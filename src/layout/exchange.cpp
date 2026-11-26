#include "exchange.hpp"

// C/C++
#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <vector>

// snap
#include <snap/mesh/meshblock.hpp>

#include "cubed_sphere_constants.h"
#include "cubed_sphere_layout.hpp"  // CS_FACE_EDGES
#include "layout.hpp"

namespace snap {

namespace {

// a split x1 column's peers in the layout, {below, above}
std::pair<int, int> raw_x1_peers(LayoutImpl const& layout) {
  auto iloc = layout.loc_of(layout.options->rank());
  return {layout.neighbor_rank(iloc, {0, 0, -1}),
          layout.neighbor_rank(iloc, {0, 0, 1})};
}

}  // namespace

FaceKind Exchange::kind(int face) const {
  TORCH_CHECK(face >= 0 && face < 6, "[Exchange:kind] face ", face,
              " is not in [0, 6)");
  int side = face % 2 == 0 ? -1 : 1;
  int dy = face / 2 == 2 ? side : 0;  // x3
  int dx = face / 2 == 1 ? side : 0;  // x2
  int dz = face / 2 == 0 ? side : 0;  // x1
  // a boundary function on the face answers for it
  auto k = pmb_->options->face_kind(dy, dx, dz);
  if (k != FaceKind::interior) return k;

  auto layout = pmb_->get_layout();
  if (!layout) return FaceKind::interior;
  int rank = layout->options->rank();
  auto iloc = layout->loc_of(rank);
  int nb = layout->neighbor_rank(iloc, {dy, dx, dz});
  if (nb < 0) return FaceKind::interior;
  if (nb == rank) return FaceKind::periodic;
  if (layout->options->is_cubed_sphere() &&
      layout->panel_of(nb) != layout->panel_of(rank))
    return FaceKind::panel;
  if (dz != 0) return FaceKind::seam_x1;
  // the neighbour past a domain edge is the wrapped one
  auto [rx, ry, rz] = iloc;
  bool wrap = (dx != 0 && (rx + dx < 0 || rx + dx >= layout->options->px())) ||
              (dy != 0 && (ry + dy < 0 || ry + dy >= layout->options->py()));
  return wrap ? FaceKind::periodic : FaceKind::interior;
}

std::pair<int, int> Exchange::x1_neighbors() const {
  auto layout = pmb_->get_layout();
  if (!layout || layout->options->periodic_z() || layout->options->pz() <= 1)
    return {-1, -1};
  return raw_x1_peers(*layout);
}

void Exchange::x1_rows(std::vector<torch::Tensor> fields, int m, bool faces,
                       int tag) const {
  auto [below, above] = x1_neighbors();
  if (below < 0 && above < 0) return;
  TORCH_CHECK(!fields.empty(), "[Exchange:x1_rows] no field");
  int is = pmb_->pcoord->il(), iu = pmb_->pcoord->iu(), o = faces ? 1 : 0;
  int nf = fields.size();

  // one message per direction: the rows of every field, in order, along the
  // last dimension (one field: its rows as they are)
  auto pack = [&](int start) {
    if (nf == 1) return fields[0].narrow(-1, start, m);
    std::vector<torch::Tensor> rows;
    rows.reserve(nf);
    for (auto const& f : fields) rows.push_back(f.narrow(-1, start, m));
    return torch::cat(rows, -1);
  };
  auto unpack = [&](torch::Tensor const& rows, int start) {
    for (int k = 0; k < nf; ++k)
      fields[k].narrow(-1, start, m).copy_(rows.narrow(-1, k * m, m));
  };

  // up: my top rows are the above block's lower ghosts; down: my bottom rows
  // (past the seam face, for faces) are the below block's upper ghosts. The
  // column's two faces are the seam sides 0 (below) and 1 (above).
  std::vector<SeamSide> sides = {
      SeamSide{1, 0, below >= 0 ? pack(is + o) : torch::Tensor()},
      SeamSide{1, 1, above >= 0 ? pack(iu + 1 - m) : torch::Tensor()}};
  for (auto& s : sides) resolve_seam(s, false);
  seam_exchange(sides, tag);
  if (below >= 0) unpack(sides[0].recv, is - m);
  if (above >= 0) unpack(sides[1].recv, iu + 1 + o);
}

std::pair<int, int> Exchange::x1_seam_peers(bool process_only) const {
  auto layout = pmb_->get_layout();
  if (!layout || layout->options->pz() <= 1) return {-1, -1};
  auto [below, above] = raw_x1_peers(*layout);
  if (process_only) {
    if (above >= 0 && layout->is_local_block(above)) above = -1;
    if (below >= 0 && layout->is_local_block(below)) below = -1;
  }
  return {below, above};
}

std::pair<torch::Tensor, torch::Tensor> Exchange::x1_seam_slab(
    torch::Tensor const& lower, torch::Tensor const& upper, int tag,
    bool process_only) const {
  std::pair<torch::Tensor, torch::Tensor> theirs;
  if (!pmb_->get_layout()) return theirs;
  // the column's two faces as seam sides 0 (below) and 1 (above); row 9 on
  // every x1 seam, a periodic wrap included
  std::vector<SeamSide> sides = {SeamSide{1, 0, lower}, SeamSide{1, 1, upper}};
  bool any = false;
  for (auto& s : sides) {
    if (!resolve_seam(s, process_only)) continue;
    TORCH_CHECK(s.send.defined(), "[Exchange:x1_seam_slab] no slab for side ",
                s.side, ", which has a peer");
    any = true;
  }
  if (!any) return theirs;
  seam_exchange(sides, tag);
  // the peer's slab as posted: a board hands over the owner's tensor
  if (sides[0].peer >= 0) theirs.first = sides[0].recv;
  if (sides[1].peer >= 0) theirs.second = sides[1].recv;
  return theirs;
}

bool Exchange::resolve_seam(SeamSide& s, bool process_only) const {
  TORCH_CHECK(s.dim >= 1 && s.dim <= 3, "[Exchange:seam] dim ", s.dim,
              " is not 1, 2 or 3");
  s.peer = -1;
  s.peer_side = -1;
  s.reversed = false;
  auto layout = pmb_->get_layout();
  if (!layout) return false;
  auto iloc = layout->loc_of(layout->options->rank());
  int peer;
  if (s.dim == 1) {
    // the x1 column: side 0 faces the block below, side 1 the block above
    // (a periodic x1 wrap counts); the x1 layout has no panel seams
    TORCH_CHECK(s.side == 0 || s.side == 1, "[Exchange:seam] x1 side ", s.side,
                " is not 0 or 1");
    auto [below, above] = x1_seam_peers(process_only);
    peer = s.side == 0 ? below : above;
  } else {
    // L/R are the x2 faces, B/T the x3 faces, with the (dy, dx) of their
    // neighbours as in cubed_sphere_layout.cpp
    constexpr std::array<std::array<int, 2>, 4> kOffset = {
        {{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
    peer = layout->neighbor_rank(iloc,
                                 {kOffset[s.side][0], kOffset[s.side][1], 0});
  }
  if (peer < 0 || (process_only && layout->is_local_block(peer))) return false;
  // the panel and the sphere test come from the layout, as kind() does
  int panel = layout->panel_of(layout->options->rank());
  bool cubed = s.dim != 1 && layout->options->is_cubed_sphere();
  s.peer = peer;
  if (cubed && layout->panel_of(peer) != panel) {
    auto e = CS_FACE_EDGES[panel][s.side];
    s.peer_side = e.nside;
    s.reversed = e.rev != 0;
  } else {
    s.peer_side = s.side ^ 1;
  }
  return true;
}

void Exchange::seam_exchange(std::vector<SeamSide>& sides, int tag) const {
  auto layout = pmb_->get_layout();
  int me = layout->options->rank();
  // a same-process peer goes through a board, a remote one through the
  // process group; a periodic wrap to this block reads its own other face
  std::vector<CommWorkPtr> sends;
  std::vector<std::vector<torch::Tensor>> sbufs;
  sbufs.reserve(sides.size());
  for (auto const& s : sides) {
    if (s.peer < 0 || s.peer == me || !s.send.defined()) continue;
    int t = tag + s.peer_side;
    if (layout->is_local_block(s.peer)) {
      layout->post_to_local_block(s.peer, s.send, t);
    } else {
      sbufs.push_back({s.send.contiguous()});
      sends.push_back(layout->send_to_block(sbufs.back(), s.peer, t));
    }
  }
  for (auto& s : sides) {
    if (s.peer < 0 || !s.take) continue;
    if (s.peer == me) {
      TORCH_CHECK(s.peer_side / 2 == s.side / 2, "[Exchange:seam] block ", me,
                  " is its own neighbour across a panel seam");
      auto it =
          std::find_if(sides.begin(), sides.end(), [&](SeamSide const& o) {
            return o.dim == s.dim && o.side == s.peer_side;
          });
      TORCH_CHECK(it != sides.end() && it->send.defined(),
                  "[Exchange:seam] block ", me, " wraps to its side ",
                  s.peer_side, ", which sends nothing");
      s.recv = it->send.clone();
    } else if (layout->is_local_block(s.peer)) {
      s.recv = layout->take_from_local_block(s.peer, tag + s.side);
    } else {
      if (!s.recv.defined()) s.recv = torch::empty_like(s.send);
      std::vector<torch::Tensor> rbuf = {s.recv};
      layout->recv_from_block(rbuf, s.peer, tag + s.side)->wait();
    }
  }
  for (auto& sw : sends) sw->wait();
}

std::pair<torch::Tensor, torch::Tensor> Exchange::seam_slab(
    int dim, torch::Tensor const& lower, torch::Tensor const& upper, int tag,
    bool process_only) const {
  std::pair<torch::Tensor, torch::Tensor> theirs;
  if (!pmb_->get_layout()) return theirs;
  int s0 = dim == 2 ? SIDE_L : SIDE_B;
  std::vector<SeamSide> sides = {
      SeamSide{dim, s0, lower.defined() ? lower.contiguous() : lower},
      SeamSide{dim, s0 + 1, upper.defined() ? upper.contiguous() : upper}};
  for (auto& s : sides) {
    if (!resolve_seam(s, process_only)) continue;
    TORCH_CHECK(s.send.defined(), "[Exchange:seam_slab] no slab for side ",
                s.side, ", which has a peer");
  }
  seam_exchange(sides, tag);
  // a board hands over the owner's tensor: the caller gets its own copy, in
  // this block's along-edge order
  auto own = [](SeamSide const& s) {
    if (s.peer < 0) return torch::Tensor();
    auto got = s.recv.clone();
    return s.reversed ? got.flip(1) : got;
  };
  theirs.first = own(sides[0]);
  theirs.second = own(sides[1]);
  return theirs;
}

torch::Tensor Exchange::unpack_anchor(torch::Tensor const& like,
                                      int tag) const {
  auto [below, above] = x1_neighbors();
  if (below < 0 && above < 0) return torch::Tensor();
  return pmb_->get_layout()->take_x1_anchor(above, like, tag);
}

void Exchange::pack_anchor(torch::Tensor const& face, int tag) const {
  auto [below, above] = x1_neighbors();
  if (below < 0) return;
  pmb_->get_layout()->pass_x1_anchor(below, face, tag);
}

}  // namespace snap
