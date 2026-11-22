#include "exchange.hpp"

// C/C++
#include <array>
#include <tuple>
#include <utility>
#include <vector>

// snap
#include <snap/mesh/meshblock.hpp>

#include "cubed_sphere_constants.h"
#include "cubed_sphere_layout.hpp"
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
      layout->panel_of(nb) != std::get<2>(iloc))
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
  auto layout = pmb_->get_layout();
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

  std::vector<std::vector<torch::Tensor>> sbufs;
  std::vector<CommWorkPtr> sends;
  sbufs.reserve(2);
  auto post = [&](torch::Tensor rows, int to, int t) {
    if (layout->is_local_block(to)) {
      layout->post_to_local_block(to, rows, t);
    } else {
      sbufs.push_back({rows.contiguous()});
      sends.push_back(layout->send_to_block(sbufs.back(), to, t));
    }
  };
  auto take = [&](int from, int t) {
    if (layout->is_local_block(from))
      return layout->take_from_local_block(from, t);
    std::vector<torch::Tensor> rbuf;
    if (nf == 1) {
      rbuf = {torch::empty_like(fields[0].narrow(-1, 0, m).contiguous())};
    } else {
      auto shape = fields[0].sizes().vec();
      shape.back() = nf * m;
      rbuf = {torch::empty(shape, fields[0].options())};
    }
    layout->recv_from_block(rbuf, from, t)->wait();
    return rbuf[0];
  };

  // up: my top rows are the above block's lower ghosts; down: my bottom rows
  // (past the seam face, for faces) are the below block's upper ghosts
  if (above >= 0) post(pack(iu + 1 - m), above, tag);
  if (below >= 0) post(pack(is + o), below, tag + 1);
  if (below >= 0) unpack(take(below, tag), is - m);
  if (above >= 0) unpack(take(above, tag + 1), iu + 1 + o);
  for (auto& sw : sends) sw->wait();
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
  auto [below, above] = x1_seam_peers(process_only);
  if (above < 0 && below < 0) return theirs;
  auto layout = pmb_->get_layout();

  // a same-process peer goes through a board, a remote one through the
  // process group (S step (4): row 9 on every x1 seam)
  std::vector<CommWorkPtr> sends;
  std::vector<torch::Tensor> up_mine, dn_mine;
  auto post = [&](std::vector<torch::Tensor>& mine, torch::Tensor const& slab,
                  int to, int t) {
    if (layout->is_local_block(to)) {
      layout->post_to_local_block(to, slab, t);
    } else {
      mine = {slab};
      sends.push_back(layout->send_to_block(mine, to, t));
    }
  };
  auto take = [&](torch::Tensor const& like, int from, int t) {
    if (layout->is_local_block(from))
      return layout->take_from_local_block(from, t);
    std::vector<torch::Tensor> rbuf = {torch::empty_like(like)};
    layout->recv_from_block(rbuf, from, t)->wait();
    return rbuf[0];
  };
  if (above >= 0) post(up_mine, upper, above, tag);
  if (below >= 0) post(dn_mine, lower, below, tag + 1);
  if (above >= 0) theirs.second = take(upper, above, tag + 1);
  if (below >= 0) theirs.first = take(lower, below, tag);
  for (auto& sw : sends) sw->wait();
  return theirs;
}

std::pair<torch::Tensor, torch::Tensor> Exchange::seam_slab(
    int dim, torch::Tensor const& lower, torch::Tensor const& upper, int tag,
    bool process_only) const {
  TORCH_CHECK(dim == 2 || dim == 3, "[Exchange:seam_slab] dim ", dim,
              " is not 2 or 3");
  std::pair<torch::Tensor, torch::Tensor> theirs;
  auto layout = pmb_->get_layout();
  if (!layout) return theirs;
  int me = layout->options->rank();
  auto iloc = layout->loc_of(me);
  bool cubed =
      dynamic_cast<CubedSphereLayoutImpl const*>(layout.get()) != nullptr;
  int panel = std::get<2>(iloc);

  // the two sides of dim: L/R are the x2 faces, B/T the x3 faces, with the
  // (dy, dx) of their neighbours as in cubed_sphere_layout.cpp
  constexpr std::array<std::array<int, 2>, 4> kOffset = {
      {{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
  struct Side {
    int side, peer, peer_side;
    bool reversed;
    torch::Tensor const* mine;
    torch::Tensor* got;
  };
  int s0 = dim == 2 ? SIDE_L : SIDE_B;
  std::array<Side, 2> sides = {
      Side{s0, -1, -1, false, &lower, &theirs.first},
      Side{s0 + 1, -1, -1, false, &upper, &theirs.second}};
  for (auto& sd : sides) {
    sd.peer = layout->neighbor_rank(
        iloc, {kOffset[sd.side][0], kOffset[sd.side][1], 0});
    if (sd.peer < 0 || (process_only && layout->is_local_block(sd.peer))) {
      sd.peer = -1;
      continue;
    }
    if (cubed && layout->panel_of(sd.peer) != panel) {
      auto e = CS_FACE_EDGES[panel][sd.side];
      sd.peer_side = e.nside;
      sd.reversed = e.rev != 0;
    } else {
      sd.peer_side = sd.side ^ 1;
    }
    TORCH_CHECK(sd.mine->defined(), "[Exchange:seam_slab] no slab for side ",
                sd.side, ", which has a peer");
  }

  // a same-process peer goes through a board, a remote one through the
  // process group; a periodic wrap to this block reads its own other face
  std::vector<CommWorkPtr> sends;
  std::array<std::vector<torch::Tensor>, 2> sbufs;
  for (int n = 0; n < 2; ++n) {
    auto const& sd = sides[n];
    if (sd.peer < 0 || sd.peer == me) continue;
    auto slab = sd.mine->contiguous();
    int t = tag + sd.peer_side;
    if (layout->is_local_block(sd.peer)) {
      layout->post_to_local_block(sd.peer, slab, t);
    } else {
      sbufs[n] = {slab};
      sends.push_back(layout->send_to_block(sbufs[n], sd.peer, t));
    }
  }
  for (auto& sd : sides) {
    if (sd.peer < 0) continue;
    torch::Tensor got;
    if (sd.peer == me) {
      TORCH_CHECK(sd.peer_side / 2 == sd.side / 2,
                  "[Exchange:seam_slab] block ", me,
                  " is its own neighbour across a panel seam");
      got = (sd.peer_side == sides[0].side ? lower : upper).clone();
    } else if (layout->is_local_block(sd.peer)) {
      got = layout->take_from_local_block(sd.peer, tag + sd.side).clone();
    } else {
      std::vector<torch::Tensor> rbuf = {
          torch::empty_like(sd.mine->contiguous())};
      layout->recv_from_block(rbuf, sd.peer, tag + sd.side)->wait();
      got = rbuf[0];
    }
    *sd.got = sd.reversed ? got.flip(1) : got;
  }
  for (auto& sw : sends) sw->wait();
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
