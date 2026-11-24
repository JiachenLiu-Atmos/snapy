#include "seam_register.hpp"

// C/C++
#include <array>
#include <memory>
#include <tuple>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_rotation.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/mesh/exchange_cut.hpp>
#include <snap/mesh/meshblock.hpp>

#include "cubed_sphere_constants.h"
#include "cubed_sphere_layout.hpp"

namespace snap {

namespace {

//! a block side of the cubed sphere: L/R are x2 faces, B/T x3 faces
struct PanelSide {
  int side, dim, peer, peer_side;
  bool reversed, opposite, owner;
};

//! (dy, dx) of the L, R, B, T neighbours (cubed_sphere_layout.cpp get_side)
constexpr std::array<std::array<int, 2>, 4> kSideOffset = {
    {{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

}  // namespace

void reconcile_panel_seams(FluxRegister& reg, MeshBlockImpl const* pmb,
                           int tag) {
  auto layout = pmb->get_layout();
  if (!layout ||
      dynamic_cast<CubedSphereLayoutImpl const*>(layout.get()) == nullptr)
    return;
  auto c = pmb->pcoord;
  int rank = layout->options->rank();
  auto iloc = layout->loc_of(rank);
  int panel = std::get<2>(iloc);
  int ng = c->options->nghost();
  int is = c->il(), ni = c->iu() + 1 - is;

  std::vector<PanelSide> sides;
  for (int s = 0; s < 4; ++s) {
    int dim = s < 2 ? 2 : 3;
    if (!reg.total(dim).defined()) continue;
    int nb =
        layout->neighbor_rank(iloc, {kSideOffset[s][0], kSideOffset[s][1], 0});
    if (nb < 0 || layout->panel_of(nb) == panel) continue;
    auto e = CS_FACE_EDGES[panel][s];
    sides.push_back(
        {s, dim, nb, e.nside, e.rev != 0, (s % 2) == (e.nside % 2), rank < nb});
  }
  if (sides.empty()) return;

  // a side's face plane and its points: [n] along the edge, interior only
  auto along = [&](int s) {
    return s < 2 ? c->ku() - c->kl() + 1 : c->ju() - c->jl() + 1;
  };
  auto plane = [&](int s) {
    return s == SIDE_L   ? c->jl()
           : s == SIDE_R ? c->ju() + 1
           : s == SIDE_B ? c->kl()
                         : c->ku() + 1;
  };
  //! [rows, n, ni] view of a [rows, nc3, nc2, nc1] tensor on side s
  auto slab = [&](torch::Tensor const& t, int s) {
    auto p = s < 2 ? t.select(2, plane(s)) : t.select(1, plane(s));
    return p.narrow(1, ng, along(s)).narrow(2, is, ni);
  };
  //! the face points (alpha, beta) and face metric cosine, each [n, 1]
  auto points = [&](int s) {
    int n = along(s), p = plane(s);
    if (s < 2)
      return std::make_tuple(
          c->x2f[p], c->x3v.narrow(0, ng, n).view({n, 1}),
          c->cosine_face2_kj.select(1, p).narrow(0, ng, n).view({n, 1}));
    return std::make_tuple(
        c->x2v.narrow(0, ng, n).view({n, 1}), c->x3f[p],
        c->cosine_face3_kj.select(0, p).narrow(0, ng, n).view({n, 1}));
  };
  //! the rows S moves for dim: the total's, then each kept part's (R-4)
  auto rows_of = [&](int dim) {
    std::vector<torch::Tensor> v = {reg.total(dim)};
    for (auto const& part : reg.kept_parts(dim))
      v.push_back(reg.kept(dim, part));
    return v;
  };
  int nvar = reg.total(sides.front().dim).size(0);
  PanelRotation rot;

  // a side's rotation operands, built once per grid (SeamRotationCache)
  if (!pmb->seam_rotation_cache)
    pmb->seam_rotation_cache = std::make_shared<SeamRotationCache>();
  auto& cache = *pmb->seam_rotation_cache;
  auto operands = [&](int s) -> PanelRotation::FaceOperands const& {
    using Stamp = SeamRotationCache::Stamp;
    std::vector<Stamp> grid =
        s < 2 ? std::vector<Stamp>{Stamp(c->x2f), Stamp(c->x3v),
                                   Stamp(c->cosine_face2_kj)}
              : std::vector<Stamp>{Stamp(c->x2v), Stamp(c->x3f),
                                   Stamp(c->cosine_face3_kj)};
    std::array<int64_t, 3> extent = {panel, ni, along(s)};
    auto& side = cache.sides[s];
    if (side.grid != grid || side.extent != extent) {
      auto [alpha, beta, cosine] = points(s);
      side.op =
          PanelRotation::face_operands({along(s), ni}, alpha, beta, cosine);
      side.grid = std::move(grid);
      side.extent = extent;
      ++cache.builds;
    }
    return side.op;
  };

  // pack: owners' rows, one message per side in Cartesian momentum, into
  // the block's fixed board buffers (V3 B3: a captured segment's graphs
  // write them before the cut and the peer's graphs read them after it)
  int nsides = static_cast<int>(sides.size());
  std::vector<torch::Tensor> mine(nsides), rbufs(nsides);
  for (int k = 0; k < nsides; ++k) {
    auto const& sd = sides[k];
    if (!sd.owner) continue;
    std::vector<torch::Tensor> rows;
    for (auto const& t : rows_of(sd.dim)) rows.push_back(slab(t, sd.side));
    auto payload = torch::cat(rows, 0);
    auto const& op = operands(sd.side);
    for (int64_t r0 = 0; r0 < payload.size(0); r0 += nvar)
      rot.face_rows_to_cart_(payload.narrow(0, r0 + IVX, 3), panel, op);
    mine[k] = pmb->board_buffer({tag, sd.side, 0}, payload);
    mine[k].copy_(payload);
  }

  // receivers' own values elsewhere and, for a remote owner, a fixed receive
  // buffer
  std::array<torch::Tensor, 3> peer, faces;
  for (int k = 0; k < nsides; ++k) {
    auto const& sd = sides[k];
    if (sd.owner) continue;
    int d = sd.dim - 1;
    if (!peer[d].defined()) {
      peer[d] = torch::cat(rows_of(sd.dim), 0);  // own values elsewhere
      faces[d] = torch::zeros(reg.total(sd.dim).sizes().slice(1),
                              reg.total(sd.dim).options().dtype(torch::kBool));
    }
    if (!layout->is_local_block(sd.peer))
      rbufs[k] = pmb->board_buffer({tag, sd.side, 1},
                                   slab(peer[d], sd.side).contiguous());
  }

  // the host part, a cut in a captured segment: post or send the owners'
  // buffers, take or receive the peers'. A local take hands over the
  // owner's fixed buffer, the same tensor at every replay
  auto got = std::make_shared<std::vector<torch::Tensor>>(nsides);
  exchange_cut([layout, sides, mine, rbufs, got, tag]() {
    std::vector<std::vector<torch::Tensor>> sbufs;
    std::vector<CommWorkPtr> sends;
    sbufs.reserve(sides.size());
    for (size_t k = 0; k < sides.size(); ++k) {
      auto const& sd = sides[k];
      if (!sd.owner) continue;
      int t = tag + sd.peer_side;
      if (layout->is_local_block(sd.peer)) {
        layout->post_to_local_block(sd.peer, mine[k], t);
      } else {
        sbufs.push_back({mine[k]});
        sends.push_back(layout->send_to_block(sbufs.back(), sd.peer, t));
      }
    }
    for (size_t k = 0; k < sides.size(); ++k) {
      auto const& sd = sides[k];
      if (sd.owner) continue;
      int t = tag + sd.side;
      if (layout->is_local_block(sd.peer)) {
        (*got)[k] = layout->take_from_local_block(sd.peer, t);
      } else {
        std::vector<torch::Tensor> rbuf = {rbufs[k]};
        layout->recv_from_block(rbuf, sd.peer, t)->wait();
        (*got)[k] = rbufs[k];
      }
    }
    for (auto& sw : sends) sw->wait();
  });

  // unpack: receivers map the owner's rows into this side and replace, per
  // dim
  for (int k = 0; k < nsides; ++k) {
    auto const& sd = sides[k];
    if (sd.owner) continue;
    int d = sd.dim - 1;
    auto like = slab(peer[d], sd.side);
    auto g = (*got)[k];
    if (sd.reversed) g = g.flip(1);
    if (sd.opposite)
      g = -g;
    else
      g = g.clone();  // a board hands over the owner's tensor
    auto const& op = operands(sd.side);
    for (int64_t r0 = 0; r0 < g.size(0); r0 += nvar)
      rot.cart_to_face_rows_(g.narrow(0, r0 + IVX, 3), panel, op);
    like.copy_(g);
    slab(faces[d].unsqueeze(0), sd.side).fill_(true);
  }
  for (int d = 1; d < 3; ++d)
    if (peer[d].defined())
      reg.replace(d + 1, faces[d], peer[d], SeamRule::owner_copy);
}

void supply_prescribed(FluxRegister& reg, MeshBlockImpl const* pmb,
                       torch::Tensor const& solid,
                       std::vector<int> const& free_rows) {
  auto c = pmb->pcoord;
  // (dy, dx, dz) of the lower and upper block face of each direction
  std::array<std::array<std::array<int, 3>, 2>, 3> kFace = {
      {{{{0, 0, -1}, {0, 0, 1}}},
       {{{0, -1, 0}, {0, 1, 0}}},
       {{{-1, 0, 0}, {1, 0, 0}}}}};
  std::array<std::array<int, 2>, 3> kPlane = {
      {{c->il(), c->iu() + 1}, {c->jl(), c->ju() + 1}, {c->kl(), c->ku() + 1}}};
  for (int dim = 1; dim <= 3; ++dim) {
    auto t = reg.total(dim);
    if (!t.defined()) continue;
    int axis = 3 - dim;  // of a [nc3, nc2, nc1] face tensor
    std::vector<int64_t> planes;
    for (int lu = 0; lu < 2; ++lu) {
      auto o = kFace[dim - 1][lu];
      if (pmb->options->face_kind(o[0], o[1], o[2]) == FaceKind::wall)
        planes.push_back(kPlane[dim - 1][lu]);
    }
    if (!solid.defined()) {  // V3 E: the wall planes alone, no full mask
      reg.set_prescribed_walls(dim, planes, free_rows);
      continue;
    }
    auto faces =
        torch::zeros(t.sizes().slice(1), t.options().dtype(torch::kBool));
    for (auto p : planes) faces.select(axis, p).fill_(true);
    // face i lies between cells i - 1 and i
    auto s = (solid != 0).to(faces.device());
    int64_t n = s.size(axis);
    faces.narrow(axis, 1, n - 1)
        .logical_or_(
            s.narrow(axis, 0, n - 1).logical_or(s.narrow(axis, 1, n - 1)));
    auto mask = faces.unsqueeze(0).repeat({t.size(0), 1, 1, 1});
    for (int r : free_rows) mask[r].fill_(false);
    reg.set_prescribed(dim, mask, torch::zeros_like(t));
  }
}

}  // namespace snap
