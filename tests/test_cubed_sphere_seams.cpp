// gtest
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

constexpr int N = 16;

//! six N x N panels, one block each, all in this process
Mesh make_sphere() {
  std::string path = "test_cubed_sphere_seams.yaml";
  std::ofstream(path)
      << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species: [{name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, "
         "cv_R: 2.5}]\n"
         "dynamics: {equation-of-state: {type: ideal-gas, gammad: 1.4}}\n"
         "distribute: {layout: cubed-sphere, nb2: 1, nb3: 1, "
         "blocks_per_process: 6}\n"
         "geometry:\n"
         "  type: gnomonic-equiangle\n"
         "  cells: {nx1: 4, nx2: "
      << N << ", nx3: " << N << ", nghost: 3}\n"
      << "  bounds: {x1min: 6.4e6, x1max: 6.41e6, x2min_pi: -0.25, "
         "x2max_pi: 0.25, x3min_pi: -0.25, x3max_pi: 0.25}\n"
         "boundary-condition: {external: {x1-inner: reflecting, x1-outer: "
         "reflecting, x2-inner: custom, x2-outer: custom, x3-inner: custom, "
         "x3-outer: custom}}\n"
         "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(MeshBlockOptionsImpl::from_yaml(path));
  mesh_opts->blocks_per_process(6);
  return Mesh(mesh_opts);
}

// A seam face is one physical face: both panels hold the same area, bit for
// bit (with an x2 arc on one side and an x3 arc on the other, some differed).
TEST(CubedSphereSeams, seam_face_areas_are_single_valued) {
  auto g = make_sphere()->blocks[0]->pcoord;
  int ng = g->options->nghost(), is = g->il(), ie = g->iu() + 1;
  auto a2 = g->face_area2(), a3 = g->face_area3();
  auto side = [&](int s, int p) {
    if (s == SIDE_L || s == SIDE_R)
      return a2.select(1, s == SIDE_L ? ng : ng + N).select(0, ng + p);
    return a3.select(0, s == SIDE_B ? ng : ng + N).select(0, ng + p);
  };
  int differ = 0;
  for (int f = 0; f < 6; ++f)
    for (int s = 0; s < 4; ++s) {
      auto e = CS_FACE_EDGES[f][s];
      for (int p = 0; p < N; ++p)
        differ += !torch::equal(
            side(s, p).slice(0, is, ie),
            side(e.nside, e.rev ? N - 1 - p : p).slice(0, is, ie));
    }
  EXPECT_EQ(differ, 0) << "seam face pairs that differ";
}

//! a cubed-sphere layout whose every neighbour is the block itself
struct SelfNeighbourLayout : CubedSphereLayoutImpl {
  using CubedSphereLayoutImpl::CubedSphereLayoutImpl;
  int neighbor_rank(std::tuple<int, int, int>,
                    std::tuple<int, int, int>) const override {
    return options->rank();
  }
};

// A self-send across a seam would fill the wrong ghosts; it is refused.
TEST(CubedSphereSeams, self_send_across_a_seam_is_refused) {
  auto mesh = make_sphere();
  auto pmb = mesh->blocks[0].get();
  SelfNeighbourLayout layout(pmb->options->layout(), pmb);
  std::vector<CommWorkPtr> works;
  try {
    layout.exchange_remote(pmb, SyncOptions(), works);
    FAIL() << "a cubed-sphere self-send was not refused";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("its own neighbour"),
              std::string::npos);
  }
}

namespace {

//! a smooth state with wind on every panel: a solid-body wind plus a
//! meridional part and a pressure bump, so every seam carries momentum
void fill_sphere(Mesh mesh, MeshVariables& vars) {
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto blk = mesh->blocks[b];
    auto c = blk->pcoord;
    auto lay = blk->get_layout();
    int face = std::get<2>(lay->loc_of(lay->options->rank()));
    auto g = torch::meshgrid({c->x3v, c->x2v}, "ij");
    auto beta = g[0], alpha = g[1];
    auto [lon, lat] = cs_ab_to_lonlat(CS_FACE_NAMES[face], alpha, beta);
    auto vel = torch::zeros({3, alpha.size(0), alpha.size(1)}, alpha.options());
    vel[2] = 20. * lat.cos();
    vel[1] = 5. * (2. * lon).sin() * lat.cos();
    cs_sph_to_contra_(vel, alpha, beta, face);
    auto d = ((lat.sin() * std::sin(0.3) +
               lat.cos() * std::cos(0.3) * (lon - 1.).cos())
                  .clamp(-1., 1.))
                 .acos();
    auto p = 1.e5 * (1. + 0.05 * (-(d / 0.3).square()).exp());
    int nvar = blk->phydro->peos->nvar();
    auto w = torch::zeros(
        {nvar, c->options->nc3(), c->options->nc2(), c->options->nc1()},
        c->x1v.options());
    w[IDN] = (p / (287. * 300.)).unsqueeze(-1);
    w[IPR] = p.unsqueeze(-1);
    for (int n = 0; n < 3; ++n) w[IVX + n] = vel[n].unsqueeze(-1);
    vars[b]["hydro_w"] = w;
  }
}

//! one stage of the six-panel sphere
Mesh stepped_sphere() {
  auto mesh = make_sphere();
  mesh->to(torch::kCPU, torch::kFloat64);
  MeshVariables vars(mesh->blocks.size());
  fill_sphere(mesh, vars);
  mesh->initialize(vars);
  double dt = mesh->max_time_step(vars);
  mesh->forward(vars, dt, 0);
  return mesh;
}

//! side s of a block's face fluxes as [nvar, N, nx1] along the edge, with the
//! face points (alpha, beta) and the face metric cosine, each [N, nx1]
struct SideSlab {
  torch::Tensor rows, alpha, beta, cosine;
};

SideSlab side_slab(MeshBlock blk, int s) {
  auto c = blk->pcoord;
  auto const& reg = blk->phydro->flux_register();
  int ng = c->options->nghost(), is = c->il(), ni = c->iu() + 1 - is;
  SideSlab out;
  auto along = [&](torch::Tensor v) {  // [N] -> [N, ni]
    return v.narrow(0, ng, N).view({N, 1}).expand({N, ni});
  };
  if (s == SIDE_L || s == SIDE_R) {
    int j = s == SIDE_L ? c->jl() : c->ju() + 1;
    out.rows = reg.total(2).select(2, j).narrow(1, ng, N).narrow(2, is, ni);
    out.alpha = c->x2f[j].expand({N, ni});
    out.beta = along(c->x3v);
    out.cosine = along(c->cosine_face2_kj.select(1, j));
  } else {
    int k = s == SIDE_B ? c->kl() : c->ku() + 1;
    out.rows = reg.total(3).select(1, k).narrow(1, ng, N).narrow(2, is, ni);
    out.alpha = along(c->x2v);
    out.beta = c->x3f[k].expand({N, ni});
    out.cosine = along(c->cosine_face3_kj.select(0, k));
  }
  return out;
}

}  // namespace

// S step (4), row 16r (S.rst rev 2.5 section 4): after reconcile every panel
// seam face holds one flux. Mass and energy rows of the two sides agree bit
// for bit once mapped by the edge (reverse, normal sign); momentum rows agree
// after the rotation from one panel's covariant face basis to the other's
// (S.rst: 1e-14 relative, a hypothesis; checked here at 1e-13).
TEST(CubedSphereSeams, panel_seam_fluxes_are_single_valued_after_a_stage) {
  auto mesh = stepped_sphere();
  std::array<MeshBlock, 6> by_face;
  for (auto& blk : mesh->blocks) {
    auto lay = blk->get_layout();
    by_face[std::get<2>(lay->loc_of(lay->options->rank()))] = blk;
  }
  int64_t scalar_differ = 0, scalar_total = 0;
  double scalar_rel = 0.;
  double mom_err = 0., mom_scale = 0.;
  for (int f = 0; f < 6; ++f)
    for (int s = 0; s < 4; ++s) {
      auto e = CS_FACE_EDGES[f][s];
      auto own = side_slab(by_face[f], s);
      auto peer = side_slab(by_face[e.nface], e.nside);
      double sign = (s % 2) == (e.nside % 2) ? -1. : 1.;
      auto pr = e.rev ? peer.rows.flip(1) : peer.rows;
      for (int v : {static_cast<int>(IDN), static_cast<int>(IPR)}) {
        auto mapped = sign * pr[v];
        scalar_differ += (own.rows[v] != mapped).sum().item<int64_t>();
        scalar_rel =
            std::max(scalar_rel, ((own.rows[v] - mapped).abs().max() /
                                  own.rows[v].abs().max().clamp_min(1e-300))
                                     .item<double>());
        scalar_total += own.rows[v].numel();
      }
      // peer momentum: its covariant face rows -> contravariant -> Cartesian
      // at its face points -> this panel's contravariant -> covariant
      auto m = peer.rows.narrow(0, IVX, 3).clone();
      coord_vec_raise_(m, peer.cosine.contiguous());
      cs_contra_to_cart_(m, peer.alpha, peer.beta, e.nface);
      if (e.rev) m = m.flip(1);
      cs_cart_to_contra_(m, own.alpha, own.beta, f);
      coord_vec_lower_(m, own.cosine.contiguous());
      auto own_m = own.rows.narrow(0, IVX, 3);
      mom_err =
          std::max(mom_err, (own_m - sign * m).abs().max().item<double>());
      mom_scale = std::max(mom_scale, own_m.abs().max().item<double>());
    }
  std::printf(
      "panel seams: %ld of %ld mass/energy face values differ (max relative "
      "%.3e); momentum max |difference| / max |flux| = %.3e\n",
      static_cast<long>(scalar_differ), static_cast<long>(scalar_total),
      scalar_rel, mom_err / mom_scale);
  EXPECT_EQ(scalar_differ, 0);
  EXPECT_LE(mom_err / mom_scale, 1e-13);
}

// S step (4), boundary.prescribed (S.rst rev 2.5 section 4): the mass and
// energy flux through a reflecting x1 wall is zero after the stage, and R
// books what the mask removed as "boundary.prescribed"
TEST(CubedSphereSeams, wall_mass_and_energy_flux_is_prescribed_zero) {
  auto mesh = stepped_sphere();
  double worst = 0.;
  for (auto& blk : mesh->blocks) {
    auto c = blk->pcoord;
    auto t = blk->phydro->flux_register().total(1);
    for (int i : {c->il(), c->iu() + 1})
      for (int v : {static_cast<int>(IDN), static_cast<int>(IPR)})
        worst = std::max(worst, t[v].select(-1, i).abs().max().item<double>());
  }
  std::printf("max |mass or energy flux| on the x1 walls = %.3e\n", worst);
  EXPECT_EQ(worst, 0.);
}
