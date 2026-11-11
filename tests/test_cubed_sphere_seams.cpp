// gtest
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <fstream>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

std::string species() {
  return "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "dynamics:\n"
         "  equation-of-state: {type: ideal-gas, gammad: 1.4}\n";
}

std::string panel_bounds() {
  return "x2min_pi: -0.25, x2max_pi: 0.25, x3min_pi: -0.25, x3max_pi: 0.25";
}

//! one gnomonic panel block, N x N, nghost 3, four x1 cells
std::string panel_card(int n) {
  std::string path =
      "test_cubed_sphere_seams_panel_" + std::to_string(n) + ".yaml";
  std::ofstream o(path);
  o << species()
    << "geometry:\n"
       "  type: gnomonic-equiangle\n"
       "  cells: {nx1: 4, nx2: "
    << n << ", nx3: " << n
    << ", nghost: 3}\n"
       "  bounds: {x1min: 6.4e6, x1max: 6.41e6, "
    << panel_bounds()
    << "}\n"
       "boundary-condition:\n"
       "  external:\n"
       "    x1-inner: reflecting\n"
       "    x1-outer: reflecting\n"
       "    x2-inner: custom\n"
       "    x2-outer: custom\n"
       "    x3-inner: custom\n"
       "    x3-outer: custom\n";
  return path;
}

//! six panels, nb x nb blocks each, all in this process
std::string sphere_card(int nb, int n) {
  std::string path =
      "test_cubed_sphere_seams_sphere_" + std::to_string(nb) + ".yaml";
  std::ofstream o(path);
  o << species()
    << "distribute:\n"
       "  layout: cubed-sphere\n"
       "  nb2: "
    << nb << "\n  nb3: " << nb << "\n  blocks_per_process: " << 6 * nb * nb
    << "\n  verbose: false\n"
       "geometry:\n"
       "  type: gnomonic-equiangle\n"
       "  cells: {nx1: 4, nx2: "
    << n << ", nx3: " << n
    << ", nghost: 3}\n"
       "  bounds: {x1min: 6.4e6, x1max: 6.41e6, "
    << panel_bounds()
    << "}\n"
       "boundary-condition:\n"
       "  external:\n"
       "    x1-inner: reflecting\n"
       "    x1-outer: reflecting\n"
       "    x2-inner: custom\n"
       "    x2-outer: custom\n"
       "    x3-inner: custom\n"
       "    x3-outer: custom\n"
       "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  return path;
}

Mesh make_sphere(int nb, int n) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml(sphere_card(nb, n));
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(6 * nb * nb);
  return Mesh(mesh_opts);
}

}  // namespace

// Every panel evaluates the arcs of its own edge faces. A seam face is one
// physical face, so both panels must hold the same area, bit for bit.
TEST(CubedSphereSeams, seam_face_areas_are_single_valued) {
  for (int n : {16, 48}) {
    auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(panel_card(n)));
    auto g = block->pcoord;
    int ng = g->options->nghost();
    int js = ng, je = ng + n, ks = ng, ke = ng + n;
    int is = g->il(), ie = g->iu() + 1;
    auto a2 = g->face_area2();
    auto a3 = g->face_area3();
    auto side = [&](int s, int p) {
      if (s == SIDE_L)
        return a2.select(1, js).select(0, ks + p).slice(0, is, ie);
      if (s == SIDE_R)
        return a2.select(1, je).select(0, ks + p).slice(0, is, ie);
      if (s == SIDE_B)
        return a3.select(0, ks).select(0, js + p).slice(0, is, ie);
      return a3.select(0, ke).select(0, js + p).slice(0, is, ie);
    };
    int differ = 0, faces = 0;
    for (int f = 0; f < 6; ++f)
      for (int s = 0; s < 4; ++s) {
        auto e = CS_FACE_EDGES[f][s];
        if (e.nface < f) continue;  // each physical edge once
        for (int p = 0; p < n; ++p) {
          int p2 = e.rev ? n - 1 - p : p;
          differ += !torch::equal(side(s, p), side(e.nside, p2));
          ++faces;
        }
      }
    EXPECT_EQ(differ, 0) << "N = " << n << ": " << differ << " of " << faces
                         << " seam faces differ between the two panels";
  }
}

// A uniform state at rest comes back from the cross-panel exchange exactly:
// interpolating between equal neighbours returns the neighbour.
TEST(CubedSphereSeams, uniform_state_survives_the_exchange_bitwise) {
  for (int nb : {1, 2}) {
    auto mesh = make_sphere(nb, 16);
    MeshVariables vars(mesh->blocks.size());
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto c = mesh->blocks[i]->pcoord;
      int nvar = mesh->blocks[i]->phydro->peos->nvar();
      auto w = torch::zeros(
          {nvar, c->options->nc3(), c->options->nc2(), c->options->nc1()},
          torch::kFloat64);
      w[IDN].fill_(1.3);
      w[IPR].fill_(1.e5);
      vars[i]["hydro_w"] = w;
    }
    mesh->initialize(vars);
    long bad = 0, total = 0;
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto c = mesh->blocks[i]->pcoord;
      auto u = vars[i].at("hydro_u");
      int ng = c->options->nghost();
      int n2 = c->options->nc2(), n3 = c->options->nc3();
      int il = c->il(), iu = c->iu() + 1, jl = c->jl(), ju = c->ju() + 1,
          kl = c->kl(), ku = c->ku() + 1;
      auto ref = u.select(1, kl).select(1, jl).select(1, il);  // (nvar)
      std::vector<torch::Tensor> strips = {
          u.slice(1, kl, ku).slice(2, 0, ng).slice(3, il, iu),
          u.slice(1, kl, ku).slice(2, ju, n2).slice(3, il, iu),
          u.slice(1, 0, ng).slice(2, jl, ju).slice(3, il, iu),
          u.slice(1, ku, n3).slice(2, jl, ju).slice(3, il, iu)};
      for (auto const& s : strips) {
        auto want = ref.view({-1, 1, 1, 1}).expand_as(s);
        bad += (s != want).sum().item<long>();
        total += s.numel();
      }
    }
    EXPECT_EQ(bad, 0) << "nb = " << nb << ": " << bad << " of " << total
                      << " edge ghost entries are not the uniform state";
  }
}

// The connectivity table and the panel face map describe the same cube: the
// table is an involution, and both panels place every seam vertex at the same
// point (to round-off).
TEST(CubedSphereSeams, connectivity_table_matches_the_face_map) {
  int n = 16;
  double d = M_PI / (2. * n), worst = 0.;
  for (int f = 0; f < 6; ++f)
    for (int s = 0; s < 4; ++s) {
      auto e = CS_FACE_EDGES[f][s];
      auto back = CS_FACE_EDGES[e.nface][e.nside];
      EXPECT_EQ(back.nface, f);
      EXPECT_EQ(back.nside, s);
      EXPECT_EQ(back.rev, e.rev);
      for (int p = 0; p <= n; ++p) {
        double t = -M_PI / 4 + p * d;
        int p2 = e.rev ? n - p : p;
        double t2 = -M_PI / 4 + p2 * d;
        auto ab = [&](int side, double tt, double* a, double* b) {
          double edge = side == SIDE_L || side == SIDE_B ? -M_PI / 4 : M_PI / 4;
          if (side == SIDE_L || side == SIDE_R) {
            *a = edge, *b = tt;
          } else {
            *a = tt, *b = edge;
          }
        };
        double a1, b1, a2, b2;
        ab(s, t, &a1, &b1);
        ab(e.nside, t2, &a2, &b2);
        auto P = cs_ab_to_xyz(CS_FACE_NAMES[f], a1, b1);
        auto Q = cs_ab_to_xyz(CS_FACE_NAMES[e.nface], a2, b2);
        worst = std::max(worst, std::sqrt((P.x - Q.x) * (P.x - Q.x) +
                                          (P.y - Q.y) * (P.y - Q.y) +
                                          (P.z - Q.z) * (P.z - Q.z)));
      }
    }
  EXPECT_LT(worst, 1.e-14);
}

// Corners are filled only where something reads them. On the cubed sphere no
// stencil reads the corner ghosts (between an x2 and an x3 ghost strip): the
// interior after one stage does not depend on them.
TEST(CubedSphereSeams, corner_ghosts_are_not_read) {
  auto run = [](bool poison) {
    auto mesh = make_sphere(2, 8);
    MeshVariables vars(mesh->blocks.size());
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto c = mesh->blocks[i]->pcoord;
      int nvar = mesh->blocks[i]->phydro->peos->nvar();
      auto m = torch::meshgrid({c->x3v, c->x2v, c->x1v}, "ij");
      auto w = torch::zeros(
          {nvar, c->options->nc3(), c->options->nc2(), c->options->nc1()},
          torch::kFloat64);
      w[IDN] = 1. + 0.1 * torch::sin(3. * m[1]) * torch::cos(2. * m[0]);
      w[IPR] = 1.e5 * (1. + 0.01 * torch::cos(3. * m[0]));
      w[IVY] = 5. * torch::cos(m[0]);
      vars[i]["hydro_w"] = w;
    }
    mesh->initialize(vars);
    if (poison) {
      for (size_t i = 0; i < mesh->blocks.size(); ++i) {
        auto c = mesh->blocks[i]->pcoord;
        int ng = c->options->nghost();
        int n2 = c->options->nc2(), n3 = c->options->nc3();
        for (auto name : {"hydro_u", "hydro_w"}) {
          auto t = vars[i].at(name);
          for (int k0 : {0, n3 - ng})
            for (int j0 : {0, n2 - ng})
              t.slice(1, k0, k0 + ng).slice(2, j0, j0 + ng).fill_(NAN);
        }
      }
    }
    mesh->forward(vars, mesh->max_time_step(vars), 0);
    std::vector<torch::Tensor> out;
    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto c = mesh->blocks[i]->pcoord;
      out.push_back(vars[i]
                        .at("hydro_u")
                        .slice(1, c->kl(), c->ku() + 1)
                        .slice(2, c->jl(), c->ju() + 1)
                        .slice(3, c->il(), c->iu() + 1)
                        .clone());
    }
    return out;
  };
  auto a = run(false), b = run(true);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_TRUE(torch::equal(a[i], b[i])) << "block " << i;
  }
}

// On a slab, the x1 and x2 boundary functions fill the corner ghosts (which
// the face-flux covariance reads) with the reflected and periodic image of
// the interior.
TEST(CubedSphereSeams, slab_corner_ghosts_are_the_boundary_images) {
  std::string path = "test_cubed_sphere_seams_slab.yaml";
  {
    std::ofstream o(path);
    o << species()
      << "geometry:\n"
         "  type: cartesian\n"
         "  cells: {nx1: 16, nx2: 24, nx3: 1, nghost: 3}\n"
         "  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., "
         "x3max: 1.}\n"
         "distribute: {layout: slab, nb1: 1, nb2: 1, nb3: 1, "
         "blocks_per_process: 1}\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: reflecting\n"
         "    x1-outer: reflecting\n"
         "    x2-inner: periodic\n"
         "    x2-outer: periodic\n"
         "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  }
  auto block_opts = MeshBlockOptionsImpl::from_yaml(path);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);
  auto mesh = Mesh(mesh_opts);
  auto c = mesh->blocks[0]->pcoord;
  int nvar = mesh->blocks[0]->phydro->peos->nvar();
  auto m = torch::meshgrid({c->x3v, c->x2v, c->x1v}, "ij");
  auto w = torch::zeros(
      {nvar, c->options->nc3(), c->options->nc2(), c->options->nc1()},
      torch::kFloat64);
  w[IDN] = 1. + 0.1 * torch::sin(2. * M_PI * m[1]) * torch::cos(M_PI * m[2]);
  w[IPR].fill_(1.e5);
  MeshVariables vars(1);
  vars[0]["hydro_w"] = w;
  mesh->initialize(vars);
  auto r = vars[0].at("hydro_u")[IDN].select(0, 0);  // (nc2, nc1)
  int ng = c->options->nghost(), nx1 = c->options->nx1(),
      nx2 = c->options->nx2();
  int bad = 0;
  for (int jc : {0, 1})
    for (int ic : {0, 1})
      for (int a = 0; a < ng; ++a)
        for (int q = 0; q < ng; ++q) {
          int i = ic ? ng + nx1 + a : a, j = jc ? ng + nx2 + q : q;
          int ii = ic ? 2 * (ng + nx1) - 1 - i : 2 * ng - 1 - i;
          int jj = jc ? j - nx2 : j + nx2;
          bad += r[j][i].item<double>() != r[jj][ii].item<double>();
        }
  EXPECT_EQ(bad, 0);
}

namespace {

//! a cubed-sphere layout whose every neighbour is the block itself, which no
//! real cube produces: it reaches the self-send branch of exchange_remote
class SelfNeighbourLayout : public CubedSphereLayoutImpl {
 public:
  using CubedSphereLayoutImpl::CubedSphereLayoutImpl;
  int neighbor_rank(std::tuple<int, int, int>,
                    std::tuple<int, int, int>) const override {
    return options->rank();
  }
};

}  // namespace

// A block that is its own neighbour across a seam would receive into the
// reversed direction's slot, which is not where the rotated peer offset
// points; the exchange refuses it instead of filling the wrong ghosts.
TEST(CubedSphereSeams, self_send_across_a_seam_is_refused) {
  auto mesh = make_sphere(1, 8);
  auto pmb = mesh->blocks[0].get();
  SelfNeighbourLayout layout(pmb->options->layout(), pmb);
  std::vector<CommWorkPtr> works;
  try {
    layout.exchange_remote(pmb, SyncOptions(), works);
    FAIL() << "a cubed-sphere self-send was not refused";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("is its own neighbour"),
              std::string::npos)
        << e.what_without_backtrace();
  }
}
