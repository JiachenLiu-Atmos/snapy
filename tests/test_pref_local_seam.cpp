// RED on main 3f7ad96. One process, the same column as one block and as two
// blocks. p_ref of the cell under the seam should be one value. It is not.
//
// Two interior cells is too small: a block whose nx1 is 1 stores no ghosts
// (CoordinateOptions::nc1() is 1, and il() is 0). p_ref then takes the edge
// stencil instead of the hydrostatic face. Four interior cells, two per
// block, is the smallest split where that cell is an ordinary interior cell.
//
// HydroImpl passes the running face down the column only when the split is
// across processes (hydro.cpp, the x1 relay). A second block in the same
// process gets an empty anchor, and the scan starts again from that block's
// own top cell (hydro_ref_x1_scan_impl, anchor == nullptr). This is the p_ref
// seam jump reported on #250. The harness that measured it is
// chengcli/snapy study/250-t4-harness at
// 1ffb3c16d8c6e182461876a30734af73320a45a9. This test does not fix it.

// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// torch
#include <torch/torch.h>
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

// _hydro_ref_x1 is protected. A derived class may name it; the member
// pointer applies to any HydroImpl. Same access as the T4 harness.
struct RefPeek : HydroImpl {
  static auto ptr() { return &RefPeek::_hydro_ref_x1; }
};

// Return is (psf_lo, pref, dsf, dref). The seam check uses pref.
torch::Tensor pref_of(Hydro const& hydro, torch::Tensor const& w) {
  return std::get<1>(((*hydro).*RefPeek::ptr())(w));
}

Mesh make_mesh(int nb1) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_pref_local_seam.yaml");
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  return Mesh(mesh_opts);
}

torch::Tensor uniform_w(MeshBlock block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  return w;
}

double at_cell(torch::Tensor const& field, int i) {
  return field.select(-1, i).item<double>();
}

// Cell-center p_ref if the scan restarts on this block. The anchor is this
// cell's own top face, then half a cell of g*rho*dx. grav1 is stored < 0.
double restarted_pref(MeshBlock block, torch::Tensor const& w) {
  auto coord = block->pcoord;
  int iu = coord->iu();
  double g = -block->phydro->options->grav()->grav1();
  double dx = coord->dx1f.select(-1, iu).item<double>();
  double pres = w[IPR].select(-1, iu).item<double>();
  double rho = w[IDN].select(-1, iu).item<double>();
  double anchor = pres * std::exp(-g * 0.5 * dx / (pres / rho));
  return anchor + 0.5 * g * rho * dx;
}

// Uniform rho = 1 and p = 1e5, one process, dx = 1. p_ref of the cell under
// the seam is fixed by the cells above it. Splitting the column inside one
// process must not change it.
TEST(HydroRefX1, local_blocks_restart_the_reference_at_the_seam) {
  auto one = make_mesh(1);
  ASSERT_EQ(one->blocks.size(), 1u);
  auto block = one->blocks[0];
  auto w = uniform_w(block);
  auto pref = pref_of(block->phydro, w);
  int il = block->pcoord->il();
  int iu = block->pcoord->iu();
  int nint = iu - il + 1;
  ASSERT_GE(nint, 4);
  ASSERT_EQ(nint % 2, 0);
  int below = il + nint / 2 - 1;  // interior cell just under the mid seam
  int above = below + 1;
  double unsplit_below = at_cell(pref, below);
  double unsplit_above = at_cell(pref, above);

  auto two = make_mesh(2);
  ASSERT_EQ(two->blocks.size(), 2u);
  MeshBlock lower = two->blocks[0];
  MeshBlock upper = two->blocks[1];
  if (lower->pcoord->options->x1min() > upper->pcoord->options->x1min())
    std::swap(lower, upper);
  ASSERT_LT(lower->pcoord->options->x1min(), upper->pcoord->options->x1min());
  ASSERT_EQ(lower->pcoord->iu() - lower->pcoord->il() + 1, nint / 2);
  ASSERT_EQ(upper->pcoord->iu() - upper->pcoord->il() + 1, nint / 2);

  auto wu = uniform_w(upper);
  auto wl = uniform_w(lower);
  // The two blocks exchange reference rows, so they run concurrently, as the
  // mesh worker threads run them.
  torch::Tensor pu, pl;
  std::thread tu([&]() { pu = pref_of(upper->phydro, wu); });
  std::thread tl([&]() { pl = pref_of(lower->phydro, wl); });
  tu.join();
  tl.join();
  double from_above = at_cell(pu, upper->pcoord->il());
  double from_below = at_cell(pl, lower->pcoord->iu());

  // The block that still owns the domain top reproduces the unsplit cell.
  EXPECT_NEAR(from_above, unsplit_above, 1e-6)
      << "upper block p_ref " << from_above << " unsplit " << unsplit_above;

  // RED. The lower block does not. Its scan started again from its own top
  // cell, so this p_ref is not the unsplit value.
  EXPECT_NEAR(from_below, unsplit_below, 1e-6)
      << "lower block p_ref " << from_below << " unsplit " << unsplit_below
      << " restart-from-own-top " << restarted_pref(lower, wl);
}

// RED on b94e5ed. The anchor now crosses an in-process seam, but the ghost
// rows of p_ref and dref are still exchanged only between processes
// (hydro.cpp, the block gated by x1_split). After 200 steps a 4-block
// in-process column therefore leaves the one-block state. On this 32 x 8
// polytrope the difference is rho 6.4e-5, p 1.1e-6, |dv| 1.4e-4 m/s. The
// #250 tropopause column is the larger one Cheng measured at 2.49e-4.
const char* kDriftYaml = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 28.9703e-3
    density-floor: 1.e-300
    pressure-floor: 1.e-300
    limiter: false
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars

integration:
  type: rk3
  cfl: 0.5
  implicit-scheme: 0

forcing:
  const-gravity:
    grav1: -9.81

distribute:
  layout: slab
  nb2: 1
  nb3: 1
  blocks_per_process: 1

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6400., x2min: 0., x2max: 6400., x3min: -0.5, x3max: 0.5}
  cells: {nx1: 32, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic
)";

Mesh make_column(int nb1) {
  char fname[] = "/tmp/pref-drift-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << kDriftYaml;
  out.close();
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(torch::kCPU, torch::kFloat64);
  return mesh;
}

void fill_column(Mesh mesh, MeshVariables& vars) {
  constexpr double g = 9.81;
  constexpr double Rd = 287.0;
  constexpr double cp = 1004.5;
  constexpr double T0 = 300.;
  constexpr double p0 = 1.e5;
  constexpr double lapse = g / cp;
  constexpr double xc = 3200.;
  constexpr double zc = 2000.;
  constexpr double xr = 1600.;
  constexpr double zr = 800.;
  constexpr double dT = -20.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int nc1 = coord->options->nc1();
    int nc2 = coord->options->nc2();
    int nc3 = coord->options->nc3();
    auto z = coord->x1v;
    auto x = coord->x2v;
    auto Tbg = T0 - lapse * z;
    auto p = p0 * (Tbg / T0).pow(g / (Rd * lapse));
    auto bubble = ((x - xc) / xr).pow(2).view({1, nc2, 1}) +
                  ((z - zc) / zr).pow(2).view({1, 1, nc1});
    auto T = Tbg.view({1, 1, nc1}) + dT * torch::exp(-bubble);
    auto pp = p.view({1, 1, nc1}).expand({nc3, nc2, nc1});
    auto w =
        torch::zeros({mesh->blocks[b]->phydro->peos->nvar(), nc3, nc2, nc1},
                     torch::kFloat64);
    w[IDN].copy_(pp / (Rd * T));
    w[IPR].copy_(pp);
    vars[b]["hydro_w"] = w;
  }
}

void step_column(Mesh mesh, MeshVariables& vars) {
  auto dt = mesh->max_time_step(vars);
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
}

torch::Tensor column_state(Mesh mesh, MeshVariables const& vars) {
  std::vector<MeshBlock> blocks(mesh->blocks.begin(), mesh->blocks.end());
  std::sort(blocks.begin(), blocks.end(),
            [](MeshBlock const& a, MeshBlock const& b) {
              return a->pcoord->options->x1min() < b->pcoord->options->x1min();
            });
  std::vector<torch::Tensor> parts;
  for (auto const& block : blocks) {
    size_t b = 0;
    for (; b < mesh->blocks.size(); ++b)
      if (mesh->blocks[b]->pcoord->options->x1min() ==
          block->pcoord->options->x1min())
        break;
    auto coord = block->pcoord;
    auto w = vars[b].at("hydro_w");
    parts.push_back(w.slice(-1, coord->il(), coord->iu() + 1)
                        .slice(-2, coord->jl(), coord->ju() + 1));
  }
  return torch::cat(parts, -1);
}

double rel_diff(torch::Tensor const& a, torch::Tensor const& b, int var,
                double floor) {
  double da = (a[var] - b[var]).abs().max().item<double>();
  double sc = std::max(a[var].abs().max().item<double>(),
                       b[var].abs().max().item<double>());
  return da / std::max(sc, floor);
}

}  // namespace

// One process. Four blocks along x1 against one block, 200 steps. The p_ref
// and dref ghost rows go to every x1 neighbour, same-process blocks included,
// so the split column tracks the one-block state. This keeps it under 1e-12.
TEST(HydroRefX1, in_process_split_matches_one_block_after_200_steps) {
  torch::set_num_threads(1);
  auto one = make_column(1);
  auto split = make_column(4);
  ASSERT_EQ(one->blocks.size(), 1u);
  ASSERT_EQ(split->blocks.size(), 4u);

  MeshVariables v1(one->blocks.size());
  MeshVariables v4(split->blocks.size());
  fill_column(one, v1);
  fill_column(split, v4);
  one->initialize(v1);
  split->initialize(v4);

  for (int step = 0; step < 200; ++step) {
    step_column(one, v1);
    step_column(split, v4);
  }

  auto a = column_state(one, v1);
  auto b = column_state(split, v4);
  ASSERT_TRUE(a.sizes() == b.sizes()) << a.sizes() << " vs " << b.sizes();
  double rho = rel_diff(a, b, IDN, 0.);
  double pres = rel_diff(a, b, IPR, 0.);
  double vx = rel_diff(a, b, IVX, 1.);
  double state = std::max(rho, std::max(pres, vx));
  std::printf(
      "4 x1 blocks vs 1 after 200 steps: rel diff rho %.3e p %.3e vx %.3e\n",
      rho, pres, vx);
  // round-off of the chained in-process anchor scan: vx 5.4e-14, 200 steps
  EXPECT_LE(state, 1e-12) << "rho " << rho << " p " << pres << " vx " << vx;
}
