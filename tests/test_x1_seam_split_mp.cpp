// Two processes, test_x1_seam_split's column split into two x1 blocks with one
// block per process, so the x1 seam exchange crosses ranks. Rank 0 also steps
// the one-block column and the two-block column of one process locally; the
// two ranks' interiors are summed into one zero-padded column and compared on
// rank 0: max over rho, rho v1 and E of |split - ref| / max|ref|.
//
// The arm is read from the environment, one ctest entry each
// (tests/CMakeLists.txt): switches off checks the cross-rank split against
// the in-process split only; X1_SEAM_ARM=face (gravity-work: face with its
// corrected-PE work, whose seam slope is centred across ranks, #303),
// SNAP_X1_CENTROID_EXACT and SNAP_WB_REF4 check it against the in-process
// split and against one block.
// The face arm also checks the split E + P summed over the two ranks.
//
// Run: torchrun --no-python --nproc-per-node=2 test_x1_seam_split_mp.<build>

// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/hydro/wb_ref4.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

constexpr double kR0 = 5.0, kLz = 2.0, kGamma = 1.4, kSeed = 0.05;

// test_x1_seam_split's column
std::string column_yaml(int nx1, double nh, char const* gw) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: spherical-polar
  bounds: {x1min: %g, x1max: %g, x2min: 1.5207963267948966, x2max: 1.6207963267948966, x3min: 0., x3max: 0.1}
  cells: {nx1: %d, nx2: 1, nx3: 1, nghost: 3}
dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    temperature-floor: 1.e-12
    limiter: true
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars
integration:
  type: rk3
  cfl: 0.4
  implicit-scheme: 0
forcing:
  const-gravity: {grav1: -1., non-hydrostatic: %g, gravity-work: %s}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)",
                kR0, kR0 + kLz, nx1, nh, gw);
  return buf;
}

// nb1 x1 blocks; split: one block per process (the 2 ranks), else all blocks
// in this process alone, with no process group
Mesh make_column(int nb1, bool split, int nx1, double nh, char const* gw) {
  char fname[] = "/tmp/x1-seam-split-mp-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << column_yaml(nx1, nh, gw);
  out.close();
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  block_opts->hydro()->grav()->grav1(-1.);
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  if (!split) block_opts->layout()->process_rank(0).process_world_size(1);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(split ? 1 : nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(torch::kCPU, torch::kFloat64);
  return mesh;
}

// test_x1_seam_split's fill_column
void fill_column(Mesh mesh, MeshVariables& vars, bool cold) {
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int nc1 = coord->options->nc1();
    auto z = coord->x1v - kR0;
    auto temp = cold ? 0.25 * torch::exp(1.1 - z) : torch::ones_like(z);
    auto p = cold ? torch::exp(-4. * (torch::exp(z - 1.1) - std::exp(-1.1)))
                  : torch::exp(-z);
    auto rho =
        p / temp * (1. + 0.02 * torch::exp(-((z - 0.5 * kLz) / 0.2).square()));
    auto in = torch::logical_and(z > 0., z < kLz);
    auto v1 =
        torch::where(in, kSeed * std::sqrt(kGamma) * torch::sin(M_PI * z / kLz),
                     torch::zeros_like(z));
    auto w = torch::zeros({mesh->blocks[b]->phydro->peos->nvar(),
                           coord->options->nc3(), coord->options->nc2(), nc1},
                          coord->x1v.options());
    w[IDN].copy_(rho.view({1, 1, nc1}));
    w[IPR].copy_(p.view({1, 1, nc1}));
    w[IVX].copy_(v1.view({1, 1, nc1}));
    vars[b]["hydro_w"] = w;
  }
}

void step(Mesh mesh, MeshVariables& vars, double dt) {
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
}

// the blocks' interiors in one zero-padded nx1 column, x1 order
torch::Tensor padded_column(Mesh mesh, MeshVariables const& vars, int nx1) {
  torch::Tensor full;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    auto u = vars[b].at("hydro_u");
    if (!full.defined())
      full = torch::zeros({u.size(0), u.size(1), u.size(2), nx1}, u.options());
    int i0 = std::lround((coord->options->x1min() - kR0) / (kLz / nx1));
    full.narrow(-1, i0, coord->iu() - coord->il() + 1)
        .copy_(u.slice(-1, coord->il(), coord->iu() + 1));
  }
  return full;
}

torch::Tensor allreduce_sum(Mesh mesh, torch::Tensor t) {
  std::vector<torch::Tensor> values = {t};
  mesh->blocks.front()->get_layout()->comm->allreduce(values,
                                                      c10d::ReduceOp::SUM);
  return values[0];
}

// sum V [E + rho phi(x1v) - g1 <(x1 - x1v)^2> s[rho]] over the local blocks:
// the E + P that the corrected-PE work of gravity-work: face conserves
// (meshblock.cpp pe=)
double energy_p(Mesh mesh, MeshVariables const& vars) {
  double total = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int is = coord->il(), ie = coord->iu() + 1;
    auto u = vars[b].at("hydro_u");
    auto rho = u[IDN];
    auto pe = rho * coord->x1v;
    // across the seam the ghost rho is the neighbour's (exchanged after every
    // stage), so this is the P of one block (#303)
    auto [below, above] = mesh->blocks[b]->phydro->x1_neighbors();
    pe.slice(-1, is, ie) -=
        corrected_pe_work_seam(rho, coord->x1f, coord->x1v, is, ie, -1.,
                               X1Measure::radial, below >= 0, above >= 0);
    auto vol = coord->cell_volume();
    total += ((u[IPR] + pe) * vol).slice(-1, is, ie).sum().item<double>();
  }
  return total;
}

// max over rho, rho v1 and E of |a - ref| / max|ref|
double rel_gap(torch::Tensor const& a, torch::Tensor const& ref) {
  double gap = 0.;
  for (int var : {IDN, IVX, IPR}) {
    double d = (a[var] - ref[var]).abs().max().item<double>();
    gap = std::max(gap, d / ref[var].abs().max().item<double>());
  }
  return gap;
}

struct Gaps {
  double one = 0., two = 0., drift = 0.;
};

// the 2-rank split after nstep steps against one block and against the
// in-process split, both on rank 0 only; drift: the split E + P, both ranks
Gaps split_gaps(int rank, int nx1, double nh, char const* gw, int nstep,
                bool cold) {
  // the 2-rank split first, the in-process split last: process-local layouts
  // are registered by (process rank, local block), so the last one owns them
  auto mp = make_column(2, true, nx1, nh, gw);
  EXPECT_EQ(mp->blocks.size(), 1u);
  Mesh one = nullptr, two = nullptr;
  if (rank == 0) {
    one = make_column(1, false, nx1, nh, gw);
    two = make_column(2, false, nx1, nh, gw);
    EXPECT_EQ(one->blocks.size(), 1u);
    EXPECT_EQ(two->blocks.size(), 2u);
  }
  MeshVariables vm(1), v1(1), v2(2);
  fill_column(mp, vm, cold);
  mp->initialize(vm);
  if (rank == 0) {
    fill_column(one, v1, cold);
    fill_column(two, v2, cold);
    one->initialize(v1);
    two->initialize(v2);
  }
  double dt = 0.3 * (kLz / nx1) / std::sqrt(kGamma);
  auto global_ep = [&] {
    return allreduce_sum(mp, torch::tensor({energy_p(mp, vm)},
                                           torch::dtype(torch::kFloat64)))
        .item<double>();
  };
  Gaps g;
  double ep0 = global_ep();
  for (int n = 0; n < nstep; ++n) {
    step(mp, vm, dt);
    if (rank == 0) {
      step(one, v1, dt);
      step(two, v2, dt);
    }
    g.drift = std::max(g.drift, std::abs(global_ep() - ep0) / ep0);
  }
  auto c = allreduce_sum(mp, padded_column(mp, vm, nx1));
  if (rank == 0) {
    g.one = rel_gap(c, padded_column(one, v1, nx1));
    g.two = rel_gap(c, padded_column(two, v2, nx1));
  }
  return g;
}

}  // namespace

TEST(X1SeamSplitMp, split_across_two_ranks_matches_one_process) {
  bool centroid = std::getenv("SNAP_X1_CENTROID_EXACT") != nullptr;
  char const* seam_arm = std::getenv("X1_SEAM_ARM");
  bool radial = seam_arm && std::string(seam_arm) == "face";
  bool wb_ref4 = std::getenv("SNAP_WB_REF4") != nullptr;
  ASSERT_LE(centroid + radial + wb_ref4, 1) << "one switch per arm";
  ASSERT_EQ(x1_centroid_exact_enabled(), centroid);
  if (wb_ref4) ASSERT_TRUE(wb_ref4_enabled());
  torch::set_num_threads(1);

  auto probe = make_column(2, true, 16, 1., "cell");
  auto layout = probe->blocks.front()->get_layout();
  ASSERT_TRUE(layout->has_process_group() &&
              layout->options->process_world_size() == 2 &&
              probe->blocks.size() == 1)
      << "the two x1 blocks must be in two processes";
  int rank = layout->options->process_rank();
  char const* arm = centroid  ? "SNAP_X1_CENTROID_EXACT"
                    : radial  ? "X1_SEAM_ARM=face"
                    : wb_ref4 ? "SNAP_WB_REF4"
                              : "switches off";

  // EXPECT, not ASSERT, below: both ranks must reach every collective
  if (radial) {
    for (int nx1 : {32, 64, 128}) {
      auto g = split_gaps(rank, nx1, 1., "face", 20 * nx1 / 32, false);
      if (rank == 0) {
        std::printf(
            "%s, nz %d: 2 ranks vs 2 blocks %.3e, vs 1 block %.3e; split E+P "
            "drift %.3e\n",
            arm, nx1, g.two, g.one, g.drift);
        EXPECT_LE(g.two, 1e-13) << "nz " << nx1;
        // the seam takes one block's centred slope (#303): one block's state,
        // to the round-off of the other arms
        EXPECT_LE(g.one, 1e-13) << "nz " << nx1;
      }
      EXPECT_LE(g.drift, 1e-13) << "rank " << rank << ", nz " << nx1;
    }
    return;
  }
  // SNAP_WB_REF4: the cold column, its flag switching on above the seam
  int nx1 = wb_ref4 ? 16 : 32;
  for (double nh : {1., 0.}) {
    auto g = split_gaps(rank, nx1, nh, "cell", 20, wb_ref4);
    if (rank != 0) continue;
    std::printf(
        "%s, non-hydrostatic %g: 2 ranks vs 2 blocks %.3e, vs 1 block %.3e\n",
        arm, nh, g.two, g.one);
    // round-off: 20 steps of the column, as test_x1_seam_split
    EXPECT_LE(g.two, 1e-13) << "non-hydrostatic " << nh;
    if (centroid || wb_ref4)
      EXPECT_LE(g.one, 1e-13) << "non-hydrostatic " << nh;
  }
}
