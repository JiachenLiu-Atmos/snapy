// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

// The capture save list (StageExecutor::save_list) must hold everything a
// stage writes, or its three warm-up passes and the capture pass leave their
// writes behind. The register monitor's amounts (_step_amount, _stage_amount,
// _stage_post) are written in every stage (apply_prescribed books
// "boundary.prescribed" on the walls; end_stage folds the stage into the step
// amount), so a cuda-graph block must report the same amounts as an eager
// one, step by step, bitwise, with dt changing every step. Since V3 D dt is a
// graph input, so each stage is captured once (its warm-up and capture are
// the passes that leak) and replayed at every dt.
namespace {

std::string deck(char const* execution) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
reference-state: {Tref: 0., Pref: 1.e5}
species:
  - name: dry
    composition: {O: 0.4200039819342909, N: 1.560014790041652, Ar: 0.01}
    cv_R: 2.5
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6400., x2min: 0., x2max: 12800., x3min: 0., x3max: 1.}
  cells: {nx1: 32, nx2: 64, nx3: 1, nghost: 3}
distribute: {layout: slab, nb2: 1, nb3: 1, verbose: false}
dynamics:
  equation-of-state:
    type: ideal-moist
    limiter: false
    density-floor: 1.e-10
    pressure-floor: 1.e-10
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: hllc}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
integration:
  type: rk3
  cfl: 0.9
  implicit-scheme: 0
  nlim: 1
  ncycle_out: 0
  execution: %s
  execution-strict: %s
forcing:
  const-gravity: {grav1: -9.8}
)",
                execution,
                std::string(execution) == "cuda-graph" ? "true" : "false");
  return buf;
}

MeshBlock make_block(char const* execution, Variables& vars) {
  char fname[] = "/tmp/test-graph-monitor-amounts-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << deck(execution);
  auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname);
  block->to(torch::Device(torch::kCUDA, 0));
  auto coord = block->pcoord;
  auto z = coord->x1v.view({1, 1, -1});
  auto x = coord->x2v.view({1, -1, 1});
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        coord->x1v.options());
  // a stratified column with a warm pressure bump: flow reaches every wall
  w[IDN] = 1.2 * torch::exp(-z / 8000.) + 0. * x;
  w[IPR] = 1.e5 * torch::exp(-z / 8000.) *
           (1. + 0.02 * torch::exp(-((x - 6400.) * (x - 6400.) +
                                     (z - 2000.) * (z - 2000.)) /
                                   1.e6));
  vars["hydro_w"] = w;
  block->initialize(vars);
  return block;
}

}  // namespace

TEST(GraphSaveList, monitor_amounts_match_eager_bitwise) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";

  Variables gvars, evars;
  auto graph = make_block("cuda-graph", gvars);
  auto eager = make_block("eager", evars);
  ASSERT_TRUE(graph->execution_capability(gvars).eager_reasons.empty())
      << "the deck must capture, or this test gates nothing";

  auto& gmon = graph->phydro->flux_register().monitor();
  auto& emon = eager->phydro->flux_register().monitor();
  int nstage = graph->pintg->stages.size();
  double dt0 = eager->max_time_step(evars);
  double largest = 0.;
  for (int n = 1; n <= 10; ++n) {
    double dt = 0.3 * dt0 * (1. + 0.2 * ((7 * n) % 5) / 4.);  // new every step
    for (int s = 0; s < nstage; ++s) {
      graph->forward(gvars, dt, s);
      eager->forward(evars, dt, s);
    }
    ASSERT_TRUE(torch::equal(gvars.at("hydro_u"), evars.at("hydro_u")))
        << "step " << n << ": the state already differs";
    ASSERT_EQ(gmon.amount_names(), emon.amount_names()) << "step " << n;
    for (auto const& name : emon.amount_names()) {
      auto e = emon.amounts(name), g = gmon.amounts(name);
      largest = std::max(largest, e.abs().max().item<double>());
      EXPECT_TRUE(torch::equal(g, e))
          << "step " << n << ", amount " << name << ": graph " << g.cpu()
          << " vs eager " << e.cpu();
    }
  }
  EXPECT_GT(largest, 0.) << "no amount was ever nonzero: nothing was compared";
  auto report = graph->execution_report();
  EXPECT_EQ(report.at("captures"), nstage)
      << "dt is a graph input (V3 D): one capture per stage, replayed at "
         "every dt";
  EXPECT_EQ(report.at("eager"), 0);
}

// V3 S4: a rolled-back attempt on the cuda-graph path. The redo
// replays the captured stages, so the monitor's host stage bookkeeping
// (_stage, _ended, the stage weights) must survive FluxRegister::rollback():
// when rollback reset it, fold_ read S = 0 for the redo and for every
// replayed step after it, dropping every rate and the later stages' posts
// while the eager block, whose redo runs begin_stage/end_stage again, was
// right. The redo's amounts and the next step's must equal eager's, bitwise,
// with no eager fallback and no new capture.
TEST(GraphSaveList, monitor_amounts_match_eager_after_a_rollback) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";

  Variables gvars, evars;
  auto graph = make_block("cuda-graph", gvars);
  auto eager = make_block("eager", evars);
  ASSERT_TRUE(graph->execution_capability(gvars).eager_reasons.empty())
      << "the deck must capture, or this test gates nothing";

  auto& gmon = graph->phydro->flux_register().monitor();
  auto& emon = eager->phydro->flux_register().monitor();
  int nstage = graph->pintg->stages.size();
  double dt0 = eager->max_time_step(evars);
  auto step = [&](double dt) {
    for (int s = 0; s < nstage; ++s) {
      graph->forward(gvars, dt, s);
      eager->forward(evars, dt, s);
    }
  };
  auto same_amounts = [&](char const* when) {
    ASSERT_TRUE(torch::equal(gvars.at("hydro_u"), evars.at("hydro_u")))
        << when << ": the state differs";
    double largest = 0.;
    for (auto const& name : emon.amount_names()) {
      auto e = emon.amounts(name), g = gmon.amounts(name);
      largest = std::max(largest, e.abs().max().item<double>());
      EXPECT_TRUE(torch::equal(g, e))
          << when << ", amount " << name << ": graph " << g.cpu()
          << " vs eager " << e.cpu();
    }
    EXPECT_GT(largest, 0.) << when << ": no amount was nonzero";
  };
  step(0.30 * dt0);  // captured
  step(0.33 * dt0);  // replayed
  // a rejected attempt, in MeshBlock::check_redo's order: the register rolls
  // back, the state is restored, the step is redone at a smaller dt
  auto gu = gvars.at("hydro_u").clone(), eu = evars.at("hydro_u").clone();
  auto gw = gvars.at("hydro_w").clone(), ew = evars.at("hydro_w").clone();
  step(0.36 * dt0);
  graph->phydro->flux_register().rollback();
  eager->phydro->flux_register().rollback();
  gvars.at("hydro_u").copy_(gu);
  evars.at("hydro_u").copy_(eu);
  gvars.at("hydro_w").copy_(gw);
  evars.at("hydro_w").copy_(ew);
  step(0.18 * dt0);
  same_amounts("the redo");
  step(0.30 * dt0);
  same_amounts("the step after the redo");
  auto report = graph->execution_report();
  EXPECT_EQ(report.at("captures"), nstage) << "the redo must replay";
  EXPECT_EQ(report.at("eager"), 0);
}
