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
#include <snap/mesh/stage_executor.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

// V3 A2/A3a: a block's stage is a schedule of ops in segments: the stage-entry
// exchange is a host-exchange segment (eager by design, a boundary), then
// advance_local, captured as one graph per (block, stage, segment).
// advance_local's dt reads (the RK combine's w2 dt and
// FluxRegister::end_stage(dt)) take the block's device dt (V3 D), so the
// segment keeps no dt in its key: a new dt replays. The result equals eager
// bitwise.
namespace {

std::string deck(char const* execution, bool strict) {
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
                execution, strict ? "true" : "false");
  return buf;
}

MeshBlock make_block(torch::Device device, char const* execution, bool strict,
                     Variables& vars) {
  char fname[] = "/tmp/test-stage-segments-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << deck(execution, strict);
  auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname);
  block->to(device);
  auto coord = block->pcoord;
  auto z = coord->x1v.view({1, 1, -1});
  auto x = coord->x2v.view({1, -1, 1});
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        coord->x1v.options());
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

TEST(StageSegments, exchange_is_a_host_segment_then_one_graph) {
  Variables vars;
  auto block = make_block(torch::kCPU, "cuda-graph", false, vars);
  auto const& sched = block->executor()->schedule(0);
  ASSERT_EQ(sched.segments().size(), 2u);
  auto const& ex = sched.segments()[0];
  EXPECT_EQ(ex.names, (std::vector<std::string>{"exchange"}));
  EXPECT_EQ(ex.kind, SegmentKind::host_exchange);
  EXPECT_FALSE(ex.capturable);
  EXPECT_FALSE(ex.reads_host_dt);
  auto const& seg = sched.segments()[1];
  EXPECT_EQ(seg.names, (std::vector<std::string>{"advance_local"}));
  EXPECT_EQ(seg.kind, SegmentKind::compute);
  EXPECT_TRUE(seg.capturable);
  EXPECT_FALSE(seg.reads_host_dt)
      << "the RK combine (w2 dt) and end_stage(dt) read the device dt (V3 D)";

  // on CPU the case is eager: the graph row carries the case's reason; the
  // host-exchange row is a boundary, not a refusal
  auto c = block->execution_capability(vars);
  ASSERT_EQ(c.segments.size(), 2u);
  EXPECT_TRUE(c.segments[0].host);
  EXPECT_TRUE(c.segments[0].reasons.empty());
  EXPECT_FALSE(c.segments[1].host);
  EXPECT_FALSE(c.segments[1].capturable);
  EXPECT_FALSE(c.segments[1].reads_host_dt);
  EXPECT_EQ(c.segments[1].ops, seg.names);
  EXPECT_EQ(c.segments[1].reasons, c.eager_reasons);
  EXPECT_FALSE(c.eager_reasons.empty());
}

TEST(StageSegments, strict_names_the_eager_segment) {
  Variables vars;
  auto block = make_block(torch::kCPU, "cuda-graph", true, vars);
  try {
    block->forward(vars, 0.1, 0);
    FAIL() << "a strict cuda-graph block on CPU ran";
  } catch (c10::Error const& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("cannot run as a CUDA graph"), std::string::npos) << msg;
    EXPECT_NE(msg.find("segment 1 (advance_local) of block 0, "
                       "stage 0 runs eager"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("not on a CUDA device"), std::string::npos) << msg;
  }
}

TEST(StageSegments, graph_equals_eager_and_a_new_dt_replays) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  torch::Device cuda(torch::kCUDA, 0);
  Variables gvars, evars;
  auto graph = make_block(cuda, "cuda-graph", true, gvars);
  auto eager = make_block(cuda, "eager", false, evars);
  auto c = graph->execution_capability(gvars);
  ASSERT_TRUE(c.eager_reasons.empty())
      << "the host-exchange segment is no reason for eager";
  ASSERT_EQ(c.segments.size(), 2u);
  EXPECT_TRUE(c.segments[0].host);
  EXPECT_TRUE(c.segments[1].capturable);

  int nstage = graph->pintg->stages.size();
  double dt0 = eager->max_time_step(evars);
  auto step = [&](double dt, int n) {
    for (int s = 0; s < nstage; ++s) {
      graph->forward(gvars, dt, s);
      eager->forward(evars, dt, s);
    }
    for (auto name : {"hydro_u", "hydro_w"})
      ASSERT_TRUE(torch::equal(gvars.at(name), evars.at(name)))
          << "step " << n << " " << name;
  };
  // a constant dt: one capture per (block, stage, segment), then replays
  for (int n = 1; n <= 3; ++n) step(0.3 * dt0, n);
  auto r = graph->execution_report();
  EXPECT_EQ(r.at("captures"), nstage);
  EXPECT_EQ(r.at("replays"), 3 * nstage);
  EXPECT_EQ(r.at("graphs"), nstage);
  // a new dt every step: dt is a graph input (V3 D), so the graphs replay
  for (int n = 4; n <= 6; ++n) step(0.3 * dt0 * (1. + 0.05 * n), n);
  r = graph->execution_report();
  EXPECT_EQ(r.at("captures"), nstage);
  EXPECT_EQ(r.at("replays"), 6 * nstage);
  EXPECT_EQ(r.at("graphs"), nstage);
  EXPECT_EQ(r.at("eager"), 0);
  EXPECT_EQ(r.at("host"), 6 * nstage) << "one exchange per stage, on the host";
}
