// external
#include <gtest/gtest.h>

// C/C++
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

// POSIX
#include <unistd.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/stage_executor.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

// V3 B3 (d)/(e): a Mesh whose blocks disagree runs eager. Two Cartesian
// blocks along x2 with conversion-order 4 (the hydro step exchanges primitive
// ghosts across the blocks inside the stage, a cut) and an outflow x2-outer
// boundary: only the outer block refuses capture ("an outflow boundary is set
// with conversion-order 4"). Before the rule, the inner block warmed up and
// captured while the outer one ran eagerly, so their in-stage exchanges met
// the same-process rendezvous a different number of times and the Mesh hung.
// Now the whole Mesh runs eager, the capturable block naming "a Mesh whose
// blocks disagree", strict refuses before any launch, and the result equals
// an eager Mesh bitwise. Blocks whose execution modes differ are the other
// half of the rule. Every Mesh call runs under a watchdog: a hang fails the
// test in bounded time instead of stalling ctest.
namespace {

constexpr int kWatchdogSeconds = 240;

// runs f on a thread; if it does not finish within the watchdog, the test
// binary exits with a failure (a hung thread cannot be joined)
void bounded(char const* what, std::function<void()> f) {
  std::mutex m;
  std::condition_variable cv;
  bool done = false;
  std::exception_ptr error;
  std::thread t([&]() {
    try {
      f();
    } catch (...) {
      error = std::current_exception();
    }
    std::lock_guard<std::mutex> lock(m);
    done = true;
    cv.notify_all();
  });
  std::unique_lock<std::mutex> lock(m);
  if (!cv.wait_for(lock, std::chrono::seconds(kWatchdogSeconds),
                   [&]() { return done; })) {
    std::fprintf(stderr,
                 "[  FAILED  ] %s did not finish within %d s: the Mesh hung "
                 "(its blocks' in-stage exchanges never met)\n",
                 what, kWatchdogSeconds);
    std::fflush(stderr);
    std::_Exit(1);
  }
  lock.unlock();
  t.join();
  if (error) std::rethrow_exception(error);
}

std::string deck(char const* execution, bool strict, char const* precision,
                 char const* x2_outer) {
  char buf[2560];
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
distribute: {layout: slab, nb2: 2, nb3: 1, verbose: false}
dynamics:
  precision: %s
  anomaly: true
  perturbation-flux: true
  conversion-order: 4
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
    x2-outer: %s
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
                precision, x2_outer, execution, strict ? "true" : "false");
  return buf;
}

Mesh make_mesh(char const* execution, bool strict, char const* precision,
               char const* x2_outer, MeshVariables& vars) {
  char fname[] = "/tmp/test-mesh-disagree-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << deck(execution, strict, precision, x2_outer);
  auto bo = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  auto mo = MeshOptionsImpl::create();
  mo->block(bo);
  mo->blocks_per_process(2);
  auto mesh = Mesh(mo);
  mesh->to(torch::Device(torch::kCUDA, 0));
  vars.assign(mesh->blocks.size(), Variables{});
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto b = mesh->blocks[i];
    auto coord = b->pcoord;
    auto z = coord->x1v.view({1, 1, -1});
    auto x = coord->x2v.view({1, -1, 1});
    auto w = torch::zeros({b->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          coord->x1v.options().dtype(torch::kFloat64));
    w[IDN] = 1.2 * torch::exp(-z / 8000.) + 0. * x;
    w[IPR] = 1.e5 * torch::exp(-z / 8000.) *
             (1. + 0.02 * torch::exp(-((x - 6400.) * (x - 6400.) +
                                       (z - 2000.) * (z - 2000.)) /
                                     1.e6));
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  return mesh;
}

void step(Mesh& mesh, MeshVariables& vars, double dt) {
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int s = 0; s < nstage; ++s) mesh->forward(vars, dt, s);
}

// a reason that starts with `what` (a Mesh reason quotes the other block's)
bool names(std::vector<std::string> const& reasons, char const* what) {
  for (auto const& r : reasons)
    if (r.rfind(what, 0) == 0) return true;
  return false;
}

// graph (not strict) vs eager over 4 steps with dt changing, bitwise
void graph_equals_eager(Mesh& graph, MeshVariables& gv, Mesh& eager,
                        MeshVariables& ev, char const* precision) {
  double dt0 = eager->max_time_step(ev);
  for (int n = 0; n < 4; ++n) {
    double dt = 0.3 * dt0 * (1. + 0.05 * n);
    bounded("a graph Mesh step", [&]() { step(graph, gv, dt); });
    step(eager, ev, dt);
    for (size_t i = 0; i < gv.size(); ++i)
      for (auto name : {"hydro_u", "hydro_w"})
        ASSERT_TRUE(torch::equal(gv[i].at(name), ev[i].at(name)))
            << precision << " step " << n << " block " << i << " " << name;
  }
}

}  // namespace

TEST(MeshDisagree, one_refusing_block_runs_the_whole_mesh_eager) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  for (auto precision : {"float64", "float32"}) {
    MeshVariables gv, ev;
    auto graph = make_mesh("cuda-graph", false, precision, "outflow", gv);
    auto eager = make_mesh("eager", false, precision, "outflow", ev);
    ASSERT_EQ(graph->blocks.size(), 2u);
    // the outer block refuses for itself; the inner one for the Mesh
    std::vector<std::vector<std::string>> why;
    for (size_t i = 0; i < 2; ++i)
      why.push_back(
          graph->blocks[i]->execution_capability(gv[i]).eager_reasons);
    int outflow = names(why[0], "an outflow boundary") ? 0 : 1;
    EXPECT_TRUE(names(why[outflow], "an outflow boundary")) << precision;
    EXPECT_TRUE(names(why[1 - outflow], "a Mesh whose blocks disagree"))
        << precision << ": the capturable block must run eager with the Mesh";
    graph_equals_eager(graph, gv, eager, ev, precision);
    auto r = graph->execution_report();
    EXPECT_EQ(r.at("captures"), 0) << precision << ": nothing captures";
    EXPECT_EQ(r.at("graphs"), 0) << precision;
  }
}

TEST(MeshDisagree, strict_refuses_a_mesh_whose_blocks_disagree) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  MeshVariables gv;
  auto graph = make_mesh("cuda-graph", true, "float64", "outflow", gv);
  bool threw = false;
  std::string msg;
  bounded("a strict Mesh step", [&]() {
    try {
      step(graph, gv, 0.1);
    } catch (c10::Error const& e) {
      threw = true;
      msg = e.what();
    }
  });
  EXPECT_TRUE(threw) << "a strict Mesh whose blocks disagree ran";
  EXPECT_NE(msg.find("cannot run as a CUDA graph"), std::string::npos) << msg;
  EXPECT_NE(msg.find("a Mesh whose blocks disagree"), std::string::npos) << msg;
  EXPECT_EQ(graph->execution_report().at("captures"), 0);
}

TEST(MeshDisagree, blocks_whose_execution_modes_differ_run_eager) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  MeshVariables gv, ev;
  auto graph = make_mesh("cuda-graph", false, "float64", "reflecting", gv);
  auto eager = make_mesh("eager", false, "float64", "reflecting", ev);
  // one block switched to eager after construction: the modes differ
  graph->blocks[1]->options->execution("eager");
  auto why = graph->blocks[0]->execution_capability(gv[0]).eager_reasons;
  EXPECT_TRUE(names(why, "a Mesh whose blocks disagree"));
  graph_equals_eager(graph, gv, eager, ev, "float64");
  EXPECT_EQ(graph->execution_report().at("captures"), 0)
      << "the cuda-graph block must not capture alone";
}
