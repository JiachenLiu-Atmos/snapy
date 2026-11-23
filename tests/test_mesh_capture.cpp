// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

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

// V3 B2: a Mesh's blocks run their stages as CUDA graphs, one graph per
// (block, stage, segment), captured on the block worker pool (thread-local
// capture mode), with the Mesh's own exchange, global phase and chemistry
// between them as before. Two Cartesian blocks along x2 (a dry deck: no
// scalar or species, so no exchange runs inside a stage) equal an eager Mesh
// bitwise, FP64 and FP32, with dt changing.
// V3 B3: a within-stage cross-block exchange is a cut (pack and unpack in
// the segment's graphs, the same-process rendezvous on the host between
// them): the scalar positivity limiter's theta sync (a scalar with
// flux-covariance) and conversion-order 4's primitive ghosts capture, all
// blocks together, and equal an eager Mesh bitwise, FP64 and FP32.
namespace {

std::string deck(char const* execution, bool strict, char const* precision,
                 bool scalar, int conversion_order = 2) {
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
  conversion-order: %d
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
%s)",
                precision, conversion_order, execution,
                strict ? "true" : "false",
                scalar ? "scalar:\n  nvar: 1\n  names: [tracer]\n" : "");
  return buf;
}

Mesh make_mesh(char const* execution, bool strict, char const* precision,
               bool scalar, MeshVariables& vars, int conversion_order = 2) {
  char fname[] = "/tmp/test-mesh-capture-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << deck(execution, strict, precision, scalar,
                               conversion_order);
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
    if (scalar)
      vars[i]["scalar_r"] =
          (0.001 * (z / 6400.) + 0. * x).unsqueeze(0).to(torch::kFloat64);
  }
  mesh->initialize(vars);
  return mesh;
}

void step(Mesh& mesh, MeshVariables& vars, double dt) {
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int s = 0; s < nstage; ++s) mesh->forward(vars, dt, s);
}

}  // namespace

TEST(MeshCapture, two_blocks_capture_and_equal_an_eager_mesh) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  for (auto precision : {"float64", "float32"}) {
    MeshVariables gv, ev;
    auto graph = make_mesh("cuda-graph", true, precision, false, gv);
    auto eager = make_mesh("eager", false, precision, false, ev);
    ASSERT_EQ(graph->blocks.size(), 2u);
    for (auto& b : graph->blocks)
      ASSERT_TRUE(b->execution_capability(gv[0]).eager_reasons.empty())
          << precision << ": the two-block dry deck must capture";
    int nstage = graph->blocks.front()->pintg->stages.size();
    double dt0 = eager->max_time_step(ev);
    int nsteps = 6;
    for (int n = 0; n < nsteps; ++n) {
      double dt = n < 2 ? 0.3 * dt0 : 0.3 * dt0 * (1. + 0.05 * n);
      step(graph, gv, dt);
      step(eager, ev, dt);
      for (size_t i = 0; i < gv.size(); ++i)
        for (auto name : {"hydro_u", "hydro_w"})
          ASSERT_TRUE(torch::equal(gv[i].at(name), ev[i].at(name)))
              << precision << " step " << n << " block " << i << " " << name;
    }
    auto r = graph->execution_report();
    // two constant-dt steps, then a new dt every step: per block, one capture
    // per stage, then replays only (dt is a graph input, V3 D)
    EXPECT_EQ(r.at("captures"), 2 * nstage) << precision;
    EXPECT_EQ(r.at("replays"), 2 * nstage * nsteps) << precision;
    EXPECT_EQ(r.at("graphs"), 2 * nstage) << precision;
    EXPECT_EQ(r.at("pools"), 2) << precision << ": one pool per block";
    EXPECT_EQ(r.at("eager"), 0) << precision;
    EXPECT_EQ(r.at("host"), 0)
        << precision << ": a Mesh exchanges last, outside";
  }
}

// graph vs eager over 6 steps (2 at one dt, then a new dt every step),
// bitwise after each; returns the graph Mesh's execution report
std::map<std::string, int64_t> graph_equals_eager(char const* precision,
                                                  bool scalar, int co) {
  MeshVariables gv, ev;
  auto graph = make_mesh("cuda-graph", true, precision, scalar, gv, co);
  auto eager = make_mesh("eager", false, precision, scalar, ev, co);
  for (size_t i = 0; i < graph->blocks.size(); ++i) {
    auto c = graph->blocks[i]->execution_capability(gv[i]);
    std::string why;
    for (auto const& r : c.eager_reasons) why += "\n  - " + r;
    EXPECT_TRUE(c.eager_reasons.empty()) << precision << ": block " << i << why;
  }
  double dt0 = eager->max_time_step(ev);
  for (int n = 0; n < 6; ++n) {
    double dt = n < 2 ? 0.3 * dt0 : 0.3 * dt0 * (1. + 0.05 * n);
    step(graph, gv, dt);
    step(eager, ev, dt);
    for (size_t i = 0; i < gv.size(); ++i)
      for (auto name : {"hydro_u", "hydro_w", "scalar_s", "scalar_r"}) {
        if (!gv[i].count(name)) continue;
        EXPECT_TRUE(torch::equal(gv[i].at(name), ev[i].at(name)))
            << precision << " step " << n << " block " << i << " " << name
            << ": max |graph - eager| "
            << (gv[i].at(name).to(torch::kFloat64) -
                ev[i].at(name).to(torch::kFloat64))
                   .abs()
                   .max()
                   .item<double>();
      }
  }
  return graph->execution_report();
}

TEST(MeshCapture, a_within_stage_theta_sync_is_a_cut_and_equals_eager) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  for (auto precision : {"float64", "float32"}) {
    auto r = graph_equals_eager(precision, /*scalar=*/true, /*co=*/2);
    // one capture per (block, stage, segment), all blocks together; dt is a
    // graph input, so the new dts replay
    EXPECT_EQ(r.at("captures"), 2 * 3) << precision;
    EXPECT_EQ(r.at("replays"), 2 * 3 * 6) << precision;
    EXPECT_EQ(r.at("eager"), 0) << precision;
  }
}

TEST(MeshCapture, conversion_order_4_primitive_ghosts_are_a_cut) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  for (auto precision : {"float64", "float32"}) {
    auto r = graph_equals_eager(precision, /*scalar=*/false, /*co=*/4);
    EXPECT_EQ(r.at("captures"), 2 * 3) << precision;
    EXPECT_EQ(r.at("eager"), 0) << precision;
  }
}
