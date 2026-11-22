// V3 lane X, X4: the no-source baseline of X.rst section 5's chemistry-slot
// gate. One deck, test_parentless_cloud_nb1.yaml, as one block against two
// (the split of tests/test_parentless_cloud_nb1.cpp:33-45) through both
// branches of Mesh::forward; the EOS limiter set false before the Mesh is
// built; filled as test_parentless_cloud_nb1.cpp:57-66 with deficit 0. Every
// owned cell of hydro_u after each of 10 steps must match bitwise. (The deck
// has no scalars, so scalar_s is not compared.) That baseline runs the
// closure and the fixer off, so it never reaches global_phase_sum; a second
// test does, through Mesh::forward's N-block global phase: six cubed-sphere
// panels with conserve-axial-momentum exact (its closure) keep their absolute
// axial angular momentum only if the panels' partial sums are summed.

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/mesh/mesh.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

//! the owned hydro_u of the global column after each step
std::vector<torch::Tensor> run_column(int nb1, torch::Device device,
                                      torch::Dtype dtype, int nstep) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_parentless_cloud_nb1.yaml");
  block_opts->hydro()->eos()->limiter() = false;
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);

  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto coord = block->pcoord;
    int il = coord->il(), iu = coord->iu();
    double x1min = coord->options->x1min();
    double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
    auto cons =
        torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                      coord->options->nc2(), coord->options->nc1()},
                     torch::device(device).dtype(dtype));
    cons[IDN].fill_(1.);
    cons[IPR].fill_(1.e8);
    cons[ICY].fill_(0.01);       // vapor
    cons[ICY + 1].fill_(0.001);  // cloud
    for (int i = 0; i < coord->options->nc1(); ++i) {
      int g = static_cast<int>(x1min + (i - il + 0.5) * dx);   // global cell
      cons[ICY + 2].select(-1, i).fill_(g >= 8 ? 0.001 : 0.);  // deficit 0
    }
    vars[b]["hydro_w"] = block->phydro->peos->compute("U->W", {cons});
  }
  mesh->initialize(vars);

  // blocks in x1 order, then their owned cells side by side
  std::vector<size_t> order(mesh->blocks.size());
  for (size_t b = 0; b < order.size(); ++b) order[b] = b;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return mesh->blocks[a]->pcoord->options->x1min() <
           mesh->blocks[b]->pcoord->options->x1min();
  });
  auto owned = [&]() {
    std::vector<torch::Tensor> parts;
    for (auto b : order) {
      auto coord = mesh->blocks[b]->pcoord;
      parts.push_back(vars[b]
                          .at("hydro_u")
                          .slice(-1, coord->il(), coord->iu() + 1)
                          .cpu()
                          .clone());
    }
    return torch::cat(parts, -1);
  };

  std::vector<torch::Tensor> out;
  int nstage = static_cast<int>(mesh->blocks[0]->pintg->stages.size());
  for (int n = 0; n < nstep; ++n) {
    double dt = mesh->max_time_step(vars);
    for (int s = 0; s < nstage; ++s) mesh->forward(vars, dt, s);
    EXPECT_EQ(mesh->check_redo(vars), 0) << "nb1 = " << nb1 << " step " << n;
    out.push_back(owned());
  }
  return out;
}

//! six cubed-sphere panels in one process, exact AAM, Coriolis about z and
//! the closure on or off (as test_global_phase_closure.cpp)
Mesh cubed_mesh(bool closure, double omega, torch::Device device) {
  auto cfg = YAML::LoadFile("test_axial_momentum_flux_form.yaml");
  cfg["dynamics"]["conserve-axial-momentum"] = "exact";
  cfg["geometry"]["cells"]["interp_order"] = 4;
  cfg["forcing"]["coriolis"]["type"] = "xyz";
  cfg["forcing"]["coriolis"]["omega1"] = omega;
  cfg["forcing"]["const-gravity"]["grav1"] = 0.;  // the uniform state at rest
  char path[] = "test_global_phase_XXXXXX.yaml";
  auto fd = mkstemps(path, 5);
  TORCH_CHECK(fd >= 0, "mkstemps failed");
  {
    std::ofstream f(path);
    f << YAML::Dump(cfg);
  }
  close(fd);
  auto opts = MeshOptionsImpl::from_yaml(path);
  std::remove(path);
  opts->blocks_per_process(6);
  {  // as python/csrc/pymesh.cpp set_local_horizontal_cells(8, 8)
    auto coord = opts->block()->coord();
    auto layout = opts->block()->layout();
    if (coord->global_nx2() == 0) {
      coord->global_x2min(coord->x2min());
      coord->global_x2max(coord->x2max());
    }
    if (coord->global_nx3() == 0) {
      coord->global_x3min(coord->x3min());
      coord->global_x3max(coord->x3max());
    }
    coord->nx2(8);
    coord->nx3(8);
    coord->global_nx2(8 * layout->px());
    coord->global_nx3(8 * layout->py());
  }
  auto mesh = Mesh(opts);
  // exact includes the closure; off is lane A's test hook (A.rst 3.2), the
  // old full without conserve-axial-momentum-closure
  for (auto& b : mesh->blocks) b->phydro->paam->closure = closure;
  mesh->to(device, torch::kFloat64);
  return mesh;
}

//! the absolute axial angular momentum of the owned cells, as
//! test_axial_momentum_flux_form.py aam(): V r k . m + omega rho r^2 k . k V
double absolute_aam(Mesh const& mesh, MeshVariables const& vars, double omega) {
  double total = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto pg = dynamic_cast<GnomonicEquiangleImpl const*>(block->pcoord.get());
    auto u = vars[b].at("hydro_u");
    auto r = pg->x1v, vol = pg->cell_volume();
    auto rho = u[IDN];
    if (u.size(0) > ICY) rho = rho + u.narrow(0, ICY, u.size(0) - ICY).sum(0);
    auto l =
        vol * r * (pg->aam_k_c[VEL2] * u[IVY] + pg->aam_k_c[VEL3] * u[IVZ]) +
        omega * vol * r * r * pg->aam_kk_c * rho;
    auto in = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    total += l.index(in).sum().item<double>();
  }
  return total;
}

//! the largest |L(n) - L(0)| / |L(0)| over nstep steps
double aam_drift(bool closure, torch::Device device, int nstep) {
  double const omega = 5.e-6;
  auto mesh = cubed_mesh(closure, omega, device);
  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto c = block->pcoord->options;
    auto w = torch::zeros(
        {block->phydro->peos->nvar(), c->nc3(), c->nc2(), c->nc1()},
        torch::device(device).dtype(torch::kFloat64));
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    w[IVY].fill_(10.);  // a wind, so the Coriolis torque is not zero
    vars[b]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  double l0 = absolute_aam(mesh, vars, omega), drift = 0.;
  int nstage = static_cast<int>(mesh->blocks[0]->pintg->stages.size());
  for (int n = 0; n < nstep; ++n) {
    double dt = mesh->max_time_step(vars);
    for (int s = 0; s < nstage; ++s) mesh->forward(vars, dt, s);
    EXPECT_EQ(mesh->check_redo(vars), 0) << "step " << n;
    drift = std::max(
        drift, std::abs(absolute_aam(mesh, vars, omega) - l0) / std::abs(l0));
  }
  return drift;
}

}  // namespace

TEST_P(DeviceTest, chemistry_slot_no_source_baseline_one_block_equals_two) {
  auto one = run_column(1, device, dtype, 10);
  auto two = run_column(2, device, dtype, 10);
  ASSERT_EQ(one.size(), two.size());
  for (size_t n = 0; n < one.size(); ++n)
    EXPECT_TRUE(torch::equal(one[n], two[n]))
        << "step " << n << ": max |two - one| = "
        << (two[n].to(torch::kFloat64) - one[n].to(torch::kFloat64))
               .abs()
               .max()
               .item<double>();
}

TEST(GlobalPhase, closure_sums_the_panels_partials_through_mesh_forward) {
  for (auto device :
       {torch::Device(torch::kCPU), torch::Device(torch::kCUDA)}) {
    if (device.is_cuda() && !snapy_cuda_test_enabled()) continue;
    double on = aam_drift(true, device, 10), off = aam_drift(false, device, 10);
    std::printf("%s: max |dL| / |L| closure on %.3e, off %.3e\n",
                device.str().c_str(), on, off);
    EXPECT_LE(on, 1.e-12) << device << ": closure on, max |dL| / |L|";
    EXPECT_GE(off, 1.e3 * on) << device << ": the closure must matter here";
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
