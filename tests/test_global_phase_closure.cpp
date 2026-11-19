// V3 lane X, X4: the axial closure's partial sums stay on the device.
// Six cubed-sphere panels in one process (test_axial_momentum_flux_form.yaml)
// with full AAM, Coriolis and conserve-axial-momentum-closure. After the
// blocks' advance_local with the closure deferred (as Mesh::forward does), the
// block's partial {R, I} must be FP64 on the state's CUDA device: the global
// phase sums and applies it there (no host copy, no host read). The six
// blocks run the stage together, as Mesh::forward runs them. CUDA only.

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

TEST(GlobalPhase, axial_closure_partial_sums_stay_on_the_device) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  auto cfg = YAML::LoadFile("test_axial_momentum_flux_form.yaml");
  cfg["dynamics"]["conserve-axial-momentum"] = "full";
  cfg["dynamics"]["conserve-axial-momentum-closure"] = true;
  cfg["geometry"]["cells"]["interp_order"] = 4;
  cfg["forcing"]["coriolis"]["type"] = "xyz";
  cfg["forcing"]["coriolis"]["omega1"] = 5.e-6;
  char path[] = "test_global_phase_closure_XXXXXX.yaml";
  auto fd = mkstemps(path, 5);
  ASSERT_GE(fd, 0);
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
  mesh->to(torch::Device(torch::kCUDA), torch::kFloat64);

  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto c = block->pcoord->options;
    auto w = torch::zeros(
        {block->phydro->peos->nvar(), c->nc3(), c->nc2(), c->nc1()},
        torch::device(torch::kCUDA).dtype(torch::kFloat64));
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    w[IVY].fill_(10.);  // a wind, so the Coriolis torque is not zero
    vars[b]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  double dt = mesh->max_time_step(vars);

  // the blocks' stages together (they exchange seams inside a stage), the
  // closure deferred to the global phase, which is not run here
  std::vector<std::thread> threads;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    mesh->blocks[b]->defer_axial_closure = true;
    threads.emplace_back(
        [&, b] { mesh->blocks[b]->advance_local(vars[b], dt, 0); });
  }
  for (auto& t : threads) t.join();
  for (auto const& block : mesh->blocks) {
    auto const& sums = block->phydro->axial_closure_sums();
    ASSERT_TRUE(sums.defined());
    EXPECT_TRUE(sums.is_cuda()) << "partial sums on " << sums.device();
    EXPECT_EQ(sums.scalar_type(), torch::kFloat64);
    EXPECT_EQ(sums.numel(), 2);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
