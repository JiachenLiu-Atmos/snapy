// external
#include <gtest/gtest.h>

// C/C++
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>
#include <snap/mesh/stage_executor.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

// V3 lane B: a Mesh owns one StageExecutor for all its blocks and its forward
// runs each block's local stage through it; a bare MeshBlock (or a clone of a
// Mesh's block) keeps a one-block executor of its own. The executor changes
// who runs the stage, not what runs: two blocks stepped by Mesh::forward equal
// the pre-executor sequence (advance_local per block, then the exchange),
// bitwise.
namespace {

Mesh make_mesh(torch::Device device, torch::Dtype dtype) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  block_opts->layout()->py(1);  // nb2 = 2, nb3 = 1: two blocks along x2
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);
  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);
  return mesh;
}

// a pressure and velocity field that varies across the block seam
MeshVariables initial(Mesh mesh) {
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto pcoord = mesh->blocks[i]->pcoord;
    auto opts = pcoord->x2v.options();
    auto x2 = pcoord->x2v.view({1, -1, 1});
    auto x3 = pcoord->x3v.view({-1, 1, 1});
    auto w = torch::zeros({5, pcoord->options->nc3(), pcoord->options->nc2(),
                           pcoord->options->nc1()},
                          opts);
    w[IDN].fill_(1.);
    w[IPR] = 1.e5 * (1. + 0.1 * torch::sin(6.283185307179586 * x2) +
                     0.05 * torch::cos(6.283185307179586 * x3));
    w[IVY] = 10. * torch::cos(6.283185307179586 * x2) + 0. * x3;
    vars[i]["hydro_w"] = w;
  }
  mesh->initialize(vars);
  return vars;
}

}  // namespace

TEST_P(DeviceTest, mesh_owns_one_executor_and_steps_through_it) {
  auto mesh = make_mesh(device, dtype);
  ASSERT_EQ(mesh->blocks.size(), 2u);
  auto ex = mesh->executor();
  ASSERT_TRUE(ex);
  EXPECT_EQ(ex->num_blocks(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(mesh->blocks[i]->executor(), ex) << "block " << i;
    EXPECT_EQ(ex->block_at(i), mesh->blocks[i].get()) << "block " << i;
  }

  // the reference: the same deck stepped as Mesh::forward did before lane B
  auto ref = make_mesh(device, dtype);
  auto vars = initial(mesh);
  auto rvars = initial(ref);
  std::vector<torch::Tensor> u_init;
  for (auto& v : vars) u_init.push_back(v.at("hydro_u").clone());

  int nstage = mesh->blocks.front()->pintg->stages.size();
  int nstep = 3;
  double dt = 0.5 * mesh->max_time_step(vars);
  for (int n = 0; n < nstep; ++n)
    for (int s = 0; s < nstage; ++s) {
      mesh->forward(vars, dt, s);
      for (size_t i = 0; i < ref->blocks.size(); ++i) {
        if (s == 0) ref->blocks[i]->step_dt = dt;
        ref->blocks[i]->defer_chemistry = true;
        ref->blocks[i]->advance_local(rvars[i], dt, s);
      }
      // same-process blocks meet in the exchange: one thread each
      std::vector<std::thread> jobs;
      for (size_t i = 0; i < ref->blocks.size(); ++i)
        jobs.emplace_back([&, i] {
          c10::OptionalDeviceGuard guard(device);
          ref->blocks[i]->exchange_ghost_zones(rvars[i]);
        });
      for (auto& j : jobs) j.join();
    }

  for (size_t i = 0; i < mesh->blocks.size(); ++i)
    for (auto name : {"hydro_u", "hydro_w"})
      EXPECT_TRUE(torch::equal(vars[i].at(name), rvars[i].at(name)))
          << "block " << i << " " << name << " differs from the pre-B sequence";
  for (size_t i = 0; i < mesh->blocks.size(); ++i)
    EXPECT_FALSE(torch::equal(vars[i].at("hydro_u"), u_init[i]))
        << "block " << i
        << ": the deck did not move, so equality gates nothing";

  // the Mesh's block-stages ran through its executor, eagerly
  auto report = mesh->execution_report();
  EXPECT_EQ(report.at("eager"), 2 * nstage * nstep);
  EXPECT_EQ(report.at("captures"), 0);
  EXPECT_EQ(report.at("graphs"), 0);
  EXPECT_EQ(mesh->blocks.front()->execution_report(), report);
  // the reference mesh's executor ran nothing
  EXPECT_EQ(ref->execution_report().at("eager"), 0);
}

TEST_P(DeviceTest, bare_block_and_clone_own_their_executor) {
  auto mesh = make_mesh(device, dtype);
  auto bare = MeshBlock(mesh->blocks.front()->options);
  ASSERT_TRUE(bare->executor());
  EXPECT_EQ(bare->executor()->num_blocks(), 1);
  EXPECT_EQ(bare->executor()->block_at(0), bare.get());
  EXPECT_NE(bare->executor(), mesh->executor());

  auto copy =
      std::dynamic_pointer_cast<MeshBlockImpl>(mesh->blocks.back()->clone());
  ASSERT_TRUE(copy);
  ASSERT_TRUE(copy->executor());
  EXPECT_NE(copy->executor(), mesh->executor())
      << "a clone of a Mesh's block must not run through the Mesh's executor";
  EXPECT_EQ(copy->executor()->block_at(0), copy.get());
}
