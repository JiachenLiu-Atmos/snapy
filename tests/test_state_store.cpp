// Lane X, X2: the state store's fixed storage, its adoption of a rebound
// hydro_u/scalar_s and its generation counter.

// gtest
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/mesh/state_store.hpp>

using snap::PrecisionPolicy;
using snap::StateStore;
using snap::Variables;

namespace {

auto const kF64 = torch::kFloat64;

Variables random_state() {
  return {{"hydro_u", torch::rand({5, 2, 3, 4}, kF64)},
          {"scalar_s", torch::rand({1, 2, 3, 4}, kF64)},
          {"scalar_r", torch::rand({1, 2, 3, 4}, kF64)}};
}

snap::MeshBlock make_block(Variables& vars) {
  auto b = snap::MeshBlock(
      snap::MeshBlockOptionsImpl::from_yaml("test_tracer_dry_convention.yaml"));
  auto pc = b->pcoord;
  auto w = torch::zeros({b->phydro->peos->nvar(), pc->options->nc3(),
                         pc->options->nc2(), pc->options->nc1()},
                        kF64);
  auto z = pc->x1v.view({1, 1, -1});
  w[snap::IDN].copy_(torch::exp(-z / 8.e3).expand_as(w[snap::IDN]));
  w[snap::IPR].copy_(1.e5 * torch::exp(-z / 8.e3).expand_as(w[snap::IPR]));
  w[snap::ICY].fill_(1.e-3);
  w[snap::IVY].copy_(torch::sin(pc->x2v).view({1, -1, 1}).expand_as(w[0]));
  vars = {{"hydro_w", w},
          {"scalar_r",
           torch::full({1, w.size(1), w.size(2), w.size(3)}, 0.5, kF64)}};
  b->initialize(vars);
  return b;
}

void step(snap::MeshBlock& b, Variables& vars, double dt) {
  for (int s = 0; s < static_cast<int>(b->pintg->stages.size()); ++s)
    b->forward(vars, dt, s);
}

}  // namespace

TEST(StateStore, AdoptKeepsTheAddressAndTheCallersHandle) {
  StateStore s;
  s.reset(PrecisionPolicy("float64"), 5, 1, 2, 3, 4);
  EXPECT_EQ(s.generation(), 1);
  auto addr = s.hydro_u().data_ptr();

  auto v = random_state();
  auto u = v["hydro_u"];
  auto given = u.clone();
  s.adopt(v);
  EXPECT_EQ(s.generation(), 2);  // three keys, one bump
  EXPECT_TRUE(v["hydro_u"].is_same(u));
  EXPECT_TRUE(s.hydro_u().is_same(u));
  EXPECT_EQ(u.data_ptr(), addr);
  EXPECT_TRUE(torch::equal(u, given));

  s.adopt(v);  // already the store's: nothing to do
  EXPECT_EQ(s.generation(), 2);

  // a caller rebinds between steps: copied in, address kept, one bump
  v["hydro_u"] = 2. * u;
  auto rebound = v["hydro_u"];
  given = rebound.clone();
  s.adopt(v);
  EXPECT_EQ(s.generation(), 3);
  EXPECT_TRUE(s.hydro_u().is_same(rebound));
  EXPECT_EQ(rebound.data_ptr(), addr);
  EXPECT_TRUE(torch::equal(rebound, given));

  v["hydro_u"] = torch::zeros({5, 2, 3, 5}, kF64);
  EXPECT_THROW(s.adopt(v), c10::Error);
  v["hydro_u"] = torch::zeros({5, 2, 3, 4}, torch::kFloat32);
  EXPECT_THROW(s.adopt(v), c10::Error);
  v["hydro_u"] = rebound;

  // scalar_r is tracked, not copied: its dtype follows the divide until G1
  v["scalar_r"] = torch::rand({1, 2, 3, 4}, torch::kFloat32);
  s.adopt(v);
  EXPECT_TRUE(s.scalar_r().is_same(v["scalar_r"]));
}

TEST(StateStore, BackupRestoreAndReallocate) {
  StateStore s;
  s.reset(PrecisionPolicy("float32"), 5, 1, 2, 3, 4);
  EXPECT_EQ(s.u0().dtype(), torch::kFloat32);
  EXPECT_EQ(s.hydro_u().dtype(), torch::kFloat32);
  s.hydro_u().uniform_();
  s.scalar_s().uniform_();
  auto u = s.hydro_u().clone(), q = s.scalar_s().clone();
  s.backup();
  s.hydro_u().zero_();
  s.scalar_s().zero_();
  s.restore();
  EXPECT_TRUE(torch::equal(s.hydro_u(), u));
  EXPECT_TRUE(torch::equal(s.scalar_s(), q));

  int g = s.generation();
  auto addr = s.hydro_u().data_ptr();
  s.reallocate();
  EXPECT_EQ(s.generation(), g + 1);
  EXPECT_NE(s.hydro_u().data_ptr(), addr);
}

TEST(StateStore, MeshBlockAdoptsARebindBetweenSteps) {
  Variables va, vb;
  auto a = make_block(va);
  auto b = make_block(vb);
  double dt = 0.5 * a->max_time_step(va);

  step(a, va, dt);
  step(b, vb, dt);
  int ga = a->state_generation(), gb = b->state_generation();
  auto addr = va["hydro_u"].data_ptr();

  // as tests/test_straka_cuda_graph.py:32-33 does at step 50
  vb["hydro_u"] = vb["hydro_u"].clone();
  vb["scalar_s"] = vb["scalar_s"].clone();
  auto ub = vb["hydro_u"];

  for (int n = 0; n < 3; ++n) {
    step(a, va, dt);
    step(b, vb, dt);
  }
  EXPECT_EQ(a->state_generation(), ga);  // out=: no new storage per stage
  EXPECT_EQ(va["hydro_u"].data_ptr(), addr);
  EXPECT_EQ(b->state_generation(), gb + 1);
  EXPECT_TRUE(vb["hydro_u"].is_same(ub));  // the caller's tensor is updated
  for (auto key : {"hydro_u", "scalar_s", "scalar_r", "hydro_w"})
    EXPECT_TRUE(torch::equal(va[key], vb[key])) << key;

  a->to(torch::kCPU, kF64);
  EXPECT_EQ(a->state_generation(), ga + 1);
  a->to(torch::Device(torch::kCPU));
  EXPECT_EQ(a->state_generation(), ga + 2);
}

TEST(StateStore, MeshBlockToDeviceReallocates) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "no CUDA device";
  Variables v;
  auto b = make_block(v);
  int g = b->state_generation();
  b->to(torch::Device(torch::kCUDA));  // Python's block.to(device)
  EXPECT_EQ(b->state_generation(), g + 1);
  for (auto& [key, t] : v) t = t.to(torch::kCUDA);
  step(b, v, 0.5 * b->max_time_step(v));
  EXPECT_TRUE(v["hydro_u"].is_cuda());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
