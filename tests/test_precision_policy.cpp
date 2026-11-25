// Lane X, section 5 "dtype per quantity": each row of section 6 in both modes.

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/mesh/precision_policy.hpp>

// tests
#include "cuda_test_gate.hpp"

using snap::PrecisionPolicy;
using snap::Quantity;

namespace {

//! a moist Cartesian column with implicit-scheme 1 (VIC), on the CPU
snap::MeshBlock vic_block() {
  return snap::MeshBlock(
      snap::MeshBlockOptionsImpl::from_yaml("test_tracer_dry_convention.yaml"));
}

bool narrow_row(Quantity q) {
  return q == Quantity::kHydroState || q == Quantity::kBackup ||
         q == Quantity::kScalarState || q == Quantity::kFlux ||
         q == Quantity::kImplicit;
}

void expect_rows(PrecisionPolicy const& p, torch::Dtype narrow,
                 torch::Dtype wide) {
  for (int i = 0; i < static_cast<int>(Quantity::kCount); ++i) {
    auto q = static_cast<Quantity>(i);
    // kImplicit stores its FP64 stage (the boundary and the retry) wide
    auto storage = narrow_row(q) && q != Quantity::kImplicit ? narrow : wide;
    auto compute = narrow_row(q) || q == Quantity::kReference ? narrow : wide;
    EXPECT_EQ(p.storage(q), storage) << "row " << i;
    EXPECT_EQ(p.compute(q), compute) << "row " << i;
  }
}

}  // namespace

TEST(PrecisionPolicy, Float64) {
  PrecisionPolicy p("float64");
  EXPECT_FALSE(p.fp32());
  expect_rows(p, torch::kFloat64, torch::kFloat64);
}

TEST(PrecisionPolicy, Float32) {
  PrecisionPolicy p("float32");
  EXPECT_TRUE(p.fp32());
  expect_rows(p, torch::kFloat32, torch::kFloat64);
}

TEST(PrecisionPolicy, MpsHoldsNoFloat64) {
  for (auto precision : {"float64", "float32"}) {
    PrecisionPolicy p(precision, torch::Device(torch::kMPS));
    expect_rows(p, torch::kFloat32, torch::kFloat32);
  }
}

TEST(PrecisionPolicy, CastAndStore) {
  PrecisionPolicy p("float32");
  auto x = torch::rand({4}, torch::kFloat64);
  auto s = p.store(Quantity::kHydroState, x);
  EXPECT_EQ(s.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(s, x.to(torch::kFloat32)));
  // already in the dtype: the tensor itself, no copy
  EXPECT_TRUE(p.store(Quantity::kHydroState, s).is_same(s));
  EXPECT_TRUE(p.cast(Quantity::kPrimitive, x).is_same(x));
  EXPECT_EQ(p.cast(Quantity::kPrimitive, s).scalar_type(), torch::kFloat64);
}

// the block policy is built for the block's device, by reset() and to()
TEST(PrecisionPolicy, BlockPolicyFollowsTheBlockDevice) {
  std::vector<torch::Device> devices = {torch::kCPU};
  if (snapy_cuda_test_enabled()) devices.push_back(torch::kCUDA);
  if (torch::hasMPS()) devices.push_back(torch::kMPS);
  auto b = vic_block();
  for (auto device : devices) {
    auto wide = PrecisionPolicy::wide(device);
    b->to(device, wide);
    EXPECT_EQ(b->policy.storage(Quantity::kImplicit), wide) << device;
    EXPECT_EQ(b->policy.storage(Quantity::kHydroState), wide) << device;
  }
}

// MPS has no FP64: the implicit stage runs there, without an FP64 cast
// ('Cannot convert a MPS Tensor to float64' with a CPU-built policy)
TEST(PrecisionPolicy, ImplicitStageRunsOnMps) {
  if (!torch::hasMPS()) GTEST_SKIP() << "MPS is not available";
  auto b = vic_block();
  auto pc = b->pcoord;
  auto w = torch::zeros({b->phydro->peos->nvar(), pc->options->nc3(),
                         pc->options->nc2(), pc->options->nc1()},
                        torch::kFloat64);
  auto z = pc->x1v.view({1, 1, -1});
  w[snap::IDN].copy_(torch::exp(-z / 8.e3).expand_as(w[snap::IDN]));
  w[snap::IPR].copy_(1.e5 * torch::exp(-z / 8.e3).expand_as(w[snap::IPR]));
  w[snap::ICY].fill_(1.e-3);
  snap::Variables v = {
      {"hydro_w", w},
      {"scalar_r", torch::full({1, w.size(1), w.size(2), w.size(3)}, 0.5,
                               torch::kFloat64)}};
  auto mps = torch::Device(torch::kMPS);
  b->to(mps, torch::kFloat32);
  for (auto& [key, t] : v) t = t.to(mps, torch::kFloat32);
  b->initialize(v);
  // the first stage only: the last one's saturation adjustment has no MPS
  // kernel (test_mps_matches_cpu.py)
  EXPECT_NO_THROW(b->forward(v, 0.5 * b->max_time_step(v), 0));
  EXPECT_TRUE(v["hydro_u"].is_mps());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
