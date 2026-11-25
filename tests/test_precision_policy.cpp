// Lane X, section 5 "dtype per quantity": each row of section 6 in both modes.

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/mesh/precision_policy.hpp>

using snap::PrecisionPolicy;
using snap::Quantity;

namespace {

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
