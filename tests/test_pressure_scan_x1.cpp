#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <limits>
#include <snap/hydro/hydro_reference.hpp>

#include "device_testing.hpp"

namespace {

// Independent FP64 oracle: each face sums only the cells between it and the
// supplied anchor, without using the implementation's running face recurrence.
std::pair<torch::Tensor, torch::Tensor> oracle(torch::Tensor rho,
                                               torch::Tensor dx,
                                               torch::Tensor anchor, int iu,
                                               double g) {
  rho = rho.to(torch::kCPU, torch::kFloat64);
  dx = dx.to(torch::kCPU, torch::kFloat64);
  anchor = anchor.to(torch::kCPU, torch::kFloat64);
  auto dp = g * rho * dx;
  auto faces =
      torch::empty({rho.size(0), rho.size(1), rho.size(2) + 1}, rho.options());
  for (int f = 0; f <= rho.size(-1); ++f) {
    auto value = anchor.squeeze(-1).clone();
    if (f <= iu) value += dp.slice(-1, f, iu + 1).sum(-1);
    if (f > iu + 1) value -= dp.slice(-1, iu + 1, f).sum(-1);
    faces.select(-1, f).copy_(value);
  }
  return {faces.slice(-1, 0, -1), faces.slice(-1, 1)};
}

void close(torch::Tensor a, torch::Tensor b, torch::Dtype dtype) {
  EXPECT_TRUE(torch::allclose(a.to(torch::kCPU, torch::kFloat64),
                              b.to(torch::kCPU, torch::kFloat64),
                              dtype == torch::kFloat32 ? 2.e-6 : 1.e-13,
                              dtype == torch::kFloat32 ? 1.e-12 : 1.e-20));
}

}  // namespace

TEST_P(DeviceTest, signed_pressure_scan_matches_oracle_and_tensor_backend) {
  auto opts = torch::TensorOptions().device(device).dtype(dtype);
  // Noncontiguous input with distinct columns, signed density and anchor.
  auto rho =
      (torch::arange(60, opts).reshape({3, 2, 10}) - 31.).transpose(0, 1) *
      1.e-7;
  auto dx = torch::linspace(.1, .4, 10, torch::kFloat64);
  auto anchor = (torch::arange(6, opts).reshape({2, 3, 1}) - 3.) * 1.e-6;
  auto before = rho.clone(), abefore = anchor.clone();
  for (int iu : {0, 7, 9}) {
    snap::HydroReferenceX1Operator op(dx, iu, false, false, false, false);
    for (double g : {0., -3., 10.}) {
      auto [lo, hi] = op.scan_pressure(rho, g, anchor);
      auto [elo, ehi] = oracle(rho, dx.to(opts), anchor, iu, g);
      close(lo, elo, dtype);
      close(hi, ehi, dtype);
      EXPECT_EQ(lo.scalar_type(), dtype);
      EXPECT_EQ(lo.device(), rho.device());
      EXPECT_TRUE(torch::equal(hi.select(-1, iu), anchor.squeeze(-1)));
      EXPECT_TRUE(torch::equal(hi.slice(-1, 0, -1), lo.slice(-1, 1)));
      auto tl = torch::empty_like(lo), th = torch::empty_like(hi);
      snap::hydro_pressure_scan_x1_tensor(rho, dx.to(opts), anchor, tl, th, iu,
                                          g);
      close(lo, tl, dtype);
      close(hi, th, dtype);
    }
  }
  EXPECT_TRUE(torch::equal(rho, before));
  EXPECT_TRUE(torch::equal(anchor, abefore));
}

TEST_P(DeviceTest, pressure_scan_preserves_sub_ulp_increments_and_linearity) {
  auto opts = torch::TensorOptions().device(device).dtype(dtype);
  auto dx = torch::ones({10}, torch::kFloat64);
  snap::HydroReferenceX1Operator op(dx, 7, false, true, true, true);
  auto drho = torch::full({1, 1, 10}, 1.e-8, opts);
  auto anchor = torch::full({1, 1, 1}, -2.e-8, opts);
  // Even FP64 totals can erase this increment. The delta path must not form
  // and subtract them, nor apply a positive pressure floor to signed output.
  auto background = torch::full_like(drho, 1.e12);
  EXPECT_TRUE(torch::equal(background + drho, background));
  auto [lo, hi] = op.scan_pressure(drho, 1., anchor);
  auto [elo, ehi] = oracle(drho, dx, anchor, 7, 1.);
  close(lo, elo, dtype);
  close(hi, ehi, dtype);
  EXPECT_LT(hi.min().item<double>(), 0.);
  EXPECT_GT(lo.max().item<double>(), 0.);
  auto [twolo, twohi] = op.scan_pressure(-2. * drho, 1., -2. * anchor);
  close(twolo, -2. * lo, dtype);
  close(twohi, -2. * hi, dtype);
  auto [sumlo, sumhi] =
      op.scan_pressure(drho + (-2. * drho), 1., anchor + (-2. * anchor));
  close(sumlo, lo + twolo, dtype);
  close(sumhi, hi + twohi, dtype);
  auto [zero_lo, zero_hi] =
      op.scan_pressure(torch::zeros_like(drho), 1., torch::zeros_like(anchor));
  EXPECT_EQ(zero_lo.abs().max().item<double>(), 0.);
  EXPECT_EQ(zero_hi.abs().max().item<double>(), 0.);
}

TEST_P(DeviceTest, pressure_scan_split_column_anchor_relay) {
  auto opts = torch::TensorOptions().device(device).dtype(dtype);
  auto dx = torch::linspace(.1, .4, 16, torch::kFloat64);
  auto rho = (torch::arange(16, opts).reshape({1, 1, 16}) - 9.) * 1.e-7;
  auto anchor = torch::full({1, 1, 1}, -1.e-6, opts);
  snap::HydroReferenceX1Operator whole(dx, 13, false, true, true, true);
  snap::HydroReferenceX1Operator upper(dx.slice(0, 6, 16), 7, false, false,
                                       true, true);
  snap::HydroReferenceX1Operator lower(dx.slice(0, 0, 10), 7, false, true,
                                       false, true);
  auto [lo, hi] = whole.scan_pressure(rho, 10., anchor);
  auto [ul, uh] = upper.scan_pressure(rho.slice(-1, 6, 16), 10., anchor);
  // Pass the upper block's lower OWNED face, not its first ghost face.
  auto [ll, lh] =
      lower.scan_pressure(rho.slice(-1, 0, 10), 10., ul.slice(-1, 2, 3));
  close(ul, lo.slice(-1, 6, 16), dtype);
  close(uh, hi.slice(-1, 6, 16), dtype);
  close(ll, lo.slice(-1, 0, 10), dtype);
  close(lh, hi.slice(-1, 0, 10), dtype);
  EXPECT_TRUE(torch::equal(lh.select(-1, 7), ul.select(-1, 2)));
}

TEST_P(DeviceTest, pressure_scan_agrees_with_unclamped_default_faces) {
  auto opts = torch::TensorOptions().device(device).dtype(dtype);
  for (bool stretched : {false, true}) {
    auto dx = stretched ? torch::linspace(.1, .4, 10, torch::kFloat64)
                        : torch::full({10}, .1, torch::kFloat64);
    snap::HydroReferenceX1Operator op(dx, 7, stretched, true, true, true);
    auto w = torch::zeros({5, 2, 3, 10}, opts);
    w[snap::IDN].fill_(1.);
    w[snap::IPR].fill_(100.);
    auto anchor = torch::full({2, 3, 1}, 90., opts);
    snap::HydroReferenceX1 out(w);
    op.evaluate(w, 10., anchor, out);
    auto [lo, hi] = op.scan_pressure(w[snap::IDN], 10., anchor);
    EXPECT_TRUE(torch::equal(lo, out.psf_lo));
    EXPECT_TRUE(torch::equal(hi, out.psf_hi));
    // The existing total evaluator still floors an invalid anchor.
    w[snap::IDN].zero_();
    anchor.fill_(-1.);
    op.evaluate(w, 10., anchor, out);
    EXPECT_GT(out.psf_lo.min().item<double>(), 0.);
    auto signed_faces = op.scan_pressure(w[snap::IDN], 10., anchor);
    EXPECT_TRUE(
        torch::equal(signed_faces.first, torch::full_like(out.psf_lo, -1.)));
  }
}

TEST(PressureScanX1, rejects_ambiguous_shapes_and_missing_anchor) {
  auto dx = torch::ones({10}, torch::kFloat64);
  snap::HydroReferenceX1Operator op(dx, 7, false, true, true, true);
  auto rho = torch::ones({2, 3, 10}, torch::kFloat32);
  auto anchor = torch::zeros({2, 3, 1}, torch::kFloat64);
  EXPECT_THROW(op.scan_pressure(rho, 1., torch::Tensor()), c10::Error);
  EXPECT_THROW(op.scan_pressure(rho, 1., anchor[0]), c10::Error);
  EXPECT_THROW(op.scan_pressure(rho.slice(-1, 0, 9), 1., anchor), c10::Error);
  EXPECT_THROW(op.scan_pressure(rho.to(torch::kInt64), 1., anchor), c10::Error);
}
