// C/C++
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// fvm
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>

#include "interp_simple.hpp"

// tests
#include "device_testing.hpp"

using namespace snap;

TEST_P(DeviceTest, interp_plm) {
  double phim1 = 1.0;
  double phi = 2.0;
  double phip1 = 3.0;
  double result = interp_plm(phim1, phi, phip1);
  double expected_result = 1.5;
  EXPECT_NEAR(result, expected_result, 1.E-10);
}

TEST_P(DeviceTest, interp_plm_torch1) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l = interp_plm(phi[0].item<float>(), phi[1].item<float>(),
                                 phi[2].item<float>());
      auto result1r = interp_plm(phi[2].item<float>(), phi[1].item<float>(),
                                 phi[0].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl.item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr.item<float>(), 2.E-6);
    } else {
      auto result1l = interp_plm(phi[0].item<double>(), phi[1].item<double>(),
                                 phi[2].item<double>());
      auto result1r = interp_plm(phi[2].item<double>(), phi[1].item<double>(),
                                 phi[0].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 0);
      EXPECT_NEAR(result1l, resultl.item<double>(), 1.E-6);
      EXPECT_NEAR(result1r, resultr.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_plm_torch2) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l =
          interp_plm(phi[0][0].item<float>(), phi[0][1].item<float>(),
                     phi[0][2].item<float>());
      auto result1r =
          interp_plm(phi[0][2].item<float>(), phi[0][1].item<float>(),
                     phi[0][0].item<float>());
      auto result2l =
          interp_plm(phi[1][0].item<float>(), phi[1][1].item<float>(),
                     phi[1][2].item<float>());
      auto result2r =
          interp_plm(phi[1][2].item<float>(), phi[1][1].item<float>(),
                     phi[1][0].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 1);

      EXPECT_NEAR(result1l, resultl[0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[1].item<float>(), 2.E-6);
    } else {
      auto result1l =
          interp_plm(phi[0][0].item<double>(), phi[0][1].item<double>(),
                     phi[0][2].item<double>());
      auto result1r =
          interp_plm(phi[0][2].item<double>(), phi[0][1].item<double>(),
                     phi[0][0].item<double>());
      auto result2l =
          interp_plm(phi[1][0].item<double>(), phi[1][1].item<double>(),
                     phi[1][2].item<double>());
      auto result2r =
          interp_plm(phi[1][2].item<double>(), phi[1][1].item<double>(),
                     phi[1][0].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 1);

      EXPECT_NEAR(result1l, resultl[0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[1].item<double>(), 2.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_plm_torch3) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({3, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l =
          interp_plm(phi[0][0].item<float>(), phi[1][0].item<float>(),
                     phi[2][0].item<float>());

      auto result1r =
          interp_plm(phi[2][0].item<float>(), phi[1][0].item<float>(),
                     phi[0][0].item<float>());

      auto result2l =
          interp_plm(phi[0][1].item<float>(), phi[1][1].item<float>(),
                     phi[2][1].item<float>());

      auto result2r =
          interp_plm(phi[2][1].item<float>(), phi[1][1].item<float>(),
                     phi[0][1].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl[0][0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0][0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[0][1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[0][1].item<float>(), 2.E-6);
    } else {
      auto result1l =
          interp_plm(phi[0][0].item<double>(), phi[1][0].item<double>(),
                     phi[2][0].item<double>());

      auto result1r =
          interp_plm(phi[2][0].item<double>(), phi[1][0].item<double>(),
                     phi[0][0].item<double>());

      auto result2l =
          interp_plm(phi[0][1].item<double>(), phi[1][1].item<double>(),
                     phi[2][1].item<double>());

      auto result2r =
          interp_plm(phi[2][1].item<double>(), phi[1][1].item<double>(),
                     phi[0][1].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl[0][0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0][0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[0][1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[0][1].item<double>(), 2.E-6);
    }
  }
}

// Round-off-sized slopes: the old FLT_MIN-guarded denominator divided 0/0.
TEST_P(DeviceTest, interp_plm_round_off_slopes_stay_finite) {
  PLMInterp interp;
  interp->to(device, dtype);

  std::vector<std::array<double, 3>> stencils = {
      {std::numeric_limits<float>::min(), 0., 0.}};
  if (dtype == torch::kFloat64) {
    stencils.push_back({-1.97568603841971563e-23, -1.97568603841971622e-23,
                        -1.97568603841971681e-23});
    stencils.push_back({3.81738396164642948e-24, 3.81738396164642948e-24,
                        3.81738396164641773e-24});
  }
  for (auto const &s : stencils) {
    auto phi = torch::tensor({s[0], s[1], s[2]}, torch::kFloat64)
                   .to(torch::device(device).dtype(dtype));
    auto [resultl, resultr] = interp->forward(phi, 0);
    ASSERT_TRUE(torch::isfinite(resultl).all().item<bool>());
    ASSERT_TRUE(torch::isfinite(resultr).all().item<bool>());
    if (dtype == torch::kFloat64) {
      EXPECT_EQ(interp_plm(s[0], s[1], s[2]), resultl.item<double>());
      EXPECT_EQ(interp_plm(s[2], s[1], s[0]), resultr.item<double>());
    }
  }
}

TEST_P(DeviceTest, reconstruct_preserves_a_constant_field) {
  auto w = torch::full({1, 8, 9, 10}, 2., torch::device(device).dtype(dtype));
  for (const auto *type : {"dc", "plm", "weno5"}) {
    auto options = ReconstructOptionsImpl::create();
    options->interp()->type(type);
    Reconstruct recon(options);
    recon->to(device, dtype);
    for (int dim : {1, 2, 3}) {
      auto result = recon->forward(w, dim);
      auto shape = w.sizes().vec();
      shape.insert(shape.begin(), 2);
      EXPECT_EQ(result.sizes().vec(), shape) << type << " dim=" << dim;
      EXPECT_TRUE(torch::allclose(result, torch::full_like(result, 2.)))
          << type << " dim=" << dim;
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
