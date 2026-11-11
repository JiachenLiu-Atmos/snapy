// C/C++
#include <algorithm>
#include <cmath>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/recon/interpolation.hpp>

#include "interp_simple.hpp"

// tests
#include "device_testing.hpp"

using namespace snap;

TEST_P(DeviceTest, interp_cp5m_torch1) {
  Center5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({5}, torch::device(device).dtype(dtype));

    if (dtype == torch::kFloat32) {
      auto result1 = interp_cp5(phi[0].item<float>(), phi[1].item<float>(),
                                phi[2].item<float>(), phi[3].item<float>(),
                                phi[4].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_cp5(phi[0].item<double>(), phi[1].item<double>(),
                                phi[2].item<double>(), phi[3].item<double>(),
                                phi[4].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_cp5m_torch2) {
  Center5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_cp5(phi[0][0].item<float>(), phi[0][1].item<float>(),
                     phi[0][2].item<float>(), phi[0][3].item<float>(),
                     phi[0][4].item<float>());
      auto result2 =
          interp_cp5(phi[1][0].item<float>(), phi[1][1].item<float>(),
                     phi[1][2].item<float>(), phi[1][3].item<float>(),
                     phi[1][4].item<float>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_cp5(phi[0][0].item<double>(), phi[0][1].item<double>(),
                     phi[0][2].item<double>(), phi[0][3].item<double>(),
                     phi[0][4].item<double>());
      auto result2 =
          interp_cp5(phi[1][0].item<double>(), phi[1][1].item<double>(),
                     phi[1][2].item<double>(), phi[1][3].item<double>(),
                     phi[1][4].item<double>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_cp5m_torch3) {
  Center5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({5, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_cp5(phi[0][0].item<float>(), phi[1][0].item<float>(),
                     phi[2][0].item<float>(), phi[3][0].item<float>(),
                     phi[4][0].item<float>());
      auto result2 =
          interp_cp5(phi[0][1].item<float>(), phi[1][1].item<float>(),
                     phi[2][1].item<float>(), phi[3][1].item<float>(),
                     phi[4][1].item<float>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_cp5(phi[0][0].item<double>(), phi[1][0].item<double>(),
                     phi[2][0].item<double>(), phi[3][0].item<double>(),
                     phi[4][0].item<double>());
      auto result2 =
          interp_cp5(phi[0][1].item<double>(), phi[1][1].item<double>(),
                     phi[2][1].item<double>(), phi[3][1].item<double>(),
                     phi[4][1].item<double>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_cp5p_torch4) {
  Center5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 = interp_cp5(phi[4].item<float>(), phi[3].item<float>(),
                                phi[2].item<float>(), phi[1].item<float>(),
                                phi[0].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_cp5(phi[4].item<double>(), phi[3].item<double>(),
                                phi[2].item<double>(), phi[1].item<double>(),
                                phi[0].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3a) {
  double phim1 = 1.0;
  double phi = 2.0;
  double phip1 = 3.0;
  double result = interp_weno3(phim1, phi, phip1);
  double expected_result = 1.5;
  EXPECT_NEAR(result, expected_result, 1.E-10);
}

TEST_P(DeviceTest, interp_weno3_smooth_limit) {
  Weno3Interp interp;
  interp->to(device, torch::kFloat64);

  constexpr double eps = 1.e-7;
  auto phi = torch::tensor({1. + eps, 1. + 2. * eps, 1. + 4. * eps},
                           torch::device(device).dtype(torch::kFloat64));

  auto left_expected = interp_cp3(phi[0].item<double>(), phi[1].item<double>(),
                                  phi[2].item<double>());
  auto left_result = torch::zeros({1}, phi.options());
  interp->left(phi, 0, left_result);
  EXPECT_NEAR(left_expected, left_result.item<double>(), 1.E-6);

  auto right_expected = interp_cp3(phi[2].item<double>(), phi[1].item<double>(),
                                   phi[0].item<double>());
  auto right_result = torch::zeros({1}, phi.options());
  interp->right(phi, 0, right_result);
  EXPECT_NEAR(right_expected, right_result.item<double>(), 1.E-6);
}

TEST_P(DeviceTest, interp_weno5b) {
  double phim2 = 1.0;
  double phim1 = 2.0;
  double phi = 3.0;
  double phip1 = 4.0;
  double phip2 = 5.0;
  double result = interp_weno5(phim2, phim1, phi, phip1, phip2);
  double expected_result = 2.5000000000000004;
  EXPECT_NEAR(result, expected_result, 1.E-10);
}

TEST_P(DeviceTest, interp_weno3m_torch1) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 = interp_weno3(phi[0].item<float>(), phi[1].item<float>(),
                                  phi[2].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_weno3(phi[0].item<double>(), phi[1].item<double>(),
                                  phi[2].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3m_torch2) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno3(phi[0][0].item<float>(), phi[0][1].item<float>(),
                       phi[0][2].item<float>());
      auto result2 =
          interp_weno3(phi[1][0].item<float>(), phi[1][1].item<float>(),
                       phi[1][2].item<float>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno3(phi[0][0].item<double>(), phi[0][1].item<double>(),
                       phi[0][2].item<double>());
      auto result2 =
          interp_weno3(phi[1][0].item<double>(), phi[1][1].item<double>(),
                       phi[1][2].item<double>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3m_torch3) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({3, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno3(phi[0][0].item<float>(), phi[1][0].item<float>(),
                       phi[2][0].item<float>());
      auto result2 =
          interp_weno3(phi[0][1].item<float>(), phi[1][1].item<float>(),
                       phi[2][1].item<float>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno3(phi[0][0].item<double>(), phi[1][0].item<double>(),
                       phi[2][0].item<double>());
      auto result2 =
          interp_weno3(phi[0][1].item<double>(), phi[1][1].item<double>(),
                       phi[2][1].item<double>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3p_torch4) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 = interp_weno3(phi[2].item<float>(), phi[1].item<float>(),
                                  phi[0].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_weno3(phi[2].item<double>(), phi[1].item<double>(),
                                  phi[0].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3p_torch5) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno3(phi[0][2].item<float>(), phi[0][1].item<float>(),
                       phi[0][0].item<float>());
      auto result2 =
          interp_weno3(phi[1][2].item<float>(), phi[1][1].item<float>(),
                       phi[1][0].item<float>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->right(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno3(phi[0][2].item<double>(), phi[0][1].item<double>(),
                       phi[0][0].item<double>());
      auto result2 =
          interp_weno3(phi[1][2].item<double>(), phi[1][1].item<double>(),
                       phi[1][0].item<double>());
      auto result = torch::zeros({2, 1}, phi.options());
      interp->right(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno3p_torch6) {
  Weno3Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({3, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno3(phi[2][0].item<float>(), phi[1][0].item<float>(),
                       phi[0][0].item<float>());
      auto result2 =
          interp_weno3(phi[2][1].item<float>(), phi[1][1].item<float>(),
                       phi[0][1].item<float>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->right(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno3(phi[2][0].item<double>(), phi[1][0].item<double>(),
                       phi[0][0].item<double>());
      auto result2 =
          interp_weno3(phi[2][1].item<double>(), phi[1][1].item<double>(),
                       phi[0][1].item<double>());
      auto result = torch::zeros({1, 2}, phi.options());
      interp->right(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5m_torch1) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 = interp_weno5(phi[0].item<float>(), phi[1].item<float>(),
                                  phi[2].item<float>(), phi[3].item<float>(),
                                  phi[4].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_weno5(phi[0].item<double>(), phi[1].item<double>(),
                                  phi[2].item<double>(), phi[3].item<double>(),
                                  phi[4].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->left(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5m_torch2) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno5(phi[0][0].item<float>(), phi[0][1].item<float>(),
                       phi[0][2].item<float>(), phi[0][3].item<float>(),
                       phi[0][4].item<float>());
      auto result2 =
          interp_weno5(phi[1][0].item<float>(), phi[1][1].item<float>(),
                       phi[1][2].item<float>(), phi[1][3].item<float>(),
                       phi[1][4].item<float>());
      torch::Tensor result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<float>(), 2.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 2.E-6);
    } else {
      auto result1 =
          interp_weno5(phi[0][0].item<double>(), phi[0][1].item<double>(),
                       phi[0][2].item<double>(), phi[0][3].item<double>(),
                       phi[0][4].item<double>());
      auto result2 =
          interp_weno5(phi[1][0].item<double>(), phi[1][1].item<double>(),
                       phi[1][2].item<double>(), phi[1][3].item<double>(),
                       phi[1][4].item<double>());
      torch::Tensor result = torch::zeros({2, 1}, phi.options());
      interp->left(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5m_torch3) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({5, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno5(phi[0][0].item<float>(), phi[1][0].item<float>(),
                       phi[2][0].item<float>(), phi[3][0].item<float>(),
                       phi[4][0].item<float>());
      auto result2 =
          interp_weno5(phi[0][1].item<float>(), phi[1][1].item<float>(),
                       phi[2][1].item<float>(), phi[3][1].item<float>(),
                       phi[4][1].item<float>());
      torch::Tensor result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno5(phi[0][0].item<double>(), phi[1][0].item<double>(),
                       phi[2][0].item<double>(), phi[3][0].item<double>(),
                       phi[4][0].item<double>());
      auto result2 =
          interp_weno5(phi[0][1].item<double>(), phi[1][1].item<double>(),
                       phi[2][1].item<double>(), phi[3][1].item<double>(),
                       phi[4][1].item<double>());
      torch::Tensor result = torch::zeros({1, 2}, phi.options());
      interp->left(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5p_torch4) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 = interp_weno5(phi[4].item<float>(), phi[3].item<float>(),
                                  phi[2].item<float>(), phi[1].item<float>(),
                                  phi[0].item<float>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<float>(), 1.E-6);
    } else {
      auto result1 = interp_weno5(phi[4].item<double>(), phi[3].item<double>(),
                                  phi[2].item<double>(), phi[1].item<double>(),
                                  phi[0].item<double>());
      auto result2 = torch::zeros({1}, phi.options());
      interp->right(phi, 0, result2);
      EXPECT_NEAR(result1, result2.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5p_torch5) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 5}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno5(phi[0][4].item<float>(), phi[0][3].item<float>(),
                       phi[0][2].item<float>(), phi[0][1].item<float>(),
                       phi[0][0].item<float>());
      auto result2 =
          interp_weno5(phi[1][4].item<float>(), phi[1][3].item<float>(),
                       phi[1][2].item<float>(), phi[1][1].item<float>(),
                       phi[1][0].item<float>());
      torch::Tensor result = torch::zeros({2, 1}, phi.options());
      interp->right(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno5(phi[0][4].item<double>(), phi[0][3].item<double>(),
                       phi[0][2].item<double>(), phi[0][1].item<double>(),
                       phi[0][0].item<double>());
      auto result2 =
          interp_weno5(phi[1][4].item<double>(), phi[1][3].item<double>(),
                       phi[1][2].item<double>(), phi[1][1].item<double>(),
                       phi[1][0].item<double>());
      torch::Tensor result = torch::zeros({2, 1}, phi.options());
      interp->right(phi, 1, result);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_weno5p_torch6) {
  Weno5Interp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({5, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1 =
          interp_weno5(phi[4][0].item<float>(), phi[3][0].item<float>(),
                       phi[2][0].item<float>(), phi[1][0].item<float>(),
                       phi[0][0].item<float>());
      auto result2 =
          interp_weno5(phi[4][1].item<float>(), phi[3][1].item<float>(),
                       phi[2][1].item<float>(), phi[1][1].item<float>(),
                       phi[0][1].item<float>());
      torch::Tensor result = torch::zeros({1, 2}, phi.options());
      interp->right(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<float>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<float>(), 1.E-6);
    } else {
      auto result1 =
          interp_weno5(phi[4][0].item<double>(), phi[3][0].item<double>(),
                       phi[2][0].item<double>(), phi[1][0].item<double>(),
                       phi[0][0].item<double>());
      auto result2 =
          interp_weno5(phi[4][1].item<double>(), phi[3][1].item<double>(),
                       phi[2][1].item<double>(), phi[1][1].item<double>(),
                       phi[0][1].item<double>());
      torch::Tensor result = torch::zeros({1, 2}, phi.options());
      interp->right(phi, 0, result);
      result = result.squeeze(0);
      EXPECT_NEAR(result1, result[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result2, result[1].item<double>(), 1.E-6);
    }
  }
}

TEST(weno5, interp_weno5b) {
  double result = interp_weno5<double>(1, 2, 3, 4, 5);

  std::cout << "=== 1 pixels ===" << std::endl;

  result = interp_weno5<double>(1e9, 2, 3, 4, 5);
  std::cout << "1) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 3, 4, 5);
  std::cout << "2) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 1e9, 4, 5);
  std::cout << "x3) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 3, 1e9, 5);
  std::cout << "4) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 3, 4, 1e9);
  std::cout << "5) " << result << std::endl;

  std::cout << "=== 2 pixels ===" << std::endl;

  result = interp_weno5<double>(1e9, 1e9, 3, 4, 5);
  std::cout << "1) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 1e9, 4, 5);
  std::cout << "x2) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 3, 1e9, 5);
  std::cout << "3) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 3, 4, 1e9);
  std::cout << "4) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 1e9, 4, 5);
  std::cout << "x5) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 3, 1e9, 5);
  std::cout << "6) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 3, 4, 1e9);
  std::cout << "7) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 1e9, 1e9, 5);
  std::cout << "x8) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 1e9, 4, 1e9);
  std::cout << "x9) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 3, 1e9, 1e9);
  std::cout << "10) " << result << std::endl;

  std::cout << "=== 3 pixels ===" << std::endl;
  result = interp_weno5<double>(1e9, 1e9, 1e9, 4, 5);
  std::cout << "x1) " << result << std::endl;

  result = interp_weno5<double>(1e9, 1e9, 3, 1e9, 5);
  std::cout << "2) " << result << std::endl;

  result = interp_weno5<double>(1e9, 1e9, 3, 4, 1e9);
  std::cout << "3) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 1e9, 1e9, 5);
  std::cout << "x4) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 1e9, 4, 1e9);
  std::cout << "x5) " << result << std::endl;

  result = interp_weno5<double>(1e9, 2, 3, 1e9, 1e9);
  std::cout << "6) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 1e9, 1e9, 5);
  std::cout << "x7) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 1e9, 4, 1e9);
  std::cout << "x8) " << result << std::endl;

  result = interp_weno5<double>(1, 1e9, 3, 1e9, 1e9);
  std::cout << "9) " << result << std::endl;

  result = interp_weno5<double>(1, 2, 1e9, 1e9, 1e9);
  std::cout << "x10) " << result << std::endl;

  std::cout << "=== 4 pixels ===" << std::endl;
  result = interp_weno5<double>(1e9, 1e9, 3, 1e9, 1e9);
  std::cout << "1) " << result << std::endl;

  result = interp_weno5<double>(1, 1, -3 / 1e9, 1, 1);
  std::cout << "1) " << result << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
