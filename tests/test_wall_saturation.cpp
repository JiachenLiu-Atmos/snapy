#include <gtest/gtest.h>
#include <kintera/constants.h>
#include <torch/torch.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <snap/mesh/meshblock.hpp>

#include "cuda_test_gate.hpp"

using namespace snap;

class WallSaturation : public testing::TestWithParam<torch::DeviceType> {};

TEST_P(WallSaturation, phase_change_preserves_energy_and_water) {
  auto device = torch::Device(GetParam());
  if (device.is_cuda() && !snapy_cuda_test_enabled()) GTEST_SKIP();
  torch::set_num_threads(1);
  for (bool condense : {true, false}) {
    SCOPED_TRACE(condense ? "condensation" : "evaporation");
    for (bool upper : {false, true}) {
      SCOPED_TRACE(upper ? "upper wall" : "lower wall");
      auto b = MeshBlock(
          MeshBlockOptionsImpl::from_yaml("test_wall_saturation.yaml"));
      b->to(device, torch::kFloat64);
      auto c = b->pcoord->options;
      auto w =
          torch::zeros({b->phydro->peos->nvar(), c->nc3(), c->nc2(), c->nc1()},
                       torch::dtype(torch::kFloat64).device(device));
      double q0 = condense ? 2.7e-3 : 0.005;
      double temp = condense ? 278.15 : 320.;
      auto rgas =
          kintera::constants::Rgas * ((1. - q0) / 0.0289700 + q0 / 0.01801528);
      w[IPR].fill_(1.e5);
      w[IDN].fill_(1.e5 / (rgas * temp));
      auto z = b->pcoord->x1v;
      if (upper) z = 64. - z;
      if (condense) {
        w[ICY].copy_((q0 * (1. + 2. * torch::exp(-z / 6.))).view({1, 1, -1}));
      } else {
        w[ICY].fill_(q0);
        w[ICY + 1].select(-1, upper ? c->nc1() - 4 : 3).fill_(1.e-3);
      }
      Variables v{{"hydro_w", w}};
      b->initialize(v);
      auto interior = b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
      auto volume = b->pcoord->cell_volume().index(interior);
      auto total = [&](int slot) {
        return (v.at("hydro_u")[slot].index(interior) * volume)
            .sum()
            .item<double>();
      };
      double e0 = total(IPR), water0 = total(ICY) + total(ICY + 1);
      double cloud0 = total(ICY + 1);
      double dt_max = b->max_time_step(v);
      ASSERT_GT(dt_max, 0.);
      ASSERT_TRUE(std::isfinite(dt_max));
      int steps = std::ceil(1. / dt_max);
      ASSERT_LT(steps, 10000);
      for (int n = 0; n < steps; ++n) {
        for (int stage = 0; stage < b->pintg->stages.size(); ++stage)
          b->forward(v, 1. / steps, stage);
      }
      double de = std::abs(total(IPR) / e0 - 1.);
      double dw = std::abs((total(ICY) + total(ICY + 1)) / water0 - 1.);
      std::cout << std::setprecision(12) << device.str() << " "
                << (condense ? "condensation " : "evaporation ")
                << (upper ? "upper" : "lower") << " energy=" << de
                << " water=" << dw << '\n';
      if (condense)
        EXPECT_GT(total(ICY + 1), cloud0 + 1.e-6);
      else
        EXPECT_LT(total(ICY + 1), cloud0 * 1.e-6);
      EXPECT_LT(de, 1.e-12);
      EXPECT_LT(dw, 1.e-12);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Devices, WallSaturation,
    testing::Values(torch::kCPU
#ifdef USE_CUDA
                    ,
                    torch::kCUDA
#endif
                    ),
    [](const testing::TestParamInfo<torch::DeviceType>& info) {
      return torch::Device(info.param).str();
    });
