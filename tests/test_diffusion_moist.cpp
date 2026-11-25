// C/C++
#include <string>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"
#include "diffusion_energy.hpp"

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block(std::string eos_type) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  options->hydro()->eos()->type() = std::move(eos_type);
  return std::make_shared<MeshBlockImpl>(options);
}

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block,
                             torch::Device device, torch::Dtype dtype) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::device(device).dtype(dtype));
  w[IDN] = 1.;
  w[IPR] = 1.e5;
  w[ICY] = 0.1;
  w[ICY + 1] = 0.2;
  return w;
}

}  // namespace

TEST_P(DeviceTest, moist_conduction_uses_local_mixture_specific_heat) {
  for (auto const& eos_type :
       std::vector<std::string>{"ideal-moist", "moist-mixture"}) {
    auto block = make_block(eos_type);
    block->to(device, dtype);
    auto peos = block->phydro->peos;
    auto w = make_primitive(block, device, dtype);
    auto x = block->pcoord->x1v.to(device, dtype);
    auto temp = x.square().view({1, 1, -1});
    auto cv = peos->specific_heat_cv(w, temp);
    auto expected_cv = 0.7 * peos->species_cv_ref(0) +
                       0.1 * peos->species_cv_ref(1) +
                       0.2 * peos->species_cv_ref(2);
    EXPECT_TRUE(
        torch::allclose(cv, torch::full_like(cv, expected_cv), 1.e-5, 1.e-5));

    auto du = torch::zeros_like(w);
    diffuse(*block, du, w, temp, 0.1);
    EXPECT_NEAR(du[IPR][0][0][4].item<double>(), 0.05 * expected_cv, 1.e-3);
  }
}

TEST(diffusion,
     dynamic_conduction_timestep_uses_local_volumetric_heat_capacity) {
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_diffusion_dynamic_cv.yaml");
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto peos = block->phydro->peos;
  auto diffusion = block->phydro->pdiffusion;
  auto w = make_primitive(block, torch::kCPU, torch::kFloat64);
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int ng = block->pcoord->options->nghost();

  // Vary density and composition independently. The minimum density is dry,
  // while the minimum rho*cv is a denser, vapor-rich cell; multiplying
  // separate minima therefore gives neither cell's volumetric heat capacity.
  w[IDN].fill_(1.1);
  w[ICY].fill_(0.9);
  w[ICY + 1].zero_();
  w[IDN].narrow(-1, ng, 1).fill_(1.);
  w[ICY].narrow(-1, ng, 1).zero_();
  w[IPR].fill_(1.e5);

  auto temp = peos->compute("W->T", {w});
  auto rho_cv = w[IDN] * peos->specific_heat_cv(w, temp);
  auto rho_cv_min = rho_cv.index(interior).min().item<double>();
  auto rho_min = w[IDN].index(interior).min().item<double>();
  ASSERT_LT(rho_cv_min, rho_min * peos->species_cv_ref());

  // dx=1 and ndim=1 in this fixture.
  double expected = rho_cv_min / (2. * diffusion->options->kappa_iso());
  EXPECT_NEAR(diffusion->max_time_step({w}), expected, 1.e-12 * expected);
}

TEST_P(DeviceTest, conserved_limiter_uses_nucleation_parent_metadata) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  options->hydro()->eos()->limiter() = true;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);

  auto coord = block->pcoord;
  auto cons = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
  cons[IDN].fill_(1.);
  cons[IPR].fill_(1.e8);
  cons[ICY].fill_(0.3);
  cons[ICY + 1].fill_(-0.1);
  auto total_before = cons[IDN] + cons.narrow(0, ICY, 2).sum(0);

  block->phydro->peos->apply_conserved_limiter_(cons);

  auto total_after = cons[IDN] + cons.narrow(0, ICY, 2).sum(0);
  EXPECT_TRUE(torch::allclose(total_after, total_before, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(cons[ICY], torch::full_like(cons[ICY], 0.2),
                              1.e-6, 1.e-6));
  EXPECT_TRUE(torch::equal(cons[ICY + 1], torch::zeros_like(cons[ICY + 1])));
}
