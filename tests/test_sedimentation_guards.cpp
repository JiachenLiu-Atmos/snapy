// C/C++
#include <cstdio>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {
//! the one card this binary loads: kintera's species table is process-global.
//! It settles `cloud` at const-vsed -2 and carries no forcing block.
constexpr char const* kCard = "test_gravity_sedimentation.yaml";

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[ICY].fill_(0.01);
  w[ICY + 1].fill_(0.02);
  return w;
}

//! A rising cloud (const-vsed +2) leaves the cell BELOW a face, so face i must
//! carry rho*y*vsed and the energy of that mass from cell i-1; settling takes
//! cell i. The cloud differs per cell, so the two donors differ. Both walls
//! stay sealed. grav1 is -1e-12, not 0 (sedimentation is a null op at 0), so
//! the Riemann flux of the resting column is negligible.
void expect_rising_cloud_taken_from_below(torch::Device device) {
  auto card = YAML::LoadFile(kCard);
  card["sedimentation"]["const-vsed"]["cloud"] = 2.;
  std::string name = "test_sedimentation_rising.yaml";
  {
    std::ofstream(name) << card;
  }
  auto options = MeshBlockOptionsImpl::from_yaml(name);
  std::remove(name.c_str());
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.e-12);
  options->hydro()->grav() = gravity;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device);

  int il = block->pcoord->il(), iu = block->pcoord->iu();
  auto w = make_primitive(block);
  for (int i = il; i <= iu; ++i) {
    w[ICY + 1].select(-1, i).fill_(0.01 * (i - il + 1));
  }
  auto peos = block->phydro->peos;
  auto u = peos->compute("W->U", {w.to(device)});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(u);
  block->phydro->forward(0.1, u, vars);

  auto flux = block->phydro->flux1().cpu();
  auto en = peos->compute("W->E", {w.to(device)}).cpu()[1];  // the cloud
  auto at = [](torch::Tensor const& t, int i) {
    return t[0][0][i].item<double>();
  };
  for (int i = il + 1; i <= iu; ++i) {
    double below = 2. * at(w[IDN], i - 1) * at(w[ICY + 1], i - 1);
    double above = 2. * at(w[IDN], i) * at(w[ICY + 1], i);
    EXPECT_NEAR(at(flux[ICY + 1], i), below, 1.e-9 * below)
        << "face " << i << ": the cell above gives " << above;
    double e_below = 2. * at(en, i - 1);
    EXPECT_NEAR(at(flux[IPR], i), e_below, 1.e-9 * std::abs(e_below))
        << "face " << i << ": the cell above gives " << 2. * at(en, i);
  }
  EXPECT_EQ(at(flux[ICY + 1], il), 0.) << "the bottom wall leaks";
  EXPECT_EQ(at(flux[ICY + 1], iu + 1), 0.) << "the top wall leaks";
}
}  // namespace

// EF-6 (D8): the one kernel takes the species energy from the EOS, so a
// settling face carries exactly vsed times "W->E" of its donor cell (the old
// ideal-moist kernel rebuilt it inline and rounded differently)
TEST(sedimentation, settling_energy_is_the_eos_species_energy) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  options->hydro()->grav() = gravity;
  auto block = std::make_shared<MeshBlockImpl>(options);
  int il = block->pcoord->il(), iu = block->pcoord->iu();
  auto w = make_primitive(block);
  for (int i = il; i <= iu; ++i) {
    w[IVX].select(-1, i).fill_(0.3 * (i - il) - 0.7);
    w[IVY].select(-1, i).fill_(1.9 + 0.13 * i);
    w[ICY].select(-1, i).fill_(0.0137 * (i - il + 1));
    w[ICY + 1].select(-1, i).fill_(0.0071 * (i - il + 2));
    w[IPR].select(-1, i).fill_(1.e5 * (1. - 0.03 * (i - il)));
  }
  auto flux = block->phydro->psed->forward(w);
  auto en = block->phydro->peos->compute("W->E", {w})[1];  // the cloud
  for (int i = il + 1; i <= iu; ++i)  // settling at -2 takes cell i
    EXPECT_EQ(flux[IPR][0][0][i].item<double>(),
              -2. * en[0][0][i].item<double>())
        << "face " << i;
}

// EF-12 (D14): SedVel is folded into SedHydro, which holds the particle
// buffers itself, rebuilt from the options at reset()
TEST(sedimentation, particle_buffers_are_sedhydros_own) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  options->hydro()->grav() = gravity;
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto buffers = block->phydro->psed->named_buffers();
  for (auto name : {"radius", "density", "const_vsed"})
    EXPECT_TRUE(buffers.contains(name)) << name;
  EXPECT_FALSE(buffers.contains("sedvel.radius"));
  if (buffers.contains("const_vsed"))
    EXPECT_EQ(buffers["const_vsed"][0].item<double>(), -2.);
}

// sedimentation reads grav1 in forward(); without const-gravity that was a null
// dereference on the first step, so the card is refused at setup instead
TEST(sedimentation, refuses_a_card_without_const_gravity) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  ASSERT_TRUE(options->hydro()->sed());
  ASSERT_FALSE(options->hydro()->grav());
  try {
    std::make_shared<MeshBlockImpl>(options);
    ADD_FAILURE() << "sedimentation without const-gravity accepted";
  } catch (std::exception const& e) {
    EXPECT_NE(std::string(e.what()).find("set forcing/const-gravity"),
              std::string::npos)
        << e.what();
  }
}

// With the x1 flux off nothing rewrites _flux1, so a settling flux added into
// it grew by one more copy on every call. from_yaml zeroes grav1 under
// disable-flux-x1, which makes sedimentation return early, so only options
// built in code reach this.
TEST(sedimentation, is_skipped_when_the_x1_flux_is_off) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  options->hydro()->grav() = gravity;
  options->hydro()->disable_flux_x1() = true;
  auto block = std::make_shared<MeshBlockImpl>(options);

  auto w = make_primitive(block);
  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);

  auto du1 = block->phydro->forward(0.1, u.clone(), vars).clone();
  auto du2 = block->phydro->forward(0.1, u.clone(), vars).clone();

  // no x1 flux of any kind: the cloud does not move
  EXPECT_EQ(du1[ICY + 1].abs().max().item<double>(), 0.);
  EXPECT_TRUE(torch::equal(du1, du2))
      << "max |du2 - du1| = " << (du2 - du1).abs().max().item<double>();
}

TEST(sedimentation, rising_cloud_is_taken_from_the_cell_below) {
  expect_rising_cloud_taken_from_below(torch::kCPU);
}

TEST(sedimentation, rising_cloud_is_taken_from_the_cell_below_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA device";
  expect_rising_cloud_taken_from_below(torch::kCUDA);
}
