// C/C++
#include <cmath>
#include <cstdio>

// external
#include <gtest/gtest.h>

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

// Reflecting column, WENO5, gravity on, implicit correction off.
// One upper cell is dipped (1e-4) and the cell below it is cut to 0.2, so the
// well-balanced reconstruction overshoots non-positive and the positivity
// floor fires. When the floor writes the adjacent cell's density, the mass
// flux at that face is -6.3e-11. Writing the density reference dsf instead,
// with that same reference, makes it -9.5e-8. When the reference is still the
// bottom-anchored isentrope, it is -1.3e-6.
std::shared_ptr<MeshBlockImpl> make_block(torch::Device device) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_face_floor.yaml");
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-10.);
  options->hydro()->grav() = gravity;
  options->hydro()->icorr() = nullptr;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, torch::kFloat64);
  return block;
}

//! the dipped column, one forward; returns the block (fluxes in flux1())
std::shared_ptr<MeshBlockImpl> dipped_column(torch::Device device,
                                             double pscale) {
  auto block = make_block(device);
  auto coord = block->pcoord;
  int iu = coord->iu();
  int dipped = iu - 1;

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::dtype(torch::kFloat64).device(device));
  auto rho = torch::exp(-coord->x1v / 2.);
  rho.select(-1, dipped).mul_(1.e-4);
  rho.select(-1, dipped - 1).mul_(0.2);
  w[IDN].copy_(rho);
  w[IPR].copy_(torch::exp(-coord->x1v / 2.) * pscale);

  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  block->phydro->forward(1.e-4, u, vars);
  return block;
}

void face_floor_uses_adjacent_density(torch::Device device) {
  auto block = dipped_column(device, 1.e5);
  int dipped = block->pcoord->iu() - 1;

  auto mass = block->phydro->flux1()[IDN];
  auto mom = block->phydro->flux1()[IVX];
  ASSERT_EQ(mass.device(), device);
  double dipped_mass = mass.select(-1, dipped).item<double>();
  double dipped_mom = mom.select(-1, dipped).item<double>();
  double lower_mass = mass.select(-1, 4).item<double>();

  // The column is not in balance, so a face away from the dip still carries
  // an O(1) mass flux. A run that dropped every flux would not pass.
  EXPECT_NEAR(lower_mass, 4.07888, 1.e-3);
  // #289, in every arm (ctest test_face_floor_wb_ref4): the switch replaces
  // only the FACE density reference, so the cell reference still does not
  // follow the dip, rho' there stays large and the floor fires on the same
  // face. The values below are the switched arms' too, to every digit.
  EXPECT_LT(std::abs(dipped_mass), 1.e-9);
  EXPECT_GT(dipped_mom, 0.03042);
}

// #289, with or without SNAP_WB_REF4: at p = 10 e^{-x/2} a cell's scan
// pressure drops by e^{g dz rho/p} = e^1 > e^{0.5}, so every cell is flagged
// and keeps the kernel's reference, which does not follow the dip; the
// overshoot is then floored to the adjacent density, as with the switch off:
// the same flux in every arm (with the linear/ln wall continuation of rho/p)
void face_floor_fires_on_an_unresolved_column(torch::Device device) {
  auto block = dipped_column(device, 10.);
  int dipped = block->pcoord->iu() - 1;
  double dipped_mass =
      block->phydro->flux1()[IDN].select(-1, dipped).item<double>();
  std::printf("wb_ref4 %d, p = 10 e^{-x/2}: dipped face mass flux %.4e\n",
              static_cast<int>(HydroImpl::wb_ref4()), dipped_mass);
  EXPECT_NEAR(dipped_mass, 2.83191e-8, 1.e-5 * 2.83191e-8);
}

}  // namespace

TEST(hydro, face_floor_uses_adjacent_density) {
  face_floor_uses_adjacent_density(torch::kCPU);
}

TEST(hydro, face_floor_fires_on_an_unresolved_column) {
  face_floor_fires_on_an_unresolved_column(torch::kCPU);
}

TEST(hydro, face_floor_uses_adjacent_density_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  face_floor_uses_adjacent_density(torch::Device(torch::kCUDA, 0));
}

// the unresolved column on the GPU floors the same face: its mass flux is the
// CPU's, to the CPU check's relative tolerance (the CPU value is pinned above)
TEST(hydro, face_floor_fires_on_an_unresolved_column_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  auto dipped_mass = [](torch::Device device) {
    auto block = dipped_column(device, 10.);
    int dipped = block->pcoord->iu() - 1;
    return block->phydro->flux1()[IDN].select(-1, dipped).item<double>();
  };
  double cpu = dipped_mass(torch::kCPU);
  double cuda = dipped_mass(torch::Device(torch::kCUDA, 0));
  std::printf(
      "wb_ref4 %d, p = 10 e^{-x/2}: dipped face mass flux %.4e (CUDA)"
      ", %.4e (CPU)\n",
      static_cast<int>(HydroImpl::wb_ref4()), cuda, cpu);
  EXPECT_NEAR(cuda, cpu, 1.e-5 * std::abs(cpu));
}
