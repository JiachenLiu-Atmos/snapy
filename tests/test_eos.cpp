// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snapy
#include <snap/eos/fix_vapor_impl.h>
#include <snap/snap.h>

#include <cmath>
#include <snap/eos/ideal_moist.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block(std::string eos_type,
                                          std::string file = "test_eos.yaml") {
  auto options = MeshBlockOptionsImpl::from_yaml(file);
  options->hydro()->eos()->type() = std::move(eos_type);
  return std::make_shared<MeshBlockImpl>(options);
}

// uniform cell volumes for the fix_vapor_impl columns below
std::vector<double> const vol(4, 1.);

}  // namespace

TEST(eos_limiter, accepts_zero_vapor_column) {
  std::vector<double> vapor(4, 0.);
  std::vector<double> major(4, 1.);

  EXPECT_EQ(
      fix_vapor_impl(vapor.data(), major.data(), vol.data(), vapor.size()), 0);
  for (double value : vapor) EXPECT_DOUBLE_EQ(value, 0.);

  vapor = {0.2, -0.1, 0., 0.};
  EXPECT_EQ(
      fix_vapor_impl(vapor.data(), major.data(), vol.data(), vapor.size()), 0);
  EXPECT_DOUBLE_EQ(vapor[0], 0.1);
  EXPECT_DOUBLE_EQ(vapor[1], 0.);
  EXPECT_DOUBLE_EQ(vapor[2], 0.);
  EXPECT_DOUBLE_EQ(vapor[3], 0.);
}

namespace {

double column_sum(std::vector<double> const& v) {
  double s = 0.;
  for (double x : v) s += x;
  return s;
}

}  // namespace

// A bottom-cell deficit has nothing below it; the measured column
TEST(eos_limiter, repairs_bottom_cell_from_above) {
  std::vector<double> major = {1.00, 0.85, 0.72, 0.61};
  std::vector<double> vapor = {-1.7e-21, 4.4e-11, 1.0e-11, 1.0e-11};
  double before = column_sum(vapor);

  EXPECT_EQ(
      fix_vapor_impl(vapor.data(), major.data(), vol.data(), vapor.size()), 0);
  for (double value : vapor) EXPECT_GE(value, 0.);
  EXPECT_NEAR(column_sum(vapor), before, 1.e-14 * std::abs(before));

  // the repaired cell is zeroed, not left holding what it just borrowed
  EXPECT_DOUBLE_EQ(vapor[0], 0.);
  // Bounded transfer, asserted on the TRANSFER: the deficit is 1.7e-21 against
  // a vapor[1] of 4.4e-11, so any tolerance that survives round-off on the
  // latter also passes a supplier that gave nothing. The subtraction is exact
  // (Sterbenz), leaving only the one rounding in `vapor[1] -= take`.
  EXPECT_NEAR(4.4e-11 - vapor[1], 1.7e-21, 1.e-25);
  EXPECT_DOUBLE_EQ(vapor[2], 1.0e-11);
  EXPECT_DOUBLE_EQ(vapor[3], 1.0e-11);
}

// a column in net deficit is unrepairable: report it, leave it untouched
TEST(eos_limiter, rejects_column_in_net_deficit_without_writing) {
  std::vector<double> major(4, 1.);
  std::vector<double> vapor = {-0.3, 0.1, 0.1, 0.};
  std::vector<double> const expected = vapor;

  EXPECT_EQ(
      fix_vapor_impl(vapor.data(), major.data(), vol.data(), vapor.size()), 1);
  EXPECT_EQ(vapor, expected);
}

// nothing to borrow from in either direction
TEST(eos_limiter, rejects_single_cell_column) {
  std::vector<double> major = {1.};
  std::vector<double> vapor = {-1.e-30};

  EXPECT_EQ(fix_vapor_impl(vapor.data(), major.data(), vol.data(), 1), 1);
}

// a deficit the cells below can cover is borrowed from them, in proportion
// to their vapor, whatever the dry density: no longer flattened to one mass
// fraction
TEST(eos_limiter, downward_branch_borrows_the_deficit) {
  std::vector<double> major = {1.00, 0.85, 0.72, 0.61};
  std::vector<double> vapor = {0.2, -0.1, 0., 0.};
  double before = column_sum(vapor);

  EXPECT_EQ(
      fix_vapor_impl(vapor.data(), major.data(), vol.data(), vapor.size()), 0);
  EXPECT_DOUBLE_EQ(vapor[0], 0.2 - 0.2 * (0.1 / 0.2));
  EXPECT_DOUBLE_EQ(vapor[1], 0.);
  EXPECT_NEAR(column_sum(vapor), before, 1.e-14 * std::abs(before));
}

TEST_P(DeviceTest, moist_mixture) {
  auto block = make_block("moist-mixture");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto cons =
      torch::empty({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));

  cons.uniform_(0., 1.);
  cons[IDN].add_(1.);
  // Keep thermal energy well-conditioned without scaling it with grid size.
  cons[IPR].mul_(1.e5).add_(1.e5);
  cons[IPR] += 0.5 * cons.narrow(0, IVX, 3).pow(2).sum(0) / cons[IDN];

  auto prim = peos->forward(cons);
  auto cons2 = peos->compute("W->U", {prim});

  EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-6, 1.E-6));

  auto gamma = peos->compute("W->A", {prim});

  EXPECT_TRUE(gamma.allclose(torch::ones_like(gamma) * 1.4, 1.E-6, 1.E-6));

  auto cs = peos->compute("WA->L", {prim, gamma});
  std::cout << "cs min = " << cs.min() << std::endl;
  std::cout << "cs max = " << cs.max() << std::endl;

  // Cache validation must follow tensor identity and ATen's mutation version,
  // without retaining a full primitive-field copy. Mutating pressure must
  // invalidate the cached temperature, and a distinct tensor with identical
  // contents must still produce the same refreshed result.
  auto temp_before = peos->compute("W->T", {prim}).clone();
  prim[IPR].mul_(1.25);
  auto temp_after = peos->compute("W->T", {prim});
  auto temp_reference = peos->compute("W->T", {prim.clone()});
  EXPECT_FALSE(torch::allclose(temp_before, temp_after, 1.E-6, 1.E-6));
  EXPECT_TRUE(torch::allclose(temp_after, temp_reference, 1.E-6, 1.E-6));
}

// Repeating a public W->T call must build a fresh backward graph.
TEST_P(DeviceTest, ideal_moist_temperature_repeated_backward) {
  auto block = make_block("ideal-moist", "test_eos_temp2inteng.yaml");
  block->to(device, dtype);
  auto eos = block->phydro->peos;
  auto prim = torch::ones(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::device(device).dtype(dtype));
  prim[IPR].fill_(1.e5);
  prim.narrow(0, ICY, eos->nvar() - ICY).fill_(0.01);
  prim.requires_grad_(true);
  eos->compute("W->T", {prim}).sum().backward();
  auto expected = prim.grad().clone();
  prim.mutable_grad() = torch::Tensor();
  auto temp = eos->compute("W->T", {prim});
  EXPECT_EQ(temp.device(), prim.device());
  EXPECT_EQ(temp.scalar_type(), dtype);
  temp.sum().backward();
  EXPECT_TRUE(torch::equal(prim.grad(), expected));
  {
    torch::NoGradGuard no_grad;
    auto input_before = prim.clone();
    auto temp_before = temp.clone();
    auto second = eos->compute("W->T", {prim});
    EXPECT_FALSE(temp.is_alias_of(prim));
    EXPECT_FALSE(second.is_alias_of(prim));
    EXPECT_FALSE(second.is_alias_of(temp));
    EXPECT_TRUE(torch::equal(second, temp_before));
    second.add_(1);
    EXPECT_TRUE(torch::equal(temp, temp_before));
    EXPECT_TRUE(torch::equal(prim, input_before));
  }
}

// Warming enthalpy in inference mode must not poison a later tracked call.
TEST_P(DeviceTest, ideal_moist_enthalpy_after_inference) {
  auto block = make_block("ideal-moist", "test_eos_temp2inteng.yaml");
  block->to(device, dtype);
  auto eos = block->phydro->peos;
  auto prim = torch::ones(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::device(device).dtype(dtype));
  prim[IPR].fill_(1.e5);
  prim.narrow(0, ICY, eos->nvar() - ICY).fill_(0.01);
  {
    c10::InferenceMode inference;
    eos->species_enthalpy(prim);
  }
  auto tracked = prim.clone().requires_grad_(true);
  auto enthalpy = eos->species_enthalpy(tracked);
  EXPECT_EQ(enthalpy.device(), tracked.device());
  EXPECT_EQ(enthalpy.scalar_type(), dtype);
  enthalpy.sum().backward();
  EXPECT_TRUE(torch::isfinite(tracked.grad()).all().item<bool>());
  auto fresh = make_block("ideal-moist", "test_eos_temp2inteng.yaml");
  fresh->to(device, dtype);
  auto reference = prim.clone().requires_grad_(true);
  auto expected = fresh->phydro->peos->species_enthalpy(reference);
  expected.sum().backward();
  EXPECT_TRUE(torch::equal(enthalpy, expected));
  EXPECT_TRUE(torch::equal(tracked.grad(), reference.grad()));
}

TEST_P(DeviceTest, ideal_moist_internal_energy_offset) {
  auto block = make_block("ideal-moist");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  int nvar = peos->nvar();

  auto hydro =
      torch::randn({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  auto offset = peos->internal_energy_offset(hydro);

  auto ideal_moist = std::dynamic_pointer_cast<IdealMoistImpl>(peos);
  ASSERT_TRUE(ideal_moist);

  auto expected = hydro[IDN] * ideal_moist->u0[0].to(dtype).to(device);
  EXPECT_TRUE(torch::allclose(offset, expected, 1.E-6, 1.E-6));

  auto tendency = 0.25 * hydro;
  auto tendency_offset = peos->internal_energy_offset(tendency);
  EXPECT_TRUE(torch::allclose(tendency_offset, 0.25 * offset, 1.E-6, 1.E-6));
}

TEST_P(DeviceTest, ideal_moist_pressure_perturbations_preserve_small_heating) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  options->hydro()->anomaly() = true;
  auto block = MeshBlock(options);
  block->to(device, torch::kFloat64);
  auto eos = std::dynamic_pointer_cast<IdealMoistImpl>(block->phydro->peos);
  auto w = torch::zeros(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::device(device).dtype(torch::kFloat64));
  w[IDN].fill_(10.);
  w[IPR].fill_(1.e7);
  // Exactly representable species densities after W->U, so their initial
  // conversion to float32 cannot masquerade as a thermal perturbation.
  w[ICY].fill_(0.0125);
  w[ICY + 1].fill_(0.025);
  auto u0 = eos->compute("W->U", {w});
  block->phydro->set_reference(u0);
  auto stored = u0.clone();
  block->phydro->sub_reference_(stored);
  block->to(device, dtype);
  stored = stored.to(dtype);
  stored[IPR].fill_(0.01);
  auto before = stored.clone();
  auto result = eos->perturbations(stored);
  auto rho = torch::cat({u0[IDN].unsqueeze(0), u0.narrow(0, ICY, 2)}, 0);
  // Use options for the offset oracle too: module conversion may round u0.
  auto offset = torch::tensor(eos->pthermo->options->uref_R(), w.options()) *
                kintera::constants::Rgas /
                torch::tensor(eos->pthermo->options->mu(), w.options());
  auto expected = stored[IPR].to(torch::kFloat64) * w[IPR] /
                  (u0[IPR] - (rho * offset.view({-1, 1, 1, 1})).sum(0));
  EXPECT_EQ(result.pressure_delta.scalar_type(), dtype);
  EXPECT_EQ(result.pressure0.scalar_type(), torch::kFloat64);
  EXPECT_TRUE(torch::equal(stored, before));
  EXPECT_TRUE(torch::allclose(result.pressure_delta.to(torch::kFloat64),
                              expected, 2.e-6, 1.e-10));
  EXPECT_TRUE(torch::equal(result.density_delta,
                           torch::zeros_like(result.density_delta)));
  EXPECT_TRUE(torch::allclose(result.pressure0, w[IPR], 1.e-14, 1.e-8));
  if (dtype == torch::kFloat32)
    EXPECT_TRUE(
        torch::equal(u0[IPR].to(dtype) + stored[IPR], u0[IPR].to(dtype)));
}

TEST_P(DeviceTest, ideal_moist_pressure_perturbations_include_species_and_ke) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  options->hydro()->anomaly() = true;
  auto block = MeshBlock(options);
  block->to(device, torch::kFloat64);
  auto eos = std::dynamic_pointer_cast<IdealMoistImpl>(block->phydro->peos);
  auto w = torch::zeros(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::device(device).dtype(torch::kFloat64));
  w[IDN].fill_(10.);
  w[IPR].fill_(1.e7);
  w[ICY].fill_(0.0125);
  w[ICY + 1].fill_(0.025);
  auto u0 = eos->compute("W->U", {w});
  block->phydro->set_reference(u0);
  auto stored = u0.clone();
  block->phydro->sub_reference_(stored);
  stored[IDN].fill_(0.001);
  stored[IPR].fill_(100.);
  stored[IVX].fill_(0.03);
  stored[ICY] += 0.001;
  stored[ICY + 1] -= 0.001;
  stored = stored.to(dtype);
  // Oracle uses the actual stored species/increments before module demotion.
  auto total = block->phydro->total_u(stored.to(torch::kFloat64));
  auto expected = eos->compute("U->W", {total});
  block->to(device, dtype);
  auto result = eos->perturbations(stored);
  EXPECT_TRUE(torch::allclose(result.pressure_delta.to(torch::kFloat64),
                              expected[IPR] - w[IPR], 2.e-6, 1.e-7));
  EXPECT_TRUE(torch::allclose(result.density_delta.to(torch::kFloat64),
                              expected[IDN] - w[IDN], 2.e-6, 1.e-12));
  block->phydro->pref->set("species0", torch::empty({0}, w.options()));
  EXPECT_THROW(eos->perturbations(stored), c10::Error);
}

TEST_P(DeviceTest, ideal_moist_pressure_perturbations_without_condensables) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos.yaml");
  options->hydro()->eos()->type() = "ideal-moist";
  options->hydro()->anomaly() = true;
  auto block = MeshBlock(options);
  block->to(device, torch::kFloat64);
  auto eos = std::dynamic_pointer_cast<IdealMoistImpl>(block->phydro->peos);
  auto w = torch::zeros(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::device(device).dtype(torch::kFloat64));
  w[IDN].fill_(10.);
  w[IPR].fill_(1.e7);
  auto u0 = eos->compute("W->U", {w});
  block->phydro->set_reference(u0);
  auto stored = torch::zeros_like(u0).to(dtype);
  stored[IPR].fill_(0.01);
  block->to(device, dtype);
  auto result = eos->perturbations(stored);
  EXPECT_TRUE(torch::allclose(result.pressure_delta,
                              torch::full_like(stored[IPR], 0.004), 2.e-7,
                              1.e-10));
  EXPECT_TRUE(torch::equal(result.density_delta,
                           torch::zeros_like(result.density_delta)));
  // Reject singular and negative total heat capacity before pressure division.
  for (double drho : {-10., -20.}) {
    stored[IDN].fill_(drho);
    EXPECT_THROW(eos->perturbations(stored), c10::Error);
  }
  stored[IDN].zero_();
  auto bad_ref = block->phydro->u_ref().clone();
  bad_ref[0].zero_();
  block->phydro->pref->set("u0", bad_ref);
  EXPECT_THROW(eos->perturbations(stored), c10::Error);
}

TEST(eos_limiter,
     float_anomaly_floor_is_representable_and_preserves_other_cells) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  options->hydro()->anomaly() = true;
  auto block = MeshBlock(options);
  block->to(torch::kCPU, torch::kFloat64);
  auto eos = block->phydro->peos;
  auto w = torch::zeros(
      {eos->nvar(), block->pcoord->options->nc3(),
       block->pcoord->options->nc2(), block->pcoord->options->nc1()},
      torch::kFloat64);
  w[IDN].fill_(10.);
  w[IPR].fill_(1.e7);
  auto u0 = eos->compute("W->U", {w});
  block->phydro->set_reference(u0);
  auto stored = u0.clone();
  block->phydro->sub_reference_(stored);
  stored = stored.to(torch::kFloat32);
  stored[IDN].fill_(1.e-8);
  auto dry = block->phydro->dry_density(stored);
  EXPECT_EQ(dry.scalar_type(), torch::kFloat64);
  EXPECT_GT(dry.min().item<double>(), 10.);
  stored[IDN].zero_();
  // total_u must add the exact reference after promotion, not before it.
  stored[IPR].fill_(0.01);
  auto reconstructed = block->phydro->total_u(stored);
  EXPECT_EQ(reconstructed.scalar_type(), torch::kFloat64);
  EXPECT_TRUE(torch::allclose(reconstructed[IPR] - u0[IPR],
                              stored[IPR].to(torch::kFloat64), 1.e-6, 1.e-8));
  block->to(torch::kCPU, torch::kFloat32);
  eos->options->limiter() = true;
  auto before = stored.clone();
  int i = block->pcoord->il();
  int j = block->pcoord->jl();
  int k = block->pcoord->kl();
  stored[IDN][k][j][i] = -10.;
  stored[IPR][k][j][i + 1] = -1.e9;
  eos->apply_conserved_limiter_(stored, /*whole_column=*/false);
  auto total = block->phydro->total_u(stored);
  EXPECT_GE(total[IDN][k][j][i].item<double>(), eos->options->density_floor());
  auto ke = eos->compute("U->K", {total});
  auto min_ie = eos->compute(
      "UT->I",
      {total, torch::full_like(ke, eos->options->temperature_floor())});
  EXPECT_GE(total[IPR][k][j][i + 1].item<double>(),
            (ke + min_ie)[k][j][i + 1].item<double>());
  stored[IDN][k][j][i] = before[IDN][k][j][i];
  stored[IPR][k][j][i + 1] = before[IPR][k][j][i + 1];
  EXPECT_TRUE(torch::equal(stored, before));
}

TEST_P(DeviceTest, ideal_gas_internal_energy_offset_is_zero) {
  auto block = make_block("ideal-gas");
  block->to(device, dtype);

  auto peos = block->phydro->peos;
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  auto hydro = torch::randn({peos->nvar(), nc3, nc2, nc1},
                            torch::device(device).dtype(dtype));
  auto offset = peos->internal_energy_offset(hydro);
  EXPECT_TRUE(
      torch::allclose(offset, torch::zeros_like(hydro[IDN]), 1.E-12, 1.E-12));
}

/*TEST_P(DeviceTest, cons2prim_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - ICY - ncloud;

  auto cons =
      torch::randn({NHYDRO, 1, 200, 200}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 200, 200}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto start = std::chrono::high_resolution_clock::now();

  auto prim = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);
  auto cons2 = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;

  std::cout << (cons - cons2).min() << std::endl;
  std::cout << (cons - cons2).max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-2, 1.E-2));
  } else {
    EXPECT_TRUE(torch::allclose(cons, cons2, 1.E-9, 1.E-9));
  }
}

TEST_P(DeviceTest, prim2cons_hydro_ideal_ncloud5) {
  int32_t NHYDRO = 14;
  int32_t ncloud = 5;
  int32_t nvapor = NHYDRO - ICY - ncloud;

  auto prim =
      torch::randn({NHYDRO, 1, 5, 5}, torch::device(device).dtype(dtype));
  auto gammad = torch::ones({1, 5, 5}, torch::device(device).dtype(dtype));
  auto rmu =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));
  auto rcv =
      torch::randn({nvapor + ncloud}, torch::device(device).dtype(dtype));

  gammad *= 1.4;
  rmu.normal_(0, 1);
  rcv.normal_(0, 1);

  auto cons = eos_prim2cons_hydro_ideal(prim, gammad, rmu, rcv, ncloud);
  auto prim2 = eos_cons2prim_hydro_ideal(cons, gammad, rmu, rcv, ncloud);

  std::cout << (prim - prim2).abs().max() << std::endl;

  if (dtype == torch::kFloat32) {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-4, 1.E-4));
  } else {
    EXPECT_TRUE(torch::allclose(prim, prim2, 1.E-12, 1.E-12));
  }
}*/

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
