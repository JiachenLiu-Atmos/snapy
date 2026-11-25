// C/C++
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/bc/bc_func.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block() {
  return std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml"));
}

void make_x1_periodic(MeshBlockOptions const& options) {
  options->bfuncs()[BoundaryFace::kInnerX1] =
      get_bc_func().at("periodic_inner");
  options->bfuncs()[BoundaryFace::kOuterX1] =
      get_bc_func().at("periodic_outer");
  options->bcnames()[BoundaryFace::kInnerX1] = "periodic_inner";
  options->bcnames()[BoundaryFace::kOuterX1] = "periodic_outer";
}

std::shared_ptr<MeshBlockImpl> make_periodic_block(int nx1) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->coord()->global_nx1() = nx1;
  options->coord()->nx1() = nx1;
  options->coord()->global_x1max() = 2. * M_PI;
  options->coord()->x1max() = 2. * M_PI;
  options->hydro()->diffusion()->kappa_iso() = 0.;
  make_x1_periodic(options);
  return std::make_shared<MeshBlockImpl>(options);
}

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block,
                             torch::Device device, torch::Dtype dtype) {
  auto coord = block->pcoord;
  auto w = torch::ones(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IPR] = 1.e5;
  w.narrow(0, IVX, 3).zero_();
  return w;
}

void fill_periodic_x1(torch::Tensor const& var, int nghost) {
  BoundaryFuncOptions options;
  options.type(kPrimitive).nghost(nghost);
  get_bc_func().at("periodic_inner")(var, 3, options);
  get_bc_func().at("periodic_outer")(var, 3, options);
}

void fill_reflecting_x1(torch::Tensor const& var, int nghost) {
  BoundaryFuncOptions options;
  options.type(kPrimitive).nghost(nghost);
  get_bc_func().at("reflecting_inner")(var, 3, options);
  get_bc_func().at("reflecting_outer")(var, 3, options);
}

//! Density linear in x1 with slope kRhoSlope, then reflected so that the x1
//! ghosts hold the MIRROR of the first active cell instead of the value the
//! linear profile has there -- exactly what `reflecting` installs in a
//! stratified atmosphere. Temperature is linear over ALL cells, ghosts
//! included, so dT/dn is the same at every face and the conductive flux is
//! proportional to the face density alone.
constexpr double kRho0 = 1.0, kRhoSlope = 0.2, kTemp0 = 300.0,
                 kTempSlope = 10.0;

torch::Tensor make_linear_state(std::shared_ptr<MeshBlockImpl> const& block,
                                torch::Device device, torch::Dtype dtype,
                                torch::Tensor* temp) {
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  *temp = kTemp0 + kTempSlope * x;
  w[IDN] = kRho0 + kRhoSlope * x;
  fill_reflecting_x1(w, coord->options->nghost());
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * (*temp);
  return w;
}

}  // namespace

TEST(diffusion_options, parse_and_reject_legacy_keys) {
  auto options = DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {nu_iso: 2.0, kappa_iso: 3.0}"));
  ASSERT_TRUE(options);
  EXPECT_DOUBLE_EQ(options->nu_iso(), 2.);
  EXPECT_DOUBLE_EQ(options->kappa_iso(), 3.);

  EXPECT_ANY_THROW(DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {K: 2.0, type: theta}")));
  EXPECT_ANY_THROW(
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {nu_iso: -1.0}")));
}

TEST(diffusion_options, dynamic_rejects_non_bool) {
  for (char const* bad : {"1", "maybe", "yes", "on", "~", "[]"}) {
    auto yaml = std::string("diffusion: {dynamic: ") + bad + "}";
    EXPECT_THROW(DiffusionOptionsImpl::from_yaml(YAML::Load(yaml)), c10::Error)
        << bad;
  }
  auto off = DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {dynamic: false}"));
  EXPECT_FALSE(off->dynamic());
  auto bare = DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {}"));
  EXPECT_FALSE(bare->dynamic());
  EXPECT_DOUBLE_EQ(bare->nu_iso(), 0.);
  EXPECT_DOUBLE_EQ(bare->kappa_iso(), 0.);
  auto on =
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {dynamic: true}"));
  EXPECT_TRUE(on->dynamic());
}

TEST(diffusion_options, coefficients_reject_negative_or_non_numeric) {
  for (char const* key : {"nu_iso", "kappa_iso"}) {
    for (char const* bad : {"banana", "maybe", "~", "[]", "-1"}) {
      auto yaml = std::string("diffusion: {") + key + ": " + bad + "}";
      EXPECT_THROW(DiffusionOptionsImpl::from_yaml(YAML::Load(yaml)),
                   c10::Error)
          << key << " " << bad;
    }
    auto yaml = std::string("diffusion: {") + key + ": 0.5}";
    auto options = DiffusionOptionsImpl::from_yaml(YAML::Load(yaml));
    ASSERT_TRUE(options);
    if (std::string(key) == "nu_iso") {
      EXPECT_DOUBLE_EQ(options->nu_iso(), 0.5);
    } else {
      EXPECT_DOUBLE_EQ(options->kappa_iso(), 0.5);
    }
    auto zero_yaml = std::string("diffusion: {") + key + ": 0}";
    auto zero = DiffusionOptionsImpl::from_yaml(YAML::Load(zero_yaml));
    ASSERT_TRUE(zero);
    if (std::string(key) == "nu_iso") {
      EXPECT_DOUBLE_EQ(zero->nu_iso(), 0.);
    } else {
      EXPECT_DOUBLE_EQ(zero->kappa_iso(), 0.);
    }
  }
}

TEST(diffusion_options, reject_enabled_curved_coordinates) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->coord()->type() = "spherical-polar";
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));

  options->hydro()->diffusion()->nu_iso() = 0.;
  options->hydro()->diffusion()->kappa_iso() = 0.;
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST(diffusion_options, reject_conduction_without_reference_specific_heat) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->hydro()->eos()->type() = "shallow-water";
  options->hydro()->diffusion()->nu_iso() = 0.;
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST(diffusion_options, cartesian_temperature_requires_reflecting_x1) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->hydro()->diffusion()->temperature_inner(300.);
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options));
  options->bcnames()[BoundaryFace::kInnerX1] = "reflecting_custom";
  EXPECT_THROW(std::make_shared<MeshBlockImpl>(options), c10::Error);
  make_x1_periodic(options);
  EXPECT_THROW(std::make_shared<MeshBlockImpl>(options), c10::Error);
}

TEST_P(DeviceTest, uniform_state_has_zero_tendency) {
  auto block = make_block();
  block->to(device, dtype);
  EXPECT_TRUE(
      torch::allclose(block->pcoord->center_distance1(),
                      torch::ones_like(block->pcoord->center_distance1())));
  auto w = make_primitive(block, device, dtype);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  EXPECT_TRUE(torch::allclose(du, torch::zeros_like(du), 1.e-6, 1.e-6));
}

TEST_P(DeviceTest, cartesian_fixed_temperature_preserves_linear_conduction) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->hydro()
      ->diffusion()
      ->dynamic(true)
      .nu_iso(0.)
      .kappa_iso(2.)
      .temperature_inner(300.)
      .temperature_outer(360.);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto c = block->pcoord;
  auto temperature = 300. + 10. * c->x1v.to(device, dtype).view({1, 1, -1});
  double gas = block->phydro->peos->species_cv_ref() *
               (block->phydro->peos->options->gammad() - 1.);
  w[IPR].copy_(w[IDN] * gas * temperature);
  fill_reflecting_x1(w, c->options->nghost());
  temperature = block->phydro->peos->compute("W->T", {w});
  auto du = block->phydro->pdiffusion->forward(torch::zeros_like(w), w,
                                               temperature, 1.);
  auto sl = block->part({0, 0, 0}, PartOptions().exterior(false));
  EXPECT_LT(du.index(sl).abs().max().item<double>(),
            dtype == torch::kFloat64 ? 1.e-9 : 1.e-3);
  // Raising only the lower wall injects K * deltaT / (dz/2) into the layer.
  block->phydro->pdiffusion->options->temperature_inner(305.);
  du = block->phydro->pdiffusion->forward(torch::zeros_like(w), w, temperature,
                                          1.);
  auto spatial = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto heat = (du[IPR] * c->cell_volume()).index(spatial).sum().item<double>();
  EXPECT_NEAR(heat, 20., dtype == torch::kFloat64 ? 1.e-9 : 1.e-3);
}

TEST_P(DeviceTest, transverse_velocity_uses_viscous_laplacian) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto x = block->pcoord->x1v.to(device, dtype);
  w[IVY] = x.square();
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  auto expected = torch::zeros_like(du[IVY]);
  expected.index(block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3)))
      .fill_(0.1);
  EXPECT_TRUE(torch::allclose(du[IVY], expected, 1.e-5, 1.e-5));
}

// A.rst 3.7 on a Cartesian grid as on the cubed sphere: post_fluxes puts the
// energy rows of the face fluxes into the register as "diffusion" and add()
// keeps the others. Rows IDN..IVZ of du are the same bits; the energy row, now
// the register's divergence, differs from the old one only by rounding.
TEST_P(DeviceTest, cartesian_energy_post_reproduces_the_energy_row) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto x = block->pcoord->x1v.to(device, dtype).view({1, 1, -1});
  w[IVY] = x.square();
  auto temp = (300. + x.square()).expand_as(w[IDN]).contiguous();
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * temp;
  auto diffusion = block->phydro->pdiffusion;
  auto old = diffusion->forward(torch::zeros_like(w), w, temp, 1.);

  auto& reg = block->phydro->flux_register();
  for (int d = 1; d <= 3; ++d)
    if (reg.total(d).defined()) reg.total(d).zero_();
  diffusion->post_fluxes(reg, {w, temp, 1.});
  auto du = diffusion->forward(torch::zeros_like(w), w, temp, 1.);
  EXPECT_TRUE(torch::equal(du.narrow(0, 0, IPR), old.narrow(0, 0, IPR)));
  EXPECT_EQ(du[IPR].abs().max().item<double>(), 0.);  // not in add()

  auto sl = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto e =
      -block->pcoord->divergence(reg.total(1), reg.total(2), reg.total(3))[IPR];
  double worst = (e - old[IPR]).index(sl).abs().max().item<double>();
  double scale = old[IPR].index(sl).abs().max().item<double>();
  ASSERT_GT(scale, 0.);
  EXPECT_LE(worst, (dtype == torch::kFloat64 ? 1.e-13 : 1.e-5) * scale);
}

// The hydro rewrites the register total of a direction each stage only where
// its flux is on; a disabled direction's total is never rewritten, so a
// diffusion post there would add one more energy flux every stage. With
// disable-flux-x1 or -x2 the disabled total must not grow with the number of
// stages at a frozen state (add() then keeps the energy rows).
TEST_P(DeviceTest, disabled_direction_total_does_not_accumulate) {
  for (int off : {1, 2}) {
    auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_2d.yaml");
    if (off == 1) options->hydro()->disable_flux_x1(true);
    if (off == 2) options->hydro()->disable_flux_x2(true);
    auto block = std::make_shared<MeshBlockImpl>(options);
    block->to(device, dtype);
    auto w = make_primitive(block, device, dtype);
    auto x = block->pcoord->x1v.to(device, dtype).view({1, 1, -1});
    auto y = block->pcoord->x2v.to(device, dtype).view({1, -1, 1});
    auto temp = (300. + 5. * x + 3. * y).expand_as(w[IDN]).contiguous();
    auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
    w[IPR] = w[IDN] * Rd * temp;
    Variables vars;
    vars["hydro_w"] = w;
    block->initialize(vars);
    auto u0 = vars["hydro_u"].clone();
    int nstage = block->pintg->stages.size();
    auto& reg = block->phydro->flux_register();
    std::vector<double> energy;
    for (int n = 1; n <= 24; ++n) {
      vars["hydro_u"].copy_(u0);  // frozen state
      block->forward(vars, 1.e-4, (n - 1) % nstage);
      if (n == 3 || n == 6 || n == 12 || n == 24)
        energy.push_back(reg.total(off)[IPR].abs().max().item<double>());
    }
    std::printf(
        "disable-flux-x%d: max|total(%d)[IPR]| after 3/6/12/24 "
        "stages = %.6g %.6g %.6g %.6g\n",
        off, off, energy[0], energy[1], energy[2], energy[3]);
    EXPECT_LE(energy[3], energy[0] * (1. + 1.e-6) + 1.e-12)
        << "disable-flux-x" << off;
  }
}

// With a hydro flux direction off, the diffusion energy flux still goes
// through its owner, the register: over one step the disabled direction's
// total holds the diffusion energy row (and no other row), and its step field,
// the boundary transfer the budget reads, carries it.
TEST_P(DeviceTest, disabled_direction_register_carries_the_diffusion_energy) {
  for (int off : {1, 2}) {
    auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_2d.yaml");
    if (off == 1) options->hydro()->disable_flux_x1(true);
    if (off == 2) options->hydro()->disable_flux_x2(true);
    auto block = std::make_shared<MeshBlockImpl>(options);
    block->to(device, dtype);
    auto w = make_primitive(block, device, dtype);
    auto x = block->pcoord->x1v.to(device, dtype).view({1, 1, -1});
    auto y = block->pcoord->x2v.to(device, dtype).view({1, -1, 1});
    auto temp = (300. + 5. * x + 3. * y).expand_as(w[IDN]).contiguous();
    auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
    w[IPR] = w[IDN] * Rd * temp;
    Variables vars;
    vars["hydro_w"] = w;
    block->initialize(vars);
    auto& reg = block->phydro->flux_register();
    reg.request_step_field(off);
    for (int s = 0; s < static_cast<int>(block->pintg->stages.size()); ++s)
      block->forward(vars, 1.e-4, s);
    auto t = reg.total(off).to(torch::kFloat64);
    double energy = t[IPR].abs().max().item<double>();
    double others = t.narrow(0, 0, IPR).abs().max().item<double>();
    double transfer = reg.step_field(off)[IPR].abs().max().item<double>();
    std::printf(
        "disable-flux-x%d: max|total(%d)[IPR]| = %.6g, other rows %.6g, "
        "max|step field(%d)[IPR]| = %.6g\n",
        off, off, energy, others, off, transfer);
    EXPECT_GT(energy, 0.) << "disable-flux-x" << off;
    EXPECT_EQ(others, 0.) << "disable-flux-x" << off;
    EXPECT_GT(transfer, 0.) << "disable-flux-x" << off;
  }
}

TEST_P(DeviceTest, temperature_uses_conductive_laplacian) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto x = block->pcoord->x1v.to(device, dtype);
  auto temp = x.square().view({1, 1, -1});
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * temp;
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  auto cv_ref = block->phydro->peos->species_cv_ref();
  EXPECT_NEAR(du[IPR][0][0][4].item<double>(), 0.05 * cv_ref, 1.e-3);
  EXPECT_TRUE(torch::allclose(du[IDN], torch::zeros_like(du[IDN])));
  EXPECT_TRUE(torch::allclose(du.narrow(0, IVX, 3),
                              torch::zeros_like(du.narrow(0, IVX, 3))));
}

TEST_P(DeviceTest, viscous_sine_mode_matches_analytic_decay) {
  constexpr int nx1 = 64;
  constexpr int nsteps = 100;
  auto block = make_periodic_block(nx1);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = make_primitive(block, device, dtype);
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  w[IVY] = torch::sin(x);
  fill_periodic_x1(w, coord->options->nghost());

  auto nu = block->phydro->pdiffusion->options->nu_iso();
  auto dx = (coord->options->x1max() - coord->options->x1min()) / nx1;
  auto dt = 0.1 * dx * dx / nu;
  for (int n = 0; n < nsteps; ++n) {
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    block->phydro->pdiffusion->forward(du, w, temp, dt);
    w[IVY] += du[IVY];
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto time = nsteps * dt;
  auto expected = torch::sin(x) * std::exp(-nu * time);
  EXPECT_TRUE(torch::allclose(w[IVY].index(interior), expected.index(interior),
                              3.e-4, 3.e-4));
}

// On a linear density profile the two-cell average is exact on
// every INTERIOR face, so a uniform dT/dn gives the same tendency in every
// interior cell -- unless the wall face reads the ghost, which the reflecting
// fill has put off the profile. Reading the ghost halves the tendency in the
// two wall cells while leaving the interior right.
TEST_P(DeviceTest, wall_face_coefficient_reads_no_ghost) {
  auto block = make_block();
  block->to(device, dtype);
  torch::Tensor temp;
  auto w = make_linear_state(block, device, dtype, &temp);
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto got = du[IPR].index(interior);
  auto cv_ref = block->phydro->peos->species_cv_ref();
  auto expected = 0.1 * block->phydro->pdiffusion->options->kappa_iso() *
                  cv_ref * kTempSlope * kRhoSlope;
  EXPECT_TRUE(
      torch::allclose(got, torch::full_like(got, expected), 1.e-5, 1.e-5))
      << "du[IPR] over the interior: " << got;
}

// x1 ONLY, deliberately. The same construction along x2 must therefore give the
// two-cell-average answer, not the extrapolated one: at a lateral `reflecting`
// face the mirror may be a genuine symmetry, where the average is the better
// estimate. This pins the decision rather than leaving it to inspection.
TEST_P(DeviceTest, x2_wall_is_not_one_sided) {
  auto block = std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml("test_diffusion_2d.yaml"));
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  auto y = coord->x2v.to(device, dtype).view({1, -1, 1});
  auto temp = (kTemp0 + kTempSlope * y).expand_as(w[IDN]).contiguous();
  w[IDN] = kRho0 + kRhoSlope * y;
  {
    BoundaryFuncOptions bops;
    bops.type(kPrimitive).nghost(coord->options->nghost());
    get_bc_func().at("reflecting_inner")(w, 2, bops);
    get_bc_func().at("reflecting_outer")(w, 2, bops);
  }
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * temp;

  auto du = torch::zeros_like(w);
  block->phydro->pdiffusion->forward(du, w, temp, 0.1);

  int ng = coord->options->nghost();
  auto cv_ref = block->phydro->peos->species_cv_ref();
  auto full = 0.1 * block->phydro->pdiffusion->options->kappa_iso() * cv_ref *
              kTempSlope * kRhoSlope;
  // mirrored ghost, two-cell average => the wall row sits half a slope short
  EXPECT_NEAR(du[IPR][0][ng][ng].item<double>(), 0.5 * full, 1.e-4 * full);
}

// The same state under a periodic x1: the ghost is then the true wrapped
// neighbour, so the two-cell average is correct and the wall extrapolation
// must NOT fire. The deliberately off-profile ghost makes the two answers
// differ, so this discriminates.
TEST_P(DeviceTest, periodic_x1_face_is_not_extrapolated) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  make_x1_periodic(options);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);
  torch::Tensor temp;
  auto w = make_linear_state(block, device, dtype, &temp);
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);

  int nghost = block->pcoord->options->nghost();
  auto cv_ref = block->phydro->peos->species_cv_ref();
  auto full = 0.1 * block->phydro->pdiffusion->options->kappa_iso() * cv_ref *
              kTempSlope * kRhoSlope;
  // mirrored ghost => the wrap face average sits half a slope short
  EXPECT_NEAR(du[IPR][0][0][nghost].item<double>(), 0.5 * full, 1.e-4 * full);
}

// reflecting puts a physical state at the wrong place and fixed_temperature
// engineers a state that is none; the wall branch may extrapolate past both.
// Everything else -- including a caller-supplied function whose name was never
// recorded -- must keep the two-cell average.
TEST(diffusion_options, wall_names_are_an_exact_whitelist) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  EXPECT_TRUE(options->is_wall_boundary(0, 0, -1));
  EXPECT_TRUE(options->is_wall_boundary(0, 0, 1));

  for (std::string name : {"periodic", "outflow", "custom"}) {
    options->bfuncs()[BoundaryFace::kInnerX1] =
        get_bc_func().at(name + "_inner");
    options->bcnames()[BoundaryFace::kInnerX1] = name + "_inner";
    EXPECT_FALSE(options->is_wall_boundary(0, 0, -1)) << name;
    EXPECT_TRUE(options->is_physical_boundary(0, 0, -1)) << name;
  }

  // `solid` writes a bare 1 into every variable, so its GRADIENT is nonsense
  // too and a coefficient fix would not rescue the face
  options->bfuncs()[BoundaryFace::kInnerX1] = get_bc_func().at("solid_inner");
  options->bcnames()[BoundaryFace::kInnerX1] = "solid_inner";
  EXPECT_FALSE(options->is_wall_boundary(0, 0, -1));

  // an unnamed function, e.g. one installed from Python, is not a wall
  options->bcnames()[BoundaryFace::kInnerX1] = "";
  EXPECT_FALSE(options->is_wall_boundary(0, 0, -1));

  // a name that merely BEGINS with the whitelisted one is not it either
  options->bfuncs()[BoundaryFace::kInnerX1] =
      get_bc_func().at("reflecting_inner");
  options->bcnames()[BoundaryFace::kInnerX1] = "reflecting_sponge_inner";
  EXPECT_FALSE(options->is_wall_boundary(0, 0, -1));

  // an engineered fixed-temperature wall is a wall
  options->bcnames()[BoundaryFace::kInnerX1] = "fixed_temperature_inner";
  EXPECT_TRUE(options->is_wall_boundary(0, 0, -1));

  // and neither is anything once the two records disagree in length
  options->bcnames().clear();
  EXPECT_FALSE(options->is_wall_boundary(0, 0, 1));
}

// The property the wall branch asserts is that the coefficient
// reads no ghost, so the wall flux must not depend on the ghost DENSITY at all.
// Two states differing ONLY in that ghost must give the same tendency.
//
// This is the only gate here that reaches the VISCOUS wall coefficient, which
// is the branch that is live in production: at a reflecting wall the mirror
// makes dT/dn vanish, so the conductive coefficient multiplies zero and a
// conduction-only gate can be satisfied by code that still reads the ghost for
// the momentum flux. The normal velocity is mirrored ODD here, so the wall
// carries a real stress and rho_face genuinely matters.
TEST_P(DeviceTest, wall_flux_does_not_depend_on_the_ghost_density) {
  auto arm = [&](bool ghost_on_profile) {
    auto block = make_block();
    block->to(device, dtype);
    auto coord = block->pcoord;
    int ng = coord->options->nghost();
    int nc1 = coord->options->nc1();
    auto w =
        torch::zeros({5, coord->options->nc3(), coord->options->nc2(), nc1},
                     torch::device(device).dtype(dtype));
    auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
    auto profile = kRho0 + kRhoSlope * x;
    auto temp = kTemp0 + kTempSlope * x;

    w[IDN] = profile;
    w[IVX] = 1.0;               // reflected ODD below: a live wall stress
    fill_reflecting_x1(w, ng);  // rho ghost becomes the mirror
    if (ghost_on_profile) {     // ... or the linear continuation instead
      w[IDN].narrow(-1, 0, ng).copy_(profile.narrow(-1, 0, ng));
      w[IDN].narrow(-1, nc1 - ng, ng).copy_(profile.narrow(-1, nc1 - ng, ng));
    }
    // pressure from the PROFILE in both arms, so the ghost density is the only
    // difference between them
    auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
    w[IPR] = profile * Rd * temp;

    auto du = torch::zeros_like(w);
    block->phydro->pdiffusion->forward(du, w, temp, 0.1);
    return du;
  };

  auto mirrored = arm(false);
  auto continued = arm(true);
  auto block = make_block();
  int ng = block->pcoord->options->nghost();
  // The quantity under test must be NONZERO, or this gate degenerates exactly
  // the way the conduction-only gates did: equality of two arms is satisfied
  // trivially by anything that makes the wall flux vanish.
  EXPECT_GT(std::abs(mirrored[IVX][0][0][ng].item<double>()), 1.e-6)
      << "the wall viscous tendency is zero, so this gate proves nothing";
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  for (int n = 0; n < 5; ++n) {
    auto a = mirrored[n].index(interior);
    auto b = continued[n].index(interior);
    EXPECT_TRUE(torch::allclose(a, b, 1.e-10, 1.e-10))
        << "du[" << n << "] moved with the ghost density: " << a - b;
  }
}

TEST_P(DeviceTest, timestep_uses_largest_diffusivity) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step({w}), 1., 1.e-12);

  w[IPR] = 1.e-6;
  EXPECT_NEAR(block->phydro->max_time_step(w), 1., 1.e-6);
}

TEST(diffusion, timestep_rejects_a_non_positive_bound) {
  auto block = make_block();
  auto w = make_primitive(block, torch::kCPU, torch::kFloat64);
  w[IDN].fill_(-1.);
  block->phydro->pdiffusion->options->dynamic(true);
  EXPECT_THROW(block->phydro->pdiffusion->max_time_step({w}), c10::Error);
}

// std::min(dt, NaN) returns dt: a NaN diffusivity used to drop the bound
TEST_P(DeviceTest, timestep_rejects_a_non_finite_diffusivity) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  block->phydro->pdiffusion->options->nu_iso(NAN);
  EXPECT_ANY_THROW(block->phydro->max_time_step(w));
}

// std::max(finite nu, NaN kappa) returns nu, and a zero nu returned early: a
// NaN conductivity passed, in both modes. Each coefficient is checked alone.
TEST_P(DeviceTest, timestep_rejects_each_non_finite_coefficient) {
  struct Case {
    double nu, kappa;
    bool dynamic;
  };
  for (auto c :
       {Case{0.5, NAN, false}, Case{0.5, NAN, true}, Case{0., NAN, false},
        Case{0., NAN, true}, Case{NAN, 0.25, true}}) {
    SCOPED_TRACE("nu=" + std::to_string(c.nu) +
                 " kappa=" + std::to_string(c.kappa) +
                 " dynamic=" + std::to_string(c.dynamic));
    auto block = make_block();
    block->to(device, dtype);
    auto w = make_primitive(block, device, dtype);
    auto opts = block->phydro->pdiffusion->options;
    opts->dynamic(c.dynamic);
    opts->nu_iso(c.nu);
    opts->kappa_iso(c.kappa);
    EXPECT_ANY_THROW(block->phydro->max_time_step(w));
  }
}

// dynamic: true reads nu_iso as mu and kappa_iso as k: on rho = 1 + x/10 the
// tendencies are dt*mu*d2(vy)/dx2 and dt*k*d2T/dx2 in every cell, with no rho
// (or rho*cv) at the faces or the centres; kinematic gives dt*nu*(rho v')'
TEST(diffusion, dynamic_coefficients_carry_no_density) {
  auto block = make_block();
  auto w = make_primitive(block, torch::kCPU, torch::kFloat64);
  auto x = block->pcoord->x1v.to(torch::kFloat64).view({1, 1, -1});
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  auto peos = block->phydro->peos;
  auto opts = block->phydro->pdiffusion->options;
  w[IDN] = 1. + 0.1 * x;
  auto run = [&](bool dynamic, bool heat, torch::Tensor* cv = nullptr) {
    opts->dynamic(dynamic);
    auto v = w.clone();
    auto temp = (300. + (heat ? x.square() : 0. * x)).expand_as(v[IDN]);
    v[IPR] = v[IDN] * Rd * temp;
    if (!heat) v[IVY] = x.square();
    auto du = torch::zeros_like(v);
    block->phydro->pdiffusion->forward(du, v, temp, 0.1);
    if (cv) *cv = peos->specific_heat_cv(v, temp).index(interior);
    return du[heat ? IPR : IVY].index(interior);
  };
  auto xi = x.expand_as(w[IDN]).index(interior);
  torch::Tensor cv;
  auto kin_heat = run(false, true, &cv);
  EXPECT_TRUE(
      torch::allclose(run(false, false), 0.05 * (2. + 0.4 * xi), 1.e-12, 0.));
  EXPECT_TRUE(
      torch::allclose(kin_heat, 0.025 * cv * (2. + 0.4 * xi), 1.e-12, 0.));
  auto shear = run(true, false), heat = run(true, true);
  EXPECT_TRUE(torch::allclose(shear, torch::full_like(shear, 0.5 * 2. * 0.1),
                              1.e-12, 0.))
      << "dynamic viscous tendency " << shear;
  EXPECT_TRUE(torch::allclose(heat, torch::full_like(heat, 0.25 * 2. * 0.1),
                              1.e-12, 0.))
      << "dynamic conductive tendency " << heat;
  // max_time_step bounds by mu/rho_min and k/(rho_min*cv_ref); dx = 1, one dim
  double rho_min = 1.05,
         coeff =
             std::max(0.5 / rho_min, 0.25 / (rho_min * peos->species_cv_ref()));
  opts->dynamic(true);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step({w}), 1. / (2. * coeff),
              1.e-12);
}
