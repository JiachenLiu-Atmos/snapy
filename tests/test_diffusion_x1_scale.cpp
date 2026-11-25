// C/C++
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/bc/bc_func.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"
#include "diffusion_energy.hpp"

using namespace snap;

// x1 profiles of the kinematic diffusion coefficients (nu_scale_x1,
// kappa_scale_x1): a tensor over the block's x1 cell centres, or a YAML table
// {x1: [...], scale: [...]} interpolated onto them.

namespace {

std::string table_yaml(char const* key, std::vector<double> const& x1,
                       std::vector<double> const& scale) {
  std::ostringstream os;
  os << std::setprecision(17) << key << ": {x1: [";
  for (size_t i = 0; i < x1.size(); ++i) os << (i ? ", " : "") << x1[i];
  os << "], scale: [";
  for (size_t i = 0; i < scale.size(); ++i) os << (i ? ", " : "") << scale[i];
  os << "]}";
  return os.str();
}

DiffusionOptions parse(std::string const& body) {
  return DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {nu_iso: 0.5, kappa_iso: 0.25, " + body + "}"));
}

MeshBlockOptions base_options() {
  return MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
}

std::shared_ptr<MeshBlockImpl> build(MeshBlockOptions const& options) {
  return std::make_shared<MeshBlockImpl>(options);
}

void make_x1_periodic(MeshBlockOptions const& options, int nx1) {
  options->coord()->global_nx1() = nx1;
  options->coord()->nx1() = nx1;
  options->coord()->global_x1max() = 2. * M_PI;
  options->coord()->x1max() = 2. * M_PI;
  options->bfuncs()[BoundaryFace::kInnerX1] =
      get_bc_func().at("periodic_inner");
  options->bfuncs()[BoundaryFace::kOuterX1] =
      get_bc_func().at("periodic_outer");
  options->bcnames()[BoundaryFace::kInnerX1] = "periodic_inner";
  options->bcnames()[BoundaryFace::kOuterX1] = "periodic_outer";
}

void fill_periodic_x1(torch::Tensor const& var, int nghost) {
  BoundaryFuncOptions options;
  options.type(kPrimitive).nghost(nghost);
  get_bc_func().at("periodic_inner")(var, 3, options);
  get_bc_func().at("periodic_outer")(var, 3, options);
}

//! rho linear, v2 = x^2, T = 300 + x^2: every term of both fluxes is non-zero
torch::Tensor sheared_state(std::shared_ptr<MeshBlockImpl> const& block,
                            torch::Device device, torch::Dtype dtype,
                            torch::Tensor* temp) {
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  *temp = (300. + x.square()).expand_as(w[IDN]).clone();
  w[IDN] = 1. + 0.1 * x;
  w[IVY] = x.square();
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * (*temp);
  return w;
}

torch::Tensor tendency(std::shared_ptr<MeshBlockImpl> const& block,
                       torch::Tensor const& w, torch::Tensor const& temp) {
  auto du = torch::zeros_like(w);
  diffuse(*block, du, w, temp, 0.1);
  return du;
}

std::vector<double> centres(MeshBlockOptions const& options) {
  auto x = build(options)->pcoord->x1v.to(torch::kFloat64).contiguous();
  return std::vector<double>(x.data_ptr<double>(),
                             x.data_ptr<double>() + x.numel());
}

DiffusionOptions coefficients() {
  auto op = DiffusionOptionsImpl::create();
  op->nu_iso(0.5);
  op->kappa_iso(0.25);
  return op;
}

//! test_diffusion.yaml's column at 16 cells, as nb1 blocks along x1 in one
//! process (cubed layout; the slab layout rejects nb1 > 1)
Mesh split_column(int nb1, DiffusionOptions const& diffusion) {
  auto block = base_options();
  block->coord()->global_nx1() = 16;
  block->coord()->nx1() = 16;
  block->coord()->global_x1max() = 16.;
  block->coord()->x1max() = 16.;
  block->hydro()->diffusion() = diffusion;
  if (nb1 > 1) {
    block->layout()->type() = "cubed";
    block->layout()->pz(nb1);
  }
  auto options = MeshOptionsImpl::create();
  options->block(block);
  options->blocks_per_process(nb1);
  return Mesh(options);
}

//! the sheared state's tendency on every block, interiors joined along x1
torch::Tensor column_tendency(Mesh const& mesh, torch::Device device,
                              torch::Dtype dtype) {
  std::vector<torch::Tensor> parts;
  for (auto const& block : mesh->blocks) {
    torch::Tensor temp;
    auto w = sheared_state(block.ptr(), device, dtype, &temp);
    auto il = block->pcoord->il(), iu = block->pcoord->iu();
    parts.push_back(tendency(block.ptr(), w, temp).slice(-1, il, iu + 1));
  }
  return torch::cat(parts, -1);
}

//! a density falling ~10x over x1 in [0, 1], fastest at the top
torch::Tensor stratified_rho(torch::Tensor const& x) {
  return torch::pow(1.25 - x, 1.5);
}

//! test_diffusion.yaml's column (reflecting x1 walls) on nx1 cells over
//! [0, 1], both kinematic coefficients scaled by `scale` of the x1 centres;
//! on a CUDA device it is moved there in double precision
std::shared_ptr<MeshBlockImpl> scaled_column(
    int nx1, std::function<torch::Tensor(torch::Tensor const&)> const& scale,
    torch::Device device = torch::kCPU) {
  auto options = base_options();
  options->coord()->global_nx1() = nx1;
  options->coord()->nx1() = nx1;
  options->coord()->global_x1max() = 1.;
  options->coord()->x1max() = 1.;
  auto s = scale(build(options)->pcoord->x1v.to(torch::kFloat64));
  options->hydro()->diffusion()->nu_scale_x1(s);
  options->hydro()->diffusion()->kappa_scale_x1(s.clone());
  auto block = build(options);
  if (device.is_cuda()) block->to(device, torch::kFloat64);
  return block;
}

//! interior tendency (dt = 1) of the stratified column, at rest with
//! temperature `field` (heat) or at 300 K with v2 = `field` (shear); cv out;
//! the tendency is returned on the CPU
torch::Tensor stratified_tendency(std::shared_ptr<MeshBlockImpl> const& block,
                                  torch::Tensor const& field, bool heat,
                                  double* cv) {
  auto coord = block->pcoord;
  auto peos = block->phydro->peos;
  auto x = coord->x1v.to(torch::kFloat64).view({1, 1, -1});
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      x.options());
  w[IDN] = stratified_rho(x);
  auto temp = (heat ? field : 300. + 0. * x).expand_as(w[IDN]).clone();
  if (!heat) w[IVY] = field;
  auto Rd = 8.31446261815324 / peos->options->weight();
  w[IPR] = w[IDN] * Rd * temp;
  *cv = peos->specific_heat_cv(w, temp).max().item<double>();
  auto du = torch::zeros_like(w);
  diffuse(*block, du, w, temp, 1.);
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  return du[heat ? IPR : IVY].index(interior).reshape(-1).cpu();
}

}  // namespace

TEST(diffusion_x1_scale, yaml_table_parses) {
  auto op = parse(table_yaml("nu_scale_x1", {0., 3., 6.}, {1., 2., 4.}));
  ASSERT_TRUE(op->nu_scale_x1_table().defined());
  EXPECT_FALSE(op->kappa_scale_x1_table().defined());
  EXPECT_FALSE(op->nu_scale_x1().defined());
  EXPECT_TRUE(torch::equal(
      op->nu_scale_x1_table(),
      torch::tensor({0., 3., 6., 1., 2., 4.}, torch::kFloat64).view({2, 3})));
}

TEST(diffusion_x1_scale, yaml_table_rejects_bad_input) {
  for (char const* bad : {
           "nu_scale_x1: 2.0",                                // not a table
           "nu_scale_x1: {x1: [0., 6.]}",                     // no scale
           "nu_scale_x1: {x1: [0., 6.], scale: [1.]}",        // lengths
           "nu_scale_x1: {x1: [0.], scale: [1.]}",            // one point
           "nu_scale_x1: {x1: [6., 0.], scale: [1., 1.]}",    // decreasing
           "nu_scale_x1: {x1: [0., 0.], scale: [1., 1.]}",    // repeated
           "nu_scale_x1: {x1: [0., 6.], scale: [1., 0.]}",    // zero
           "nu_scale_x1: {x1: [0., 6.], scale: [1., -2.]}",   // negative
           "nu_scale_x1: {x1: [0., 6.], scale: [1., .nan]}",  // NaN
           "nu_scale_x1: {x1: [0., 6.], scale: [1., x]}",     // not a number
           "kappa_scale_x1: {x1: [0., 6.], scale: [1., 1.], z: [0.]}",
       }) {
    EXPECT_THROW(parse(bad), c10::Error) << bad;
  }
}

// a profile of ones takes the scaled path and must not change one bit
TEST_P(DeviceTest, unity_profile_is_bitwise_no_profile) {
  torch::Tensor temp;
  auto plain = build(base_options());
  plain->to(device, dtype);
  auto w = sheared_state(plain, device, dtype, &temp);
  auto expected = tendency(plain, w, temp);

  auto by_cells = base_options();
  auto nc1 = plain->pcoord->options->nc1();
  by_cells->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  by_cells->hydro()->diffusion()->kappa_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  auto block = build(by_cells);
  block->to(device, dtype);
  EXPECT_TRUE(torch::equal(tendency(block, w, temp), expected));

  auto by_table = base_options();
  by_table->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}) + ", " +
            table_yaml("kappa_scale_x1", {0., 6.}, {1., 1.}));
  block = build(by_table);
  block->to(device, dtype);
  EXPECT_TRUE(torch::equal(tendency(block, w, temp), expected));
}

// v2 = sin(x) on a periodic box, nu scaled by a uniform s: exp(-s nu k^2 t),
// k = 1. Same scheme and tolerance as test_diffusion's unscaled sine mode.
TEST_P(DeviceTest, viscous_sine_mode_decays_at_the_scaled_rate) {
  constexpr int nx1 = 64, nsteps = 100;
  constexpr double s = 2.5;
  auto options = base_options();
  make_x1_periodic(options, nx1);
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 2. * M_PI}, {s, s}));
  options->hydro()->diffusion()->kappa_iso(0.);
  auto block = build(options);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = torch::ones(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IPR] = 1.e5;
  w.narrow(0, IVX, 3).zero_();
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  w[IVY] = torch::sin(x);
  fill_periodic_x1(w, coord->options->nghost());

  auto nu = block->phydro->pdiffusion->options->nu_iso();
  auto dx = 2. * M_PI / nx1;
  auto dt = 0.1 * dx * dx / (s * nu);
  for (int n = 0; n < nsteps; ++n) {
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    diffuse(*block, du, w, temp, dt);
    w[IVY] += du[IVY];  // rho = 1
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto expected = torch::sin(x) * std::exp(-s * nu * nsteps * dt);
  EXPECT_TRUE(torch::allclose(w[IVY].index(interior), expected.index(interior),
                              3.e-4, 3.e-4));
}

// T = 300 + 100 sin(x) at rest, kappa scaled by a uniform s: the energy
// tendency is kappa s rho cv T'' and, at fixed rho, dT = dE / (rho cv), so the
// mode decays as exp(-s kappa k^2 t), k = 1.
TEST_P(DeviceTest, conductive_sine_mode_decays_at_the_scaled_rate) {
  constexpr int nx1 = 64, nsteps = 100;
  constexpr double s = 2.5, amp = 100.;
  auto options = base_options();
  make_x1_periodic(options, nx1);
  options->hydro()->diffusion() =
      parse(table_yaml("kappa_scale_x1", {0., 2. * M_PI}, {s, s}));
  options->hydro()->diffusion()->nu_iso(0.);
  auto block = build(options);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto peos = block->phydro->peos;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IDN] = 1.;
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  auto Rd = 8.31446261815324 / peos->options->weight();
  w[IPR] = Rd * (300. + amp * torch::sin(x));
  fill_periodic_x1(w, coord->options->nghost());

  auto kappa = block->phydro->pdiffusion->options->kappa_iso();
  auto dx = 2. * M_PI / nx1;
  auto dt = 0.1 * dx * dx / (s * kappa);
  for (int n = 0; n < nsteps; ++n) {
    auto temp = peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    diffuse(*block, du, w, temp, dt);
    // p = rho Rd T and dT = dE / (rho cv): dp = (Rd / cv) dE
    w[IPR] += du[IPR] * Rd / peos->specific_heat_cv(w, temp);
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto mode = (peos->compute("W->T", {w}) - 300.) / amp;
  auto expected = torch::sin(x) * std::exp(-s * kappa * nsteps * dt);
  EXPECT_TRUE(torch::allclose(mode.index(interior),
                              expected.expand_as(mode).index(interior), 3.e-4,
                              3.e-4));
}

// s(x) = 1 + x/10 at rho = 1 is the kinematic flux of rho = 1 + x/10 at s = 1:
// for v2 = x^2 (T uniform) and T = 300 + x^2 (at rest) the tendencies are
// dt nu (2 + 0.4 x) and dt kappa cv (2 + 0.4 x) in every cell, the wall cells
// included (the wall face extrapolates the linear product exactly), as
// test_diffusion's dynamic_coefficients_carry_no_density has for rho.
TEST(diffusion_x1_scale, linear_profile_gives_the_analytic_tendency) {
  auto options = base_options();
  auto x1v = centres(options);
  std::vector<double> ends = {1. + 0.1 * x1v.front(), 1. + 0.1 * x1v.back()};
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {x1v.front(), x1v.back()}, ends) + ", " +
            table_yaml("kappa_scale_x1", {x1v.front(), x1v.back()}, ends));
  auto block = build(options);
  auto peos = block->phydro->peos;
  auto x = block->pcoord->x1v.to(torch::kFloat64).view({1, 1, -1});
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto Rd = 8.31446261815324 / peos->options->weight();
  auto run = [&](bool heat, torch::Tensor* cv) {
    torch::Tensor unused;
    auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &unused);
    w[IDN] = 1.;
    auto temp = (300. + (heat ? x.square() : 0. * x)).expand_as(w[IDN]);
    w[IPR] = w[IDN] * Rd * temp;
    w[IVY] = heat ? 0. * x : x.square();
    if (cv) *cv = peos->specific_heat_cv(w, temp).index(interior);
    return tendency(block, w, temp)[heat ? IPR : IVY].index(interior);
  };
  auto shape = (2. + 0.4 * x).expand({1, 1, x.size(2)}).index(interior);
  torch::Tensor cv;
  auto heat = run(true, &cv);
  auto shear = run(false, nullptr);
  EXPECT_TRUE(torch::allclose(shear, 0.05 * shape, 1.e-12, 0.)) << shear;
  EXPECT_TRUE(torch::allclose(heat, 0.025 * cv * shape, 1.e-12, 0.)) << heat;
}

namespace {

// A constant DYNAMIC coefficient over a stratified column: s = 1/rho makes
// mu = nu s rho and k = kappa s rho cv uniform, so a linear T at rest and a
// linear v2 carry a uniform flux and no tendency in any cell. The face
// coefficient is the mean of the two cells' products s rho, exact here. The
// product of the means, mean(s) mean(rho) = 1 + (drho)^2 / (4 rho_a rho_b),
// gives an O(dx^2) tendency inside and an O(dx) one in the wall cells, whose
// extrapolated wall face is exact (docs/derivations/
// diffusion-face-coefficient.md).
void constant_dynamic_coefficient_column_has_no_tendency(torch::Device device) {
  for (int nx1 : {16, 64}) {
    auto block = scaled_column(
        nx1, [](torch::Tensor const& x) { return 1. / stratified_rho(x); },
        device);
    auto x = block->pcoord->x1v.to(torch::kFloat64).view({1, 1, -1});
    auto dx = 1. / nx1;
    double cv;
    auto heat = stratified_tendency(block, 300. + 50. * x, true, &cv);
    // |flux| / dx: kappa cv 50 / dx and nu 1 / dx
    auto heat_scale = 0.25 * cv * 50. / dx;
    auto shear = stratified_tendency(block, x, false, &cv);
    auto shear_scale = 0.5 / dx;
    EXPECT_LT(heat.abs().max().item<double>(), 1.e-12 * heat_scale)
        << "nx1 = " << nx1 << ": " << heat / heat_scale;
    EXPECT_LT(shear.abs().max().item<double>(), 1.e-12 * shear_scale)
        << "nx1 = " << nx1 << ": " << shear / shear_scale;
  }
}

}  // namespace

TEST(diffusion_x1_scale, constant_dynamic_coefficient_column_has_no_tendency) {
  constant_dynamic_coefficient_column_has_no_tendency(torch::kCPU);
}

TEST(diffusion_x1_scale,
     constant_dynamic_coefficient_column_has_no_tendency_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  constant_dynamic_coefficient_column_has_no_tendency(
      torch::Device(torch::kCUDA, 0));
}

namespace {

// A smooth variable coefficient, s = 1 + cos(3 x) / 2 on the stratified rho:
// the face fluxes, recovered from the tendency by summing up from the lower
// wall (F_{i+1/2} - F_{1/2} = -dx sum_{j <= i} du_j at dt = 1), converge to the
// exact flux at second order, every face measured, the wall faces included.
void smooth_profile_flux_converges_at_second_order(torch::Device device) {
  auto s_of = [](torch::Tensor const& x) {
    return 1. + 0.5 * torch::cos(3. * x);
  };
  double prev[2] = {0., 0.};
  for (int nx1 : {32, 64, 128}) {
    auto block = scaled_column(nx1, s_of, device);
    auto x = block->pcoord->x1v.to(torch::kFloat64).view({1, 1, -1});
    auto dx = 1. / nx1;
    auto xf = torch::arange(nx1 + 1, torch::kFloat64) * dx;
    auto c = s_of(xf) * stratified_rho(xf);
    double cv;
    auto heat =
        stratified_tendency(block, 300. + 50. * torch::sin(2. * x), true, &cv);
    auto shear = stratified_tendency(block, torch::sin(2. * x), false, &cv);
    // exact fluxes: -kappa cv c T' and -nu c v2'
    std::array<torch::Tensor, 2> exact = {
        -0.25 * cv * c * 100. * torch::cos(2. * xf),
        -0.5 * c * 2. * torch::cos(2. * xf)};
    std::array<torch::Tensor, 2> du = {heat, shear};
    for (int k = 0; k < 2; ++k) {
      auto got = -dx * torch::cumsum(du[k], 0);
      auto want = exact[k].slice(0, 1) - exact[k][0];
      auto err =
          ((got - want).abs().max() / exact[k].abs().max()).item<double>();
      if (prev[k] > 0.) {
        EXPECT_GT(std::log2(prev[k] / err), 1.85)
            << (k ? "shear" : "heat") << " nx1 = " << nx1 << ": " << prev[k]
            << " -> " << err;
      }
      prev[k] = err;
    }
  }
}

}  // namespace

TEST(diffusion_x1_scale, smooth_profile_flux_converges_at_second_order) {
  smooth_profile_flux_converges_at_second_order(torch::kCPU);
}

TEST(diffusion_x1_scale, smooth_profile_flux_converges_at_second_order_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  smooth_profile_flux_converges_at_second_order(torch::Device(torch::kCUDA, 0));
}

// YAML knots on the cell centres give the profile set as a tensor, bit for bit
TEST_P(DeviceTest, yaml_table_equals_the_cells_profile) {
  auto x1v = centres(base_options());
  std::vector<double> values(x1v.size());
  for (size_t i = 0; i < x1v.size(); ++i)
    values[i] = 1. + 0.5 * std::cos(x1v[i]);

  auto by_cells = base_options();
  auto cells = torch::tensor(values, torch::kFloat64);
  by_cells->hydro()->diffusion()->nu_scale_x1(cells);
  by_cells->hydro()->diffusion()->kappa_scale_x1(cells.flip(0));
  auto a = build(by_cells);
  a->to(device, dtype);

  std::vector<double> reversed(values.rbegin(), values.rend());
  auto by_table = base_options();
  by_table->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", x1v, values) + ", " +
            table_yaml("kappa_scale_x1", x1v, reversed));
  auto b = build(by_table);
  b->to(device, dtype);

  torch::Tensor temp;
  auto w = sheared_state(a, device, dtype, &temp);
  auto expected = tendency(a, w, temp);
  EXPECT_TRUE(torch::equal(tendency(b, w, temp), expected));
  auto plain = build(base_options());
  plain->to(device, dtype);
  EXPECT_FALSE(torch::equal(tendency(plain, w, temp), expected));
}

// dx = 1, one active dimension: dt = 1 / (2 max(nu max s_nu, kappa max s_k))
TEST(diffusion_x1_scale, timestep_uses_the_largest_scaled_coefficient) {
  auto w_of = [](std::shared_ptr<MeshBlockImpl> const& block) {
    torch::Tensor temp;
    return sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
  };
  auto options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 4.}));
  auto block = build(options);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step({w_of(block)}),
              1. / (2. * 0.5 * 4.), 1.e-12);

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("kappa_scale_x1", {0., 6.}, {1., 4.}));
  block = build(options);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step({w_of(block)}),
              1. / (2. * 0.25 * 4.), 1.e-12);
}

TEST(diffusion_x1_scale, reset_refuses_bad_profiles) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  auto expect_refused = [](MeshBlockOptions const& options, char const* why) {
    EXPECT_THROW(build(options), c10::Error) << why;
  };

  auto options = base_options();
  options->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1 + 1}, torch::kFloat64));
  expect_refused(options, "wrong length");

  options = base_options();
  auto bad = torch::ones({nc1}, torch::kFloat64);
  bad[0] = 0.;
  options->hydro()->diffusion()->kappa_scale_x1(bad);
  expect_refused(options, "non-positive");

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}));
  options->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  expect_refused(options, "table and tensor both set");

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}));
  options->hydro()->diffusion()->dynamic(true);
  expect_refused(options, "dynamic: true");

  // a profile set after the block is built never reached reset
  auto block = build(base_options());
  torch::Tensor temp;
  auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
  block->phydro->pdiffusion->options->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  EXPECT_THROW(tendency(block, w, temp), c10::Error);
}

// The table is interpolated onto each block's own cells, so a column split
// into nb1 = 2 or 4 blocks along x1 gets the tendency of the unsplit column.
TEST_P(DeviceTest, table_profile_is_independent_of_the_x1_split) {
  auto profile = [] {
    return parse(
        table_yaml("nu_scale_x1", {0., 5.3, 11., 16.}, {1., 3., 2., 5.}) +
        ", " + table_yaml("kappa_scale_x1", {2.2, 9.}, {0.5, 4.}));
  };
  auto one = split_column(1, profile());
  one->to(device, dtype);
  auto expected = column_tendency(one, device, dtype);
  ASSERT_EQ(expected.size(-1), 16);

  auto plain = split_column(1, coefficients());
  plain->to(device, dtype);
  EXPECT_FALSE(torch::equal(column_tendency(plain, device, dtype), expected));

  for (int nb1 : {2, 4}) {
    auto mesh = split_column(nb1, profile());
    mesh->to(device, dtype);
    ASSERT_EQ(static_cast<int>(mesh->blocks.size()), nb1);
    auto got = column_tendency(mesh, device, dtype);
    EXPECT_TRUE(torch::equal(got, expected))
        << "nb1 = " << nb1 << ": max |difference| "
        << (got - expected).abs().max().item<double>();
  }
}

// A per-cell tensor holds one block's cells, and every block of a mesh shares
// the options: with x1 split, each block would take the same block-sized
// profile. It is refused there; with one block along x1 it is accepted.
TEST(diffusion_x1_scale, cells_profile_is_refused_when_x1_is_split) {
  auto per_cell = [](char const* which, int64_t nc1) {
    auto op = coefficients();
    auto cells = torch::linspace(1., 2., nc1, torch::kFloat64);
    if (std::string(which) == "nu") {
      op->nu_scale_x1(cells);
    } else {
      op->kappa_scale_x1(cells);
    }
    return op;
  };
  for (char const* which : {"nu", "kappa"}) {
    EXPECT_NO_THROW(split_column(1, per_cell(which, 16 + 4))) << which;
    EXPECT_THROW(split_column(2, per_cell(which, 8 + 4)), c10::Error) << which;
    EXPECT_THROW(split_column(4, per_cell(which, 4 + 4)), c10::Error) << which;
  }
  // the table is the way there
  auto op = coefficients();
  op->nu_scale_x1_table(
      torch::tensor({0., 16., 1., 2.}, torch::kFloat64).view({2, 2}));
  EXPECT_NO_THROW(split_column(2, op));
}

// reset reads the profile once: setting, replacing, clearing or writing into
// it afterwards would be silently ignored, so forward refuses it
TEST(diffusion_x1_scale, profile_change_after_build_is_refused) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  auto cells = [&] { return torch::linspace(1., 2., nc1, torch::kFloat64); };
  auto table = [] {
    return torch::tensor({0., 6., 1., 2.}, torch::kFloat64).view({2, 2});
  };
  using Change = std::function<void(DiffusionOptions const&)>;
  struct Case {
    char const* what;
    bool by_cells;
    Change change;
  };
  std::vector<Case> cases = {
      {"cells replaced", true,
       [&](DiffusionOptions const& op) { op->nu_scale_x1(cells()); }},
      {"cells written into", true,
       [](DiffusionOptions const& op) { op->nu_scale_x1()[3] = 5.; }},
      {"cells cleared", true,
       [](DiffusionOptions const& op) { op->nu_scale_x1(torch::Tensor()); }},
      {"table replaced", false,
       [&](DiffusionOptions const& op) { op->nu_scale_x1_table(table()); }},
      {"table written into", false,
       [](DiffusionOptions const& op) { op->nu_scale_x1_table()[1][0] = 3.; }},
      {"cells added to a table", false,
       [&](DiffusionOptions const& op) { op->kappa_scale_x1(cells()); }},
  };
  for (auto const& c : cases) {
    auto options = base_options();
    if (c.by_cells) {
      options->hydro()->diffusion()->nu_scale_x1(cells());
    } else {
      options->hydro()->diffusion()->nu_scale_x1_table(table());
    }
    auto block = build(options);
    torch::Tensor temp;
    auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
    EXPECT_NO_THROW(tendency(block, w, temp)) << c.what;
    c.change(block->phydro->pdiffusion->options);
    EXPECT_THROW(tendency(block, w, temp), c10::Error) << c.what;
  }
}

// An inference tensor keeps no version counter, so an in-place write to it
// under inference mode after the build could not be detected: it is refused at
// build, per cell and as a table. A YAML table parsed under inference mode is
// an ordinary tensor and is accepted.
TEST_P(DeviceTest, inference_tensor_profile_is_refused) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  torch::Tensor cells, table;
  {
    c10::InferenceMode guard;  // as torch.inference_mode() in Python
    cells = torch::ones({nc1}, torch::device(device).dtype(torch::kFloat64));
    table = torch::tensor({0., 6., 1., 2.}, torch::kFloat64)
                .view({2, 2})
                .to(device);
  }
  ASSERT_TRUE(cells.is_inference());
  auto options = base_options();
  options->hydro()->diffusion()->nu_scale_x1(cells);
  EXPECT_THROW(build(options), c10::Error) << "per cell";
  options = base_options();
  options->hydro()->diffusion()->kappa_scale_x1_table(table);
  EXPECT_THROW(build(options), c10::Error) << "table";

  options = base_options();
  {
    c10::InferenceMode guard;
    options->hydro()->diffusion() =
        parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 2.}));
  }
  EXPECT_FALSE(
      options->hydro()->diffusion()->nu_scale_x1_table().is_inference());
  EXPECT_NO_THROW(build(options));
}

// max_time_step reads the cached profile maxima too: a profile replaced or
// written into after the build, or dynamic set to true beside it, is refused
// there even before the first forward, rather than bounding dt by the stale
// (smaller) coefficient. An unchanged profile still gives its bound.
TEST_P(DeviceTest, profile_change_is_refused_by_max_time_step) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  using Change = std::function<void(DiffusionOptions const&)>;
  struct Case {
    char const* what;
    bool by_cells;
    Change change;
  };
  std::vector<Case> cases = {
      {"table replaced by a larger one", false,
       [](DiffusionOptions const& op) {
         op->nu_scale_x1_table(
             torch::tensor({0., 6., 1., 4.}, torch::kFloat64).view({2, 2}));
       }},
      {"cells written into", true,
       [](DiffusionOptions const& op) { op->nu_scale_x1()[3] = 4.; }},
      {"dynamic set to true", false,
       [](DiffusionOptions const& op) { op->dynamic(true); }},
  };
  for (auto const& c : cases) {
    auto options = base_options();
    if (c.by_cells) {
      options->hydro()->diffusion()->nu_scale_x1(
          torch::ones({nc1}, torch::kFloat64));
    } else {
      options->hydro()->diffusion()->nu_scale_x1_table(
          torch::tensor({0., 6., 1., 1.}, torch::kFloat64).view({2, 2}));
    }
    auto block = build(options);
    block->to(device, dtype);
    torch::Tensor temp;
    auto w = sheared_state(block, device, dtype, &temp);
    auto diffusion = block->phydro->pdiffusion;
    // dx = 1, one active dimension: 1 / (2 nu max s) with max s = 1
    EXPECT_NEAR(diffusion->max_time_step({w}), 1. / (2. * 0.5), 1.e-12)
        << c.what;
    c.change(diffusion->options);
    EXPECT_THROW(diffusion->max_time_step({w}), c10::Error) << c.what;
    EXPECT_THROW(tendency(block, w, temp), c10::Error) << c.what;
  }
}

// Writes that reach the profile's storage without bumping its version counter:
// a tensor over outside memory (torch.from_numpy in Python, from_blob here) and
// the .data alias (variable_data), which shares the storage but not the
// counter. A profile changed after the build must still be refused, at
// max_time_step (before any forward) and at forward.
TEST(diffusion_x1_scale, profile_write_through_outside_storage_is_refused) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  std::vector<double> outside(nc1, 1.);
  auto owned = torch::ones({nc1}, torch::kFloat64);
  struct Case {
    char const* what;
    torch::Tensor cells;
    std::function<void()> write;
  };
  std::vector<Case> cases = {
      {"outside memory (from_numpy)",
       torch::from_blob(outside.data(), {nc1}, torch::kFloat64),
       [&] { outside[3] = 4.; }},
      {".data alias", owned, [&] { owned.variable_data()[3] = 4.; }},
  };
  for (auto const& c : cases) {
    auto options = base_options();
    options->hydro()->diffusion()->nu_scale_x1(c.cells);
    auto block = build(options);
    torch::Tensor temp;
    auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
    auto diffusion = block->phydro->pdiffusion;
    EXPECT_NEAR(diffusion->max_time_step({w}), 1. / (2. * 0.5), 1.e-12)
        << c.what;
    c.write();
    EXPECT_EQ(c.cells[3].item<double>(), 4.) << c.what;  // the write landed
    EXPECT_THROW(diffusion->max_time_step({w}), c10::Error) << c.what;
    EXPECT_THROW(tendency(block, w, temp), c10::Error) << c.what;
  }
}
