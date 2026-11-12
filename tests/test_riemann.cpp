// C/C++
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>

// external
#include <gtest/gtest.h>

// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

#include "device_testing.hpp"

namespace snap {
void call_hllc_mps(at::TensorIterator&, int);
}

// kintera
#include <kintera/species.hpp>

// snap
#include <snap/riemann/hllc_impl.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/riemann/riemann_solver.hpp>

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

const char* small_ideal_gas_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    density-floor:  1.e-10
    pressure-floor: 1.e-10
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 10, nx2: 8, nx3: 6, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char* small_ideal_moist_lmars_cloud_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

  - name: vapor
    composition: {H: 2, O: 1}
    cv_R: 2.5
    u0_R: 0.

  - name: cloud
    composition: {H: 2, O: 1}
    cv_R: 9.0
    u0_R: -3430.

dynamics:
  equation-of-state:
    type: ideal-moist
    density-floor: 1.e-10
    pressure-floor: 1.e-10
    limiter: true

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

forcing:
  const-gravity:
    grav1: -9.8

reactions:
  - equation: vapor <=> cloud
    type: nucleation
    rate-constant: {formula: h2o_ideal}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1.e3, x2min: 0., x2max: 1.e3, x3min: 0., x3max: 1.e3}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

using namespace snap;

TEST(RiemannSolver, hllc_writes_face_pressure_output) {
  auto config = std::regex_replace(std::string(small_ideal_gas_config),
                                   std::regex("type: lmars"), "type: hllc");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(77);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto gamma_l = block->phydro->peos->compute("W->A", {wl});
  auto gamma_r = block->phydro->peos->compute("W->A", {wr});
  auto cl = block->phydro->peos->compute("WA->L", {wl, gamma_l});
  auto cr = block->phydro->peos->compute("WA->L", {wr, gamma_r});

  auto rhoa = 0.5 * (wl[IDN] + wr[IDN]);
  auto ca = 0.5 * (cl + cr);
  auto pmid = 0.5 * (wl[IPR] + wr[IPR] + (wl[IVX] - wr[IVX]) * rhoa * ca);
  auto ql =
      torch::sqrt(1.0 + (gamma_l + 1) / (2 * gamma_l) * (pmid / wl[IPR] - 1.0));
  ql = torch::where(pmid <= wl[IPR], torch::ones_like(ql), ql);
  auto qr =
      torch::sqrt(1.0 + (gamma_r + 1) / (2 * gamma_r) * (pmid / wr[IPR] - 1.0));
  qr = torch::where(pmid <= wr[IPR], torch::ones_like(qr), qr);
  auto al = wl[IVX] - cl * ql;
  auto ar = wr[IVX] + cr * qr;
  auto vxl = wl[IVX] - al;
  auto vxr = wr[IVX] - ar;
  auto tl = wl[IPR] + vxl * wl[IDN] * wl[IVX];
  auto tr = wr[IPR] + vxr * wr[IDN] * wr[IVX];
  auto ml = wl[IDN] * vxl;
  auto mr = -(wr[IDN] * vxr);
  auto expected = ((ml * tr + mr * tl) / (ml + mr)).clamp_min(0.0);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
}

TEST(RiemannSolver, lmars_writes_face_pressure_output) {
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(78);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto gamma_l = block->phydro->peos->compute("W->A", {wl});
  auto gamma_r = block->phydro->peos->compute("W->A", {wr});
  auto rhobar = 0.5 * (wl[IDN] + wr[IDN]);
  auto gamma_bar = 0.5 * (gamma_l + gamma_r);
  auto cbar = torch::sqrt(gamma_bar * 0.5 * (wl[IPR] + wr[IPR]) / rhobar);
  auto expected =
      0.5 * (wl[IPR] + wr[IPR]) + 0.5 * (rhobar * cbar) * (wl[IVX] - wr[IVX]);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
}

TEST(RiemannSolver, roe_writes_face_pressure_output) {
  auto config = std::regex_replace(std::string(small_ideal_gas_config),
                                   std::regex("type: lmars"), "type: roe");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(79);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto ul = block->phydro->peos->compute("W->U", {wl});
  auto ur = block->phydro->peos->compute("W->U", {wr});
  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);
  auto hbar =
      ((ul[IPR] + wl[IPR]) / sqrtdl + (ur[IPR] + wr[IPR]) / sqrtdr) * isdlpdr;
  auto vbar =
      (sqrtdl * wl.narrow(0, IVX, 3) + sqrtdr * wr.narrow(0, IVX, 3)) * isdlpdr;
  auto gamma_l = block->phydro->peos->compute("W->A", {wl});
  auto gamma_r = block->phydro->peos->compute("W->A", {wr});
  auto cs = torch::sqrt((0.5 * (gamma_l + gamma_r) - 1.0) *
                        (hbar - 0.5 * vbar.square().sum(0)));
  auto expected =
      0.5 * (wl[IPR] + wr[IPR] + sqrtdl * sqrtdr * cs * (wl[IVX] - wr[IVX]));

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
  EXPECT_GT(torch::abs(flx).sum().item<double>(), 0.0);
}

TEST(RiemannSolver, roe_writes_face_pressure_output_ideal_moist) {
  auto config =
      std::regex_replace(std::string(small_ideal_moist_lmars_cloud_config),
                         std::regex("type: lmars"), "type: roe");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  kintera::init_species_from_yaml(fname);
  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(80);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].mul_(0.1).add_(1.0);
  wr[IDN].mul_(0.1).add_(1.0);
  wl[IPR].mul_(1000.0).add_(1.e5);
  wr[IPR].mul_(1000.0).add_(1.e5);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wl[ICY].mul_(1.e-3).add_(1.5e-2);
  wr[ICY].mul_(1.e-3).add_(1.5e-2);
  wl[ICY + 1].mul_(1.e-3).add_(4.e-3);
  wr[ICY + 1].mul_(1.e-3).add_(4.e-3);
  wr.copy_(wl);

  auto expected = wl[IPR].clone();

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
  EXPECT_GT(torch::abs(flx).sum().item<double>(), 0.0);
}

TEST(HLLCPerturbation, all_directions_and_contact_branches_match_total_flux) {
  constexpr int n = ICY + 2;
  constexpr double p0 = 1.e7;
  for (int dim : {1, 2, 3}) {
    for (double velocity : {-1.e4, -10., 0., 10., 1.e4}) {
      std::array<double, n> wl{}, wr{}, total{}, shifted{};
      wl[IDN] = 10.;
      wr[IDN] = 10.001;
      wl[IPR] = p0 + 25.;
      wr[IPR] = p0 - 12.;
      wl[IPR - dim] = velocity;
      wr[IPR - dim] = velocity + 0.001;
      wl[ICY] = wr[ICY] = 0.01;
      wl[ICY + 1] = wr[ICY + 1] = 0.02;
      double el = wl[IPR] / 0.4, er = wr[IPR] / 0.4;
      double cl = std::sqrt(1.4 * wl[IPR] / wl[IDN]);
      double cr = std::sqrt(1.4 * wr[IPR] / wr[IDN]);
      double cp, dp;
      hllc_impl(total.data(), wl.data(), wr.data(), el, er, 1.4, 1.4, cl, cr,
                dim, 2, 1, 1, &cp);
      wl[IPR] -= p0;
      wr[IPR] -= p0;
      hllc_impl(shifted.data(), wl.data(), wr.data(), el, er, 1.4, 1.4, cl, cr,
                dim, 2, 1, 1, &dp, p0);
      total[IPR - dim] -= p0;
      for (int row = 0; row < n; ++row)
        EXPECT_NEAR(shifted[row], total[row],
                    2.e-12 * std::abs(total[row]) + 2.e-6)
            << "dim=" << dim << " velocity=" << velocity << " row=" << row;
      EXPECT_NEAR(dp, cp - p0, 1.e-7);
    }
  }
}

TEST(HLLCPerturbation, float_pressure_difference_survives_large_background) {
  constexpr int n = ICY;
  std::array<float, n> wl{}, wr{}, flux{};
  wl[IDN] = wr[IDN] = 10.f;
  wl[IPR] = 0.01f;
  wr[IPR] = -0.01f;
  constexpr float p0 = 1.e7f;
  ASSERT_EQ(p0 + wl[IPR], p0 + wr[IPR]);
  float sound = std::sqrt(1.4f * p0 / 10.f), pressure;
  hllc_impl(flux.data(), wl.data(), wr.data(), p0 / 0.4f, p0 / 0.4f, 1.4f, 1.4f,
            sound, sound, DIM1, 0, 1, 1, &pressure, p0);
  EXPECT_GT(flux[IDN], 0.f);
  EXPECT_GT(flux[IPR], 0.f);
  // A negative pressure anomaly is physical; only the TOTAL is floored.
  wl[IPR] = wr[IPR] = -0.01f;
  hllc_impl(flux.data(), wl.data(), wr.data(), p0 / 0.4f, p0 / 0.4f, 1.4f, 1.4f,
            sound, sound, DIM1, 0, 1, 1, &pressure, p0);
  EXPECT_NEAR(pressure, -0.01f, 1.e-8f);
  EXPECT_NEAR(flux[IVX], -0.01f, 1.e-8f);
  EXPECT_EQ(flux[IDN], 0.f);
}

TEST_P(DeviceTest, hllc_pressure_split_dispatch_and_tensor_backend) {
  auto opts = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  opts->hydro()->riemann()->type() = "hllc";
  auto block = MeshBlock(opts);
  block->to(device, dtype);
  auto solver =
      std::dynamic_pointer_cast<HLLCSolverImpl>(block->phydro->priemann);
  ASSERT_TRUE(solver);
  auto coord = block->pcoord;
  auto wl = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                          coord->options->nc2(), coord->options->nc1()},
                         torch::device(device).dtype(dtype));
  wl[IDN].fill_(10.);
  wl[IPR].fill_(0.01);
  wl[ICY].fill_(0.01);
  wl[ICY + 1].fill_(0.02);
  auto wr = wl.clone();
  wr[IPR].fill_(-0.01);
  auto p0 = torch::full_like(wl[IPR], 1.e7);
  auto flux = torch::zeros_like(wl);
  auto dp = torch::empty_like(p0);
  solver->forward_perturbations(wl.clone(), wr.clone(), DIM1, flux, p0, dp);
  EXPECT_GT(flux[IDN].min().item<double>(), 0.);
  auto mixture_flux = flux[IDN] + flux.narrow(0, ICY, 2).sum(0);
  EXPECT_TRUE(torch::allclose(flux[ICY], 0.01 * mixture_flux, 2.e-6, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(flux[ICY + 1], 0.02 * mixture_flux, 2.e-6, 1.e-12));

  // Exercise the tensor fallback on CPU too, independent of MPS availability.
  auto tensor_flux = torch::zeros_like(flux);
  auto tensor_dp = torch::empty_like(dp);
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .declare_static_shape(flux.sizes(), /*squash_dims=*/0)
                  .add_output(tensor_flux)
                  .add_owned_output(tensor_dp.unsqueeze(0))
                  .add_input(wl)
                  .add_input(wr)
                  .add_input(solver->elr)
                  .add_input(solver->glr)
                  .add_input(solver->clr)
                  .add_owned_input(p0.unsqueeze(0))
                  .build();
  snap::call_hllc_mps(iter, DIM1);
  EXPECT_TRUE(torch::allclose(tensor_flux, flux, 2.e-5, 1.e-6));
  EXPECT_TRUE(torch::allclose(tensor_dp, dp, 2.e-5, 1.e-7));

  auto total_l = wl.clone(), total_r = wr.clone();
  total_l[IPR] += p0;
  total_r[IPR] += p0;
  auto full_flux = torch::zeros_like(flux);
  solver->forward(total_l, total_r, DIM1, full_flux);
  if (dtype == torch::kFloat32)
    EXPECT_EQ(full_flux[IDN].abs().max().item<double>(), 0.);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
