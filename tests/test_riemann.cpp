// C/C++
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <regex>

// POSIX
#include <unistd.h>

// external
#include <gtest/gtest.h>

// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

#include "device_testing.hpp"

// kintera
#include <kintera/species.hpp>

// snap
#include <snap/riemann/riemann_kernels.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/hydro/sweep.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/riemann/riemann.hpp>
#include <snap/utils/on_host.hpp>

namespace snap {
void riemann_cpu(at::TensorIterator&, int, int, RiemannParams const&,
                 torch::Tensor const&, torch::Tensor const&,
                 torch::Tensor const&);
}

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

// the filename reader reads the section it is given, not always dynamics:
// (audit 1 row R5)
TEST(RiemannSolverOptions, from_yaml_reads_its_section) {
  char fname[] = "/tmp/test-riemann-section-XXXXXX";
  int fd = mkstemp(fname);
  ASSERT_NE(fd, -1);
  close(fd);
  std::ofstream(fname) << "dynamics:\n  riemann-solver: {type: lmars}\n"
                       << "other:\n  riemann-solver: {type: roe, dir: yz}\n"
                       << "bad:\n  riemann-solver: {type: roe, typo: 1}\n";
  auto other = RiemannSolverOptionsImpl::from_yaml(fname, "other");
  EXPECT_EQ(other->type(), "roe");
  EXPECT_EQ(other->dir(), "yz");
  EXPECT_EQ(RiemannSolverOptionsImpl::from_yaml(fname, "dynamics")->type(),
            "lmars");
  EXPECT_ANY_THROW(RiemannSolverOptionsImpl::from_yaml(fname, "bad"))
      << "the key check must look at the section it reads";
  std::remove(fname);
}

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

TEST(HLLCFloat, shocks_rarefactions_and_boosts_match_double) {
  constexpr int n = ICY + 1;
  // Sod, strong shock, expansion, and an equal-state Galilean boost.
  const double cases[][6] = {{1., 1., 0., .125, .1, 0.},
                             {1., 1000., 0., 1., .01, 0.},
                             {1., .4, -2., 1., .4, 2.},
                             {1., 1., 0., 1., 1., 0.}};
  for (int dim : {1, 2, 3}) {
    for (auto const& c : cases) {
      for (float boost : {-1.e8f, -100.f, 0.f, 100.f, 1.e8f}) {
        for (float p0 : {0.f, 1.e7f}) {
          std::array<float, n> l{}, r{}, f{};
          l[IDN] = c[0];
          r[IDN] = c[3];
          l[IPR] = c[1];
          r[IPR] = c[4];
          l[IPR - dim] = boost + c[2];
          r[IPR - dim] = boost + c[5];
          l[ICY] = r[ICY] = .01f;
          const float gamma = 1.4f;
          float el = (p0 + l[IPR]) / (gamma - 1),
                er = (p0 + r[IPR]) / (gamma - 1);
          float cl = std::sqrt(gamma * (p0 + l[IPR]) / l[IDN]);
          float cr = std::sqrt(gamma * (p0 + r[IPR]) / r[IDN]);
          float fp;
          hllc_impl(f.data(), l.data(), r.data(), el, er, gamma, gamma, cl, cr,
                    dim, 1, 1, 1, &fp, p0);
          std::array<double, n> ld{}, rd{}, ref{};
          for (int i = 0; i < n; ++i) {
            ld[i] = l[i];
            rd[i] = r[i];
          }
          double dp;
          hllc_impl(ref.data(), ld.data(), rd.data(), double(el), double(er),
                    double(gamma), double(gamma), double(cl), double(cr), dim,
                    1, 1, 1, &dp, double(p0));
          EXPECT_TRUE(std::isfinite(fp));
          EXPECT_GE(double(fp) + p0, 0.);
          for (int i = 0; i < n; ++i) {
            EXPECT_TRUE(std::isfinite(f[i]));
            EXPECT_NEAR(f[i], ref[i], 3.e-4 * std::max(1., std::abs(ref[i])))
                << "dim=" << dim << " boost=" << boost << " p0=" << p0
                << " row=" << i;
          }
        }
      }
    }
  }
}

TEST_P(DeviceTest, hllc_pressure_split_dispatch_and_on_host) {
  auto opts = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  opts->hydro()->riemann()->type() = "hllc";
  auto block = MeshBlock(opts);
  block->to(device, dtype);
  auto solver = block->phydro->priemann;
  ASSERT_TRUE(solver->kind == RiemannKind::hllc);
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

  // MPS runs the CPU kernel through OnHost; run that here on every device.
  // On CPU it is the dispatched kernel itself, so the result is bitwise.
  auto host_flux = torch::zeros_like(flux);
  auto host_dp = torch::empty_like(dp);
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .declare_static_shape(flux.sizes(), /*squash_dims=*/0)
                  .add_output(host_flux)
                  .add_owned_output(host_dp.unsqueeze(0))
                  .add_input(wl)
                  .add_input(wr)
                  .add_input(solver->elr)
                  .add_input(solver->glr)
                  .add_input(solver->clr)
                  .add_owned_input(p0.unsqueeze(0))
                  .build();
  snap::OnHost<&snap::riemann_cpu>::run(
      iter, static_cast<int>(snap::RiemannKind::hllc), DIM1,
      snap::RiemannParams{}, torch::Tensor(), torch::Tensor(), torch::Tensor());
  if (device.is_cpu()) {
    EXPECT_TRUE(torch::equal(host_flux, flux));
    EXPECT_TRUE(torch::equal(host_dp, dp));
  } else {
    EXPECT_TRUE(torch::allclose(host_flux, flux, 2.e-5, 1.e-6));
    EXPECT_TRUE(torch::allclose(host_dp, dp, 2.e-5, 1.e-7));
  }

  auto total_l = wl.clone(), total_r = wr.clone();
  total_l[IPR] += p0;
  total_r[IPR] += p0;
  auto full_flux = torch::zeros_like(flux);
  solver->forward(total_l, total_r, DIM1, full_flux);
  if (dtype == torch::kFloat32)
    EXPECT_EQ(full_flux[IDN].abs().max().item<double>(), 0.);
}

// C5: the solve writes riemann.central, the mass rows (dry and species) of
// 1/2 (F(w_L) + F(w_R)), in the kernel. It replaces the two extra solves of
// hydro_forward.cpp (F(w, w) of each side), so it must equal them to
// round-off (C.rst section 5: 1e-14 relative, FP64); the other rows of the
// part are not written.
TEST_P(DeviceTest, riemann_central_matches_the_two_extra_solves) {
  for (std::string type : {"hllc", "lmars", "roe"}) {
    auto opts = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
    opts->hydro()->riemann()->type() = type;
    auto block = MeshBlock(opts);
    block->to(device, dtype);
    auto solver = block->phydro->priemann;
    auto coord = block->pcoord;
    int nvar = block->phydro->peos->nvar(), ny = nvar - ICY;
    ASSERT_GT(ny, 0);
    auto shape =
        std::vector<int64_t>{nvar, coord->options->nc3(), coord->options->nc2(),
                             coord->options->nc1()};
    auto o = torch::device(device).dtype(dtype);
    torch::manual_seed(7);
    auto state = [&] {
      auto w = torch::zeros(shape, o);
      w[IDN] = 1. + torch::rand_like(w[IDN]);
      w.narrow(0, IVX, 3).copy_(20. * torch::rand_like(w.narrow(0, IVX, 3)) -
                                10.);
      w[IPR] = 1.e5 * (0.9 + 0.2 * torch::rand_like(w[IPR]));
      w.narrow(0, ICY, ny).copy_(0.01 * torch::rand_like(w.narrow(0, ICY, ny)));
      return w;
    };
    auto wl = state(), wr = state();
    for (int dim : {DIM1, DIM2, DIM3}) {
      auto flux = torch::zeros(shape, o), cen = torch::zeros(shape, o);
      solver->forward(wl.clone(), wr.clone(), dim, flux, torch::Tensor(), cen);
      auto fa = torch::zeros(shape, o), fb = torch::zeros(shape, o);
      solver->forward(wl.clone(), wl.clone(), dim, fa);
      solver->forward(wr.clone(), wr.clone(), dim, fb);
      auto ref = 0.5 * (fa + fb);
      auto mass = [&](torch::Tensor const& f) {
        return torch::cat({f.narrow(0, IDN, 1), f.narrow(0, ICY, ny)});
      };
      double tol = dtype == torch::kFloat64 ? 1.e-14 : 2.e-6;
      auto err = ((mass(cen) - mass(ref)).abs().max() / mass(ref).abs().max())
                     .item<double>();
      EXPECT_LE(err, tol) << type << " dim " << dim;
      EXPECT_EQ(cen.narrow(0, IVX, 4).abs().max().item<double>(), 0.)
          << type << " dim " << dim;
    }
  }
}

// A-ph4 (Xi's ruling, kickoff 1795248311): riemann.central on the HLLC
// perturbation-flux path. One Cartesian ideal-moist HLLC block; the same total
// face states go through the sweep's riemann_solve on the total path and on the
// perturbation path (pressure rows shifted by the static face gauge built from
// a p0 field). The central is the total-density 1/2 (F(w_L) + F(w_R)) of the
// mass rows on both paths: it reads only the density, normal velocity and
// species rows, which the perturbation path leaves as totals.
TEST_P(DeviceTest, hllc_perturbation_central_matches_total_central) {
  auto config =
      std::regex_replace(std::string(small_ideal_moist_lmars_cloud_config),
                         std::regex("type: lmars"), "type: hllc");
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();
  kintera::init_species_from_yaml(fname);
  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(device, dtype);
  std::remove(fname);

  auto& hydro = *block->phydro;
  ASSERT_TRUE(hydro.priemann->kind == RiemannKind::hllc);
  ASSERT_EQ(hydro.peos->options->type(), "ideal-moist");
  ASSERT_EQ(block->pcoord->options->type(), "cartesian");
  auto c = block->pcoord;
  int nvar = hydro.peos->nvar(), ny = nvar - ICY;
  ASSERT_EQ(ny, 2);
  auto shape = std::vector<int64_t>{nvar, c->options->nc3(), c->options->nc2(),
                                    c->options->nc1()};
  auto o = torch::device(device).dtype(dtype);
  torch::manual_seed(11);
  // a p0 field that varies along every axis, and total states near it
  auto p0 =
      1.e5 * (0.95 + 0.1 * torch::rand({shape[1], shape[2], shape[3]}, o));
  auto state = [&] {
    auto w = torch::zeros(shape, o);
    w[IDN] = 1. + 0.1 * torch::rand_like(w[IDN]);
    w.narrow(0, IVX, 3).copy_(20. * torch::rand_like(w.narrow(0, IVX, 3)) -
                              10.);
    w[IPR] = p0 * (1. + 1.e-3 * (torch::rand_like(p0) - 0.5));
    w[ICY] = 1.5e-2 + 1.e-3 * torch::rand_like(w[ICY]);
    w[ICY + 1] = 4.e-3 + 1.e-3 * torch::rand_like(w[ICY + 1]);
    return w;
  };
  auto wl = state(), wr = state();
  auto lr = torch::stack({wl, wr});
  IdealMoistPerturbations perturb;
  perturb.pressure0 = p0;
  auto mass = [&](torch::Tensor const& f) {
    return torch::cat({f.narrow(0, IDN, 1), f.narrow(0, ICY, ny)});
  };
  // tolerance: C.rst section 5 round-off, 1e-14 relative in FP64 (as in
  // riemann_central_matches_the_two_extra_solves), 2e-6 in FP32
  double tol = dtype == torch::kFloat64 ? 1.e-14 : 2.e-6;
  for (int dim : {DIM1, DIM2, DIM3}) {
    auto flux_t = torch::zeros(shape, o), cen_t = torch::zeros(shape, o);
    auto flux_p = torch::zeros(shape, o), cen_p = torch::zeros(shape, o);
    torch::Tensor bg_t, bg_p;
    riemann_solve(hydro, lr.clone(), dim, flux_t, torch::Tensor(), bg_t,
                  nullptr, cen_t);
    riemann_solve(hydro, lr.clone(), dim, flux_p, torch::Tensor(), bg_p,
                  &perturb, cen_p);
    ASSERT_TRUE(bg_p.defined());
    EXPECT_GT(bg_p.abs().min().item<double>(), 0.9e5);
    // the total-density central from the face states themselves
    int ivx = IPR - dim;
    auto ml = wl[IDN] * wl[ivx], mr = wr[IDN] * wr[ivx];
    auto rdl = 1. - wl.narrow(0, ICY, ny).sum(0);
    auto rdr = 1. - wr.narrow(0, ICY, ny).sum(0);
    auto ref = torch::cat(
        {(0.5 * (ml * rdl + mr * rdr)).unsqueeze(0),
         0.5 * (ml * wl.narrow(0, ICY, ny) + mr * wr.narrow(0, ICY, ny))});
    auto scale = mass(cen_t).abs().max().item<double>();
    ASSERT_GT(scale, 0.);
    double abs_tp = (mass(cen_p) - mass(cen_t)).abs().max().item<double>();
    double abs_pr = (mass(cen_p) - ref).abs().max().item<double>();
    std::printf(
        "A-ph4 central %s dim %d: max|p-t| %.3e (rel %.3e), max|p-ref| %.3e "
        "(rel %.3e)\n",
        torch::toString(dtype), dim, abs_tp, abs_tp / scale, abs_pr,
        abs_pr / scale);
    EXPECT_LE(abs_tp / scale, tol) << "dim " << dim;
    EXPECT_LE(abs_pr / scale, tol) << "dim " << dim;
    // the other rows are not written on either path
    EXPECT_EQ(cen_p.narrow(0, IVX, 4).abs().max().item<double>(), 0.);
    EXPECT_EQ(cen_t.narrow(0, IVX, 4).abs().max().item<double>(), 0.);
    // FP64: the perturbation path solved the same faces (its mass flux
    // agrees with the total path's to the pressure-split round-off)
    if (dtype == torch::kFloat64) {
      auto dflux = mass(flux_p - flux_t).abs().max();
      EXPECT_LE((dflux / mass(flux_t).abs().max()).item<double>(), 1.e-9)
          << "dim " << dim;
    }
  }
}

// G-R8 (R.rst; lane A's G2, A.rst at c405d491): the kernel's riemann.central
// against today's central, 1/2 (F(w_L, w_L) + F(w_R, w_R)), on the face
// states the deck's reconstruction gives, every x2 and x3 face, one stage,
// FP64: e_inf = max_f |dF_c| / max_f rho (|u_n| + c) <= 1e-13. The decks are
// test_axial_momentum_flux_form's 3D arm (lmars, weno5, interp_order 4) and
// W2 C16 (shallow water, shallow-roe yz), one panel each, under a smooth
// non-uniform wind.
const char* g_r8_3d_config = R"(
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - name: dry
    composition: {H: 1.64813, He: 0.17591}
    cv_R: 2.5
geometry:
  type: gnomonic-equiangle
  bounds: {x1min: 6.0e6, x1max: 6.4e6, x2min_pi: -0.25, x2max_pi: 0.25, x3min_pi: -0.25, x3max_pi: 0.25}
  cells: {nx1: 8, nx2: 8, nx3: 8, nghost: 3, interp_order: 4}
dynamics:
  equation-of-state: {type: ideal-gas, density-floor: 1.e-20, pressure-floor: 1.e-20, limiter: false}
  reconstruct:
    vertical:   {type: weno5, scale: true, shock: false}
    horizontal: {type: weno5, scale: true, shock: false}
  riemann-solver: {type: lmars}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: custom, x2-outer: custom, x3-inner: custom, x3-outer: custom}
)";

const char* g_r8_w2_config = R"(
geometry:
  type: gnomonic-equiangle
  bounds: {x1min: 6371000.0, x1max: 6371001.0, x2min_pi: -0.25, x2max_pi: 0.25, x3min_pi: -0.25, x3max_pi: 0.25}
  cells: {nx1: 1, nx2: 16, nx3: 16, nghost: 3}
dynamics:
  equation-of-state: {type: shallow-water}
  reconstruct:
    vertical:   {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: shallow-roe, dir: yz}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: custom, x2-outer: custom, x3-inner: custom, x3-outer: custom}
)";

// the decks' initial states, as test_axial_momentum_flux_form.py builds them:
// the 3D arm's isothermal column in the scheme's balance (balance_column),
// times the rigid-rotation factor, with the perturbed jet (arm (f) at
// Omega = the wind's rate); W2's steady zonal flow at the Earth's rate
constexpr double kWsb = 5.e-6, kUpert = 10., kP0 = 1.e5, kRho0 = 0.1;
constexpr double kSwOmega = 7.292e-5, kSwU0 = 38.6, kSwPhi0 = 2.94e4;

torch::Tensor g_r8_3d_state(MeshBlock block) {
  auto coord = block->pcoord;
  auto o = coord->x1v.options().dtype(torch::kFloat64);
  int ng = coord->options->nghost(), nx1 = coord->options->nx1();
  double g = 10.;
  auto r = coord->x1v.narrow(0, ng, nx1).cpu().to(torch::kFloat64);
  auto dr = coord->dx1f.narrow(0, ng, nx1).cpu().to(torch::kFloat64);
  auto p = kP0 * torch::exp(-(r - r[0]) * kRho0 * g / kP0);
  auto col = torch::zeros({IPR + 1, 1, 1, nx1}, torch::kFloat64);
  col[IDN][0][0] = p * kRho0 / kP0;
  col[IPR][0][0] = p;
  auto wb = std::get<0>(balance_column(col, dr.contiguous(), g));
  int nvar = block->phydro->peos->nvar();
  auto w = torch::zeros({nvar, coord->options->nc3(), coord->options->nc2(),
                         coord->options->nc1()},
                        o);
  for (int c : {IDN, IPR}) {
    auto row = wb[c][0][0].to(o);
    w[c].narrow(-1, ng, nx1).copy_(row);
    w[c].narrow(-1, 0, ng).copy_(row.narrow(0, 0, 1).expand({ng}));
    w[c].narrow(-1, ng + nx1, ng).copy_(row.narrow(0, nx1 - 1, 1).expand({ng}));
  }
  auto bufs = block->named_buffers();
  auto lon = bufs["coord.longitude"].to(o), lat = bufs["coord.latitude"].to(o);
  auto rc = coord->x1v.to(o) * torch::cos(lat);
  auto fac = torch::exp((kWsb * kWsb + 2. * kWsb * kWsb) * rc * rc /
                        (2. * kP0 / kRho0));
  w[IDN] *= fac;
  w[IPR] *= fac;
  auto k = bufs["coord.aam_k_c"].to(o);
  auto u_east =
      kWsb * rc + kUpert * torch::cos(lat).pow(2) * torch::sin(2. * lon);
  w[IVY] = u_east * k[1] / torch::cos(lat);
  w[IVZ] = u_east * k[2] / torch::cos(lat);
  return w;
}

torch::Tensor g_r8_w2_state(MeshBlock block) {
  auto coord = block->pcoord;
  auto o = coord->x1v.options().dtype(torch::kFloat64);
  int nvar = block->phydro->peos->nvar();
  auto w = torch::zeros({nvar, coord->options->nc3(), coord->options->nc2(),
                         coord->options->nc1()},
                        o);
  auto bufs = block->named_buffers();
  auto lat = bufs["coord.latitude"].to(o), a = coord->x1v.to(o);
  auto k = bufs["coord.aam_k_c"].to(o);
  w[IDN] = kSwPhi0 - (a * kSwOmega * kSwU0 + 0.5 * kSwU0 * kSwU0) *
                         torch::sin(lat).pow(2);
  w[IVY] = kSwU0 * k[1] + 0. * a;
  w[IVZ] = kSwU0 * k[2] + 0. * a;
  return w;
}

double g_r8_e_inf(const char* config,
                  std::function<torch::Tensor(MeshBlock)> state,
                  torch::Device device) {
  auto fname = std::string("test_riemann_g_r8.yaml");
  {
    std::ofstream f(fname);
    f << config;
  }
  auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname.c_str());
  block->to(device, torch::kFloat64);
  auto hydro = block->phydro;
  auto peos = hydro->peos;
  int nvar = peos->nvar(), ny = std::max(nvar - ICY, 0);
  auto w = state(block).to(device);

  double e = 0.;
  for (int dim : {DIM2, DIM3}) {
    auto lr = hydro->precon23->forward(w, dim);
    auto like = torch::zeros_like(w);
    auto flux = like.clone(), cen = like.clone();
    hydro->priemann->forward(lr[ILT].clone(), lr[IRT].clone(), dim, flux,
                             torch::Tensor(), cen);
    auto fa = like.clone(), fb = like.clone();
    hydro->priemann->forward(lr[ILT].clone(), lr[ILT].clone(), dim, fa);
    hydro->priemann->forward(lr[IRT].clone(), lr[IRT].clone(), dim, fb);
    auto ref = 0.5 * (fa + fb);
    auto mass = [&](torch::Tensor const& f) {
      return ny > 0 ? torch::cat({f.narrow(0, IDN, 1), f.narrow(0, ICY, ny)})
                    : f.narrow(0, IDN, 1);
    };
    int ivn = dim == DIM2 ? IVY : IVZ;
    double scale = 0.;
    for (int side : {ILT, IRT}) {
      auto s = lr[side];
      auto c = peos->compute("WA->L", {s, peos->compute("W->A", {s})});
      scale =
          std::max(scale, (s[IDN] * (s[ivn].abs() + c)).max().item<double>());
    }
    // the kernel wrote the mass rows: the gate cannot pass vacuously
    EXPECT_GT(mass(cen).abs().max().item<double>(), 0.);
    e = std::max(e, (mass(cen) - mass(ref)).abs().max().item<double>() / scale);
  }
  return e;
}

TEST_P(DeviceTest, g_r8_riemann_central_on_the_aam_decks) {
  if (dtype != torch::kFloat64) GTEST_SKIP() << "G-R8 is an FP64 gate";
  double e3d = g_r8_e_inf(g_r8_3d_config, g_r8_3d_state, device);
  double ew2 = g_r8_e_inf(g_r8_w2_config, g_r8_w2_state, device);
  std::printf("G-R8 e_inf: 3D arm deck %.3e, W2 C16 deck %.3e\n", e3d, ew2);
  EXPECT_LE(e3d, 1.e-13);
  EXPECT_LE(ew2, 1.e-13);
}

// C5 events (C.rst 3.6, R.rst 2.6 names riemann.pstar_clamp and riemann.llf):
// a gentle jump clamps nothing; a strong expansion (u = -/+2000 m/s) makes
// hllc's contact pressure non-positive on every interior face and sends roe
// to its LLF fallback.
TEST_P(DeviceTest, riemann_events_count_clamped_and_llf_faces) {
  for (std::string type : {"hllc", "roe"}) {
    auto opts = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
    opts->hydro()->riemann()->type() = type;
    auto block = MeshBlock(opts);
    block->to(device, dtype);
    auto solver = block->phydro->priemann;
    auto c = block->pcoord;
    int nvar = block->phydro->peos->nvar();
    auto shape = std::vector<int64_t>{nvar, c->options->nc3(),
                                      c->options->nc2(), c->options->nc1()};
    auto o = torch::device(device).dtype(dtype);
    auto state = [&](double u) {
      auto w = torch::zeros(shape, o);
      w[IDN].fill_(1.);
      w[IVX].fill_(u);
      w[IPR].fill_(1.e5);
      w[ICY].fill_(0.01);
      return w;
    };
    int64_t n1 = std::min<int64_t>(c->iu() + 2, shape[3]) - c->il();
    int64_t n2 = c->ju() + 1 - c->jl(), n3 = c->ku() + 1 - c->kl();
    int64_t interior = n1 * n2 * n3;
    for (double u : {1., 2000.}) {
      auto flux = torch::zeros(shape, o);
      solver->forward(state(-u), state(u), DIM1, flux);
      auto ev = solver->event_counts();
      int k = type == "hllc" ? 0 : 1;
      if (u < 10.) {
        EXPECT_EQ(ev[0].item<int64_t>() + ev[1].item<int64_t>(), 0) << type;
      } else if (type == "hllc") {
        EXPECT_EQ(ev[0].item<int64_t>(), interior) << type;
      } else {
        EXPECT_GT(ev[k].item<int64_t>(), 0) << type;
        EXPECT_LE(ev[k].item<int64_t>(), interior) << type;
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
