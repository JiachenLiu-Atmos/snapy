// One default per option, whichever way the options are built, and the two
// face-flux scheme choices as per-block deck keys:
//  - EOS floors: a deck whose equation-of-state block leaves the floor keys out
//    gets the floors of EquationOfStateOptionsImpl::create();
//  - Riemann solver: a riemann-solver block without `type` gets the type of
//    RiemannSolverOptionsImpl::create() (hllc), as a deck with no block does;
//  - dynamics/flux-covariance (default on) and dynamics/x1-mass-covariance
//    (default off) are read per block: blocks with different values live in
//    one process and keep them.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/riemann/riemann_solver.hpp>
#include <string>

using namespace snap;

namespace {

//! a sheared, stratified ideal-gas box; `dyn` adds lines under dynamics:
std::string deck(
    std::string const& dyn = "", int nx1 = 8,
    std::string const& eos = "{type: ideal-gas}",
    std::string const& riemann = "  riemann-solver: {type: lmars}\n",
    int nghost = 3, std::string const& recon = "weno5") {
  return "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "geometry:\n"
         "  type: cartesian\n"
         "  bounds: {x1min: 0., x1max: 1000., x2min: 0., x2max: 1000., "
         "x3min: 0., x3max: 1.}\n"
         "  cells: {nx1: " +
         std::to_string(nx1) +
         ", nx2: 8, nx3: 1, nghost: " + std::to_string(nghost) +
         "}\n"
         "dynamics:\n"
         "  equation-of-state: " +
         eos +
         "\n"
         "  reconstruct:\n"
         "    vertical: {type: " +
         recon +
         ", scale: false, shock: false}\n"
         "    horizontal: {type: " +
         recon + ", scale: false, shock: false}\n" + riemann + dyn +
         "forcing:\n"
         "  const-gravity: {grav1: -9.8}\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: reflecting\n"
         "    x1-outer: reflecting\n"
         "    x2-inner: periodic\n"
         "    x2-outer: periodic\n"
         "    x3-inner: periodic\n"
         "    x3-outer: periodic\n";
}

std::string write_temp(std::string const& text) {
  char fname[] = "/tmp/test-options-defaults-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << text;
  return fname;
}

MeshBlockOptions options_of(std::string const& text) {
  auto fname = write_temp(text);
  MeshBlockOptions op;
  try {
    op = MeshBlockOptionsImpl::from_yaml(fname);
  } catch (...) {
    std::remove(fname.c_str());
    throw;
  }
  std::remove(fname.c_str());
  return op;
}

//! the x2 face flux of one forward from the sheared, stratified state (the
//! state is uniform in x2, so the tendency would not see a change of it)
torch::Tensor x2_flux(std::string const& text) {
  auto block = std::make_shared<MeshBlockImpl>(options_of(text));
  auto coord = block->pcoord;
  auto z = (coord->x1v - 500.) / 1000.;  // -0.5 .. 0.5 over the interior
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].copy_((1. - 0.1 * z).expand_as(w[IDN]));
  w[IPR].copy_((1.e5 * (1. - 0.2 * z)).expand_as(w[IPR]));
  w[IVY].copy_((10. * z).expand_as(w[IVY]));
  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  block->phydro->forward(1., vars.at("hydro_u"), vars);
  return block->phydro->flux2().clone();
}

}  // namespace

TEST(OptionsDefaults, eos_floors_without_keys_are_the_create_floors) {
  auto made = EquationOfStateOptionsImpl::create();
  auto read = options_of(deck())->hydro()->eos();
  EXPECT_EQ(read->density_floor(), made->density_floor());
  EXPECT_EQ(read->pressure_floor(), made->pressure_floor());
  EXPECT_EQ(read->temperature_floor(), made->temperature_floor());
  // a deck that sets them keeps its own
  auto own = options_of(deck("", 8,
                             "{type: ideal-gas, density-floor: 1.e-9, "
                             "pressure-floor: 1.e-8, temperature-floor: 5.}"))
                 ->hydro()
                 ->eos();
  EXPECT_EQ(own->density_floor(), 1.e-9);
  EXPECT_EQ(own->pressure_floor(), 1.e-8);
  EXPECT_EQ(own->temperature_floor(), 5.);
}

TEST(OptionsDefaults, riemann_block_without_type_is_the_create_type) {
  auto made = RiemannSolverOptionsImpl::create();
  EXPECT_EQ(made->type(), "hllc");
  // a block without type, read directly and through a deck
  EXPECT_EQ(
      RiemannSolverOptionsImpl::from_yaml(YAML::Load("{dir: omni}"))->type(),
      made->type());
  auto with_block = options_of(
      deck("", 8, "{type: ideal-gas}", "  riemann-solver: {dir: omni}\n"));
  EXPECT_EQ(with_block->hydro()->riemann()->type(), made->type());
  // and a deck with no riemann-solver block at all
  auto without = options_of(deck("", 8, "{type: ideal-gas}", ""));
  EXPECT_EQ(without->hydro()->riemann()->type(), made->type());
  // a block that names its type keeps it
  auto named = options_of(deck());
  EXPECT_EQ(named->hydro()->riemann()->type(), "lmars");
}

TEST(OptionsDefaults, flux_covariance_is_a_per_block_key) {
  // three blocks in one process: the key unset, off and on
  auto unset = x2_flux(deck());
  auto off = x2_flux(deck("  flux-covariance: false\n"));
  auto on = x2_flux(deck("  flux-covariance: true\n"));
  EXPECT_TRUE(torch::equal(unset, on)) << "the default is on";
  EXPECT_FALSE(torch::equal(on, off)) << "the key changed nothing";
  // and back: a later unset block is again on
  EXPECT_TRUE(torch::equal(x2_flux(deck()), on));
}

TEST(OptionsDefaults, x1_mass_covariance_is_a_per_block_key) {
  // the one-sided wall rho_1 spans three cells: with the key on, setup refuses
  // a two-cell block (nghost 2, donor cell); without it the same block builds
  auto two = [](std::string const& dyn) {
    return deck(dyn, 2, "{type: ideal-gas}",
                "  riemann-solver: {type: lmars}\n", 2, "dc");
  };
  try {
    std::make_shared<MeshBlockImpl>(
        options_of(two("  x1-mass-covariance: true\n")));
    ADD_FAILURE() << "nx1 2 accepted with x1-mass-covariance on";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("needs at least 3 x1 cells"),
              std::string::npos)
        << e.what();
  }
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options_of(two(""))));
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(
      options_of(two("  x1-mass-covariance: false\n"))));
}
