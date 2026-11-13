// One RED case per behaviour change of the option defaults and deck keys.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

// an ideal-gas column, donor cell, nghost 2; `dyn` is added under dynamics:
static std::shared_ptr<MeshBlockImpl> block(std::string const& dyn,
                                            int nx1 = 8) {
  char f[] = "/tmp/test-options-defaults-XXXXXX";
  close(mkstemp(f));
  std::ofstream(f) << "geometry: {type: cartesian, bounds: {x1min: 0., x1max: "
                      "1000., x2min: 0., x2max: 1000., x3min: 0., x3max: 1.}, "
                      "cells: {nx1: "
                   << nx1 << ", nx2: 8, nx3: 1, nghost: 2}}\n"
                   << "dynamics: {equation-of-state: {type: ideal-gas}, "
                      "reconstruct: {vertical: {type: dc}, horizontal: {type: "
                      "dc}}, riemann-solver: {type: lmars}"
                   << dyn << "}\nforcing: {const-gravity: {grav1: -9.8}}\n"
                   << "boundary-condition: {external: {x1-inner: reflecting, "
                      "x1-outer: reflecting, x2-inner: periodic, x2-outer: "
                      "periodic, x3-inner: periodic, x3-outer: periodic}}\n";
  auto b = std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(f));
  std::remove(f);
  return b;
}

TEST(OptionsDefaults, floors) {  // create() floors were 1e-10 / 1e-10
  auto made = EquationOfStateOptionsImpl::create();
  EXPECT_EQ(made->density_floor(), 1.e-6);
  EXPECT_EQ(made->pressure_floor(), 1.e-3);
}

TEST(OptionsDefaults, riemann_type) {  // a block without type was roe
  auto rs = RiemannSolverOptionsImpl::from_yaml(YAML::Load("{dir: omni}"));
  EXPECT_EQ(rs->type(), "hllc");
}

TEST(OptionsDefaults, flux_covariance_key) {  // x2 flux, shear in x1
  auto flux2 = [](std::string const& dyn) {
    auto b = block(dyn);
    auto z = (b->pcoord->x1v - 500.) / 1000.;
    auto w = torch::zeros({5, 1, 12, 12}, torch::kFloat64);
    w[IDN] = 1. - 0.1 * z, w[IPR] = 1.e5 * (1. - 0.2 * z), w[IVY] = 10. * z;
    Variables v{{"hydro_w", w}};
    b->initialize(v), b->phydro->forward(1., v.at("hydro_u"), v);
    return b->phydro->flux2().clone();
  };
  EXPECT_FALSE(torch::equal(flux2(""), flux2(", flux-covariance: false")));
}

TEST(OptionsDefaults, x1_mass_covariance_key) {  // needs 3 x1 cells when on
  EXPECT_THROW(block(", x1-mass-covariance: true", 2), c10::Error);
  EXPECT_NO_THROW(block(", x1-mass-covariance: false", 2));
}

TEST(OptionsDefaults, replaced_variables_warn_once) {
  struct : c10::WarningHandler {
    std::string all;
    void process(c10::Warning const& w) override { all += w.msg(); }
  } cap;
  setenv("SNAP_FLUX_COVARIANCE", "0", 1),
      setenv("SNAP_X1_MASS_COVARIANCE", "1", 1);
  {
    c10::WarningUtils::WarningHandlerGuard g(&cap);
    block(""), block("");
  }
  unsetenv("SNAP_FLUX_COVARIANCE"), unsetenv("SNAP_X1_MASS_COVARIANCE");
  for (auto key : {"dynamics/flux-covariance", "dynamics/x1-mass-covariance"})
    EXPECT_TRUE(cap.all.find(key) != std::string::npos &&  // exactly once
                cap.all.find(key) == cap.all.rfind(key))
        << cap.all;
}
