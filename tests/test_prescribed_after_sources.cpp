// V3 S4: the post-implicit apply_prescribed (R-5: S's prescribed x1
// faces on the total before any reader, after the implicit correction) must
// also run when a source's post_fluxes wrote total(1) after the x1 sweep's
// own apply: diffusion adds its IPR row on every x1 face, walls included.
// The first V3 S4 version skipped the apply without the implicit correction,
// which left such a flux on the register's wall faces, unbooked; main
// re-prescribed them unconditionally and booked the removal as
// boundary.prescribed.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <snap/forcing/source_term.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/mesh/meshblock.hpp>
#include <tuple>

using namespace snap;

namespace {

char const* kDeck = R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6.4e3, x2min: 0., x2max: 12.8e3, x3min: 0., x3max: 12.8e3}
  cells: {nx1: 32, nx2: 32, nx3: 1, nghost: 3}
distribute: {layout: slab, nb2: 1, nb3: 1, verbose: false}
dynamics:
  equation-of-state: {type: ideal-gas, gammad: 1.4, weight: 29.e-3,
                      density-floor: 1.e-10, pressure-floor: 1.e-10, limiter: false}
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: lmars}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: reflecting,
             x2-outer: reflecting, x3-inner: reflecting, x3-outer: reflecting}
integration: {type: rk3, cfl: 0.9, implicit-scheme: 0, nlim: -1, tlim: 1.e6,
              ncycle_out: 0}
forcing:
  const-gravity: {grav1: -9.8}
)";

// Straka's cold bubble on an adiabatic rest state, eager, no implicit step
MeshBlock bubble(Variables& vars) {
  char fname[] = "/tmp/test-prescribed-after-sources-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << kDeck;
  auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname);
  block->to(torch::kCPU, torch::kFloat64);
  auto co = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), co->options->nc3(),
                         co->options->nc2(), co->options->nc1()},
                        torch::kFloat64);
  auto x1 = co->x1v.view({1, 1, -1}), x2 = co->x2v.view({1, -1, 1});
  double const Rd = 8.314462618 / 29.e-3, cp = 3.5 * Rd, Ts = 300.;
  auto temp = Ts - 9.8 * x1 / cp;
  w[IPR] = 1.e5 * (temp / Ts).pow(cp / Rd);
  auto L =
      (((x2 - 6.4e3) / 4.e3).square() + ((x1 - 3.e3) / 2.e3).square()).sqrt();
  temp = temp + torch::where(L <= 1, -15. * ((L * M_PI).cos() + 1.) / 2., 0.);
  w[IDN] = w[IPR] / (Rd * temp);
  vars["hydro_w"] = w;
  block->initialize(vars);
  return block;
}

// a source whose divergence-phase post puts one unit of energy flux on every
// x1 face, the wall planes included, as a diffusion module's IPR row does
struct WallPoster : SourceTerm {
  explicit WallPoster(bool walls = true) : walls_(walls) {
    decl_.name = "test.wall_poster";
    decl_.rows = kEnergy;
  }
  void add(torch::Tensor, StageState const&) override {}
  void post_fluxes(FluxRegister& reg, StageState const&) override {
    auto t = reg.total(1);
    auto one = torch::ones({1, t.size(1), t.size(2), t.size(3)}, t.options());
    if (!walls_) {  // the interior faces only: 0 on the two wall planes
      one.select(-1, il_).fill_(0.);
      one.select(-1, iu1_).fill_(0.);
    }
    reg.add(1, decl_.name, {IPR}, one);
    ++posts;
  }
  bool walls_;
  int il_ = 0, iu1_ = 0;  // the wall planes (set by the test)
  int posts = 0;
};

}  // namespace

TEST(FluxRegister, a_source_post_on_the_walls_is_prescribed_again) {
  Variables vars;
  auto block = bubble(vars);
  ASSERT_FALSE(block->phydro->picorr) << "the deck must have no implicit step";
  auto poster = std::make_shared<WallPoster>();
  block->phydro->sources.add(poster);
  double dt = 0.5 * block->max_time_step(vars);
  block->forward(vars, dt, 0);
  ASSERT_EQ(poster->posts, 1);

  auto& freg = block->phydro->flux_register();
  auto t = freg.total(1);  // [nvar, nc3, nc2, x1 faces]
  auto co = block->pcoord;
  for (int p : {co->il(), co->iu() + 1}) {
    auto plane = t[IPR].select(-1, p);
    EXPECT_EQ(plane.abs().max().item<double>(), 0.)
        << "x1 wall plane " << p << " keeps the source's energy flux";
  }
  // the interior faces keep it, and the walls' removal is booked
  EXPECT_GT(t[IPR].select(-1, co->il() + 1).min().item<double>(), 0.5);
  auto booked = freg.monitor().amounts("boundary.prescribed")[IPR];
  EXPECT_NE(booked.item<double>(), 0.);
}

// R-5: the prescribed faces are prescribed before any reader. A source's
// divergence-phase post writes the totals after the sweeps' applies, and the
// divergence reads them next: the wall planes must be prescribed again
// before it, or the wall flux reaches du while the register books its
// removal. The same post on the interior faces only must give the same du.
TEST(FluxRegister, a_source_post_on_the_walls_does_not_reach_du) {
  auto run = [](bool walls) {
    Variables vars;
    auto block = bubble(vars);
    auto poster = std::make_shared<WallPoster>(walls);
    poster->il_ = block->pcoord->il();
    poster->iu1_ = block->pcoord->iu() + 1;
    block->phydro->sources.add(poster);
    double dt = 0.5 * block->max_time_step(vars);
    auto du = block->phydro->forward(dt, vars.at("hydro_u"), vars);
    EXPECT_EQ(poster->posts, 1);
    return du;
  };
  auto with = run(true), without = run(false);
  auto diff = (with[IPR] - without[IPR]).abs();
  EXPECT_EQ(diff.max().item<double>(), 0.)
      << "the wall post reached du, max |d du_IPR| = "
      << diff.max().item<double>();
}

// a conducting column: diffusion with a prescribed wall temperature on the
// reflecting x1 walls (the Cartesian form the diffusion module accepts), a
// uniform density and a temperature linear in x1
char const* kConductingDeck = R"(
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 6, nx2: 1, nx3: 1, nghost: 3}
distribute: {layout: slab, nb2: 1, nb3: 1, verbose: false}
dynamics:
  equation-of-state: {type: ideal-gas, limiter: false}
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: lmars}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting}
integration: {type: rk3, cfl: 0.9, implicit-scheme: 0, nlim: -1, tlim: 1.e6,
              ncycle_out: 0}
forcing:
  diffusion: {dynamic: true, nu_iso: 0., kappa_iso: 2.}
)";

MeshBlock conducting_column(Variables& vars, double t_inner, double t_outer) {
  char fname[] = "/tmp/test-conducting-column-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << kConductingDeck;
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  options->hydro()->diffusion()->temperature_inner(t_inner).temperature_outer(
      t_outer);
  auto block = MeshBlock(options);
  block->to(torch::kCPU, torch::kFloat64);
  auto co = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), co->options->nc3(),
                         co->options->nc2(), co->options->nc1()},
                        torch::kFloat64);
  auto temperature = 300. + 10. * co->x1v.view({1, 1, -1});
  double gas = block->phydro->peos->species_cv_ref() *
               (block->phydro->peos->options->gammad() - 1.);
  w[IDN].fill_(1.);
  w[IPR] = gas * temperature;  // rho = 1
  vars["hydro_w"] = w;
  block->initialize(vars);
  return block;
}

// A wall whose boundary prescribes a temperature has no mass flux but
// conducts: the diffusion's energy flux on that plane reaches du in a full
// step and nothing is booked as its removal. Raising the lower wall 5 K above
// the linear profile injects K dT / (dz/2) = 20 per unit time; the walls
// without a prescribed temperature stay at zero flux and book nothing.
TEST(FluxRegister, a_prescribed_wall_temperature_conducts_in_a_full_step) {
  auto run = [](double t_inner, double t_outer) {
    Variables vars;
    auto block = conducting_column(vars, t_inner, t_outer);
    auto u0 = vars.at("hydro_u").clone();
    double dt = 0.5 * block->max_time_step(vars);
    block->forward(vars, dt, 0);  // rk3 stage 0: u1 = u0 + dt du
    auto cells = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto volume = block->pcoord->cell_volume();
    double de = ((vars.at("hydro_u") - u0)[IPR] * volume)
                    .index(cells)
                    .sum()
                    .item<double>();
    auto booked = block->phydro->flux_register()
                      .monitor()
                      .amounts("boundary.prescribed")[IPR]
                      .item<double>();
    // round-off of a mirrored wall's zero-gradient flux is measured against
    // the column's energy, not against the conduction
    double tol = 1.e-12 * (u0[IPR] * volume).index(cells).sum().item<double>();
    return std::make_tuple(de, booked, dt, tol);
  };
  auto [de, booked, dt, tol] = run(305., 360.);
  EXPECT_NEAR(de, 20. * dt, 1.e-9 * 20. * dt)
      << "the prescribed-temperature wall did not conduct into du";
  EXPECT_NEAR(booked, 0., tol) << "the conduction was booked as a removal";
  auto [de0, booked0, dt0, tol0] = run(0., 0.);
  EXPECT_LT(tol0, 1.e-3 * 20. * dt0)
      << "the round-off bound reaches the signal";
  EXPECT_NEAR(de0, 0., tol0) << "a reflecting wall conducted";
  EXPECT_NEAR(booked0, 0., tol0);
}

// The mark the post-implicit apply reads: every add, scale and replace sets
// it, apply_prescribed clears it; the prescription itself and keep do not
// set it
TEST(FluxRegister, the_written_mark_follows_the_writers) {
  Variables vars;
  auto block = bubble(vars);
  auto& freg = block->phydro->flux_register();
  auto t = freg.total(1);
  ASSERT_TRUE(t.defined());
  auto one = torch::ones({1, t.size(1), t.size(2), t.size(3)}, t.options());
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.add(1, "test.writes", {IPR}, one);
  EXPECT_TRUE(freg.written_since_prescribed(1));
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.scale(1, "test.writes", {IPR}, one[0]);
  EXPECT_TRUE(freg.written_since_prescribed(1));
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  auto faces =
      torch::zeros(t.sizes().slice(1), t.options().dtype(torch::kBool));
  faces.select(-1, 3).fill_(true);
  std::vector<torch::Tensor> rows = {t.clone()};  // the dense peer: the
  for (auto const& part : freg.kept_parts(1))     // total's rows, then the
    rows.push_back(freg.kept(1, part).clone());   // kept parts'
  freg.replace(1, faces, torch::cat(rows, 0), SeamRule::mean);
  EXPECT_TRUE(freg.written_since_prescribed(1));
}
