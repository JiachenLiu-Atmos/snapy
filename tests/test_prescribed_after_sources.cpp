// V3 S4 round 7: the post-implicit apply_prescribed (R-5: S's prescribed x1
// faces on the total before any reader, after the implicit correction) must
// also run when a source's post_fluxes wrote total(1) after the x1 sweep's
// own apply: diffusion adds its IPR row on every x1 face, walls included.
// Round 1 skipped the apply without the implicit correction, which left such
// a flux on the register's wall faces, unbooked; main re-prescribed them
// unconditionally and booked the removal as boundary.prescribed.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <snap/forcing/source_term.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/mesh/meshblock.hpp>

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
  WallPoster() {
    decl_.name = "test.wall_poster";
    decl_.rows = kEnergy;
  }
  void add(torch::Tensor, StageState const&) override {}
  void post_fluxes(FluxRegister& reg, StageState const&) override {
    auto t = reg.total(1);
    reg.add(1, decl_.name, {IPR},
            torch::ones({1, t.size(1), t.size(2), t.size(3)}, t.options()));
    ++posts;
  }
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

// round 7c: the mark the post-implicit apply reads. add, scale and replace
// set it, a write that keeps the prescribed faces (the positivity limiter's)
// does not, apply_prescribed clears it; the prescription itself and keep do
// not set it (c3 paid one needless wall apply per block and stage)
TEST(FluxRegister, the_written_mark_follows_the_writers) {
  Variables vars;
  auto block = bubble(vars);
  auto& freg = block->phydro->flux_register();
  auto t = freg.total(1);
  ASSERT_TRUE(t.defined());
  auto one = torch::ones({1, t.size(1), t.size(2), t.size(3)}, t.options());
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.add(1, "test.keeps", {IPR}, one, /*keeps_prescribed=*/true);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.scale(1, "test.keeps", {IPR}, one[0], /*keeps_prescribed=*/true);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.add(1, "test.writes", {IPR}, one);
  EXPECT_TRUE(freg.written_since_prescribed(1));
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  freg.scale(1, "test.writes", {IPR}, one[0]);
  EXPECT_TRUE(freg.written_since_prescribed(1));
  freg.apply_prescribed(1);
  EXPECT_FALSE(freg.written_since_prescribed(1));
  auto faces = torch::zeros(t.sizes().slice(1), t.options().dtype(torch::kBool));
  faces.select(-1, 3).fill_(true);
  std::vector<torch::Tensor> rows = {t.clone()};  // the dense peer: the
  for (auto const& part : freg.kept_parts(1))      // total's rows, then the
    rows.push_back(freg.kept(1, part).clone());   // kept parts'
  freg.replace(1, faces, torch::cat(rows, 0), SeamRule::mean);
  EXPECT_TRUE(freg.written_since_prescribed(1));
}
