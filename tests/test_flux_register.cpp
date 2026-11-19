// V3 lane R, PR R-1: the flux register with no reader. Hydro's F1/F2/F3 are
// the register's stage totals T_d (G-R1 by construction: the producers write
// the same tensors as before), and the step field equals the host FP64
// recompute sum_s w2_s prod_{t>s} w1_t dt T_s over rk3 steps (G-R3).
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <limits>
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

// Straka's cold bubble on an adiabatic rest state: every face carries flux
MeshBlock bubble(Variables& vars) {
  {
    std::ofstream f("test_flux_register.yaml");
    f << kDeck;
  }
  auto block =
      MeshBlock(MeshBlockOptionsImpl::from_yaml("test_flux_register.yaml"));
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

}  // namespace

TEST(RegisterMonitor, census_counts_minima_and_retry_flags) {
  RegisterRequest req;
  req.counters = {"ef.pos_hits", "ef.nan"};
  req.minima = {"ef.pos_min"};
  req.retry = {"ef.floor_patch", "ef.nan"};
  FluxRegister reg;
  reg.reset(2, {1, 1, 4}, {true, false, false}, torch::kFloat64, req);
  auto& m = reg.monitor();
  EXPECT_EQ(m.counters.scalar_type(), torch::kInt64);
  EXPECT_EQ(m.retry_flags.scalar_type(), torch::kInt32);
  m.count("ef.pos_hits", torch::tensor({3, 4}));
  m.count("ef.pos_hits", torch::tensor(2));
  m.record_min("ef.pos_min", torch::tensor({0.5, 0.25}));
  m.record_min("ef.pos_min", torch::tensor(0.75));
  m.flag_retry("ef.nan", torch::tensor({false, true}));
  m.flag_retry("ef.nan", torch::tensor(false));  // an OR: stays set
  EXPECT_EQ(m.counters[0].item<int64_t>(), 9);
  EXPECT_EQ(m.counters[1].item<int64_t>(), 0);
  EXPECT_EQ(m.minima[0].item<double>(), 0.25);
  EXPECT_EQ(m.retry_flags[0].item<int>(), 0);
  EXPECT_EQ(m.retry_flags[1].item<int>(), 1);
  EXPECT_THROW(m.count("ef.unknown", torch::tensor(1)), c10::Error);
  // a rejected attempt leaves nothing behind (RULING V3-2)
  reg.rollback();
  EXPECT_EQ(m.counters.abs().sum().item<int64_t>(), 0);
  EXPECT_EQ(m.retry_flags.abs().sum().item<int>(), 0);
  EXPECT_EQ(m.minima[0].item<double>(),
            std::numeric_limits<double>::infinity());
}

TEST(FluxRegister, hydro_fluxes_are_the_stage_totals) {
  Variables vars;
  auto block = bubble(vars);
  auto& reg = block->phydro->flux_register();
  EXPECT_TRUE(block->phydro->flux1().is_same(reg.total(1)));
  EXPECT_TRUE(block->phydro->flux2().is_same(reg.total(2)));
  EXPECT_FALSE(reg.total(3).defined());  // nx3 = 1
  for (int stage = 0; stage < block->pintg->stages.size(); ++stage)
    block->forward(vars, block->max_time_step(vars), stage);
  // the producers write T_d in place, so the module buffers hold the fluxes
  auto buffers = block->phydro->named_buffers(false);
  EXPECT_TRUE(buffers["F1"].is_same(reg.total(1)));
  EXPECT_GT(reg.total(1).abs().max().item<double>(), 0.);
  EXPECT_GT(reg.total(2).abs().max().item<double>(), 0.);
  // a module move reaches the register (Module::to uses set_data)
  block->to(torch::kCPU, torch::kFloat32);
  EXPECT_EQ(reg.total(1).scalar_type(), torch::kFloat32);
  EXPECT_TRUE(block->phydro->flux1().is_same(reg.total(1)));
}

// R-2: C's riemann solve posts T_d through slot(d, "riemann")
TEST(FluxRegister, the_riemann_solve_posts_its_slot) {
  Variables vars;
  auto block = bubble(vars);
  auto& reg = block->phydro->flux_register();
  EXPECT_EQ(reg.slot_name(1), "");
  block->forward(vars, block->max_time_step(vars), 0);
  for (int d : {1, 2}) {
    EXPECT_EQ(reg.slot_name(d), "riemann") << d;
    EXPECT_TRUE(reg.slot(d, "riemann").is_same(reg.total(d))) << d;
  }
  EXPECT_THROW(reg.slot(3, "riemann"), c10::Error);  // nx3 = 1
  EXPECT_FALSE(reg.keeps(2, "riemann.central"));     // AAM full is off
  EXPECT_THROW(reg.part_slot(2, "riemann.central"), c10::Error);
}

// R-2: a kept part is allocated at reset like T_d, for an active direction
TEST(FluxRegister, kept_parts_are_requested_at_reset) {
  FluxRegister reg;
  RegisterRequest req;
  req.keep = {"riemann.central@2", "riemann.central@3"};
  reg.reset(5, {2, 3, 4}, {true, true, true}, torch::kFloat64, req);
  auto part = reg.part_slot(2, "riemann.central");
  EXPECT_EQ(part.sizes(), reg.total(2).sizes());
  EXPECT_FALSE(part.is_same(reg.total(2)));
  part.fill_(2.5);  // written in place by its producer, read back as kept
  EXPECT_TRUE(reg.kept(2, "riemann.central").is_same(part));
  EXPECT_EQ(reg.kept(2, "riemann.central").sum().item<double>(), 2.5 * 120);
  EXPECT_TRUE(reg.keeps(3, "riemann.central"));
  EXPECT_FALSE(reg.keeps(1, "riemann.central"));
  req.keep = {"riemann.central@1"};  // x1 inactive
  EXPECT_THROW(
      reg.reset(5, {2, 3, 4}, {false, true, true}, torch::kFloat64, req),
      c10::Error);
}

// G-R3: rk3, 10 steps, the recursion against the host FP64 product form
TEST(FluxRegister, step_field_recursion_equals_the_weighted_stage_sum) {
  Variables vars;
  auto block = bubble(vars);
  auto& reg = block->phydro->flux_register();
  reg.request_step_field(1);
  reg.request_step_field(2);
  auto const& stages = block->pintg->stages;
  ASSERT_EQ(stages.size(), 3u);
  double worst = 0.;
  for (int step = 0; step < 10; ++step) {
    double dt = block->max_time_step(vars);
    std::vector<std::array<torch::Tensor, 2>> T;
    for (int s = 0; s < stages.size(); ++s) {
      block->forward(vars, dt, s);
      T.push_back({reg.total(1).to(torch::kFloat64).clone(),
                   reg.total(2).to(torch::kFloat64).clone()});
    }
    for (int d = 0; d < 2; ++d) {
      auto ref = torch::zeros_like(T[0][d]);
      for (int s = 0; s < stages.size(); ++s) {
        double c = stages[s].wght2();
        for (int t = s + 1; t < stages.size(); ++t) c *= stages[t].wght1();
        ref += (c * dt) * T[s][d];
      }
      auto S = reg.step_field(d + 1);
      ASSERT_TRUE(S.defined());
      EXPECT_EQ(S.scalar_type(), torch::kFloat64);
      double scale = ref.abs().max().item<double>();
      ASSERT_GT(scale, 0.);
      double rel = (S - ref).abs().max().item<double>() / scale;
      worst = std::max(worst, rel);
      EXPECT_LE(rel, 1.e-15) << "step " << step << " dim " << d + 1;
    }
    ASSERT_EQ(block->check_redo(vars), 0);
  }
  std::printf("G-R3: worst max|S - sum c_s dt T_s| / max|sum| = %.3e\n", worst);
}
