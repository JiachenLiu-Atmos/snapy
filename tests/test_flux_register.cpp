// V3 lane R, PR R-1: the flux register with no reader. Hydro's F1/F2/F3 are
// the register's stage totals T_d (G-R1 by construction: the producers write
// the same tensors as before), and the step field equals the host FP64
// recompute sum_s w2_s prod_{t>s} w1_t dt T_s over rk3 steps (G-R3).
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <snap/hydro/flux_register.hpp>
#include <snap/layout/seam_register.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/mesh/precision_policy.hpp>

// tests
#include "cuda_test_gate.hpp"

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
  m.flush_counts();  // a mask waits in the stage's batch (V3 S4)
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

// R-3: add() in call order rounds as the pin's in-place updates, E = (E -
// c_adv) - c_sed (flux_positivity.cpp), not E - (c_adv + c_sed)
TEST(FluxRegister, add_in_call_order_keeps_the_pin_rounding) {
  torch::manual_seed(5);
  FluxRegister reg;
  RegisterRequest req;
  req.keep = {"sedimentation@1"};
  reg.reset(5, {1, 2, 64}, {true, false, false}, torch::kFloat64, req);
  auto f0 = torch::randn({5, 1, 2, 64}, torch::kFloat64);
  auto adv = 1.e-3 * torch::randn({1, 1, 2, 64}, torch::kFloat64);
  auto sed = 1.e5 * torch::randn({1, 1, 2, 64}, torch::kFloat64);
  reg.total(1).copy_(f0);
  reg.add(1, "positivity.carry", {IPR}, -adv);
  reg.add(1, "sedimentation", {IPR}, -sed);
  auto pin = f0.clone();
  pin[IPR] -= adv[0];
  pin[IPR] -= sed[0];
  auto fused = f0.clone();
  fused[IPR] -= adv[0] + sed[0];
  EXPECT_TRUE(torch::equal(reg.total(1), pin));
  EXPECT_FALSE(torch::equal(reg.total(1), fused));  // the operands make it so
  // the requested part of the same name holds the increment, other rows 0
  EXPECT_TRUE(torch::equal(reg.kept(1, "sedimentation")[IPR], -sed[0]));
  EXPECT_EQ(reg.kept(1, "sedimentation")[IDN].abs().sum().item<double>(), 0.);
  EXPECT_THROW(reg.add(1, "bad", {IDN, IPR}, adv), c10::Error);
  EXPECT_THROW(reg.add(2, "bad", {IDN}, adv), c10::Error);  // inactive
}

// R-3: scale() multiplies the rows in place, by one factor or one per row
TEST(FluxRegister, scale_multiplies_the_rows_in_place) {
  torch::manual_seed(6);
  FluxRegister reg;
  reg.reset(5, {1, 2, 8}, {true, false, false}, torch::kFloat64, {});
  auto f0 = torch::randn({5, 1, 2, 8}, torch::kFloat64);
  auto th = torch::rand({1, 2, 8}, torch::kFloat64);
  auto th2 = torch::rand({2, 1, 2, 8}, torch::kFloat64);
  reg.total(1).copy_(f0);
  reg.scale(1, "positivity.withhold", {IDN}, th);
  reg.scale(1, "positivity.withhold", {IVY, IPR}, th2);
  auto pin = f0.clone();
  pin[IDN].mul_(th);
  pin[IVY].mul_(th2[0]);
  pin[IPR].mul_(th2[1]);
  EXPECT_TRUE(torch::equal(reg.total(1), pin));
  EXPECT_TRUE(torch::equal(reg.total(1)[IVX], f0[IVX]));  // untouched
  EXPECT_THROW(reg.scale(1, "bad", {IDN}, th2), c10::Error);
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

// R-4 (lane R's ruling): a register reset with the bubble's coordinate, so
// replace and apply_prescribed book sum A_f (after - before) per row
namespace {

struct Seam {
  Variables vars;
  MeshBlock block;
  int nvar;
  Seam() : block(bubble(vars)), nvar(block->phydro->peos->nvar()) {}
  FluxRegister make(RegisterRequest const& req, int seed) {
    FluxRegister reg;
    reg.reset(block->pcoord.get(), PrecisionPolicy("float64"), nvar, req);
    torch::manual_seed(seed);
    for (int d : {1, 2}) reg.total(d).copy_(torch::randn_like(reg.total(d)));
    for (auto& [key, part] : reg.parts()) part.copy_(torch::randn_like(part));
    return reg;
  }
  // A_f of T_d's faces, FP64
  torch::Tensor area(int dim, FluxRegister& reg) {
    auto t = reg.total(dim);
    return dim == 1 ? block->pcoord->face_area1().narrow(2, 0, t.size(3))
                    : block->pcoord->face_area2().narrow(1, 0, t.size(2));
  }
  // the face's dense peer: the total's rows, then the kept parts'
  torch::Tensor peer(FluxRegister& reg, int dim) {
    std::vector<torch::Tensor> rows = {reg.total(dim).clone()};
    for (auto const& part : reg.kept_parts(dim))
      rows.push_back(reg.kept(dim, part).clone());
    return torch::cat(rows, 0);
  }
  // a fresh step, then stage 0 (rk3: w2 = 1), so the step amount is dt a_s
  void stage(FluxRegister& reg) {
    reg.begin_step();
    reg.begin_stage(0, block->pintg->stages[0]);
  }
};

double rel(torch::Tensor const& a, torch::Tensor const& b) {
  return (a - b).abs().max().item<double>() / b.abs().max().item<double>();
}

}  // namespace

TEST(FluxRegister, the_amounts_are_three_flux_corrections) {
  Seam s;
  auto reg = s.make({}, 1);
  auto const& m = reg.monitor();
  EXPECT_EQ(m.amount_names(),
            (std::vector<std::string>{"reconcile.x1", "reconcile.panel",
                                      "boundary.prescribed"}));
  for (auto const& name : m.amount_names()) {
    EXPECT_EQ(m.kind(name), AmountKind::flux_correction) << name;
    EXPECT_EQ(m.amounts(name).sizes(), (std::vector<int64_t>{s.nvar}));
    EXPECT_EQ(m.amounts(name).scalar_type(), torch::kFloat64);
  }
  EXPECT_THROW(m.amounts("vic.cell"), c10::Error);
  // the coordinate reset equals R-1's for hydro's register
  auto& hydro = s.block->phydro->flux_register();
  for (int d : {1, 2, 3}) {
    EXPECT_EQ(reg.total(d).defined(), hydro.total(d).defined()) << d;
    if (reg.total(d).defined())
      EXPECT_EQ(reg.total(d).sizes(), hydro.total(d).sizes()) << d;
  }
  EXPECT_EQ(reg.total(1).scalar_type(), torch::kFloat64);
}

// both sides of an x1 seam call mean: equal faces, bitwise, and the
// amounts book the change each side made
TEST(FluxRegister, replace_mean_gives_both_sides_the_same_faces) {
  Seam s;
  RegisterRequest req;
  req.keep = {"pressure_base1@1", "riemann.central@2"};
  auto a = s.make(req, 2), b = s.make(req, 3);
  ASSERT_EQ(a.kept_parts(1), (std::vector<std::string>{"pressure_base1"}));
  int nc1 = a.total(1).size(3);
  auto faces = torch::zeros(a.total(1).sizes().slice(1), torch::kBool);
  faces.select(2, 3).fill_(true);  // a's lower x1 face, il
  auto a0 = a.total(1).clone(), a2 = a.total(2).clone();
  auto pa = s.peer(a, 1), pb = s.peer(b, 1);
  s.stage(a);
  s.stage(b);
  a.replace(1, faces, pb, SeamRule::mean);
  b.replace(1, faces, pa, SeamRule::mean);
  EXPECT_TRUE(torch::equal(a.total(1).select(3, 3), b.total(1).select(3, 3)));
  EXPECT_TRUE(torch::equal(a.kept(1, "pressure_base1").select(3, 3),
                           b.kept(1, "pressure_base1").select(3, 3)));
  EXPECT_TRUE(torch::equal(a.total(1).narrow(3, 4, nc1 - 4),
                           a0.narrow(3, 4, nc1 - 4)));  // unmasked: untouched
  EXPECT_TRUE(torch::equal(a.total(2), a2));            // another direction
  double dt = 0.7;
  a.end_stage(dt);
  b.end_stage(dt);
  auto area = s.area(1, a);
  auto ref = dt * (area * (a.total(1) - a0)).sum({1, 2, 3});
  auto got = a.monitor().amounts("reconcile.x1");
  EXPECT_LE(rel(got, ref), 1.e-15);
  EXPECT_GT(ref.abs().max().item<double>(), 0.);
  // a mean moves both sides by half the jump, in opposite directions
  auto other = b.monitor().amounts("reconcile.x1");
  EXPECT_LE(
      (got + other).abs().max().item<double>() / ref.abs().max().item<double>(),
      1.e-14);
  EXPECT_EQ(a.monitor().amounts("reconcile.panel").abs().sum().item<double>(),
            0.);
}

// on a panel seam only the non-owner calls owner_copy; its faces become the
// owner's mapped slab, and the change is booked as reconcile.panel
TEST(FluxRegister, replace_owner_copy_takes_the_peer_on_the_mask) {
  Seam s;
  auto owner = s.make({}, 4), other = s.make({}, 5);
  auto faces = torch::zeros(owner.total(2).sizes().slice(1), torch::kBool);
  faces.select(1, 3).narrow(1, 3, 32).fill_(true);  // jl, interior i
  auto before = other.total(2).clone();
  auto mine = owner.total(2).clone();
  s.stage(other);
  other.replace(2, faces, s.peer(owner, 2), SeamRule::owner_copy);
  EXPECT_TRUE(torch::equal(other.total(2), mine.where(faces, before)));
  EXPECT_TRUE(torch::equal(owner.total(2), mine));  // the owner is untouched
  other.end_stage(1.);
  auto ref = (s.area(2, other) * torch::where(faces, mine - before, 0.))
                 .sum({1, 2, 3});
  EXPECT_LE(rel(other.monitor().amounts("reconcile.panel"), ref), 1.e-15);
  EXPECT_EQ(other.monitor().amounts("reconcile.x1").abs().sum().item<double>(),
            0.);
  // the layout is checked: a mask with a row axis, a peer without its parts
  EXPECT_THROW(other.replace(2, faces.unsqueeze(0), s.peer(owner, 2),
                             SeamRule::owner_copy),
               c10::Error);
  EXPECT_THROW(other.replace(2, faces.to(torch::kFloat64), s.peer(owner, 2),
                             SeamRule::owner_copy),
               c10::Error);
  EXPECT_THROW(other.replace(2, faces, s.peer(owner, 2).narrow(0, 0, 2),
                             SeamRule::owner_copy),
               c10::Error);
  // R-1's reset has no coordinate, so nothing can be booked
  FluxRegister plain;
  plain.reset(s.nvar, {1, 38, 38}, {true, true, false}, torch::kFloat64, {});
  EXPECT_THROW(
      plain.replace(2, faces, plain.total(2).clone(), SeamRule::owner_copy),
      c10::Error);
}

TEST(FluxRegister, apply_prescribed_writes_the_value_and_books_the_change) {
  Seam s;
  auto reg = s.make({}, 6);
  s.stage(reg);
  auto t0 = reg.total(1).clone();
  reg.apply_prescribed(1);  // S set no mask: a no-op
  EXPECT_TRUE(torch::equal(reg.total(1), t0));
  // the wall faces at il and iu + 1, mass and energy rows; value 0
  auto mask = torch::zeros_like(t0, torch::kBool);
  int nc1 = t0.size(3);
  for (int r : {int(IDN), int(IPR)}) {
    mask[r].select(2, 3).fill_(true);
    mask[r].select(2, nc1 - 3).fill_(true);
  }
  auto value = torch::zeros_like(t0);
  reg.set_prescribed(1, mask, value);
  reg.apply_prescribed(1);
  EXPECT_TRUE(torch::equal(reg.total(1), value.where(mask, t0)));
  reg.end_stage(2.);
  auto ref =
      2. * (s.area(1, reg) * torch::where(mask, value - t0, 0.)).sum({1, 2, 3});
  auto got = reg.monitor().amounts("boundary.prescribed");
  EXPECT_LE(rel(got, ref), 1.e-15);
  EXPECT_EQ(got[IVX].item<double>(), 0.);  // an unmasked row books nothing
  // the second call (start of post_implicit) changes nothing more
  s.stage(reg);
  reg.apply_prescribed(1);
  reg.end_stage(2.);
  EXPECT_EQ(
      reg.monitor().amounts("boundary.prescribed").abs().sum().item<double>(),
      0.);
  EXPECT_THROW(reg.set_prescribed(1, mask[0], value), c10::Error);
}

// R.rst 4.1 item 4: kept parts are reconciled and prescribed but never
// scaled. apply_prescribed writes S's value on every kept part of dim, with
// the total's mask; only the total's change is booked
TEST(FluxRegister, apply_prescribed_prescribes_the_kept_parts) {
  Seam s;
  RegisterRequest req;
  req.keep = {"vic.mass_total@1", "riemann.central@1", "riemann.central@2"};
  auto reg = s.make(req, 8);
  ASSERT_EQ(reg.kept_parts(1),
            (std::vector<std::string>{"vic.mass_total", "riemann.central"}));
  s.stage(reg);
  auto t0 = reg.total(1).clone();
  auto p0 = reg.kept(1, "vic.mass_total").clone();
  auto c0 = reg.kept(1, "riemann.central").clone();
  auto c2 = reg.kept(2, "riemann.central").clone();
  auto mask = torch::zeros_like(t0, torch::kBool);
  int nc1 = t0.size(3);
  for (int r : {int(IDN), int(IPR)}) {
    mask[r].select(2, 3).fill_(true);
    mask[r].select(2, nc1 - 3).fill_(true);
  }
  auto value = torch::zeros_like(t0);
  reg.set_prescribed(1, mask, value);
  reg.apply_prescribed(1);
  EXPECT_TRUE(torch::equal(reg.total(1), value.where(mask, t0)));
  EXPECT_TRUE(
      torch::equal(reg.kept(1, "vic.mass_total"), value.where(mask, p0)));
  EXPECT_TRUE(
      torch::equal(reg.kept(1, "riemann.central"), value.where(mask, c0)));
  EXPECT_TRUE(torch::equal(reg.kept(2, "riemann.central"), c2));  // dim 2
  // the kept parts change no amount: the booking is the total's alone
  reg.end_stage(1.);
  auto ref =
      (s.area(1, reg) * torch::where(mask, value - t0, 0.)).sum({1, 2, 3});
  EXPECT_LE(rel(reg.monitor().amounts("boundary.prescribed"), ref), 1.e-15);
}

// V3 E: without a solid, S's boundary.prescribed covers the wall block
// faces only, so the register works on the wall face planes alone. A face
// off the walls keeps its bits and never enters the booking: a non-finite
// flux there leaves "boundary.prescribed" finite
TEST(FluxRegister, wall_prescription_works_on_the_wall_planes_only) {
  Seam s;
  RegisterRequest req;
  req.keep = {"riemann.central@1"};
  auto reg = s.make(req, 9);
  s.stage(reg);
  auto c = s.block->pcoord;
  int il = c->il(), iw = c->iu() + 1;
  // an x1 face between two interior cells, off both walls
  reg.total(1)[IDN].select(2, il + 2).select(1, c->jl()).fill_(NAN);
  auto t0 = reg.total(1).clone();
  auto k0 = reg.kept(1, "riemann.central").clone();
  supply_prescribed(reg, s.block.get(), torch::Tensor(),
                    {IVX, IVX + 1, IVX + 2});
  reg.apply_prescribed(1);
  auto mask = torch::zeros_like(t0, torch::kBool);
  for (int r = 0; r < t0.size(0); ++r) {
    if (r >= IVX && r <= IVX + 2) continue;  // free slip: momentum keeps
    mask[r].select(2, il).fill_(true);
    mask[r].select(2, iw).fill_(true);
  }
  auto same = [](torch::Tensor const& a, torch::Tensor const& b) {
    return torch::allclose(a, b, 0., 0., /*equal_nan=*/true);
  };
  EXPECT_TRUE(same(reg.total(1), torch::where(mask, 0., t0)));
  EXPECT_TRUE(same(reg.kept(1, "riemann.central"), torch::where(mask, 0., k0)));
  reg.end_stage(1.);
  auto got = reg.monitor().amounts("boundary.prescribed");
  auto ref = (s.area(1, reg) * torch::where(mask, -t0, 0.)).sum({1, 2, 3});
  EXPECT_TRUE(torch::isfinite(got).all().item<bool>())
      << "a face off the walls entered the booking: " << got;
  EXPECT_LE(rel(got, ref), 1.e-15);
}

// A float32 register on the device (CUDA when available): apply_prescribed
// builds its signed row weights in the wide dtype on the total's device. A
// where() with two Scalar operands would build them as float64 scalar
// tensors on that device, which a device without float64 (MPS) refuses; the
// booking must equal the FP64 reference either way
TEST(FluxRegister, wall_prescription_books_in_float32_on_the_device) {
  auto device = snapy_cuda_test_enabled() ? torch::Device(torch::kCUDA, 0)
                                          : torch::Device(torch::kCPU);
  Variables vars;
  auto block = bubble(vars);
  block->to(device, torch::kFloat32);
  auto& reg = block->phydro->flux_register();
  reg.set_wall_batch(nullptr);  // book at the call, as a register alone does
  ASSERT_EQ(reg.total(1).scalar_type(), torch::kFloat32);
  ASSERT_EQ(reg.total(1).device(), device);
  torch::manual_seed(11);
  reg.total(1).copy_(torch::randn_like(reg.total(1)));
  reg.begin_step();
  reg.begin_stage(0, block->pintg->stages[0]);
  auto c = block->pcoord;
  int il = c->il(), iw = c->iu() + 1;
  auto t0 = reg.total(1).clone();
  supply_prescribed(reg, block.get(), torch::Tensor(), {IVX, IVX + 1, IVX + 2});
  reg.apply_prescribed(1);
  auto mask = torch::zeros_like(t0, torch::kBool);
  for (int r = 0; r < t0.size(0); ++r) {
    if (r >= IVX && r <= IVX + 2) continue;  // free slip: momentum keeps
    mask[r].select(2, il).fill_(true);
    mask[r].select(2, iw).fill_(true);
  }
  EXPECT_TRUE(torch::equal(reg.total(1), torch::where(mask, 0.f, t0)));
  reg.end_stage(1.);
  auto got = reg.monitor().amounts("boundary.prescribed").cpu();
  auto area = c->face_area1().narrow(2, 0, t0.size(3)).to(torch::kFloat64);
  auto ref = (area * torch::where(mask, -t0.to(torch::kFloat64), 0.))
                 .sum({1, 2, 3})
                 .cpu();
  EXPECT_GT(ref.abs().max().item<double>(), 0.);
  EXPECT_LE(rel(got, ref), 1.e-12);
}

// reset() starts with no prescription: S's wall planes from before the
// reset are gone, as its full mask is, so apply_prescribed is a no-op until
// S supplies again
TEST(FluxRegister, reset_drops_the_wall_planes) {
  Seam s;
  auto reg = s.make({}, 10);
  s.stage(reg);
  supply_prescribed(reg, s.block.get(), torch::Tensor(),
                    {IVX, IVX + 1, IVX + 2});
  reg.reset(s.block->pcoord.get(), PrecisionPolicy("float64"), s.nvar, {});
  for (int d : {1, 2}) reg.total(d).copy_(torch::randn_like(reg.total(d)));
  s.stage(reg);
  auto t0 = reg.total(1).clone();
  reg.apply_prescribed(1);
  EXPECT_TRUE(torch::equal(reg.total(1), t0))
      << "apply_prescribed after reset() changed the total";
  reg.end_stage(1.);
  EXPECT_EQ(
      reg.monitor().amounts("boundary.prescribed").abs().sum().item<double>(),
      0.);
}

// G-R14 (flux corrections only): the amounts' recursion equals
// sum_s w2_s prod_{t>s} w1_t dt a_s over rk3; a rollback clears them
TEST(FluxRegister, amounts_fold_like_the_step_field) {
  Seam s;
  auto reg = s.make({}, 7);
  auto const& stages = s.block->pintg->stages;
  auto faces = torch::zeros(reg.total(1).sizes().slice(1), torch::kBool);
  faces.select(2, 3).fill_(true);
  double dt = 0.37, worst = 0.;
  for (int step = 0; step < 3; ++step) {
    reg.begin_step();
    auto ref = torch::zeros({s.nvar}, torch::kFloat64);
    for (int st = 0; st < stages.size(); ++st) {
      reg.begin_stage(st, stages[st]);
      auto before = reg.total(1).clone();
      auto peer = torch::randn_like(before);
      reg.replace(1, faces, peer, SeamRule::mean);
      auto a = (s.area(1, reg) * (reg.total(1) - before)).sum({1, 2, 3});
      double c = stages[st].wght2();
      for (int t = st + 1; t < stages.size(); ++t) c *= stages[t].wght1();
      ref += (c * dt) * a;
      reg.end_stage(dt);
    }
    double r = rel(reg.monitor().amounts("reconcile.x1"), ref);
    worst = std::max(worst, r);
    EXPECT_LE(r, 1.e-15) << step;
  }
  std::printf("G-R14 (reconcile.x1): worst relative difference %.3e\n", worst);
  reg.rollback();
  EXPECT_EQ(reg.monitor().amounts("reconcile.x1").abs().sum().item<double>(),
            0.);
}

// V3 S4 (ruling 2026-11-24, condition b): a rejected attempt leaves the
// per-stage slots as if it never ran, so the redone step's amounts are those
// of a clean step with the redo's dt, bitwise
TEST(FluxRegister, a_rolled_back_attempt_leaves_the_amounts_of_the_redo) {
  Seam s;
  auto const& stages = s.block->pintg->stages;
  auto a = s.make({}, 7), b = s.make({}, 7);
  auto faces = torch::zeros(a.total(1).sizes().slice(1), torch::kBool);
  faces.select(2, 3).fill_(true);
  auto run = [&](FluxRegister& reg, double dt, int seed, int nstages) {
    torch::manual_seed(seed);
    for (int st = 0; st < nstages; ++st) {
      reg.begin_stage(st, stages[st]);
      reg.replace(1, faces, torch::randn_like(reg.total(1)), SeamRule::mean);
      reg.end_stage(dt);
    }
  };
  a.begin_step();
  run(a, 0.41, 11, 2);  // the attempt: two stages in, then rejected
  a.rollback();
  a.total(1).copy_(b.total(1));  // the redo restores the step's state
  run(a, 0.29, 23, stages.size());
  b.begin_step();
  run(b, 0.29, 23, stages.size());  // a clean step with the redo's dt
  auto ra = a.monitor().amounts("reconcile.x1");
  auto rb = b.monitor().amounts("reconcile.x1");
  EXPECT_GT(rb.abs().sum().item<double>(), 0.);
  EXPECT_TRUE(torch::equal(ra, rb)) << ra << " vs " << rb;
}

// R-5: keep writes a computed one-quantity part on row IDN of its nvar-row
// part; the other rows stay zero
TEST(FluxRegister, keep_writes_row_idn_of_the_part) {
  FluxRegister reg;
  RegisterRequest req;
  req.keep = {"vic.mass_total@1"};
  reg.reset(5, {1, 2, 8}, {true, false, false}, torch::kFloat64, req);
  torch::manual_seed(8);
  auto f = torch::randn({1, 2, 8}, torch::kFloat64);
  reg.keep(1, "vic.mass_total", f);
  auto part = reg.kept(1, "vic.mass_total");
  EXPECT_TRUE(torch::equal(part[IDN], f));
  EXPECT_EQ(part.narrow(0, 1, 4).abs().sum().item<double>(), 0.);
  reg.keep(1, "vic.mass_total", 2. * f.unsqueeze(0));  // [1, faces...] too
  EXPECT_TRUE(torch::equal(part[IDN], 2. * f));
  EXPECT_THROW(reg.keep(1, "vic.mass_total", f.narrow(-1, 0, 4)), c10::Error);
  EXPECT_THROW(reg.keep(1, "vic.mass_dry", f), c10::Error);  // not requested
}

// R-5: a cell_amount name is created at its first post, with its kind; a
// later post with another kind is an error
TEST(FluxRegister, cell_amount_names_are_made_at_their_first_post) {
  FluxRegister reg;
  reg.reset(5, {1, 2, 8}, {true, false, false}, torch::kFloat64, {});
  auto& m = reg.monitor();
  ASSERT_EQ(m.amount_names().size(), 3u);
  reg.begin_step();
  auto v = torch::arange(5 * 16, torch::kFloat64).view({5, 1, 2, 8});
  reg.cell_amount("vic.cell", v, 1., AmountKind::cell_source);
  ASSERT_EQ(m.amount_names().size(), 4u);
  EXPECT_EQ(m.amount_names()[3], "vic.cell");
  EXPECT_EQ(m.kind("vic.cell"), AmountKind::cell_source);
  // outside a stage it goes straight into the step amount, summed per row
  EXPECT_TRUE(torch::equal(m.amounts("vic.cell"), v.sum({1, 2, 3})));
  reg.cell_amount("vic.cell", v.sum({1, 2, 3}), 1., AmountKind::cell_source);
  EXPECT_TRUE(torch::equal(m.amounts("vic.cell"), 2. * v.sum({1, 2, 3})));
  EXPECT_THROW(reg.cell_amount("vic.cell", v, 1., AmountKind::floor),
               c10::Error);
  EXPECT_THROW(reg.cell_amount("reconcile.x1", v, 1., AmountKind::cell_source),
               c10::Error);  // R-4's names are flux corrections
  EXPECT_THROW(reg.cell_amount("bad", v.narrow(0, 0, 3), 1., AmountKind::floor),
               c10::Error);
  EXPECT_EQ(m.amount_names().size(), 5u);  // "bad" was made, then refused
  reg.rollback();
  EXPECT_EQ(m.amounts("vic.cell").abs().sum().item<double>(), 0.);
}

// G-R14 for cell amounts (R.rst 3.4): a stage's posts with stage_weight
// w2_s and after_combine posts with 1.0 fold as A <- w1 A + sum w_k a_k, so
// the step amount is sum_s prod_{t>s} w1_t (w2_s a_s + b_s)
TEST(FluxRegister, cell_amounts_fold_with_their_stage_weight) {
  Seam s;
  FluxRegister reg;
  reg.reset(s.nvar, {1, 2, 8}, {true, false, false}, torch::kFloat64, {});
  auto const& stages = s.block->pintg->stages;
  torch::manual_seed(9);
  double worst = 0.;
  for (int step = 0; step < 3; ++step) {
    reg.begin_step();
    auto ref = torch::zeros({s.nvar}, torch::kFloat64);
    for (int st = 0; st < stages.size(); ++st) {
      reg.begin_stage(st, stages[st]);
      auto a = torch::randn({s.nvar, 1, 2, 8}, torch::kFloat64);
      reg.cell_amount("vic.cell", a, stages[st].wght2(),
                      AmountKind::cell_source);
      reg.end_stage(0.37);
      auto b = torch::randn({s.nvar}, torch::kFloat64);  // after_combine
      reg.cell_amount("fixer", b, 1., AmountKind::fixer);
      double later = 1.;
      for (int t = st + 1; t < stages.size(); ++t) later *= stages[t].wght1();
      ref += later * (stages[st].wght2() * a.sum({1, 2, 3}) + b);
    }
    auto got =
        reg.monitor().amounts("vic.cell") + reg.monitor().amounts("fixer");
    double r = rel(got, ref);
    worst = std::max(worst, r);
    EXPECT_LE(r, 1.e-15) << step;
    EXPECT_EQ(reg.monitor().amounts("reconcile.x1").abs().sum().item<double>(),
              0.);
  }
  std::printf("G-R14 (cell amounts): worst relative difference %.3e\n", worst);
}

// R-5 on R-4.rows: apply_prescribed(1) prescribes lane I's kept x1 parts on
// every row the mask sets (IDN and IPR here), with the total's value, unbooked
TEST(FluxRegister, apply_prescribed_sets_every_masked_row_of_lane_i_parts) {
  Seam s;
  RegisterRequest req;
  req.keep = {"vic.mass_total@1", "vic.momentum_t@1"};
  auto reg = s.make(req, 10);
  s.stage(reg);
  auto t0 = reg.total(1).clone();
  auto m0 = reg.kept(1, "vic.mass_total").clone();
  auto p0 = reg.kept(1, "vic.momentum_t").clone();
  // IDN at the lower wall face, IPR at the upper: both rows apply
  auto mask = torch::zeros_like(t0, torch::kBool);
  int nc1 = t0.size(3);
  mask[IDN].select(2, 3).fill_(true);
  mask[IPR].select(2, nc1 - 3).fill_(true);
  auto value = torch::full_like(t0, 0.25);
  reg.set_prescribed(1, mask, value);
  reg.apply_prescribed(1);
  for (auto const& [name, before] :
       {std::pair{"vic.mass_total", m0}, std::pair{"vic.momentum_t", p0}}) {
    ASSERT_FALSE(
        torch::equal(before.masked_select(mask), value.masked_select(mask)))
        << name;
    auto part = reg.kept(1, name);
    EXPECT_TRUE(
        torch::equal(part.masked_select(mask), value.masked_select(mask)))
        << name;
  }
  reg.end_stage(1.);
  auto ref =
      (s.area(1, reg) * torch::where(mask, value - t0, 0.)).sum({1, 2, 3});
  EXPECT_LE(rel(reg.monitor().amounts("boundary.prescribed"), ref), 1.e-15);
}

// R-5: hydro requests lane I's vic parts with the implicit solve and
// mass.dry.limited@d with scalars; the scalars' flux read is that part, row
// IDN of the total as limit ends, the same values as the total after the
// hydro forward (the switch is bitwise until I-4 posts into total(1))
TEST(FluxRegister, hydro_keeps_the_parts_its_readers_need) {
  Variables plain;
  auto bub = bubble(plain);
  auto& r0 = bub->phydro->flux_register();
  EXPECT_FALSE(r0.keeps(1, "vic.mass_total"));
  EXPECT_FALSE(r0.keeps(1, "mass.dry.limited"));

  std::string deck = kDeck;
  auto at = deck.find("implicit-scheme: 0");
  ASSERT_NE(at, std::string::npos);
  deck.replace(at, 18, "implicit-scheme: 1");
  deck += "scalar: {nvar: 1, names: [trace], reconstruct: {type: weno5}}\n";
  {
    std::ofstream f("test_flux_register_scalar.yaml");
    f << deck;
  }
  auto block = MeshBlock(
      MeshBlockOptionsImpl::from_yaml("test_flux_register_scalar.yaml"));
  block->to(torch::kCPU, torch::kFloat64);
  auto& reg = block->phydro->flux_register();
  for (auto const* p : {"vic.mass_total", "vic.mass_dry", "vic.momentum_t"})
    EXPECT_TRUE(reg.keeps(1, p)) << p;
  EXPECT_TRUE(reg.keeps(1, "mass.dry.limited"));
  EXPECT_TRUE(reg.keeps(2, "mass.dry.limited"));
  EXPECT_FALSE(reg.keeps(3, "mass.dry.limited"));  // nx3 = 1

  Variables vars;
  vars["hydro_w"] = plain.at("hydro_w").clone();
  vars["scalar_r"] = torch::full_like(vars["hydro_w"].narrow(0, 0, 1), 1.e-3);
  block->initialize(vars);
  double dt = block->max_time_step(vars);
  for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
    block->forward(vars, dt, stage);
    for (int d : {1, 2}) {
      auto part = reg.kept(d, "mass.dry.limited");
      if (d == 1) {  // I-4: total(1) also holds the implicit dry transfer
        auto rest = reg.total(1)[IDN] - reg.kept(1, "vic.mass_dry")[IDN];
        EXPECT_LE((part[IDN] - rest).abs().max().item<double>(),
                  1.e-15 * reg.total(1)[IDN].abs().max().item<double>());
      } else {
        EXPECT_TRUE(torch::equal(part[IDN], reg.total(d)[IDN])) << d;
      }
      EXPECT_EQ(part.narrow(0, 1, part.size(0) - 1).abs().sum().item<double>(),
                0.)
          << d;
    }
  }
  EXPECT_GT(reg.kept(1, "mass.dry.limited").abs().max().item<double>(), 0.);
  EXPECT_GT(vars.at("scalar_s").abs().max().item<double>(), 0.);
}

// V3 I-4: the vertical solve posts its face transfers to the register as
// M_f / (A_f dt). The kept vic.mass_total and vic.mass_dry give back the
// solve's moved masses, the vic.momentum_t posts telescope to the solve's
// tangential cell increments, and G-R11: m1's A_f total(1) equals the pin's
// Phi = A_f F_explicit + mc / dt to round-off (F_explicit is the scalars'
// mass.dry.limited@1, total(1)[IDN] as limit ends; the deck is dry). The
// G-R11 rounding is in ulp of the larger operand, max(|A F|, |mc / dt|):
// a face where the two nearly cancel keeps their absolute error
TEST(FluxRegister, the_vertical_solve_posts_its_face_transfers) {
  for (int scheme : {1, 9}) {
    std::string deck = kDeck;
    auto at = deck.find("implicit-scheme: 0");
    ASSERT_NE(at, std::string::npos);
    deck.replace(at, 18, "implicit-scheme: " + std::to_string(scheme));
    deck += "scalar: {nvar: 1, names: [trace], reconstruct: {type: weno5}}\n";
    {
      std::ofstream f("test_flux_register_vic.yaml");
      f << deck;
    }
    Variables plain;
    auto bub = bubble(plain);
    auto block = MeshBlock(
        MeshBlockOptionsImpl::from_yaml("test_flux_register_vic.yaml"));
    block->to(torch::kCPU, torch::kFloat64);
    Variables vars;
    vars["hydro_w"] = plain.at("hydro_w").clone();
    vars["scalar_r"] = torch::full_like(vars["hydro_w"].narrow(0, 0, 1), 1e-3);
    block->initialize(vars);
    auto& reg = block->phydro->flux_register();
    auto co = block->pcoord;
    int nc1 = co->options->nc1(), is = co->il(), ie = co->iu() + 1;
    auto A = co->face_area1().narrow(-1, 0, nc1);
    auto vol = co->cell_volume();
    auto rel = [](torch::Tensor const& a, torch::Tensor const& b) {
      return (a - b).abs().max().item<double>() / b.abs().max().item<double>();
    };
    double worst_ulp = 0., worst_m = 0., worst_t = 0.;
    for (int step = 0; step < 2; ++step) {
      double dt = block->max_time_step(vars);
      for (int st = 0; st < block->pintg->stages.size(); ++st) {
        block->forward(vars, dt, st);
        auto mc = block->phydro->picorr->mass_correction();
        ASSERT_GT(mc[IVZ].abs().max().item<double>(), 0.) << scheme;
        // the kept parts give back the moved masses
        auto total = reg.kept(1, "vic.mass_total")[IDN] * A * dt;
        auto dry = reg.kept(1, "vic.mass_dry")[IDN] * A * dt;
        worst_m = std::max({worst_m, rel(total, mc[IVZ]), rel(dry, mc[IVY])});
        // G-R11
        auto phi_new = A * reg.total(1)[IDN];
        auto phi_old = A * reg.kept(1, "mass.dry.limited")[IDN] + mc[IVZ] / dt;
        auto opscale =
            torch::maximum((A * reg.kept(1, "mass.dry.limited")[IDN]).abs(),
                           (mc[IVZ] / dt).abs());
        auto ulp = ((phi_new - phi_old).abs() /
                    (opscale * std::ldexp(1., -52)).clamp_min(1e-300))
                       .slice(-1, is, ie + 1);
        worst_ulp = std::max(worst_ulp, ulp.max().item<double>());
        // the tangential posts telescope to the solve's cell increments
        auto P = reg.kept(1, "vic.momentum_t").narrow(0, IVY, 2) * A * dt;
        auto div = (P.slice(-1, is, ie) - P.slice(-1, is + 1, ie + 1))
                       .slice(-2, co->jl(), co->ju() + 1);
        auto inc = (block->phydro->picorr->correction().narrow(0, IVY, 2) * vol)
                       .slice(-1, is, ie)
                       .slice(-2, co->jl(), co->ju() + 1);
        // relative to the transfers. Partial matrix: correction() is
        // (du0 + sum p / V) - du0 and keeps du0's rounding, |du0 V| ~ 1e3 |P|
        // here (measured with du0 exposed: <= 0.93 ulp of max(|du0 V|,
        // |inc|)); the full matrix posts are built from correction() itself
        double pmax = P.abs().max().item<double>();
        if (pmax > 0.)
          worst_t =
              std::max(worst_t, (div - inc).abs().max().item<double>() / pmax);
      }
      ASSERT_EQ(block->check_redo(vars), 0);
    }
    std::printf(
        "I-4 scheme %d: G-R11 worst |A total(1) - Phi| %.2f ulp; kept masses "
        "rel %.3e; momentum_t telescoping / max|P| %.3e\n",
        scheme, worst_ulp, worst_m, worst_t);
    EXPECT_LE(worst_ulp, 4.) << scheme;
    EXPECT_LE(worst_m, 1.e-14) << scheme;
    EXPECT_LE(worst_t, scheme == 1 ? 1.e-11 : 1.e-14) << scheme;
    auto const& names = reg.monitor().amount_names();
    auto kind_of = [&](char const* n) {
      bool has = std::find(names.begin(), names.end(), n) != names.end();
      EXPECT_TRUE(has) << n << " scheme " << scheme;
      return has ? reg.monitor().kind(n) : AmountKind::floor;
    };
    EXPECT_EQ(kind_of("vic.cell"), AmountKind::cell_source);
    EXPECT_EQ(kind_of("vic.mass_clamped"), AmountKind::flux_correction);
    if (scheme == 9)
      EXPECT_EQ(kind_of("vic.momentum_t.top"), AmountKind::diagnostic);
  }
}

// V3 I-5: the hydro register declares lane I's retry cause implicit.column
// (R.rst 4.5) and hydro flags it with the column status OR; slot 5 of the
// redo read, solve_failed_tensor(), is the same OR until lane X reads
// retry_flags
TEST(FluxRegister, a_failed_column_flags_implicit_column) {
  std::string deck = kDeck;
  auto at = deck.find("implicit-scheme: 0");
  ASSERT_NE(at, std::string::npos);
  deck.replace(at, 18, "implicit-scheme: 1");
  {
    std::ofstream f("test_flux_register_i5.yaml");
    f << deck;
  }
  Variables plain;
  auto bub = bubble(plain);
  auto block =
      MeshBlock(MeshBlockOptionsImpl::from_yaml("test_flux_register_i5.yaml"));
  block->to(torch::kCPU, torch::kFloat64);
  auto& mon = block->phydro->flux_register().monitor();
  int slot = mon.retry_index("implicit.column");
  auto& ic = block->phydro->picorr;
  for (bool bad : {false, true}) {
    Variables vars;
    vars["hydro_w"] = plain.at("hydro_w").clone();
    block->initialize(vars);
    double dt = block->max_time_step(vars);
    int j = block->pcoord->jl() + 5;
    if (bad)  // one column's state: its x1 sweep and solve see a NaN
      vars.at("hydro_u")[IPR].select(-2, j).fill_(
          std::numeric_limits<double>::quiet_NaN());
    block->forward(vars, dt, 0);
    EXPECT_EQ(mon.retry_flags[slot].item<int>(), bad ? 1 : 0) << bad;
    EXPECT_EQ(ic->status().item<bool>(), bad);
    EXPECT_EQ(ic->solve_failed_tensor(torch::kFloat64).item<double>(),
              bad ? 1. : 0.);
    if (bad) {
      auto st = ic->col_status();
      EXPECT_NE(st[0][j].item<int>(), 0);
    }
  }
}

// V3 I-7: gravity-work: cell is removed. A deck that selects cell is
// refused, from YAML and from code, with a message that names face. The
// global fixer went in V3 X5: gravity-work-fixer: true is refused with X5's
// message, false still loads.
TEST(GravityWork, the_cell_form_is_refused_and_names_face) {
  auto refused = [](std::function<void()> f) {
    try {
      f();
    } catch (c10::Error const& e) {
      return std::string(e.what_without_backtrace());
    }
    return std::string();
  };
  auto deck = [](std::string const& gravity) {
    std::string d = kDeck;
    auto at = d.find("const-gravity: {grav1: -9.8}");
    EXPECT_NE(at, std::string::npos);
    d.replace(at, 28, "const-gravity: {grav1: -9.8, " + gravity + "}");
    std::ofstream f("test_flux_register_gw.yaml");
    f << d;
    return std::string("test_flux_register_gw.yaml");
  };
  // from YAML
  auto msg = refused(
      [&] { MeshBlockOptionsImpl::from_yaml(deck("gravity-work: cell")); });
  EXPECT_NE(msg.find("gravity-work: face"), std::string::npos) << msg;
  msg = refused([&] {
    MeshBlockOptionsImpl::from_yaml(deck("gravity-work-fixer: true"));
  });
  EXPECT_NE(msg.find("removed in V3"), std::string::npos) << msg;
  EXPECT_EQ(
      refused([&] {
        MeshBlockOptionsImpl::from_yaml(deck("gravity-work-fixer: false"));
      }),
      "");
  // from code: refused at Hydro construction
  auto opts = MeshBlockOptionsImpl::from_yaml(deck("gravity-work: face"));
  opts->hydro()->grav()->gravity_work("cell");
  msg = refused([&] { auto block = MeshBlock(opts); });
  EXPECT_NE(msg.find("gravity-work: face"), std::string::npos) << msg;
}

// R-6 follow-up (R.rst 3.4, skeleton R:79-81): a radial post_split lands in
// a direct part of S_d that no later w1 scales. The step-start post (before
// stage 0, after begin_step) and the step-end post (after the last
// end_stage) both enter with their own weight; begin_step clears them.
TEST(FluxRegister, post_split_is_never_scaled_by_a_later_w1) {
  Seam s;
  RegisterRequest req;
  req.step_field = {1};
  auto reg = s.make(req, 7);
  auto plain = s.make(req, 7);  // the same stages without the split posts
  auto const& stages = s.block->pintg->stages;
  ASSERT_EQ(stages.size(), 3u);
  double const dt = 0.37;
  torch::manual_seed(11);
  auto lo = torch::randn_like(reg.total(1));
  auto hi = torch::randn_like(reg.total(1));

  EXPECT_THROW(reg.post_split(2, "diffusion.radial", lo, 0.5, dt),
               c10::Error);  // no step field for x2
  reg.begin_step();
  plain.begin_step();
  reg.post_split(1, "diffusion.radial", lo, 0.5, dt);  // the step-start half
  for (int st = 0; st < stages.size(); ++st) {
    auto t = torch::randn_like(reg.total(1));
    for (auto* r : {&reg, &plain}) {
      r->begin_stage(st, stages[st]);
      r->total(1).copy_(t);
    }
    EXPECT_THROW(reg.post_split(1, "diffusion.radial", hi, 0.5, dt),
                 c10::Error);  // a split post is never inside a stage
    reg.end_stage(dt);
    plain.end_stage(dt);
  }
  reg.post_split(1, "diffusion.radial", hi, 0.5, dt);  // the step-end half

  // S_d = direct + sum_s w2_s prod_{t>s} w1_t dt T_s, the recursion part
  // computed exactly as without the posts
  auto direct = torch::zeros_like(plain.step_field(1));
  direct.add_(lo.to(direct.scalar_type()), 0.5 * dt);
  direct.add_(hi.to(direct.scalar_type()), 0.5 * dt);
  EXPECT_TRUE(torch::equal(reg.step_field(1), plain.step_field(1) + direct));
  EXPECT_GT(direct.abs().max().item<double>(), 0.);

  // a redone step starts from begin_step: the posts are gone
  reg.begin_step();
  EXPECT_EQ(reg.step_field(1).abs().max().item<double>(), 0.);
}
