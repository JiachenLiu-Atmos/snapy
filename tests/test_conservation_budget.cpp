// Conservation budget on closed decks to C n eps, every step: mass, species,
// E + P (face work; P the corrected PE) or E + rho phi (cell work + fixer), x2
// momentum where periodic. Per stage: x1 walls pass no mass, x2 wrap faces
// agree.
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/mesh/mesh.hpp>

#include "cuda_test_gate.hpp"
using namespace snap;
struct Case {  // gw: cell (+ fixer) or face; per: periodic x2/x3; vsed: rain
  char const *name, *deck, *gw;  // qv: vapor, supersaturated above ~0.022
  int scheme;
  bool per, pocket;
  double vsed = 0., qv = 0.01;
};
double const C = 4., kEps = 2.220446049250313e-16, g = 10.;
bool const kMp = getenv("SNAP_BUDGET_MP") != nullptr;  // E2: three ranks

std::string card(Case const& c) {  // the in-tree deck, closed and small
  auto d = YAML::LoadFile(c.deck);
  d.remove("outputs");
  auto cells = d["geometry"]["cells"];
  if (cells["nx2"].as<int>() == 1) cells["nx2"] = 4, cells["nx3"] = 4;
  for (auto f : {"x2-inner", "x2-outer", "x3-inner", "x3-outer"})
    if (c.per || !d["boundary-condition"]["external"][f])
      d["boundary-condition"]["external"][f] =
          c.per ? "periodic" : "reflecting";
  d["forcing"]["const-gravity"]["grav1"] = -g;
  d["forcing"]["const-gravity"]["gravity-work"] = c.gw;
  if (c.vsed != 0.)
    d["sedimentation"] =
        YAML::Load("{radius: {}, density: {}, const-vsed: {rain: " +
                   std::to_string(c.vsed) + "}}");
  if (kMp)  // three ranks: at two, both x2 neighbours are the same peer
    d["distribute"] = YAML::Load("{layout: slab, nb2: 3, nb3: 1}"),
    cells["nx2"] = 6;
  d["integration"] = YAML::Load("{type: rk3, cfl: 0.5}");
  d["integration"]["implicit-scheme"] = c.scheme;
  std::string path = std::string("budget_") + c.name +  // one per rank
                     (getenv("RANK") ? getenv("RANK") : "") + ".yaml";
  std::ofstream(path) << d;
  return path;
}
struct Budget : testing::TestWithParam<Case> {};
TEST_P(Budget, closes_to_round_off) {
  auto c = GetParam();
  bool cuda = getenv("SNAP_BUDGET_CUDA");  // the _cuda entry
  if (cuda && !snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA";
  if (kMp != (std::string(c.name) == "e2_mp"))
    GTEST_SKIP() << "mp only / not mp";
  if (kMp && cuda) GTEST_SKIP() << "E2 runs on CPU ranks only";
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(card(c)));
  int nb = mesh->blocks.size(),
      ny = mesh->blocks[0]->phydro->peos->nvar() - ICY;
  bool face = std::string(c.gw) == "face";
  MeshVariables vars(nb);
  std::vector<torch::Tensor> vol(nb), phi(nb);
  for (int i = 0; i < nb; ++i) {
    auto b = mesh->blocks[i];
    auto co = b->pcoord;
    auto m = torch::meshgrid({co->x3v, co->x2v, co->x1v}, "ij");
    double x0 = co->options->x1min(), h = co->options->x1max() - x0;
    auto r = m[2], s = (r - x0) / h;
    // isothermal column: scale height h when dry, 300 K air with tracers
    double rho0 = ny ? 1.16 : 1.e5 / (g * h), u0 = std::sqrt(10. / rho0);
    auto w = r.new_zeros({ICY + ny, r.size(0), r.size(1), r.size(2)});
    w[IDN] = rho0 * (-(r - x0) * rho0 * g / 1.e5).exp() *
             (1. + 0.01 * (3. * m[1]).sin() * m[0].cos());
    w[IPR] = w[IDN] / rho0 * 1.e5;
    w[IVX] = u0 * (M_PI * s).sin() * (2. * m[1]).cos();
    w[IVY] = u0 * (M_PI * s).sin();
    for (int n = 0; n < ny; ++n)  // vapor qv, first condensate empty
      w[ICY + n] = (n == 1 ? 0. : n ? 0.01 : c.qv) * (1. + 0.5 * s);
    if (c.pocket) {  // the clamp at the wall and theta < 1 at the x2 edge
      w[ICY].select(-1, co->il()).fill_(1.e-12);
      w[ICY].select(-2, co->jl()).fill_(1.e-12);
    }
    vol[i] = co->cell_volume();
    phi[i] = g * (co->options->type() == "cartesian" ? r - x0 : r);
    if (cuda) b->to(torch::kCUDA);
    for (auto* t : {&vol[i], &phi[i]}) *t = t->to(co->x1v.device());
    vars[i]["hydro_w"] = w.to(co->x1v.device());
  }
  mesh->initialize(vars);
  auto P = PartOptions().exterior(false);
  auto in = mesh->blocks[0]->part({}, P),
       in3 = mesh->blocks[0]->part({}, P.ndim(3));
  auto totals = [&]() {  // mass, energy, x2 momentum, then each species
    auto t = torch::zeros({3 + ny}, torch::kFloat64);
    for (int i = 0; i < nb; ++i) {
      auto co = mesh->blocks[i]->pcoord;
      int il = co->il(), iu = co->iu();
      auto u = vars[i].at("hydro_u").index(in);
      auto dV = vol[i].index(in3), rho = u[IDN] + u.narrow(0, ICY, ny).sum(0);
      auto pe = rho * phi[i].index(in3);
      auto f = co->x1f.slice(0, il, iu + 2), v = co->x1v.slice(0, il, iu + 1);
      auto ms = x1_measure(co->options->type());  // P by hand, not through
      if (face)  // corrected_pe_work: P = sum V [rho phi(r_c) - g1 <(x1 -
        pe += g * x1_variance(f, ms != X1Measure::plain) *  // r_c)^2> rho']
              centroid_slope(rho, v);
      if (face && ms == X1Measure::radial_mid)  // r_c != x1v
        pe += g * rho * x1_centroid_offset(f, v);
      std::vector<torch::Tensor> x = {rho * dV, (u[IPR] + pe) * dV,
                                      u[IVY] * dV};
      for (int n = 0; n < ny; ++n) x.push_back(u[ICY + n] * dV);
      for (int k = 0; k < 3 + ny; ++k)
        t[k] += x[k].sum().cpu().to(torch::kFloat64);
    }
    if (kMp) {
      std::vector<torch::Tensor> v = {t};
      mesh->blocks[0]->get_layout()->comm->allreduce(v, c10d::ReduceOp::SUM);
    }
    return t;
  };
  auto stage_checks = [&](double dt) {  // the x1 walls, the x2 wrap, the solve
    for (int i = 0; i < nb; ++i) {
      auto b = mesh->blocks[i];
      auto co = b->pcoord;
      int il = co->il(), iu = co->iu(), jl = co->jl(), ju = co->ju() + 1;
      auto kj = [&](torch::Tensor x) {  // interior x3, x2 of a face slice
        return x.slice(-2, co->kl(), co->ku() + 1).slice(-1, jl, ju);
      };
      auto f1 = b->phydro->flux1(), a1 = co->face_area1();
      auto rows = torch::cat({f1.narrow(0, IDN, 1), f1.narrow(0, ICY, ny)});
      auto u = vars[i].at("hydro_u").index(in);
      auto mw = (u[IDN] + u.narrow(0, ICY, ny).sum(0)) * vol[i].index(in3);
      auto sum = [](torch::Tensor x) { return x.abs().sum().item<double>(); };
      double wall = 0., mwall = sum(mw.select(-1, 0)) + sum(mw.select(-1, -1));
      for (int f : {il, iu + 1})
        wall += dt * sum(kj(rows.select(-1, f) * a1.select(-1, f)));
      EXPECT_LE(wall, 10. * kEps * mwall) << "x1 wall flux, stage dt " << dt;
      if (c.per && !kMp) {  // one block: the two copies of the x2 wrap face
        auto f2 = b->phydro->flux2()
                      .slice(1, co->kl(), co->ku() + 1)
                      .slice(3, il, iu + 1);
        EXPECT_TRUE(torch::equal(f2.select(2, jl), f2.select(2, ju)))
            << "periodic x2 face copies differ";
      }
      auto const& lm = b->phydro->peos->limiter_marks();
      EXPECT_TRUE(!lm.defined() || !lm.any().item<bool>()) << "limiter fired";
      if (b->phydro->picorr)  // a failed solve skips the fixer
        EXPECT_EQ(b->phydro->picorr->solve_failed_tensor(torch::kFloat64)
                      .item<double>(),
                  0.);
    }
  };
  auto t0 = totals();
  for (int step = 1; step <= 20; ++step) {
    double dt = mesh->max_time_step(vars);
    for (int st = 0; st < 3; ++st) {
      mesh->forward(vars, dt, st);
      stage_checks(dt);
    }
    auto t = totals();
    printf("budget %s step %2d rel", c.name, step);
    for (int k = 0; k < 3 + ny; ++k) {
      if ((k == 2 && !c.per) || (k > 2 && c.qv > .02)) continue;  // x2 mom.:
      // periodic only; species: the saturating row gates their sum (mass)
      double x0 = t0[k].item<double>(), dx = (t[k] - t0[k]).item<double>();
      double rel = std::abs(dx) / std::abs(x0 ? x0 : t0[0].item<double>());
      printf(" %.2e", rel / (step * kEps));
      EXPECT_LE(rel, C * step * kEps) << "total " << k << " step " << step;
    }
    printf("  (units of n eps)\n");
    double dc = c.qv > .02 ? (t[4] - t0[4]).item<double>() : 1.;  // cloud
    if (c.qv > .02) printf("condensed %.3e kg\n", dc);
    EXPECT_GT(dc, 0.) << "the saturating row made no cloud";
  }
}
char const *A = "test_gravity_energy.yaml", *B = "test_species_face_clamp.yaml",
           *Cs = "test_fix_vapor_volume.yaml",
           *D = "test_cubed_sphere_cell_volume.yaml";
Case const kDecks[] = {{"a_cell_s0", A, "cell", 0},
                       {"a_cell_s1", A, "cell", 1},
                       {"a_cell_s9", A, "cell", 9},
                       {"a_face_s0", A, "face", 0},
                       {"a_face_s1", A, "face", 1, true},
                       {"a_face_s9", A, "face", 9},
                       {"a_per", A, "face", 0, true},
                       {"b_moist_slab", B, "face", 0, true, true},
                       {"b_sat", B, "face", 0, true, false, 0., 0.03},
                       {"c_face_s0", Cs, "face", 0, false, false, -0.5},
                       {"c_face_s9", Cs, "face", 9},
                       {"c_cell_s9", Cs, "cell", 9},
                       {"d_cubed", D, "face", 0},
                       {"d_cubed_s9", D, "face", 9},
                       {"e2_mp", A, "face", 0, true}};
INSTANTIATE_TEST_SUITE_P(Decks, Budget, testing::ValuesIn(kDecks),
                         [](auto const& p) {
                           return std::string(p.param.name);
                         });
