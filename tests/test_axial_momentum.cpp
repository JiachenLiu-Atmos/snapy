// Lane A phase 2 (A.rst 3.3, 5.1-5.3): the AAM consumer and its closure fixer
// on six cubed-sphere panels in one process (test_axial_momentum_flux_form.yaml
// with conserve-axial-momentum: exact and a Coriolis force).
//   G6  the closure's declared AAM change, -w2 dt R, is the measured one;
//       its partial sums {R, I} are FP64 on the state's device and it books
//       "aam.closure" as a fixer amount on rows IVY and IVZ only;
//   G7  a sponge's and a bottom drag's booked AAM share, sum V r K . du of
//       the source's own increment, is the AAM change it makes.

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <thread>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

Mesh six_panels(int nx, YAML::Node const& forcing = YAML::Node()) {
  auto cfg = YAML::LoadFile("test_axial_momentum_flux_form.yaml");
  cfg["dynamics"]["conserve-axial-momentum"] = "exact";
  cfg["geometry"]["cells"]["interp_order"] = 4;
  cfg["forcing"]["coriolis"]["type"] = "xyz";
  cfg["forcing"]["coriolis"]["omega1"] = 5.e-6;
  for (auto const& f : forcing)
    cfg["forcing"][f.first.as<std::string>()] = f.second;
  char path[] = "test_axial_momentum_XXXXXX.yaml";
  auto fd = mkstemps(path, 5);
  EXPECT_GE(fd, 0);
  {
    std::ofstream f(path);
    f << YAML::Dump(cfg);
  }
  close(fd);
  auto opts = MeshOptionsImpl::from_yaml(path);
  std::remove(path);
  opts->blocks_per_process(6);
  {  // as python/csrc/pymesh.cpp set_local_horizontal_cells(nx, nx)
    auto coord = opts->block()->coord();
    auto layout = opts->block()->layout();
    if (coord->global_nx2() == 0) {
      coord->global_x2min(coord->x2min());
      coord->global_x2max(coord->x2max());
    }
    if (coord->global_nx3() == 0) {
      coord->global_x3min(coord->x3min());
      coord->global_x3max(coord->x3max());
    }
    coord->nx2(nx);
    coord->nx3(nx);
    coord->global_nx2(nx * layout->px());
    coord->global_nx3(nx * layout->py());
  }
  return Mesh(opts);
}

//! a wind that is not solid-body, so the Coriolis residual R is not zero
MeshVariables windy(Mesh const& mesh) {
  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto c = block->pcoord;
    auto m = torch::meshgrid({c->x3v, c->x2v, c->x1v}, "ij");
    auto w = torch::zeros(
        {block->phydro->peos->nvar(), m[0].size(0), m[0].size(1), m[0].size(2)},
        torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    w[IVY] = 10. + 3. * torch::sin(2. * m[1]) * torch::cos(m[0]);
    w[IVZ] = 2. * torch::cos(3. * m[1]);
    vars[b]["hydro_w"] = w;
  }
  return vars;
}

double total_aam(Mesh const& mesh, MeshVariables const& vars) {  // L
  double L = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto u = vars[b].at("hydro_u");
    L += mesh->blocks[b]->phydro->paam->aam(u[IVY], u[IVZ]).item<double>();
  }
  return L;
}

}  // namespace

struct Closure : testing::TestWithParam<int> {};

// G6 (A.rst 5.3): declared against measured, to 1e-12 relative
TEST_P(Closure, declared_change_is_the_measured_one) {
  auto mesh = six_panels(GetParam());
  auto vars = windy(mesh);
  mesh->initialize(vars);
  double dt = mesh->max_time_step(vars);
  int nb = mesh->blocks.size();

  // the blocks' stage together, the closure deferred as Mesh::forward does
  std::vector<std::thread> threads;
  for (int b = 0; b < nb; ++b) {
    mesh->blocks[b]->defer_axial_closure = true;
    threads.emplace_back(
        [&, b] { mesh->blocks[b]->advance_local(vars[b], dt, 0); });
  }
  for (auto& t : threads) t.join();

  auto global = torch::zeros({2}, torch::kFloat64);
  for (auto const& block : mesh->blocks) {
    auto sums = block->phydro->axial_closure_sums();
    ASSERT_EQ(sums.numel(), 2);
    EXPECT_EQ(sums.scalar_type(), torch::kFloat64);
    EXPECT_EQ(sums.device(), block->pcoord->x1v.device());
    global += sums;
  }
  double R = global[0].item<double>(), I = global[1].item<double>();
  ASSERT_GT(I, 0.);
  ASSERT_NE(R, 0.);

  double w2 = mesh->blocks[0]->pintg->stages[0].wght2();
  // The increment is measured on a zero state, where u + du is du exactly;
  // on the state itself u + du - u rounds at eps |u| per cell, which is
  // 1e-11 of this increment (reported below, not gated).
  double declared = 0., measured = 0., on_state = 0.;
  double L = total_aam(mesh, vars);
  for (int b = 0; b < nb; ++b) {
    auto hydro = mesh->blocks[b]->phydro;
    auto paam = hydro->paam;
    auto u = vars[b].at("hydro_u");
    auto du = torch::zeros_like(u);
    auto& mon = hydro->flux_register().monitor();
    declared += AxialClosureFixer(paam.get())
                    .apply(du, global, dt, w2, hydro->flux_register())
                    .item<double>();
    measured += paam->aam(du[IVY], du[IVZ]).item<double>();
    auto u1 = u + du;
    on_state += paam->aam(u1[IVY] - u[IVY], u1[IVZ] - u[IVZ]).item<double>();
    // rows IVY and IVZ only, and a fixer amount under the fixer's name
    for (int row : {IDN, IVX, IPR})
      EXPECT_EQ(du[row].abs().max().item<double>(), 0.);
    ASSERT_GE(mon.amount_index("aam.closure"), 0);
    EXPECT_EQ(mon.kind("aam.closure"), AmountKind::fixer);
    auto booked = mon.amounts("aam.closure");
    auto vol = mesh->blocks[b]->pcoord->cell_volume();
    for (int row : {IVY, IVZ}) {
      auto vdu = vol * du[row];
      EXPECT_GT(vdu.abs().sum().item<double>(), 0.);
      EXPECT_NEAR(booked[row].item<double>(), vdu.sum().item<double>(),
                  1.e-12 * vdu.abs().sum().item<double>());
    }
    EXPECT_EQ(booked[IDN].item<double>(), 0.);
    EXPECT_EQ(booked[IPR].item<double>(), 0.);
    // the stage's sums are consumed: a second apply does nothing
    EXPECT_FALSE(AxialClosureFixer(paam.get())
                     .apply(du, global, dt, w2, hydro->flux_register())
                     .defined());
  }
  printf(
      "G6 nx %d: L %.6e R %.6e I %.6e declared %.15e measured %.15e rel "
      "%.3e; on the state rel %.3e; (declared + w2 dt R) / declared "
      "%.3e\n",
      GetParam(), L, R, I, declared, measured,
      std::abs(measured - declared) / std::abs(declared),
      std::abs(on_state - declared) / std::abs(declared),
      std::abs(declared + w2 * dt * R) / std::abs(declared));
  EXPECT_LE(std::abs(declared + w2 * dt * R), 1.e-13 * std::abs(declared));
  EXPECT_LE(std::abs(measured - declared), 1.e-12 * std::abs(declared));
}

// 8: no cell on the axis; 9: one per polar panel, which hands its share on
INSTANTIATE_TEST_SUITE_P(Panels, Closure, testing::Values(8, 9));

// a non-finite R or I <= 0 leaves the state alone, with no host branch
TEST(ClosureMask, bad_sums_change_nothing) {
  auto mesh = six_panels(8);
  auto vars = windy(mesh);
  mesh->initialize(vars);
  double dt = mesh->max_time_step(vars);
  auto block = mesh->blocks[0];
  std::vector<std::thread> threads;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    mesh->blocks[b]->defer_axial_closure = true;
    threads.emplace_back(
        [&, b] { mesh->blocks[b]->advance_local(vars[b], dt, 0); });
  }
  for (auto& t : threads) t.join();
  auto u = vars[0].at("hydro_u");
  auto u0 = u.clone();
  auto bad = torch::tensor({std::nan(""), 1.}, torch::kFloat64);
  auto dl = AxialClosureFixer(block->phydro->paam.get())
                .apply(u, bad, dt, 1., block->phydro->flux_register());
  EXPECT_TRUE(torch::equal(u, u0));
  EXPECT_EQ(dl.item<double>(), 0.);
}

// G7 (A.rst 5.3): a source's booked AAM share is the AAM change it makes
TEST(SourceShare, sponge_and_drag_torque_is_the_aam_change) {
  auto mesh = six_panels(
      8, YAML::Load("top-sponge-lyr: {tau: 3.e3, width: 1.e5}\n"
                    "relax-bot-velo: {tau: 2.e3, bvx: 0., bvy: 0.}\n"));
  auto vars = windy(mesh);
  mesh->initialize(vars);
  double dt = mesh->max_time_step(vars);
  std::map<std::string, double> booked, change;
  double L = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto hydro = mesh->blocks[b]->phydro;
    auto paam = hydro->paam;
    auto w = vars[b].at("hydro_w"), u = vars[b].at("hydro_u");
    StageState stage{w, hydro->peos->compute("W->T", {w}), dt};
    double L0 = paam->aam(u[IVY], u[IVZ]).item<double>();
    L += L0;
    auto du = torch::zeros_like(u);
    for (auto const& r : hydro->sources.runs(SourcePhase::pre_implicit))
      hydro->sources.run(
          r, du, stage, [&](std::string const& name, torch::Tensor s) {
            booked[name] += paam->aam(s[IVY], s[IVZ]).item<double>();
            auto u1 = u + s;
            change[name] += paam->aam(u1[IVY], u1[IVZ]).item<double>() - L0;
          });
  }
  for (auto name : {"top-sponge-lyr", "relax-bot-velo"}) {
    ASSERT_TRUE(booked.count(name)) << name;
    printf("G7 %s: booked %.15e change %.15e |diff| / |L| %.3e (L %.6e)\n",
           name, booked[name], change[name],
           std::abs(booked[name] - change[name]) / std::abs(L), L);
    EXPECT_NE(booked[name], 0.) << name;
    EXPECT_LE(std::abs(booked[name] - change[name]), 1.e-15 * std::abs(L))
        << name;
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
