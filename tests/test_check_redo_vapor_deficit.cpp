// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

using namespace snap;

// #317: a step whose source after the stages (the driver's kinetics) takes
// more vapor out of a column than the column holds leaves a deficit with no
// donors, which fix_vapor cannot repair. The same step at dt/2 takes half as
// much and is fine, so check_redo must redo it with a smaller dt instead of
// throwing from the limiter before the redo flags are combined.
TEST(MeshBlock, check_redo_redoes_an_uncoverable_vapor_deficit) {
  auto block = MeshBlock(
      MeshBlockOptionsImpl::from_yaml("test_check_redo_vapor_deficit.yaml"));
  block->to(torch::kCPU, torch::kFloat64);
  auto pcoord = block->pcoord;

  double const q = 1.e-3;  // vapor mass fraction, every cell
  auto hydro_w =
      torch::zeros({block->phydro->peos->nvar(), pcoord->options->nc3(),
                    pcoord->options->nc2(), pcoord->options->nc1()},
                   torch::kFloat64);
  hydro_w[IDN].fill_(1.);
  hydro_w[IPR].fill_(1.e5);
  hydro_w[ICY].fill_(q);
  Variables vars;
  vars["hydro_w"] = hydro_w;
  block->initialize(vars);

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  double const vapor0 =
      vars["hydro_u"].index(interior)[ICY].min().item<double>();
  double const dt0 = block->max_time_step(vars);
  // a sink rate that empties the column 1.5 times over at dt0
  double const rate = 1.5 * vapor0 / dt0;

  std::vector<double> dts;
  std::vector<int> errs;
  for (int attempt = 0; attempt < 4; ++attempt) {
    double dt = block->max_time_step(vars);
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage)
      block->forward(vars, dt, stage);
    vars["hydro_u"].index(interior)[ICY] -= rate * dt;

    int err = -2;
    try {
      err = block->check_redo(vars);
    } catch (c10::Error const& e) {
      FAIL() << "check_redo threw at dt=" << dt << " (attempt " << attempt
             << "): " << e.what_without_backtrace();
    }
    std::printf("attempt %d: dt %.6g, check_redo %d\n", attempt, dt, err);
    dts.push_back(dt);
    errs.push_back(err);
    if (err <= 0) break;
  }

  // redo at dt0, then the step succeeds at dt0/2
  ASSERT_EQ(errs.size(), 2);
  EXPECT_EQ(errs[0], 1);
  EXPECT_EQ(errs[1], 0);
  EXPECT_DOUBLE_EQ(dts[0], dt0);
  EXPECT_DOUBLE_EQ(dts[1], 0.5 * dt0);
  auto vapor = vars["hydro_u"].index(interior)[ICY];
  EXPECT_GE(vapor.min().item<double>(), 0.);
  EXPECT_NEAR(vapor.min().item<double>(), 0.25 * vapor0, 1.e-12);
  EXPECT_EQ(block->pintg->current_redo, 0);
}
