// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/gravity_work.hpp>

#include "../src/implicit/vic_gravity_work_impl.h"

using namespace snap;

// V3 I-3: the operator's gravity-work rows come from GravityWork's
// coefficients. The Cartesian midpoint-potential branch of the energy row is
// gone; a Cartesian column takes the general face-work rows with
// GravityWork's face weights, which are 0.25 on a uniform grid. RED before
// I-3: there is no GravityWork, and vic_gravity_work_rows still takes the
// Cartesian switch.

namespace {

// a column of nx1 cells (nghost each side) along x1, one cell in x2 and x3
struct Column {
  torch::Tensor x1f, x2f, x3f, x1v, area, vol;
  int ng, nx1;
  Column(double x1min, double dx, int nx1_, int ng_, bool spherical)
      : ng(ng_), nx1(nx1_) {
    int n = nx1 + 2 * ng;
    x1f = x1min + dx * (torch::arange(n + 1, torch::kFloat64) - ng);
    x2f = torch::tensor({0., 1.}, torch::kFloat64);
    x3f = torch::tensor({0., 1.}, torch::kFloat64);
    auto lo = x1f.narrow(0, 0, n), hi = x1f.narrow(0, 1, n);
    if (spherical) {
      x1v = .75 * (hi.pow(4) - lo.pow(4)) / (hi.pow(3) - lo.pow(3));
      area = (x1f * x1f).view({1, 1, n + 1});
      vol = ((hi.pow(3) - lo.pow(3)) / 3.).view({1, 1, n});
    } else {
      x1v = .5 * (lo + hi);
      area = torch::ones({1, 1, n + 1}, torch::kFloat64);
      vol = (hi - lo).view({1, 1, n});
    }
  }
  // the coordinate interface GravityWork::update reads
  torch::Tensor face_area1() const { return area; }
  torch::Tensor cell_volume() const { return vol; }
  int il() const { return ng; }
  int iu() const { return ng + nx1 - 1; }
};

GravityWork weights_of(Column const& col) {
  int n = col.nx1 + 2 * col.ng;
  auto w = torch::zeros({5, 1, 1, n}, torch::kFloat64);
  auto gamma = torch::zeros({1, 1, n}, torch::kFloat64);
  GravityWork gw;
  gw.update(col, w, gamma, -23.1);
  return gw;
}

}  // namespace

TEST(gravity_work_rows, cartesian_face_weights_are_a_quarter) {
  // the harness's column (#348) at the origin and 7e7 m away (#288)
  for (double x1min : {0., 7.e7}) {
    Column col(x1min, 29946.8085106, 45, 3, false);
    auto gw = weights_of(col);
    auto lo = gw.face_lo(), hi = gw.face_hi();
    // the column's own coordinates round: x1min + dx * k with ghost cells
    double tol = x1min == 0. ? 1.e-13 : 1.e-11;
    EXPECT_LT((lo - .25).abs().max().item<double>(), tol) << x1min;
    EXPECT_LT((hi - .25).abs().max().item<double>(), tol) << x1min;
  }
}

TEST(gravity_work_rows, spherical_face_weights_follow_the_geometry) {
  Column col(7.e7, 29946.8085106, 45, 3, true);
  auto gw = weights_of(col);
  int n = col.nx1 + 2 * col.ng;
  auto expect_lo = .5 * col.area.narrow(-1, 0, n) *
                   (col.x1v - col.x1f.narrow(0, 0, n)) / col.vol;
  EXPECT_TRUE(torch::equal(gw.face_lo(), expect_lo.unsqueeze(0).contiguous()));
}

// the general face-work rows with weights 0.25 give the removed Cartesian
// branch's rows to round-off: the same values in another operation order
TEST(gravity_work_rows, general_rows_equal_the_midpoint_form_to_round_off) {
  using M = Eigen::Matrix<double, 5, 5>;
  using R = Eigen::Matrix<double, 1, 5>;
  R am, ap;
  am << .7, -.3, .2, .1, -.4;
  ap << -.2, .6, .1, -.5, .3;
  double grav = -23.1;
  int e = IPR, m = IVX;
  M a = M::Constant(1.), b = M::Constant(2.), c = M::Constant(3.);
  vic_gravity_work_rows<double, 5>(a, b, c, am, ap, e, m, grav, true, .25, .25);
  // the midpoint-potential rows the Cartesian branch wrote
  M a0 = M::Constant(1.), b0 = M::Constant(2.), c0 = M::Constant(3.);
  R em = R::Zero();
  em(m) = 1.;
  a0(e, m) += grav;
  a0.row(e) -= 0.5 * grav * (em + 0.5 * (ap - am));
  b0.row(e) -= 0.5 * grav * (0.5 * em + 0.5 * am);
  c0.row(e) -= 0.5 * grav * (0.5 * em - 0.5 * ap);
  double scale = std::abs(grav) * 4.;
  EXPECT_LT((a - a0).cwiseAbs().maxCoeff(), 8. * 2.3e-16 * scale);
  EXPECT_LT((b - b0).cwiseAbs().maxCoeff(), 8. * 2.3e-16 * scale);
  EXPECT_LT((c - c0).cwiseAbs().maxCoeff(), 8. * 2.3e-16 * scale);
  EXPECT_GT((a - M::Constant(1.)).cwiseAbs().maxCoeff(), 1.);  // exercised
}
