#include <gtest/gtest.h>

#include <cstring>

#include "../src/implicit/vic_assemble_full_impl.h"
#include "../src/implicit/vic_assemble_partial_impl.h"

// The partial matrix with the m_t fold must be the full (5x5) matrix with
// dm_t,j = v_t,j drho_j + q_t,j substituted, q_t,j = du_t,j - v_t,j du_rho,j:
//   a3 = S a5 P_i, b3 = S b5 P_{i-1}, c3 = S c5 P_{i+1},
//   rhs_corr = S (a5 T q_i + b5 T q_{i-1} + c5 T q_{i+1}),
// with S the rows rho, m1, E; a face the wall mirror merged into a[i] uses
// cell i's v_t and q_t.
namespace {

using M5 = Eigen::Matrix<double, 5, 5>;
using M3 = Eigen::Matrix<double, 3, 3>;
using V3 = Eigen::Matrix<double, 3, 1>;

constexpr int kStore = 5;  // ghost, cells 0..2, ghost
constexpr int stride1 = kStore, stride2 = 1;
constexpr int is = 0, ie = 2;

struct Column {
  double w[5 * kStore], du[5 * kStore];
  double gamma[kStore], area[kStore + 1], vol[kStore];
  double work_lo[kStore], work_hi[kStore];
  explicit Column(double wind) {
    double rho[kStore] = {1.1, 1., .8, .6, .5};
    for (int j = 0; j < kStore; ++j) {
      w[snap::IDN * stride1 + j] = rho[j];
      w[snap::IVX * stride1 + j] = .12 - .05 * j;
      w[snap::IVY * stride1 + j] = wind * (.9 + .1 * j);
      w[snap::IVZ * stride1 + j] = wind * (-.2 + .07 * j);
      w[snap::IPR * stride1 + j] = 2. - .3 * j;
      du[snap::IDN * stride1 + j] = .01 * (j - 2);
      du[snap::IVX * stride1 + j] = .02;
      du[snap::IVY * stride1 + j] = wind * .03 * (j + 1);
      du[snap::IVZ * stride1 + j] = -wind * .01 * j;
      du[snap::IPR * stride1 + j] = .05;
      gamma[j] = 1.4;
      vol[j] = 1. + .1 * j;
      work_lo[j] = .25;
      work_hi[j] = .25;
    }
    for (int j = 0; j <= kStore; ++j) area[j] = 1. + .05 * j;
  }
  // pointers to cell 0
  double* W() { return w + 1; }
  double* DU() { return du + 1; }
  double v(int t, int j) { return w[(snap::IVY + t) * stride1 + j + 1]; }
  double q(int t, int j) {
    return du[(snap::IVY + t) * stride1 + j + 1] -
           v(t, j) * du[snap::IDN * stride1 + j + 1];
  }
};

Eigen::Matrix<double, 3, 5> S() {
  Eigen::Matrix<double, 3, 5> s;
  s.setZero();
  s(0, snap::IDN) = s(1, snap::IVX) = s(2, snap::IPR) = 1.;
  return s;
}
Eigen::Matrix<double, 5, 3> P(Column& col, int j) {
  Eigen::Matrix<double, 5, 3> p;
  p.setZero();
  p(snap::IDN, 0) = p(snap::IVX, 1) = p(snap::IPR, 2) = 1.;
  p(snap::IVY, 0) = col.v(0, j);
  p(snap::IVZ, 0) = col.v(1, j);
  return p;
}
Eigen::Matrix<double, 5, 1> Tq(Column& col, int j) {
  Eigen::Matrix<double, 5, 1> t;
  t.setZero();
  t(snap::IVY) = col.q(0, j);
  t(snap::IVZ) = col.q(1, j);
  return t;
}

void check(int flag, int i) {
  Column col(1.);
  double dt = .5, grav = -9.8;
  M5 a5[3], b5[3], c5[3];
  M3 a3[3], b3[3], c3[3];
  V3 corr[3];
  snap::vic_assemble_full_impl(a5, b5, c5, col.W(), col.gamma + 1, col.area + 1,
                               col.vol + 1, col.work_lo + 1, col.work_hi + 1, i,
                               is, ie, dt, grav, flag, 0, stride1, stride2,
                               true, true, false);
  snap::vic_assemble_partial_impl(
      a3, b3, c3, col.W(), col.gamma + 1, col.area + 1, col.vol + 1,
      col.work_lo + 1, col.work_hi + 1, i, is, ie, dt, grav, flag, 0, stride1,
      stride2, true, true, false, false, col.DU(), corr);

  M3 ea = S() * a5[i] * P(col, i);
  V3 er = S() * a5[i] * Tq(col, i);
  EXPECT_LT((a3[i] - ea).cwiseAbs().maxCoeff(), 1.e-12) << "a, cell " << i;
  if (i > is) {
    M3 eb = S() * b5[i] * P(col, i - 1);
    EXPECT_LT((b3[i] - eb).cwiseAbs().maxCoeff(), 1.e-12) << "b, cell " << i;
    er += S() * b5[i] * Tq(col, i - 1);
  }
  if (i < ie) {
    M3 ec = S() * c5[i] * P(col, i + 1);
    EXPECT_LT((c3[i] - ec).cwiseAbs().maxCoeff(), 1.e-12) << "c, cell " << i;
    er += S() * c5[i] * Tq(col, i + 1);
  }
  EXPECT_GT(er.cwiseAbs().maxCoeff(), 1.e-6);  // the term is exercised
  EXPECT_LT((corr[i] - er).cwiseAbs().maxCoeff(), 1.e-12) << "rhs, cell " << i;
}

}  // namespace

// V3 I-8 (I-8.tests): the face work is always assembled, so the arm without
// it (kVicFaceWork off with gravity on) is gone
TEST(vic_partial_tangential, equals_the_reduced_full_matrix) {
  for (int i = is; i <= ie; ++i) check(0, i);
}

TEST(vic_partial_tangential, a_wind_free_column_is_unchanged_bitwise) {
  Column col(0.);
  double dt = .5, grav = -9.8;
  int flag = 0;
  for (int i = is; i <= ie; ++i) {
    M3 a0[3], b0[3], c0[3], a1[3], b1[3], c1[3];
    V3 corr[3];
    snap::vic_assemble_partial_impl(a0, b0, c0, col.W(), col.gamma + 1,
                                    col.area + 1, col.vol + 1, col.work_lo + 1,
                                    col.work_hi + 1, i, is, ie, dt, grav, flag,
                                    0, stride1, stride2, true, true);
    snap::vic_assemble_partial_impl(
        a1, b1, c1, col.W(), col.gamma + 1, col.area + 1, col.vol + 1,
        col.work_lo + 1, col.work_hi + 1, i, is, ie, dt, grav, flag, 0, stride1,
        stride2, true, true, false, false, col.DU(), corr);
    EXPECT_EQ(std::memcmp(a0[i].data(), a1[i].data(), sizeof(M3)), 0);
    EXPECT_EQ(std::memcmp(b0[i].data(), b1[i].data(), sizeof(M3)), 0);
    EXPECT_EQ(std::memcmp(c0[i].data(), c1[i].data(), sizeof(M3)), 0);
    for (int r = 0; r < 3; ++r) {
      EXPECT_EQ(corr[i](r), 0.);
      EXPECT_FALSE(std::signbit(corr[i](r)));
    }
  }
}
