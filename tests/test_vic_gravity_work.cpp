// C/C++
#include <cmath>

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/implicit/vic_assemble_full_impl.h>
#include <snap/implicit/vic_assemble_partial_impl.h>
#include <snap/implicit/vic_gravity_work_impl.h>
#include <snap/snap.h>

#include <snap/implicit/implicit_dispatch.hpp>

namespace {

constexpr int kNy = 0;
constexpr int kNlayer = 4;
constexpr int kNc = kNlayer + 2;  // one ghost cell on each side

// one column, primitive variables stored [var][cell]; the assembly of cell i
// reads cells i - 1 and i + 1, so the pointers below start at cell 1
struct Column {
  double w_[5 * kNc], gamma_[kNc], area[kNlayer + 1], vol[kNlayer], lo[kNlayer],
      hi[kNlayer];
  double* w = w_ + 1;
  double* gamma = gamma_ + 1;
  Column() {
    for (int i = 0; i < kNc; ++i) {
      w_[snap::IDN * kNc + i] = 1.2 - 0.1 * i;
      w_[snap::IVX * kNc + i] = 3. * std::sin(1. + i);
      w_[snap::IVY * kNc + i] = 2. * std::cos(0.3 * i);
      w_[snap::IVZ * kNc + i] = -1.5 + 0.4 * i;
      w_[snap::IPR * kNc + i] = 1.e5 * (1. - 0.08 * i);
      gamma_[i] = 1.4;
    }
    for (int i = 0; i < kNlayer; ++i) {
      vol[i] = 1. + 0.05 * i;
      lo[i] = 0.23 + 0.01 * i;
      hi[i] = 0.27 - 0.01 * i;
    }
    for (int i = 0; i <= kNlayer; ++i) area[i] = 1. + 0.1 * i;
  }
};

}  // namespace

// the partial matrix is the full one on (rho, m1, E) when the tangential
// momenta are frozen, gravity-work rows included, for every work mode
TEST(VicGravityWork, partial_rows_are_the_full_rows) {
  using M3 = Eigen::Matrix<double, 3, 3>;
  using M5 = Eigen::Matrix<double, 5, 5>;
  int const rows[3] = {snap::IDN, snap::IVX, snap::IPR};
  int const modes[] = {0, snap::kVicDiffusiveCell, snap::kVicFaceWork,
                       snap::kVicFaceWork | snap::kVicCartesianFaceWork};
  for (int mode : modes) {
    Column col;
    M3 a3[kNlayer], b3[kNlayer], c3[kNlayer];
    M5 a5[kNlayer], b5[kNlayer], c5[kNlayer];
    for (int i = 0; i < kNlayer; ++i) {
      snap::vic_assemble_partial_impl(
          a3, b3, c3, col.w, col.gamma, col.area, col.vol, col.lo, col.hi, i, 0,
          kNlayer - 1, 2., -9.8, mode, kNy, kNc, 1, true, true);
      snap::vic_assemble_full_impl(
          a5, b5, c5, col.w, col.gamma, col.area, col.vol, col.lo, col.hi, i, 0,
          kNlayer - 1, 2., -9.8, mode, kNy, kNc, 1, true, true, false);
      for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k) {
          EXPECT_EQ(a3[i](r, k), a5[i](rows[r], rows[k]))
              << "mode " << mode << " cell " << i << " a(" << r << k << ")";
          EXPECT_EQ(b3[i](r, k), b5[i](rows[r], rows[k]))
              << "mode " << mode << " cell " << i << " b(" << r << k << ")";
          EXPECT_EQ(c3[i](r, k), c5[i](rows[r], rows[k]))
              << "mode " << mode << " cell " << i << " c(" << r << k << ")";
        }
    }
  }
}

// with no |A| mass rows the face work changes only the energy row's
// x1-momentum column: grav (1 - lower - upper) on the cell, -grav lower and
// -grav upper on its two neighbours (Cartesian: grav / 2 and -grav / 4 each)
TEST(VicGravityWork, face_work_rows_without_mass_diffusion) {
  using M3 = Eigen::Matrix<double, 3, 3>;
  using R3 = Eigen::Matrix<double, 1, 3>;
  M3 a = M3::Zero(), b = M3::Zero(), c = M3::Zero();
  R3 zero = R3::Zero();
  double lower = 0.25, upper = 0.375, grav = -2.;
  snap::vic_gravity_work_rows<double, 3>(a, b, c, zero, zero, 2, 1, grav, true,
                                         false, false, lower, upper);
  EXPECT_EQ(a(2, 1), grav * (1. - lower - upper));
  EXPECT_EQ(b(2, 1), -grav * lower);
  EXPECT_EQ(c(2, 1), -grav * upper);
  a(2, 1) = b(2, 1) = c(2, 1) = 0.;
  EXPECT_TRUE(a.isZero(0.) && b.isZero(0.) && c.isZero(0.));
  a.setZero();
  b.setZero();
  c.setZero();
  snap::vic_gravity_work_rows<double, 3>(a, b, c, zero, zero, 2, 1, grav, true,
                                         false, true, lower, upper);
  EXPECT_EQ(a(2, 1), grav - 0.5 * grav);
  EXPECT_EQ(b(2, 1), -0.25 * grav);
  EXPECT_EQ(c(2, 1), -0.25 * grav);
  // no work mode: nothing
  a.setZero();
  b.setZero();
  c.setZero();
  snap::vic_gravity_work_rows<double, 3>(a, b, c, zero, zero, 2, 1, grav, false,
                                         false, false, lower, upper);
  EXPECT_TRUE(a.isZero(0.) && b.isZero(0.) && c.isZero(0.));
}
