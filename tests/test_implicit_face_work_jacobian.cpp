#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include "../src/implicit/vic_assemble_full_impl.h"
#include "../src/implicit/vic_assemble_partial_impl.h"

namespace {

using Vector = Eigen::Matrix<double, 5, 1>;
using Matrix = Eigen::Matrix<double, 5, 5>;

Matrix roe_diffusion(double* left, double* right) {
  double prim[5];
  snap::RoeAverage(prim, 0.4, left, right);
  double cs = snap::SoundSpeed(prim, 0.4);
  Matrix r, ri;
  Vector lambda;
  snap::Eigenvalue(lambda, prim[snap::IVX], cs);
  snap::Eigenvector(r, ri, prim, cs, 0.4, 0);
  return r * lambda.asDiagonal() * ri;
}

template <int N>
void check_energy_row(int flag = snap::kVicFaceWork, bool curved = false,
                      bool centroid = false) {
  using Block = Eigen::Matrix<double, N, N>;
  double w[15] = {1.,  .8,   .6,   .12,  .09, .15, .07, .07,
                  .07, -.03, -.03, -.03, 2.,  1.7, 1.4};
  double gamma[3] = {1.4, 1.4, 1.4};
  double area[4] = {1., 1., 1., 1.};
  double volume[3] = {1., 1., 1.};
  double face[4] = {1., 2., 4., 7.};
  if (curved) {
    for (int j = 0; j < 4; ++j) area[j] = face[j] * face[j];
    for (int j = 0; j < 3; ++j)
      volume[j] = centroid
                      ? (std::pow(face[j + 1], 3) - std::pow(face[j], 3)) / 3.
                      : .5 * (area[j] + area[j + 1]) * (face[j + 1] - face[j]);
  }
  double center = centroid
                      ? .75 * (std::pow(face[2], 4) - std::pow(face[1], 4)) /
                            (std::pow(face[2], 3) - std::pow(face[1], 3))
                      : 3.;
  double lower_weight = curved ? area[1] * (center - face[1]) / volume[1] : .5;
  double upper_weight = curved ? area[2] * (face[2] - center) / volume[1] : .5;
  double work_lo[3] = {.25, .5 * lower_weight, .25};
  double work_hi[3] = {.25, .5 * upper_weight, .25};
  double wl[5], wc[5], wr[5];
  for (int n = 0; n < 5; ++n) {
    wl[n] = w[3 * n];
    wc[n] = w[3 * n + 1];
    wr[n] = w[3 * n + 2];
  }
  Matrix am = roe_diffusion(wl, wc);
  Matrix ap = roe_diffusion(wc, wr);
  constexpr double grav = -9.8;
  auto face_work = [&](std::array<Vector, 3> const& q) {
    double lower = .5 * (q[0](snap::IVX) + q[1](snap::IVX)) -
                   .5 * (am.row(snap::IDN) * (q[1] - q[0]))(0);
    double upper = .5 * (q[1](snap::IVX) + q[2](snap::IVX)) -
                   .5 * (ap.row(snap::IDN) * (q[2] - q[1]))(0);

    if (flag == snap::kVicDiffusiveCell) {
      double lower_diffusion = lower - .5 * (q[0](snap::IVX) + q[1](snap::IVX));
      double upper_diffusion = upper - .5 * (q[1](snap::IVX) + q[2](snap::IVX));
      return grav * (q[1](snap::IVX) + lower_weight * lower_diffusion +
                     upper_weight * upper_diffusion);
    }
    return grav * (lower_weight * lower + upper_weight * upper);
  };

  Block a[3], b[3], c[3], a0[3], b0[3], c0[3];
  auto assemble = [&](Block* aa, Block* bb, Block* cc, double g) {
    if constexpr (N == 5) {
      snap::vic_assemble_full_impl(aa, bb, cc, w, gamma, area, volume, work_lo,
                                   work_hi, 1, 0, 2, .5, g, flag, 0, 3, 1,
                                   false, false, false);
    } else {
      snap::vic_assemble_partial_impl(aa, bb, cc, w, gamma, area, volume,
                                      work_lo, work_hi, 1, 0, 2, .5, g, flag, 0,
                                      3, 1, false, false);
    }
  };
  assemble(a, b, c, grav);
  assemble(a0, b0, c0, 0.);
  std::array<Block, 3> source = {b0[1] - b[1], a0[1] - a[1], c0[1] - c[1]};
  std::array<int, N> vars;
  if constexpr (N == 5)
    vars = {0, 1, 2, 3, 4};
  else
    vars = {snap::IDN, snap::IVX, snap::IPR};

  // Freeze Roe coefficients, as VIC does; differentiate the face work,
  // not the separate nonlinear WENO/Riemann explicit flux.
  double worst = 0.;
  for (int cell = 0; cell < 3; ++cell) {
    for (int n = 0; n < N; ++n) {
      std::array<Vector, 3> plus, minus;
      for (int j = 0; j < 3; ++j) plus[j].setOnes();
      minus = plus;
      constexpr double h = 1.e-5;
      plus[cell](vars[n]) += h;
      minus[cell](vars[n]) -= h;
      double fd = (face_work(plus) - face_work(minus)) / (2. * h);
      double error = std::abs(fd - source[cell](N - 1, n));
      worst = std::max(worst, error);
      EXPECT_NEAR(fd, source[cell](N - 1, n), 1.e-5)
          << "N=" << N << " cell=" << cell << " variable=" << vars[n];
    }
  }
  std::cout << "flag " << flag << " VIC " << N << "x" << N
            << " face-work FD max error: " << worst << '\n';
}

}  // namespace

TEST(implicit_face_work, full_energy_row_matches_frozen_roe_flux) {
  check_energy_row<5>();
}

TEST(implicit_face_work, partial_energy_row_matches_frozen_roe_flux) {
  check_energy_row<3>();
}

TEST(implicit_face_work, full_cell_work_includes_roe_mass_diffusion) {
  check_energy_row<5>(snap::kVicDiffusiveCell);
}

TEST(implicit_face_work, partial_cell_work_includes_roe_mass_diffusion) {
  check_energy_row<3>(snap::kVicDiffusiveCell);
}

TEST(implicit_face_work, full_cell_work_uses_curved_face_metrics) {
  for (bool centroid : {false, true})
    check_energy_row<5>(snap::kVicDiffusiveCell, true, centroid);
}

TEST(implicit_face_work, partial_cell_work_uses_curved_face_metrics) {
  for (bool centroid : {false, true})
    check_energy_row<3>(snap::kVicDiffusiveCell, true, centroid);
}

TEST(implicit_face_work, full_face_work_uses_curved_face_metrics) {
  for (bool centroid : {false, true})
    check_energy_row<5>(snap::kVicFaceWork, true, centroid);
}

TEST(implicit_face_work, partial_face_work_uses_curved_face_metrics) {
  for (bool centroid : {false, true})
    check_energy_row<3>(snap::kVicFaceWork, true, centroid);
}
