#pragma once

// C/C++
#include <cmath>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/math/ludcmp.h>
#include <snap/math/luminv.h>
#include <snap/snap.h>

#include "vic_solve_failure.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// K = 2 (Woodbury): the same factorisation also sweeps two unit right-hand
// sides, e_E at cells k0 and k1, into z0 and z1 (forward-eliminated; the
// caller back-substitutes them with a, as for delta)
template <int K = 0, typename T, int N>
bool DISPATCH_MACRO
ForwardSweep(Eigen::Matrix<T, N, N>* a, Eigen::Matrix<T, N, N>* b,
             Eigen::Matrix<T, N, N>* c, Eigen::Matrix<T, N, 1>* delta, T* du,
             double dt, int il, int iu, int dir, int ny, int stride1,
             int stride2, bool first_block, bool last_block,
             bool rhs_in_delta = false, Eigen::Matrix<T, N, 1>* z0 = nullptr,
             Eigen::Matrix<T, N, 1>* z1 = nullptr, int k0 = 0, int k1 = 0) {
  Eigen::Matrix<T, N, 1> rhs;
  auto unit = [&](int i, int k) {
    Eigen::Matrix<T, N, 1> e = Eigen::Matrix<T, N, 1>::Zero();
    if (i == k) e(N - 1) = 1;
    return e;
  };
  // partial matrix: the assembly left the m_t right-hand-side correction in
  // delta[i] (vic_assemble_partial_impl); read it before delta[i] is solved
  auto correct = [&](int i) {
    if (rhs_in_delta)
      for (int row = 0; row < N; ++row)
        if (delta[i](row) != 0) rhs(row) -= delta[i](row);
  };

  if constexpr (N == 3) {  // partial matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IPR, il) / dt;
    correct(il);
  } else {  // full matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, il) / dt;
    rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, il) / dt;
    rhs(4) = DU(IPR, il) / dt;
  }

  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A;
  Eigen::Matrix<T, N, N + 1 + K, Eigen::RowMajor> solved;

  // if (!first_block) {
  // RecvBuffer(a[il - 1], delta[il - 1], bblock);
  // a[il] = (a[il] - b[il] * a[il - 1]).inverse().eval();
  // delta[il] = a[il] * (rhs - b[il] * delta[il - 1]);
  // a[il] *= c[il];
  //} else {
  if constexpr (N > 4) {
    A = a[il];
    for (int n = 0; n < N; ++n) indx[n] = n;
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    solved.template leftCols<N>() = c[il];
    solved.col(N) = rhs;
    if constexpr (K == 2) {
      solved.col(N + 1) = unit(il, k0);
      solved.col(N + 2) = unit(il, k1);
    }
    lubksb(A, indx, solved);
    for (int row = 0; row < N; ++row)
      for (int col = 0; col < N + 1 + K; ++col)
        if (!std::isfinite(solved(row, col)))
          return vic_fail_column(delta, il, iu);
    a[il] = solved.template leftCols<N>();
    delta[il] = solved.col(N);
    if constexpr (K == 2)
      z0[il] = solved.col(N + 1), z1[il] = solved.col(N + 2);
  } else {  // retain the nonsingular small-matrix arithmetic
    A = a[il];
    if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
    a[il] = a[il].inverse().eval();
    delta[il] = a[il] * rhs;
    if constexpr (K == 2)
      z0[il] = a[il] * unit(il, k0), z1[il] = a[il] * unit(il, k1);
    a[il] = a[il] * c[il];
  }
  for (int row = 0; row < N; ++row) {
    if (!std::isfinite(delta[il](row))) return vic_fail_column(delta, il, iu);
    for (int col = 0; col < N; ++col)
      if (!std::isfinite(a[il](row, col)))
        return vic_fail_column(delta, il, iu);
  }
  //}

  for (int i = il + 1; i <= iu; ++i) {
    if constexpr (N == 3) {  // partial matrix
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IPR, i) / dt;
      correct(i);
    } else {
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
      rhs(4) = DU(IPR, i) / dt;
    }

    a[i] -= b[i] * a[i - 1];

    if constexpr (N > 4) {
      A = a[i];
      for (int n = 0; n < N; ++n) indx[n] = n;
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      solved.template leftCols<N>() = c[i];
      solved.col(N) = rhs - b[i] * delta[i - 1];
      if constexpr (K == 2) {
        solved.col(N + 1) = unit(i, k0) - b[i] * z0[i - 1];
        solved.col(N + 2) = unit(i, k1) - b[i] * z1[i - 1];
      }
      lubksb(A, indx, solved);
      for (int row = 0; row < N; ++row)
        for (int col = 0; col < N + 1 + K; ++col)
          if (!std::isfinite(solved(row, col)))
            return vic_fail_column(delta, il, iu);
      a[i] = solved.template leftCols<N>();
      delta[i] = solved.col(N);
      if constexpr (K == 2)
        z0[i] = solved.col(N + 1), z1[i] = solved.col(N + 2);
    } else {  // checked bypass path, retaining its original inverse
      A = a[i];
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      a[i] = a[i].inverse().eval();
      delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
      if constexpr (K == 2) {
        z0[i] = a[i] * (unit(i, k0) - b[i] * z0[i - 1]);
        z1[i] = a[i] * (unit(i, k1) - b[i] * z1[i - 1]);
      }
      a[i] = a[i] * c[i];
    }
    for (int row = 0; row < N; ++row) {
      if (!std::isfinite(delta[i](row))) return vic_fail_column(delta, il, iu);
      for (int col = 0; col < N; ++col)
        if (!std::isfinite(a[i](row, col)))
          return vic_fail_column(delta, il, iu);
    }
  }

  // SaveCoefficients(a, delta, il, iu);
  // if (!last_block) SendBuffer(a[iu], delta[iu], tblock);
  return true;
}

}  // namespace snap

#undef VOL
#undef DU
#undef W
