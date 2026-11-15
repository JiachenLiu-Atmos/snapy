#pragma once

// C/C++
#include <cmath>
#include <type_traits>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/math/lubksb.h>
#include <snap/math/ludcmp.h>
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
  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A;
  Eigen::Matrix<T, N, N + 1 + K, Eigen::RowMajor> solved;

  // one elimination step per cell; the first cell has no row above it
  // FP32: scale each block row by a power of two of its [B A C] max, then
  // pivot on those rows (FP64 keeps ludcmp's implicit diagonal-block scaling)
  constexpr bool rowscale = std::is_same_v<T, float> && N > 4;
  Eigen::Matrix<T, N, 1> s;
  for (int i = il; i <= iu; ++i) {
    if constexpr (rowscale)
      for (int r = 0; r < N; ++r)
        s(r) = std::exp2(-std::round(std::log2(std::fmax(
            std::fmax(a[i].row(r).cwiseAbs().maxCoeff(),
                      b[i].row(r).cwiseAbs().maxCoeff()),
            std::fmax(c[i].row(r).cwiseAbs().maxCoeff(), T(1e-30))))));
    rhs(0) = DU(IDN, i);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, i) / dt;
    if constexpr (N == 3) {  // partial matrix
      rhs(2) = DU(IPR, i) / dt;
      // the assembly left the m_t right-hand-side correction in delta[i]
      // (vic_assemble_partial_impl); read it before delta[i] is solved
      if (rhs_in_delta)
        for (int row = 0; row < N; ++row)
          if (delta[i](row) != 0) rhs(row) -= delta[i](row);
    } else {  // full matrix
      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
      rhs(4) = DU(IPR, i) / dt;
    }

    if (i > il) a[i] -= b[i] * a[i - 1];

    A = a[i];
    if constexpr (N > 4) {
      for (int n = 0; n < N; ++n) indx[n] = n;
      if constexpr (rowscale) A = s.asDiagonal() * A;
      if (ludcmp<T, N, !rowscale>(A, indx) == 0)
        return vic_fail_column(delta, il, iu);
      solved.template leftCols<N>() = c[i];
      if (i > il) {
        solved.col(N) = rhs - b[i] * delta[i - 1];
        if constexpr (K == 2) {
          solved.col(N + 1) = unit(i, k0) - b[i] * z0[i - 1];
          solved.col(N + 2) = unit(i, k1) - b[i] * z1[i - 1];
        }
      } else {
        solved.col(N) = rhs;
        if constexpr (K == 2) {
          solved.col(N + 1) = unit(i, k0);
          solved.col(N + 2) = unit(i, k1);
        }
      }
      if constexpr (rowscale) solved = s.asDiagonal() * solved;
      lubksb(A, indx, solved);
      for (int row = 0; row < N; ++row)
        for (int col = 0; col < N + 1 + K; ++col)
          if (!std::isfinite(solved(row, col)))
            return vic_fail_column(delta, il, iu);
      a[i] = solved.template leftCols<N>();
      delta[i] = solved.col(N);
      if constexpr (K == 2)
        z0[i] = solved.col(N + 1), z1[i] = solved.col(N + 2);
    } else {  // retain the nonsingular small-matrix arithmetic
      if (ludcmp(A, indx) == 0) return vic_fail_column(delta, il, iu);
      a[i] = a[i].inverse().eval();
      if (i > il) {
        delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
        if constexpr (K == 2) {
          z0[i] = a[i] * (unit(i, k0) - b[i] * z0[i - 1]);
          z1[i] = a[i] * (unit(i, k1) - b[i] * z1[i - 1]);
        }
      } else {
        delta[i] = a[i] * rhs;
        if constexpr (K == 2)
          z0[i] = a[i] * unit(i, k0), z1[i] = a[i] * unit(i, k1);
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
  return true;
}

}  // namespace snap

#undef VOL
#undef DU
#undef W
