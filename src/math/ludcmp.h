#pragma once

// C/C++
#include <cmath>
#include <limits>

// Eigen
#include <Eigen/Dense>

// base
#include <configure.h>

//! \brief Perform LU decomposition with partial pivoting
//!
//! Given a matrix a, this routine replaces it by the LU decomposition of a
//! rowwise permutation of itself. The decomposition is used with lubksb to
//! solve linear equations or invert a matrix.
//!
//! \tparam T Scalar type (e.g., float, double)
//! \tparam N Matrix dimension
//! \param[in,out] a Input matrix, replaced by LU decomposition on output
//! \param[out] indx Output vector recording row permutation from partial
//! pivoting
//! \return +1 or -1 depending on whether row interchanges were even or odd;
//!         0 indicates failure (singular/near-singular or nonfinite matrix)
//!
//! \note Adapted from Numerical Recipes in C, 2nd Ed., p. 46.
//! \note A pivot must exceed 8*N*epsilon times its original row magnitude.
//! \note Implicit=false pivots on the rows as given (the caller scaled them).
template <typename T, int N, bool Implicit = true>
inline DISPATCH_MACRO int ludcmp(Eigen::Matrix<T, N, N, Eigen::RowMajor> &a,
                                 int *indx) {
  int i, imax = 0, j, k, d;
  T big, dum, sum, temp;
  T vv[N], scale[N];
  const T tolerance = T(8 * N) * std::numeric_limits<T>::epsilon();
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col)
      if (!std::isfinite(a(row, col))) return 0;

  d = 1;
  for (i = 0; i < N; i++) {
    big = 0.0;
    for (j = 0; j < N; j++)
      if ((temp = fabs(a(i, j))) > big) big = temp;
    if (big == 0.0) return 0;
    scale[i] = big;
    vv[i] = Implicit ? 1.0 / big : 1.0;
    if (!std::isfinite(vv[i])) return 0;
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < j; i++) {
      sum = a(i, j);
      for (k = 0; k < i; k++) sum -= a(i, k) * a(k, j);
      if (!std::isfinite(sum)) return 0;
      a(i, j) = sum;
    }
    big = 0.0;
    for (i = j; i < N; i++) {
      sum = a(i, j);
      for (k = 0; k < j; k++) sum -= a(i, k) * a(k, j);
      if (!std::isfinite(sum)) return 0;
      a(i, j) = sum;
      dum = vv[i] * fabs(sum);
      if (!std::isfinite(dum)) return 0;
      if (dum >= big) {
        big = dum;
        imax = i;
      }
    }
    if (j != imax) {
      for (k = 0; k < N; k++) {
        dum = a(imax, k);
        a(imax, k) = a(j, k);
        a(j, k) = dum;
      }
      d = -d;
      vv[imax] = vv[j];
      temp = scale[imax];
      scale[imax] = scale[j];
      scale[j] = temp;
    }
    indx[j] = imax;
    if (fabs(a(j, j)) / scale[j] <= tolerance) return 0;
    if (j != N - 1) {
      dum = (1.0 / a(j, j));
      for (i = j + 1; i < N; i++) a(i, j) *= dum;
    }
  }

  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col)
      if (!std::isfinite(a(row, col))) return 0;
  return d;
}
