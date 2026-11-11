#pragma once

// base
#include <configure.h>

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])
#define SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
DISPATCH_MACRO inline T _vvdot(T const* v1, T const* v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

// polynomial; coeff is one table, or one face's row of a per-face x1 table
template <typename T, int N>
DISPATCH_MACRO void interp_poly_impl(T* out, T const* inp, T const* coeff,
                                     int stride1, int stride2, int stride_out,
                                     int nvar) {
  for (int j = 0; j < nvar; ++j) {
    OUT(j) = 0.;
    for (int i = 0; i < N; ++i) {
      OUT(j) += coeff[i] * INP(j, i);
    }
  }
};

// WENO 3 interpolation
template <typename T>
DISPATCH_MACRO void interp_weno3_impl(T* out, T* inp, T* coeff, int stride1,
                                      int stride2, int stride_out, int nvar,
                                      double scale) {
  T* c1 = coeff;
  T* c2 = c1 + 3;
  T* c3 = c2 + 3;
  T* c4 = c3 + 3;

  T phi[3];

  for (int j = 0; j < nvar; ++j) {
    T vscale = scale
                   ? (fabs(INP(j, 0)) + fabs(INP(j, 1)) + fabs(INP(j, 2))) / 3.0
                   : 1.0;

    if (vscale != 0.0) {
      phi[0] = INP(j, 0) / vscale;
      phi[1] = INP(j, 1) / vscale;
      phi[2] = INP(j, 2) / vscale;
    } else {
      OUT(j) = 0.0;
      continue;
    }

    T p0 = _vvdot<3>(phi, c1);
    T p1 = _vvdot<3>(phi, c2);

    T beta0 = SQR(_vvdot<3>(phi, c3));
    T beta1 = SQR(_vvdot<3>(phi, c4));

    T alpha0 = (2.0 / 3.0) / SQR(beta0 + 1e-6);
    T alpha1 = (1.0 / 3.0) / SQR(beta1 + 1e-6);

    OUT(j) = (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
  }
};

// WENO 5 interpolation. Nu: coeff is one face's row of a per-face x1 table
// (kWeno5NuCoeff values), its three linear weights d after the nine stencils;
// otherwise the uniform table with the weights .3, .6, .1
template <typename T, bool Nu = false>
DISPATCH_MACRO void interp_weno5_impl(T* out, T const* inp, T const* coeff,
                                      int stride1, int stride2, int stride_out,
                                      int nvar, double scale) {
  T const* c1 = coeff;
  T const* c2 = c1 + 5;
  T const* c3 = c2 + 5;
  T const* c4 = c3 + 5;
  T const* c5 = c4 + 5;
  T const* c6 = c5 + 5;
  T const* c7 = c6 + 5;
  T const* c8 = c7 + 5;
  T const* c9 = c8 + 5;

  T phi[5];

  for (int j = 0; j < nvar; ++j) {
    T vscale = scale ? (fabs(INP(j, 0)) + fabs(INP(j, 1)) + fabs(INP(j, 2)) +
                        fabs(INP(j, 3)) + fabs(INP(j, 4))) /
                           5.0
                     : 1.0;

    if (vscale != 0.0) {
      for (int k = 0; k < 5; ++k) {
        phi[k] = INP(j, k) / vscale;
      }
    } else {
      OUT(j) = 0.0;
      continue;
    }

    T p0 = _vvdot<5>(phi, c1);
    T p1 = _vvdot<5>(phi, c2);
    T p2 = _vvdot<5>(phi, c3);

    T beta0 =
        13. / 12. * SQR(_vvdot<5>(phi, c4)) + .25 * SQR(_vvdot<5>(phi, c5));
    T beta1 =
        13. / 12. * SQR(_vvdot<5>(phi, c6)) + .25 * SQR(_vvdot<5>(phi, c7));
    T beta2 =
        13. / 12. * SQR(_vvdot<5>(phi, c8)) + .25 * SQR(_vvdot<5>(phi, c9));

    T alpha0, alpha1, alpha2;
    if constexpr (Nu) {
      T const* d = c9 + 5;
      alpha0 = d[0] / SQR(beta0 + 1e-6);
      alpha1 = d[1] / SQR(beta1 + 1e-6);
      alpha2 = d[2] / SQR(beta2 + 1e-6);
    } else {
      alpha0 = .3 / SQR(beta0 + 1e-6);
      alpha1 = .6 / SQR(beta1 + 1e-6);
      alpha2 = .1 / SQR(beta2 + 1e-6);
    }

    OUT(j) = vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
             (alpha0 + alpha1 + alpha2);
  }
};

}  // namespace snap

#undef OUT
#undef SQR
#undef INP
