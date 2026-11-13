#pragma once

// base
#include <configure.h>

// One definition of each reconstruction at one face, shared by the CPU line
// loop (interp_impl.h) and the CUDA bodies (interp_impl.cuh). src points at
// the first of the stencil's inputs, which are stride apart; c holds the
// variant's coefficient rows. With scale, the stencil is divided by its mean
// absolute value before the nonlinear weights and multiplied back after; a
// stencil of zeros gives 0.

#define SNAP_SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
DISPATCH_MACRO inline T _vvdot(T const* v1, T const* v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

//! polynomial: sum_k c[k] * src[k]
template <typename T, int N>
DISPATCH_MACRO inline T poly_point(T const* src, int stride, T const* c) {
  T out = 0.;
  for (int k = 0; k < N; ++k) {
    out += c[k] * src[k * stride];
  }
  return out;
}

//! WENO3: c = two candidate rows, then two smoothness rows (3 each)
template <typename T>
DISPATCH_MACRO inline T weno3_point(T const* src, int stride, T const* c,
                                    bool scale) {
  T const* c1 = c;
  T const* c2 = c1 + 3;
  T const* c3 = c2 + 3;
  T const* c4 = c3 + 3;
  T vscale =
      scale ? (fabs(src[0]) + fabs(src[stride]) + fabs(src[2 * stride])) / 3.0
            : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[3];
  for (int k = 0; k < 3; ++k) {
    phi[k] = src[k * stride] / vscale;
  }

  T p0 = _vvdot<3>(phi, c1);
  T p1 = _vvdot<3>(phi, c2);

  T beta0 = SNAP_SQR(_vvdot<3>(phi, c3));
  T beta1 = SNAP_SQR(_vvdot<3>(phi, c4));

  T alpha0 = (2.0 / 3.0) / SNAP_SQR(beta0 + 1e-6);
  T alpha1 = (1.0 / 3.0) / SNAP_SQR(beta1 + 1e-6);

  return (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
}

//! WENO5: c = three candidate rows, then six smoothness rows (5 each). Nu: c
//! is one face's row of a per-face x1 table (kWeno5NuCoeff values), its three
//! linear weights d after the nine stencils; otherwise the weights are .3, .6,
//! .1
template <typename T, bool Nu = false>
DISPATCH_MACRO inline T weno5_point(T const* src, int stride, T const* c,
                                    bool scale) {
  T const* c1 = c;
  T const* c2 = c1 + 5;
  T const* c3 = c2 + 5;
  T const* c4 = c3 + 5;
  T const* c5 = c4 + 5;
  T const* c6 = c5 + 5;
  T const* c7 = c6 + 5;
  T const* c8 = c7 + 5;
  T const* c9 = c8 + 5;
  T vscale = scale ? (fabs(src[0]) + fabs(src[stride]) + fabs(src[2 * stride]) +
                      fabs(src[3 * stride]) + fabs(src[4 * stride])) /
                         5.0
                   : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = src[k * stride] / vscale;
  }

  T p0 = _vvdot<5>(phi, c1);
  T p1 = _vvdot<5>(phi, c2);
  T p2 = _vvdot<5>(phi, c3);

  T beta0 = 13. / 12. * SNAP_SQR(_vvdot<5>(phi, c4)) +
            .25 * SNAP_SQR(_vvdot<5>(phi, c5));
  T beta1 = 13. / 12. * SNAP_SQR(_vvdot<5>(phi, c6)) +
            .25 * SNAP_SQR(_vvdot<5>(phi, c7));
  T beta2 = 13. / 12. * SNAP_SQR(_vvdot<5>(phi, c8)) +
            .25 * SNAP_SQR(_vvdot<5>(phi, c9));

  T alpha0, alpha1, alpha2;
  if constexpr (Nu) {
    T const* d = c9 + 5;
    alpha0 = d[0] / SNAP_SQR(beta0 + 1e-6);
    alpha1 = d[1] / SNAP_SQR(beta1 + 1e-6);
    alpha2 = d[2] / SNAP_SQR(beta2 + 1e-6);
  } else {
    alpha0 = .3 / SNAP_SQR(beta0 + 1e-6);
    alpha1 = .6 / SNAP_SQR(beta1 + 1e-6);
    alpha2 = .1 / SNAP_SQR(beta2 + 1e-6);
  }

  return vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
         (alpha0 + alpha1 + alpha2);
}

}  // namespace snap

#undef SNAP_SQR
