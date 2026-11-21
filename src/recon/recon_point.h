#pragma once

// base
#include <configure.h>

// One definition of each reconstruction at one face, shared by the CPU line
// loop (recon_dispatch.cpp) and the CUDA line kernel (recon_line.cuh). src
// points at the first of the stencil's inputs, which are stride apart; c holds
// the scheme's coefficient rows. With scale, the stencil is divided by its
// mean absolute value before the nonlinear weights and multiplied back after;
// a stencil of zeros gives 0.

#define SNAP_SQR(x) ((x) * (x))

namespace snap {

//! every reconstruction scheme. dc, plm and ppm are tensor code; the other
//! four are stencil kernels with a coefficient table
enum class Scheme : int { dc, plm, ppm, cp3, weno3, cp5, weno5 };

//! stencil width and coefficient count of each scheme. ncoeff is the uniform
//! table; ncoeff_nu is one face's row of a per-face x1 table (0: the scheme
//! has none). A weno5 row holds its three linear weights d after the nine
//! stencil rows (45 + 3).
template <Scheme S>
struct SchemeTraits {
  static constexpr int stencil =
      S == Scheme::dc                                              ? 1
      : S == Scheme::plm || S == Scheme::cp3 || S == Scheme::weno3 ? 3
                                                                   : 5;
  static constexpr int ncoeff = S == Scheme::cp3     ? 3
                                : S == Scheme::weno3 ? 12
                                : S == Scheme::cp5   ? 5
                                : S == Scheme::weno5 ? 45
                                                     : 0;
  static constexpr int ncoeff_nu = S == Scheme::cp5     ? 5
                                   : S == Scheme::weno5 ? 48
                                                        : 0;
};

constexpr int kWeno5NuCoeff = SchemeTraits<Scheme::weno5>::ncoeff_nu;
constexpr int kPoly5NuCoeff = SchemeTraits<Scheme::cp5>::ncoeff_nu;

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

//! one face of a stencil scheme: the loop body of the CPU loop and the CUDA
//! line kernel. row is the uniform table, or with Nu one face's row of a
//! per-face x1 table (cp5, weno5)
template <Scheme S, typename T, bool Nu = false>
DISPATCH_MACRO inline T recon_point(T const* src, int stride, T const* row,
                                    bool scale) {
  if constexpr (S == Scheme::cp3) {
    return poly_point<T, 3>(src, stride, row);
  } else if constexpr (S == Scheme::cp5) {
    return poly_point<T, 5>(src, stride, row);
  } else if constexpr (S == Scheme::weno3) {
    return weno3_point(src, stride, row, scale);
  } else {
    static_assert(S == Scheme::weno5, "dc, plm and ppm are tensor code");
    return weno5_point<T, Nu>(src, stride, row, scale);
  }
}

}  // namespace snap

#undef SNAP_SQR
