#pragma once

// C/C++
#include <cmath>

// base
#include <configure.h>

// snap
#include <snap/utils/sqr.h>

// One definition of each reconstruction at one face, for the CPU and CUDA
// line loops; constants typed T (V3 C9). See lane-c-notes.md.

namespace snap {

//! every reconstruction scheme. dc, plm and ppm are tensor code; the other
//! four are stencil kernels with a coefficient table
enum class Scheme : int { dc, plm, ppm, cp3, weno3, cp5, weno5 };

//! stencil width and coefficient count of each scheme. ncoeff is the uniform
//! table; ncoeff_nu is one face's row of a per-face x1 table (0: the scheme
//! has none). A weno5 row, uniform or per-face, holds its three linear weights
//! d after the nine stencil rows (45 + 3).
template <Scheme S>
struct SchemeTraits {
  static constexpr int stencil =
      S == Scheme::dc                                              ? 1
      : S == Scheme::plm || S == Scheme::cp3 || S == Scheme::weno3 ? 3
                                                                   : 5;
  static constexpr int ncoeff = S == Scheme::cp3     ? 3
                                : S == Scheme::weno3 ? 12
                                : S == Scheme::cp5   ? 5
                                : S == Scheme::weno5 ? 48
                                                     : 0;
  static constexpr int ncoeff_nu = S == Scheme::cp5     ? 5
                                   : S == Scheme::weno5 ? 48
                                                        : 0;
};

constexpr int kWeno5NuCoeff = SchemeTraits<Scheme::weno5>::ncoeff_nu;
constexpr int kPoly5NuCoeff = SchemeTraits<Scheme::cp5>::ncoeff_nu;

template <int N, typename T>
DISPATCH_MACRO inline T _vvdot(T const* v1, T const* v2) {
  T out = T(0);
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

//! polynomial: sum_k c[k] * src[k]
template <typename T, int N>
DISPATCH_MACRO inline T poly_point(T const* src, int stride, T const* c) {
  T out = T(0);
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
  T vscale = scale ? (std::fabs(src[0]) + std::fabs(src[stride]) +
                      std::fabs(src[2 * stride])) /
                         T(3.0)
                   : T(1.0);

  if (vscale == T(0)) return T(0);

  T phi[3];
  for (int k = 0; k < 3; ++k) {
    phi[k] = src[k * stride] / vscale;
  }

  T p0 = _vvdot<3>(phi, c1);
  T p1 = _vvdot<3>(phi, c2);

  T beta0 = sqr(_vvdot<3>(phi, c3));
  T beta1 = sqr(_vvdot<3>(phi, c4));

  T alpha0 = T(2.0 / 3.0) / sqr(beta0 + T(1e-6));
  T alpha1 = T(1.0 / 3.0) / sqr(beta1 + T(1e-6));

  return (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
}

//! WENO5: c = three candidate rows, six smoothness rows (5 each), then the
//! three linear weights d. c is one face's row of a per-face x1 table or the
//! uniform table, whose d is .3, .6, .1 (V3 C9: both read d, so the weights
//! are in the table's dtype)
template <typename T>
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
  T vscale = scale ? (std::fabs(src[0]) + std::fabs(src[stride]) +
                      std::fabs(src[2 * stride]) + std::fabs(src[3 * stride]) +
                      std::fabs(src[4 * stride])) /
                         T(5.0)
                   : T(1.0);

  if (vscale == T(0)) return T(0);

  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = src[k * stride] / vscale;
  }

  T p0 = _vvdot<5>(phi, c1);
  T p1 = _vvdot<5>(phi, c2);
  T p2 = _vvdot<5>(phi, c3);

  T beta0 =
      T(13. / 12.) * sqr(_vvdot<5>(phi, c4)) + T(.25) * sqr(_vvdot<5>(phi, c5));
  T beta1 =
      T(13. / 12.) * sqr(_vvdot<5>(phi, c6)) + T(.25) * sqr(_vvdot<5>(phi, c7));
  T beta2 =
      T(13. / 12.) * sqr(_vvdot<5>(phi, c8)) + T(.25) * sqr(_vvdot<5>(phi, c9));

  T const* d = c9 + 5;
  T alpha0 = d[0] / sqr(beta0 + T(1e-6));
  T alpha1 = d[1] / sqr(beta1 + T(1e-6));
  T alpha2 = d[2] / sqr(beta2 + T(1e-6));

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
    return weno5_point<T>(src, stride, row, scale);
  }
}

}  // namespace snap
