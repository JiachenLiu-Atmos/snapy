#pragma once

// base
#include <configure.h>

// snap
#include "interp_point.h"

namespace snap {

// One face per call: the nvar variables of the stencil starting at inp
// (variable stride2, cell stride1) go to out (variable stride stride_out).
// The arithmetic is the point function's (interp_point.h).

// polynomial; coeff is one table, or one face's row of a per-face x1 table
template <typename T, int N>
DISPATCH_MACRO void interp_poly_impl(T* out, T const* inp, T const* coeff,
                                     int stride1, int stride2, int stride_out,
                                     int nvar) {
  for (int j = 0; j < nvar; ++j) {
    out[j * stride_out] = poly_point<T, N>(inp + j * stride2, stride1, coeff);
  }
};

// WENO 3 interpolation
template <typename T>
DISPATCH_MACRO void interp_weno3_impl(T* out, T* inp, T* coeff, int stride1,
                                      int stride2, int stride_out, int nvar,
                                      double scale) {
  for (int j = 0; j < nvar; ++j) {
    out[j * stride_out] =
        weno3_point(inp + j * stride2, stride1, coeff, scale != 0.);
  }
};

// WENO 5 interpolation. Nu: coeff is one face's row of a per-face x1 table
// (kWeno5NuCoeff values), its three linear weights d after the nine stencils;
// otherwise the uniform table with the weights .3, .6, .1
template <typename T, bool Nu = false>
DISPATCH_MACRO void interp_weno5_impl(T* out, T const* inp, T const* coeff,
                                      int stride1, int stride2, int stride_out,
                                      int nvar, double scale) {
  for (int j = 0; j < nvar; ++j) {
    out[j * stride_out] =
        weno5_point<T, Nu>(inp + j * stride2, stride1, coeff, scale != 0.);
  }
};

}  // namespace snap
