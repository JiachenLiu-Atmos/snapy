#pragma once

#include <cuda_runtime.h>

// snap
#include <snap/snap.h>

#include "x1_coeffs_device.h"

#define INP(j, i) (inp[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])
#define SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
inline __device__ T _vvdot(T const *v1, T const *v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

template <typename T, int N>
__device__ T interp_shared_poly_coeff_impl(T const *line, T const *coeff, int v,
                                           int start, int axis_size) {
  T out = 0.;
  for (int k = 0; k < N; ++k) {
    out += coeff[k] * line[v * axis_size + start + k];
  }
  return out;
}

template <typename T>
__device__ T interp_shared_weno3_coeff_impl(T const *line, T const *coeff,
                                            int v, int start, int axis_size,
                                            bool scale) {
  T const *c1 = coeff;
  T const *c2 = c1 + 3;
  T const *c3 = c2 + 3;
  T const *c4 = c3 + 3;
  T const *src = line + v * axis_size + start;
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2])) / 3.0 : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[3];
  phi[0] = src[0] / vscale;
  phi[1] = src[1] / vscale;
  phi[2] = src[2] / vscale;

  T p0 = _vvdot<3>(phi, c1);
  T p1 = _vvdot<3>(phi, c2);

  T beta0 = SQR(_vvdot<3>(phi, c3));
  T beta1 = SQR(_vvdot<3>(phi, c4));

  T alpha0 = (2.0 / 3.0) / SQR(beta0 + 1e-6);
  T alpha1 = (1.0 / 3.0) / SQR(beta1 + 1e-6);

  return (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
}

// Nu: coeff is one face's row of a per-face x1 table (kWeno5NuCoeff values),
// its three linear weights d after the nine stencils; otherwise the uniform
// table with the weights .3, .6, .1
template <typename T, bool Nu = false>
__device__ T interp_shared_weno5_coeff_impl(T const *line, T const *coeff,
                                            int v, int start, int axis_size,
                                            bool scale) {
  T const *c1 = coeff;
  T const *c2 = c1 + 5;
  T const *c3 = c2 + 5;
  T const *c4 = c3 + 5;
  T const *c5 = c4 + 5;
  T const *c6 = c5 + 5;
  T const *c7 = c6 + 5;
  T const *c8 = c7 + 5;
  T const *c9 = c8 + 5;
  T const *src = line + v * axis_size + start;
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2]) +
                      fabs(src[3]) + fabs(src[4])) /
                         5.0
                   : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = src[k] / vscale;
  }

  T p0 = _vvdot<5>(phi, c1);
  T p1 = _vvdot<5>(phi, c2);
  T p2 = _vvdot<5>(phi, c3);

  T beta0 = 13. / 12. * SQR(_vvdot<5>(phi, c4)) + .25 * SQR(_vvdot<5>(phi, c5));
  T beta1 = 13. / 12. * SQR(_vvdot<5>(phi, c6)) + .25 * SQR(_vvdot<5>(phi, c7));
  T beta2 = 13. / 12. * SQR(_vvdot<5>(phi, c8)) + .25 * SQR(_vvdot<5>(phi, c9));

  T alpha0, alpha1, alpha2;
  if constexpr (Nu) {
    T const *d = c9 + 5;
    alpha0 = d[0] / SQR(beta0 + 1e-6);
    alpha1 = d[1] / SQR(beta1 + 1e-6);
    alpha2 = d[2] / SQR(beta2 + 1e-6);
  } else {
    alpha0 = .3 / SQR(beta0 + 1e-6);
    alpha1 = .6 / SQR(beta1 + 1e-6);
    alpha2 = .1 / SQR(beta2 + 1e-6);
  }

  return vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
         (alpha0 + alpha1 + alpha2);
}

// polynomial. Nu: coeff is a per-face x1 table, one row of N per output,
// read from global memory; otherwise one table, loaded into shared memory
template <typename T, int N, bool Nu = false>
__device__ void interp_poly_impl(T *out, T *inp, T const *coeff, int nvar,
                                 int stride_in1, int stride_in2,
                                 int stride_out1, int stride_out2, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T const *row;
  if constexpr (Nu) {
    row = coeff + static_cast<int64_t>(id) * N;
  } else {
    T *scoeff = smem + nt * nvar;
    for (int i = id; i < N; i += nt) {
      scoeff[i] = coeff[i];
    }
    row = scoeff;
  }

  bool active = id <= nt - N;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) = interp_shared_poly_coeff_impl<T, N>(sinp, row, j, id, nt);
  }
};

// weno3
template <typename T>
__device__ void interp_weno3_impl(T *out, T *inp, T *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + nt * nvar;
  constexpr int N = 12;  // Number of coefficients for WENO3
  for (int i = id; i < N; i += nt) {
    scoeff[i] = coeff[i];
  }

  bool active = id <= nt - 3;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) = interp_shared_weno3_coeff_impl(sinp, scoeff, j, id, nt, scale);
  }
};

// weno5. Nu: coeff is a per-face x1 table, one row of kWeno5NuCoeff per
// output, read from global memory; otherwise one table, loaded into shared
// memory
template <typename T, bool Nu = false>
__device__ void interp_weno5_impl(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T const *row;
  if constexpr (Nu) {
    row = coeff + static_cast<int64_t>(id) * kWeno5NuCoeff;
  } else {
    T *scoeff = smem + nt * nvar;
    constexpr int N = 45;  // Number of coefficients for WENO5
    for (int i = id; i < N; i += nt) {
      scoeff[i] = coeff[i];
    }
    row = scoeff;
  }

  bool active = id <= nt - 5;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) =
        interp_shared_weno5_coeff_impl<T, Nu>(sinp, row, j, id, nt, scale);
  }
};

// one thread per output of a per-face x1 table: weno5 rows, or poly rows of
// NCoeff weights
template <int NCoeff, typename T>
__device__ void interp_nu_untiled(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  if constexpr (NCoeff == kWeno5NuCoeff) {
    interp_weno5_impl<T, true>(out, inp, coeff, nvar, stride_in1, stride_in2,
                               stride_out1, stride_out2, scale, smem);
  } else {
    interp_poly_impl<T, NCoeff, true>(out, inp, coeff, nvar, stride_in1,
                                      stride_in2, stride_out1, stride_out2,
                                      smem);
  }
}

// Coeff counts of the matrices in weno3.cpp / weno5.cpp. Polynomial
// reconstruction uses Stencil coefficients, so it does not need one.
constexpr int kWeno3Stencil = 3;
constexpr int kWeno3Coeff = 12;
constexpr int kWeno5Stencil = 5;
constexpr int kWeno5Coeff = 45;

struct Weno3Op {
  template <typename T>
  __device__ static T eval(T const *line, T const *coeff, int v, int start,
                           int axis, bool scale) {
    return interp_shared_weno3_coeff_impl(line, coeff, v, start, axis, scale);
  }
};

struct Weno5Op {
  template <typename T>
  __device__ static T eval(T const *line, T const *coeff, int v, int start,
                           int axis, bool scale) {
    return interp_shared_weno5_coeff_impl(line, coeff, v, start, axis, scale);
  }
};

// one face's row of a per-face x1 weno5 table
struct Weno5NuOp {
  template <typename T>
  __device__ static T eval(T const *line, T const *row, int v, int start,
                           int axis, bool scale) {
    return interp_shared_weno5_coeff_impl<T, true>(line, row, v, start, axis,
                                                   scale);
  }
};

template <int N>
struct PolyOp {
  template <typename T>
  __device__ static T eval(T const *line, T const *coeff, int v, int start,
                           int axis, bool /*scale*/) {
    return interp_shared_poly_coeff_impl<T, N>(line, coeff, v, start, axis);
  }
};

// One CUDA block cannot hold a line longer than 1024 threads. Walk that
// line in tiles of blockDim.x outputs. A tile loads blockDim.x + Stencil - 1
// inputs (the overhang) and calls the same shared-memory weights as the
// one-thread-per-cell kernel. Nu: coeff is a per-face x1 table, one row of
// NCoeff per output, read from global memory.
template <typename T, int Stencil, int NCoeff, typename Op, bool Nu = false>
__device__ void interp_line_tiled(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  int nline, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;
  int overhang = Stencil - 1;
  int window_max = nt + overhang;
  int nout = nline - overhang;

  T *scoeff = smem + window_max * nvar;
  if constexpr (!Nu) {
    for (int i = id; i < NCoeff; i += nt) {
      scoeff[i] = coeff[i];
    }
  }

  for (int start = 0; start < nout; start += nt) {
    int n_out = nout - start;
    if (n_out > nt) n_out = nt;
    int window = n_out + overhang;

    T *sinp = smem;
    for (int j = 0; j < nvar; ++j) {
      for (int i = id; i < window; i += nt) {
        sinp[j * window + i] = INP(j, start + i);
      }
    }

    __syncthreads();

    if (id < n_out) {
      T const *row = scoeff;
      if constexpr (Nu) row = coeff + static_cast<int64_t>(start + id) * NCoeff;
      for (int j = 0; j < nvar; ++j) {
        OUT(j, start + id) =
            Op::template eval<T>(sinp, row, j, id, window, scale);
      }
    }

    __syncthreads();
  }
}

}  // namespace snap

#undef SQR
#undef INP
#undef OUT
