#pragma once

#include <cuda_runtime.h>

// snap
#include <snap/snap.h>

#include "interp_point.h"
#include "x1_coeffs_device.h"

#define INP(j, i) (inp[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])

namespace snap {

// The shared-memory line holds variable v's inputs at line[v * axis_size + i];
// each output is the point function (interp_point.h) at its first input.
template <typename T, int N>
__device__ T interp_shared_poly_coeff_impl(T const *line, T const *coeff, int v,
                                           int start, int axis_size) {
  return poly_point<T, N>(line + v * axis_size + start, 1, coeff);
}

template <typename T>
__device__ T interp_shared_weno3_coeff_impl(T const *line, T const *coeff,
                                            int v, int start, int axis_size,
                                            bool scale) {
  return weno3_point(line + v * axis_size + start, 1, coeff, scale);
}

// Nu: coeff is one face's row of a per-face x1 table (kWeno5NuCoeff values),
// its three linear weights d after the nine stencils; otherwise the uniform
// table with the weights .3, .6, .1
template <typename T, bool Nu = false>
__device__ T interp_shared_weno5_coeff_impl(T const *line, T const *coeff,
                                            int v, int start, int axis_size,
                                            bool scale) {
  return weno5_point<T, Nu>(line + v * axis_size + start, 1, coeff, scale);
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

// One line per CUDA block, one output per thread: the block loads the line
// (blockDim.x inputs per variable) into shared memory, and the NCoeff weights
// too unless Nu; the last Stencil - 1 threads only load. Nu: coeff is a
// per-face x1 table, one row of NCoeff per output, read from global memory.
template <typename T, int Stencil, int NCoeff, typename Op, bool Nu = false>
__device__ void interp_line_block(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  T const *row;
  if constexpr (Nu) {
    row = coeff + static_cast<int64_t>(id) * NCoeff;
  } else {
    T *scoeff = smem + nt * nvar;
    for (int i = id; i < NCoeff; i += nt) {
      scoeff[i] = coeff[i];
    }
    row = scoeff;
  }

  bool active = id <= nt - Stencil;

  __syncthreads();

  if (!active) return;

  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) = Op::template eval<T>(sinp, row, j, id, nt, scale);
  }
}

// one thread per output of a per-face x1 table: weno5 rows, or poly rows of
// NCoeff weights
template <int NCoeff, typename T>
__device__ void interp_nu_untiled(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  if constexpr (NCoeff == kWeno5NuCoeff) {
    interp_line_block<T, kWeno5Stencil, NCoeff, Weno5NuOp, true>(
        out, inp, coeff, nvar, stride_in1, stride_in2, stride_out1,
        stride_out2, scale, smem);
  } else {
    interp_line_block<T, NCoeff, NCoeff, PolyOp<NCoeff>, true>(
        out, inp, coeff, nvar, stride_in1, stride_in2, stride_out1,
        stride_out2, scale, smem);
  }
}

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

#undef INP
#undef OUT
