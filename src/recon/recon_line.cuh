#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <array>
#include <limits>

// snap
#include <snap/snap.h>

#include "recon_point.h"

// The CUDA line kernel of every stencil scheme and its tiled launch
// (recon_dispatch.cu). Include snap/utils/loops.cuh first: it has no include
// guard, so this header does not include it again.

#define INP(j, i) (inp[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])

namespace snap {

// The shared-memory line holds variable v's inputs at line[v * axis_size + i];
// each output is recon_point at its first input. Nu: row is one face's row of
// a per-face x1 table; otherwise the uniform table.
template <Scheme S, bool Nu = false>
struct ReconOp {
  template <typename T>
  __device__ static T eval(T const *line, T const *row, int v, int start,
                           int axis, bool scale) {
    return recon_point<S, T, Nu>(line + v * axis + start, 1, row, scale);
  }
};

// One line per CUDA block, one output per thread: the block loads the line
// (blockDim.x inputs per variable) into shared memory, and the NCoeff weights
// too unless Nu; the last Stencil - 1 threads only load. Nu: coeff is a
// per-face x1 table, one row of NCoeff per output, read from global memory.
template <typename T, int Stencil, int NCoeff, typename Op, bool Nu = false>
__device__ void recon_line_block(T *out, T *inp, T const *coeff, int nvar,
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

// One CUDA block cannot hold a line longer than 1024 threads. Walk that
// line in tiles of blockDim.x outputs. A tile loads blockDim.x + Stencil - 1
// inputs (the overhang) and calls the same shared-memory weights as the
// one-thread-per-cell kernel. Nu: coeff is a per-face x1 table, one row of
// NCoeff per output, read from global memory.
template <typename T, int Stencil, int NCoeff, typename Op, bool Nu = false>
__device__ void recon_line_tiled(T *out, T *inp, T const *coeff, int nvar,
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

// 0 means the line fits in one CUDA block, so the caller keeps stencil_kernel.
// Otherwise returns the tile width and the dynamic shared-memory size for
// (tile + stencil - 1) inputs per variable plus ncoeff weights (0 for a
// per-face x1 table, read from global memory).
inline int recon_tile_width(int64_t nline, int nvar, int stencil,
                            int64_t ncoeff, size_t scalar_size,
                            size_t* shared_bytes) {
  if (nline <= 1024) {
    return 0;
  }
  TORCH_CHECK(nline <= std::numeric_limits<int>::max(),
              "recon line does not fit in int: ", nline);
  TORCH_CHECK(nvar > 0 && stencil >= 2 && ncoeff >= 0);
  constexpr size_t kSharedLimit = 48 * 1024;
  int tile = 256;
  auto bytes = [&](int t) -> size_t {
    return (static_cast<size_t>(t + stencil - 1) * static_cast<size_t>(nvar) +
            static_cast<size_t>(ncoeff)) *
           scalar_size;
  };
  while (tile > 1 && bytes(tile) > kSharedLimit) {
    tile >>= 1;
  }
  TORCH_CHECK(bytes(tile) <= kSharedLimit,
              "recon tile does not fit in shared memory, nvar=", nvar,
              " stencil=", stencil, " ncoeff=", ncoeff);
  *shared_bytes = bytes(tile);
  return tile;
}

// Same grid as stencil_kernel (one block per line), but block.x is the
// tile width instead of the line length. grid.x stays 1: blockIdx is the
// line id, so a second block along the line would read the wrong offset.
template <typename scalar_t, int Arity, typename func_t>
void stencil_kernel_tiled(at::TensorIterator& iter, int dim, int block_x,
                          size_t shared_bytes, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);
  TORCH_CHECK(block_x >= 1 && block_x <= 1024);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.input().numel();

  TORCH_INTERNAL_ASSERT(numel >= 0 &&
                        numel <= std::numeric_limits<int32_t>::max());
  if (numel == 0) {
    return;
  }

  int len[3] = {1, 1, 1};
  int ndim = iter.input().dim();
  len[3 + dim - ndim] = at::native::ensure_nonempty_size(iter.input(), dim);
  native::_left_shift<3>(len, dim + 1 - ndim);
  dim3 line_block(len[2], len[1], len[0]);
  TORCH_CHECK(line_block.y == 1 && line_block.z == 1);
  TORCH_CHECK(line_block.x > static_cast<unsigned>(block_x));

  if (ndim <= 3) {
    len[3 - ndim] = at::native::ensure_nonempty_size(iter.input(), 0);
  }
  for (int i = 1; i < ndim; ++i) {
    len[3 + i - ndim] = at::native::ensure_nonempty_size(iter.input(), i);
  }
  native::_left_shift<3>(len, dim + 1 - ndim);

  dim3 grid(len[2] / line_block.x, len[1] / line_block.y,
            len[0] / line_block.z);
  TORCH_CHECK(grid.x == 1);
  dim3 block(block_x, line_block.y, line_block.z);

  auto stream = at::cuda::getCurrentCUDAStream();
  native::reduce_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      numel, [=] __device__(int bid, scalar_t* smem) {
        auto offsets = offset_calc.get(bid);
        f(data.data(), offsets.data(), smem);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

#undef INP
#undef OUT
