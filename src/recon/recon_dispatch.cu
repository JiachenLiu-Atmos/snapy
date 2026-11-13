// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

// snap
#include <snap/utils/loops.cuh>

#include "interp_impl.cuh"
#include "recon_dispatch.hpp"
#include "recon_launch.cuh"

namespace snap {

template <int N>
void call_poly_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_poly_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, N, coeff.numel(),
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_poly_impl<scalar_t, N>(out, w, c, nvar, stride_in1,
                                            stride_in2, stride_out1,
                                            stride_out2, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_poly_impl<scalar_t, N>(out, w, c, nv, stride_in1,
                                            stride_in2, stride_out1,
                                            stride_out2, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == N);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, N, N, PolyOp<N>>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              /*scale=*/false, nline, smem);
        });
  });
}

void call_weno3_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                     bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno3_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, kWeno3Stencil, kWeno3Coeff,
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_weno3_impl(out, w, c, nvar, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_weno3_impl(out, w, c, nv, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == kWeno3Coeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, kWeno3Stencil, kWeno3Coeff, Weno3Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}

void call_weno5_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                     bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno5_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, kWeno5Stencil, kWeno5Coeff,
                                sizeof(scalar_t), &shared);

    // 1024 threads is the CUDA block limit. This launch is the one that
    // already ships, so a line that fits stays on it.
    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_weno5_impl(out, w, c, nvar, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_weno5_impl(out, w, c, nv, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == kWeno5Coeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, kWeno5Stencil, kWeno5Coeff, Weno5Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly3, &snap::call_poly_cuda<3>);
REGISTER_CUDA_DISPATCH(call_poly5, &snap::call_poly_cuda<5>);
REGISTER_CUDA_DISPATCH(call_weno3, &snap::call_weno3_cuda);
REGISTER_CUDA_DISPATCH(call_weno5, &snap::call_weno5_cuda);

}  // namespace at::native
