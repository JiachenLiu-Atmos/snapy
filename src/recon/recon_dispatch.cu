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

// One launcher for every variant: a line that fits one CUDA block (1024
// threads, the block limit) keeps the one-block kernel, split into launches of
// at most `per` variables when their shared memory does not fit; a longer line
// is walked in tiles (interp_line_tiled). Every output is the same point
// function either way.
template <int Stencil, int NCoeff, typename Op>
void call_interp_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                      bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_interp_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, Stencil, NCoeff,
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
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
              interp_line_block<scalar_t, Stencil, NCoeff, Op>(
                  out, w, c, nv, stride_in1, stride_in2, stride_out1,
                  stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == NCoeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, Stencil, NCoeff, Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}

template <int N>
void call_poly_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  call_interp_cuda<N, N, PolyOp<N>>(iter, coeff, dim, /*scale=*/false);
}

void call_weno3_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                     bool scale) {
  call_interp_cuda<kWeno3Stencil, kWeno3Coeff, Weno3Op>(iter, coeff, dim,
                                                        scale);
}

void call_weno5_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                     bool scale) {
  call_interp_cuda<kWeno5Stencil, kWeno5Coeff, Weno5Op>(iter, coeff, dim,
                                                        scale);
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly3, &snap::call_poly_cuda<3>);
REGISTER_CUDA_DISPATCH(call_poly5, &snap::call_poly_cuda<5>);
REGISTER_CUDA_DISPATCH(call_weno3, &snap::call_weno3_cuda);
REGISTER_CUDA_DISPATCH(call_weno5, &snap::call_weno5_cuda);

}  // namespace at::native
