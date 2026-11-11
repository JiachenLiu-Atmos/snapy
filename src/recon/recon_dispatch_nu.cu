#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>
#include <snap/utils/loops.cuh>

#include "interp_impl.cuh"
#include "recon_dispatch_nu.hpp"
#include "recon_launch.cuh"
#include "x1_coeffs.hpp"

namespace snap {

template <int Stencil, int NCoeff, typename Op>
void call_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                  bool scale, char const* what) {
  at::cuda::CUDAGuard device_guard(iter.device());
  x1_coeffs_check_rows(iter, coeff, dim, NCoeff, what);

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_nu_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);
    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, Stencil, /*ncoeff=*/0,
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      // One launch holds every variable's whole line in shared memory; more
      // variables than fit go in parts of at most `per`. The rows are in
      // global memory and indexed by face, so each part reads the same rows.
      int per = native::stencil_vars_per_launch(nline64, 0, sizeof(scalar_t));
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, /*buffers=*/0,
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_nu_untiled<NCoeff>(out, w, c, nv, stride_in1, stride_in2,
                                        stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, Stencil, NCoeff, Op, true>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}

void call_weno5_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                        bool scale) {
  call_nu_cuda<kWeno5Stencil, kWeno5NuCoeff, Weno5NuOp>(iter, coeff, dim, scale,
                                                        "weno5_nu");
}

void call_poly5_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  call_nu_cuda<kPoly5NuCoeff, kPoly5NuCoeff, PolyOp<5>>(iter, coeff, dim, false,
                                                        "poly5_nu");
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly5_nu, &snap::call_poly5_nu_cuda);
REGISTER_CUDA_DISPATCH(call_weno5_nu, &snap::call_weno5_nu_cuda);

}  // namespace at::native
