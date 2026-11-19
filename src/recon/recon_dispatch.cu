// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

// snap
#include <snap/utils/loops.cuh>

#include "recon_dispatch.hpp"
#include "recon_line.cuh"
#include "x1_coeffs.hpp"

namespace snap {

// One launcher for every stencil scheme: a line that fits one CUDA block (1024
// threads, the block limit) keeps the one-block kernel, split into launches of
// at most `per` variables when their shared memory does not fit; a longer line
// is walked in tiles (recon_line_tiled). Uniform: the table goes to shared
// memory. Nu: the per-face rows stay in global memory, indexed by face, so
// each part reads the same rows. Every output is the same point function
// either way.
template <Scheme S, bool Nu>
void recon_line_cuda(at::TensorIterator& iter, at::Tensor rows, int dim,
                     bool scale, char const* what) {
  constexpr int Stencil = SchemeTraits<S>::stencil;
  constexpr int NCoeff =
      Nu ? SchemeTraits<S>::ncoeff_nu : SchemeTraits<S>::ncoeff;
  using Op = ReconOp<S, Nu>;

  at::cuda::CUDAGuard device_guard(iter.device());
  if constexpr (Nu) x1_coeffs_check_rows(iter, rows, dim, NCoeff, what);

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "recon_line_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = rows.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    int64_t nshared = Nu ? 0 : rows.numel();
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, Stencil, Nu ? 0 : NCoeff,
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per =
          native::stencil_vars_per_launch(nline64, nshared, sizeof(scalar_t));
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, nshared,
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              recon_line_block<scalar_t, Stencil, NCoeff, Op, Nu>(
                  out, w, c, nv, stride_in1, stride_in2, stride_out1,
                  stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    if constexpr (!Nu) TORCH_CHECK(rows.numel() == NCoeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          recon_line_tiled<scalar_t, Stencil, NCoeff, Op, Nu>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}

void call_recon_cuda(at::TensorIterator& iter, at::Tensor rows, int dim,
                     int scheme, bool scale, int64_t row_stride) {
  switch (static_cast<Scheme>(scheme)) {
    case Scheme::cp3:
      return recon_line_cuda<Scheme::cp3, false>(iter, rows, dim, false,
                                                 "cp3");
    case Scheme::weno3:
      return recon_line_cuda<Scheme::weno3, false>(iter, rows, dim, scale,
                                                   "weno3");
    case Scheme::cp5:
      if (row_stride) {
        return recon_line_cuda<Scheme::cp5, true>(iter, rows, dim, false,
                                                  "poly5_nu");
      }
      return recon_line_cuda<Scheme::cp5, false>(iter, rows, dim, false,
                                                 "cp5");
    case Scheme::weno5:
      if (row_stride) {
        return recon_line_cuda<Scheme::weno5, true>(iter, rows, dim, scale,
                                                    "weno5_nu");
      }
      return recon_line_cuda<Scheme::weno5, false>(iter, rows, dim, scale,
                                                   "weno5");
    default:
      TORCH_CHECK(false, "[call_recon] scheme ", scheme,
                  " has no stencil kernel");
  }
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_recon, &snap::call_recon_cuda);

}  // namespace at::native
