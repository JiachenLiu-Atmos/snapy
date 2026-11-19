// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/utils/loops.cuh>

#include "riemann_dispatch.hpp"
#include "riemann_kernels.h"

namespace snap {

// One CUDA loop for every face-function kind: N operands, each face is
// riemann_face<K>.
template <RiemannKind K, int N>
void riemann_cuda_loop(at::TensorIterator &iter, int dim,
                       RiemannParams const &params,
                       torch::Tensor const &inv_mu_ratio_m1,
                       torch::Tensor const &cv_ratio_m1,
                       torch::Tensor const &u0) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "riemann_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    bool moist = params.ideal_moist;
    RiemannArgs<scalar_t> a{
        dim,
        static_cast<int>(nhydro - ICY),
        static_cast<int>(stride),
        N == 8,
        moist,
        params.nvapor,
        scalar_t(params.gammad),
        moist ? inv_mu_ratio_m1.data_ptr<scalar_t>() : nullptr,
        moist ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr,
        moist ? u0.data_ptr<scalar_t>() : nullptr,
        params.dir_yz};

    native::gpu_kernel<N>(
        iter, [=] GPU_LAMBDA(char *const data[N], unsigned int strides[N]) {
          scalar_t *p[8];
#pragma unroll
          for (int k = 0; k < N; ++k)
            p[k] = reinterpret_cast<scalar_t *>(data[k] + strides[k]);
          riemann_face<K>(p, a);
        });
  });
}

void riemann_cuda(at::TensorIterator &iter, int kind, int dim,
                  RiemannParams const &params,
                  torch::Tensor const &inv_mu_ratio_m1,
                  torch::Tensor const &cv_ratio_m1, torch::Tensor const &u0) {
  at::cuda::CUDAGuard device_guard(iter.device());
  switch (static_cast<RiemannKind>(kind)) {
    case RiemannKind::hllc:
      if (iter.ninputs() == 6)
        return riemann_cuda_loop<RiemannKind::hllc, 8>(
            iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
      return riemann_cuda_loop<RiemannKind::hllc, 7>(
          iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
    case RiemannKind::lmars:
      return riemann_cuda_loop<RiemannKind::lmars, 6>(
          iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
    case RiemannKind::roe:
      return riemann_cuda_loop<RiemannKind::roe, 7>(
          iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
    case RiemannKind::shallow_roe:
      return riemann_cuda_loop<RiemannKind::shallow_roe, 3>(
          iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
    default:
      TORCH_CHECK(false, "[call_riemann] kind ", kind,
                  " is a torch formula, not a face function");
  }
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_riemann, &snap::riemann_cuda);

}  // namespace at::native
