// torch
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/native/cuda/Loops.cuh>

// snap
#include "step_dt.hpp"

namespace snap {

//! s = fma(a, x, s) elementwise, a a 0-dim device scalar: add's alpha form
//! (a + alpha * b, contracted), so s.add_(x, alpha) with alpha on the device
void axpy_device_cuda(at::Tensor& s, at::Tensor const& x, at::Tensor const& a) {
  at::cuda::CUDAGuard device_guard(s.device());
  auto alpha = a.to(s.scalar_type());
  auto xs = x.to(s.scalar_type());
  auto iter = at::TensorIteratorConfig()
                  .add_output(s)
                  .add_const_input(s)
                  .add_const_input(xs)
                  .add_const_input(alpha)
                  .build();
  AT_DISPATCH_FLOATING_TYPES(s.scalar_type(), "axpy_device_cuda", [&]() {
    at::native::gpu_kernel(iter,
                           [] __host__ __device__(scalar_t sv, scalar_t xv,
                                                  scalar_t av) -> scalar_t {
                             return ::fma(av, xv, sv);
                           });
  });
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(axpy_device, &snap::axpy_device_cuda);

}  // namespace at::native
