// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "hllc_impl.h"
#include "lmars_impl.h"
#include "riemann_dispatch.hpp"
#include "roe_impl.h"
#include "shallow_roe_impl.h"
#include <snap/utils/loops.cuh>

namespace snap {

void call_lmars_cuda(at::TensorIterator &iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_lmars_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;

    native::gpu_kernel<6>(iter, [=] GPU_LAMBDA(char *const data[6],
                                               unsigned int strides[6]) {
      auto out = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
      auto face_pressure = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
      auto wl = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
      auto wr = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
      auto elr = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
      auto glr = reinterpret_cast<scalar_t *>(data[5] + strides[5]);
      lmars_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride), dim,
                 ny, stride, stride, face_pressure);
    });
  });
}

template <int N>
void hllc_cuda_impl(at::TensorIterator &iter, int dim) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_hllc_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;

    native::gpu_kernel<N>(iter, [=] GPU_LAMBDA(char *const data[N],
                                               unsigned int strides[N]) {
      auto out = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
      auto face_pressure = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
      auto wl = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
      auto wr = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
      auto elr = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
      auto glr = reinterpret_cast<scalar_t *>(data[5] + strides[5]);
      auto clr = reinterpret_cast<scalar_t *>(data[6] + strides[6]);
      scalar_t p0 = 0;
      if constexpr (N == 8)
        p0 = *reinterpret_cast<scalar_t *>(data[7] + strides[7]);
      hllc_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride), *clr,
                *(clr + stride), dim, ny, stride, stride, face_pressure, p0);
    });
  });
}

void call_hllc_cuda(at::TensorIterator &iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());
  if (iter.ninputs() == 6)
    hllc_cuda_impl<8>(iter, dim);
  else
    hllc_cuda_impl<7>(iter, dim);
}

void call_roe_cuda(at::TensorIterator &iter, int dim, bool ideal_moist,
                   int nvapor, double gammad,
                   torch::Tensor const &inv_mu_ratio_m1,
                   torch::Tensor const &cv_ratio_m1, torch::Tensor const &u0) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_roe_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;
    auto inv_mu = ideal_moist ? inv_mu_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto cv = ideal_moist ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto energy0 = ideal_moist ? u0.data_ptr<scalar_t>() : nullptr;

    native::gpu_kernel<7>(iter, [=] GPU_LAMBDA(char *const data[7],
                                               unsigned int strides[7]) {
      auto out = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
      auto face_pressure = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
      auto wl = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
      auto wr = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
      auto elr = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
      auto glr = reinterpret_cast<scalar_t *>(data[5] + strides[5]);
      auto clr = reinterpret_cast<scalar_t *>(data[6] + strides[6]);
      roe_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride), *clr,
               *(clr + stride), dim, ny, ideal_moist, nvapor, scalar_t(gammad),
               inv_mu, cv, energy0, stride, stride, face_pressure);
    });
  });
}

void call_shallow_roe_cuda(at::TensorIterator &iter, int dim, int dir_yz) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_shallow_roe_cuda", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    native::gpu_kernel<3>(
        iter, [=] GPU_LAMBDA(char *const data[3], unsigned int strides[3]) {
          auto out = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto wl = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto wr = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          shallow_roe_impl(out, wl, wr, dim, dir_yz, stride, stride);
        });
  });
}
} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_lmars, &snap::call_lmars_cuda);
REGISTER_CUDA_DISPATCH(call_hllc, &snap::call_hllc_cuda);
REGISTER_CUDA_DISPATCH(call_roe, &snap::call_roe_cuda);
REGISTER_CUDA_DISPATCH(call_shallow_roe, &snap::call_shallow_roe_cuda);

} // namespace at::native
