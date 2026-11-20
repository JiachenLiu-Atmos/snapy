// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/utils/loops.cuh>
#include "eos_dispatch.hpp"
#include "ideal_gas_impl.h"
#include "fix_vapor_impl.h"

namespace snap {

void ideal_gas_cons2prim_cuda(at::TensorIterator& iter, double gammad) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ideal_gas_cons2prim_cuda", [&]() {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    native::gpu_kernel<3>(
        iter, [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3]) {
          auto prim = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto cons = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto cos_theta = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          ideal_gas_cons2prim(prim, cons, cos_theta, gammad, stride);
        });
  });
}

int call_fix_vapor_cuda(at::TensorIterator& iter, bool count,
                        at::Tensor const& sink) {
  at::cuda::CUDAGuard device_guard(iter.device());

  // count failures on the device (a host int captured by value never reported)
  // only when the caller reads them: the read synchronises the device. A
  // defined sink takes the count instead, read later by the caller.
  bool read = count && !sink.defined();
  auto nerr = !count ? at::Tensor()
              : read ? at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(iter.device()))
                     : sink;
  int* nerr_ptr = count ? nerr.data_ptr<int>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_fix_vapor_cuda", [&] {
    auto nx1 = at::native::ensure_nonempty_size(iter.output(), -1);

    native::gpu_kernel<3>(
        iter, [=] __device__(char* const data[3], unsigned int strides[3]) {
          auto vapor = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto major = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto vol = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          int err = fix_vapor_impl(vapor, major, vol, nx1);
          if (err && nerr_ptr) atomicAdd(nerr_ptr, err);
        });
  });

  return read ? nerr.item<int>() : 0;
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(ideal_gas_cons2prim, &snap::ideal_gas_cons2prim_cuda);
REGISTER_CUDA_DISPATCH(call_fix_vapor, &snap::call_fix_vapor_cuda);

}  // namespace at::native
