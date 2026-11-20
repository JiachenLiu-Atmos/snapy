// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// snap
#include <snap/snap.h>

#include <snap/utils/on_host.hpp>

#include "riemann_dispatch.hpp"
#include "shallow_roe_impl.h"

namespace snap {

void call_shallow_roe_cpu(at::TensorIterator& iter, int dim, int dir_yz) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_shallow_roe_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t i = 0; i < n; ++i) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto wl = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wr = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            shallow_roe_impl(out, wl, wr, dim, dir_yz, stride, stride);
          }
        },
        grain_size);
  });
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_shallow_roe);
REGISTER_ALL_CPU_DISPATCH(call_shallow_roe, &snap::call_shallow_roe_cpu);
REGISTER_MPS_DISPATCH(call_shallow_roe,
                      &snap::OnHost<&snap::call_shallow_roe_cpu>::run);

}  // namespace at::native
