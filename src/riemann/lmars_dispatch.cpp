// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/utils/on_host.hpp>

#include "riemann_dispatch.hpp"

// impl
#include "lmars_impl.h"

namespace snap {

void call_lmars_cpu(at::TensorIterator& iter, int dim) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_lmars_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto face_pressure =
                reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wl = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wr = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto elr = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto glr = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            lmars_impl(out, wl, wr, *elr, *(elr + stride), *glr,
                       *(glr + stride), dim, ny, stride, stride, face_pressure);
          }
        },
        grain_size);
  });
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_lmars);
REGISTER_ALL_CPU_DISPATCH(call_lmars, &snap::call_lmars_cpu);
REGISTER_MPS_DISPATCH(call_lmars, &snap::OnHost<&snap::call_lmars_cpu>::run);

}  // namespace at::native
