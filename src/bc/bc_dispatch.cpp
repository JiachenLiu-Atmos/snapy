// C/C++
#include <atomic>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "bc_dispatch.hpp"
#include "flip_zero_impl.h"

namespace snap {

int flip_zero_cpu(at::TensorIterator& iter, int dim, int dir) {
  std::atomic<int> num_flips{0};
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "flip_zero_cpu", [&] {
    if constexpr (std::is_signed<scalar_t>::value) {
      auto len = at::native::ensure_nonempty_size(iter.output(), dim);
      auto stride = at::native::ensure_nonempty_stride(iter.output(), dim);

      iter.for_each(
          [&](char** data, const int64_t* strides, int64_t n) {
            for (int i = 0; i < n; i++) {
              auto solid =
                  reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
              auto dp = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
              auto fromLen =
                  reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
              auto fromBit =
                  reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
              auto usedFlip =
                  reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);

              num_flips.fetch_add(
                  compute_min_flips<scalar_t>(solid, len, /*minRun0=*/3,
                                              /*minRun1=*/2,
                                              /*allowBothFlips=*/0, dir, stride,
                                              dp, fromLen, fromBit, usedFlip),
                  std::memory_order_relaxed);

              reconstruct_solution<scalar_t>(solid, len, /*minRun0=*/3,
                                             /*minRun1=*/2,
                                             /*allowBothFlips=*/0, dir, stride,
                                             fromLen, fromBit, usedFlip);
            }
          },
          grain_size);
    }
  });

  return num_flips.load(std::memory_order_relaxed);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(flip_zero);

REGISTER_ALL_CPU_DISPATCH(flip_zero, &snap::flip_zero_cpu);
REGISTER_MPS_DISPATCH(flip_zero, &snap::OnHost<&snap::flip_zero_cpu>::run);

}  // namespace at::native
