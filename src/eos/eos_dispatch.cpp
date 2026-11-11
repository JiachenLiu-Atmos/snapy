// C/C++
#include <atomic>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/utils/on_host.hpp>

#include "eos_dispatch.hpp"
#include "fix_vapor_impl.h"
#include "ideal_gas_impl.h"
// #include "ideal_moist_impl.h"

namespace snap {

void ideal_gas_cons2prim_cpu(at::TensorIterator& iter, double gammad) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ideal_gas_cons2prim_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto prim = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto cons = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto cos_theta =
                reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            ideal_gas_cons2prim(prim, cons, cos_theta, gammad, stride);
          }
        },
        grain_size);
  });
}

void ideal_gas_cons2prim_mps(at::TensorIterator& iter, double gammad) {
  auto prim = iter.output();
  auto cons = iter.input(0);
  auto cos_theta = iter.input(1);

  // den -> den
  prim[IDN] = cons[IDN];

  // mom -> vel
  prim.narrow(0, IVX, 3) = cons.narrow(0, IVX, 3) / prim[IDN];

  coord_vec_raise_(prim.narrow(0, IVX, 3), cos_theta[0]);

  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // eng -> pr
  prim[IPR] = (gammad - 1) * (cons[IPR] - ke);
}

/*void call_ideal_moist_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_ideal_moist_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nmass = nhydro - 5;

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto prim = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto cons = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto gammad = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto feps = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        auto fsig = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
        ideal_moist_cons2prim(prim, cons, gammad, feps, fsig, nmass, stride);
      }
    });
  });
}*/

int call_fix_vapor_cpu(at::TensorIterator& iter, bool /*count*/,
                       at::Tensor const& /*sink*/) {
  int grain_size = iter.numel() / at::get_num_threads();
  // for_each is parallel
  std::atomic<int> all_err{0};

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_fix_vapor_cpu", [&] {
    auto nx1 = at::native::ensure_nonempty_size(iter.output(), -1);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto vapor = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto major = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto vol = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            int err = fix_vapor_impl(vapor, major, vol, nx1);
            if (err) all_err.fetch_add(err, std::memory_order_relaxed);
          }
        },
        grain_size);
  });

  return all_err.load();
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(ideal_gas_cons2prim);
DEFINE_DISPATCH(call_fix_vapor);

REGISTER_ALL_CPU_DISPATCH(ideal_gas_cons2prim, &snap::ideal_gas_cons2prim_cpu);
REGISTER_ALL_CPU_DISPATCH(call_fix_vapor, &snap::call_fix_vapor_cpu);

REGISTER_MPS_DISPATCH(ideal_gas_cons2prim, &snap::ideal_gas_cons2prim_mps);
REGISTER_MPS_DISPATCH(call_fix_vapor,
                      &snap::OnHost<&snap::call_fix_vapor_cpu>::run);

}  // namespace at::native
