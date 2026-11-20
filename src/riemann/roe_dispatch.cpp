// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// snap
#include <snap/snap.h>

#include <snap/utils/on_host.hpp>

#include "riemann_dispatch.hpp"
#include "roe_impl.h"

namespace snap {

void call_roe_cpu(at::TensorIterator& iter, int dim, bool ideal_moist,
                  int nvapor, double gammad,
                  torch::Tensor const& inv_mu_ratio_m1,
                  torch::Tensor const& cv_ratio_m1, torch::Tensor const& u0) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_roe_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;
    auto inv_mu = ideal_moist ? inv_mu_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto cv = ideal_moist ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto energy0 = ideal_moist ? u0.data_ptr<scalar_t>() : nullptr;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t i = 0; i < n; ++i) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto face_pressure =
                reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wl = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wr = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto elr = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto glr = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            auto clr = reinterpret_cast<scalar_t*>(data[6] + i * strides[6]);
            roe_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride),
                     *clr, *(clr + stride), dim, ny, ideal_moist, nvapor,
                     scalar_t(gammad), inv_mu, cv, energy0, stride, stride,
                     face_pressure);
          }
        },
        grain_size);
  });
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_roe);
REGISTER_ALL_CPU_DISPATCH(call_roe, &snap::call_roe_cpu);
REGISTER_MPS_DISPATCH(call_roe, &snap::OnHost<&snap::call_roe_cpu>::run);

}  // namespace at::native
