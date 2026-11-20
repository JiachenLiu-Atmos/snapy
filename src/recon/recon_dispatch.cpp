// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "interp_impl.h"
#include "recon_dispatch.hpp"

namespace snap {

// One CPU loop for every variant: each output is the point function
// (interp_point.h) at its face, through the line functions of interp_impl.h.
template <typename Line>
void call_interp_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                     bool scale) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_interp_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto c = coeff.data_ptr<scalar_t>();

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            Line::apply(out, w, c, stride1, stride2, stride_out, nvar, scale);
          }
        },
        grain_size);
  });
}

template <int N>
struct PolyLine {
  template <typename T>
  static void apply(T* out, T* w, T* c, int s1, int s2, int so, int nvar,
                    bool) {
    interp_poly_impl<T, N>(out, w, c, s1, s2, so, nvar);
  }
};

struct Weno3Line {
  template <typename T>
  static void apply(T* out, T* w, T* c, int s1, int s2, int so, int nvar,
                    bool scale) {
    interp_weno3_impl(out, w, c, s1, s2, so, nvar, scale);
  }
};

struct Weno5Line {
  template <typename T>
  static void apply(T* out, T* w, T* c, int s1, int s2, int so, int nvar,
                    bool scale) {
    interp_weno5_impl(out, w, c, s1, s2, so, nvar, scale);
  }
};

template <int N>
void call_poly_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim) {
  call_interp_cpu<PolyLine<N>>(iter, coeff, dim, /*scale=*/false);
}

void call_weno3_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                    bool scale) {
  call_interp_cpu<Weno3Line>(iter, coeff, dim, scale);
}

void call_weno5_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                    bool scale) {
  call_interp_cpu<Weno5Line>(iter, coeff, dim, scale);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_poly3);
DEFINE_DISPATCH(call_poly5);
DEFINE_DISPATCH(call_weno3);
DEFINE_DISPATCH(call_weno5);

REGISTER_ALL_CPU_DISPATCH(call_poly3, &snap::call_poly_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(call_poly5, &snap::call_poly_cpu<5>);
REGISTER_ALL_CPU_DISPATCH(call_weno3, &snap::call_weno3_cpu);
REGISTER_ALL_CPU_DISPATCH(call_weno5, &snap::call_weno5_cpu);

REGISTER_MPS_DISPATCH(call_poly3, &snap::OnHost<&snap::call_poly_cpu<3>>::run);
REGISTER_MPS_DISPATCH(call_poly5, &snap::OnHost<&snap::call_poly_cpu<5>>::run);
REGISTER_MPS_DISPATCH(call_weno3, &snap::OnHost<&snap::call_weno3_cpu>::run);
REGISTER_MPS_DISPATCH(call_weno5, &snap::OnHost<&snap::call_weno5_cpu>::run);

}  // namespace at::native
