// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
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

template <int N>
void call_poly_mps(at::TensorIterator& iter, torch::Tensor coeff, int dim) {
  auto out = iter.output();
  auto w = iter.input();
  torch::matmul_out(out, w.unfold(dim, N, 1), coeff);
}

void call_weno3_mps(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                    bool scale) {
  auto result = iter.output();
  auto w = iter.input();

  auto c1 = coeff[0];
  auto c2 = coeff[1];
  auto c3 = coeff[2];
  auto c4 = coeff[3];

  auto wu = w.unfold(dim, 3, 1);
  torch::Tensor wscale;
  if (scale) {
    wscale = wu.abs().mean(-1) + 1.e-10;
    wu /= wscale.unsqueeze(-1);
  }

  auto alpha1 = 2. / 3. / (wu.matmul(c3).square() + 1e-6).square();
  auto alpha2 = 1. / 3. / (wu.matmul(c4).square() + 1e-6).square();

  torch::add_out(result, alpha1 * wu.matmul(c1), alpha2 * wu.matmul(c2));
  result /= alpha1 + alpha2;

  if (scale) {
    result.mul_(wscale);
  }
}

void call_weno5_mps(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                    bool scale) {
  auto result = iter.output();
  auto w = iter.input();

  auto c1 = coeff[0];
  auto c2 = coeff[1];
  auto c3 = coeff[2];
  auto c4 = coeff[3];
  auto c5 = coeff[4];
  auto c6 = coeff[5];
  auto c7 = coeff[6];
  auto c8 = coeff[7];
  auto c9 = coeff[8];

  auto wu = w.unfold(dim, 5, 1);
  torch::Tensor wscale;
  if (scale) {
    wscale = wu.abs().mean(-1) + 1.e-10;
    wu /= wscale.unsqueeze(-1);
  }

  auto beta1 =
      13. / 12. * wu.matmul(c4).square() + 1. / 4. * wu.matmul(c5).square();
  auto beta2 =
      13. / 12. * wu.matmul(c6).square() + 1. / 4. * wu.matmul(c7).square();
  auto beta3 =
      13. / 12. * wu.matmul(c8).square() + 1. / 4. * wu.matmul(c9).square();

  auto alpha1 = 0.3 / (beta1 + 1e-6).square();
  auto alpha2 = 0.6 / (beta2 + 1e-6).square();
  auto alpha3 = 0.1 / (beta3 + 1e-6).square();

  torch::div_out(
      result,
      alpha1 * wu.matmul(c1) + alpha2 * wu.matmul(c2) + alpha3 * wu.matmul(c3),
      alpha1 + alpha2 + alpha3);

  if (scale) {
    result.mul_(wscale);
  }
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

REGISTER_MPS_DISPATCH(call_poly3, &snap::call_poly_mps<3>);
REGISTER_MPS_DISPATCH(call_poly5, &snap::call_poly_mps<5>);
REGISTER_MPS_DISPATCH(call_weno3, &snap::call_weno3_mps);
REGISTER_MPS_DISPATCH(call_weno5, &snap::call_weno5_mps);

}  // namespace at::native
