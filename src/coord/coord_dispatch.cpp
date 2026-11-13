// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "coord_dispatch.hpp"
#include "coord_utils_impl.h"
#include "cubed_sphere_utils_impl.h"

namespace snap {
void call_coord_vec_lower_cpu(at::TensorIterator& iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_coord_vec_lower_cpu", [&] {
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto v2 = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto v3 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto cth = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            coord_vec_lower_impl(v2, v3, *cth);
          }
        },
        grain_size);
  });
}

void call_coord_vec_lower_mps(at::TensorIterator& iter) {
  auto v2 = iter.output(0).clone();
  auto v3 = iter.output(1).clone();
  auto cth = iter.input(0);

  iter.output(0).copy_(v2 + v3 * cth);
  iter.output(1).copy_(v3 + v2 * cth);
}

void call_coord_vec_raise_cpu(at::TensorIterator& iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_coord_vec_raise_cpu", [&] {
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto v2 = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto v3 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto cth = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            coord_vec_raise_impl(v2, v3, *cth);
          }
        },
        grain_size);
  });
}

void call_coord_vec_raise_mps(at::TensorIterator& iter) {
  auto v2 = iter.output(0).clone();
  auto v3 = iter.output(1).clone();
  auto cth = iter.input(0);
  auto sth2 = 1. - cth * cth;

  iter.output(0).copy_(v2 / sth2 - v3 * cth / sth2);
  iter.output(1).copy_(-v2 * cth / sth2 + v3 / sth2);
}

void call_cs_matrix_vec_cpu(at::TensorIterator& iter) {
  int grain_size =
      std::max<int64_t>(1, iter.numel() / std::max(1, at::get_num_threads()));

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cs_matrix_vec_cpu", [&] {
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; ++i) {
            auto load = [&](int operand) {
              return *reinterpret_cast<scalar_t*>(data[operand] +
                                                  i * strides[operand]);
            };
            auto out0 = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto out1 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto out2 = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            scalar_t v0 = *out0;
            scalar_t v1 = *out1;
            scalar_t v2 = *out2;
            *out0 = load(3) * v0 + load(4) * v1 + load(5) * v2;
            *out1 = load(6) * v0 + load(7) * v1 + load(8) * v2;
            *out2 = load(9) * v0 + load(10) * v1 + load(11) * v2;
          }
        },
        grain_size);
  });
}

void call_cs_matrix_vec_mps(at::TensorIterator& iter) {
  auto v0 = iter.output(0).clone();
  auto v1 = iter.output(1).clone();
  auto v2 = iter.output(2).clone();
  iter.output(0).copy_(iter.input(0) * v0 + iter.input(1) * v1 +
                       iter.input(2) * v2);
  iter.output(1).copy_(iter.input(3) * v0 + iter.input(4) * v1 +
                       iter.input(5) * v2);
  iter.output(2).copy_(iter.input(6) * v0 + iter.input(7) * v1 +
                       iter.input(8) * v2);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_coord_vec_lower);
DEFINE_DISPATCH(call_coord_vec_raise);
DEFINE_DISPATCH(call_cs_matrix_vec);

REGISTER_ALL_CPU_DISPATCH(call_coord_vec_lower,
                          &snap::call_coord_vec_lower_cpu);
REGISTER_ALL_CPU_DISPATCH(call_coord_vec_raise,
                          &snap::call_coord_vec_raise_cpu);
REGISTER_ALL_CPU_DISPATCH(call_cs_matrix_vec, &snap::call_cs_matrix_vec_cpu);

REGISTER_MPS_DISPATCH(call_coord_vec_lower, &snap::call_coord_vec_lower_mps);
REGISTER_MPS_DISPATCH(call_coord_vec_raise, &snap::call_coord_vec_raise_mps);
REGISTER_MPS_DISPATCH(call_cs_matrix_vec, &snap::call_cs_matrix_vec_mps);

}  // namespace at::native
