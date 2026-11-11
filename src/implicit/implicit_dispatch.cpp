// eigen
#include <Eigen/Dense>

// C/C++
#include <vector>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "forward_sweep_impl.h"
#include "implicit_dispatch.hpp"
#include "vic_assemble_full_impl.h"
#include "vic_assemble_partial_impl.h"
#include "vic_redistribute_impl.h"

namespace snap {

void vic_assemble_partial_cpu(at::TensorIterator& iter, double dt, double grav,
                              int dir) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_assemble_partial_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;

    using Matrix = Eigen::Matrix<scalar_t, 3, 3>;
    using Vector = Eigen::Matrix<scalar_t, 3, 1>;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t col = 0; col < n; ++col) {
            auto du = reinterpret_cast<scalar_t*>(data[0] + col * strides[0]);
            auto delta = reinterpret_cast<Vector*>(data[9] + col * strides[9]);
            auto w = reinterpret_cast<scalar_t*>(data[2] + col * strides[2]);
            auto gamma =
                reinterpret_cast<scalar_t*>(data[3] + col * strides[3]);
            auto area = reinterpret_cast<scalar_t*>(data[4] + col * strides[4]);
            auto vol = reinterpret_cast<scalar_t*>(data[5] + col * strides[5]);
            auto work_lo =
                reinterpret_cast<scalar_t*>(data[11] + col * strides[11]);
            auto work_hi =
                reinterpret_cast<scalar_t*>(data[12] + col * strides[12]);
            auto a = reinterpret_cast<Matrix*>(data[6] + col * strides[6]);
            auto b = reinterpret_cast<Matrix*>(data[7] + col * strides[7]);
            auto c = reinterpret_cast<Matrix*>(data[8] + col * strides[8]);
            auto solid =
                reinterpret_cast<scalar_t*>(data[10] + col * strides[10]);

            for (int i = 0; i < nlayer; ++i) {
              if (solid[i * stride2] != 0) {
                a[i].setIdentity();
                a[i] /= dt;
                b[i].setZero();
                c[i].setZero();
                delta[i].setZero();  // no m_t right-hand-side correction
                continue;
              }
              bool lower = i > 0 && solid[(i - 1) * stride2] != 0;
              bool upper = i + 1 < nlayer && solid[(i + 1) * stride2] != 0;
              vic_assemble_partial_impl(a, b, c, w, gamma, area, vol, work_lo,
                                        work_hi, i, 0, nlayer - 1, dt, grav,
                                        dir, ny, stride1, stride2, first_block,
                                        last_block, lower, upper, du, delta);
            }
          }
        },
        grain_size);
  });
}

void vic_assemble_full_cpu(at::TensorIterator& iter, double dt, double grav,
                           int dir) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_assemble_full_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;
    bool periodic = false;

    using Matrix = Eigen::Matrix<scalar_t, 5, 5>;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t col = 0; col < n; ++col) {
            auto w = reinterpret_cast<scalar_t*>(data[2] + col * strides[2]);
            auto gamma =
                reinterpret_cast<scalar_t*>(data[3] + col * strides[3]);
            auto area = reinterpret_cast<scalar_t*>(data[4] + col * strides[4]);
            auto vol = reinterpret_cast<scalar_t*>(data[5] + col * strides[5]);
            auto work_lo =
                reinterpret_cast<scalar_t*>(data[11] + col * strides[11]);
            auto work_hi =
                reinterpret_cast<scalar_t*>(data[12] + col * strides[12]);
            auto a = reinterpret_cast<Matrix*>(data[6] + col * strides[6]);
            auto b = reinterpret_cast<Matrix*>(data[7] + col * strides[7]);
            auto c = reinterpret_cast<Matrix*>(data[8] + col * strides[8]);
            auto solid =
                reinterpret_cast<scalar_t*>(data[10] + col * strides[10]);

            for (int i = 0; i < nlayer; ++i) {
              if (solid[i * stride2] != 0) {
                a[i].setIdentity();
                a[i] /= dt;
                b[i].setZero();
                c[i].setZero();
                continue;
              }
              bool lower = i > 0 && solid[(i - 1) * stride2] != 0;
              bool upper = i + 1 < nlayer && solid[(i + 1) * stride2] != 0;
              vic_assemble_full_impl(a, b, c, w, gamma, area, vol, work_lo,
                                     work_hi, i, 0, nlayer - 1, dt, grav, dir,
                                     ny, stride1, stride2, first_block,
                                     last_block, periodic, lower, upper);
            }
          }
        },
        grain_size);
  });
}

// K = 2 (Woodbury, operands 13 and 14): the same factorisation also solves the
// unit E rows at the column's two wall cells
template <int N, int K>
void vic_solve_cpu(at::TensorIterator& iter, double dt, double grav, int dir) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;
    bool first_block = true;
    bool last_block = true;

    using Matrix = Eigen::Matrix<scalar_t, N, N>;
    using Vector = Eigen::Matrix<scalar_t, N, 1>;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t col = 0; col < n; ++col) {
            auto du = reinterpret_cast<scalar_t*>(data[0] + col * strides[0]);
            auto a = reinterpret_cast<Matrix*>(data[6] + col * strides[6]);
            auto b = reinterpret_cast<Matrix*>(data[7] + col * strides[7]);
            auto c = reinterpret_cast<Matrix*>(data[8] + col * strides[8]);
            auto delta = reinterpret_cast<Vector*>(data[9] + col * strides[9]);
            Vector *z0 = nullptr, *z1 = nullptr;
            if constexpr (K == 2) {
              z0 = reinterpret_cast<Vector*>(data[13] + col * strides[13]);
              z1 = reinterpret_cast<Vector*>(data[14] + col * strides[14]);
            }

            // partial: delta holds the assembly's m_t right-hand-side term
            if (!ForwardSweep<K>(a, b, c, delta, du, dt, 0, nlayer - 1, dir, ny,
                                 stride1, stride2, first_block, last_block,
                                 N == 3, z0, z1, 0, nlayer - 1))
              continue;
            vic_backward_substitute(a, delta, 0, nlayer - 1);
            if constexpr (K == 2) {
              vic_backward_substitute(a, z0, 0, nlayer - 1);
              vic_backward_substitute(a, z1, 0, nlayer - 1);
            }
          }
        },
        grain_size);
  });
}

template void vic_solve_cpu<3, 0>(at::TensorIterator&, double, double, int);
template void vic_solve_cpu<5, 0>(at::TensorIterator&, double, double, int);
template void vic_solve_cpu<3, 2>(at::TensorIterator&, double, double, int);
template void vic_solve_cpu<5, 2>(at::TensorIterator&, double, double, int);

template <int N>
void vic_redistribute_cpu(at::TensorIterator& iter, double /*dt*/,
                          double /*grav*/, int dir) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_redistribute_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    auto stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);

    int ny = nhydro - ICY;

    using Vector = Eigen::Matrix<scalar_t, N, 1>;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t col = 0; col < n; ++col) {
            auto du = reinterpret_cast<scalar_t*>(data[0] + col * strides[0]);
            auto mass_fix =
                reinterpret_cast<scalar_t*>(data[1] + col * strides[1]);
            auto w = reinterpret_cast<scalar_t*>(data[2] + col * strides[2]);
            auto vol = reinterpret_cast<scalar_t*>(data[5] + col * strides[5]);
            auto delta = reinterpret_cast<Vector*>(data[9] + col * strides[9]);

            auto solid =
                reinterpret_cast<scalar_t*>(data[10] + col * strides[10]);
            for (int lo = 0; lo < nlayer;) {
              if (solid[lo * stride2] != 0) {
                ++lo;
                continue;
              }
              int hi = lo + 1;
              while (hi < nlayer && solid[hi * stride2] == 0) ++hi;
              vic_constituent_column<scalar_t, N>(
                  du + lo * stride2, w + lo * stride2, mass_fix + lo * stride2,
                  delta + lo, vol + lo * stride2, hi - lo, dir, ny, stride1,
                  stride2);
              if constexpr (N == 3)
                vic_tangential_momentum_column(
                    du + lo * stride2, w + lo * stride2,
                    mass_fix + lo * stride2, vol + lo * stride2, hi - lo, dir,
                    stride1, stride2);
              for (int i = lo; i < hi; ++i)
                vic_redistribute_cell(du, mass_fix, delta, i, dir, ny, stride1,
                                      stride2);
              lo = hi;
            }
          }
        },
        grain_size);
  });
}

template void vic_redistribute_cpu<3>(at::TensorIterator&, double, double, int);
template void vic_redistribute_cpu<5>(at::TensorIterator&, double, double, int);

// the donor sweep of MeshBlock's implicit tracer transfer, skipped when no
// donor can overdraw
void donor_cap_cpu(at::Tensor& P, at::Tensor const& held, int is, int ie) {
  int nf = ie - is;
  if (nf <= 0) return;
  auto qf = P.narrow(-1, is + 1, nf);
  auto cap = held.narrow(-1, is, nf).clone();
  cap.narrow(-1, 1, nf - 1) += P.narrow(-1, is + 1, nf - 1);
  auto binds = (qf > cap) | (qf < -held.narrow(-1, is + 1, nf));
  if (binds.any().item<bool>()) {
    auto avail = held.select(-1, is);
    for (int i = is + 1; i <= ie; ++i) {  // q: the face between i-1 and i
      auto q = P.select(-1, i);
      auto avail_up = held.select(-1, i);
      q.copy_(torch::maximum(torch::minimum(q, avail), -avail_up));
      avail = avail_up + q;
    }
  }
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(vic_assemble_partial);
DEFINE_DISPATCH(vic_assemble_full);
DEFINE_DISPATCH(vic_solve_partial);
DEFINE_DISPATCH(vic_solve_full);
DEFINE_DISPATCH(vic_solve_woodbury_partial);
DEFINE_DISPATCH(vic_solve_woodbury_full);
DEFINE_DISPATCH(vic_redistribute_partial);
DEFINE_DISPATCH(vic_redistribute_full);
DEFINE_DISPATCH(donor_cap);

REGISTER_ALL_CPU_DISPATCH(vic_assemble_partial,
                          &snap::vic_assemble_partial_cpu);
REGISTER_ALL_CPU_DISPATCH(vic_assemble_full, &snap::vic_assemble_full_cpu);
REGISTER_ALL_CPU_DISPATCH(vic_solve_partial, (&snap::vic_solve_cpu<3, 0>));
REGISTER_ALL_CPU_DISPATCH(vic_solve_full, (&snap::vic_solve_cpu<5, 0>));
REGISTER_ALL_CPU_DISPATCH(vic_solve_woodbury_partial,
                          (&snap::vic_solve_cpu<3, 2>));
REGISTER_ALL_CPU_DISPATCH(vic_solve_woodbury_full,
                          (&snap::vic_solve_cpu<5, 2>));
REGISTER_ALL_CPU_DISPATCH(vic_redistribute_partial,
                          &snap::vic_redistribute_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(vic_redistribute_full,
                          &snap::vic_redistribute_cpu<5>);
REGISTER_ALL_CPU_DISPATCH(donor_cap, &snap::donor_cap_cpu);

REGISTER_MPS_DISPATCH(vic_assemble_partial,
                      &snap::OnHost<&snap::vic_assemble_partial_cpu>::run);
REGISTER_MPS_DISPATCH(vic_assemble_full,
                      &snap::OnHost<&snap::vic_assemble_full_cpu>::run);
REGISTER_MPS_DISPATCH(vic_solve_partial,
                      (&snap::OnHost<&snap::vic_solve_cpu<3, 0>>::run));
REGISTER_MPS_DISPATCH(vic_solve_full,
                      (&snap::OnHost<&snap::vic_solve_cpu<5, 0>>::run));
REGISTER_MPS_DISPATCH(vic_solve_woodbury_partial,
                      (&snap::OnHost<&snap::vic_solve_cpu<3, 2>>::run));
REGISTER_MPS_DISPATCH(vic_solve_woodbury_full,
                      (&snap::OnHost<&snap::vic_solve_cpu<5, 2>>::run));
REGISTER_MPS_DISPATCH(vic_redistribute_partial,
                      &snap::OnHost<&snap::vic_redistribute_cpu<3>>::run);
REGISTER_MPS_DISPATCH(vic_redistribute_full,
                      &snap::OnHost<&snap::vic_redistribute_cpu<5>>::run);
REGISTER_MPS_DISPATCH(donor_cap, &snap::donor_cap_cpu);  // plain torch ops

}  // namespace at::native
