#pragma once

// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

// base
#include <configure.h>

// snap
#include "forward_sweep_impl.h"
#include "implicit_dispatch.hpp"
#include "vic_assemble_impl.h"
#include "vic_redistribute_impl.h"

namespace snap {

//! The operands of the VIC TensorIterator (ImplicitHydroImpl::vic_config);
//! the Woodbury solve adds z0 and z1.
enum VicOperand : int {
  kVicDu = 0,
  kVicMassFix,
  kVicW,
  kVicGamma,
  kVicArea,
  kVicVol,
  kVicA,
  kVicB,
  kVicC,
  kVicDelta,
  kVicSolid,
  kVicWorkLo,
  kVicWorkHi,
  kVicZ0,
  kVicZ1
};

//! a column's layout, the same for every column of the iterator
struct VicLayout {
  int nlayer, ny, stride1, stride2;
};

template <typename T>
DISPATCH_MACRO T* vic_op(char* const* p, int k) {
  return reinterpret_cast<T*>(p[k]);
}

//! assemble a[i], b[i], c[i] of cell i; a solid cell gets (1/dt) I
template <typename T, int N>
DISPATCH_MACRO void vic_assemble_cell(char* const* p, VicLayout const& L, int i,
                                      double dt, double grav, int dir) {
  using Matrix = Eigen::Matrix<T, N, N>;
  using Vector = Eigen::Matrix<T, N, 1>;
  auto du = vic_op<T>(p, kVicDu);
  auto delta = vic_op<Vector>(p, kVicDelta);
  auto w = vic_op<T>(p, kVicW);
  auto gamma = vic_op<T>(p, kVicGamma);
  auto area = vic_op<T>(p, kVicArea);
  auto vol = vic_op<T>(p, kVicVol);
  auto work_lo = vic_op<T>(p, kVicWorkLo);
  auto work_hi = vic_op<T>(p, kVicWorkHi);
  auto a = vic_op<Matrix>(p, kVicA);
  auto b = vic_op<Matrix>(p, kVicB);
  auto c = vic_op<Matrix>(p, kVicC);
  auto solid = vic_op<T>(p, kVicSolid);
  int nlayer = L.nlayer, stride2 = L.stride2;
  if (solid[i * stride2] != 0) {
    a[i].setIdentity();
    a[i] /= dt;
    b[i].setZero();
    c[i].setZero();
    if constexpr (N == 3) delta[i].setZero();  // no m_t right-hand side
    return;
  }
  bool lower = i > 0 && solid[(i - 1) * stride2] != 0;
  bool upper = i + 1 < nlayer && solid[(i + 1) * stride2] != 0;
  if constexpr (N == 3) {
    vic_assemble_partial_impl(a, b, c, w, gamma, area, vol, work_lo, work_hi, i,
                              0, nlayer - 1, dt, grav, dir, L.ny, L.stride1,
                              stride2, /*first_block=*/true,
                              /*last_block=*/true, lower, upper, du, delta);
  } else {
    vic_assemble_full_impl(a, b, c, w, gamma, area, vol, work_lo, work_hi, i, 0,
                           nlayer - 1, dt, grav, dir, L.ny, L.stride1, stride2,
                           /*first_block=*/true, /*last_block=*/true,
                           /*periodic=*/false, lower, upper);
  }
}

//! the column solve: forward sweep and back substitution. K = 2 (Woodbury):
//! the same factorisation also solves the unit E rows at the two wall cells
template <typename T, int N, int K>
DISPATCH_MACRO void vic_solve_column(char* const* p, VicLayout const& L,
                                     double dt, int dir) {
  using Matrix = Eigen::Matrix<T, N, N>;
  using Vector = Eigen::Matrix<T, N, 1>;
  auto du = vic_op<T>(p, kVicDu);
  auto a = vic_op<Matrix>(p, kVicA);
  auto b = vic_op<Matrix>(p, kVicB);
  auto c = vic_op<Matrix>(p, kVicC);
  auto delta = vic_op<Vector>(p, kVicDelta);
  Vector *z0 = nullptr, *z1 = nullptr;
  if constexpr (K == 2) {
    z0 = vic_op<Vector>(p, kVicZ0);
    z1 = vic_op<Vector>(p, kVicZ1);
  }
  int nlayer = L.nlayer;
  // partial: delta holds the assembly's m_t right-hand-side term
  if (!ForwardSweep<K>(a, b, c, delta, du, dt, 0, nlayer - 1, dir, L.ny,
                       L.stride1, L.stride2, /*first_block=*/true,
                       /*last_block=*/true, N == 3, z0, z1, 0, nlayer - 1))
    return;
  vic_backward_substitute(a, delta, 0, nlayer - 1, z0, z1);
}

//! the redistribution's serial column pass, per fluid segment: constituents
//! (and, for the partial matrix, the tangential momenta). It writes mass_fix
//! and the m_t rows of du, which vic_redistribute_one does not touch.
template <typename T, int N>
DISPATCH_MACRO void vic_redistribute_column(char* const* p, VicLayout const& L,
                                            int dir) {
  using Vector = Eigen::Matrix<T, N, 1>;
  auto du = vic_op<T>(p, kVicDu);
  auto mass_fix = vic_op<T>(p, kVicMassFix);
  auto w = vic_op<T>(p, kVicW);
  auto vol = vic_op<T>(p, kVicVol);
  auto delta = vic_op<Vector>(p, kVicDelta);
  auto solid = vic_op<T>(p, kVicSolid);
  int nlayer = L.nlayer, stride1 = L.stride1, stride2 = L.stride2;
  for (int lo = 0; lo < nlayer;) {
    if (solid[lo * stride2] != 0) {
      ++lo;
      continue;
    }
    int hi = lo + 1;
    while (hi < nlayer && solid[hi * stride2] == 0) ++hi;
    vic_constituent_column<T, N>(
        du + lo * stride2, w + lo * stride2, mass_fix + lo * stride2,
        delta + lo, vol + lo * stride2, hi - lo, dir, L.ny, stride1, stride2);
    if constexpr (N == 3)
      vic_tangential_momentum_column(
          du + lo * stride2, w + lo * stride2, mass_fix + lo * stride2,
          vol + lo * stride2, hi - lo, dir, stride1, stride2);
    lo = hi;
  }
}

//! the redistribution of fluid cell i
template <typename T, int N>
DISPATCH_MACRO void vic_redistribute_one(char* const* p, VicLayout const& L,
                                         int i, int dir) {
  using Vector = Eigen::Matrix<T, N, 1>;
  auto solid = vic_op<T>(p, kVicSolid);
  if (solid[i * L.stride2] != 0) return;
  vic_redistribute_cell(vic_op<T>(p, kVicDu), vic_op<T>(p, kVicMassFix),
                        vic_op<Vector>(p, kVicDelta), i, dir, L.ny, L.stride1,
                        L.stride2);
}

//! One VIC stage for N x N blocks, on the launcher's columns. Launch provides
//!   columns<NOPS, NT>(iter, f): f(p) once per column, p the column's operand
//!     pointers (NT: CUDA threads per block);
//!   cells<NOPS>(iter, nlayer, f): f(p, i) once per (column, layer).
//! On a CPU every column runs its cells in order, as one serial loop.
template <class Launch, typename T, int N>
void vic_stage_run(at::TensorIterator& iter, VicLayout L,
                   VicParams const& prm) {
  double dt = prm.dt, grav = prm.grav;
  int dir = prm.dir;
  switch (prm.stage) {
    case VicStage::assemble:
      Launch::template cells<13>(
          iter, L.nlayer, [=] DISPATCH_MACRO(char* const* p, int i) {
            vic_assemble_cell<T, N>(p, L, i, dt, grav, dir);
          });
      break;
    case VicStage::solve:
      if (prm.woodbury)  // 64 threads per block: more registers per thread
        Launch::template columns<15, 64>(
            iter, [=] DISPATCH_MACRO(char* const* p) {
              vic_solve_column<T, N, 2>(p, L, dt, dir);
            });
      else
        Launch::template columns<13, 128>(
            iter, [=] DISPATCH_MACRO(char* const* p) {
              vic_solve_column<T, N, 0>(p, L, dt, dir);
            });
      break;
    case VicStage::redistribute:
      Launch::template redistribute<13>(
          iter, L.nlayer,
          [=] DISPATCH_MACRO(char* const* p) {
            vic_redistribute_column<T, N>(p, L, dir);
          },
          [=] DISPATCH_MACRO(char* const* p, int i) {
            vic_redistribute_one<T, N>(p, L, i, dir);
          });
      break;
  }
}

//! the one VIC kernel definition; CpuColumns and CudaColumns instantiate it
template <class Launch>
void vic_stage(at::TensorIterator& iter, VicParams const& prm) {
  TORCH_CHECK(prm.n == 3 || prm.n == 5, "VIC blocks are 3x3 or 5x5");
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_stage", [&] {
    VicLayout L;
    L.nlayer = at::native::ensure_nonempty_size(iter.output(), 3);
    L.ny = at::native::ensure_nonempty_size(iter.output(), 0) - ICY;
    L.stride1 = at::native::ensure_nonempty_stride(iter.output(), 0);
    L.stride2 = at::native::ensure_nonempty_stride(iter.output(), 3);
    if (prm.n == 3)
      vic_stage_run<Launch, scalar_t, 3>(iter, L, prm);
    else
      vic_stage_run<Launch, scalar_t, 5>(iter, L, prm);
  });
}

}  // namespace snap
