// eigen
#include <Eigen/Dense>

// C/C++
#include <array>

// torch
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/utils/loops.cuh>

#include "implicit_dispatch.hpp"
#include "vic_kernels.h"

namespace snap {

//! vic_stage's launcher on CUDA: one thread per column, or per (column,
//! layer) for the cell-parallel parts (the assembly and the redistribution)
struct CudaColumns {
  template <int NOPS, int NT = 128, typename F>
  static void columns(at::TensorIterator& iter, F const& f) {
    at::cuda::CUDAGuard device_guard(iter.device());
    std::array<char*, NOPS> data;
    for (int k = 0; k < NOPS; ++k) data[k] = (char*)iter.data_ptr(k);
    auto offset_calc = ::make_offset_calculator<NOPS>(iter);
    at::native::launch_legacy_kernel<NT, 1>(
        iter.numel(), [=] __device__(int col) {
          auto offsets = offset_calc.get(col);
          char* p[NOPS];
          for (int k = 0; k < NOPS; ++k) p[k] = data[k] + offsets[k];
          f(p);
        });
  }

  template <int NOPS, typename F>
  static void cells(at::TensorIterator& iter, int nlayer, F const& f) {
    at::cuda::CUDAGuard device_guard(iter.device());
    std::array<char*, NOPS> data;
    for (int k = 0; k < NOPS; ++k) data[k] = (char*)iter.data_ptr(k);
    auto offset_calc = ::make_offset_calculator<NOPS>(iter);
    int64_t total = iter.numel() * (int64_t)nlayer;
    at::native::launch_legacy_kernel<128, 1>(total, [=] __device__(int idx) {
      int col = idx / nlayer;
      int i = idx % nlayer;
      auto offsets = offset_calc.get(col);
      char* p[NOPS];
      for (int k = 0; k < NOPS; ++k) p[k] = data[k] + offsets[k];
      f(p, i);
    });
  }

  //! the serial column pass of every column, then every cell in parallel:
  //! the column pass writes only mass_fix and, for the partial matrix, the
  //! m_t rows of du, which the cell pass does not touch
  template <int NOPS, typename F, typename G>
  static void redistribute(at::TensorIterator& iter, int nlayer,
                           F const& column, G const& cell) {
    columns<NOPS>(iter, column);
    cells<NOPS>(iter, nlayer, cell);
  }
};

template void vic_stage<CudaColumns>(at::TensorIterator&, VicParams const&);

// the donor sweep of MeshBlock's implicit tracer transfer, one thread per
// column, so no host read decides whether it binds
void donor_cap_cuda(at::Tensor &P, at::Tensor const &held, int is, int ie) {
  at::cuda::CUDAGuard device_guard(P.device());
  TORCH_CHECK(P.is_contiguous() && P.sizes() == held.sizes());
  auto h = held.contiguous();
  int64_t nc1 = P.size(-1), ncol = P.numel() / nc1;

  AT_DISPATCH_FLOATING_TYPES(P.scalar_type(), "donor_cap_cuda", [&]() {
    auto q = P.data_ptr<scalar_t>();
    auto hd = h.data_ptr<scalar_t>();
    at::native::launch_legacy_kernel<128, 1>(ncol, [=] __device__(int col) {
      auto qc = q + col * nc1, hc = hd + col * nc1;
      scalar_t avail = hc[is];
      for (int i = is + 1; i <= ie; ++i) {
        qc[i] = ::max(::min(qc[i], avail), -hc[i]);
        avail = hc[i] + qc[i];
      }
    });
  });
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(vic_stage, &snap::vic_stage<snap::CudaColumns>);
REGISTER_CUDA_DISPATCH(donor_cap, &snap::donor_cap_cuda);

}  // namespace at::native
