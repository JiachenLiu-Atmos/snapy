// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "recon_dispatch.hpp"
#include "recon_point.h"
#include "x1_coeffs.hpp"

namespace snap {
namespace {

void check_chunk(char const* base, int64_t nrow, int64_t rowbytes, char** data,
                 const int64_t* strides, int64_t n, char const* what) {
  char const* lo = data[2];
  char const* hi = data[2] + (n - 1) * strides[2];
  if (hi < lo) std::swap(lo, hi);
  TORCH_CHECK((lo - base) % rowbytes == 0 && (hi - base) % rowbytes == 0, "[",
              what, "] coefficient row pointer is not row aligned");
  TORCH_CHECK(lo >= base && (hi - base) / rowbytes < nrow, "[", what,
              "] coefficient row pointer is outside the table");
}

// One CPU loop for every stencil scheme: each output is recon_point at its
// face. Uniform: one table for every face. Nu: the iterator's second input
// is the per-face row view, one row of ncoeff_nu per face.
template <Scheme S, bool Nu>
void recon_line_cpu(at::TensorIterator& iter, torch::Tensor const& rows,
                    int dim, bool scale, char const* what) {
  constexpr int NCoeff = SchemeTraits<S>::ncoeff_nu;
  if constexpr (Nu) {
    x1_coeffs_check_rows(iter, rows, dim, NCoeff, what);
    TORCH_CHECK(iter.ninputs() == 2, "[", what,
                "] the CPU iterator needs the coefficient row view; "
                "build it with snap::x1_nu_iterator");
  }
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "recon_line_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto c = rows.data_ptr<scalar_t>();
    auto const* base = reinterpret_cast<char const*>(c);
    int64_t nrow = rows.size(0);
    int64_t rowbytes = NCoeff * static_cast<int64_t>(sizeof(scalar_t));

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          if constexpr (Nu) {
            if (n <= 0) return;
            check_chunk(base, nrow, rowbytes, data, strides, n, what);
          }
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            scalar_t const* row = c;
            if constexpr (Nu) {
              row = reinterpret_cast<scalar_t const*>(data[2] + i * strides[2]);
            }
            for (int j = 0; j < nvar; ++j) {
              out[j * stride_out] = recon_point<S, scalar_t, Nu>(
                  w + j * stride2, stride1, row, scale);
            }
          }
        },
        grain_size);
  });
}

}  // namespace

void call_recon_cpu(at::TensorIterator& iter, torch::Tensor rows, int dim,
                    int scheme, bool scale, int64_t row_stride) {
  switch (static_cast<Scheme>(scheme)) {
    case Scheme::cp3:
      return recon_line_cpu<Scheme::cp3, false>(iter, rows, dim, scale, "cp3");
    case Scheme::weno3:
      return recon_line_cpu<Scheme::weno3, false>(iter, rows, dim, scale,
                                                  "weno3");
    case Scheme::cp5:
      if (row_stride) {
        return recon_line_cpu<Scheme::cp5, true>(iter, rows, dim, scale,
                                                 "poly5_nu");
      }
      return recon_line_cpu<Scheme::cp5, false>(iter, rows, dim, scale, "cp5");
    case Scheme::weno5:
      if (row_stride) {
        return recon_line_cpu<Scheme::weno5, true>(iter, rows, dim, scale,
                                                   "weno5_nu");
      }
      return recon_line_cpu<Scheme::weno5, false>(iter, rows, dim, scale,
                                                  "weno5");
    default:
      TORCH_CHECK(false, "[call_recon] scheme ", scheme,
                  " has no stencil kernel");
  }
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_recon);

REGISTER_ALL_CPU_DISPATCH(call_recon, &snap::call_recon_cpu);

REGISTER_MPS_DISPATCH(call_recon, &snap::OnHost<&snap::call_recon_cpu>::run);

}  // namespace at::native
