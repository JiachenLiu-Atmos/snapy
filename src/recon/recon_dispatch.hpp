#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

// One stub for every stencil scheme (snap::Scheme cp3, weno3, cp5, weno5).
// rows is the scheme's uniform table (row_stride 0), or a per-face x1 table
// of row_stride coefficients per face (cp5, weno5); then the CPU iterator
// carries the row view as its second input (snap::x1_nu_iterator).
using recon_fn = void (*)(at::TensorIterator &iter, at::Tensor rows, int dim,
                          int scheme, bool scale, int64_t row_stride);

DECLARE_DISPATCH(recon_fn, call_recon);

}  // namespace at::native
