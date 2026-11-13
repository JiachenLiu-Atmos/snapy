#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using coord_vec_fn = void (*)(at::TensorIterator &iter);
using cs_matrix_vec_fn = void (*)(at::TensorIterator &iter);

DECLARE_DISPATCH(coord_vec_fn, call_coord_vec_lower);
DECLARE_DISPATCH(coord_vec_fn, call_coord_vec_raise);
DECLARE_DISPATCH(cs_matrix_vec_fn, call_cs_matrix_vec);

}  // namespace at::native
