// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Center3InterpImpl::reset() {
  cm = register_buffer(
      "cm", torch::tensor({1. / 3., 5. / 6., -1. / 6.}, torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));
}

void Center3InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly3(out.device().type(), iter, cm.to(w.options()), dim);
}

void Center3InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly3(out.device().type(), iter, cp.to(w.options()), dim);
}
}  // namespace snap
