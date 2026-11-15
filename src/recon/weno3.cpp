// torch
#include <ATen/TensorIterator.h>

// snap
#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Weno3InterpImpl::reset() {
  cm = register_buffer("cm", torch::tensor({{1. / 2., 1. / 2., 0.},
                                            {0., 3. / 2., -1. / 2.},
                                            {1., -1., 0.},
                                            {0., 1., -1.}},
                                           torch::kFloat64));

  cp = register_buffer("cp", cm.flip({1}));
}

void Weno3InterpImpl::left(torch::Tensor w, int dim, torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  // add dim to squash dim if w in on cuda
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

  at::native::call_weno3(out.device().type(), iter, cm.to(w.options()), dim,
                         options->scale());
}

void Weno3InterpImpl::right(torch::Tensor w, int dim,
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

  at::native::call_weno3(out.device().type(), iter, cp.to(w.options()), dim,
                         options->scale());
}

}  // namespace snap
