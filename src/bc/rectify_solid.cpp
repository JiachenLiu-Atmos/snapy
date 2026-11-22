// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "bc.hpp"
#include "bc_dispatch.hpp"
#include "internal_boundary.hpp"

namespace snap {
enum { DIM1 = 2, DIM2 = 1, DIM3 = 0 };

// run dimension 1
int run_flip_dim1(torch::Tensor& solid, int dir) {
  constexpr int MAXRUN = InternalBoundaryOptionsImpl::MAXRUN;

  int nc3 = solid.size(0);
  int nc2 = solid.size(1);
  int nc1 = solid.size(2);
  if (nc1 == 1) return 0;

  auto dp = torch::empty({nc3, nc2, nc1 + 1, MAXRUN * 2}, solid.options());
  auto fromLen = torch::empty({nc3, nc2, nc1 + 1, MAXRUN * 2}, solid.options());
  auto fromBit = torch::empty({nc3, nc2, nc1 + 1, MAXRUN * 2}, solid.options());
  auto usedFlip =
      torch::empty({nc3, nc2, nc1 + 1, MAXRUN * 2}, solid.options());

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape({nc3, nc2, nc1, 1}, /*squash_dims=*/{2, 3})
          .add_owned_output(solid.unsqueeze(-1))
          .add_input(dp)
          .add_input(fromLen)
          .add_input(fromBit)
          .add_input(usedFlip)
          .build();

  int num_flips = at::native::flip_zero(solid.device().type(), iter, DIM1, dir);
  return num_flips;
}

// run dimension 2
int run_flip_dim2(torch::Tensor& solid, int dir) {
  constexpr int MAXRUN = InternalBoundaryOptionsImpl::MAXRUN;

  int nc3 = solid.size(0);
  int nc2 = solid.size(1);
  int nc1 = solid.size(2);
  if (nc2 == 1) return 0;

  auto dp = torch::empty({nc3, nc2 + 1, nc1, MAXRUN * 2}, solid.options());
  auto fromLen = torch::empty({nc3, nc2 + 1, nc1, MAXRUN * 2}, solid.options());
  auto fromBit = torch::empty({nc3, nc2 + 1, nc1, MAXRUN * 2}, solid.options());
  auto usedFlip =
      torch::empty({nc3, nc2 + 1, nc1, MAXRUN * 2}, solid.options());

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape({nc3, nc2, nc1, 1}, /*squash_dims=*/{1, 3})
          .add_owned_output(solid.unsqueeze(-1))
          .add_input(dp)
          .add_input(fromLen)
          .add_input(fromBit)
          .add_input(usedFlip)
          .build();

  int num_flips = at::native::flip_zero(solid.device().type(), iter, DIM2, dir);
  return num_flips;
}

// run dimension 3
int run_flip_dim3(torch::Tensor& solid, int dir) {
  constexpr int MAXRUN = InternalBoundaryOptionsImpl::MAXRUN;

  int nc3 = solid.size(0);
  int nc2 = solid.size(1);
  int nc1 = solid.size(2);
  if (nc3 == 1) return 0;

  auto dp = torch::empty({nc3 + 1, nc2, nc1, MAXRUN * 2}, solid.options());
  auto fromLen = torch::empty({nc3 + 1, nc2, nc1, MAXRUN * 2}, solid.options());
  auto fromBit = torch::empty({nc3 + 1, nc2, nc1, MAXRUN * 2}, solid.options());
  auto usedFlip =
      torch::empty({nc3 + 1, nc2, nc1, MAXRUN * 2}, solid.options());

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape({nc3, nc2, nc1, 1}, /*squash_dims=*/{0, 3})
          .add_owned_output(solid.unsqueeze(-1))
          .add_input(dp)
          .add_input(fromLen)
          .add_input(fromBit)
          .add_input(usedFlip)
          .build();

  int num_flips = at::native::flip_zero(solid.device().type(), iter, DIM3, dir);
  return num_flips;
}

torch::Tensor InternalBoundaryImpl::rectify_solid(
    torch::Tensor solid_in, int& total_num_flips,
    std::vector<bcfunc_t> const& bfuncs) const {
  int nc3 = solid_in.size(0);
  int nc2 = solid_in.size(1);

  auto solid = solid_in.contiguous();
  int nghost = pmb ? pmb->options->coord()->nghost() : 1;

  ///-----  set all ghost zones to 1  -----///
  BoundaryFuncOptions op;
  op.nghost(nghost);
  op.type(kScalar);

  auto solid_inner = get_bc_func()["solid_inner"];
  auto solid_outer = get_bc_func()["solid_outer"];

  solid_inner(solid, DIM1, op);
  solid_outer(solid, DIM1, op);

  if (nc2 > 1) {
    solid_inner(solid, DIM2, op);
    solid_outer(solid, DIM2, op);
  }

  if (nc3 > 1) {
    solid_inner(solid, DIM3, op);
    solid_outer(solid, DIM3, op);
  }

  total_num_flips = 0;
  int i = 0;
  int dir = 1;
  for (; i < options->max_iter(); ++i) {
    auto num_flips = run_flip_dim1(solid, dir);

    if (nc2 > 1) {
      num_flips += run_flip_dim2(solid, dir);
    }

    if (nc3 > 1) {
      num_flips += run_flip_dim3(solid, dir);
    }

    if (num_flips == 0) break;

    total_num_flips += num_flips;
    dir *= -1;
  }

  TORCH_CHECK(i < options->max_iter(), "rectify_solid did not converge after ",
              options->max_iter(), " iterations");

  ///-----  set proper boundary conditions  -----///
  for (int i = 0; i < bfuncs.size(); ++i) {
    bfuncs[i](solid, 2 - i / 2, op);
  }

  return solid;
}
}  // namespace snap
