// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "coord_dispatch.hpp"
#include "coord_utils.hpp"

namespace snap {

void coord_vec_lower_(torch::Tensor const& vel, torch::Tensor cth) {
  TORCH_CHECK(cth.dim() == vel[VEL2].dim(),
              "coord_vec_lower_::cth has incompatible dimension with vel",
              "Expected dim ", vel[VEL2].dim(), " but got ", cth.dim());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(vel[VEL2].sizes())
                  .add_owned_output(vel[VEL2])
                  .add_owned_output(vel[VEL3])
                  .add_owned_input(cth.to(vel.options()).expand_as(vel[VEL2]))
                  .build();

  at::native::call_coord_vec_lower(vel.device().type(), iter);
}

void coord_vec_raise_(torch::Tensor const& vel, torch::Tensor cth) {
  TORCH_CHECK(cth.dim() == vel[VEL2].dim(),
              "coord_vec_raise_::cth has incompatible dimension with vel",
              "Expected dim ", vel[VEL2].dim(), " but got ", cth.dim());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(vel[VEL2].sizes())
                  .add_owned_output(vel[VEL2])
                  .add_owned_output(vel[VEL3])
                  .add_owned_input(cth.to(vel.options()).expand_as(vel[VEL2]))
                  .build();

  at::native::call_coord_vec_raise(vel.device().type(), iter);
}

}  // namespace snap
