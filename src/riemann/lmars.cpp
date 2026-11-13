// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void LmarsSolverImpl::reset() {
  TORCH_CHECK(phydro, "[LmarsSolver] Parent Hydro is null");
  auto pcoord = phydro->pmb->pcoord;

  // register buffers
  auto nc1 = pcoord->options->nc1();
  auto nc2 = pcoord->options->nc2();
  auto nc3 = pcoord->options->nc3();

  elr =
      register_buffer("elr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
  clr =
      register_buffer("clr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
  glr =
      register_buffer("glr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor flx,
                                       torch::Tensor face_pressure) {
  auto pcoord = phydro->pmb->pcoord;
  auto peos = phydro->peos;

  elr[ILT] = peos->compute("W->I", {wl}) / wl[IDN];

  if (peos->options->type() == "aneos") {
    clr[ILT] = peos->compute("W->L", {wl});
    glr[ILT] = peos->compute("WL->A", {wl, clr[ILT]});
  } else {
    glr[ILT] = peos->compute("W->A", {wl});
  }

  elr[IRT] = peos->compute("W->I", {wr}) / wr[IDN];

  if (peos->options->type() == "aneos") {
    clr[IRT] = peos->compute("W->L", {wr});
    glr[IRT] = peos->compute("WL->A", {wr, clr[IRT]});
  } else {
    glr[IRT] = peos->compute("W->A", {wr});
  }

  to_face_frame_(*pcoord, dim, wl, wr);

  auto face_pressure_out =
      face_pressure.defined() ? face_pressure : torch::empty_like(wl[IDN]);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
                  .add_output(flx)
                  .add_owned_output(face_pressure_out.unsqueeze(0))
                  .add_input(wl)
                  .add_input(wr)
                  .add_input(elr)
                  .add_input(glr)
                  .build();

  at::native::call_lmars(flx.device().type(), iter, dim);

  to_global_frame_(*pcoord, dim, flx);

  return flx;
}

}  // namespace snap
