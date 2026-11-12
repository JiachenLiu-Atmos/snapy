// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void HLLCSolverImpl::reset() {
  TORCH_CHECK(phydro, "[HLLCSolver] Parent Hydro is null");
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

torch::Tensor HLLCSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                      int dim, torch::Tensor flx,
                                      torch::Tensor face_pressure) {
  return _forward(wl, wr, dim, flx, face_pressure, torch::Tensor());
}

torch::Tensor HLLCSolverImpl::forward_perturbations(
    torch::Tensor wl, torch::Tensor wr, int dim, torch::Tensor flx,
    torch::Tensor reference_pressure, torch::Tensor face_pressure_delta) {
  TORCH_CHECK(reference_pressure.defined(),
              "HLLC perturbations require a shared face pressure background");
  return _forward(wl, wr, dim, flx, face_pressure_delta, reference_pressure);
}

torch::Tensor HLLCSolverImpl::_forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor flx,
                                       torch::Tensor face_pressure,
                                       torch::Tensor reference_pressure) {
  auto pcoord = phydro->pmb->pcoord;
  auto peos = phydro->peos;
  auto wl_total = wl, wr_total = wr;
  if (reference_pressure.defined()) {
    reference_pressure = reference_pressure.to(wl.options()).expand_as(wl[IPR]);
    wl_total = wl.clone();
    wr_total = wr.clone();
    wl_total[IPR] += reference_pressure;
    wr_total[IPR] += reference_pressure;
  }

  elr[ILT] = peos->compute("W->I", {wl_total});

  if (peos->options->type() == "aneos") {
    clr[ILT] = peos->compute("W->L", {wl_total});
    glr[ILT] = peos->compute("WL->A", {wl_total, clr[ILT]});
  } else {
    glr[ILT] = peos->compute("W->A", {wl_total});
    clr[ILT] = peos->compute("WA->L", {wl_total, glr[ILT]});
  }

  elr[IRT] = peos->compute("W->I", {wr_total});

  if (peos->options->type() == "aneos") {
    clr[IRT] = peos->compute("W->L", {wr_total});
    glr[IRT] = peos->compute("WL->A", {wr_total, clr[IRT]});
  } else {
    glr[IRT] = peos->compute("W->A", {wr_total});
    clr[IRT] = peos->compute("WA->L", {wr_total, glr[IRT]});
  }

  switch (dim) {
    case 1:
      pcoord->prim2local3_(wl);
      pcoord->prim2local3_(wr);
      break;
    case 2:
      pcoord->prim2local2_(wl);
      pcoord->prim2local2_(wr);
      break;
    case 3:
      pcoord->prim2local1_(wl);
      pcoord->prim2local1_(wr);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }

  auto face_pressure_out =
      face_pressure.defined() ? face_pressure : torch::empty_like(wl[IDN]);

  at::TensorIteratorConfig config;
  config.resize_outputs(false)
      .check_all_same_dtype(false)
      .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
      .add_output(flx)
      .add_owned_output(face_pressure_out.unsqueeze(0))
      .add_input(wl)
      .add_input(wr)
      .add_input(elr)
      .add_input(glr)
      .add_input(clr);
  if (reference_pressure.defined())
    config.add_owned_input(reference_pressure.unsqueeze(0));
  auto iter = config.build();

  at::native::call_hllc(flx.device().type(), iter, dim);

  switch (dim) {
    case 1:
      pcoord->flux2global3_(flx);
      break;
    case 2:
      pcoord->flux2global2_(flx);
      break;
    case 3:
      pcoord->flux2global1_(flx);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }

  return flx;
}

}  // namespace snap
