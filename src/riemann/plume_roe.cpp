// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>

#include "riemann_solver.hpp"

namespace snap {

//! calculate flux for plume equations
/*!
 * w = [R, W, B, V]
 */
static void calculate_flux(torch::Tensor& flx, torch::Tensor const& prim) {
  auto R = prim[0];
  auto W = prim[1];
  auto B = prim[2];
  auto V = prim[3];

  flx[0] = (1.0 - 1.0 / std::exp(1.0)) * R * R * W;
  flx[1] = 0.5 * R * R * W * V - 0.5 * R * R * R * V - 0.125 * R * R * V * V;
  flx[2] = 0.5 * R * R * B * W;
  flx[3] = 0.25 * R * R * R * V * W + 0.5 * R * R * R * R * W;
}

static torch::Tensor lax_friedrichs_flux(torch::Tensor const& priml,
                                         torch::Tensor const& primr,
                                         torch::Tensor const& csl,
                                         torch::Tensor const& csr) {
  auto flxl = torch::zeros_like(priml);
  auto flxr = torch::zeros_like(primr);

  calculate_flux(flxl, priml);
  calculate_flux(flxr, primr);

  auto alpha = torch::max(csl, csr);

  return 0.5 * (flxl + flxr) - 0.5 * alpha * (primr - priml);
}

void PlumeRoeSolverImpl::reset() {
  TORCH_CHECK(phydro, "[PlumeRoeSolver] parent is nullptr");
}

torch::Tensor PlumeRoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                          int dim, torch::Tensor flx,
                                          torch::Tensor face_pressure) {
  TORCH_CHECK(!face_pressure.defined(),
              "Face-pressure output is not implemented for PlumeRoeSolver");
  auto peos = phydro->peos;

  if (dim != 1) {
    TORCH_CHECK(false,
                "PlumeRoeSolver only supports dim = 1, but got dim = ", dim);
  }

  // Approximate sound speeds
  auto csl = peos->compute("W->L", {wl});
  auto csr = peos->compute("W->L", {wr});

  // Lax-Friedrichs flux
  auto lf_flux = lax_friedrichs_flux(wl, wr, csl, csr);

  return lf_flux;
}

}  // namespace snap
