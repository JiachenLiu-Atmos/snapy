#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

enum X1RefGrid : int {
  kX1RefNonUniform = 0,
  kX1RefUniform = 1,
  kX1RefStretched = 2
};

//! The CPU kernel, which MPS runs through OnHost; also callable from tests.
void hydro_pressure_scan_x1_cpu(torch::Tensor const& rho,
                                torch::Tensor const& dx1f,
                                torch::Tensor const& anchor,
                                torch::Tensor const& lo,
                                torch::Tensor const& hi, int iu, double grav);

}  // namespace snap

namespace at::native {

using hydro_ref_x1_fn =
    void (*)(torch::Tensor const& w, torch::Tensor const& dx1f,
             torch::Tensor const& anchor, torch::Tensor const& psf_lo,
             torch::Tensor const& psf_hi, torch::Tensor const& pref,
             torch::Tensor const& dsf, torch::Tensor const& dref, int iu,
             double grav, int x1_grid, bool phys_in, bool phys_out,
             bool wall_clamp, torch::Tensor const& rows);

DECLARE_DISPATCH(hydro_ref_x1_fn, call_hydro_ref_x1);

using hydro_pressure_scan_x1_fn = void (*)(torch::Tensor const& rho,
                                           torch::Tensor const& dx1f,
                                           torch::Tensor const& anchor,
                                           torch::Tensor const& lo,
                                           torch::Tensor const& hi, int iu,
                                           double grav);
DECLARE_DISPATCH(hydro_pressure_scan_x1_fn, call_hydro_pressure_scan_x1);

}  // namespace at::native
