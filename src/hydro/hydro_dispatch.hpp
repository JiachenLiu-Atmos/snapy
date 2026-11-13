#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

class EquationOfStateImpl;

//! dynamics/conversion-order 4's per-cell fallback test (hydro_forward.cpp):
//! true where `a` is admissible -- finite, rho above the density floor and,
//! with `limiter` on, no species below zero (u - L u, prim false), and p
//! above the pressure floor too (its primitives, prim true). The floors are
//! zero with `limiter` off. balance_column passes the run's limiter flag.
torch::Tensor conv4_admissible(torch::Tensor const& a, bool prim,
                               EquationOfStateImpl* eos, bool limiter);

enum X1RefGrid : int {
  kX1RefNonUniform = 0,
  kX1RefUniform = 1,
  kX1RefStretched = 2
};

torch::Tensor hydro_ref_x1_stretched_rows(torch::Tensor const& dx1f, int iu,
                                          bool phys_in, bool phys_out,
                                          bool wall_clamp);

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

}  // namespace at::native
