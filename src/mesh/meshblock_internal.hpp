#pragma once

// Helpers shared by the six parts of MeshBlock (src/mesh/meshblock*.cpp; V3
// X5). Internal to src/mesh.

// torch
#include <torch/torch.h>

// snap
#include <snap/eos/equation_of_state.hpp>

#include "meshblock.hpp"

namespace snap {

//! scalar_r from scalar_s and the total dry density, into vars["scalar_r"]
void set_scalar_primitive(Variables &vars, torch::Tensor const &scalar_s,
                          torch::Tensor const &hydro_u, HydroImpl const &hydro);

//! the CUDA vapor repair counts its failures on the device while in scope
struct DeferVaporCheck {
  EquationOfStateImpl *eos;
  explicit DeferVaporCheck(EquationOfStateImpl *e) : eos(e) {
    eos->defer_vapor_check(true);
  }
  ~DeferVaporCheck() { eos->defer_vapor_check(false); }
};

}  // namespace snap
