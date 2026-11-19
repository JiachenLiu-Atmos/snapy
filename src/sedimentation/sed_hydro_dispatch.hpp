#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! The scalars of the one sedimentation kernel (SedKernelArgs, EF.rst D8)
struct SedParams {
  int il = -1, iu = 0;  //!< faces at or below il and above iu are sealed
  double grav = 0.;     //!< the signed x1 acceleration
  double gas_diameter = 0., gas_epsilon_lj = 0., gas_mass = 0.;
  double upper_limit = 0.;
};

//! The settling flux of particles hydro_ids added onto the x1 faces of flux,
//! and their face velocities into vsed (nparticle, nc3, nc2, nc1). temp and
//! energy (rows ICY..: rho y e) come from the EOS, "W->T" and "W->E".
void sedimentation_flux(torch::Tensor w, torch::Tensor temp,
                        torch::Tensor energy, torch::Tensor flux,
                        torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                        torch::Tensor radius, torch::Tensor density,
                        torch::Tensor const_vsed, torch::Tensor hydro_ids,
                        SedParams p);

}  // namespace snap

namespace at::native {

using sedimentation_flux_fn = void (*)(torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor,
                                       torch::Tensor, torch::Tensor,
                                       snap::SedParams);

DECLARE_DISPATCH(sedimentation_flux_fn, call_sedimentation_flux);

}  // namespace at::native
