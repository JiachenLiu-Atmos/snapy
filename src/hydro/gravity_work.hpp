#pragma once

#include <torch/torch.h>

namespace snap {

//! Work from area-integrated face mass fluxes (or mass increments). The result
//! has the interior cell shape [is, ie); callers multiply by dt only for
//! fluxes. Difference geometry BEFORE casting to the working dtype, so a large
//! origin cannot erase a short cell width when FP64 master coordinates are
//! supplied.
inline torch::Tensor gravity_work_from_face_mass(torch::Tensor const& face_mass,
                                                 torch::Tensor const& x1f,
                                                 torch::Tensor const& x1v,
                                                 torch::Tensor const& volume,
                                                 int is, int ie, double grav1) {
  auto centres = x1v.slice(0, is, ie);
  auto hi = (grav1 * (x1f.slice(0, is + 1, ie + 1) - centres))
                .to(face_mass.options());
  auto lo = (grav1 * (x1f.slice(0, is, ie) - centres)).to(face_mass.options());
  return (hi * face_mass.slice(-1, is + 1, ie + 1) -
          lo * face_mass.slice(-1, is, ie)) /
         volume.slice(-1, is, ie).to(face_mass.options());
}

}  // namespace snap
