#pragma once

// torch
#include <torch/torch.h>

namespace snap {

class HydroImpl;

//! x1 mass-flux w (A.rst 3.4, H3): m1 = rho w from the stage's final x1 face
//! mass transfers Phi (explicit flux plus implicit transfer), split as
//!   m1 = [Phi_lo (r^2 - r_lo^2) + Phi_hi (r_hi^2 - r^2)] / (2 r V dt),
//! so a column's Coriolis torque equals the planetary angular momentum its x1
//! transfers move, and a wall face that moves no mass exerts none. m1 is not
//! private to Coriolis: coord's x1_mass_source_ reads it too, with or without
//! Coriolis. Lane A's class; Hydro holds it (R-6) and builds m1 once per stage
//! in post_implicit.
class X1Momentum {
 public:
  //! #355's gate, found at reset: not shallow water, nc1 > 1, and a
  //! gnomonic-equiangle grid or a spherical-polar one with Coriolis
  explicit X1Momentum(HydroImpl const* hydro);
  bool applies() const { return applies_; }

  //! pre_implicit: the Coriolis input wc1 = (rho, 0, v2, v3): Coriolis sees
  //! w = 0 before the solve. Rebuilds the grid factors and the stage buffers
  //! when the x1 faces, the dtype, the device or the block shape change
  torch::Tensor const& before_solve(torch::Tensor const& w);
  //! post_implicit: the face transfer rates phi = A_f (dry + ny species rows
  //! of flux1, the x1 total, which holds the solve's posts since I-4); then
  //! m1 into row IVX of wc2 = (1, m1, 0, 0); returns wc2
  torch::Tensor const& after_solve(torch::Tensor const& flux1, int ny);
  //! m1 of the last after_solve, x1 ghosts 0
  torch::Tensor m1() const;

 private:
  HydroImpl const* hydro_;
  bool applies_ = false;
  //! the grid-only factors (x1 face area and the split weights
  //! (r^2 - r_lo^2) / (2 r V), (r_hi^2 - r^2) / (2 r V) of the interior
  //! cells) and the stage buffers: phi and the Coriolis inputs wc1, wc2
  torch::Tensor x1f_, area_, lo_, hi_, phi_, wc1_, wc2_;
  int64_t ver_ = -1;
};

}  // namespace snap
