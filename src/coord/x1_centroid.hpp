#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! SNAP_X1_CENTROID_EXACT: on spherical-polar grids a cell value is the r^2 dr
//! average, located at x1v = rbar + O(dx1^2 / r), while every x1 cell-to-face
//! map (the reconstruction, the hydrostatic scan, the reference) is a
//! uniform-measure formula. With the switch:
//! (i)+(ii) every x1 reconstruction input is first converted from r^2 means to
//!   plain means (x1_plain_means), so the reconstruction, the scan step
//!   g dx1f <rho> and the well-balanced reference see the values their
//!   formulas assume;
//! (iii) the x1 pressure source is (2/V) int r p~ dr, p~ the quintic
//!   through the six nearest face pressures, so the pressure force is the r^2
//!   average of -d_r p~ and a hydrostatic column of r^2 means is at rest.
//! See docs/derivations/x1-centroid-spherical.md.
//! Read once per process; off unless it is set.
bool x1_centroid_exact_enabled();

//! plain mean of cell i from the r^2 means of five cells, exact for
//! polynomials of degree <= 4. Window centred, kept inside the owned cells at a
//! physical x1 wall (clamp_in/out), inside the array elsewhere.
struct X1PlainMeanStencils {
  int is = 0, iu = -1, nc1 = 0;
  bool clamp_in = false, clamp_out = false;
  //! false: fewer than five cells to fit; nothing changes
  bool usable = false;
  //! (nc1, 5) cell indices and weights
  torch::Tensor idx, wt;
  //! rebuild for a state of another device or dtype; an unusable cache has
  //! no weights to move and is never stale (#298)
  bool stale(torch::Tensor const& ref) const {
    return usable && (wt.device() != ref.device() ||
                      wt.scalar_type() != ref.scalar_type());
  }
};

X1PlainMeanStencils x1_plain_mean_stencils(torch::Tensor const& x1f, int is,
                                           int iu, bool clamp_in,
                                           bool clamp_out,
                                           torch::TensorOptions const& options);

//! w (nvar, nc3, nc2, nc1) as plain means along x1, a new tensor. Ghost cells
//! past a clamped wall get the mirrored correction of the owned cells, odd in
//! the row ivx when odd_ivx (a reflecting wall), so a mirrored state stays
//! mirrored.
torch::Tensor x1_plain_means(X1PlainMeanStencils const& st,
                             torch::Tensor const& w, int ivx, bool odd_in,
                             bool odd_out);

//! (2/V) int r p~ dr on the owned cells, p~ the quintic through the six
//! nearest faces: inside is..iu+1 at a clamped end (a physical wall), two
//! ghost faces past it elsewhere (filled by the x1 neighbour)
struct X1PressureSourceStencils {
  int is = 0, iu = -1;
  bool usable = false;
  //! (nown, 6) face indices and weights
  torch::Tensor idx, wt;
  //! as X1PlainMeanStencils::stale (#298)
  bool stale(torch::Tensor const& ref) const {
    return usable && (wt.device() != ref.device() ||
                      wt.scalar_type() != ref.scalar_type());
  }
};

X1PressureSourceStencils x1_pressure_source_stencils(
    torch::Tensor const& x1f, int is, int iu, bool clamp_in, bool clamp_out,
    torch::TensorOptions const& options);

//! face_pressure1 (..., nc1 + 1) -> the source on the owned cells (..., nown)
torch::Tensor x1_pressure_source(X1PressureSourceStencils const& st,
                                 torch::Tensor const& face_pressure1);

}  // namespace snap
