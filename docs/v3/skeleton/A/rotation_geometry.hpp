// V3 lane A skeleton: the AAM geometry, proposed for lane S ownership
// (A.rst 3.8; S-1). Declarations only.
#pragma once

#include <torch/torch.h>

namespace snap::v3 {

//! Geometry of rotation about the omega1 axis, built once per block; from
//! Coordinate::rotation_geometry(), null without an axis (Cartesian). Replaces
//! the four AAM hooks on Coordinate (coordinate.hpp:370-392 at 9e5de2d4).
class RotationGeometry {
 public:
  virtual ~RotationGeometry() = default;
  //! k = z x n at centres (contra / covariant), k . k = cos^2(lat)
  virtual torch::Tensor k_center() const = 0;
  virtual torch::Tensor k_center_lower() const = 0;
  virtual torch::Tensor kk_center() const = 0;
  //! exact face average of k (face flux basis), cos^2(lat) at x2/x3 faces
  virtual torch::Tensor k_face(int dir) const = 0;
  virtual torch::Tensor c2_face(int dir) const = 0;
  //! set the K component of two covariant horizontal rows to `target`
  //! (AAM rate per volume), orthogonal part unchanged; on-axis cell rule
  virtual void rate_to_rows(torch::Tensor row2, torch::Tensor row3,
                            torch::Tensor target) const = 0;
  //! Omega (about omega1, or the full vector) in the local orthonormal
  //! frame at cell centres [3, nk, nj, ni]; moved from coriolis.cpp:85-151
  virtual torch::Tensor omega_local(double o1, double o2, double o3) const = 0;
  //! #355: m1 = phi_lo * lo + phi_hi * hi; phi a mass rate; lo/hi hold 1/(2rV)
  virtual torch::Tensor x1_split_lo() const = 0;
  virtual torch::Tensor x1_split_hi() const = 0;
  virtual bool has_on_axis_cell() const = 0;  //!< at reset; no host read
};

}  // namespace snap::v3
