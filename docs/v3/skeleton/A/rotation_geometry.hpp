// V3 lane A skeleton: the rotation data lane A reads from coord. Owned by
// lane S (S-1 accepted; A.rst 3.8). Declarations only.
#pragma once

#include <torch/torch.h>

namespace snap {

//! Rotation about the declared axis, built once per block by coord (lane S);
//! from Coordinate::rotation_geometry(), null without an axis (Cartesian).
//! Replaces the four AAM hooks on Coordinate (coordinate.hpp:370-392).
class RotationGeometry {
 public:
  virtual ~RotationGeometry() = default;
  //! the axis; only z is supported in V3, others refused (FOR CHENG)
  virtual int axis() const = 0;
  //! k = z x n at centres (contra / covariant), k . k = cos^2(lat)
  virtual torch::Tensor k_center() const = 0;
  virtual torch::Tensor k_center_lower() const = 0;
  virtual torch::Tensor kk_center() const = 0;
  //! exact face average of k, in each face's panel frame (S-2), and
  //! cos^2(lat) at x2/x3 faces
  virtual torch::Tensor k_face(int dir) const = 0;
  virtual torch::Tensor c2_face(int dir) const = 0;
  //! set the K component of two covariant horizontal rows to `target`
  //! (AAM rate per volume), orthogonal part unchanged; on-axis cell rule
  virtual void rate_to_rows(torch::Tensor row2, torch::Tensor row3,
                            torch::Tensor target) const = 0;
  virtual bool has_on_axis_cell() const = 0;  //!< at reset; no host read
};

}  // namespace snap
