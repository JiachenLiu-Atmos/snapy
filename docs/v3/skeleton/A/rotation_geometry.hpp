// V3 lane A skeleton: lane A's read view of S's AxialRotationData and
// RotationAxis (S coord_rotation.hpp:15-18, 48-55 at d03e4888; A.rst 3.8).
#pragma once

#include <torch/torch.h>

namespace snap {

//! Rotation about the declared axis, built once per block by coord (lane S);
//! from Coordinate::rotation_geometry(), null without an axis (Cartesian).
//! Replaces the four AAM hooks on Coordinate (coordinate.hpp:370-392).
class RotationGeometry {
 public:
  virtual ~RotationGeometry() = default;
  //! Cartesian unit vector (RotationAxis::unit); only z in V3 (FOR CHENG)
  virtual torch::Tensor axis_unit() const = 0;
  //! k = z x n at centres (contra / covariant), k . k = cos^2(lat)
  virtual torch::Tensor k_center() const = 0;
  virtual torch::Tensor k_center_lower() const = 0;
  virtual torch::Tensor kk_center() const = 0;
  //! exact face average of k, in coord's flux2global<d>_ frame (S-2), and
  //! cos^2(lat) at x2/x3 faces
  virtual torch::Tensor k_face(int dir) const = 0;
  virtual torch::Tensor c2_face(int dir) const = 0;
  //! on-axis mask, set at reset (pending S-4: rate_to_rows moves to A, H4)
  virtual torch::Tensor on_axis_mask() const = 0;
  virtual bool has_on_axis_cell() const = 0;  //!< at reset; no host read
};

}  // namespace snap
