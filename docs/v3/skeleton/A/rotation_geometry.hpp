// V3 lane A skeleton: lane A reads lane S's AxialRotationData and defines no
// view of its own (A.rst 3.8, D24). S's fields (fallback cf5a22a9): axis,
// k_cell, k_face2, k_face3, cos2_lat, K_f2, K_f3, s2, on_axis. Declarations only.
#pragma once

namespace snap {

//! lane S's type (coord_rotation.hpp); reached through S's accessor
//! CoordinateImpl::rotation_geometry() const, null without an axis
struct AxialRotationData;

}  // namespace snap
