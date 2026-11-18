// V3 lane A skeleton: lane A reads lane S's AxialRotationData and defines no
// view of its own (A.rst 3.8, D24). S rev 2.1 fields (4b96f4ea): axis, k_cell,
// k_cell_lower, cos2_lat, cos2_face2/3, K_f2/3, s2, on_axis, has_on_axis_cell.
#pragma once

namespace snap {

//! lane S's type (coord_rotation.hpp); reached through S's accessor
//! CoordinateImpl::rotation_geometry() const, null without an axis
struct AxialRotationData;

}  // namespace snap
