#pragma once

#include <array>

namespace snap {
// Additive metadata: legacy readers still have totals and exact anomalies.
// Version 1 stores [schema version, stored-state bits] and reference grid
// faces.
inline constexpr int kHydroRestartVersion = 1;
inline constexpr char kHydroRestartMetadata[] = "hydro_restart_metadata";
inline constexpr std::array<char const*, 3> kHydroReferenceGrid = {
    "hydro_ref_x1f", "hydro_ref_x2f", "hydro_ref_x3f"};
}  // namespace snap
