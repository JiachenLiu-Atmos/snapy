#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// One module for restart write, bundle and read. Replaces restart.cpp,
// combine_restart.cpp, read_restart_file.*. Stored vs derived: O.rst sec. 3.
// Register requirement 10 (R header 543ff84f:23-24, "Nothing to restart"):
// the FluxRegister and RegisterMonitor are derived, never written here.
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

// file contract, unchanged (#367: restart_metadata.hpp:9-12)
inline constexpr int kHydroRestartVersion = 1;
inline constexpr char kHydroRestartMetadata[] = "hydro_restart_metadata";
inline constexpr std::array<char const*, 3> kHydroReferenceGrid = {
    "hydro_ref_x1f", "hydro_ref_x2f", "hydro_ref_x3f"};
inline constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";

//! the stored keys (Xi note 1). Every other Variables entry and every
//! register/StateStore tensor stays out (req 10); scalar_r is rebuilt.
inline constexpr std::array<char const*, 12> kRestartStoredKeys = {
    "hydro_u", "hydro_w", "hydro_u_stored", "hydro_ref", "hydro_ref_species",
    "scalar_s", "fill_solid_hydro_u", "boundary_reference_w",
    "boundary_reference_r", "last_time", "last_cycle", "output_key"};

struct OutputSchedule {
  std::vector<int64_t> file_number;  // by position (out<n>)
  std::vector<double> next_time;     // advanced for outputs firing now
  std::vector<int64_t> key;          // N legacy keys, then N v2 keys
};

//! budget sums since the last record (FP64), so an interval closes across
//! a restart; written only when a budget output is requested (Xi note 10)
struct BudgetCarry {  // all tensors float64, stored on CPU
  double last_record_time = 0.;
  std::vector<std::string> rows, fixers;
  torch::Tensor total_prev;     // (nrows,) totals at the last record
  torch::Tensor pending;        // (nrows,) boundary transfer since then
  torch::Tensor fixer_pending;  // (nfixers, nrows) source-fixer amounts
};

void write_restart(MeshBlockImpl* pmb, Variables const& vars, double time,
                   bool final_write, OutputSchedule const& schedule,
                   BudgetCarry const& carry);
//! throws (file named) on a bad header, truncated payload or missing block
Variables read_restart(std::string const& path, int block_rank);
//! #367 checks (meshblock.cpp:1623-1653) next to the contract
void check_restart_metadata(Variables const& data,
                            std::array<torch::Tensor, 3> const& faces);
}  // namespace snap
