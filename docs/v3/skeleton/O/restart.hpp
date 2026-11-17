#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// One module for restart write, bundle and read. Replaces restart.cpp,
// combine_restart.cpp, read_restart_file.*. Stored vs derived: O.rst sec. 3.
// Register requirement 10: the flux register and monitor are derived and are
// never written here.
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

struct OutputSchedule {
  std::vector<int64_t> file_number;  // by position (out<n>)
  std::vector<double> next_time;     // advanced for outputs firing now
  std::vector<int64_t> key;          // N legacy keys, then N v2 keys
};

//! budget sums since the last record (FP64), so an interval closes across
//! a restart; output bookkeeping, not register state
struct BudgetCarry {
  double last_record_time = 0.;
  std::vector<std::string> rows;
  torch::Tensor pending;
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
