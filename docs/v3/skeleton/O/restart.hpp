#pragma once
// V3 lane O skeleton (declarations only; not built).
//
// Restart: one module for write, bundle and read. Replaces at pin 9e5de2d4
// output/restart.cpp (130), output/combine_restart.cpp (123) and
// input/read_restart_file.{hpp,cpp} (257), which carry the bundle magic twice
// (combine_restart.cpp:16, read_restart_file.cpp:23) and fail on a bad bundle
// by printing to std::cerr and returning an empty map
// (read_restart_file.cpp:157-220).
//
// Rule (V3_ROADMAP.md, "Derived caches and graph executables are never
// serialized into restart files"): a restart stores the minimum from which
// every other buffer is rebuilt. The tables below are normative.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

// ---- file contract (kept from #367: restart_metadata.hpp:9-12) ----------
inline constexpr int kHydroRestartVersion = 1;
inline constexpr char kHydroRestartMetadata[] = "hydro_restart_metadata";
inline constexpr std::array<char const*, 3> kHydroReferenceGrid = {
    "hydro_ref_x1f", "hydro_ref_x2f", "hydro_ref_x3f"};
inline constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";

//! STORED: written by every restart, read back verbatim.
//!   hydro_u           totals, FP64 (restart.cpp:44)
//!   hydro_u_stored    the anomaly bits in the run dtype (restart.cpp:45)
//!   hydro_ref, hydro_ref_species, hydro_ref_x{1,2,3}f   FP64 background and
//!                     the grid it belongs to (#367; restart.cpp:46-61)
//!   hydro_restart_metadata  [version, stored-state bits] (restart.cpp:50-53)
//!   scalar_s          (scalar_r is rebuilt; restart.cpp:30-34)
//!   fill_solid_hydro_u, boundary_reference_w/_r  when the block has them
//!   last_time, last_cycle
//!   OutputSchedule    (below)
//!   BudgetCarry       (below; new in V3)
//!
//! DERIVED: never written; rebuilt on load or on first use.
//!   hydro_w, scalar_r               from U (meshblock.cpp:1827-1829)
//!   flux register, monitor counters reset at load (lane R); the budget
//!                                   restarts its interval from BudgetCarry
//!   CUDA graph executables, captured workspaces, cast caches
//!                                   (utils/tensor_cast_cache.hpp) - lane X
//!   coordinate/metric tensors, reconstruction coefficients, EOS tables
struct OutputSchedule {
  std::vector<int64_t> file_number;  //!< per output, by position (out<n>)
  std::vector<double> next_time;     //!< advanced for outputs firing now (#313)
  std::vector<int64_t> key;          //!< N legacy keys then N v2 keys (#313)
};

//! New: what the budget time series needs to continue across a restart
//! without a jump. The tech report at 919936da notes run-to-date meters are
//! not carried across a restart today (chapter 16, known limits).
struct BudgetCarry {
  double last_record_time = 0.;
  //! per budget row: the integrated boundary flux and fixer contributions
  //! since the last record (FP64), so an interval that straddles the restart
  //! closes. Row names are stored next to the values.
  std::vector<std::string> rows;
  torch::Tensor pending;  //!< (nrows,) float64
};

//! Write this block's part file and, on the root, bundle the parts.
//! The bundle format and file names are unchanged from the pin.
void write_restart(class MeshBlockImpl* pmb, Variables const& vars,
                   double time, bool final_write,
                   OutputSchedule const& schedule, BudgetCarry const& carry);

//! Read the part of `block_rank` from a bundle or a single part file.
//! Throws (instead of returning {}) on a missing block, a truncated payload
//! or a bad header; the message names the file.
Variables read_restart(std::string const& path, int block_rank);

//! Validate the #367 metadata against the block (version, dtype bits, shape,
//! reference grid faces); throws with the existing messages
//! (meshblock.cpp:1623-1653). Lives here so the file contract and its check
//! are in one place; MeshBlock calls it.
void check_restart_metadata(Variables const& data,
                            std::array<torch::Tensor, 3> const& faces);

}  // namespace snap
