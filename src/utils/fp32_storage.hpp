#pragma once

// C/C++
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

//! FP32 storage emulation (fp32 roadmap, Step 0).
//!
//! The arithmetic stays in the state's dtype; only the *stored* state is
//! rounded through float32 (round to nearest even) at the points a float32
//! memory would round it. It measures what float32 storage does to a run
//! before any float32 arithmetic exists.
//!
//! Read once per process from the environment:
//!
//!   SNAP_FP32_STORAGE       off (default) | step | stage
//!     step:  round once per step, after the last RK stage
//!     stage: round after every RK stage
//!   SNAP_FP32_STORAGE_ROWS  all (default) or a comma list of
//!     IDN, IVX, IVY, IVZ, IPR, ICY (all species rows), ICY+k (one species)
//!     or plain row numbers, e.g. "IDN,IPR" or "IPR,ICY+2"
struct Fp32StorageConfig {
  enum Mode { kOff = 0, kStep = 1, kStage = 2 };
  Mode mode = kOff;
  //! the rows to round; empty means every row
  std::vector<std::string> rows;
};

//! the process-wide configuration, parsed once from the environment
Fp32StorageConfig const& fp32_storage_config();

//! parse a mode string ("off", "step", "stage"); throws on anything else
Fp32StorageConfig::Mode parse_fp32_storage_mode(std::string const& s);

//! parse a rows string into row tokens; "" or "all" gives an empty list
std::vector<std::string> parse_fp32_storage_rows(std::string const& s);

//! resolve row tokens to row numbers of a state with `nvar` rows
std::vector<int64_t> resolve_fp32_storage_rows(
    std::vector<std::string> const& rows, int64_t nvar);

//! true when a stage's output must be rounded under `cfg`
bool fp32_storage_due(Fp32StorageConfig const& cfg, int stage, int nstages);

//! Round `u` (rows `rows`, every row if empty) through float32 in place.
//! A float32 state is left unchanged. Ghost cells are rounded with the rest:
//! periodic and reflecting fills are copies or negations, which commute with
//! round-to-nearest-even.
void round_through_fp32_(torch::Tensor u, std::vector<int64_t> const& rows);

}  // namespace snap
