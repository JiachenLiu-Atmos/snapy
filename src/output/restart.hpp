#pragma once

// C/C++
#include <array>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/layout/layout.hpp>

#include "restart_metadata.hpp"

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

//! the restart variables of block `block_rank`, from a bundle or from a
//! single part file; throws if the file cannot be read, the bundle is
//! malformed or truncated, or it holds no part for the block
Variables load_restart(std::string const& path, int block_rank = get_rank());

//! the #367 checks: the hydro metadata, the stored dtype and shape, and the
//! reference grid against this block's faces x1f, x2f, x3f; perturbation-flux
//! needs the original reference
void check_restart_metadata(Variables const& data,
                            std::array<torch::Tensor, 3> const& grid,
                            bool perturbation_flux);

//! a budget interval still open at a restart (O.rst 3.4, Xi note 10): the
//! block's sums since its last record, FP64 (counts int64) on the CPU.
//! Written only when a budget output is requested; a file without one starts
//! a new interval at the first observe. The fixer sums come with O8-cols.
struct BudgetCarry {
  std::vector<std::string> rows;  // row names; resumed only if they match
  torch::Tensor total_prev;       // (nrows,) totals at the interval start
  torch::Tensor pending;          // (nrows,) boundary transfer since then
  torch::Tensor counter;          // (ncounters,) monitor counts since then
  torch::Tensor minimum;          // (nminima,) monitor minima since then
  double interval = 0.;           // accepted time since then
  bool defined() const { return total_prev.defined(); }
};

//! restart keys of the carry: kBudgetCarryPrefix + its field
inline constexpr char kBudgetCarryPrefix[] = "budget_carry.";
void write_budget_carry(BudgetCarry const& carry, Variables& out);
//! undefined when the file holds none or one with other rows
BudgetCarry read_budget_carry(Variables const& data,
                              std::vector<std::string> const& rows);

}  // namespace snap
