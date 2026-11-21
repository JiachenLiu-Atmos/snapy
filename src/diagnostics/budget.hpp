#pragma once

// C/C++
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// snap
#include <snap/output/restart.hpp>

#include "consumer.hpp"

namespace snap {

class LayoutImpl;

//! one interval of the budget, summed over blocks (and processes once
//! reduced); every sum is FP64 on the CPU. O8-cols adds the source and
//! residual columns.
struct BudgetRecord {
  std::vector<std::string> rows, counters, minima;  // names
  torch::Tensor total;     // (nrow,) domain totals at the record
  torch::Tensor change;    // (nrow,) total - total at the interval start
  torch::Tensor boundary;  // (nrow,) outward transfer through the boundary
  torch::Tensor counter;   // (ncounter,) int64 monitor counts
  torch::Tensor minimum;   // (nminimum,) monitor minima (+inf: unset)
  double time = 0., interval = 0., dt = 0.;
  int64_t cycle = 0;
};

//! The budget consumer (O.rst section 3.2) without the fixer, source and
//! residual columns (O8-cols, after lane R's R-7). Per block, in FP64:
//! the row totals of the conserved state, the outward register transfer
//! through the fluid domain's boundary (S's wall, outflow and custom faces
//! whose interior cell is fluid, and every face of an immersed solid with
//! one solid neighbour, outward from the fluid; never interior, x1-seam,
//! panel or periodic faces), and the monitor's counters and minima. An
//! interval starts at a record, from a BudgetCarry, or (neither yet) at the
//! first observe, whose own step then stays out.
class BudgetConsumer : public Consumer {
 public:
  //! the row totals of the conserved state u (totals, not anomalies) over
  //! the interior: the cycle line's totals, in its summation order
  static torch::Tensor totals(MeshBlockImpl const& block,
                              torch::Tensor const& u);

  //! the row names: the register's (RegisterRequest::row_names), else mass,
  //! momentum1-3, energy and species/<n> by the conserved layout
  static std::vector<std::string> row_names(MeshBlockImpl const& block);

  //! the outward transfer of the step fields through the boundary faces
  torch::Tensor boundary(MeshBlockImpl const& block, Variables const& vars,
                         FluxRegister const& reg);

  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegister const& reg, RegisterMonitor const& mon,
               double dt) override;

  //! close the interval; local() is this block's part until the next flush
  void flush(MeshBlockImpl const& block, Variables const& vars,
             double time) override;
  BudgetRecord const& local() const { return _local; }

  //! the sums since the interval start, for a restart; `fires_now`: the
  //! record at this time is still to come in the same output pass
  BudgetCarry carry(MeshBlockImpl const& block, Variables const& vars,
                    bool fires_now) const;
  //! resume the interval on `device`, the block's (a restart loads the carry
  //! on the host); a carry with other rows starts a new one
  void resume(BudgetCarry const& carry, torch::Device device);
  bool started() const { return _started; }

 private:
  void start(MeshBlockImpl const& block, Variables const& vars);

  bool _started = false;
  torch::Tensor _total_start, _pending, _counter, _minimum;
  double _interval = 0., _dt = 0.;
  //! per direction: the outward weight (sign x face area) of the counted
  //! faces, and its window in the step field (begin, size along k, j, i)
  std::array<torch::Tensor, 3> _weight;
  std::array<std::array<int64_t, 6>, 3> _window{};
  bool _weights_ready = false;
  BudgetRecord _local;
};

//! sum the local blocks' parts in order, then over the processes (every
//! process gets the record)
BudgetRecord reduce_budget(std::vector<BudgetRecord const*> const& parts,
                           LayoutImpl const& layout);

//! the cycle line (O.rst 1371-1372, content unchanged): the conserved
//! totals are BudgetConsumer::totals; ke, pe and the run-to-date meters are
//! read from hydro. Local blocks first, then one collective per value.
void print_cycle_line(
    std::vector<std::pair<MeshBlockImpl const*, Variables const*>> const&
        blocks,
    double time, double dt, int precision, char const* energy_label);

}  // namespace snap
