#pragma once

// C/C++
#include <string>

// snap
#include <snap/diagnostics/budget.hpp>

#include "output_type.hpp"

namespace snap {

//! The budget time series (O.rst 3.2, budget_series.cdl): one NetCDF4 file
//! per run, <basename>.budget.nc in the output directory, appended by the
//! root process, one record per interval; every sum is NC_DOUBLE, counts
//! NC_INT64. Variables: time, cycle, dt, interval; row_name, counter_name,
//! minimum_name; total, change, boundary (time, row); counter, minimum.
//! The source and residual variables come with O8-cols.
class BudgetSeries {
 public:
  //! append `record` to `path`, creating the file on this process's first
  //! record unless `resume` and the file exists; throws without NETCDF
  static void append(std::string const& path, BudgetRecord const& record,
                     bool resume);
};

//! `- {type: budget, dt: ...}`: observes every accepted step and records
//! the budget at its cadence and on the final write. The local blocks of a
//! process are summed in block order, then over the processes, once the
//! last of them records.
class BudgetOutput : public OutputType {
 public:
  explicit BudgetOutput(OutputOptions const& options_);

  void bind(MeshBlockImpl* pmb) override;
  void observe(MeshBlockImpl const* pmb, Variables const& vars,
               double dt) override;
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
  void save_carry(MeshBlockImpl const* pmb, Variables const& vars,
                  bool fires_now, Variables& out) const override;
  void load_carry(Variables const& data) override;

  BudgetConsumer const& consumer() const { return _consumer; }
  //! the domain record of the last interval (every process holds it)
  BudgetRecord const& record() const { return _record; }
  //! the series file of block pmb
  static std::string path(MeshBlockImpl const* pmb);

 private:
  BudgetConsumer _consumer;
  BudgetRecord _record;
  std::vector<std::string> _rows;
  bool _resumed = false;
};

}  // namespace snap
