#pragma once
// V3 lane O skeleton (declarations only; not built).
//
// Diagnostics as consumers of the flux register and the monitor (lane R).
// Replaces the whole of src/diagnostics/ at pin 9e5de2d4: Athena++/canoe
// code that is outside every CMake glob (src/CMakeLists.txt:35-53 lists no
// diagnostics/; src/diagnostics/CMakeLists.txt is reached by no
// add_subdirectory) and includes headers that do not exist in snapy
// (<athena/...>, <exchanger/...>: diagnostics.hpp:8-13).
//
// The live per-cell diagnostics (thermo, div, curl, ic_*, path_*, avg_*) stay
// field loaders of the writer core (output/load_diag_output_data.cpp); they
// get a registry instead of the ContainVariable chains.
//
// ASSUMED lane R interfaces (not available to this revision; see O.rst,
// "Assumptions"). Names and shapes below are placeholders for R to confirm.

#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {

class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

// ------------------------------------------------------------------------
// Read-only views onto lane R state (assumption A1, A2)
// ------------------------------------------------------------------------

//! A1: the flux register. Face fluxes of the conserved rows, integrated over
//! the accepted step with the RK weights, i.e. sum_s b_s dt F_s, per axis.
//! Device tensors, owned by R; reset by R at step start and restored by R on
//! a redo. Consumers read only between accepted steps.
struct FluxRegisterView {
  //! (nvar, nc3, nc2, nc1+1) etc.; undefined for a collapsed axis
  torch::Tensor dF1, dF2, dF3;
  //! row names in conserved order (rho, mom1..3, etot, species...)
  std::vector<std::string> rows;
  //! FP64 (assumption A1b: R accumulates in FP64 even in an FP32 run)
  torch::Dtype dtype() const;
};

//! A2: the monitor. Per-fixer conserved-quantity changes in the accepted
//! step (cell-integrated, FP64 device scalars or (nvar,) vectors) and the
//! event counters that print_cycle_diagnostics reads today
//! (meshblock.cpp:1287-1296, 1336-1337): limiter cut/flux, positivity
//! severe/min, VIC clamp residual, gravity-work fix, AAM closure sums
//! (hydro.hpp:235-311).
struct MonitorView {
  //! fixer name -> (nvar,) change of each conserved row this step
  std::map<std::string, torch::Tensor> fixer_delta;
  //! counter name -> scalar tensor
  std::map<std::string, torch::Tensor> counters;
};

// ------------------------------------------------------------------------
// Consumers
// ------------------------------------------------------------------------

//! Something that observes the state after an accepted step. `observe` runs
//! on device and does not read to the host; `flush` is the only place a
//! consumer may read to the host, and only at its output cadence.
class Consumer {
 public:
  virtual ~Consumer() = default;
  virtual std::string name() const = 0;
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegisterView const& reg, MonitorView const& mon,
                       torch::Tensor dt_device) = 0;
  //! collective over the process group (sum/min/max), then host read
  virtual void flush(double time) = 0;
};

//! The budget: for each row Q (mass, each species, E + PE, AAM about the
//! rotation axis, each passive scalar)
//!   Q(t1) - Q(t0) = - (boundary flux, from the register)
//!                   + sum over fixers (from the monitor) + residual
//! accumulated in FP64 on device and written by BudgetSeries.
class BudgetConsumer : public Consumer {
 public:
  struct Row {
    std::string name;      //!< "mass", "species/H2O", "energy", "aam", ...
    std::string units;     //!< "kg", "J", "kg m^2 s^-1"
    torch::Tensor weight;  //!< cell weight giving Q = sum(weight * u[rows])
    std::vector<int> u_rows;
  };
  explicit BudgetConsumer(std::vector<Row> rows);
  std::string name() const override { return "budget"; }
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegisterView const& reg, MonitorView const& mon,
               torch::Tensor dt_device) override;
  void flush(double time) override;  //!< hands one record to BudgetSeries

 private:
  std::vector<Row> rows_;
  torch::Tensor total_t0_;    //!< (nrows,) FP64, at the last record
  torch::Tensor boundary_;    //!< (nrows,) FP64, integrated since last record
  torch::Tensor fixers_;      //!< (nfixers, nrows) FP64
  std::vector<std::string> fixer_names_;
};

//! Horizontal-mean vertical flux profile from the register (the intent of
//! the legacy diagnostics/hydro_flux.cpp, which recomputed rho*v1 from
//! cell centres, hydro_flux.cpp:56-63). Output grid "--F" (x1 faces).
class HydroFluxProfile : public Consumer {
 public:
  std::string name() const override { return "hydro_flux"; }
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegisterView const& reg, MonitorView const& mon,
               torch::Tensor dt_device) override;
  void flush(double time) override;

 private:
  torch::Tensor sum_;  //!< (nvar, nc1+1) FP64, time-integrated face flux
  double elapsed_ = 0.;
};

//! One budget record file per run: NetCDF4, unlimited `time`, NC_DOUBLE
//! always. Root rank only. Format in O.rst, question 3.
class BudgetSeries {
 public:
  BudgetSeries(std::string path, std::vector<std::string> rows,
               std::vector<std::string> fixers);
  //! append one record; values are host FP64 after the collective
  void append(double time, int64_t cycle, double dt,
              torch::Tensor total,     // (nrows,)
              torch::Tensor boundary,  // (nrows,)
              torch::Tensor fixers,    // (nfixers, nrows)
              std::map<std::string, double> const& counters);
};

// ------------------------------------------------------------------------
// Field-diagnostic registry (replaces ContainVariable chains in
// load_diag_output_data.cpp:54-338)
// ------------------------------------------------------------------------
struct FieldDiagnostic {
  std::string key;        //!< the `variables:` token that requests it
  std::string grid;       //!< "CCC", "--C", ...
  std::string units, long_name;
  //! returns (ncomp, n3, n2, n1); runs on device
  torch::Tensor (*compute)(MeshBlockImpl const&, Variables const&);
};
std::vector<FieldDiagnostic> const& field_diagnostics();

}  // namespace snap
