#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// Diagnostics as readers of lane R's register (provisional header,
// happysky19/snapy 543ff84f docs/v3/skeleton/R/flux_register.hpp, "R:" below).
// Replaces src/diagnostics/ (never compiled). O reads after the accept and
// never posts (R:96). Points marked R-O1..3 wait on O's conflict note to R.
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
class CoordinateImpl;
class FluxRegister;     // R:62-94; O reads step_field(dim) only (R:91)
class RegisterMonitor;  // R:97-104; R owns the domain totals (R:21-22)
using Variables = std::map<std::string, torch::Tensor>;

//! observe: device only, after accept, X's host dt. flush: collective +
//! host read at the consumer's cadence, outside every captured segment.
class Consumer {
 public:
  virtual ~Consumer() = default;
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegister const& reg, RegisterMonitor const& mon,
                       double dt) = 0;
  virtual void flush(double time) = 0;
};

//! total - total_prev = -sum_d boundary(d) + sum(cell sources) + residual,
//! FP64. total = mon.domain_total() (R:101), boundary = mon.boundary(dim)
//! (R:100), sources = mon.amounts(name) (R:99) of cell_amount posts (R:84).
//! Row names (R-O1), counter names (R-O2) and amount kinds (R-O3) pending.
class BudgetConsumer : public Consumer {
 public:
  struct Row {
    std::string name, units;  // "mass", "species/H2O", "energy", "aam"
  };
  explicit BudgetConsumer(std::vector<Row> rows);
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegister const& reg, RegisterMonitor const& mon,
               double dt) override;
  void flush(double time) override;
};

//! horizontal mean of the x1 step transfer: reg.step_field(1) (R:91, FP64,
//! per unit area; dim 1 requested in RegisterRequest::step_field, R:56-59)
//! times CoordinateImpl::face_area1() (coordinate.hpp:254). Grid "--F".
class HydroFluxProfile : public Consumer {
 public:
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegister const& reg, RegisterMonitor const& mon,
               double dt) override;
  void flush(double time) override;
};

//! replaces ContainVariable chains (load_diag_output_data.cpp:54-338)
struct FieldDiagnostic {
  std::string key, grid, units, long_name;
  torch::Tensor (*compute)(MeshBlockImpl const&, Variables const&);
};
}  // namespace snap
