#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// Diagnostics as readers of lane R's register, FROZEN header rev 4
// (happysky19/snapy 175af2fc docs/v3/skeleton/R/flux_register.hpp, "R:").
// Replaces src/diagnostics/ (never compiled). O reads after the accept and
// before the next begin_step (R.rst:494), never posts (R:90).
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
class FluxRegister;     // R:43-89; O reads step_field(dim) only (R:85)
class RegisterMonitor;  // R:91-104; one per register instance (hydro, each
                        // Scalar module), owns the domain totals
using Variables = std::map<std::string, torch::Tensor>;

//! observe: device only, after accept, host dt of the accepted step. The
//! monitor's host read is X's step-end read (R.rst:394-397); flush adds only
//! the cross-rank FP64 reduction at the budget cadence, outside segments.
class Consumer {
 public:
  virtual ~Consumer() = default;
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegister const& reg, RegisterMonitor const& mon,
                       double dt) = 0;
  virtual void flush(double time) = 0;
};

//! residual = delta domain_total() + sum_d boundary(d) - sum amounts(name)
//! over kind(name) in {cell_source, floor, fixer} (R.rst:484-488), FP64.
//! Rows: row_names() (R:94), conserved then derived "e+pe", "aam".
//! boundary(d) is area-integrated (R:98); flux_correction is report only.
class BudgetConsumer : public Consumer {
 public:
  explicit BudgetConsumer(std::vector<RegisterMonitor const*> monitors);
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegister const& reg, RegisterMonitor const& mon,
               double dt) override;
  void flush(double time) override;
};

//! horizontal mean of reg.step_field(1) (R:85, kRegister, per unit area;
//! RegisterRequest::step_field, R:40) times CoordinateImpl::face_area1()
//! (coordinate.hpp:254). Grid "--F". The only place O applies an area.
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
