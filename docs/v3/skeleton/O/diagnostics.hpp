#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// Diagnostics as readers of lane R's register, frozen header rev 4.2
// (happysky19/snapy 4c59162a docs/v3/skeleton/R/flux_register.hpp, "R:").
// Replaces src/diagnostics/ (never compiled). O reads after the accept and
// before the next begin_step (R.rst:560), never posts (R:89).
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
class FluxRegister;     // R:41-88; O reads step_field(dim) only (R:84)
class RegisterMonitor;  // R:90-103; one per register instance (hydro, each
                        // Scalar module), owns the domain totals
using Variables = std::map<std::string, torch::Tensor>;

//! observe: device only, after accept, accumulates amounts every accepted
//! step. flush: O's cross-rank SUM + host read at its cadence (X.rst@bac201d1
//! :165-173); X keeps only the MAX retry read. Outside segments.
class Consumer {
 public:
  virtual ~Consumer() = default;
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegister const& reg, RegisterMonitor const& mon,
                       double dt) = 0;
  virtual void flush(double time) = 0;
};

//! residual = delta domain_total() + sum_d boundary(d) - sum amounts(name)
//! over kind(name) in {cell_source, floor, fixer} (R.rst:550-554), FP64;
//! flux_correction and diagnostic (R:30) are reported only. Rows:
//! row_names() (R:93). boundary(d) is area-integrated (R:97).
class BudgetConsumer : public Consumer {
 public:
  explicit BudgetConsumer(std::vector<RegisterMonitor const*> monitors);
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegister const& reg, RegisterMonitor const& mon,
               double dt) override;
  void flush(double time) override;
};

//! horizontal mean of reg.step_field(1) (R:84, kRegister, per unit area;
//! RegisterRequest::step_field, R:38) times CoordinateImpl::face_area1()
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
