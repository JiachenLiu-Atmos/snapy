#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// Diagnostics as consumers of the flux register and monitor (lane R,
// Linfeng). Replaces src/diagnostics/ (never compiled). Views below are
// lane O's reading of register requirements 3, 6 and 7; R owns the storage.
#include <map>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! req 3: step transfer = sum_s b_s dt F_s, per axis, conserved rows
struct FluxRegisterView {
  torch::Tensor dF1, dF2, dF3;  // undefined for a collapsed axis
  std::vector<std::string> rows;
};

//! req 6/7: device-side monitor; per-fixer amounts, axial closure included
struct MonitorView {
  std::map<std::string, torch::Tensor> fixer_delta;  // name -> (nrows,)
  std::map<std::string, torch::Tensor> counters;     // name -> scalar
};

//! observe: device only, after accept. flush: collective + host read,
//! only at the consumer's cadence, outside every captured segment.
class Consumer {
 public:
  virtual ~Consumer() = default;
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegisterView const& reg, MonitorView const& mon,
                       torch::Tensor dt_device) = 0;
  virtual void flush(double time) = 0;
};

//! total(t) - total(t0) = -boundary + sum fixer + residual, FP64
class BudgetConsumer : public Consumer {
 public:
  struct Row {
    std::string name, units;  // "mass", "species/H2O", "energy", "aam"
    std::vector<int> u_rows;
    torch::Tensor weight;
  };
  explicit BudgetConsumer(std::vector<Row> rows);
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegisterView const& reg, MonitorView const& mon,
               torch::Tensor dt_device) override;
  void flush(double time) override;
};

//! horizontal-mean x1 face flux from the register (grid "--F")
class HydroFluxProfile : public Consumer {
 public:
  void observe(MeshBlockImpl const& block, Variables const& vars,
               FluxRegisterView const& reg, MonitorView const& mon,
               torch::Tensor dt_device) override;
  void flush(double time) override;
};

//! replaces ContainVariable chains (load_diag_output_data.cpp:54-338)
struct FieldDiagnostic {
  std::string key, grid, units, long_name;
  torch::Tensor (*compute)(MeshBlockImpl const&, Variables const&);
};
}  // namespace snap
