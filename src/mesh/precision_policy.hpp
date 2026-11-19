#pragma once

// C/C++
#include <array>
#include <string>

// torch
#include <torch/torch.h>

namespace snap {

//! the quantities with a fixed dtype contract (V3 lane X, section 6)
enum class Quantity {
  kHydroState,   // hydro_u, anomaly form
  kBackup,       // u0, s0: always the storage dtype
  kScalarState,  // scalar_s
  kFlux,         // fluxes, du
  kReference,    // reference and geometry masters
  kPrimitive,    // hydro_w totals
  kGauge,        // gauges, total face pressure
  kReduction,    // dt, budget and closure sums
  kImplicit,     // vertical implicit solve
  kThermo,       // kintera totals, saturation
  kRegister,     // flux register step accumulator
  kSource,       // stage-local source accumulator
  kDiagnostic,   // diagnostic sums
  kCount
};

struct DtypeRow {
  torch::Dtype storage, compute;
};

//! One table of dtypes per quantity, built once from dynamics.precision.
//! The reference is stored FP64 and cast per use. MPS has no FP64: every FP64
//! row holds FP32 there.
class PrecisionPolicy {
 public:
  explicit PrecisionPolicy(std::string const& precision = "float64",
                           torch::Device device = torch::kCPU)
      : _fp32(precision == "float32") {
    auto wide = device.is_mps() ? torch::kFloat32 : torch::kFloat64;
    auto narrow = _fp32 ? torch::kFloat32 : wide;
    _rows.fill({wide, wide});
    for (auto q :
         {Quantity::kHydroState, Quantity::kBackup, Quantity::kScalarState,
          Quantity::kFlux, Quantity::kImplicit})
      _rows[static_cast<int>(q)] = {narrow, narrow};
    _rows[static_cast<int>(Quantity::kReference)].compute = narrow;
  }

  bool fp32() const { return _fp32; }
  DtypeRow const& row(Quantity q) const { return _rows[static_cast<int>(q)]; }
  torch::Dtype storage(Quantity q) const { return row(q).storage; }
  torch::Dtype compute(Quantity q) const { return row(q).compute; }
  //! x in the compute (cast) or storage (store) dtype of q; x if it already is
  torch::Tensor cast(Quantity q, torch::Tensor const& x) const {
    return x.to(compute(q));
  }
  torch::Tensor store(Quantity q, torch::Tensor const& x) const {
    return x.to(storage(q));
  }

 private:
  bool _fp32;
  std::array<DtypeRow, static_cast<int>(Quantity::kCount)> _rows;
};

}  // namespace snap
