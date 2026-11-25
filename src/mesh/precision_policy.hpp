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
  kImplicit,     // vertical implicit solve
  kCount
};

struct DtypeRow {
  torch::Dtype storage, compute;
};

//! One table of dtypes per quantity, built once from dynamics.precision and
//! integration.implicit-precision. The reference is stored FP64 and cast per
//! use. MPS has no FP64: every FP64 row holds FP32 there.
class PrecisionPolicy {
 public:
  explicit PrecisionPolicy(std::string const& precision = "float64",
                           torch::Device device = torch::kCPU,
                           std::string const& implicit = "auto")
      : _fp32(precision == "float32") {
    auto const w = wide(device);
    auto narrow = _fp32 ? torch::kFloat32 : w;
    _rows.fill({w, w});
    for (auto q :
         {Quantity::kHydroState, Quantity::kBackup, Quantity::kScalarState,
          Quantity::kFlux, Quantity::kImplicit})
      _rows[static_cast<int>(q)] = {narrow, narrow};
    _rows[static_cast<int>(Quantity::kReference)].compute = narrow;
    if (implicit != "auto")  // float32 or float64 overrides the block
      _rows[static_cast<int>(Quantity::kImplicit)].compute =
          implicit == "float32" ? torch::kFloat32 : w;
  }

  //! the widest dtype a device holds: FP64, except FP32 on MPS (no FP64)
  static torch::Dtype wide(torch::Device device) {
    return device.is_mps() ? torch::kFloat32 : torch::kFloat64;
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
