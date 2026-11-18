#pragma once

// C/C++
#include <array>

// torch
#include <torch/torch.h>

namespace snap {

//! The EOS/floor/positivity status of one MeshBlock, in named slots (V3
//! lane EF, D4). The hydro and scalar modules and the EOS write to it in
//! place; MeshBlock::check_redo reads the redo causes from it in its one
//! device-to-host copy. It replaces the EOS's mark queues and vapor-failure
//! count, the ideal-moist heat-capacity latch, and the positivity meters of
//! HydroImpl and ScalarImpl.
//!
//! A meter slot is the tensor its module registers as a buffer (the same
//! TensorImpl), so the buffer names stay and Module::to moves both.
class EfStatus {
 public:
  enum Slot : int {
    kMarks = 0,      //!< bool [2]: [0] a floor patched an interior cell,
                     //!< [1] an interior NaN; undefined until armed
    kVaporFail,      //!< int32 0-d: deferred vapor-repair failures
    kHeatCapacity,   //!< bool 0-d: ideal-moist heat-capacity failure latch
    kPosHits,        //!< int64 [1]: (cell, species) entries with theta < 1
    kPosSevere,      //!< int64 [1]: theta < 0.9, withheld above round-off
    kPosMin,         //!< FP64 [1]: the smallest theta
    kLimCut,         //!< FP64 [1]: x1 face flux the limiter removed
    kLimFlux,        //!< FP64 [1]: x1 face flux it was offered
    kScalarPosHits,  //!< int64 [1]: scalar entries with theta < 1
    kNumSlots
  };

  torch::Tensor& slot(Slot s) { return slots_[s]; }
  torch::Tensor const& slot(Slot s) const { return slots_[s]; }

  //! store a module's meter in slot s; returns it for register_buffer
  torch::Tensor& bind(Slot s, torch::Tensor t) {
    slots_[s] = std::move(t);
    return slots_[s];
  }

  //! arm the limiter marks, all false, on the device of like
  void reset_marks(torch::Tensor const& like) {
    auto& m = slots_[kMarks];
    if (m.defined() && m.device() == like.device())
      m.zero_();
    else
      m = torch::zeros({2}, like.options().dtype(torch::kBool));
  }

  //! OR a 0-d bool into mark i (0: floor patch, 1: NaN) if armed
  void mark(int i, torch::Tensor const& any) {
    if (slots_[kMarks].defined()) slots_[kMarks][i].logical_or_(any);
  }
  bool marks_armed() const { return slots_[kMarks].defined(); }

  //! a 0-d slot of dtype on the device of like, allocated zero on first use
  //! or after a device change; never zeroed here
  torch::Tensor& flag(Slot s, torch::Tensor const& like,
                      torch::ScalarType dtype) {
    auto& f = slots_[s];
    if (!f.defined() || f.device() != like.device())
      f = torch::zeros({}, like.options().dtype(dtype));
    return f;
  }

  //! zero a 0-d flag if it exists
  void clear(Slot s) {
    if (slots_[s].defined()) slots_[s].zero_();
  }

 private:
  std::array<torch::Tensor, kNumSlots> slots_;
};

}  // namespace snap
