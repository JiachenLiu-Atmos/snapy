#pragma once

#include <torch/torch.h>

namespace snap {

// Private, derived data: never register these tensors as module/restart
// buffers. Reuse conversions of coefficients/geometry, not evolving simulation
// state.
// Aliasing contract: sources and returned values must not receive writes
// through independently versioned aliases (including aliases made with
// set_data), raw pointers, or CUDA graph replay. Such writes need not bump the
// version checked here and can leave a stale conversion. Read-only aliases are
// fine; ordinary versioned source edits and set_data on the source itself are
// tracked. If an untracked write is possible, discard/recreate the cache before
// its next use.
class TensorCastCache {
 public:
  torch::Tensor get(torch::Tensor const& src, torch::Tensor const& like) {
    if (src.device() == like.device() &&
        src.scalar_type() == like.scalar_type())
      return src;
    // Inference tensors have no mutation counter. Autograd conversions must
    // belong to the current graph, so neither case can safely reuse this cache.
    if (src.is_inference() || src.requires_grad() ||
        c10::InferenceMode::is_enabled())
      return src.to(like.device(), like.scalar_type()).contiguous();
    if (!value_.defined() || !source_.is_same(src) ||
        source_version_ != src._version() ||
        snapshot_.data_ptr() != src.data_ptr() ||
        snapshot_.device() != src.device() ||
        snapshot_.scalar_type() != src.scalar_type() ||
        snapshot_.sizes() != src.sizes() ||
        snapshot_.strides() != src.strides() ||
        value_.device() != like.device() ||
        value_.scalar_type() != like.scalar_type() ||
        value_version_ != value_._version()) {
      value_ = src.to(like.device(), like.scalar_type()).contiguous();
      source_ = src;
      // A separate TensorImpl preserves metadata across Module::to/set_data.
      snapshot_ = src.detach();
      source_version_ = src._version();
      value_version_ = value_._version();
    }
    return value_;
  }

 private:
  torch::Tensor source_, snapshot_, value_;
  int64_t source_version_ = -1, value_version_ = -1;
};

}  // namespace snap
