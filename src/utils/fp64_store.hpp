#pragma once

// C/C++
#include <optional>
#include <string>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

namespace snap {

//! A buffer holder that stays in float64 (fp32 roadmap, Step 1, D2).
//!
//! Its buffers follow the parent through `to(device)` but ignore
//! `to(dtype)`, so a reference state or an accumulator kept here is never
//! demoted when the rest of the block is converted to float32. The buffers
//! are ordinary registered buffers: they show up in `named_buffers()` (and
//! hence in Python `block.buffer(...)`) under the child's name, e.g.
//! `hydro.ref.u0`.
class Fp64StoreImpl : public torch::nn::Cloneable<Fp64StoreImpl> {
 public:
  Fp64StoreImpl() = default;
  void reset() override {}

  //! buffers are registered lazily, after reset(), so the generic
  //! Cloneable::clone (which re-runs reset and then matches buffer counts)
  //! does not apply; copy them by name instead
  std::shared_ptr<torch::nn::Module> clone(
      const std::optional<torch::Device>& device =
          std::nullopt) const override {
    auto copy = std::make_shared<Fp64StoreImpl>();
    for (auto const& b : named_buffers(/*recurse=*/false)) {
      if (!b.value().defined()) continue;
      copy->set(b.key(), device ? b.value().to(*device) : b.value());
    }
    return copy;
  }

  //! dtype conversions are ignored; device moves are honoured
  void to(torch::Device device, torch::Dtype /*dtype*/,
          bool non_blocking = false) override {
    torch::nn::Module::to(device, non_blocking);
  }
  void to(torch::Dtype /*dtype*/, bool /*non_blocking*/ = false) override {}
  void to(torch::Device device, bool non_blocking = false) override {
    torch::nn::Module::to(device, non_blocking);
  }

  //! true if a buffer called `name` is present and defined
  bool has(std::string const& name) const {
    auto bufs = named_buffers(/*recurse=*/false);  // find() points into it
    auto p = bufs.find(name);
    return p != nullptr && p->defined();
  }

  //! the buffer called `name` (undefined if absent)
  torch::Tensor get(std::string const& name) const {
    auto bufs = named_buffers(/*recurse=*/false);
    auto p = bufs.find(name);
    return p ? *p : torch::Tensor();
  }

  //! store `t` as float64 under `name`, registering it on first use
  torch::Tensor set(std::string const& name, torch::Tensor const& t) {
    auto v = t.detach().to(torch::kFloat64).contiguous().clone();
    auto bufs = named_buffers(/*recurse=*/false);
    auto p = bufs.find(name);
    if (p != nullptr && p->defined()) {
      // Module::register_buffer refuses a second insert; repoint the
      // registered TensorImpl instead so every holder sees the new values
      torch::NoGradGuard ng;
      p->set_(v);
      return *p;
    }
    return register_buffer(name, v);
  }
};
TORCH_MODULE(Fp64Store);

}  // namespace snap
