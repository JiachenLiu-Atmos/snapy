#pragma once

// C/C++
#include <map>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include "precision_policy.hpp"

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

//! The block's state in fixed storage (V3 lane X, X2): hydro_u, scalar_s,
//! scalar_r and the RK backups u0, s0, in the storage dtype. The RK combine
//! writes into hydro_u and scalar_s (pyharp out=), so their addresses change
//! only on reset(), reallocate() (to() and restart) and adopt(), and each of
//! those bumps generation(). Other owners' tensors that a captured stage
//! reads or writes are registered by name (X3: the flux register's); a new
//! address bumps generation() too.
class StateStore {
 public:
  //! allocates every buffer; u0 and s0 are the ones MeshBlock registers
  void reset(PrecisionPolicy const& policy, int64_t nvar_hydro,
             int64_t nvar_scalar, int64_t nc3, int64_t nc2, int64_t nc1) {
    auto dtype = policy.storage(Quantity::kBackup);
    _u0 = torch::zeros({nvar_hydro, nc3, nc2, nc1}, dtype);
    _s0 = torch::zeros({nvar_scalar, nc3, nc2, nc1}, dtype);
    reallocate();
  }

  //! new hydro_u, scalar_s and scalar_r like u0 and s0 (device, dtype);
  //! the caller's tensors are adopted at the next adopt()
  void reallocate() {
    _hydro_u = torch::zeros_like(_u0);
    _scalar_s = torch::zeros_like(_s0);
    _scalar_r = torch::zeros_like(_s0);
    ++_generation;
  }

  //! Binds vars' hydro_u and scalar_s to the store. A tensor that is not the
  //! store's (first use, or a caller rebound the key between steps) is copied
  //! in and then set_ to the store's storage, so the caller's handle stays
  //! valid and the address stays fixed; this bumps generation(). scalar_r is
  //! only tracked: its divide still set_data's it (fixed storage with G1).
  void adopt(Variables& vars) {
    bool moved = adopt(vars, "hydro_u", _hydro_u);
    if (_s0.size(0) > 0) {
      moved |= adopt(vars, "scalar_s", _scalar_s);
      moved |= track(vars, "scalar_r", _scalar_r);
    }
    if (moved) ++_generation;
  }

  void backup() {  // stage 0
    _u0.copy_(_hydro_u);
    if (_s0.size(0) > 0) _s0.copy_(_scalar_s);
  }
  void restore() {  // rejected step
    _hydro_u.copy_(_u0);
    if (_s0.size(0) > 0) _scalar_s.copy_(_s0);
  }

  //! an owner's tensor, allocated by its owner; re-registering the same
  //! address is a no-op, a new one bumps generation()
  void register_storage(std::string const& name, torch::Tensor const& t) {
    auto it = _storage.find(name);
    if (it != _storage.end() && it->second.defined() == t.defined() &&
        (!t.defined() || (it->second.data_ptr() == t.data_ptr() &&
                          it->second.sizes() == t.sizes() &&
                          it->second.device() == t.device())))
      return;
    _storage[name] = t;
    ++_generation;
  }
  torch::Tensor const& storage(std::string const& name) const {
    auto it = _storage.find(name);
    TORCH_CHECK(it != _storage.end(), "StateStore: no storage named ", name);
    return it->second;
  }
  std::map<std::string, torch::Tensor> const& registered() const {
    return _storage;
  }

  torch::Tensor& hydro_u() { return _hydro_u; }
  torch::Tensor& scalar_s() { return _scalar_s; }
  torch::Tensor& scalar_r() { return _scalar_r; }
  torch::Tensor& u0() { return _u0; }
  torch::Tensor& s0() { return _s0; }
  int generation() const { return _generation; }

 private:
  bool adopt(Variables& vars, std::string const& key, torch::Tensor& slot) {
    auto it = vars.find(key);
    if (it == vars.end() || !it->second.defined() || it->second.is_same(slot))
      return false;
    auto given = it->second;
    TORCH_CHECK(given.sizes() == slot.sizes() &&
                    given.dtype() == slot.dtype() &&
                    given.device() == slot.device(),
                "StateStore: ", key, " has shape ", given.sizes(), ", ",
                given.dtype(), " on ", given.device(), "; the block holds ",
                slot.sizes(), ", ", slot.dtype(), " on ", slot.device());
    slot.copy_(given);
    given.set_(slot);
    slot = given;
    return true;
  }

  //! scalar_r: hold the caller's tensor (its dtype follows the divide)
  bool track(Variables& vars, std::string const& key, torch::Tensor& slot) {
    auto it = vars.find(key);
    if (it == vars.end() || !it->second.defined() || it->second.is_same(slot))
      return false;
    slot = it->second;
    return true;
  }

  torch::Tensor _hydro_u, _scalar_s, _scalar_r, _u0, _s0;
  std::map<std::string, torch::Tensor> _storage;  // register_storage()
  int _generation = 0;
};

}  // namespace snap
