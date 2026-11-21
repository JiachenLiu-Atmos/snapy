#pragma once

// C/C++
#include <map>
#include <string>

// torch
#include <torch/torch.h>

namespace snap {

class MeshBlockImpl;
class FluxRegister;
class RegisterMonitor;
using Variables = std::map<std::string, torch::Tensor>;

//! A diagnostic that reads lane R's register (V3 lane O step O8, O.rst
//! section 3.2): O reads after the accept and before the next begin_step,
//! and never posts. observe runs on the device after the global accept of
//! every step (never for a rejected attempt); flush closes the interval at
//! the consumer's own cadence. The collective and the host read happen
//! once per process for all its local blocks, in the caller (one block's
//! flush cannot see the others).
class Consumer {
 public:
  virtual ~Consumer() = default;

  //! the accepted step of length dt: reg's step fields, mon's census
  virtual void observe(MeshBlockImpl const& block, Variables const& vars,
                       FluxRegister const& reg, RegisterMonitor const& mon,
                       double dt) = 0;

  //! close this block's interval at time `time`
  virtual void flush(MeshBlockImpl const& block, Variables const& vars,
                     double time) = 0;
};

}  // namespace snap
