#pragma once

// C/C++
#include <map>
#include <memory>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace kintera {
class ThermoYImpl;
}

namespace snap {

class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! Everything a captured stage depends on; any change recaptures it.
struct GraphKey {
  int device_index = -1;
  //! per state tensor: its name, then dtype, shape and strides
  std::vector<std::string> names;
  std::vector<int64_t> layout;
  int store_generation = 0;   // StateStore::generation(): addresses
  int config_generation = 0;  // StageExecutor::bump_config_generation()
  int stage = -1;
  double dt = 0.;  // host double: a new dt recaptures
  //! kintera::alloc_epoch of the EOS thermo (K.rst 3.3): its counter,
  //! active set or table reallocated (reset(), a new shape) recaptures
  int64_t thermo_epoch = 0;

  bool operator==(GraphKey const& other) const;
  bool operator!=(GraphKey const& other) const { return !(*this == other); }
};

//! V3 lane X, X3: runs each stage of a bare MeshBlock's forward() eagerly or
//! as one captured CUDA graph (tests/straka_cuda_graph.py at d6a3c00c, ported
//! and removed). Eager is the default and the reference. cuda-graph captures
//! only the prototype's case (capture_reasons() empty) and runs eagerly
//! otherwise; strict mode refuses such a case before its first launch. Op
//! registration, step hooks, the retry decision and N-block graphs are X4/X5's.
class StageExecutor {
 public:
  explicit StageExecutor(MeshBlockImpl* block);
  ~StageExecutor();

  //! one stage: forward()'s exchange and advance_local, eager or replayed
  void run_stage(Variables& vars, double dt, int stage);

  //! why this case cannot be captured; empty: it can
  std::vector<std::string> capture_reasons(Variables const& vars) const;

  void invalidate();  // drops every graph
  void bump_config_generation() { ++_config_generation; }
  std::map<std::string, int64_t> report() const;

 private:
  struct Captured;

  void run_eager(Variables& vars, double dt, int stage);
  GraphKey make_key(Variables const& vars, double dt, int stage) const;
  //! the EOS's kintera ThermoY (hydro.eos.thermo), or null
  std::shared_ptr<kintera::ThermoYImpl> thermo_of() const;
  //! the save list: the state, the registered storage and hydro_w
  std::map<std::string, torch::Tensor> save_list(Variables& vars) const;
  void capture(Variables& vars, double dt, int stage);
  void replay(Variables& vars, int stage);

  MeshBlockImpl* _block;
  std::map<int, std::shared_ptr<Captured>> _graphs;  // one per stage
  int _config_generation = 0;
  int64_t _captures = 0, _replays = 0, _eager = 0;
};

}  // namespace snap
