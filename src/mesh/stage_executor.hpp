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
class LayoutImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! V3 lane X, X4: the global phase's reduction for a bare MeshBlock and a
//! Mesh's blocks. The partials are summed in block order where they live (one
//! is returned as is); a process group adds one host allreduce.
torch::Tensor global_phase_sum(std::vector<torch::Tensor> const& partials,
                               LayoutImpl const& layout);

//! Everything a captured stage depends on; any change recaptures it.
struct GraphKey {
  int device_index = -1;
  std::vector<std::string> names;  // per state tensor: its name, then
  std::vector<int64_t> layout;     // its dtype, shape and strides
  int store_generation = 0;        // StateStore::generation(): addresses
  int config_generation = 0;       // StageExecutor::bump_config_generation()
  int stage = -1;
  double dt = 0.;  // host double: a new dt recaptures
  //! kintera::alloc_epoch of the EOS thermo (K.rst 3.3): its counter,
  //! active set or table reallocated (reset(), a new shape) recaptures
  int64_t thermo_epoch = 0;

  bool operator==(GraphKey const& other) const;
  bool operator!=(GraphKey const& other) const { return !(*this == other); }
};

//! the capability matrix (X.rst section 10) for one block as configured:
//! numerical support is the eager row, graph support is separate
struct Capability {
  bool eager_fp64 = false, eager_fp32 = false;  // this block's precision
  bool graph_fp64 = false, graph_fp32 = false;  // and whether it can capture
  std::vector<std::string> eager_reasons;       // why its stages run eager
};

//! V3 lane X, X3: a bare MeshBlock's stages run eagerly (default, reference)
//! or as CUDA graphs when capture_reasons() is empty; strict refuses the rest.
class StageExecutor {
 public:
  explicit StageExecutor(MeshBlockImpl* block);
  ~StageExecutor();

  void run_stage(Variables& vars, double dt, int stage);  // eager or replayed

  //! why this case cannot be captured; empty: it can
  std::vector<std::string> capture_reasons(Variables const& vars) const;
  //! V3 X7: this block's row of the capability matrix (G7)
  Capability capability(Variables const& vars) const;

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
