#pragma once

// C/C++
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/utils/step_dt.hpp>

#include "segment.hpp"

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
  int block = -1, stage = -1,
      segment = -1;  // one graph per (block, stage, segment)
  //! the host dt, only for a segment holding an op that still reads it
  //! (Op::reads_host_dt): a new dt recaptures that segment; else 0. Every
  //! other segment reads the executor's device dt (V3 D) and keeps no dt
  double dt = 0.;
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
  //! why its stages run eager: the case's reasons and every eager op's
  std::vector<std::string> eager_reasons;
  //! one row per segment of the block's stage schedule
  struct SegmentRow {
    int index = 0;
    std::string kind;
    bool capturable = false;     //!< runs as a graph in this case
    bool reads_host_dt = false;  //!< dt is in its graph key
    bool host = false;  //!< a host-exchange or global segment (a boundary)
    std::vector<std::string> ops;
    std::vector<std::string> reasons;  //!< why it runs eager
  };
  std::vector<SegmentRow> segments;
};

//! V3 lane X, X3: a block's stages run eagerly (default, reference) or as CUDA
//! graphs when capture_reasons() is empty; strict refuses the rest. A Mesh
//! owns one executor for all its blocks (lane B); a bare MeshBlock owns a
//! one-block one. Blocks are addressed by their index in the executor.
class StageExecutor {
 public:
  explicit StageExecutor(MeshBlockImpl* block);
  explicit StageExecutor(std::vector<MeshBlockImpl*> blocks);
  ~StageExecutor();

  int num_blocks() const { return static_cast<int>(_blocks.size()); }
  //! the block at index `block` (checked)
  MeshBlockImpl* block_at(int block) const;
  //! the ops and segments of the block's stages (A2: one segment per stage,
  //! the exchange and advance_local)
  StageSchedule const& schedule(int block) const;

  //! a bare block's stage: exchange_ghost_zones, then advance_local, eager or
  //! replayed
  void run_stage(int block, Variables& vars, double dt, int stage);
  //! a Mesh block's local stage (advance_local, eager): the Mesh exchanges
  //! last and runs the global phase itself. Safe on the block worker pool.
  void run_local(int block, Variables& vars, double dt, int stage);

  //! why this block cannot be captured; empty: it can
  std::vector<std::string> capture_reasons(int block,
                                           Variables const& vars) const;
  //! V3 X7: this block's row of the capability matrix (G7)
  Capability capability(int block, Variables const& vars) const;

  void invalidate();           // drops every graph
  void invalidate(int block);  // drops this block's graphs
  void bump_config_generation() { ++_config_generation; }
  std::map<std::string, int64_t> report() const;

 private:
  struct Captured;
  using Slot = std::tuple<int, int, int>;  // (block, stage, segment)

  void build_schedule(int block);
  //! the ops of a segment (all segments: -1), in order, eagerly
  void run_eager(int block, Variables& vars, StepDt dt, int stage,
                 int segment = -1);
  GraphKey make_key(int block, int segment, Variables const& vars, double dt,
                    int stage) const;
  //! the block's device dt, which every graph of the block reads, set to dt
  //! in stream order (V3 D); one per block, so blocks replaying on their own
  //! streams (a Mesh's worker pool) never write one another's
  StepDt device_dt(int block, Variables const& vars, double dt);
  //! the EOS's kintera ThermoY (hydro.eos.thermo), or null
  std::shared_ptr<kintera::ThermoYImpl> thermo_of(int block) const;
  //! the save list: the state, the registered storage and hydro_w
  std::map<std::string, torch::Tensor> save_list(int block,
                                                 Variables& vars) const;
  void capture(int block, int segment, Variables& vars, double dt, int stage);
  void replay(int block, int segment, Variables& vars, double dt, int stage);

  std::vector<MeshBlockImpl*> _blocks;
  std::vector<StageSchedule> _schedules;  // one per block
  std::map<Slot, std::shared_ptr<Captured>> _graphs;
  //! V3 D: one persistent 0-dim FP64 dt per block on its device, shared by
  //! every graph of that block
  std::vector<torch::Tensor> _dt;
  int _config_generation = 0;
  //! run_local runs on the block worker pool
  std::atomic<int64_t> _captures{0}, _replays{0}, _eager{0};
  //! host-exchange and global segments run under cuda-graph (boundaries)
  std::atomic<int64_t> _host{0};
};

}  // namespace snap
