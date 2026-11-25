#pragma once

// C/C++
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/utils/step_dt.hpp>

#include "exchange_cut.hpp"
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
  //! the implicit correction's options its call reads (hydro.cpp), baked
  //! into a capture: a change recaptures (V3 C1; X.rst:296). The solve reads
  //! the block's device dt (C4), so its segment keeps no dt above
  std::string implicit_precision;
  bool implicit_fallback_fp64 = false;
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

//! V3 lane X, X3: a block's stages run eagerly or as CUDA graphs when
//! capture_reasons() is empty (mode(): execution auto is cuda-graph on a CUDA
//! device); a refused case runs eager and its reasons are logged once, and
//! strict refuses it instead. A Mesh
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
  //! the ops and segments of the block's stages (a bare block's begin with
  //! its stage-entry exchange, a Mesh block's do not)
  StageSchedule const& schedule(int block) const;

  //! the block's execution mode, "eager" or "cuda-graph":
  //! integration.execution, with auto resolved to cuda-graph when the block is
  //! on a CUDA device (or the case is strict) and eager otherwise
  std::string mode(int block) const;

  //! the block's stage, its schedule eager or its segments' graphs (B2). A
  //! Mesh block's has no entry exchange (the Mesh exchanges last and runs
  //! the global phase). Safe on the block worker pool: blocks capture
  //! together (thread-local capture mode, after align_captures) and replay
  //! concurrently on their streams.
  void run_stage(int block, Variables& vars, double dt, int stage);
  //! the stage's ops without the stage-entry exchange, eagerly
  //! (MeshBlockImpl::advance_local)
  void run_local(int block, Variables& vars, double dt, int stage);
  //! a multi-block executor's captures are collective (B3): its blocks meet
  //! at the same-process rendezvous inside a segment with a cut at an
  //! exchange, in the warm-up and the capture alike. Called on the calling
  //! thread before the blocks' run_stage (Mesh::forward): a segment that one
  //! block must (re)capture is dropped on every block, so all capture it
  void align_captures(std::vector<Variables>& vars, double dt, int stage);
  //! strict refuses a block whose stage has a non-host eager segment, naming
  //! it; called before any launch (Mesh::forward: on the calling thread)
  void check_strict(int block, Variables const& vars, int stage) const;

  //! why this block cannot be captured; empty: it can. Its own reasons
  //! (configuration and variables), then, in a multi-block executor, "a
  //! Mesh whose blocks disagree" when another block refuses capture or runs
  //! another execution mode: the whole Mesh then runs eager (B3 (d)/(e))
  std::vector<std::string> capture_reasons(int block,
                                           Variables const& vars) const;
  //! the blocks' variable-dependent reasons, which capture_reasons reads for
  //! the other blocks of a Mesh; on the calling thread, before the blocks
  //! run (Mesh::initialize, Mesh::forward)
  void agree(std::vector<Variables> const& vars);
  //! V3 X7: this block's row of the capability matrix (G7)
  Capability capability(int block, Variables const& vars) const;

  void invalidate();           // drops every graph
  void invalidate(int block);  // drops this block's graphs
  void bump_config_generation() { ++_config_generation; }
  std::map<std::string, int64_t> report() const;

 private:
  struct Captured;
  using Slot = std::tuple<int, int, int>;  // (block, stage, segment)

  //! (re)builds the block's schedule; an op's mode follows the block's
  //! options (an outflow boundary makes the boundaries op eager)
  void build_schedule(int block) const;
  //! whether the block has an outflow (radiating) boundary function
  bool outflow_boundary(int block) const;
  //! whether the block has user TorchScript stage forcings
  bool user_forcing(int block) const;
  //! V3 C2: a chemistry step is registered on a bare block (its slot is
  //! eager); a Mesh's block runs its step in the Mesh's global phase (C5)
  bool chemistry_slot(int block) const;
  //! what an op's mode follows: outflow (1), user forcings (2), chemistry (4)
  int schedule_signature(int block) const;
  //! the ops of a segment (all segments: -1), in order, eagerly
  void run_eager(int block, Variables& vars, StepDt dt, int stage,
                 int segment = -1);
  GraphKey make_key(int block, int segment, Variables const& vars, double dt,
                    int stage) const;
  //! the block's device dt, which every graph of the block reads, set to dt
  //! in stream order (V3 D); one per block, so blocks replaying on their own
  //! streams (a Mesh's worker pool) never write one another's
  StepDt device_dt(int block, Variables const& vars, double dt);
  //! a block's reasons from its configuration only
  std::vector<std::string> config_reasons(int block) const;
  //! reasons from a block's variables (the state's device, a solid mask)
  static std::vector<std::string> vars_reasons(Variables const& vars);
  //! the EOS's kintera ThermoY (hydro.eos.thermo), or null
  std::shared_ptr<kintera::ThermoYImpl> thermo_of(int block) const;
  //! the save list: the state, the registered storage and hydro_w
  std::map<std::string, torch::Tensor> save_list(int block,
                                                 Variables& vars) const;
  //! the stage's segments: the host ones (a bare block's exchange) and the
  //! eager ones eagerly, the rest captured or replayed
  void run_segments(int block, Variables& vars, double dt, int stage);
  //! the block's schedule, rebuilt if its signature moved (its graphs are
  //! then dropped)
  void refresh(int block);
  std::shared_ptr<Captured> graph_at(Slot const& slot) const;
  void capture(int block, int segment, Variables& vars, double dt, int stage);
  void replay(int block, int segment, Variables& vars, double dt, int stage);

  std::vector<MeshBlockImpl*> _blocks;
  //! one per block, rebuilt when its signature changes (Python can reset a
  //! block's boundary functions or stage forcings after construction); a
  //! rebuild drops the block's graphs at its next stage
  mutable std::vector<StageSchedule> _schedules;
  mutable std::vector<int> _schedule_sig;
  mutable std::vector<char> _schedule_rebuilt;
  std::map<Slot, std::shared_ptr<Captured>> _graphs;  // under _graphs_mutex
  mutable std::mutex _graphs_mutex;
  //! a lone block's captures one at a time; a multi-block executor's are
  //! collective and concurrent (align_captures)
  std::mutex _capture_mutex;
  //! V3 D: one persistent 0-dim FP64 dt per block on its device, shared by
  //! every graph of that block (each block's worker writes only its own)
  std::vector<torch::Tensor> _dt;
  //! per block, its variables' reasons as of the last agree()
  std::vector<std::vector<std::string>> _vars_reasons;
  //! one CUDA graph memory pool per block, shared by its segments' graphs
  //! (a block's graphs replay one after another on one stream); {0, 0}
  //! until its first capture
  std::vector<std::pair<uint64_t, uint64_t>> _pools;
  //! per block, its own capture stream (a cudaStream_t), never destroyed
  std::vector<void*> _capture_streams;
  int _config_generation = 0;
  //! run_stage runs on the block worker pool
  std::atomic<int64_t> _captures{0}, _replays{0}, _eager{0};
  //! the eager reasons already logged, per block (log_eager)
  std::vector<std::set<std::string>> _logged;
  std::mutex _logged_mutex;
  //! log each of a non-strict block's eager reasons once
  void log_eager(int block, std::vector<std::string> const& why);
  //! host-exchange and global segments run under cuda-graph (boundaries)
  std::atomic<int64_t> _host{0};
};

}  // namespace snap
