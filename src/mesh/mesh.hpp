#pragma once

// C/C++
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// torch
#include <torch/nn/cloneable.h>

// snap
#include <snap/mesh/meshblock.hpp>

// arg
#include <snap/add_arg.h>

namespace kintera {
struct KineticsOptionsImpl;
}

namespace snap {

using MeshVariables = std::vector<Variables>;

struct MeshOptionsImpl {
  static std::shared_ptr<MeshOptionsImpl> create() {
    return std::make_shared<MeshOptionsImpl>();
  }
  static std::shared_ptr<MeshOptionsImpl> from_yaml(std::string input_file,
                                                    bool verbose = false);

  void report(std::ostream& os) const {
    os << "-- mesh options --\n";
    if (block() && block()->layout())
      os << "* blocks_per_process = " << blocks_per_process() << "\n";
  }

  std::string device_str() const {
    return block() ? block()->device_str() : "";
  }

  ADD_ARG(MeshBlockOptions, block) = nullptr;

 public:
  //! the layout's blocks_per_process (distribute/blocks_per_process), its one
  //! home: read and written through block()->layout(), so block() comes first
  int const& blocks_per_process() const {
    return layout_().blocks_per_process();
  }
  MeshOptionsImpl& blocks_per_process(int const& n) {
    layout_().blocks_per_process(n);
    return *this;
  }

 private:
  LayoutOptionsImpl& layout_() const {
    TORCH_CHECK(block() && block()->layout(),
                "MeshOptions::blocks_per_process is the layout's: set block() "
                "(with its layout) first");
    return *block()->layout();
  }
};
using MeshOptions = std::shared_ptr<MeshOptionsImpl>;

class MeshImpl : public torch::nn::Cloneable<MeshImpl> {
 public:
  MeshOptions options;
  std::vector<MeshBlock> blocks;

  MeshImpl() : options(MeshOptionsImpl::create()) {}
  explicit MeshImpl(MeshOptions const& options_);
  ~MeshImpl() override;
  void reset() override;

  double initialize(MeshVariables& vars, char const* restart_file = nullptr);
  double max_time_step(MeshVariables const& vars);
  void forward(MeshVariables& vars, double dt, int stage);
  //! V3 S-K4: one chemistry adapter per block (Mesh::blocks order, built one
  //! after another on the calling thread), run in the last stage's global
  //! phase before the final exchange
  void set_chemistry(kintera::KineticsOptionsImpl reactions);
  void exchange(MeshVariables& vars, SyncOptions const& opts);
  void exchange_ghost_zones(MeshVariables& vars, int type = kConserved);
  void set_user_stage_forcings(std::vector<std::string> const& filenames);
  void make_outputs(MeshVariables const& vars, double current_time,
                    bool final_write = false);
  void print_cycle_info(MeshVariables const& vars, double time,
                        double dt) const;
  int check_redo(MeshVariables& vars);
  void set_cycle(int cycle);
  int finalize(MeshVariables const& vars, double time);

  //! the block worker pool's CUDA streams (cudaStream_t), one per block;
  //! empty without a CUDA pool
  std::vector<void*> worker_streams() const;

  //! the stage executor of all this Mesh's blocks (lane B), shared with them
  std::shared_ptr<StageExecutor> executor() const { return executor_; }
  //! its counters: captures, replays, eager block-stages, graphs
  std::map<std::string, int64_t> execution_report() const;

 private:
  class BlockWorkerPool;

  void run_block_jobs(std::function<void(size_t)> func);

  std::shared_ptr<BlockWorkerPool> workers_;
  std::shared_ptr<StageExecutor> executor_;
};

TORCH_MODULE(Mesh);

}  // namespace snap

#undef ADD_ARG
