// C/C++
#include <c10/core/DeviceGuard.h>
#include <configure.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/chemistry/chemistry_source.hpp>
#include <snap/diagnostics/budget.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/utils/log.hpp>

#include "stage_executor.hpp"

namespace snap {

namespace {
MeshBlockOptions clone_block_options(MeshBlockOptions const& src) {
  auto dst = std::make_shared<MeshBlockOptionsImpl>(*src);
  auto layout = std::make_shared<LayoutOptionsImpl>(*src->layout());
  dst->layout(layout);
  if (src->coord() != nullptr) {
    dst->coord(src->coord()->clone());
  }
  return dst;
}

//! Every block of a Mesh exchanges the same variables; a missing or extra
//! one would leave the same-process copy reading past a block's buffers.
void check_same_variables(MeshVariables const& vars, char const* caller) {
  auto keys = [](Variables const& v) {
    std::vector<std::string> k;
    for (auto const& [name, _] : v) k.push_back(name);
    return k;  // sorted: a std::map's order
  };
  for (size_t i = 1; i < vars.size(); ++i) {
    auto k0 = keys(vars[0]), ki = keys(vars[i]);
    if (k0 == ki) continue;
    for (auto const& name : k0)
      TORCH_CHECK(vars[i].count(name), caller, ": block ", i,
                  " has no variable '", name,
                  "' (block 0 has it); every block must pass the same "
                  "variables");
    for (auto const& name : ki)
      TORCH_CHECK(vars[0].count(name), caller, ": block ", i, " has variable '",
                  name,
                  "' that block 0 does not; every block must pass the same "
                  "variables");
  }
}

}  // namespace

class MeshImpl::BlockWorkerPool {
 public:
  BlockWorkerPool(size_t count, torch::Device device)
      : device_(std::move(device)), remaining_(count) {
#ifdef USE_CUDA
    if (device_.is_cuda()) {
      c10::cuda::CUDAGuard device_guard(device_.index());
      streams_.reserve(count);
      done_events_.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        streams_.push_back(c10::cuda::getStreamFromPool(
            /*isHighPriority=*/false, device_.index()));
        done_events_.push_back(std::make_unique<at::cuda::CUDAEvent>());
      }
      caller_ready_event_ = std::make_unique<at::cuda::CUDAEvent>();
    }
#endif

    workers_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      workers_.emplace_back([this, i]() { run(i); });
    }
  }

  ~BlockWorkerPool() { stop(); }

  //! the pool's CUDA streams: one per block on a CUDA device, else none
  std::vector<void*> streams() const {
    std::vector<void*> raw;
#ifdef USE_CUDA
    for (auto const& s : streams_) raw.push_back(s.stream());
#endif
    return raw;
  }

  void submit(std::function<void(size_t)> func) {
#ifdef USE_CUDA
    std::optional<c10::cuda::CUDAStream> caller_stream;
    if (device_.is_cuda()) {
      caller_stream.emplace(c10::cuda::getCurrentCUDAStream(device_.index()));
      caller_ready_event_->record(*caller_stream);
    }
#endif

    {
      std::lock_guard<std::mutex> lock(mutex_);
      TORCH_CHECK(!stopping_, "Mesh block worker pool is stopping");
      func_ = std::move(func);
      error_ = nullptr;
      remaining_ = workers_.size();
      generation_ += 1;
    }
    cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&]() { return remaining_ == 0; });

#ifdef USE_CUDA
    if (device_.is_cuda()) {
      for (auto const& event : done_events_) {
        event->block(*caller_stream);
      }
    }
#endif

    if (error_ != nullptr) {
      auto error = error_;
      error_ = nullptr;
      std::rethrow_exception(error);
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

 private:
  void run(size_t index) {
    c10::OptionalDeviceGuard device_guard(device_);

#ifdef USE_CUDA
    if (device_.is_cuda()) {
      c10::cuda::CUDAStreamGuard stream_guard(streams_.at(index));
      run_loop(index);
      return;
    }
#endif

    run_loop(index);
  }

  void run_loop(size_t index) {
    size_t seen_generation = 0;
    while (true) {
      std::function<void(size_t)> func;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock,
                 [&]() { return stopping_ || generation_ != seen_generation; });
        if (stopping_) return;
        seen_generation = generation_;
        func = func_;
      }

#ifdef USE_CUDA
      if (device_.is_cuda()) {
        caller_ready_event_->block(streams_.at(index));
      }
#endif

      try {
        func(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_ == nullptr) {
          error_ = std::current_exception();
        }
      }

#ifdef USE_CUDA
      if (device_.is_cuda()) {
        done_events_.at(index)->record(streams_.at(index));
      }
#endif

      {
        std::lock_guard<std::mutex> lock(mutex_);
        remaining_ -= 1;
        if (remaining_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  torch::Device device_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::function<void(size_t)> func_;
  std::exception_ptr error_;
  size_t generation_ = 0;
  size_t remaining_ = 0;
  bool stopping_ = false;

#ifdef USE_CUDA
  std::vector<c10::cuda::CUDAStream> streams_;
  std::vector<std::unique_ptr<at::cuda::CUDAEvent>> done_events_;
  std::unique_ptr<at::cuda::CUDAEvent> caller_ready_event_;
#endif
};

MeshOptions MeshOptionsImpl::from_yaml(std::string input_file, bool verbose) {
  auto block = MeshBlockOptionsImpl::from_yaml(input_file, verbose);
  auto options = MeshOptionsImpl::create();
  options->block(block);

  auto config = YAML::LoadFile(input_file);
  options->blocks_per_process(
      config["distribute"]["blocks_per_process"].as<int>(
          options->blocks_per_process()));
  return options;
}

MeshImpl::MeshImpl(MeshOptions const& options_) : options(options_) { reset(); }

MeshImpl::~MeshImpl() = default;

void MeshImpl::reset() {
  workers_.reset();
  executor_.reset();
  blocks.clear();
  TORCH_CHECK(options->block() != nullptr, "Mesh requires MeshBlockOptions");

  auto base = options->block();
  auto base_layout = base->layout();
  TORCH_CHECK(base_layout != nullptr, "Mesh requires LayoutOptions");
  TORCH_CHECK(options->blocks_per_process() > 0,
              "blocks_per_process must be positive");

  base_layout->blocks_per_process(options->blocks_per_process());
  base_layout->world_size(base_layout->process_world_size() *
                          options->blocks_per_process());

  // this Mesh's identity in the same-process exchange state (S-L1): a new
  // one per reset, so a rebuilt or second Mesh never takes these blocks' place
  static std::atomic<int64_t> next_mesh_id{1};
  int64_t mesh_id = next_mesh_id++;

  for (int i = 0; i < options->blocks_per_process(); ++i) {
    auto block_opts = clone_block_options(base);
    auto layout = block_opts->layout();
    layout->mesh_id(mesh_id);
    layout->blocks_per_process(options->blocks_per_process());
    layout->world_size(layout->process_world_size() *
                       options->blocks_per_process());
    layout->rank(layout->global_block_rank(layout->process_rank(), i));
    if (block_opts->coord() != nullptr) {
      block_opts->coord()->repartition(layout);
    }
    auto block =
        register_module("block" + std::to_string(i), MeshBlock(block_opts));
    blocks.push_back(block);
  }

  // one stage executor for all the blocks (lane B)
  std::vector<MeshBlockImpl*> members;
  for (auto& b : blocks) members.push_back(b.get());
  executor_ = std::make_shared<StageExecutor>(members);
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i)
    blocks[i]->use_executor(executor_, i);

  if (blocks.size() > 1) {
    workers_ = std::make_shared<BlockWorkerPool>(
        blocks.size(), torch::Device(blocks.front()->options->device_str()));
  }
}

std::map<std::string, int64_t> MeshImpl::execution_report() const {
  return executor_->report();
}

std::vector<void*> MeshImpl::worker_streams() const {
  return workers_ ? workers_->streams() : std::vector<void*>{};
}

void MeshImpl::run_block_jobs(std::function<void(size_t)> func) {
  if (blocks.size() <= 1) {
    if (!blocks.empty()) {
      c10::OptionalDeviceGuard device_guard(
          torch::Device(blocks.front()->options->device_str()));
      func(0);
    }
    return;
  }

  TORCH_CHECK(workers_ != nullptr,
              "Mesh block worker pool is not initialized for multi-block mesh");
  workers_->submit(std::move(func));
}

double MeshImpl::initialize(MeshVariables& vars, char const* restart_file) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::initialize expects one Variables map per local MeshBlock");

  if (restart_file) {
    double current_time = 0.;
    for (int i = 0; i < blocks.size(); ++i) {
      auto block_time = blocks[i]->initialize(vars[i], restart_file);
      if (i == 0) {
        current_time = block_time;
      }
    }
    executor_->agree(vars);  // B3 (d)/(e): capability rows see the Mesh
    return current_time;
  }

  auto layout = blocks.front()->get_layout();
  if (layout->has_process_group()) {
    layout->comm->barrier();
  }
  SignalHandler::GetInstance();

  run_block_jobs([&](size_t i) { blocks[i]->initialize_under_mesh(vars[i]); });
  executor_->agree(vars);  // B3 (d)/(e): capability rows see the Mesh

  return 0.;
}

double MeshImpl::max_time_step(MeshVariables const& vars) {
  TORCH_CHECK(
      vars.size() == blocks.size(),
      "Mesh::max_time_step expects one Variables map per local MeshBlock");

  double dt_local = 1.e99;
  for (int i = 0; i < blocks.size(); ++i) {
    dt_local = std::min(dt_local, blocks[i]->local_max_time_step(vars[i]));
  }

  return blocks.front()->global_time_step(dt_local);
}

void MeshImpl::forward(MeshVariables& vars, double dt, int stage) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::forward expects one Variables map per local MeshBlock");
  // V3 B2: the blocks' stages run as graphs on the worker pool; strict
  // refuses a block that cannot capture here, before any launch. B3 (d)/(e):
  // a block that disagrees with another (one refuses, or their modes differ)
  // makes every block eager, so the blocks first share their variables'
  // reasons
  executor_->agree(vars);
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i)
    executor_->check_strict(i, vars[i], stage);
  if (stage == 0)
    for (auto& b : blocks) b->step_dt = dt;

  for (auto& b : blocks) b->defer_chemistry = true;
  // B3: a segment one block must (re)capture is captured by all of them (its
  // exchanges meet at the same-process rendezvous)
  executor_->align_captures(vars, dt, stage);
  run_block_jobs(
      [&](size_t i) { executor_->run_local(i, vars[i], dt, stage); });
  // conserve-axial-momentum-closure: R and I summed over the local blocks,
  // one reduction, then each block's solid-body increment
  auto root = blocks.front();
  // the global phase (X4): the blocks' device partials, one reduction
  if (root->phydro->paam->closes()) {
    std::vector<torch::Tensor> partials;
    for (auto& b : blocks) partials.push_back(b->axial_closure_sums());
    auto global = global_phase_sum(partials, *root->get_layout());
    run_block_jobs([&](size_t i) {
      blocks[i]->apply_axial_closure(vars[i], dt, stage, global);
      auto& v = vars[i];
      blocks[i]->apply_boundaries(
          v, v.at("hydro_u"),
          v.count("scalar_s") ? v.at("scalar_s") : torch::Tensor());
    });
  }
  // S-K4 (K.rst 3.5): the chemistry step, last in the global phase, before
  // the final exchange. The blocks run it one after another, not on the
  // worker pool: kintera's LogSVPFunc keeps one process-global _op that
  // Evaporation::forward reassigns on every call (kintera log_svp.cpp:52-58,
  // 2.6.11), so two blocks' kinetics on two threads race on that shared_ptr
  // (a double free; K.rst 3.6's no-shared-state assumption does not hold).
  // Sequential until kintera makes it per instance; the results are the
  // same, each block's step reads only its own state.
  if (stage == static_cast<int>(root->pintg->stages.size()) - 1)
    for (size_t i = 0; i < blocks.size(); ++i) {
      auto chem = blocks[i]->chemistry();
      if (!chem) continue;
      chem->forward(*blocks[i], vars[i], dt);
      auto& v = vars[i];
      blocks[i]->apply_boundaries(
          v, v.at("hydro_u"),
          v.count("scalar_s") ? v.at("scalar_s") : torch::Tensor());
    }
  run_block_jobs([&](size_t i) { blocks[i]->exchange_ghost_zones(vars[i]); });
}

void MeshImpl::set_chemistry(kintera::KineticsOptionsImpl reactions) {
  for (auto& b : blocks) b->set_chemistry(reactions);  // one after another
}

void MeshImpl::set_user_stage_forcings(
    std::vector<std::string> const& filenames) {
  TORCH_CHECK(!blocks.empty(),
              "Mesh::set_user_stage_forcings requires at least one block");
  blocks.front()->set_user_stage_forcings(filenames);
  for (size_t i = 1; i < blocks.size(); ++i) {
    blocks[i]->user_stage_forcings = blocks.front()->user_stage_forcings;
  }
}

void MeshImpl::exchange(MeshVariables& vars, SyncOptions const& opts) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::exchange expects one Variables map per local MeshBlock");

  check_same_variables(vars, "Mesh::exchange");
  run_block_jobs([&](size_t i) { blocks[i]->exchange(vars[i], opts); });
}

void MeshImpl::exchange_ghost_zones(MeshVariables& vars, int type) {
  TORCH_CHECK(type == kPrimitive || type == kConserved || type == kScalar,
              "Mesh::exchange_ghost_zones received invalid variable type");

  SyncOptions opts;
  opts.interpolate(true).type(type);
  exchange(vars, opts);
}

void MeshImpl::make_outputs(MeshVariables const& vars, double current_time,
                            bool final_write) {
  TORCH_CHECK(
      vars.size() == blocks.size(),
      "Mesh::make_outputs expects one Variables map per local MeshBlock");

  for (int i = 0; i < blocks.size(); ++i) {
    blocks[i]->make_outputs(vars[i], current_time, final_write);
  }
}

void MeshImpl::print_cycle_info(MeshVariables const& vars, double time,
                                double dt) const {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::print_cycle_info expects one Variables map per local "
              "MeshBlock");
  std::vector<std::pair<MeshBlockImpl const*, Variables const*>> local;
  for (size_t i = 0; i < blocks.size(); ++i)
    local.emplace_back(blocks[i].get(), &vars[i]);
  print_cycle_line(local, time, dt,
                   std::numeric_limits<double>::max_digits10 - 3, " energy=");
}

int MeshImpl::check_redo(MeshVariables& vars) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::check_redo expects one Variables map per local MeshBlock");

  auto sig = snap::SignalHandler::GetInstance();
  if (!blocks.empty() && sig->CheckSignalFlags(blocks.front().get())) {
    return -1;
  }

  std::vector<MeshBlockImpl*> impls;
  std::vector<Variables const*> local;
  for (size_t i = 0; i < blocks.size(); ++i) {
    impls.push_back(blocks[i].get());
    local.push_back(&vars[i]);
  }
  int causes = MeshBlockImpl::decide_retry(impls, local);

  int out = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    int err = blocks[i]->apply_redo(vars[i], causes);
    if (err < 0) {
      // A globally rejected VIC step must restore every local block even
      // when the first block has already exhausted its retry budget.
      if (!(causes & 32)) return -1;
      out = -1;
    } else if (out >= 0) {
      out = std::max(out, err);
    }
  }
  return out;
}

void MeshImpl::set_cycle(int cycle) {
  for (auto& block : blocks) {
    block->cycle = cycle;
  }
}

int MeshImpl::finalize(MeshVariables const& vars, double time) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::finalize expects one Variables map per local MeshBlock");

  if (blocks.size() == 1) {
    return blocks.front()->finalize(vars.front(), time);
  }

  make_outputs(vars, time, /*final_write=*/true);
  std::vector<MeshBlockImpl*> impls;
  for (auto& b : blocks) impls.push_back(b.get());
  return MeshBlockImpl::finish_run(impls, time);
}

}  // namespace snap
