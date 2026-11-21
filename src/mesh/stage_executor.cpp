// base
#include <configure.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include "meshblock.hpp"
#include "stage_executor.hpp"

namespace snap {

torch::Tensor global_phase_sum(std::vector<torch::Tensor> const& partials,
                               LayoutImpl const& layout) {
  TORCH_CHECK(!partials.empty(), "global_phase_sum: no partial sums");
  auto sums = partials.front();
  if (partials.size() > 1) {
    sums = torch::zeros_like(partials.front());
    for (auto const& p : partials) sums += p;
  }
  if (layout.has_process_group()) {  // the host segment: one allreduce
    std::vector<at::Tensor> reduce = {sums.cpu()};
    layout.comm->allreduce(reduce, c10d::ReduceOp::SUM);
    sums = reduce[0].to(sums.device());
  }
  return sums;
}

bool GraphKey::operator==(GraphKey const& other) const {
  return device_index == other.device_index && names == other.names &&
         layout == other.layout && store_generation == other.store_generation &&
         config_generation == other.config_generation && stage == other.stage &&
         dt == other.dt && thermo_epoch == other.thermo_epoch;
}

struct StageExecutor::Captured {
  GraphKey key;
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAGraph> graph;
#endif
  //! per vars key, an alias of the storage the graph reads (inputs) and
  //! writes (outputs): a handle rebound by set_data is copied in before a
  //! replay and set_ to the output after it, as an eager stage leaves it
  std::map<std::string, torch::Tensor> inputs, outputs;
};

StageExecutor::StageExecutor(MeshBlockImpl* block) : _block(block) {}
StageExecutor::~StageExecutor() = default;

void StageExecutor::run_eager(Variables& vars, double dt, int stage) {
  _block->exchange_ghost_zones(vars);
  _block->advance_local(vars, dt, stage);
}

std::vector<std::string> StageExecutor::capture_reasons(
    Variables const& vars) const {
  // the prototype tests/straka_cuda_graph.py:16-31 at d6a3c00c: dry Straka
  std::vector<std::string> why;
  auto const& op = _block->options;
  auto const& hydro = op->hydro();
#ifndef USE_CUDA
  why.push_back("snapy was built without CUDA");
#endif
  auto it = vars.find("hydro_u");
  if (it == vars.end() || !it->second.defined() || !it->second.is_cuda())
    why.push_back("the state is not on a CUDA device");
  if (op->coord()->type() != "cartesian")
    why.push_back("geometry is " + op->coord()->type() + ", not cartesian");
  if (hydro->eos()->limiter()) why.push_back("the EOS limiter is on");
  if (hydro->eos()->type() != "ideal-moist")
    why.push_back("equation of state is " + hydro->eos()->type() +
                  ", not ideal-moist");
  if (hydro->riemann()->type() != "hllc")
    why.push_back("riemann solver is " + hydro->riemann()->type() +
                  ", not hllc");
  if (hydro->icorr()) why.push_back("implicit-scheme is not 0");
  if (op->intg()->type() != "rk3")
    why.push_back("integration type is " + op->intg()->type() + ", not rk3");
  auto const& lo = _block->_playout->options;
  if (lo->px() * lo->py() * lo->pz() != 1 ||
      _block->_playout->has_process_group())
    why.push_back("more than one block");
  if (!hydro->grav() || hydro->coriolis() || hydro->diffusion() ||
      hydro->bodyHeat() || hydro->botHeat() || hydro->topCool() ||
      hydro->relaxBotComp() || hydro->relaxBotTemp() || hydro->relaxBotVelo() ||
      hydro->topSpongeLyr() || hydro->botSpongeLyr() || hydro->plumeForcing() ||
      !_block->user_stage_forcings.empty())
    why.push_back("forcing is not const-gravity alone");
  if (vars.count("solid")) why.push_back("a solid mask is set");
  return why;
}

Capability StageExecutor::capability(Variables const& vars) const {
  Capability c;
  bool fp32 = _block->policy.fp32();
  c.eager_fp64 = !fp32;  // every case runs eager in its precision
  c.eager_fp32 = fp32;
  c.eager_reasons = capture_reasons(vars);
  bool graph = c.eager_reasons.empty();
  c.graph_fp64 = graph && !fp32;
  c.graph_fp32 = graph && fp32;
  return c;
}

GraphKey StageExecutor::make_key(Variables const& vars, double dt,
                                 int stage) const {
  GraphKey key;
  key.device_index = vars.at("hydro_u").device().index();
  for (auto const& [name, t] : vars) {
    if (!t.defined()) continue;
    key.names.push_back(name);
    key.layout.push_back(static_cast<int64_t>(t.scalar_type()));
    key.layout.push_back(t.dim());
    for (auto n : t.sizes()) key.layout.push_back(n);
    for (auto n : t.strides()) key.layout.push_back(n);
  }
  key.store_generation = _block->_state.generation();
  key.config_generation = _config_generation;
  key.stage = stage;
  key.dt = dt;
  if (auto pthermo = thermo_of())
    key.thermo_epoch = kintera::alloc_epoch(*pthermo);
  return key;
}

std::map<std::string, torch::Tensor> StageExecutor::save_list(
    Variables& vars) const {
  std::map<std::string, torch::Tensor> list;
  for (auto name : {"hydro_u", "scalar_s", "scalar_r", "hydro_w"}) {
    auto it = vars.find(name);
    if (it != vars.end() && it->second.defined())
      list[std::string("vars.") + name] = it->second;
  }
  auto& state = _block->_state;
  list["u0"] = state.u0();
  if (state.s0().numel() > 0) list["s0"] = state.s0();
  for (auto const& [name, t] : state.registered())
    if (t.defined()) list["storage." + name] = t;
  // K2 (K.rst 3.2): the ThermoY failure counter, so a warm-up leaves no count
  if (auto pthermo = thermo_of()) {
    if (pthermo->failures().defined())
      list["thermo.failures"] = pthermo->failures();
    // K4 (K.rst 3.4): the active-set buffers too, so a warm-up leaves them
    if (pthermo->nactive.defined()) {
      list["thermo.reaction_set"] = pthermo->reaction_set;
      list["thermo.nactive"] = pthermo->nactive;
    }
  }
  return list;
}

std::shared_ptr<kintera::ThermoYImpl> StageExecutor::thermo_of() const {
  return _block->thermo();
}

void StageExecutor::run_stage(Variables& vars, double dt, int stage) {
  auto const& op = _block->options;
  if (op->execution() != "cuda-graph") {
    ++_eager;
    run_eager(vars, dt, stage);
    return;
  }

  auto why = capture_reasons(vars);
  if (!why.empty()) {
    std::string list;
    for (auto const& w : why) list += "\n  - " + w;
    TORCH_CHECK(!op->execution_strict(),
                "integration/execution-strict: this case cannot run as a "
                "CUDA graph:",
                list);
    ++_eager;
    run_eager(vars, dt, stage);
    return;
  }

#ifdef USE_CUDA
  _block->_state.adopt(vars);  // a rebound tensor bumps the generation
  _block->register_flux_storage();
  auto it = _graphs.find(stage);
  if (it == _graphs.end() || it->second->key != make_key(vars, dt, stage))
    capture(vars, dt, stage);
  replay(vars, stage);
#endif
}

void StageExecutor::capture(Variables& vars, double dt, int stage) {
#ifdef USE_CUDA
  _graphs.erase(stage);
  auto device = vars.at("hydro_u").device().index();
  auto current = at::cuda::getCurrentCUDAStream(device);
  auto side = at::cuda::getStreamFromPool(false, device);

  // K.rst 3.5, 7: the thermo is prepared first, under the sync-exempt mode
  // the warm-up uses (its table upload is a synchronizing copy), so the save
  // list below already holds the counter and the active-set buffers on the
  // first capture; from then on no kintera table is built inside the
  // warm-up or the capture (a missing one throws)
  {
    auto& warn = c10::cuda::warning_state();
    auto mode = warn.get_sync_debug_mode();
    warn.set_sync_debug_mode(c10::cuda::SyncDebugMode::L_DISABLED);
    try {
      _block->prepare_thermo(vars.at("hydro_u"));
    } catch (...) {
      warn.set_sync_debug_mode(mode);
      throw;
    }
    warn.set_sync_debug_mode(mode);
  }
  kintera::NoTableBuild no_table_build;
  auto handles = save_list(vars);
  std::map<std::string, torch::Tensor> saved;
  for (auto const& [name, t] : handles) saved[name] = t.clone();
  auto restore = [&]() {
    for (auto const& [name, t] : handles) t.copy_(saved.at(name));
  };

  // warm-up on a side stream with the step's dt, then the save list back;
  // stream order only. Warm-up is eager and builds what a first call builds
  // (a block's first stage syncs once), so a sync-debug mode the caller set
  // applies to the capture and the replays, not here.
  at::cuda::CUDAEvent ready;
  ready.record(current);
  ready.block(side);
  {
    auto& warn = c10::cuda::warning_state();
    auto mode = warn.get_sync_debug_mode();
    warn.set_sync_debug_mode(c10::cuda::SyncDebugMode::L_DISABLED);
    c10::cuda::CUDAStreamGuard guard(side);
    try {
      for (int i = 0; i < 3; ++i) run_eager(vars, dt, stage);
    } catch (...) {
      warn.set_sync_debug_mode(mode);
      throw;
    }
    warn.set_sync_debug_mode(mode);
  }
  at::cuda::CUDAEvent warm;
  warm.record(side);
  warm.block(current);
  restore();
  _block->register_flux_storage();

  auto entry = std::make_shared<Captured>();
  for (auto const& [name, t] : vars)
    if (t.defined()) entry->inputs[name] = t.detach();

  at::cuda::CUDAEvent restored;
  restored.record(current);
  restored.block(side);
  entry->graph = std::make_unique<at::cuda::CUDAGraph>();
  {
    c10::cuda::CUDAStreamGuard guard(side);
    entry->graph->capture_begin();
    run_eager(vars, dt, stage);
    entry->graph->capture_end();
  }
  at::cuda::CUDAEvent captured;
  captured.record(side);
  captured.block(current);

  // capture runs nothing, but a handle set_data'd during it points at the
  // graph's output, which is not yet written
  for (auto const& [name, t] : vars)
    if (t.defined()) entry->outputs[name] = t.detach();
  restore();
  for (auto const& [name, t] : entry->inputs) {
    auto h = handles.find("vars." + name);
    if (h != handles.end() && t.data_ptr() != h->second.data_ptr())
      t.copy_(saved.at(h->first));
  }

  entry->key = make_key(vars, dt, stage);
  _graphs[stage] = entry;
  ++_captures;
#else
  TORCH_CHECK(false, "StageExecutor: snapy was built without CUDA");
#endif
}

void StageExecutor::replay(Variables& vars, int stage) {
#ifdef USE_CUDA
  auto& entry = *_graphs.at(stage);
  for (auto const& [name, in] : entry.inputs) {
    auto const& t = vars.at(name);
    if (t.data_ptr() != in.data_ptr()) in.copy_(t);
  }
  _block->phydro->rk_stage = stage;  // advance_local's host side
  entry.graph->replay();
  for (auto const& [name, out] : entry.outputs) {
    auto& t = vars.at(name);
    if (t.data_ptr() != out.data_ptr()) t.set_(out);
  }
  ++_replays;
#endif
}

void StageExecutor::invalidate() { _graphs.clear(); }

std::map<std::string, int64_t> StageExecutor::report() const {
  return {{"captures", _captures},
          {"replays", _replays},
          {"eager", _eager},
          {"graphs", static_cast<int64_t>(_graphs.size())}};
}

}  // namespace snap
