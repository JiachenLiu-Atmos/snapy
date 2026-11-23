// C/C++
#include <algorithm>

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
         config_generation == other.config_generation && block == other.block &&
         stage == other.stage && segment == other.segment && dt == other.dt &&
         thermo_epoch == other.thermo_epoch;
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

StageExecutor::StageExecutor(MeshBlockImpl* block)
    : StageExecutor(std::vector<MeshBlockImpl*>{block}) {}
StageExecutor::StageExecutor(std::vector<MeshBlockImpl*> blocks)
    : _blocks(std::move(blocks)) {
  TORCH_CHECK(!_blocks.empty(), "StageExecutor: no blocks");
  for (auto* b : _blocks) TORCH_CHECK(b, "StageExecutor: a null block");
  _dt.resize(_blocks.size());
  _schedules.resize(_blocks.size());
  _schedule_outflow.assign(_blocks.size(), -1);
  _schedule_rebuilt.assign(_blocks.size(), 0);
  _pools.assign(_blocks.size(), {0, 0});
  for (int b = 0; b < num_blocks(); ++b) build_schedule(b);
}
StageExecutor::~StageExecutor() = default;

bool StageExecutor::outflow_boundary(int block) const {
  for (auto const& fn : block_at(block)->options->bfuncs())
    if (fn && is_outflow(fn)) return true;
  return false;
}

void StageExecutor::build_schedule(int block) const {
  // A3a: the stage-entry exchange is its own host-exchange segment (eager by
  // design, a segment boundary), then advance_local in its slots as one graph
  StageSchedule s;
  Op exchange;
  exchange.name = "exchange";
  exchange.eager_reason = "the stage-entry ghost exchange";
  exchange.run = [this, block](Variables& vars, StageContext const&) {
    block_at(block)->exchange_ghost_zones(vars);
  };
  s.add(std::move(exchange), SegmentKind::host_exchange);

  Op stage;
  stage.name = "advance";
  // the RK combine's w2 dt and FluxRegister::end_stage(w2 dt) read the
  // block's device dt (V3 D: a StepDt), not the host dt, so this segment keeps
  // no dt in its key; a host read under capture throws (StepDt::value)
  stage.reads_host_dt = false;
  // the thermo's tables and buffers, and the register monitor's amounts on
  // the register's device, before the save list (K.rst 3.5; 86a1e571)
  stage.prepare = [this, block](torch::Device device, torch::Dtype dtype,
                                at::IntArrayRef shape) {
    auto* pmb = block_at(block);
    pmb->prepare_thermo(device, dtype, shape);
    pmb->place_flux_storage();
    pmb->register_flux_storage();
  };
  stage.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_advance(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(stage));

  // the user stage forcings (A3b), between the tendencies and the combine
  Op forcing;
  forcing.name = "forcing";
  // a user forcing is called with the host dt (dt.value()), but a case with
  // one is eager ("forcing is not const-gravity alone"); without one the op
  // reads no dt, so it keeps none in its key (V3 D)
  forcing.reads_host_dt = false;
  forcing.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_forcing(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(forcing));

  Op combine;
  combine.name = "combine";
  // end_stage(dt) and the RK combine's w2 dt read the device dt (V3 D)
  combine.reads_host_dt = false;
  combine.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_combine(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(combine));

  // the end-of-stage slots, in the pin's order (A3a)
  Op saturation;
  saturation.name = "saturation";
  saturation.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_saturation(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(saturation));

  Op chemistry;
  chemistry.name = "chemistry";
  // ChemistrySource::forward takes the host dt (dt.value()), but only when a
  // chemistry step is registered, and that case is eager as a whole
  // (capture_reasons), so the op never runs in a captured segment with it.
  // Without one the op reads no dt: no host-dt key, and a dt change replays
  // (V3 D). Were that refusal lifted, dt.value() on the device dt throws.
  chemistry.reads_host_dt = false;
  chemistry.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_chemistry(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(chemistry));

  // an outflow (radiating) ghost fill checks its inputs on the host (#314):
  // the boundaries op runs eagerly after the graph, the rest still captures
  bool outflow = outflow_boundary(block);
  Op boundaries;
  boundaries.name = "boundaries";
  if (outflow) {
    boundaries.capturable = false;
    boundaries.eager_reason = "an outflow boundary is set";
  }
  boundaries.run = [this, block](Variables& vars, StageContext const& ctx) {
    block_at(block)->stage_boundaries(vars, ctx.dt, ctx.stage);
  };
  s.add(std::move(boundaries));
  _schedules[block] = std::move(s);
  if (_schedule_outflow[block] >= 0) _schedule_rebuilt[block] = 1;
  _schedule_outflow[block] = outflow;
}

StageSchedule const& StageExecutor::schedule(int block) const {
  block_at(block);  // checked
  if (_schedule_outflow[block] != static_cast<int>(outflow_boundary(block)))
    build_schedule(block);
  return _schedules[block];
}

MeshBlockImpl* StageExecutor::block_at(int block) const {
  TORCH_CHECK(block >= 0 && block < num_blocks(), "StageExecutor: block ",
              block, " out of range [0, ", num_blocks(), ")");
  return _blocks[block];
}

void StageExecutor::run_eager(int block, Variables& vars, StepDt dt, int stage,
                              int segment) {
  StageContext ctx{block, stage, dt};
  auto const& s = schedule(block);
  if (segment < 0) {
    for (auto const& op : s.ops()) op.run(vars, ctx);
    return;
  }
  for (int i : s.segments().at(segment).ops) s.ops()[i].run(vars, ctx);
}

StepDt StageExecutor::device_dt(int block, Variables const& vars, double dt) {
  auto device = vars.at("hydro_u").device();
  auto& t = _dt.at(block);
  if (!t.defined() || t.device() != device) {
    invalidate(block);  // its graphs read the old buffer
    t = torch::empty({}, torch::dtype(torch::kFloat64).device(device));
  }
  t.fill_(dt);
  return StepDt::device(t);
}

void StageExecutor::run_local(int block, Variables& vars, double dt,
                              int stage) {
  ++_eager;
  block_at(block)->advance_local(vars, dt, stage);
}

std::vector<std::string> StageExecutor::capture_reasons(
    int block, Variables const& vars) const {
  // the prototype tests/straka_cuda_graph.py:16-31 at d6a3c00c: dry Straka
  std::vector<std::string> why;
  auto* pmb = block_at(block);
  auto const& op = pmb->options;
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
  auto const& lo = pmb->_playout->options;
  if (lo->nblocks() != 1 || pmb->_playout->has_process_group())
    why.push_back("more than one block");
  if (!hydro->grav() || hydro->coriolis() || hydro->diffusion() ||
      hydro->bodyHeat() || hydro->botHeat() || hydro->topCool() ||
      hydro->relaxBotComp() || hydro->relaxBotTemp() || hydro->relaxBotVelo() ||
      hydro->topSpongeLyr() || hydro->botSpongeLyr() || hydro->plumeForcing() ||
      !pmb->user_stage_forcings.empty())
    why.push_back("forcing is not const-gravity alone");
  if (vars.count("solid")) why.push_back("a solid mask is set");
  // an outflow (radiating) ghost fill checks its inputs on the host (#314):
  // per op (the boundaries op, eager) unless conversion-order 4 also fills
  // the primitive ghosts with it inside the hydro step (hydro_forward.cpp)
  if (hydro->conversion_order() == 4 && outflow_boundary(block))
    why.push_back(
        "an outflow boundary is set with conversion-order 4 (the hydro step "
        "refills its primitive ghosts with it)");
  // S-K4: the chemistry step runs inside the last stage; whether kintera's
  // kinetics path captures is not verified, so a block with one runs eagerly
  if (pmb->_chemistry) why.push_back("a chemistry step is registered");
  return why;
}

Capability StageExecutor::capability(int block, Variables const& vars) const {
  Capability c;
  bool fp32 = block_at(block)->policy.fp32();
  c.eager_fp64 = !fp32;  // every case runs eager in its precision
  c.eager_fp32 = fp32;
  // the case's reasons make every segment eager; an eager op adds its own
  auto why = capture_reasons(block, vars);
  c.eager_reasons = why;
  auto const& segs = schedule(block).segments();
  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    auto const& seg = segs[i];
    Capability::SegmentRow row;
    row.index = i;
    row.kind = to_string(seg.kind);
    row.reads_host_dt = seg.reads_host_dt;
    row.ops = seg.names;
    if (runs_on_host(seg.kind)) {  // a boundary, not a refusal
      row.host = true;
      c.segments.push_back(row);
      continue;
    }
    row.reasons = why;
    for (auto const& r : seg.eager_reasons) {
      row.reasons.push_back(r);
      if (std::find(c.eager_reasons.begin(), c.eager_reasons.end(), r) ==
          c.eager_reasons.end())
        c.eager_reasons.push_back(r);
    }
    row.capturable = row.reasons.empty();
    c.segments.push_back(row);
  }
  bool graph = c.eager_reasons.empty();
  c.graph_fp64 = graph && !fp32;
  c.graph_fp32 = graph && fp32;
  return c;
}

GraphKey StageExecutor::make_key(int block, int segment, Variables const& vars,
                                 double dt, int stage) const {
  auto* pmb = block_at(block);
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
  key.store_generation = pmb->_state.generation();
  key.config_generation = _config_generation;
  key.block = block;
  key.stage = stage;
  key.segment = segment;
  // dt is in the key only while an op of the segment reads the host dt; the
  // others read the block's device dt (V3 D)
  key.dt = schedule(block).segments().at(segment).reads_host_dt ? dt : 0.;
  if (auto pthermo = thermo_of(block))
    key.thermo_epoch = kintera::alloc_epoch(*pthermo);
  return key;
}

std::map<std::string, torch::Tensor> StageExecutor::save_list(
    int block, Variables& vars) const {
  auto* pmb = block_at(block);
  std::map<std::string, torch::Tensor> list;
  for (auto name : {"hydro_u", "scalar_s", "scalar_r", "hydro_w"}) {
    auto it = vars.find(name);
    if (it != vars.end() && it->second.defined())
      list[std::string("vars.") + name] = it->second;
  }
  auto& state = pmb->_state;
  list["u0"] = state.u0();
  if (state.s0().numel() > 0) list["s0"] = state.s0();
  for (auto const& [name, t] : state.registered())
    if (t.defined()) list["storage." + name] = t;
  // K2 (K.rst 3.2): the ThermoY failure counter, so a warm-up leaves no count
  if (auto pthermo = thermo_of(block)) {
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

std::shared_ptr<kintera::ThermoYImpl> StageExecutor::thermo_of(
    int block) const {
  return block_at(block)->thermo();
}

void StageExecutor::run_stage(int block, Variables& vars, double dt,
                              int stage) {
  auto* pmb = block_at(block);
  auto const& op = pmb->options;
  if (op->execution() != "cuda-graph") {
    ++_eager;
    run_eager(block, vars, dt, stage);
    return;
  }

  // the case's reasons make every segment eager; strict names the first
  // eager segment and its reasons
  auto const& segs = schedule(block).segments();
  if (_schedule_rebuilt[block]) {  // its segments moved: drop its graphs
    invalidate(block);
    _schedule_rebuilt[block] = 0;
  }
  auto why = capture_reasons(block, vars);
  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    if (runs_on_host(segs[i].kind)) continue;  // a boundary, not a refusal
    auto reasons = why;
    for (auto const& r : segs[i].eager_reasons) reasons.push_back(r);
    if (reasons.empty()) continue;
    std::string list, ops;
    for (auto const& w : reasons) list += "\n  - " + w;
    for (auto const& n : segs[i].names) ops += (ops.empty() ? "" : ", ") + n;
    TORCH_CHECK(!op->execution_strict(),
                "integration/execution-strict: this case cannot run as a "
                "CUDA graph: segment ",
                i, " (", ops, ") of block ", block, ", stage ", stage,
                " runs eager:", list);
  }
  if (!why.empty()) {
    ++_eager;
    run_eager(block, vars, dt, stage);
    return;
  }

#ifdef USE_CUDA
  pmb->_state.adopt(vars);  // a rebound tensor bumps the generation
  pmb->register_flux_storage();
  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    if (runs_on_host(segs[i].kind)) {  // the exchange: a boundary
      ++_host;
      run_eager(block, vars, dt, stage, i);
      continue;
    }
    if (!segs[i].capturable) {  // an eager op's segment
      ++_eager;
      run_eager(block, vars, dt, stage, i);
      continue;
    }
    auto it = _graphs.find({block, stage, i});
    if (it == _graphs.end() ||
        it->second->key != make_key(block, i, vars, dt, stage))
      capture(block, i, vars, dt, stage);
    replay(block, i, vars, dt, stage);
  }
#endif
}

#ifdef USE_CUDA
namespace {
// the CUDA sync-debug mode off for a scope; the caller's mode is restored on
// exit, by an exception too
struct SyncDebugOff {
  c10::cuda::WarningState& warn = c10::cuda::warning_state();
  c10::cuda::SyncDebugMode mode = warn.get_sync_debug_mode();
  SyncDebugOff() {
    warn.set_sync_debug_mode(c10::cuda::SyncDebugMode::L_DISABLED);
  }
  ~SyncDebugOff() { warn.set_sync_debug_mode(mode); }
};
}  // namespace
#endif

void StageExecutor::capture(int block, int segment, Variables& vars, double dt,
                            int stage) {
#ifdef USE_CUDA
  auto* pmb = block_at(block);
  _graphs.erase({block, stage, segment});
  auto const& sched = schedule(block);
  auto const& seg = sched.segments().at(segment);
  auto device = vars.at("hydro_u").device().index();
  auto current = at::cuda::getCurrentCUDAStream(device);
  auto side = at::cuda::getStreamFromPool(false, device);
  // the device dt, set on the current stream before the side stream waits;
  // a segment with an op that still reads the host dt captures that dt (and
  // keeps it in its key), every other one the block's device dt (V3 D)
  auto sdt = device_dt(block, vars, dt);
  StepDt wdt = seg.reads_host_dt ? StepDt(dt) : sdt;

  // K.rst 3.5, 7: the thermo is prepared first, under the sync-exempt mode
  // the warm-up uses (its table upload is a synchronizing copy), so the save
  // list below already holds the counter and the active-set buffers on the
  // first capture; from then on no kintera table is built inside the
  // warm-up or the capture (a missing one throws)
  // The segment's ops prepare (Op::prepare, outside capture) with the
  // state's device, dtype and shape: advance's prepares the thermo and
  // the monitor's amounts (still on the host before a block's first step),
  // so the save list below holds them from the first capture on.
  {
    SyncDebugOff off;
    auto const& u = vars.at("hydro_u");
    for (int i : seg.ops)
      if (sched.ops()[i].prepare)
        sched.ops()[i].prepare(u.device(), u.scalar_type(), u.sizes());
  }
  kintera::NoTableBuild no_table_build;
  auto handles = save_list(block, vars);
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
    SyncDebugOff off;
    c10::cuda::CUDAStreamGuard guard(side);
    for (int i = 0; i < 3; ++i) run_eager(block, vars, wdt, stage, segment);
  }
  at::cuda::CUDAEvent warm;
  warm.record(side);
  warm.block(current);
  restore();
  pmb->register_flux_storage();

  auto entry = std::make_shared<Captured>();
  for (auto const& [name, t] : vars)
    if (t.defined()) entry->inputs[name] = t.detach();

  at::cuda::CUDAEvent restored;
  restored.record(current);
  restored.block(side);
  entry->graph = std::make_unique<at::cuda::CUDAGraph>();
  // the block's shared pool: its graphs replay in turn on one stream, so a
  // graph's dead temporaries may be another's (CUDAGraph.h: pool handle).
  // With no live graph left (all dropped: to(), a restart, a schedule
  // rebuild) the old pool may still hold live tensors (a replay sets
  // scalar_r to a graph output) at a use count of 0, and capturing into it
  // again trips the caching allocator: start a fresh pool; the old one is
  // freed with its last tensor.
  bool live = false;
  for (auto const& [slot, g] : _graphs)
    if (std::get<0>(slot) == block) live = true;
  if (!live || _pools[block] == std::pair<uint64_t, uint64_t>{0, 0}) {
    auto id = at::cuda::graph_pool_handle();
    _pools[block] = {id.first, id.second};
  }
  {
    c10::cuda::CUDAStreamGuard guard(side);
    entry->graph->capture_begin({_pools[block].first, _pools[block].second});
    run_eager(block, vars, wdt, stage, segment);
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

  entry->key = make_key(block, segment, vars, dt, stage);
  _graphs[{block, stage, segment}] = entry;
  ++_captures;
#else
  TORCH_CHECK(false, "StageExecutor: snapy was built without CUDA");
#endif
}

void StageExecutor::replay(int block, int segment, Variables& vars, double dt,
                           int stage) {
#ifdef USE_CUDA
  auto* pmb = block_at(block);
  device_dt(block, vars, dt);  // the graph's kernels read it
  auto& entry = *_graphs.at({block, stage, segment});
  for (auto const& [name, in] : entry.inputs) {
    auto const& t = vars.at(name);
    if (t.data_ptr() != in.data_ptr()) in.copy_(t);
  }
  pmb->phydro->rk_stage = stage;  // advance_local's host side
  entry.graph->replay();
  for (auto const& [name, out] : entry.outputs) {
    auto& t = vars.at(name);
    if (t.data_ptr() != out.data_ptr()) t.set_(out);
  }
  ++_replays;
#endif
}

void StageExecutor::invalidate() { _graphs.clear(); }

void StageExecutor::invalidate(int block) {
  for (auto it = _graphs.begin(); it != _graphs.end();)
    it = std::get<0>(it->first) == block ? _graphs.erase(it) : std::next(it);
}

std::map<std::string, int64_t> StageExecutor::report() const {
  return {{"captures", _captures.load()},
          {"replays", _replays.load()},
          {"eager", _eager.load()},
          {"host", _host.load()},
          {"graphs", static_cast<int64_t>(_graphs.size())},
          {"pools", static_cast<int64_t>(std::count_if(
                        _pools.begin(), _pools.end(), [](auto const& p) {
                          return p != std::pair<uint64_t, uint64_t>{0, 0};
                        }))}};
}

}  // namespace snap
