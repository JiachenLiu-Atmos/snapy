// C/C++
#include <algorithm>
#include <functional>
#include <set>

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
#include "exchange_cut.hpp"
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
         implicit_precision == other.implicit_precision &&
         implicit_fallback_fp64 == other.implicit_fallback_fp64 &&
         thermo_epoch == other.thermo_epoch;
}

#ifdef USE_CUDA
namespace {
// a CUDAGraph registers with the CUDA generator's state at capture_begin and
// unregisters when destroyed (an unordered_set, unlocked: CUDAGeneratorImpl.h
// registered_graphs_), and capture_end and replay touch that state too. A
// Mesh's blocks capture and replay on their worker threads, so every one of
// these host calls takes this lock (the GPU work stays concurrent)
std::mutex g_graph_mutex;
struct GraphDeleter {
  void operator()(at::cuda::CUDAGraph* g) const {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    delete g;
  }
};
using GraphPtr = std::unique_ptr<at::cuda::CUDAGraph, GraphDeleter>;

//! a step of a captured segment: a graph, or the host step of a cut
struct SegmentStep {
  GraphPtr graph;
  std::function<void()> host;
};
}  // namespace
#endif

struct StageExecutor::Captured {
  GraphKey key;
#ifdef USE_CUDA
  //! the segment as run: graphs and, between them, the host steps of its
  //! cuts at an exchange (B3), in order; without a cut, one graph
  std::vector<SegmentStep> steps;
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
  _vars_reasons.resize(_blocks.size());
  _logged.resize(_blocks.size());
  _schedules.resize(_blocks.size());
  _schedule_sig.assign(_blocks.size(), -1);
  _schedule_rebuilt.assign(_blocks.size(), 0);
  _pools.assign(_blocks.size(), {0, 0});
  _capture_streams.assign(_blocks.size(), nullptr);
  for (int b = 0; b < num_blocks(); ++b) build_schedule(b);
}
StageExecutor::~StageExecutor() = default;

std::string StageExecutor::mode(int block) const {
  auto* pmb = block_at(block);
  auto const& op = pmb->options;
  if (op->execution() != "auto") return op->execution();
  if (op->execution_strict()) return "cuda-graph";
  bool cuda =
      pmb->pcoord && pmb->pcoord->x1v.defined() && pmb->pcoord->x1v.is_cuda();
  return cuda ? "cuda-graph" : "eager";
}

void StageExecutor::log_eager(int block, std::vector<std::string> const& why) {
  if (why.empty() || block_at(block)->options->execution_strict()) return;
  std::lock_guard<std::mutex> lock(_logged_mutex);
  for (auto const& r : why)
    if (_logged[block].insert(r).second)
      TORCH_WARN("integration/execution: block ", block,
                 " runs eager where it cannot be captured: ", r);
}

bool StageExecutor::user_forcing(int block) const {
  return !block_at(block)->user_stage_forcings.empty();
}

bool StageExecutor::chemistry_slot(int block) const {
  // V3 C5: a Mesh's block defers its chemistry to the Mesh's global phase
  // (Mesh::forward runs it after the stages, eagerly, before the final
  // exchange), so its chemistry op does nothing and stays in the graph
  auto* pmb = block_at(block);
  return pmb->_chemistry != nullptr && !pmb->_mesh_executor;
}

int StageExecutor::schedule_signature(int block) const {
  return (outflow_boundary(block) ? 1 : 0) | (user_forcing(block) ? 2 : 0) |
         (chemistry_slot(block) ? 4 : 0);
}

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
  // block's device dt (V3 D: a StepDt), not the host dt, and so does the
  // vertical implicit solve (C4: its kernels, its I/dt rows and its posts),
  // so this segment keeps no dt in its key; a host read under capture
  // throws (StepDt::value)
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
  // a user TorchScript forcing is opaque (X.rst section 7: an eager segment);
  // the stage's other ops still capture, the tendencies cross in stage.du/ds
  bool user = user_forcing(block);
  if (user) {
    forcing.capturable = false;
    forcing.eager_reason = "a user TorchScript stage forcing is set";
  }
  // a user forcing is called with the host dt (dt.value()); its op is then
  // eager, so it runs with the host dt and keys no graph. Without one the op
  // reads no dt. Either way the graph segments around it read the device dt
  // and keep no dt in their keys (V3 D).
  forcing.reads_host_dt = user;
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
  // ChemistrySource::forward takes the host dt (dt.value()), only when a
  // chemistry step is registered. V3 C2: that step is then an eager FP64
  // segment (X.rst section 3, K 3.6): its kintera Kinetics and ThermoX build
  // their tables on first use, which a capture refuses (NoTableBuild). It
  // runs eagerly with the host dt, as a user forcing does, and keys no graph;
  // the ops before it and the boundaries after it still capture with the
  // device dt (V3 D). It reads and writes hydro_u and hydro_w, the fixed
  // StateStore buffers the graphs replay into. Without one the op reads no dt
  bool chem = chemistry_slot(block);
  if (chem) {
    chemistry.capturable = false;
    chemistry.eager_reason =
        "a chemistry step is registered (eager FP64 kinetics, V3 C2)";
  }
  chemistry.reads_host_dt = chem;
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
  if (_schedule_sig[block] >= 0) _schedule_rebuilt[block] = 1;
  _schedule_sig[block] = schedule_signature(block);
}

StageSchedule const& StageExecutor::schedule(int block) const {
  block_at(block);  // checked
  if (_schedule_sig[block] != schedule_signature(block)) build_schedule(block);
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

#ifdef USE_CUDA
namespace {
// B3: the segment this thread captures, while StageExecutor::capture runs its
// ops under capture; exchange_cut closes the open graph there and opens the
// next one in the same pool and mode
struct CutCapture {
  std::vector<SegmentStep>* steps = nullptr;
  GraphPtr open;  // the graph being captured
  at::cuda::MempoolId_t pool;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal;
  bool in_host = false;  // running a cut's host step (not capturing)

  void begin() {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    open = GraphPtr(new at::cuda::CUDAGraph());
    open->capture_begin(pool, mode);
  }
  // ends the open capture; keeps the graph unless it captured nothing (a cut
  // right at the start, or two cuts in a row), whose "graph is empty"
  // warning is expected here and silenced
  void close() {
    size_t nodes = 1;
    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cudaStreamGetCaptureInfo(stream, &status, nullptr, &g, nullptr,
                                 nullptr) == cudaSuccess &&
        status == cudaStreamCaptureStatusActive && g != nullptr)
      C10_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
    if (nodes > 0) {
      {
        std::lock_guard<std::mutex> lock(g_graph_mutex);
        open->capture_end();
      }
      steps->push_back({std::move(open), nullptr});
      return;
    }
    struct Silent : c10::WarningHandler {
      void process(c10::Warning const&) override {}
    } silent;
    c10::WarningUtils::WarningHandlerGuard guard(&silent);
    {
      std::lock_guard<std::mutex> lock(g_graph_mutex);
      open->capture_end();
    }
    open.reset();
  }
};
thread_local CutCapture* g_cut = nullptr;
// a cut's host step is running on this thread (capture or replay)
thread_local bool g_in_cut_host = false;
struct CutHostScope {
  bool prev;
  CutHostScope() : prev(g_in_cut_host) { g_in_cut_host = true; }
  ~CutHostScope() { g_in_cut_host = prev; }
};

// sets g_cut for a scope, restored by an exception too
struct CutScope {
  CutCapture* prev;
  explicit CutScope(CutCapture* c) : prev(g_cut) { g_cut = c; }
  ~CutScope() { g_cut = prev; }
};
}  // namespace
#endif

void exchange_cut(std::function<void()> host) {
#ifdef USE_CUDA
  auto* c = g_cut;
  if (c == nullptr || c->in_host) {
    host();
    return;
  }
  c->close();
  c->in_host = true;
  try {
    CutHostScope host_step;
    host();
  } catch (...) {
    c->in_host = false;
    throw;
  }
  c->in_host = false;
  c->steps->push_back({nullptr, host});
  c->begin();
#else
  host();
#endif
}

bool in_cut_host_step() {
#ifdef USE_CUDA
  return g_in_cut_host;
#else
  return false;
#endif
}

#ifdef USE_CUDA
//! a rendezvous's seam-copy graphs, one per signature: two exchanges can
//! share a rendezvous (the same sync options, other variables, e.g. the
//! positivity limiter's theta and species syncs) and take turns
struct SeamCopies {
  std::map<std::vector<int64_t>, GraphPtr> graphs;
};
#else
struct SeamCopies {};
#endif

namespace {
// the process's seam-copy graphs: captured, replayed (execution_report)
std::atomic<int64_t> g_seam_captures{0}, g_seam_replays{0};
// past this many signatures a rendezvous's addresses keep moving: its old
// graphs are dropped
constexpr size_t kMaxSeamGraphs = 16;
}  // namespace

void run_seam_copies(std::shared_ptr<SeamCopies>& cache,
                     std::vector<int64_t> const& signature,
                     std::function<void()> const& copies) {
#ifdef USE_CUDA
  if (!g_in_cut_host) {
    copies();
    return;
  }
  if (!cache) cache = std::make_shared<SeamCopies>();
  auto it = cache->graphs.find(signature);
  if (it == cache->graphs.end()) {
    // dropped graphs are destroyed at the end of this block, outside the
    // graph lock (their deleter takes it)
    std::map<std::vector<int64_t>, GraphPtr> dropped;
    if (cache->graphs.size() >= kMaxSeamGraphs) dropped.swap(cache->graphs);
    GraphPtr g;
    {
      std::lock_guard<std::mutex> lock(g_graph_mutex);
      // captured on a stream of its own, one per device (these captures take
      // turns under the lock): the current stream at a replay is the block's
      // worker stream, which comes from the shared pool another block may be
      // launching from or capturing on. Never destroyed, as a block's own
      // capture stream
      static std::map<int, cudaStream_t> own;
      int dev = c10::cuda::current_device();
      if (!own[dev])
        C10_CUDA_CHECK(
            cudaStreamCreateWithFlags(&own[dev], cudaStreamNonBlocking));
      c10::cuda::CUDAStreamGuard on_own(
          at::cuda::getStreamFromExternal(own[dev], dev));
      g.reset(new at::cuda::CUDAGraph());
      // its own pool: the copies allocate nothing
      g->capture_begin(at::cuda::graph_pool_handle(),
                       cudaStreamCaptureModeThreadLocal);
      try {
        copies();
      } catch (...) {
        try {
          g->capture_end();
        } catch (...) {
        }
        g.release();  // leaked on this error path, as a failed stage capture
        throw;
      }
      g->capture_end();
    }
    it = cache->graphs.emplace(signature, std::move(g)).first;
    ++g_seam_captures;
  }
  {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    it->second->replay();
  }
  ++g_seam_replays;
#else
  copies();
#endif
}

bool in_segment_capture() {
#ifdef USE_CUDA
  return g_cut != nullptr && !g_cut->in_host;
#else
  return false;
#endif
}

void refuse_in_segment_capture(char const* what) {
  TORCH_CHECK(!in_segment_capture(), "StageExecutor: ", what,
              " inside a captured segment has no cut at its exchange (B3): "
              "this case should have been refused before capture; please "
              "report the deck");
}

void StageExecutor::run_local(int block, Variables& vars, double dt,
                              int stage) {
  // the stage without its entry exchange: the Mesh exchanges last
  dispatch(block, vars, dt, stage, /*with_host=*/false,
           [&]() { block_at(block)->advance_local(vars, dt, stage); });
}

void StageExecutor::align_captures(std::vector<Variables>& vars, double dt,
                                   int stage) {
#ifdef USE_CUDA
  if (num_blocks() < 2) return;
  TORCH_CHECK(static_cast<int>(vars.size()) == num_blocks(),
              "StageExecutor::align_captures: one Variables map per block");
  std::set<int> stale;  // segments some block must (re)capture
  for (int b = 0; b < num_blocks(); ++b) {
    auto* pmb = block_at(b);
    if (mode(b) != "cuda-graph") return;
    schedule(b);  // rebuilt if its signature moved
    if (_schedule_rebuilt[b]) {
      invalidate(b);
      _schedule_rebuilt[b] = 0;
    }
    if (!capture_reasons(b, vars[b]).empty()) continue;
    pmb->_state.adopt(vars[b]);  // as run_segments, before its keys
    pmb->register_flux_storage();
    auto const& segs = schedule(b).segments();
    for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
      if (runs_on_host(segs[i].kind) || !segs[i].capturable) continue;
      auto g = graph_at({b, stage, i});
      if (!g || g->key != make_key(b, i, vars[b], dt, stage)) stale.insert(i);
    }
  }
  if (stale.empty()) return;
  std::lock_guard<std::mutex> lock(_graphs_mutex);
  for (int b = 0; b < num_blocks(); ++b)
    for (int i : stale) _graphs.erase({b, stage, i});
#endif
}

void StageExecutor::agree(std::vector<Variables> const& vars) {
  TORCH_CHECK(static_cast<int>(vars.size()) == num_blocks(),
              "StageExecutor::agree: one Variables map per block");
  for (int b = 0; b < num_blocks(); ++b)
    _vars_reasons[b] = vars_reasons(vars[b]);
}

std::vector<std::string> StageExecutor::vars_reasons(Variables const& vars) {
  std::vector<std::string> why;
  auto it = vars.find("hydro_u");
  if (it == vars.end() || !it->second.defined() || !it->second.is_cuda())
    why.push_back("the state is not on a CUDA device");
  if (vars.count("solid")) why.push_back("a solid mask is set");
  return why;
}

std::vector<std::string> StageExecutor::capture_reasons(
    int block, Variables const& vars) const {
  auto why = vars_reasons(vars);
  for (auto& r : config_reasons(block)) why.push_back(std::move(r));
  if (!why.empty() || num_blocks() < 2) return why;
  // B3 (d)/(e): a Mesh whose blocks disagree runs eager as a whole. A block
  // that could capture while another runs eagerly (it refuses, or its mode
  // is eager) would meet that block's in-stage exchanges at the same-process
  // rendezvous a different number of times (warm-up and capture against one
  // eager pass), and the Mesh would hang. The other blocks' reasons are
  // their configuration's, live, and their variables' as of the last
  // agree() (Mesh::initialize and every Mesh::forward)
  auto const mine = mode(block);
  for (int b = 0; b < num_blocks(); ++b) {
    if (b == block) continue;
    auto const other = mode(b);
    if (other != mine) {
      why.push_back("a Mesh whose blocks disagree: block " + std::to_string(b) +
                    " runs " + other + ", block " + std::to_string(block) +
                    " runs " + mine + ", so the whole Mesh runs eager");
      break;
    }
    auto theirs = config_reasons(b);
    for (auto const& r : _vars_reasons[b]) theirs.push_back(r);
    if (!theirs.empty()) {
      why.push_back("a Mesh whose blocks disagree: block " + std::to_string(b) +
                    " refuses capture (" + theirs.front() +
                    "), so the whole Mesh runs eager");
      break;
    }
  }
  return why;
}

std::vector<std::string> StageExecutor::config_reasons(int block) const {
  // the prototype tests/straka_cuda_graph.py:16-31 at d6a3c00c: dry Straka
  std::vector<std::string> why;
  auto* pmb = block_at(block);
  auto const& op = pmb->options;
  auto const& hydro = op->hydro();
#ifndef USE_CUDA
  why.push_back("snapy was built without CUDA");
#endif
  // B3: the cubed sphere (gnomonic-equiangle panels of a cubed-sphere
  // layout): its in-stage panel-seam exchanges are cuts
  bool cubed_sphere = op->coord()->type() == "gnomonic-equiangle" &&
                      pmb->_playout->options->type() == "cubed-sphere";
  if (op->coord()->type() != "cartesian" && !cubed_sphere)
    why.push_back("geometry is " + op->coord()->type() +
                  ", not cartesian or a cubed sphere");
  if (hydro->eos()->limiter()) why.push_back("the EOS limiter is on");
  // an allow-list of the equations of state and Riemann solvers a capture
  // test covers (tests/test_graph_capture_lifts.py, the graph = eager tests);
  // every other one (aneos, moist-mixture, shallow-water; roe, shallow-roe,
  // upwind) is refused, named, until a capture test covers it
  static std::set<std::string> const capture_eos = {"ideal-gas", "ideal-moist"};
  static std::set<std::string> const capture_riemann = {"lmars", "hllc"};
  if (!capture_eos.count(hydro->eos()->type()))
    why.push_back("equation of state is " + hydro->eos()->type() +
                  "; capture allows ideal-gas, ideal-moist");
  if (!capture_riemann.count(hydro->riemann()->type()))
    why.push_back("riemann solver is " + hydro->riemann()->type() +
                  "; capture allows lmars, hllc");
  if (op->intg()->type() != "rk3")
    why.push_back("integration type is " + op->intg()->type() + ", not rk3");
  // N blocks (B2): the blocks of one process capture. Their exchanges inside
  // a stage (conversion-order 4's primitive ghosts, the positivity limiters'
  // theta sync) are cuts (B3: pack and unpack captured, the rendezvous on the
  // host between the graphs). Ranks and x1 splits are not cut yet
  auto const& lo = pmb->_playout->options;
  if (pmb->_playout->has_process_group())
    why.push_back(
        "more than one process: multi-rank capture is not done yet (its "
        "exchanges run on the host between segments, P1)");
  if (lo->pz() > 1)
    why.push_back(
        "an x1 split (pz > 1): its relays inside the stage are not cut into "
        "segments yet (P2, P3)");
  // no forcing block at all captures as const-gravity does
  if (hydro->coriolis() || hydro->diffusion() || hydro->bodyHeat() ||
      hydro->botHeat() || hydro->topCool() || hydro->relaxBotComp() ||
      hydro->relaxBotTemp() || hydro->relaxBotVelo() || hydro->topSpongeLyr() ||
      hydro->botSpongeLyr() || hydro->plumeForcing())
    why.push_back("forcing is not const-gravity alone");
  // an outflow (radiating) ghost fill checks its inputs on the host (#314):
  // per op (the boundaries op, eager) unless conversion-order 4 also fills
  // the primitive ghosts with it inside the hydro step (hydro_forward.cpp)
  if (hydro->conversion_order() == 4 && outflow_boundary(block))
    why.push_back(
        "an outflow boundary is set with conversion-order 4 (the hydro step "
        "refills its primitive ghosts with it)");
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
  if (auto const& icorr = pmb->options->hydro()->icorr()) {
    key.implicit_precision = icorr->precision();
    key.implicit_fallback_fp64 = icorr->fallback_fp64();
  }
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
  // the block's EfStatus (positivity census, lim_cut/lim_flux, limiter
  // marks, vapor/heat-capacity flags), so a warm-up leaves no count
  for (int s = 0; s < EfStatus::kNumSlots; ++s) {
    auto const& t = pmb->pefs->slot(EfStatus::Slot(s));
    if (t.defined()) list["efs." + std::to_string(s)] = t;
  }
  // conversion order 4's fallback count, a hydro buffer outside the EfStatus
  auto hydro_buffers = pmb->phydro->named_buffers(/*recurse=*/false);
  if (auto t = hydro_buffers.find("conversion_trips"))
    list["hydro.conversion_trips"] = *t;
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

void StageExecutor::check_strict(int block, Variables const& vars,
                                 int stage) const {
  auto const& op = block_at(block)->options;
  if (mode(block) != "cuda-graph" || !op->execution_strict()) return;
  // the case's reasons make every segment eager; strict names the first
  // eager segment and its reasons (host segments are boundaries)
  auto const& segs = schedule(block).segments();
  auto why = capture_reasons(block, vars);
  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    if (runs_on_host(segs[i].kind)) continue;  // a boundary, not a refusal
    auto reasons = why;
    for (auto const& r : segs[i].eager_reasons) reasons.push_back(r);
    if (reasons.empty()) continue;
    std::string list, ops;
    for (auto const& w : reasons) list += "\n  - " + w;
    for (auto const& n : segs[i].names) ops += (ops.empty() ? "" : ", ") + n;
    TORCH_CHECK(false,
                "integration/execution-strict: this case cannot run as a "
                "CUDA graph: segment ",
                i, " (", ops, ") of block ", block, ", stage ", stage,
                " runs eager:", list);
  }
}

void StageExecutor::run_stage(int block, Variables& vars, double dt,
                              int stage) {
  dispatch(block, vars, dt, stage, /*with_host=*/true,
           [&]() { run_eager(block, vars, dt, stage); });
}

void StageExecutor::dispatch(int block, Variables& vars, double dt, int stage,
                             bool with_host,
                             std::function<void()> const& eager) {
  if (mode(block) != "cuda-graph") {
    ++_eager;
    eager();
    return;
  }
  schedule(block);                 // rebuilt if its signature moved
  if (_schedule_rebuilt[block]) {  // its segments moved: drop its graphs
    invalidate(block);
    _schedule_rebuilt[block] = 0;
  }
  check_strict(block, vars, stage);
  if (auto why = capture_reasons(block, vars); !why.empty()) {
    log_eager(block, why);
    ++_eager;
    eager();
    return;
  }
  run_segments(block, vars, dt, stage, with_host);
}

std::shared_ptr<StageExecutor::Captured> StageExecutor::graph_at(
    Slot const& slot) const {
  std::lock_guard<std::mutex> lock(_graphs_mutex);
  auto it = _graphs.find(slot);
  return it == _graphs.end() ? nullptr : it->second;
}

void StageExecutor::run_segments(int block, Variables& vars, double dt,
                                 int stage, bool with_host) {
#ifdef USE_CUDA
  auto* pmb = block_at(block);
  auto const& segs = schedule(block).segments();
  pmb->_state.adopt(vars);  // a rebound tensor bumps the generation
  pmb->register_flux_storage();
  for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
    if (runs_on_host(segs[i].kind)) {  // the exchange: a boundary
      if (!with_host) continue;        // a Mesh exchanges last, itself
      ++_host;
      run_eager(block, vars, dt, stage, i);
      continue;
    }
    if (!segs[i].capturable) {  // an eager op's segment
      log_eager(block, segs[i].eager_reasons);
      ++_eager;
      run_eager(block, vars, dt, stage, i);
      continue;
    }
    auto g = graph_at({block, stage, i});
    if (!g || g->key != make_key(block, i, vars, dt, stage))
      capture(block, i, vars, dt, stage);
    replay(block, i, vars, dt, stage);
  }
#endif
}

#ifdef USE_CUDA
namespace {
// the CUDA sync-debug mode off for a scope; the caller's mode is restored on
// exit, by an exception too. The mode is process-wide and a Mesh's blocks
// warm up together on their worker threads (B3), so the scopes are counted:
// the first one in turns it off, the last one out restores it
std::mutex g_sync_debug_mutex;
int g_sync_debug_off = 0;
c10::cuda::SyncDebugMode g_sync_debug_saved =
    c10::cuda::SyncDebugMode::L_DISABLED;
struct SyncDebugOff {
  SyncDebugOff() {
    std::lock_guard<std::mutex> lock(g_sync_debug_mutex);
    auto& warn = c10::cuda::warning_state();
    if (g_sync_debug_off++ == 0) {
      g_sync_debug_saved = warn.get_sync_debug_mode();
      warn.set_sync_debug_mode(c10::cuda::SyncDebugMode::L_DISABLED);
    }
  }
  ~SyncDebugOff() {
    std::lock_guard<std::mutex> lock(g_sync_debug_mutex);
    if (--g_sync_debug_off == 0)
      c10::cuda::warning_state().set_sync_debug_mode(g_sync_debug_saved);
  }
};
}  // namespace
#endif

void StageExecutor::capture(int block, int segment, Variables& vars, double dt,
                            int stage) {
#ifdef USE_CUDA
  auto* pmb = block_at(block);
  // a block of a multi-block executor or process captures in thread-local
  // mode: the other blocks launch from their worker threads meanwhile. They
  // capture this segment too (align_captures: an exchange inside it meets
  // them at the same-process rendezvous, in the warm-up and at the capture's
  // cuts), so captures are concurrent there; a lone block's are serialized
  bool multi = num_blocks() > 1 ||
               pmb->_playout->options->blocks_per_process() > 1 ||
               pmb->_playout->has_process_group();
  std::unique_lock<std::mutex> one_capture(_capture_mutex, std::defer_lock);
  if (!multi) one_capture.lock();
  {
    std::lock_guard<std::mutex> lock(_graphs_mutex);
    _graphs.erase({block, stage, segment});
  }
  auto const& sched = schedule(block);
  auto const& seg = sched.segments().at(segment);
  auto device = vars.at("hydro_u").device().index();
  auto current = at::cuda::getCurrentCUDAStream(device);
  // the block's own capture stream: the shared pool hands its streams out in
  // turn (also to the Mesh's worker streams), so with many blocks two
  // concurrent captures could get one stream. Created once, never destroyed:
  // a graph's outputs can outlive the executor, and the caching allocator
  // may still record events on the stream they were captured on
  if (!_capture_streams[block]) {
    cudaStream_t s = nullptr;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    _capture_streams[block] = s;
  }
  auto side = at::cuda::getStreamFromExternal(
      static_cast<cudaStream_t>(_capture_streams[block]), device);
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
  // V3 C1: the implicit correction's scratch, status and FP64-stage predicate
  // are allocated here, not in the warm-up, and registered, so the save list
  // holds them
  auto wit = vars.find("hydro_w");
  if (wit != vars.end() && wit->second.defined())
    pmb->phydro->prepare_implicit(wit->second);
  pmb->register_flux_storage();
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
  // the block's shared pool: its graphs replay in turn on one stream, so a
  // graph's dead temporaries may be another's (CUDAGraph.h: pool handle).
  // With no live graph left (all dropped: to(), a restart, a schedule
  // rebuild) the old pool may still hold live tensors (a replay sets
  // scalar_r to a graph output) at a use count of 0, and capturing into it
  // again trips the caching allocator: start a fresh pool; the old one is
  // freed with its last tensor.
  bool live = false;
  {
    std::lock_guard<std::mutex> lock(_graphs_mutex);
    for (auto const& kv : _graphs)
      if (std::get<0>(kv.first) == block) live = true;
  }
  if (!live || _pools[block] == std::pair<uint64_t, uint64_t>{0, 0}) {
    auto id = at::cuda::graph_pool_handle();
    _pools[block] = {id.first, id.second};
  }
  {
    c10::cuda::CUDAStreamGuard guard(side);
    // a multi-block executor's other blocks keep launching from their worker
    // threads meanwhile: thread-local mode confines the capture to this one
    CutCapture cut;
    cut.steps = &entry->steps;
    cut.pool = {_pools[block].first, _pools[block].second};
    cut.mode =
        multi ? cudaStreamCaptureModeThreadLocal : cudaStreamCaptureModeGlobal;
    cut.begin();
    try {
      // a lone block's exchanges reach no other block: no cut context
      CutScope scope(multi ? &cut : nullptr);
      run_eager(block, vars, wdt, stage, segment);
    } catch (...) {
      // end an open capture so the stream leaves capture mode. A graph whose
      // capture failed is not destroyed (its generator registration may be
      // gone): it is leaked, once, on this error path
      if (cut.open) {
        std::lock_guard<std::mutex> lock(g_graph_mutex);
        try {
          cut.open->capture_end();
        } catch (...) {
        }
        cut.open.release();
      }
      throw;
    }
    cut.close();
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
  {
    std::lock_guard<std::mutex> lock(_graphs_mutex);
    _graphs[{block, stage, segment}] = entry;
  }
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
  auto g = graph_at({block, stage, segment});
  TORCH_CHECK(g, "StageExecutor: no graph for block ", block, ", stage ", stage,
              ", segment ", segment);
  auto& entry = *g;
  for (auto const& [name, in] : entry.inputs) {
    auto const& t = vars.at(name);
    if (t.data_ptr() != in.data_ptr()) in.copy_(t);
  }
  pmb->phydro->rk_stage = stage;  // advance_local's host side
  for (auto const& step : entry.steps) {
    if (step.graph) {
      std::lock_guard<std::mutex> lock(g_graph_mutex);
      step.graph->replay();
    } else {  // a cut's host step: the exchange's rendezvous or communicator
      CutHostScope host_step;
      step.host();
    }
  }
  for (auto const& [name, out] : entry.outputs) {
    auto& t = vars.at(name);
    if (t.data_ptr() != out.data_ptr()) t.set_(out);
  }
  ++_replays;
#endif
}

void StageExecutor::invalidate() {
  std::lock_guard<std::mutex> lock(_graphs_mutex);
  _graphs.clear();
}

void StageExecutor::invalidate(int block) {
  std::lock_guard<std::mutex> lock(_graphs_mutex);
  for (auto it = _graphs.begin(); it != _graphs.end();)
    it = std::get<0>(it->first) == block ? _graphs.erase(it) : std::next(it);
}

std::map<std::string, int64_t> StageExecutor::report() const {
  int64_t ngraphs, ncuts = 0;
  {
    std::lock_guard<std::mutex> lock(_graphs_mutex);
    ngraphs = static_cast<int64_t>(_graphs.size());
#ifdef USE_CUDA
    for (auto const& kv : _graphs)
      for (auto const& step : kv.second->steps) ncuts += step.graph ? 0 : 1;
#endif
  }
  return {{"captures", _captures.load()},
          {"replays", _replays.load()},
          {"eager", _eager.load()},
          {"host", _host.load()},
          {"graphs", ngraphs},
          {"cuts", ncuts},
          // the same-process seam copies' graphs, process-wide (B3)
          {"seam_captures", g_seam_captures.load()},
          {"seam_replays", g_seam_replays.load()},
          {"pools", static_cast<int64_t>(std::count_if(
                        _pools.begin(), _pools.end(), [](auto const& p) {
                          return p != std::pair<uint64_t, uint64_t>{0, 0};
                        }))}};
}

}  // namespace snap
