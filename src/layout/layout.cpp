// yaml
#include <sys/utsname.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>

// base
#include <configure.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

// snap
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "cubed_sphere_layout.hpp"
#include "layout.hpp"

namespace snap {

namespace {

int random_master_port() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(29500, 29600);
  return dist(gen);
}

std::string default_backend() {
#ifdef USE_UCX
  struct utsname system_info;
  if (uname(&system_info) == 0 && std::string(system_info.sysname) == "Darwin")
    return "gloo";
  return "ucx";
#else
  return "gloo";
#endif
}

struct LocalExchangeKey {
  int process_rank;
  int blocks_per_process;
  int dim;
  int phyid;
  int type;
  bool intra_panel_only;
  bool cross_panel_only;
  bool skip_corner;
  bool interpolate;
  std::string layout_type;
  std::string device;

  bool operator<(LocalExchangeKey const& other) const {
    return std::tie(process_rank, blocks_per_process, dim, phyid, type,
                    intra_panel_only, cross_panel_only, skip_corner,
                    interpolate, layout_type, device) <
           std::tie(other.process_rank, other.blocks_per_process, other.dim,
                    other.phyid, other.type, other.intra_panel_only,
                    other.cross_panel_only, other.skip_corner,
                    other.interpolate, other.layout_type, other.device);
  }
};

struct LocalExchangeState {
  int generation = 0;
  int arrived = 0;
  int released = 0;
  bool ready = false;
  std::exception_ptr error;
#ifdef USE_CUDA
  std::vector<std::shared_ptr<at::cuda::CUDAEvent>> arrival_events;
  std::shared_ptr<at::cuda::CUDAEvent> completion_event;
#endif
};

struct RemoteExchangeOp {
  LayoutImpl* layout;
  int remote_process;
  int local_block;
  int remote_local_block;
  int buffer_id;
  int send_tag;
  int recv_tag;
  std::tuple<int, int, int> offset;
  std::tuple<int, int, int> peer_offset;
};

std::mutex g_local_exchange_mutex;
std::condition_variable g_local_exchange_cv;
std::map<std::pair<int, int>, LayoutImpl*> g_local_layouts;
std::map<LocalExchangeKey, LocalExchangeState> g_local_exchange_states;
std::mutex g_process_comm_mutex;

//! same-process pieces of one x1 column, posted by its blocks (gather_x1)
struct ColumnBoard {
  int posted = 0;
  int taken = 0;
  bool full = false;
  std::map<int, torch::Tensor> pieces;
};
std::map<std::tuple<int, int, int>, ColumnBoard> g_column_boards;

//! same-process x1 reference anchors in flight (take_x1_anchor), keyed by
//! (process, receiving block rank)
std::map<std::pair<int, int>, torch::Tensor> g_x1_anchors;

//! same-process block-to-block messages in flight (post_to_local_block),
//! keyed by (process, sender block rank, receiver block rank, tag)
std::map<std::tuple<int, int, int, int>, torch::Tensor> g_local_mail;

std::pair<int, int> exchange_dz_bounds(LayoutImpl const& layout,
                                       SyncOptions const& opts) {
  if (layout.num_exchange_buffers() < 27) return {0, 0};
  return {opts.dz_min(), opts.dz_max()};
}

std::vector<LayoutImpl*> local_layouts_for(LayoutImpl const& layout) {
  std::vector<LayoutImpl*> layouts(layout.options->blocks_per_process(),
                                   nullptr);

  for (auto const& [key, local_layout] : g_local_layouts) {
    if (key.first == layout.options->process_rank()) {
      TORCH_CHECK(
          key.second >= 0 && key.second < layout.options->blocks_per_process(),
          "invalid local block index ", key.second,
          " for process-local layout registry");
      layouts[key.second] = local_layout;
    }
  }

  for (int i = 0; i < layouts.size(); ++i) {
    TORCH_CHECK(layouts[i] != nullptr,
                "missing process-local layout registration for process ",
                layout.options->process_rank(), " local block ", i);
  }

  return layouts;
}

LocalExchangeKey make_local_exchange_key(LayoutImpl const& layout,
                                         SyncOptions const& opts) {
  return LocalExchangeKey{
      layout.options->process_rank(),
      layout.options->blocks_per_process(),
      opts.dim(),
      opts.phyid(),
      opts.type(),
      opts.intra_panel_only(),
      opts.cross_panel_only(),
      opts.skip_corner(),
      opts.interpolate(),
      layout.options->type(),
      layout.options->device(),
  };
}

}  // namespace

std::string exchange_buffer_key(SyncOptions const& opts,
                                Variables const& vars) {
  std::ostringstream key;
  key << opts.dim() << ':' << opts.phyid() << ':' << opts.type() << ':'
      << opts.intra_panel_only() << ':' << opts.cross_panel_only() << ':'
      << opts.skip_corner() << ':' << opts.interpolate();
  for (auto const& [name, _] : vars) {
    key << ':' << name;
  }
  return key.str();
}

bool buffer_matches(torch::Tensor const& buffer, torch::Tensor const& source) {
  return buffer.defined() && buffer.sizes() == source.sizes() &&
         buffer.scalar_type() == source.scalar_type() &&
         buffer.device() == source.device();
}

void LayoutImpl::for_each_direction(
    SyncOptions const& opts,
    std::function<void(int, int, int)> const& fn) const {
  auto [dz_min, dz_max] = exchange_dz_bounds(*this, opts);
  for (int dz = dz_min; dz <= dz_max; ++dz)
    for (int dy = opts.dy_min(); dy <= opts.dy_max(); ++dy)
      for (int dx = opts.dx_min(); dx <= opts.dx_max(); ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) continue;
        if (opts.skip_corner() &&
            std::abs(dz) + std::abs(dy) + std::abs(dx) > 1)
          continue;
        fn(dy, dx, dz);
      }
}

bool LayoutImpl::_skips_direction(MeshBlockImpl const* pmb,
                                  std::tuple<int, int, int> offset) const {
  auto [dy, dx, dz] = offset;
  return pmb->options->is_physical_boundary(dy, dx, dz);
}

bool LayoutImpl::_in_phase(std::tuple<int, int, int> iloc, int nb,
                           SyncOptions const& opts) const {
  if (!opts.cross_panel_only() && !opts.intra_panel_only()) return true;
  bool same_panel = std::get<2>(iloc) == std::get<2>(loc_of(nb));
  return opts.cross_panel_only() ? !same_panel : same_panel;
}

LayoutOptionsImpl::LayoutOptionsImpl() {
  backend(get_env("BACKEND", default_backend()));

  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  auto process_rank_env = get_env("PROCESS_RANK", get_env("RANK", "0"));
  auto process_world_size_env =
      get_env("PROCESS_WORLD_SIZE", get_env("WORLD_SIZE", "1"));
  auto world_size_env = get_env("WORLD_SIZE", process_world_size_env);
  int process_world_size_value = std::stoi(process_world_size_env);
  int world_size_value = std::stoi(world_size_env);
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  auto master_port_env = std::getenv("MASTER_PORT");
  if (master_port_env) {
    master_port(std::stoi(master_port_env));
  } else {
    TORCH_CHECK(process_world_size_value == 1 && world_size_value == 1,
                "MASTER_PORT must be set for multi-process runs "
                "(PROCESS_WORLD_SIZE=",
                process_world_size_value, ", WORLD_SIZE=", world_size_value,
                ") so all ranks rendezvous on the same TCPStore");
    master_port(random_master_port());
  }
  process_rank(std::stoi(process_rank_env));
  rank(std::stoi(get_env("RANK", process_rank_env)));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  process_world_size(process_world_size_value);
  world_size(world_size_value);
  device_id(std::stoi(get_env("DEVICE_ID", "-1")));
  device(get_env("DEVICE", "cpu"));
}

LayoutOptions LayoutOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = LayoutOptionsImpl::create();
  auto config = YAML::LoadFile(filename);

  if (!config["distribute"]) return op;

  auto node = config["distribute"];
  check_keys(node, "distribute",
             {"layout", "nb1", "nb2", "nb3", "verbose", "blocks_per_process",
              "backend"});  // dead since #181; BACKEND env

  op->type() = node["layout"].as<std::string>("slab");
  op->py(node["nb3"].as<int>(1));
  op->px(node["nb2"].as<int>(1));
  op->pz(node["nb1"].as<int>(1));
  op->backend() = get_env("BACKEND", op->backend());
  op->device() = get_env("DEVICE", "cpu");
  op->verbose() = node["verbose"].as<bool>(verbose);

  if (op->verbose()) op->report(SINFO(LayoutOptions));

  return op;
}

std::shared_ptr<LayoutImpl> LayoutImpl::create(LayoutOptions const& options,
                                               MeshBlockImpl* p,
                                               std::string const& name) {
  (void)name;

  std::shared_ptr<LayoutImpl> pl;
  if (options->type() == "slab") {
    pl = std::make_shared<SlabLayoutImpl>(options, p);
  } else if (options->type() == "cubed") {
    pl = std::make_shared<CubedLayoutImpl>(options, p);
  } else if (options->type() == "cubed-sphere") {
    pl = std::make_shared<CubedSphereLayoutImpl>(options, p);
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", options->type());
  }

  if (p != nullptr) {
    std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
    g_local_layouts[{options->process_rank(),
                     options->local_block_index(options->rank())}] = pl.get();
  }

  return pl;
}

LayoutImpl::~LayoutImpl() {
  if (options == nullptr || owner() == nullptr) return;

  std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
  g_local_layouts.erase(
      {options->process_rank(), options->local_block_index(options->rank())});
}

void LayoutImpl::_prepare_local_exchange(MeshBlockImpl const* pmb,
                                         SyncOptions const& opts) {
  if (options->blocks_per_process() <= 1) return;

#ifdef USE_CUDA
  std::optional<c10::cuda::CUDAStream> current_stream;
  if (options->device() == "cuda") {
    current_stream.emplace(c10::cuda::getCurrentCUDAStream());
  }
#endif

  auto key = make_local_exchange_key(*this, opts);
  int expected = options->blocks_per_process();
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  auto& state = g_local_exchange_states[key];
  int generation = state.generation;

#ifdef USE_CUDA
  if (current_stream) {
    if (state.arrival_events.size() != expected) {
      state.arrival_events.resize(expected);
    }
    int local_block = options->local_block_index(options->rank());
    auto& arrival_event = state.arrival_events.at(local_block);
    if (!arrival_event) {
      arrival_event = std::make_shared<at::cuda::CUDAEvent>();
    }
    arrival_event->record(*current_stream);
    if (!state.completion_event) {
      state.completion_event = std::make_shared<at::cuda::CUDAEvent>();
    }
  }
#endif

  state.arrived += 1;
  if (state.arrived == expected) {
    auto layouts = local_layouts_for(*this);
    lock.unlock();
    std::exception_ptr error;
#ifdef USE_CUDA
    auto completion_event = state.completion_event;
#endif
    try {
#ifdef USE_CUDA
      if (current_stream) {
        for (auto const& event : state.arrival_events) {
          TORCH_CHECK(event != nullptr,
                      "missing CUDA arrival event for local exchange");
          event->block(*current_stream);
        }
      }
#endif
      _copy_local_exchange_buffers(layouts, opts);
#ifdef USE_CUDA
      if (current_stream && completion_event) {
        completion_event->record(*current_stream);
      }
#endif
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    state.error = error;
    state.ready = true;
    state.released = expected;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(
        lock, [&]() { return state.ready && state.generation == generation; });
  }

#ifdef USE_CUDA
  if (state.error == nullptr && state.completion_event && current_stream) {
    auto completion_event = state.completion_event;
    lock.unlock();
    completion_event->block(*current_stream);
    lock.lock();
  }
#endif

  if (state.error != nullptr) {
    auto error = state.error;
    state.released -= 1;
    if (state.released == 0) {
      state.arrived = 0;
      state.ready = false;
      state.error = nullptr;
      state.generation += 1;
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(
          lock, [&]() { return state.generation != generation; });
    }
    std::rethrow_exception(error);
  }

  state.released -= 1;
  if (state.released == 0) {
    state.arrived = 0;
    state.ready = false;
    state.error = nullptr;
    state.generation += 1;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(lock,
                             [&]() { return state.generation != generation; });
  }
}

std::tuple<int, int, int> LayoutImpl::_remap_exchange_offset(
    std::tuple<int, int, int> iloc, int dy, int dx, int dz) const {
  int dx_sgn = 1;
  int dy_sgn = 1;
  int dz_sgn = 1;

  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  if (options->periodic_z() && options->pz() == 2 && std::get<2>(iloc) == 0) {
    dz_sgn = -1;
  }

  return {dy_sgn * dy, dx_sgn * dx, dz_sgn * dz};
}

std::tuple<int, int, int> LayoutImpl::_peer_exchange_offset(
    int peer_rank, int target_rank, SyncOptions const& opts,
    std::tuple<int, int, int> offset) const {
  (void)peer_rank;
  (void)target_rank;
  (void)opts;
  auto [dy, dx, dz] = offset;
  return {-dy, -dx, -dz};
}

void LayoutImpl::_copy_local_exchange_buffers(
    std::vector<LayoutImpl*> const& layouts, SyncOptions const& opts) const {
  for (auto* layout : layouts) {
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);

    layout->for_each_direction(opts, [&](int dy_, int dx_, int dz_) {
      auto offset = layout->_remap_exchange_offset(iloc, dy_, dx_, dz_);
      int nb = layout->neighbor_rank(iloc, offset);
      if (nb < 0 || nb == rank) return;

      // this phase's peers only
      if (!layout->_in_phase(iloc, nb, opts)) return;
      if (layout->options->owner_process_rank(nb) !=
          layout->options->process_rank()) {
        return;
      }

      int bid = get_buffer_id(offset);
      auto peer = layouts.at(layout->options->local_block_index(nb));
      auto peer_offset = layout->_peer_exchange_offset(nb, rank, opts, offset);
      int peer_bid = get_buffer_id(peer_offset);

      auto& send_bufs = layout->owner()->send_bufs;
      auto& peer_recv_bufs = peer->owner()->recv_bufs;
      // a block that packed different variables than its peer would make
      // the copy below read past the peer's buffers
      TORCH_CHECK(peer_recv_bufs[peer_bid].size() == send_bufs[bid].size(),
                  "local exchange: block ", rank, " sends ",
                  send_bufs[bid].size(), " variables to block ", nb,
                  ", which expects ", peer_recv_bufs[peer_bid].size(),
                  "; every block must exchange the same variables");
      for (int n = 0; n < send_bufs[bid].size(); ++n) {
        TORCH_CHECK(
            peer_recv_bufs[peer_bid][n].numel() == send_bufs[bid][n].numel(),
            "local exchange size mismatch from rank ", rank, " to rank ", nb,
            " send_offset=(", std::get<0>(offset), ",", std::get<1>(offset),
            ",", std::get<2>(offset), ") recv_offset=(",
            std::get<0>(peer_offset), ",", std::get<1>(peer_offset), ",",
            std::get<2>(peer_offset),
            ") send_shape=", send_bufs[bid][n].sizes(),
            " recv_shape=", peer_recv_bufs[peer_bid][n].sizes());
        peer_recv_bufs[peer_bid][n].view({-1}).copy_(
            send_bufs[bid][n].reshape({-1}));
      }
    });
  }
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions const& opts) {
  if (options->verbose()) {
    SINFO(Layout) << "serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  auto& cache = pmb->exchange_buffer_cache[exchange_buffer_key(opts, vars)];
  cache.send.resize(num_exchange_buffers());
  cache.recv.resize(num_exchange_buffers());

  for_each_direction(opts, [&](int dy, int dx, int dz) {
    std::tuple<int, int, int> offset(dy, dx, dz);
    if (_skips_direction(pmb, offset)) return;
    int nb = neighbor_rank(iloc, offset);
    if (nb < 0) return;  // no neighbor

    // Get the interior part for this direction
    auto sub = pmb->part(offset, PartOptions().exterior(false));

    // Copy data from mesh to send buffer
    int bid = get_buffer_id(offset);
    int count = 0;
    auto& cached_send = cache.send[bid];
    auto& cached_recv = cache.recv[bid];
    cached_send.resize(vars.size());
    cached_recv.resize(vars.size());
    for (auto& [name, var] : vars) {
      auto source = var.index(sub);
      if (!buffer_matches(cached_send[count], source)) {
        cached_send[count] = torch::empty(source.sizes(), source.options());
      }
      cached_send[count].copy_(source);

      if (!buffer_matches(cached_recv[count], cached_send[count])) {
        cached_recv[count] = torch::empty(cached_send[count].sizes(),
                                          cached_send[count].options());
      }
      count++;
    }
    pmb->send_bufs[bid] = cached_send;
    pmb->recv_bufs[bid] = cached_recv;
  });
}

void LayoutImpl::launch_exchange(MeshBlockImpl const* pmb,
                                 SyncOptions const& opts,
                                 std::vector<CommWorkPtr>& works) {
  _prepare_local_exchange(pmb, opts);

  exchange_remote(pmb, opts, works);
}

void LayoutImpl::exchange_remote(MeshBlockImpl const* pmb,
                                 SyncOptions const& opts,
                                 std::vector<CommWorkPtr>& works) {
  TORCH_CHECK(owner() != nullptr,
              "[Layout:exchange_remote] layout has no owning MeshBlock");
  TORCH_CHECK(pmb != nullptr,
              "[Layout:exchange_remote] MeshBlock pointer is null");

  if (options->verbose()) {
    SINFO(Layout) << "performing communication\n";
  }

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  std::vector<RemoteExchangeOp> remote_ops;

  // the order of the first block is swapped for a periodic axis of two blocks
  // (_remap_exchange_offset); a cubed-sphere sync keeps this phase's peers
  for_each_direction(opts, [&](int dy_, int dx_, int dz_) {
    auto offset = _remap_exchange_offset(iloc, dy_, dx_, dz_);
    auto [dy, dx, dz] = offset;
    if (_skips_direction(pmb, offset)) return;
    int nb = neighbor_rank(iloc, offset);
    if (nb < 0) return;  // no neighbor
    if (!_in_phase(iloc, nb, opts)) return;

    int r = get_buffer_id(offset);
    int remote_process = options->owner_process_rank(nb);
    bool is_remote = remote_process != options->process_rank();

    if (is_remote) {
      int remote_local_block = options->local_block_index(nb);
      int local_block = options->local_block_index(rank);
      auto peer_offset = _peer_exchange_offset(nb, rank, opts, offset);
      int send_id =
          make_comm_tag(remote_local_block, peer_offset, opts.phyid());
      int recv_id = make_comm_tag(local_block, offset, opts.phyid());
      remote_ops.push_back({this, remote_process, local_block,
                            remote_local_block, r, send_id, recv_id, offset,
                            peer_offset});
    } else if (nb == rank) {  // self-send
      // the reversed direction is the receiving slot only on a flat layout;
      // across a cubed-sphere seam the peer's offset is rotated
      // (_peer_exchange_offset), so this copy would fill the wrong ghosts
      TORCH_CHECK(dynamic_cast<CubedSphereLayoutImpl const*>(this) == nullptr,
                  "exchange_remote: cubed-sphere block ", rank,
                  " is its own neighbour in direction (", dy, ",", dx, ",", dz,
                  "); a self-send across a panel seam is unsupported, since "
                  "its receiving slot is not the reversed direction");
      int r1 = get_buffer_id(std::tuple<int, int, int>(-dy, -dx, -dz));
      for (int n = 0; n < pmb->recv_bufs[r].size(); ++n)
        pmb->recv_bufs[r1][n].copy_(pmb->send_bufs[r][n]);
    }
  });

  if (remote_ops.empty()) return;
  TORCH_CHECK(has_process_group(),
              "[Layout:exchange_remote] remote communication requires an "
              "initialized process group");

  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  bool coalescing = comm->supports_coalescing();
  if (coalescing) {
    comm->start_coalescing();
  }
  for (auto const& op : remote_ops) {
    exchange_each_var(*comm, pmb->send_bufs[op.buffer_id],
                      pmb->recv_bufs[op.buffer_id], op.remote_process,
                      op.send_tag, op.recv_tag, works);
  }
  if (coalescing) {
    auto coalesced_work = comm->end_coalescing();
    if (coalesced_work) {
      works.push_back(coalesced_work);
    }
  }
}

void exchange_each_var(ProcessGroupContext const& comm,
                       std::vector<torch::Tensor>& sends,
                       std::vector<torch::Tensor>& recvs, int peer,
                       int send_tag, int recv_tag,
                       std::vector<CommWorkPtr>& works) {
  TORCH_CHECK(sends.size() <= 32768,
              "too many variables in one exchange: ", sends.size());
  for (int n = 0; n < sends.size(); ++n) {
    std::vector<torch::Tensor> send = {sends[n]}, recv = {recvs[n]};
    auto send_work = comm.send(send, peer, send_tag + n * 65536);
    if (send_work) works.push_back(send_work);
    auto recv_work = comm.recv(recv, peer, recv_tag + n * 65536);
    if (recv_work) works.push_back(recv_work);
  }
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars,
                             SyncOptions const& opts) const {
  if (options->verbose()) {
    SINFO(Layout) << "deserializing data from receive buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  for_each_direction(opts, [&](int dy, int dx, int dz) {
    std::tuple<int, int, int> offset(dy, dx, dz);
    if (_skips_direction(pmb, offset)) return;
    int nb = neighbor_rank(iloc, offset);
    if (nb < 0) return;  // no neighbor

    // Get the exterior (ghost zone) part for this direction
    auto sub = pmb->part(offset, PartOptions().exterior(true));

    // Copy data from receive buffer to mesh ghost zones
    int bid = get_buffer_id(offset);
    int count = 0;
    for (auto& [name, var] : vars) {
      var.index_put_(sub, pmb->recv_bufs[bid][count++]);
    }
  });
}

void LayoutImpl::fill_corners(MeshBlockImpl const* pmb, Variables& vars) const {
  auto sub_left = pmb->part({0, -1, 0}, PartOptions().exterior(true));
  auto sub_right = pmb->part({0, +1, 0}, PartOptions().exterior(true));
  auto sub_bot = pmb->part({-1, 0, 0}, PartOptions().exterior(true));
  auto sub_top = pmb->part({+1, 0, 0}, PartOptions().exterior(true));

  // Fill-in left-bot inter-panel corners
  std::tuple<int, int, int> corner(/*dy=*/-1, /*dx=*/-1, 0);
  auto sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_bot));
  }

  // Fill-in right-bot inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/-1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_bot));
  }

  // Fill-in left-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/-1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_top));
  }

  // Fill-in right-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_top));
  }
}

void LayoutImpl::finalize(MeshBlockImpl const* pmb, Variables& vars,
                          SyncOptions const& opts,
                          std::vector<CommWorkPtr>& works) {
  // Wait for all operations to complete
  for (auto& work : works) {
    if (work) {
      work->wait();
    }
  }
  // Deserialize received data into ghost zones
  deserialize(pmb, vars, opts);

  // Corners are synthesized from the edge strips, so neither half of a split
  // sync may do it; `exchange` calls fill_corners once, after both.
  if (opts.skip_corner() && !opts.cross_panel_only() &&
      !opts.intra_panel_only()) {
    fill_corners(pmb, vars);
  }

  // Completed point-to-point work is sufficient; a global barrier would
  // serialize otherwise independent exchanges.
  works.clear();
}

torch::Tensor LayoutImpl::gather_x1(torch::Tensor const& piece) {
  int pz = options->pz();
  if (pz <= 1) return piece;
  auto [rx, ry, rz] = loc_of(options->rank());
  int proc = options->process_rank();
  int bpp = std::max(1, options->blocks_per_process());
  int me = options->local_block_index(options->rank());
  std::vector<torch::Tensor> pieces(pz);
  pieces[rz] = piece.contiguous();
#ifdef USE_CUDA
  // the other blocks read this piece on their own streams
  if (piece.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
#endif

  // a remote piece is one contiguous tensor per message (Gloo)
  std::vector<int> local;
  std::vector<std::vector<torch::Tensor>> bufs;
  bufs.reserve(2 * pz);
  std::vector<CommWorkPtr> works;
  for (int k = 0; k < pz; ++k) {
    int nb = rank_of({rx, ry, k});
    int owner = options->owner_process_rank(nb);
    if (owner == proc) {
      local.push_back(k);
      continue;
    }
    TORCH_CHECK(has_process_group(),
                "[Layout:gather_x1] a remote column piece requires an "
                "initialized process group");
    int peer = options->local_block_index(nb);
    pieces[k] = torch::empty_like(pieces[rz]);  // blocks split x1 evenly
    std::lock_guard<std::mutex> lock(g_process_comm_mutex);
    bufs.push_back({pieces[rz]});  // tag: sender, receiver; above exchange tags
    auto send = comm->send(bufs.back(), owner, (1 << 24) + me * bpp + peer);
    if (send) works.push_back(send);
    bufs.push_back({pieces[k]});
    auto recv = comm->recv(bufs.back(), owner, (1 << 24) + peer * bpp + me);
    if (recv) works.push_back(recv);
  }

  int nlocal = local.size();
  if (nlocal > 1) {
    std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
    auto& board = g_column_boards[{proc, rx, ry}];
    g_local_exchange_cv.wait(lock, [&]() { return !board.full; });
    board.pieces[rz] = pieces[rz];
    if (++board.posted == nlocal) {
      board.full = true;
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(lock, [&]() { return board.full; });
    }
    for (int k : local) pieces[k] = board.pieces.at(k);
  }
  for (auto& work : works) work->wait();

  auto column = torch::cat(pieces, -1);
#ifdef USE_CUDA
  // done reading the other blocks' pieces before the board lets them go
  if (column.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  if (nlocal > 1) {
    std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
    auto& board = g_column_boards[{proc, rx, ry}];
    if (++board.taken == nlocal) {
      board = ColumnBoard();
      g_local_exchange_cv.notify_all();
    }
  }
  return column;
}

CommWorkPtr LayoutImpl::send_to_block(std::vector<torch::Tensor>& tensors,
                                      int block_rank, int tag) {
  TORCH_CHECK(has_process_group(),
              "[Layout:send_to_block] a remote block requires an initialized "
              "process group");
  int bpp = std::max(1, options->blocks_per_process());
  TORCH_CHECK(bpp <= 16,
              "[Layout:send_to_block] at most 16 blocks per process");
  int me = options->local_block_index(options->rank());
  int peer = options->local_block_index(block_rank);
  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  return comm->send(tensors, options->owner_process_rank(block_rank),
                    tag + (me * bpp + peer) * 0x10000);
}

CommWorkPtr LayoutImpl::recv_from_block(std::vector<torch::Tensor>& tensors,
                                        int block_rank, int tag) {
  TORCH_CHECK(has_process_group(),
              "[Layout:recv_from_block] a remote block requires an "
              "initialized process group");
  int bpp = std::max(1, options->blocks_per_process());
  TORCH_CHECK(bpp <= 16,
              "[Layout:recv_from_block] at most 16 blocks per process");
  int me = options->local_block_index(options->rank());
  int peer = options->local_block_index(block_rank);
  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  return comm->recv(tensors, options->owner_process_rank(block_rank),
                    tag + (peer * bpp + me) * 0x10000);
}

torch::Tensor LayoutImpl::take_x1_anchor(int above, torch::Tensor const& like,
                                         int tag) {
  if (above < 0) return torch::Tensor();
  if (!is_local_block(above)) {
    std::vector<torch::Tensor> rbuf = {torch::empty_like(like)};
    recv_from_block(rbuf, above, tag)->wait();
    return rbuf[0];
  }
  // the block above runs on another thread of this process (Mesh advances
  // local blocks together); a caller stepping them one by one must go top
  // down, or this would wait forever
  std::pair<int, int> key = {options->process_rank(), options->rank()};
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  bool posted = g_local_exchange_cv.wait_for(
      lock, std::chrono::minutes(5), [&]() { return g_x1_anchors.count(key); });
  TORCH_CHECK(posted, "[Layout:take_x1_anchor] block ", options->rank(),
              " waited 5 min for the x1 reference of block ", above,
              " above it in this process; local blocks must run concurrently "
              "or from the x1 top down");
  auto anchor = g_x1_anchors.at(key);
  g_x1_anchors.erase(key);
#ifdef USE_CUDA
  // written on the poster's stream, freed after use on this one
  if (anchor.is_cuda()) anchor.record_stream(c10::cuda::getCurrentCUDAStream());
#endif
  return anchor;
}

void LayoutImpl::pass_x1_anchor(int below, torch::Tensor const& face, int tag) {
  if (below < 0) return;
  if (!is_local_block(below)) {
    std::vector<torch::Tensor> sbuf = {face.contiguous()};
    send_to_block(sbuf, below, tag)->wait();
    return;
  }
  auto posted = face.contiguous();
#ifdef USE_CUDA
  // the block below reads this face on its own stream
  if (posted.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
  g_x1_anchors[{options->process_rank(), below}] = posted;
  g_local_exchange_cv.notify_all();
}

void LayoutImpl::post_to_local_block(int block_rank, torch::Tensor const& t,
                                     int tag) {
  TORCH_CHECK(is_local_block(block_rank), "[Layout:post_to_local_block] block ",
              block_rank, " is not in this process");
  auto posted = t.contiguous();
#ifdef USE_CUDA
  // the receiving block reads this tensor on its own stream
  if (posted.is_cuda()) c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  std::tuple<int, int, int, int> key = {options->process_rank(),
                                        options->rank(), block_rank, tag};
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  // one message per (sender, receiver, tag) in flight: wait for the previous
  // one to be taken
  bool free =
      g_local_exchange_cv.wait_for(lock, std::chrono::minutes(5),
                                   [&]() { return !g_local_mail.count(key); });
  TORCH_CHECK(free, "[Layout:post_to_local_block] block ", options->rank(),
              " waited 5 min for block ", block_rank,
              " to take its previous message, tag ", tag);
  g_local_mail[key] = posted;
  g_local_exchange_cv.notify_all();
}

torch::Tensor LayoutImpl::take_from_local_block(int block_rank, int tag) {
  TORCH_CHECK(is_local_block(block_rank),
              "[Layout:take_from_local_block] block ", block_rank,
              " is not in this process");
  std::tuple<int, int, int, int> key = {options->process_rank(), block_rank,
                                        options->rank(), tag};
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  bool posted = g_local_exchange_cv.wait_for(
      lock, std::chrono::minutes(5), [&]() { return g_local_mail.count(key); });
  TORCH_CHECK(posted, "[Layout:take_from_local_block] block ", options->rank(),
              " waited 5 min for block ", block_rank, ", tag ", tag,
              "; local blocks must run concurrently");
  auto t = g_local_mail.at(key);
  g_local_mail.erase(key);
  g_local_exchange_cv.notify_all();
#ifdef USE_CUDA
  // written on the poster's stream, freed after use on this one
  if (t.is_cuda()) t.record_stream(c10::cuda::getCurrentCUDAStream());
#endif
  return t;
}

void LayoutImpl::_init_process_group() {
  if (!use_process_group()) return;
  comm = ProcessGroupContext::create(options);
}

}  // namespace snap
