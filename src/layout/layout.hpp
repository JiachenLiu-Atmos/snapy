#pragma once

// C/C++
#include <torch/torch.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <tuple>

// snap
#include <snap/snap.h>

#include "connectivity.hpp"
#include "process_group.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {

/*!
 * \brief Calculate buffer ID from directional offsets
 *
 * Converts 3D directional offsets into a linear buffer index.
 * For 2D layouts (dz=0), returns index in range [0,8].
 * For 3D layouts, returns index in range [0,26].
 *
 * \return linear buffer index
 */
inline int get_buffer_id(std::tuple<int, int, int> offset) {
  auto [dy, dx, dz] = offset;
  return (dx % 3 + 3) % 3 + ((dy % 3 + 3) % 3) * 3 + ((dz % 3 + 3) % 3) * 9;
}

//! get environment variable with default
inline std::string get_env(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : def;
}

//! get global rank from environment variable
inline int get_rank() { return std::stoi(get_env("RANK", "0")); }

//! get local rank from environment variable
inline int get_local_rank() { return std::stoi(get_env("LOCAL_RANK", "0")); }

//! get process world size from environment variable
inline int get_world_size() { return std::stoi(get_env("WORLD_SIZE", "1")); }

struct LayoutOptionsImpl {
  static std::shared_ptr<LayoutOptionsImpl> create() {
    return std::make_shared<LayoutOptionsImpl>();
  }
  static std::shared_ptr<LayoutOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  LayoutOptionsImpl();

  void report(std::ostream& os) const {
    os << "-- layout options --\n";
    os << "* type = " << type() << "\n"
       << "* px = " << px() << "\n"
       << "* py = " << py() << "\n"
       << "* pz = " << pz() << "\n"
       << "* periodic_x = " << (periodic_x() ? "true" : "false") << "\n"
       << "* periodic_y = " << (periodic_y() ? "true" : "false") << "\n"
       << "* periodic_z = " << (periodic_z() ? "true" : "false") << "\n"
       << "* backend = " << backend() << "\n"
       << "* device = " << device() << "\n"
       << "* master_addr = " << master_addr() << "\n"
       << "* process_rank = " << process_rank() << "\n"
       << "* rank = " << rank() << "\n"
       << "* local_rank = " << local_rank() << "\n"
       << "* process_world_size = " << process_world_size() << "\n"
       << "* world_size = " << world_size() << "\n"
       << "* blocks_per_process = " << blocks_per_process() << "\n"
       << "* master_port = " << master_port() << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  int owner_process_rank(int block_rank) const {
    return block_rank / std::max(1, blocks_per_process());
  }

  int local_block_index(int block_rank) const {
    return block_rank % std::max(1, blocks_per_process());
  }

  int global_block_rank(int proc_rank, int local_block) const {
    return proc_rank * std::max(1, blocks_per_process()) + local_block;
  }

  int process_root_rank() const { return owner_process_rank(root_rank()); }

  //! type of layout
  ADD_ARG(std::string, type) = "slab";

  //! number of processors in X
  ADD_ARG(int, px) = 1;

  //! number of processors in Y
  ADD_ARG(int, py) = 1;

  //! number of processors in Z
  ADD_ARG(int, pz) = 1;

  //! periodicity in X
  ADD_ARG(bool, periodic_x) = false;

  //! periodicity in Y
  ADD_ARG(bool, periodic_y) = false;

  //! periodicity in Z
  ADD_ARG(bool, periodic_z) = false;

  ADD_ARG(std::string, backend) = "gloo";
  ADD_ARG(std::string, device) = "cpu";
  ADD_ARG(std::string, master_addr) = "127.0.0.1";
  ADD_ARG(int, process_rank) = 0;
  ADD_ARG(int, rank) = 0;
  ADD_ARG(int, root_rank) = 0;
  ADD_ARG(int, local_rank) = 0;
  ADD_ARG(int, process_world_size) = 1;
  ADD_ARG(int, world_size) = 1;
  ADD_ARG(int, blocks_per_process) = 1;
  ADD_ARG(int, master_port) = -1;
  ADD_ARG(int, device_id) = -1;
  ADD_ARG(bool, verbose) = false;
};
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

//! extra options for synchronization
struct SyncOptions {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };

  int dz_min() const { return dim() == DIM2 || dim() == DIM3 ? 0 : -1; }
  int dz_max() const { return dim() == DIM2 || dim() == DIM3 ? 0 : 1; }

  int dx_min() const { return dim() == DIM3 || dim() == DIM1 ? 0 : -1; }
  int dx_max() const { return dim() == DIM3 || dim() == DIM1 ? 0 : 1; }

  int dy_min() const { return dim() == DIM2 || dim() == DIM1 ? 0 : -1; }
  int dy_max() const { return dim() == DIM2 || dim() == DIM1 ? 0 : 1; }

  //! phase 1 of the two-phase cubed-sphere ghost sync; mirror of
  //! `cross_panel_only`
  ADD_ARG(bool, intra_panel_only) = false;
  ADD_ARG(bool, cross_panel_only) = false;
  ADD_ARG(bool, skip_corner) = true;
  ADD_ARG(bool, interpolate) = false;
  ADD_ARG(int, type) = kConserved;
  ADD_ARG(int, dim) = 0;
  //! 0 <= phyid < 64 (make_comm_tag)
  ADD_ARG(int, phyid) = 0;
};

using Variables = std::map<std::string, torch::Tensor>;

//! Cache key of the send/receive buffers of one exchange.
std::string exchange_buffer_key(SyncOptions const& opts, Variables const& vars);

//! A cached buffer can take `source`: same shape, dtype and device.
bool buffer_matches(torch::Tensor const& buffer, torch::Tensor const& source);

//! exchange_each_var adds n * 65536 for variable n, so the tag must stay in
//! [0, 65536) for the tags of different variables to stay distinct.
inline int make_comm_tag(int local_block_index,
                         std::tuple<int, int, int> offset, int phyid) {
  int tag = phyid * 1024 + local_block_index * 32 + get_buffer_id(offset);
  TORCH_CHECK(phyid >= 0 && tag >= 0 && tag < 65536, "comm tag ", tag,
              " (phyid ", phyid, ", local block ", local_block_index,
              ") is outside [0, 65536)");
  return tag;
}

//! Send, and post the receive of, each exchanged variable as its own message:
//! Gloo takes one tensor per send. Variable n adds n * 65536 to the tag.
void exchange_each_var(ProcessGroupContext const& comm,
                       std::vector<torch::Tensor>& sends,
                       std::vector<torch::Tensor>& recvs, int peer,
                       int send_tag, int recv_tag,
                       std::vector<CommWorkPtr>& works);

class MeshBlockImpl;

class LayoutImpl {
 public:
  static std::shared_ptr<LayoutImpl> create(LayoutOptions const& opts,
                                            MeshBlockImpl* p = nullptr,
                                            std::string const& name = "layout");

  //! communication
  std::shared_ptr<ProcessGroupContext> comm;

  //! options with which this `Layout` was constructed
  LayoutOptions options;

  LayoutImpl() : options(LayoutOptionsImpl::create()) {}
  LayoutImpl(const LayoutOptions& opts, MeshBlockImpl* owner = nullptr)
      : comm(nullptr), options(opts), _owner(owner) {
    int P = options->px() * options->py() * options->pz();
    _rankof.resize(P);
  }

  std::tuple<int, int, int> get_procs() const {
    return {options->px(), options->py(), options->pz()};
  }

  bool is_root() const { return options->rank() == options->root_rank(); }
  bool use_process_group() const {
    return owner() != nullptr && options->process_world_size() > 1;
  }
  bool has_process_group() const {
    return comm != nullptr && comm->initialized();
  }

  virtual ~LayoutImpl();

  //! Owning MeshBlock that provides exchange buffer storage for this layout.
  MeshBlockImpl* owner() const { return _owner; }

  //! Number of directional exchange slots required by this layout geometry.
  virtual int num_exchange_buffers() const { return 9; }

  //! Every neighbour direction of one exchange, as fn(dy, dx, dz) in dz, dy,
  //! dx order: the centre is left out, and with skip_corner every direction
  //! with more than one nonzero component; dz moves only with 27 buffers.
  void for_each_direction(SyncOptions const& opts,
                          std::function<void(int, int, int)> const& fn) const;

  virtual int rank_of(std::tuple<int, int, int> iloc) const {
    auto [rx, ry, rz] = iloc;

    int px = options->px();
    int py = options->py();
    int pz = options->pz();
    if (rx < 0 || rx >= px || ry < 0 || ry >= py || rz < 0 || rz >= pz)
      return -1;
    return _rankof[rz * (px * py) + ry * px + rx];
  }

  virtual std::tuple<int, int, int> loc_of(int rank) const { return {0, 0, 0}; }

  //! \brief Neighbor -> Z-order rank (3D)
  /*!
   * offset = (dx,dy,dz) <- {-1,0,1}. periodic flags control wrap;
   * otherwise off-domain -> -1.
   * iloc = (rx,ry,rz) are THIS rank's logical coords in the process grid (not
   * Morton code).
   */
  virtual int neighbor_rank(std::tuple<int, int, int> iloc,
                            std::tuple<int, int, int> offset) const {
    return -1;
  }

  //! Serialize variables
  virtual void serialize(MeshBlockImpl const* pmb, Variables& vars,
                         SyncOptions const& opts);

  //! Deserialize variables
  virtual void deserialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions const& opts) const;

  //! fill corners after exchange
  virtual void fill_corners(MeshBlockImpl const* pmb, Variables& vars) const;

  //! Launch only the communication phase after buffers have been serialized.
  void launch_exchange(MeshBlockImpl const* pmb, SyncOptions const& opts,
                       std::vector<CommWorkPtr>& works);

  virtual void exchange_remote(MeshBlockImpl const* pmb,
                               SyncOptions const& opts,
                               std::vector<CommWorkPtr>& works);

  void finalize(MeshBlockImpl const* pmb, Variables& vars,
                SyncOptions const& opts, std::vector<CommWorkPtr>& works);

  //! This block's x1 column: every block of the column passes its piece
  //! together; returns the pieces joined along x1 (last dim), bottom first.
  torch::Tensor gather_x1(torch::Tensor const& piece);

  //! Point-to-point messages with another block, local or remote process
  //! alike addressed by its block rank. The tag folds in both local block
  //! indices, so several blocks per process do not collide; with one block
  //! per process it is the tag given. The fold adds k * 65536 (k < 256): the
  //! low 16 bits stay the given tag, which for the 0x77xx hydro tags no
  //! make_comm_tag value has at <= 16 local blocks, and the total stays below
  //! gather_x1's 1 << 24.
  CommWorkPtr send_to_block(std::vector<torch::Tensor>& tensors, int block_rank,
                            int tag);
  CommWorkPtr recv_from_block(std::vector<torch::Tensor>& tensors,
                              int block_rank, int tag);

  //! The x1 reference relay, one column top->bottom (#254): the block
  //! `above` hands this block its running face value. Undefined when
  //! above < 0 (the column top anchors itself). A same-process neighbor
  //! goes through a board, a remote one through the process group.
  torch::Tensor take_x1_anchor(int above, torch::Tensor const& like, int tag);
  //! Hand this block's bottom face value to the block `below`.
  void pass_x1_anchor(int below, torch::Tensor const& face, int tag);

  //! A message to / from another block of this process (a board, not the
  //! process group). The blocks must run concurrently, as the mesh worker
  //! threads do. A post waits until the receiver took the previous message
  //! with the same tag; a take waits until the sender posted.
  void post_to_local_block(int block_rank, torch::Tensor const& t, int tag);
  torch::Tensor take_from_local_block(int block_rank, int tag);

  bool is_local_block(int block_rank) const {
    return options->owner_process_rank(block_rank) == options->process_rank();
  }

 protected:
  void _init_process_group();

  //! Coordinate local sibling blocks so same-process copies are visible first.
  void _prepare_local_exchange(MeshBlockImpl const* pmb,
                               SyncOptions const& opts);
  virtual std::tuple<int, int, int> _remap_exchange_offset(
      std::tuple<int, int, int> iloc, int dy, int dx, int dz = 0) const;
  virtual std::tuple<int, int, int> _peer_exchange_offset(
      int peer_rank, int target_rank, SyncOptions const& opts,
      std::tuple<int, int, int> offset) const;
  virtual void _copy_local_exchange_buffers(
      std::vector<LayoutImpl*> const& layouts, SyncOptions const& opts) const;
  //! The exchange skips this direction before asking for a neighbour: a
  //! physical boundary of the block.
  virtual bool _skips_direction(MeshBlockImpl const* pmb,
                                std::tuple<int, int, int> offset) const;
  //! Neighbour nb takes part in this exchange phase: a cubed-sphere sync runs
  //! an intra-panel and a cross-panel phase (loc_of's third index is the
  //! panel); without either flag every neighbour does.
  bool _in_phase(std::tuple<int, int, int> iloc, int nb,
                 SyncOptions const& opts) const;

  std::vector<Coord2> _coords2;
  std::vector<Coord3> _coords3;
  std::vector<int> _rankof;
  MeshBlockImpl* _owner = nullptr;
};
using Layout = std::shared_ptr<LayoutImpl>;

class SlabLayoutImpl : public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  SlabLayoutImpl() = default;
  SlabLayoutImpl(const LayoutOptions& opts, MeshBlockImpl* owner = nullptr)
      : LayoutImpl(opts, owner) {
    options->type("slab");
    _initialize();
  }

  ~SlabLayoutImpl() = default;
  void pretty_print(std::ostream& os) const;

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;

 private:
  //! Constructor-time setup that replaces the old torch-style reset() hook.
  void _initialize();
};
class CubedLayoutImpl : public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedLayoutImpl() = default;
  CubedLayoutImpl(const LayoutOptions& opts, MeshBlockImpl* owner = nullptr)
      : LayoutImpl(opts, owner) {
    options->type("cubed");
    _initialize();
  }

  ~CubedLayoutImpl() = default;
  void pretty_print(std::ostream& os) const;
  int num_exchange_buffers() const override { return 27; }

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;

 private:
  //! Constructor-time setup that replaces the old torch-style reset() hook.
  void _initialize();
};

}  // namespace snap

#undef ADD_ARG
