#pragma once

// C/C++
#include <array>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>
#include <torch/script.h>

// harp
#include <harp/integrator/integrator.hpp>

// snap
#include <snap/bc/bc_func.hpp>
#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/eos/ef_status.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/precision_policy.hpp>
#include <snap/mesh/stage_executor.hpp>
#include <snap/mesh/state_store.hpp>
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace kintera {
class ThermoYImpl;
struct KineticsOptionsImpl;
}  // namespace kintera

namespace snap {

struct OutputOptionsImpl;
using OutputOptions = std::shared_ptr<OutputOptionsImpl>;

//! \brief  container for parameters to initialize a MeshBlock
/*!
 * This struct holds all the options required to initialize a MeshBlock.
 * It can be initialized from a YAML input file using the `from_yaml` method,
 * or by setting the individual options manually.
 */
struct MeshBlockOptionsImpl {
  static std::shared_ptr<MeshBlockOptionsImpl> create() {
    return std::make_shared<MeshBlockOptionsImpl>();
  }
  static std::shared_ptr<MeshBlockOptionsImpl> from_yaml(std::string input_file,
                                                         bool verbose = false);

  MeshBlockOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- meshblock options --\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n"
       << "* basename = " << basename() << "\n";
  }

  bool is_physical_boundary(int dy, int dx, int dz) const;

  //! true if the face carries a physical wall, i.e. a boundary function whose
  //! ghost is NOT a real neighbour state. A periodic boundary function (a block
  //! wrapping onto itself) counts as physical for is_physical_boundary but is
  //! not a wall; with the layout's periodic x1 (periodic_z) the x1 boundary
  //! functions are nulled (meshblock.cpp), so is_physical_boundary is false on
  //! both x1 faces there, as at an internal x1 seam.
  bool is_wall_boundary(int dy, int dx, int dz) const;

  //! the boundary half of FaceKind (the layout's half is Exchange::kind):
  //! interior with no boundary function, else periodic, wall (reflecting,
  //! fixed temperature), outflow or custom; an unnamed function is custom
  FaceKind face_kind(int dy, int dx, int dz) const;

  //! Replace every boundary function at once. The names describe the functions
  //! they were parsed with, so this invalidates all of them; set bcnames
  //! afterwards to opt a face back into wall treatment.
  //! (Declared here, not beside bfuncs: ADD_ARG leaves the class private.)
  MeshBlockOptionsImpl& set_bfuncs(std::vector<bcfunc_t> const& funcs);

  std::string device_str() const;

  //! verbose
  ADD_ARG(bool, verbose) = false;

  //! integration.execution: eager (default) or cuda-graph; with
  //! integration.execution-strict a case that cannot be captured is refused
  //! (V3 X3; both names provisional, DESIGN.rst "pending Cheng's naming")
  ADD_ARG(std::string, execution) = "eager";
  ADD_ARG(bool, execution_strict) = false;

  //! output
  ADD_ARG(std::string, basename) = "";
  ADD_ARG(std::string, output_dir) = ".";
  ADD_ARG(std::vector<OutputOptions>, outputs);

  //! submodule options
  ADD_ARG(harp::IntegratorOptions, intg) = nullptr;
  ADD_ARG(CoordinateOptions, coord) = nullptr;
  ADD_ARG(HydroOptions, hydro) = nullptr;
  ADD_ARG(ScalarOptions, scalar) = nullptr;
  ADD_ARG(InternalBoundaryOptions, ib) = nullptr;

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! name of each installed boundary function, parallel to bfuncs
  ADD_ARG(std::vector<std::string>, bcnames);

  //! distribution layout
  ADD_ARG(LayoutOptions, layout) = nullptr;
};
using MeshBlockOptions = std::shared_ptr<MeshBlockOptionsImpl>;

using Variables = std::map<std::string, torch::Tensor>;
class OutputType;

struct ExchangeBufferSet {
  std::vector<std::vector<torch::Tensor>> send;
  std::vector<std::vector<torch::Tensor>> recv;
  std::vector<std::vector<torch::Tensor>> work;
};

struct PartOptions {
  //! if true, return the exterior part (with ghost zones);
  //! if false, return the interior part (without ghost zones)
  ADD_ARG(bool, exterior) = true;
  ADD_ARG(int, extend_x1) = 0;
  ADD_ARG(int, extend_x2) = 0;
  ADD_ARG(int, extend_x3) = 0;
  ADD_ARG(int, depth) = 99;
  ADD_ARG(int, ndim) = 4;
};

class StageExecutor;
class ChemistrySource;

class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
  friend class StageExecutor;

 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! user output
  std::function<Variables(Variables const&)> user_output_callback;

  //! immutable TorchScript forcings applied in registration order
  std::vector<std::shared_ptr<torch::jit::Module>> user_stage_forcings;

  void set_user_stage_forcings(std::vector<std::string> const& filenames);

  //! outputs
  std::vector<std::shared_ptr<OutputType>> output_types;

  //! current cycle number
  int cycle = 0;

  //! the length of the step being taken, set at its stage 0; the budget
  //! output observes it once the step is accepted
  double step_dt = 0.;

  //! Exchange buffers owned by the MeshBlock so Layout stays stateless.
  mutable std::vector<std::vector<torch::Tensor>> send_bufs, recv_bufs;
  mutable std::map<std::string, ExchangeBufferSet> exchange_buffer_cache;

  //! submodules
  harp::Integrator pintg = nullptr;
  Coordinate pcoord = nullptr;
  InternalBoundary pib = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  //! dtypes per quantity, built in reset() from dynamics.precision
  PrecisionPolicy policy;

  //! counts the state store's address changes (reset, to, restart, adopt)
  int state_generation() const { return _state.generation(); }

  //! the block's EOS/floor/positivity status: limiter marks, vapor-repair
  //! failures, heat-capacity latch and the positivity meters (V3 EF D4).
  //! Created before hydro and scalar, which register their meters in it.
  std::shared_ptr<EfStatus> pefs;

  Layout get_layout() const { return _playout; }

  //! Constructor to initialize the layers
  MeshBlockImpl() : options(MeshBlockOptionsImpl::create()) {}
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  ~MeshBlockImpl() override;
  void reset() override;
  //! An explicit precision policy owns dtypes; Module::to still moves devices.
  using torch::nn::Module::to;
  void to(torch::Device device, torch::Dtype dtype,
          bool non_blocking = false) override;
  void to(torch::Dtype dtype, bool non_blocking = false) override;
  void to(torch::Device device, bool non_blocking = false) override;

  //! index of part of the block; offset (x3, x2, x1) in {-1, 0, 1} each
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, PartOptions const& opts) const;

  //! initialize the variables; returns the initial simulation time
  double initialize(Variables& vars, char const* restart_file = nullptr);
  void initialize_local(Variables& vars);
  bool has_radiating_boundary() const;
  void apply_boundaries(Variables& vars, torch::Tensor hydro,
                        torch::Tensor tracers = {}, bool primitive = false);

  //! Mesh-owned initialization path that preserves multi-block exchange order.
  void initialize_under_mesh(Variables& vars);
  void finalize_initialization(Variables& vars);

  //! the maximum allowable time step
  double max_time_step(Variables const& vars);
  double local_max_time_step(Variables const& vars) const;
  //! the global dt from a local one: MIN over the ranks, cfl, 2^-redo
  double global_time_step(double dt_local) const;

  //! advance the variables by one stage of the integrator
  void forward(Variables& vars, double dt, int stage);
  void advance_local(Variables& vars, double dt, int stage);
  //! the same with a StepDt: a captured stage passes the executor's device
  //! dt (V3 D), which its kernels read at every replay
  void advance_stage(Variables& vars, StepDt dt, int stage);

  //! V3 S-K4 (K.rst 3.6): the chemistry step after the last stage, from a
  //! by-value copy of the reaction options; one adapter for this block,
  //! built on the calling thread. A Mesh registers its blocks through
  //! Mesh::set_chemistry, which defers each block's run to its global phase.
  void set_chemistry(kintera::KineticsOptionsImpl reactions);
  ChemistrySource* chemistry() const { return _chemistry.get(); }
  bool defer_chemistry = false;

  //! the stage executor's counters: captures, replays, eager stages, graphs
  std::map<std::string, int64_t> execution_report() const;
  //! this block's row of the capability matrix for vars (V3 X7)
  Capability execution_capability(Variables const& vars) const;
  //! an option or prescribed-mask change a captured stage depends on
  void bump_config_generation();

  //! build the anomaly reference (hydro.ref) from the initial hydro_w;
  //! called by finalize_initialization and by a restart without one
  void build_hydro_reference(Variables& vars);

  //! conserve-axial-momentum exact: this block's {R, I} of the stage
  //! (device, FP64; zeros if off) and the stage's solid-body increment from
  //! the global sums (lane A's AxialClosureFixer), applied in the global
  //! phase (forward, Mesh::forward)
  torch::Tensor axial_closure_sums() const;
  void apply_axial_closure(Variables& vars, double dt, int stage,
                           torch::Tensor const& global);
  void exchange(Variables& vars, SyncOptions const& opts) const;

  //! Serialize state into MeshBlock-owned buffers without launching comms yet.
  void begin_exchange(Variables& vars, SyncOptions const& opts) const;

  //! Launch remote exchange work after begin_exchange prepared the buffers.
  void launch_exchange(SyncOptions const& opts,
                       std::vector<CommWorkPtr>& works) const;

  //! Wait on launched work and deserialize results back into the variables.
  void finalize_exchange(Variables& vars, SyncOptions const& opts,
                         std::vector<CommWorkPtr>& works) const;
  void exchange_ghost_zones(Variables& vars);

  //! write the outputs due at current_time (all of them if final_write)
  void make_outputs(Variables const& vars, double current_time,
                    bool final_write = false);

  //! print cycle info
  void print_cycle_info(Variables const& vars, double time, double dt) const;

  //! make final output and print diagnostics
  //! \return 0, or 1 if the run terminated abnormally
  int finalize(Variables const& vars, double time);
  //! the end of a run for one block or a Mesh's blocks: the termination
  //! report, buffers released, the process group shut down; returns 1 if the
  //! run ended abnormally
  static int finish_run(std::vector<MeshBlockImpl*> const& blocks, double time);

  //! > 0: redo the step; 0: accepted; < 0: terminate
  int check_redo(Variables& vars);

  //! this block's retry causes (one host read; no reduction)
  std::array<bool, 7> local_redo_flags(Variables const& vars);
  //! Device flags (six causes plus vapor/capacity failure latches), folded
  //! into the retry causes by read_redo_flags; built without a host read.
  //! Drains kintera's saturation-failure count
  torch::Tensor local_redo_flag_tensor(Variables const& vars);
  //! one device-to-host read of a local_redo_flag_tensor (or their max):
  //! slots 0-5 are causes 0-5, slots 6-7 cause 2, slot 8 (diffusion.radial,
  //! reserved in V3) cause 6
  static std::array<bool, 7> read_redo_flags(torch::Tensor const& flags);
  //! the one retry decision for a bare block or a Mesh's blocks: their flag
  //! tensors stacked, amax, one host read, one MAX allreduce; the causes bits
  static int decide_retry(std::vector<MeshBlockImpl*> const& blocks,
                          std::vector<Variables const*> const& vars);

  //! the EOS's kintera ThermoY (hydro.eos.thermo), or null
  std::shared_ptr<kintera::ThermoYImpl> thermo() const;

  //! roll back (causes != 0: 1 floor, 2 clamp, 4 limiter, 8 nan,
  //! 16 saturation) or accept it
  int apply_redo(Variables& vars, int causes);

 protected:
  //! initialize from a restart file; returns its last simulation time
  double _init_from_restart(Variables& vars, std::string fname);

 private:
  //! one communication round; `exchange` splits a subdivided cubed-sphere
  //! sync into two of these
  void _exchange_once(Variables& vars, SyncOptions const& opts) const;
  //! exchange vars[hydro] as `type`, then vars[scalar] as kScalar
  void exchange_state(Variables& vars, char const* hydro, int type,
                      char const* scalar);

  //! K4 (K.rst 3.4): the ThermoY active set at the step's start, taken at
  //! stage 0 of each attempt and restored on a retry, so a retry repeats
  //! the first attempt (reaction_set, nactive buffers of the interior shape)
  std::pair<torch::Tensor, torch::Tensor> _active_set0;
  //! S-K4: this block's chemistry adapter (null: none registered)
  std::shared_ptr<ChemistrySource> _chemistry;
  //! K.rst 3.5: the ThermoY prepared for the interior shape (idempotent) and
  //! _active_set0 made for it; at stage 0 and before a graph warm-up
  void prepare_thermo(torch::Tensor const& hydro_u);

  //! distribution layout
  Layout _playout;

  //! hydro_u, scalar_s, scalar_r and the RK backups in fixed storage
  StateStore _state;

  //! registers the flux register's tensors in _state (a moved one bumps
  //! the store generation)
  void register_flux_storage();

  //! runs each stage of forward() eagerly or as a captured graph (X3)
  std::shared_ptr<StageExecutor> _executor;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
