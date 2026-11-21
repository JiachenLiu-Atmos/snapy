#pragma once

// C/C++
#include <array>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/thermo/thermo.hpp>
#include <kintera/utils/format.hpp>

// yaml
#include <yaml-cpp/yaml.h>

// arg
#include <snap/add_arg.h>

#include "source_term.hpp"

namespace snap {

class HydroImpl;

////// (1) Constant Gravity //////

struct ConstGravityOptionsImpl {
  static std::shared_ptr<ConstGravityOptionsImpl> create() {
    return std::make_shared<ConstGravityOptionsImpl>();
  }
  static std::shared_ptr<ConstGravityOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  ConstGravityOptionsImpl() = default;
  std::shared_ptr<ConstGravityOptionsImpl> clone() const {
    return std::make_shared<ConstGravityOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "-- constant gravity options --\n";
    os << "* grav1 = " << grav1() << "\n"
       << "* grav2 = " << grav2() << "\n"
       << "* grav3 = " << grav3() << "\n"
       << "* non-hydrostatic = " << non_hydrostatic() << "\n"
       << "* gravity-work = " << gravity_work() << "\n";
  }

  ADD_ARG(double, non_hydrostatic) = 1.;
  //! x1 gravity work in the energy equation (#283):
  //!   "face" (default): face-mass-flux work in every cell (the form before
  //!     #283), plus the work of the corrected potential energy on grids with
  //!     a form for it (derivation curved-gravity-work-weight.md sec 7)
  //!   "cell": -g rho v at cell centres, the form the implicit matrix
  //!     linearises; mass moved by sedimentation or the positivity limiter
  //!     keeps its face-form booking
  ADD_ARG(std::string, gravity_work) = "face";
  //! removed in V3 (X5, gw1): must stay false; read so old decks still load
  ADD_ARG(bool, gravity_work_fixer) = false;
  ADD_ARG(double, grav1) = 0.;
  ADD_ARG(double, grav2) = 0.;
  ADD_ARG(double, grav3) = 0.;
};
using ConstGravityOptions = std::shared_ptr<ConstGravityOptionsImpl>;

class ConstGravityImpl : public torch::nn::Cloneable<ConstGravityImpl> {
 public:
  //! options with which this `ConstGravity` was constructed
  ConstGravityOptions options;

  // Constructor to initialize the layers
  ConstGravityImpl() : options(ConstGravityOptionsImpl::create()) {}
  explicit ConstGravityImpl(ConstGravityOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward_split(torch::Tensor du, torch::Tensor w, double dt,
                              torch::Tensor rho1);
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(ConstGravity);

//////// lane-A forcing options (src/options/forcing_*.yaml) ////////

//! create() and clone(); members, report() and reader are generated
template <class T>
struct ForcingOptions {
  static std::shared_ptr<T> create() { return std::make_shared<T>(); }
  std::shared_ptr<T> clone() const {
    return std::make_shared<T>(static_cast<T const&>(*this));
  }
};

//! forcing/<key> through its generated reader; nullptr without the key
template <class T>
std::shared_ptr<T> read_forcing(YAML::Node const& forcing, char const* key,
                                void (*read)(T&, YAML::Node const&,
                                             std::string const&)) {
  if (!forcing[key]) return nullptr;
  auto op = T::create();
  read(*op, forcing[key], std::string("forcing/") + key);
  return op;
}

#define SNAP_FORCING_OPTIONS(T) \
  struct T##Impl;               \
  using T = std::shared_ptr<T##Impl>
SNAP_FORCING_OPTIONS(CoriolisOptions);
SNAP_FORCING_OPTIONS(DiffusionOptions);
SNAP_FORCING_OPTIONS(BodyHeatOptions);
SNAP_FORCING_OPTIONS(TopCoolOptions);
SNAP_FORCING_OPTIONS(BotHeatOptions);
SNAP_FORCING_OPTIONS(RelaxBotCompOptions);
SNAP_FORCING_OPTIONS(RelaxBotTempOptions);
SNAP_FORCING_OPTIONS(RelaxBotVeloOptions);
SNAP_FORCING_OPTIONS(TopSpongeLyrOptions);
SNAP_FORCING_OPTIONS(BotSpongeLyrOptions);
SNAP_FORCING_OPTIONS(PlumeForcingOptions);
#undef SNAP_FORCING_OPTIONS

struct CoriolisOptionsImpl : ForcingOptions<CoriolisOptionsImpl> {
  static CoriolisOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/coriolis_members.inc>
};

struct DiffusionOptionsImpl : ForcingOptions<DiffusionOptionsImpl> {
  static DiffusionOptions from_yaml(YAML::Node const& forcing);
  void report(std::ostream& os) const;
#include <snap_options/diffusion_members.inc>
  //! x1 profiles multiplying the kinematic nu_iso/kappa_iso, one value per x1
  //! cell centre of the block (ghosts included), set before it is built;
  //! refused when x1 is split into several blocks (give a table instead)
  ADD_ARG(torch::Tensor, nu_scale_x1);
  ADD_ARG(torch::Tensor, kappa_scale_x1);
  //! the same as a table (2, n): x1 knots, scale; interpolated onto the cells
  ADD_ARG(torch::Tensor, nu_scale_x1_table);
  ADD_ARG(torch::Tensor, kappa_scale_x1_table);
};

struct BodyHeatOptionsImpl : ForcingOptions<BodyHeatOptionsImpl> {
  static BodyHeatOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/body_heat_members.inc>
};

struct TopCoolOptionsImpl : ForcingOptions<TopCoolOptionsImpl> {
  static TopCoolOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/top_cool_members.inc>
};

struct BotHeatOptionsImpl : ForcingOptions<BotHeatOptionsImpl> {
  static BotHeatOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/bot_heat_members.inc>
};

struct RelaxBotCompOptionsImpl : ForcingOptions<RelaxBotCompOptionsImpl> {
  static RelaxBotCompOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/relax_bot_comp_members.inc>
};

struct RelaxBotTempOptionsImpl : ForcingOptions<RelaxBotTempOptionsImpl> {
  static RelaxBotTempOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/relax_bot_temp_members.inc>
};

struct RelaxBotVeloOptionsImpl : ForcingOptions<RelaxBotVeloOptionsImpl> {
  static RelaxBotVeloOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/relax_bot_velo_members.inc>
};

struct TopSpongeLyrOptionsImpl : ForcingOptions<TopSpongeLyrOptionsImpl> {
  static TopSpongeLyrOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/top_sponge_lyr_members.inc>
};

struct BotSpongeLyrOptionsImpl : ForcingOptions<BotSpongeLyrOptionsImpl> {
  static BotSpongeLyrOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/bot_sponge_lyr_members.inc>
};

struct PlumeForcingOptionsImpl : ForcingOptions<PlumeForcingOptionsImpl> {
  static PlumeForcingOptions from_yaml(YAML::Node const& forcing);
#include <snap_options/plume_forcing_members.inc>
};

//////// the lane-A sources (A.rst 3.6) ////////

//! a source that is also a torch module of its Hydro (buffers follow
//! to(device)); the base holds the options and the parent (non-owning)
template <class T, class O>
class SourceModule : public torch::nn::Cloneable<T>, public SourceTerm {
 public:
  O options;
  HydroImpl const* phydro = nullptr;

  SourceModule() = default;
  SourceModule(O op, HydroImpl const* p) : options(std::move(op)), phydro(p) {}
  void reset() override {}
};

//! one Coriolis force -2 Omega x m (A.rst 3.4), type xyz or 123; it runs
//! before the implicit solve and, under the x1 mass-flux gate, after it
class CoriolisImpl : public SourceModule<CoriolisImpl, CoriolisOptions> {
 public:
  torch::Tensor omega1, omega2, omega3, alpha, beta;
  torch::Tensor contra_to_spherical, spherical_to_covariant;
  int face_id = -1;
  bool cubed_sphere = false, shallow_water = false, traditional = false;

  CoriolisImpl() = default;
  CoriolisImpl(CoriolisOptions op, HydroImpl const* p);
  void reset() override;
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(Coriolis);

class DiffusionImpl : public SourceModule<DiffusionImpl, DiffusionOptions> {
 public:
  DiffusionImpl() = default;
  DiffusionImpl(DiffusionOptions op, HydroImpl const* p);
  void reset() override;
  void add(torch::Tensor du, StageState const& s) override;
  double max_time_step(StageState const& s) const override;
  //! cubed sphere: the energy row of the diffusion face fluxes goes to the
  //! register as "diffusion" (A.rst 3.7); the momentum rows stay in add()
  void post_fluxes(FluxRegister& reg, StageState const& s) override;
  //! radial conduction over dt; returns its x1 face heat flux density (the
  //! register's [nvar, faces...] layout, row IPR), undefined when off
  torch::Tensor radial_conduction_step(torch::Tensor u, double dt);
  void begin_split(torch::Tensor u, double dt) override;
  void finish_split(torch::Tensor u, double dt) override;
  //! the same, posting each half's x1 heat flux with
  //! post_split(1, "diffusion.radial", flux, 0.5, dt) (A.rst 3.7)
  void begin_split(torch::Tensor u, double dt, FluxRegister& reg) override;
  void finish_split(torch::Tensor u, double dt, FluxRegister& reg) override;
  void restore_split(torch::Tensor u) const override;

 private:
  torch::Tensor forward_cubed_sphere(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt);
  //! the cubed-sphere face fluxes (rows IDN..IPR), seams shared
  std::array<torch::Tensor, 3> cubed_sphere_fluxes(torch::Tensor w,
                                                   torch::Tensor temp);
  //! this stage's fluxes whose energy rows post_fluxes put in the register;
  //! add() takes their other rows and clears them
  std::array<torch::Tensor, 3> posted_;
  torch::Tensor split_initial_;
  //! the x1 profiles in w's device and dtype, so a profile of ones changes
  //! no bit
  void profiles_on(torch::Tensor const& w);
  //! refuses a profile changed since reset; every entry point that reads the
  //! cached profiles (forward, max_time_step) calls it first
  void check_profiles() const;

  //! the x1 profiles on this block's cells (float64, CPU), their maxima, and
  //! copies in the state's device and dtype
  torch::Tensor nu_scale_, kappa_scale_, nu_scale_w_, kappa_scale_w_;
  double nu_scale_max_ = 1., kappa_scale_max_ = 1.;
  //! the profile options reset took, their versions and a copy of their
  //! values, to refuse a change
  std::array<torch::Tensor, 4> profile_options_, profile_values_;
  std::array<int64_t, 4> profile_versions_{};
};
TORCH_MODULE(Diffusion);

class BodyHeatImpl : public SourceModule<BodyHeatImpl, BodyHeatOptions> {
 public:
  BodyHeatImpl() = default;
  BodyHeatImpl(BodyHeatOptions op, HydroImpl const* p);
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(BodyHeat);

//! top-cool and bot-heat: a heat flux into a layer at one x1 wall
class LayerHeatImpl : public SourceModule<LayerHeatImpl, TopCoolOptions> {
 public:
  LayerHeatImpl() = default;
  LayerHeatImpl(TopCoolOptions op, HydroImpl const* p)
      : LayerHeatImpl(std::move(op), p, true) {}
  LayerHeatImpl(BotHeatOptions const& op, HydroImpl const* p);
  void reset() override;
  void add(torch::Tensor du, StageState const& s) override;

 private:
  LayerHeatImpl(TopCoolOptions op, HydroImpl const* p, bool top);
  bool top_ = true;
};
TORCH_MODULE(LayerHeat);

//! top-sponge-lyr and bot-sponge-lyr: a velocity sponge at one x1 wall
class SpongeImpl : public SourceModule<SpongeImpl, TopSpongeLyrOptions> {
 public:
  SpongeImpl() = default;
  SpongeImpl(TopSpongeLyrOptions op, HydroImpl const* p)
      : SpongeImpl(std::move(op), p, true) {}
  SpongeImpl(BotSpongeLyrOptions const& op, HydroImpl const* p);
  void add(torch::Tensor du, StageState const& s) override;

 private:
  SpongeImpl(TopSpongeLyrOptions op, HydroImpl const* p, bool top);
  bool top_ = true;
};
TORCH_MODULE(Sponge);

//! the relax-bot-* sources: the first interior x1 layer at the lower wall
template <class T, class O>
class BottomSlab : public SourceModule<T, O> {
 public:
  BottomSlab() = default;
  BottomSlab(O op, HydroImpl const* p, char const* name, uint32_t rows)
      : SourceModule<T, O>(std::move(op), p) {
    this->decl_ = {name, SourcePhase::pre_implicit, false, SourceKind::physical,
                   rows, SourceRegion::bottom};
  }
};

class RelaxBotCompImpl
    : public BottomSlab<RelaxBotCompImpl, RelaxBotCompOptions> {
 public:
  kintera::ThermoY pthermo_y = nullptr;
  kintera::ThermoX pthermo_x = nullptr;
  //! configured species indices in thermodynamic ordering
  std::vector<int64_t> species_ids;

  RelaxBotCompImpl() = default;
  RelaxBotCompImpl(RelaxBotCompOptions op, HydroImpl const* p);
  void reset() override;
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(RelaxBotComp);

class RelaxBotTempImpl
    : public BottomSlab<RelaxBotTempImpl, RelaxBotTempOptions> {
 public:
  RelaxBotTempImpl() = default;
  RelaxBotTempImpl(RelaxBotTempOptions op, HydroImpl const* p)
      : BottomSlab(std::move(op), p, "relax-bot-temp", kEnergy) {}
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(RelaxBotTemp);

class RelaxBotVeloImpl
    : public BottomSlab<RelaxBotVeloImpl, RelaxBotVeloOptions> {
 public:
  RelaxBotVeloImpl() = default;
  RelaxBotVeloImpl(RelaxBotVeloOptions op, HydroImpl const* p)
      : BottomSlab(std::move(op), p, "relax-bot-velo", kMomentum) {}
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(RelaxBotVelo);

//! plume-eos only; primitive variables w = [R, W, B, V]
class PlumeForcingImpl
    : public SourceModule<PlumeForcingImpl, PlumeForcingOptions> {
 public:
  PlumeForcingImpl() = default;
  explicit PlumeForcingImpl(PlumeForcingOptions op)
      : SourceModule(std::move(op), nullptr) {
    decl_ = {"plume-forcing", SourcePhase::pre_implicit, false,
             SourceKind::physical, kMass | kMomentum | kEnergy};
  }
  void add(torch::Tensor du, StageState const& s) override;
};
TORCH_MODULE(PlumeForcing);

}  // namespace snap

#undef ADD_ARG
