#pragma once

// C/C++
#include <utility>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// arg
#include <snap/add_arg.h>

#include "ef_status.hpp"

namespace snap {

struct EquationOfStateOptionsImpl {
  static std::shared_ptr<EquationOfStateOptionsImpl> create() {
    return std::make_shared<EquationOfStateOptionsImpl>();
  }
  static std::shared_ptr<EquationOfStateOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  EquationOfStateOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- equation of state options --\n";
    os << "* type = " << type() << "\n"
       << "* gammad = " << gammad() << "\n"
       << "* weight = " << weight() << "\n"
       << "* density_floor = " << density_floor() << "\n"
       << "* pressure_floor = " << pressure_floor() << "\n"
       << "* temperature_floor = " << temperature_floor() << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n"
       << "* limiter = " << (limiter() ? "true" : "false") << "\n"
       << "* eos_file = " << eos_file() << "\n";
    if (thermo()) {
      os << "-- thermo options --\n";
      thermo()->report(os);
    }
  }

  ADD_ARG(std::string, type) = "moist-mixture";
  ADD_ARG(double, gammad) = 1.4;     // ratio of specific heats (cp/cv)
  ADD_ARG(double, weight) = 29.e-3;  // mean molecular weight in kg/mol

  //! floors (SI): the limiter clamps to them, and the outflow boundary and
  //! check_redo read them with the limiter off too. One default, the same
  //! whether the options come from create() or from a deck without the key.
  ADD_ARG(double, density_floor) = 1.e-6;
  ADD_ARG(double, pressure_floor) = 1.e-3;
  ADD_ARG(double, temperature_floor) = 20.;
  ADD_ARG(bool, limiter) = false;
  ADD_ARG(std::string, eos_file) = "";
  ADD_ARG(bool, verbose) = false;

  //! submodules options
  ADD_ARG(kintera::ThermoOptions, thermo) = nullptr;
};
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

class HydroImpl;

class EquationOfStateImpl {
 public:
  //! Create and register an `EquationOfState` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `EquationOfState` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `EquationOfState` module
   */
  static std::shared_ptr<EquationOfStateImpl> create(
      EquationOfStateOptions const& opts, torch::nn::Module* p,
      std::string const& name = "eos");

  //! options with which this `EquationOfState` was constructed
  EquationOfStateOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  EquationOfStateImpl() : options(EquationOfStateOptionsImpl::create()) {}
  explicit EquationOfStateImpl(EquationOfStateOptions const& options_,
                               torch::nn::Module* p = nullptr);
  virtual ~EquationOfStateImpl() = default;

  virtual int nvar() const { return 5; }

  //! \brief Return the molecular weight of species \p n.
  //!
  //! \param[in] n Index of the species for which to return the molecular
  //!              weight (defaults to 0).
  //! \return Molecular weight of the requested species, in kg/mol (or the
  //!         units consistent with the underlying thermodynamic model).
  virtual double species_weight(int n = 0) const { return 0.; }

  //! dry gas constant Rgas / weight, in J/(kg K)
  double gas_constant_dry() const;

  //! \brief Return the reference specific heat at constant volume of species \p
  //! n.
  //!
  //! \param[in] n Index of the species for which to return the reference
  //!              specific heat (defaults to 0).
  //! \return Reference specific heat at constant volume for the requested
  //!         species, in J/(kg·K) (or the units consistent with the underlying
  //!         thermodynamic model).
  virtual double species_cv_ref(int n = 0) const { return 0.; }

  //! \brief Return the local mixture specific heat at constant volume.
  //!
  //! \param[in] prim Primitive hydrodynamic variables.
  //! \param[in] temp Temperature.
  //! \return Specific heat at constant volume in J/(kg·K), or units
  //!         consistent with the underlying thermodynamic model.
  virtual torch::Tensor specific_heat_cv(torch::Tensor prim,
                                         torch::Tensor temp);

  //! \brief Return the additive internal-energy offset carried by conserved
  //! constituent densities.
  virtual torch::Tensor internal_energy_offset(torch::Tensor hydro_like) const;

  //! \brief Energy per unit mass each species carries through a face: its
  //! internal and kinetic energy plus its partial pressure over its density.
  //! \return (ny, nc3, nc2, nc1); undefined when the EOS has no such split.
  virtual torch::Tensor species_enthalpy(torch::Tensor prim) { return {}; }

  //! \brief Computes hydrodynamic variables from the given abbreviation
  /*!
   * These five abbreviations should be supported:
   *  - "W->U": convert primitive variables to conserved variables
   *  - "U->W": convert conserved variables to primitive variables
   *
   * Under anomaly storage (fp32 roadmap Step 1; ideal-moist only) "U->W" and
   * "W->U" keep meaning totals; "U'->W" and "W->U'" take or give the stored
   * representation, whose rows {IDN, IPR} are anomalies about hydro.ref.
   * `forward` converts the stored representation.
   *  - "WA->L": compute sound speed from primitive variables and adiabatic
   *  - "W->A": compute adiabatic index from conserved variables
   *  - "W->T": compute temperature
   *
   * \param[in] ab    abbreviation for the computation
   * \param[in] args  arguments for the computation
   * \return computed hydrodynamic variables
   */
  virtual torch::Tensor compute(std::string ab,
                                std::vector<torch::Tensor> const& args = {});

  //! adiabatic index and sound speed of the primitive state w, {gamma, c}:
  //! "W->A", then "WA->L" (ANEOS: "W->L", then "WL->A")
  virtual std::pair<torch::Tensor, torch::Tensor> gamma_sound_speed(
      torch::Tensor const& w);

  //! adiabatic index of w alone: "W->A" (ANEOS: "W->L", then "WL->A")
  virtual torch::Tensor adiabatic_index(torch::Tensor const& w);

  //! sound speed of w alone: "W->A", then "WA->L" (ANEOS: "W->L")
  virtual torch::Tensor sound_speed(torch::Tensor const& w);

  torch::Tensor forward(torch::Tensor cons,
                        torch::optional<torch::Tensor> out = torch::nullopt);

  //! \brief Apply the conserved variable limiter in place.
  //! \param whole_column repair a parentless cloud and the vapor over the
  //!        whole x1 column when it spans meshblocks; all blocks of the column
  //!        must then call together, as MeshBlockImpl::advance_local does
  //!        (#232).
  //! \param totals   `cons` holds totals even though the block stores
  //!        anomalies (fp32 roadmap Step 1): the "U->W"/"W->U" keys. Without a
  //!        hydro reference it changes nothing. With one and `totals` false,
  //!        rows {IDN, IPR} of `cons` are anomalies of the full block and the
  //!        floors act in anomaly space, bit-exact where they are inactive.
  virtual void apply_conserved_limiter_(torch::Tensor const& cons,
                                        bool whole_column = false,
                                        bool totals = false);

  //! \brief Apply the primitive variable limiter in place.
  virtual void apply_primitive_limiter_(torch::Tensor const& prim);

  //! \brief Zero the step's limiter marks on the device of \p like.
  void reset_limiter_marks(torch::Tensor const& like) {
    ef_status().reset_marks(like);
  }

  //! [0]: a limiter call floored an interior density or energy; [1]: it found
  //! an interior NaN. Undefined (nothing marked) before the first reset.
  //! A side channel: the limiter calls OR them in place as a side effect; the
  //! MeshBlock resets them at stage 0 and after a redo decision, and reads them
  //! in check_redo (MeshBlock and Mesh). Slot kMarks of the block's EfStatus.
  torch::Tensor const& limiter_marks() const {
    return ef_status().slot(EfStatus::kMarks);
  }

  //! the block's EfStatus (MeshBlockImpl::pefs); an EOS without a block
  //! keeps its own
  EfStatus& ef_status() const;

  //! While on (a MeshBlock stage, and check_redo's conversion), the vapor
  //! repair adds its failures to vapor_failures() instead of raising (on CUDA,
  //! without reading them back); check_redo redoes the step on a nonzero count
  //! and zeroes it, and stage 0 zeroes it. Off (any other call), a failure
  //! raises at once.
  void defer_vapor_check(bool on) { defer_vapor_check_ = on; }
  //! the block's deferred failure count (0-d int32), undefined until used
  torch::Tensor const& vapor_failures() const {
    return ef_status().slot(EfStatus::kVaporFail);
  }

 protected:
  //! "W->U" and "U->W", the same for every type: args {w[, u]} / {u[, w]},
  //! the output allocated like the input when not given. True, with *out
  //! set, if ab is one of the two.
  bool convert_(std::string const& ab, std::vector<torch::Tensor> const& args,
                torch::Tensor* out);

  //! "WA->L" of an ideal gas (dry or moist): sqrt(gamma p / rho)
  static torch::Tensor ideal_sound_speed_(torch::Tensor const& w,
                                          torch::Tensor const& gamma);

  //! kinetic energy 1/2 v . m from the primitive velocity rows of prim and
  //! the matching (lowered velocity or momentum) rows m
  static torch::Tensor kinetic_energy_(torch::Tensor const& prim,
                                       torch::Tensor const& m);

  //! kinetic energy of the conserved state, 1/2 m . raise(m) / rho, with
  //! rho = dry + the first ny species (species_total order)
  torch::Tensor cons_kinetic_energy_(torch::Tensor const& cons,
                                     torch::Tensor const& dry, int ny) const;

  //! \brief Convert primitive variables to conserved variables.
  virtual void _prim2cons(torch::Tensor prim, torch::Tensor& cons);
  //! \brief Convert conserved variables to primitive variables.
  virtual void _cons2prim(torch::Tensor cons, torch::Tensor& prim);

  //! the momentum rows of cons from prim: density times velocity, lowered by
  //! the coordinate metric
  void _mom_from_vel_(torch::Tensor const& prim, torch::Tensor& cons) const;

 private:
  bool defer_vapor_check_ = false;
  // not a buffer: stage forcings get named_buffers()
  mutable std::shared_ptr<EfStatus> own_status_;

  //! Parent vapor slots and normalized stoichiometric mass fractions by cloud.
  std::vector<std::vector<std::pair<int, double>>> cloud_parent_cache_;

  void cache_cloud_parents_();
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

}  // namespace snap

#undef ADD_ARG
