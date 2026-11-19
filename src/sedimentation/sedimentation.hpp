#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/utils/format.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct ConstGravityOptionsImpl;
using ConstGravityOptions = std::shared_ptr<ConstGravityOptionsImpl>;

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

class EquationOfStateImpl;
using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

struct SedVelOptionsImpl {
  static std::shared_ptr<SedVelOptionsImpl> create() {
    return std::make_shared<SedVelOptionsImpl>();
  }
  static std::shared_ptr<SedVelOptionsImpl> from_yaml(YAML::Node const& node);

  void report(std::ostream& os) const {
    os << "-- sedimentation velocity options --\n";
    os << "* particle_ids = " << fmt::format("{}", particle_ids()) << "\n"
       << "* radius = " << fmt::format("{}", radius()) << "\n"
       << "* density = " << fmt::format("{}", density()) << "\n"
       << "* const_vsed = " << fmt::format("{}", const_vsed()) << "\n"
       << "* a_diameter = " << a_diameter() << "\n"
       << "* a_epsilon_LJ = " << a_epsilon_LJ() << "\n"
       << "* a_mass = " << a_mass() << "\n"
       << "* upper_limit = " << upper_limit() << "\n";
  }
  //! \return species names
  std::vector<std::string> species() const;

  //! id of precipitating particles in kintera's species table
  ADD_ARG(std::vector<int>, particle_ids) = {};

  //! names of precipitating particles, in particle_ids order. Kept from the
  //! card this was parsed from: kintera's species table is process-global and
  //! refills when a later card declares other species (kintera #121), so
  //! particle_ids cannot be resolved through it afterwards (issue #234).
  ADD_ARG(std::vector<std::string>, particle_names) = {};

  //! radius and density of particles
  //! if specified, must be the same size of cloud particles in thermo
  ADD_ARG(std::vector<double>, radius) = {};
  ADD_ARG(std::vector<double>, density) = {};

  //! additional constant sedimentation velocity
  ADD_ARG(std::vector<double>, const_vsed) = {};

  //! default H2-atmosphere properties
  //! diameter of molecule [m]
  ADD_ARG(double, a_diameter) = 2.827e-10;

  //! Lennard-Jones potential [J]
  //! H2 characteristic temperature: 59.7K
  ADD_ARG(double, a_epsilon_LJ) = 8.24e-22;

  //! molecular mass of background atmosphere, default to H2 [kg]
  ADD_ARG(double, a_mass) = 3.34e-27;

  //! upper limit of sedimentation velocity [m/s]
  ADD_ARG(double, upper_limit) = 5.e3;
};
using SedVelOptions = std::shared_ptr<SedVelOptionsImpl>;

struct SedHydroOptionsImpl {
  static std::shared_ptr<SedHydroOptionsImpl> create() {
    return std::make_shared<SedHydroOptionsImpl>();
  }
  static std::shared_ptr<SedHydroOptionsImpl> from_yaml(
      std::string const& filename);

  void report(std::ostream& os) const {
    os << "-- sedimentation hydro options --\n";
    os << "* hydro_ids = " << fmt::format("{}", hydro_ids()) << "\n";
    sedvel()->report(os);
  }

  //! id of precipitating particles in hydro
  ADD_ARG(std::vector<int>, hydro_ids) = {};

  //! submodules options
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(SedVelOptions, sedvel);
};
using SedHydroOptions = std::shared_ptr<SedHydroOptionsImpl>;

class FluxRegister;
class HydroImpl;

//! The tensors of the one sedimentation kernel in the compute dtype and
//! device, built on first use and again after a move (EF.rst D8)
struct SedKernelArgs {
  torch::Tensor radius, density, const_vsed;  //!< (nparticle,)
  torch::Tensor hydro_ids;                    //!< (nparticle,) int64
  torch::Tensor cosine_cell_kj;               //!< (nc3, nc2)
};
class SedHydroImpl;

class SedVelImpl : public torch::nn::Cloneable<SedVelImpl> {
 public:
  //! Create and register a `SedVel` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] options  options for creating the `SedVel` module
   * \param[in] p        parent module for registering the created module
   * \param[in] name     name for registering the created module
   * \return           created `SedVel` module
   */
  static std::shared_ptr<SedVelImpl> create(SedVelOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name = "sedvel");

  //! particle radius and density
  //! 1D tensor of number of particles
  //! radius and density must have the same size
  torch::Tensor radius, density, const_vsed;

  //! options with which this `SedVel` was constructed
  SedVelOptions options;

  //! non-owning reference to parent
  SedHydroImpl const* psed = nullptr;

  //! Constructor to initialize the layers
  SedVelImpl() : options(SedVelOptionsImpl::create()) {}
  explicit SedVelImpl(SedVelOptions const& options_,
                      torch::nn::Module* p = nullptr);
  void reset() override;
};
TORCH_MODULE(SedVel);

class SedHydroImpl : public torch::nn::Cloneable<SedHydroImpl> {
 public:
  static std::shared_ptr<SedHydroImpl> create(SedHydroOptions const& opts,
                                              torch::nn::Module* p,
                                              std::string const& name = "sed");

  //! cache
  torch::Tensor vsed;

  //! particle indices in hydro
  torch::Tensor hydro_ids;

  //! the x1 settling flux of the stage (P1): EF's buffer, kept until 4.C
  torch::Tensor settling;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  //! submodules
  SedVel psedvel = nullptr;

  //! options with which this `SedHydro` was constructed
  SedHydroOptions options;

  //! Constructor to initialize the layers
  SedHydroImpl() : options(SedHydroOptionsImpl::create()) {}
  explicit SedHydroImpl(SedHydroOptions const& options_,
                        torch::nn::Module* p = nullptr);
  void reset() override;

  //! Calculate sedimentation velocites
  /*!
   * \param wr        hydro primitive variables at the right interface
   * \param out       optional output tensor to store the result
   * \return          4D tensor of sedimentation flux (mass, momentum, energy).
   */
  torch::Tensor forward(torch::Tensor wr,
                        torch::optional<torch::Tensor> out = torch::nullopt);

  //! the kernel's tensors for `like`'s dtype and device
  SedKernelArgs& kernel_args(torch::Tensor const& like);

  //! whether forward adds anything (gravity and a settling species)
  bool settles() const;

  //! P1: the settling flux into `settling`, then add(1, "sedimentation") of
  //! its rows (species, IVX..IVZ, IPR) onto the register's x1 total
  torch::Tensor const& post(FluxRegister& reg, torch::Tensor wr);

 private:
  SedKernelArgs args_;
};
TORCH_MODULE(SedHydro);

}  // namespace snap

#undef ADD_ARG
