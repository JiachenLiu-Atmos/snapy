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

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

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

  //! per particle: radius, density and a constant velocity (0: Stokes)
  ADD_ARG(std::vector<double>, radius) = {};
  ADD_ARG(std::vector<double>, density) = {};
  ADD_ARG(std::vector<double>, const_vsed) = {};

  //! the background gas, H2 by default: molecular diameter [m], Lennard-Jones
  //! energy [J] (59.7 K), molecular mass [kg]; the velocity cap [m/s]
  ADD_ARG(double, a_diameter) = 2.827e-10;
  ADD_ARG(double, a_epsilon_LJ) = 8.24e-22;
  ADD_ARG(double, a_mass) = 3.34e-27;
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

//! The kernel's tensors in the compute dtype and device, built on first use
//! and again after a move (D8): (nparticle,) and cosine_cell_kj (nc3, nc2)
struct SedKernelArgs {
  torch::Tensor radius, density, const_vsed, hydro_ids, cosine_cell_kj;
};

class SedHydroImpl : public torch::nn::Cloneable<SedHydroImpl> {
 public:
  static std::shared_ptr<SedHydroImpl> create(SedHydroOptions const& opts,
                                              torch::nn::Module* p,
                                              std::string const& name = "sed");

  //! the face velocities (cache), the particles' hydro rows, and the x1
  //! settling flux of the stage (P1: EF's buffer, kept until 4.C)
  torch::Tensor vsed, hydro_ids, settling;
  //! particle radius, density, const_vsed (nparticle,), from the options at
  //! reset() (D14: SedVel folded in)
  torch::Tensor radius, density, const_vsed;

  HydroImpl const* phydro = nullptr;  //!< non-owning reference to parent
  SedHydroOptions options;

  SedHydroImpl() : options(SedHydroOptionsImpl::create()) {}
  explicit SedHydroImpl(SedHydroOptions const& options_,
                        torch::nn::Module* p = nullptr);
  void reset() override;

  //! the settling flux of primitives wr added onto out (or zeros)
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
