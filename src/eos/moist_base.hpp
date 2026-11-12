#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

//! number of species rows (vapors after the dry one, plus clouds) a
//! thermodynamics card adds to the hydro state; 0 without a card or species
inline int thermo_species_rows(kintera::ThermoOptions const& thermo) {
  if (!thermo) return 0;
  int n = thermo->vapor_ids().size() + thermo->cloud_ids().size();
  return n > 0 ? n - 1 : 0;
}

//! \brief What the two moist equations of state share.
/*!
 * Both IdealMoistImpl (closed-form) and MoistMixtureImpl (kintera
 * thermodynamics) carry the same species layout: total density in IDN on the
 * primitive side and dry density on the conserved side, mass fractions /
 * species densities in ICY.., momenta and total energy. The conversions of
 * that layout, the kinetic energy and the species weights are written here
 * once; each type keeps its own internal energy, temperature and pressure.
 * IdealMoistImpl stays its own type: the closed-form kernels dispatch on it.
 */
class MoistEOSImpl : public EquationOfStateImpl {
 public:
  //! submodules
  kintera::ThermoY pthermo = nullptr;

  MoistEOSImpl() = default;
  explicit MoistEOSImpl(EquationOfStateOptions const& options_,
                        torch::nn::Module* p = nullptr)
      : EquationOfStateImpl(options_, p) {}

  int nvar() const override { return 4 + nspecies() + 1; }
  double species_weight(int n = 0) const override;
  double species_cv_ref(int n = 0) const override;

 protected:
  //! species rows of the state (vapors after the dry one, plus clouds)
  int nspecies() const { return thermo_species_rows(pthermo->options); }

  //! make gammad and weight consistent with the dry species of the card
  void sync_dry_options_();

  //! the conversions every moist type routes the same way; true if handled
  bool compute_common_(std::string const& ab,
                       std::vector<torch::Tensor> const& args,
                       torch::Tensor* out);

  //! \brief Convert primitive variables to conserved variables.
  void _prim2cons(torch::Tensor prim, torch::Tensor& cons);

  //! \brief First half of the conserved-to-primitive conversion: density,
  //! mass fractions and velocities into prim; returns the internal energy
  //! (total energy minus kinetic energy) the caller turns into pressure.
  torch::Tensor _cons2prim_layout(torch::Tensor cons, torch::Tensor& prim);

  //! \brief Convert conserved variables to kinetic energy.
  torch::Tensor _cons2ke(torch::Tensor cons);

  //! the conserved limiter of the layout code, on a state of totals
  virtual void limit_totals_(torch::Tensor const& cons) {
    apply_conserved_limiter_(cons);
  }

  virtual void _cons2prim(torch::Tensor cons, torch::Tensor& prim) = 0;
  virtual torch::Tensor _prim2intEng(torch::Tensor prim) = 0;
  virtual torch::Tensor _prim2temp(torch::Tensor prim) = 0;
  virtual torch::Tensor _prim2speciesEng(torch::Tensor prim) = 0;
  virtual torch::Tensor _temp2intEng(torch::Tensor cons,
                                     torch::Tensor temp) = 0;
};

}  // namespace snap
