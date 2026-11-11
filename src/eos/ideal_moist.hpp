#pragma once

// snap
#include "moist_base.hpp"

namespace snap {

//! Static totals and small changes, kept separate through reconstruction.
struct IdealMoistPerturbations {
  torch::Tensor density0, pressure0, density_delta, pressure_delta;
};

class IdealMoistImpl final : public torch::nn::Cloneable<IdealMoistImpl>,
                             public MoistEOSImpl {
 public:
  //! data
  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;

  // Constructor to initialize the layers
  IdealMoistImpl() = default;
  explicit IdealMoistImpl(EquationOfStateOptions const& options_,
                          torch::nn::Module* p = nullptr)
      : MoistEOSImpl(options_, p) {
    reset();
  }
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  torch::Tensor specific_heat_cv(torch::Tensor prim,
                                 torch::Tensor temp) override;
  torch::Tensor internal_energy_offset(torch::Tensor hydro_like) const override;
  torch::Tensor species_enthalpy(torch::Tensor prim) override;

  //! Read a full-block stored state without limiting or rewriting it. Reference
  //! and nonlinear thermodynamic arithmetic are FP64; deltas return in the
  //! stored dtype. Requires the original species background (new restarts).
  IdealMoistPerturbations perturbations(torch::Tensor const& stored) const;

  /*torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }*/

  //! \brief Implementation of ideal gas equation of state.
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

  //! \brief Virtual factor
  /*!
   *! Eq.16 in Li2019
   *! $ \frac{R}{R_d} = \frac{\mu_d}{\mu}$
   *! \return $1/\mu$
   */
  torch::Tensor f_eps(torch::Tensor const& yfrac) const;

  //! \brief Correction to specific heat capacity at constant volume
  /*!
   *! Eq.17 in Li2019
   *! $ f_\sigma = 1 + \sum_i (\frac{c_{v,i}}{c_{v,d}} - 1.) y_i$
   *! \return $f_\sigma$
   */
  torch::Tensor f_sig(torch::Tensor const& yfrac) const;

 private:
  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out) override {
    _cons2prim(cons, out, /*limit=*/true);
  }
  //! `cons` holds totals; `limit` false skips the conserved limiter (the
  //! anomaly key "U'->W" has already limited the stored anomalies)
  void _cons2prim(torch::Tensor cons, torch::Tensor& out, bool limit);
  //! the layout code's conserved limiter: its `cons` holds totals
  void limit_totals_(torch::Tensor const& cons) override {
    apply_conserved_limiter_(cons, /*whole_column=*/false, /*totals=*/true);
  }

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \return          total internal energy [J/m^3]
   */
  torch::Tensor _prim2intEng(torch::Tensor prim) override;

  //! \brief calculate temperature.
  /*
   * \param[in] prim  primitive variables
   * \return          temperature
   */
  torch::Tensor _prim2temp(torch::Tensor prim) override;

  //! \brief calculate species energy (internal + kinetic)
  /*
   * \param[in] prim  primitive variables
   * \return          individual species energy (ie + ke)
   */
  torch::Tensor _prim2speciesEng(torch::Tensor prim) override;

  //! \brief Convert temperature to kinetic energy.
  /*
   * \param[in] cons    conserved variables
   * \param[in] temp    temperature
   * \return            internal energy
   */
  torch::Tensor _temp2intEng(torch::Tensor cons, torch::Tensor temp) override;
};
TORCH_MODULE(IdealMoist);

}  // namespace snap
