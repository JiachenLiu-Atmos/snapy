#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/eos/aneos/aneos_thermo.hpp>

#include "equation_of_state.hpp"

namespace snap {

class ANEOSImpl : public EquationOfStateImpl,
                  public torch::nn::Cloneable<ANEOSImpl> {
 public:
  // submodules
  ANEOSThermo pthermo = nullptr;

  ANEOSImpl() = default;
  explicit ANEOSImpl(const EquationOfStateOptions& options_,
                     torch::nn::Module* p = nullptr)
      : EquationOfStateImpl(options_, p) {
    reset();
  }
  void reset() override;
  using EquationOfStateImpl::forward;

  //! The following transformations are need to implement the EOS
  /*!
   * W->U: convert primitive variables to conserved variables
   * U->W: convert conserved variables to primitive variables
   * W->A: compute adiabatic index from primitive variables
   * W->L: compute sound speed from primitive variables
   * W->T: compute temperature
   * WL->A: compute sound speed from primitive variables and adiabatic index
   */
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

  //! the sound speed first ("W->L"), then the index from it ("WL->A")
  std::pair<torch::Tensor, torch::Tensor> gamma_sound_speed(
      torch::Tensor const& w) override;
  torch::Tensor adiabatic_index(torch::Tensor const& w) override;
  torch::Tensor sound_speed(torch::Tensor const& w) override;

 private:
  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out) override;

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out) override;

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \return          total internal energy [J/m^3]
   */
  torch::Tensor _prim2intEng(torch::Tensor prim);

  //! \brief calculate temperature
  /*
   * \param[in] prim  primitive variables
   * \return          temperature
   */
  torch::Tensor _prim2temp(torch::Tensor prim);

  //! \brief Convert temperature to kinetic energy.
  /*
   * \param[in] cons    conserved variables
   * \param[in] temp    temperature
   * \return            internal energy
   */
  torch::Tensor _temp2intEng(torch::Tensor cons, torch::Tensor temp);
};
TORCH_MODULE(ANEOS);

}  // namespace snap
