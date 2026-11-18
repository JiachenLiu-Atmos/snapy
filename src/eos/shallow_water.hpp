#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

class ShallowWaterImpl final : public torch::nn::Cloneable<ShallowWaterImpl>,
                               public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  ShallowWaterImpl() = default;
  ShallowWaterImpl(EquationOfStateOptions const& options_,
                   torch::nn::Module* p = nullptr)
      : EquationOfStateImpl(options_, p) {
    reset();
  }
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  int nvar() const override { return 4; }

  /*torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }*/

  //! \brief Implementation of shallow water equation of state.
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out) override;

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out) override;
};
TORCH_MODULE(ShallowWater);

}  // namespace snap
