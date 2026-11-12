#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {
struct RiemannSolverOptionsImpl {
  static std::shared_ptr<RiemannSolverOptionsImpl> create() {
    return std::make_shared<RiemannSolverOptionsImpl>();
  }
  static std::shared_ptr<RiemannSolverOptionsImpl> from_yaml(
      std::string const& filename, std::string const& section);
  static std::shared_ptr<RiemannSolverOptionsImpl> from_yaml(
      YAML::Node const& node);

  RiemannSolverOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- riemann solver options --\n";
    os << "* type = " << type() << "\n"
       << "* dir = " << dir() << "\n";
  }

  // type of Riemann solver
  ADD_ARG(std::string, type) = "hllc";

  // used in shallow water equations
  ADD_ARG(std::string, dir) = "omni";
};
using RiemannSolverOptions = std::shared_ptr<RiemannSolverOptionsImpl>;

class HydroImpl;

class RiemannSolverImpl {
 public:
  //! Create and register a RiemannSolver module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `RiemannSolver` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return created      `RiemannSolver` module
   */
  static std::shared_ptr<RiemannSolverImpl> create(
      RiemannSolverOptions const& opts, torch::nn::Module* p,
      std::string const& name = "riemann");

  //! data
  torch::Tensor elr, clr, glr;

  //! options with which this `RiemannSolver` was constructed
  RiemannSolverOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  RiemannSolverImpl() : options(RiemannSolverOptionsImpl::create()) {}
  explicit RiemannSolverImpl(const RiemannSolverOptions& options_,
                             torch::nn::Module* p = nullptr);
  virtual ~RiemannSolverImpl() = default;

  //! Solver the Riemann problem
  virtual torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                                torch::Tensor vel_or_flux,
                                torch::Tensor face_pressure = torch::Tensor());
};
using RiemannSolver = std::shared_ptr<RiemannSolverImpl>;

class UpwindSolverImpl : public torch::nn::Cloneable<UpwindSolverImpl>,
                         public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  UpwindSolverImpl() = default;
  explicit UpwindSolverImpl(const RiemannSolverOptions& options_,
                            torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override {}
  using RiemannSolverImpl::forward;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(UpwindSolver);

class RoeSolverImpl : public torch::nn::Cloneable<RoeSolverImpl>,
                      public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  RoeSolverImpl() = default;
  explicit RoeSolverImpl(const RiemannSolverOptions& options_,
                         torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out,
                        torch::Tensor face_pressure = torch::Tensor()) override;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(RoeSolver);

class LmarsSolverImpl : public torch::nn::Cloneable<LmarsSolverImpl>,
                        public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  LmarsSolverImpl() = default;
  explicit LmarsSolverImpl(const RiemannSolverOptions& options_,
                           torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out,
                        torch::Tensor face_pressure = torch::Tensor()) override;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(LmarsSolver);

class HLLCSolverImpl : public torch::nn::Cloneable<HLLCSolverImpl>,
                       public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  HLLCSolverImpl() = default;
  explicit HLLCSolverImpl(const RiemannSolverOptions& options_,
                          torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out,
                        torch::Tensor face_pressure = torch::Tensor()) override;

  //! Pressure rows are perturbations about one shared face background.
  //! Returns perturbation normal-momentum flux and contact pressure; energy
  //! flux includes the total pressure. Other primitive rows retain their
  //! normal meanings. This is separate from the total-state forward contract.
  torch::Tensor forward_perturbations(
      torch::Tensor wl, torch::Tensor wr, int dim, torch::Tensor out,
      torch::Tensor reference_pressure,
      torch::Tensor face_pressure_delta = torch::Tensor());

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})

 private:
  torch::Tensor _forward(torch::Tensor wl, torch::Tensor wr, int dim,
                         torch::Tensor out, torch::Tensor face_pressure,
                         torch::Tensor reference_pressure);
};
TORCH_MODULE(HLLCSolver);

class ShallowRoeSolverImpl : public torch::nn::Cloneable<ShallowRoeSolverImpl>,
                             public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  ShallowRoeSolverImpl() = default;
  explicit ShallowRoeSolverImpl(const RiemannSolverOptions& options_,
                                torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out,
                        torch::Tensor face_pressure = torch::Tensor()) override;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(ShallowRoeSolver);

class PlumeRoeSolverImpl : public torch::nn::Cloneable<PlumeRoeSolverImpl>,
                           public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  PlumeRoeSolverImpl() = default;
  explicit PlumeRoeSolverImpl(const RiemannSolverOptions& options_,
                              torch::nn::Module* p = nullptr)
      : RiemannSolverImpl(options_, p) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  /*!
   * F1 = (1 - 1/e) * R^2 * W
   * F2 = 1/2 * R^2 * W * W - 1/2 * R^3 * V - 1/8 * R^2 * V^2
   * F3 = 1/2 * R^2 * B * W
   * F4 = 1/4 * R^3 * V * W + 1/2 * R^4 * W
   *
   * d(Q1)/dt = - d(F1)/dx
   * d(Q2)/dt = - d(F2)/dx
   * d(Q3)/dt = - d(F3)/dx
   * d(Q4)/dt = - d(F4)/dx
   */
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out,
                        torch::Tensor face_pressure = torch::Tensor()) override;

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())})
};
TORCH_MODULE(PlumeRoeSolver);
}  // namespace snap

#undef ADD_ARG
