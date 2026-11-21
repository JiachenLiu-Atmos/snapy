#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

#include "riemann_dispatch.hpp"

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
class CoordinateImpl;

//! rotate the left/right primitive states at an x<dim> face into the
//! face-local orthonormal frame, in place (dim 1 -> prim2local3_, 2 -> 2, 3 ->
//! 1), left before right
void to_face_frame_(CoordinateImpl const& coord, int dim,
                    torch::Tensor const& wl, torch::Tensor const& wr);

//! rotate an x<dim> face flux from the face-local frame back to the global
//! frame, in place (dim 1 -> flux2global3_, 2 -> 2, 3 -> 1)
void to_global_frame_(CoordinateImpl const& coord, int dim,
                      torch::Tensor const& flx);

//! the kind named by a riemann-solver `type`
RiemannKind parse_riemann_kind(std::string const& type);

//! One Riemann solver module for every kind (V3 C.rst 3.6). hllc, lmars, roe
//! and shallow-roe solve through one dispatch stub, call_riemann; upwind and
//! plume-roe keep their torch formulas. hllc and lmars keep their EOS buffers
//! elr, clr and glr.
class RiemannSolverImpl : public torch::nn::Cloneable<RiemannSolverImpl> {
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

  //! the solver, from options->type()
  RiemannKind kind = RiemannKind::hllc;

  //! the last face-function solve's events on its interior faces, int64
  //! (2,): riemann.pstar_clamp (hllc's contact pressure clamped) and
  //! riemann.llf (roe's LLF fallback)
  torch::Tensor events;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  RiemannSolverImpl() : options(RiemannSolverOptionsImpl::create()) {}
  explicit RiemannSolverImpl(const RiemannSolverOptions& options_,
                             torch::nn::Module* p = nullptr);
  RiemannSolverImpl(const RiemannSolverOptions& options_, RiemannKind kind_,
                    torch::nn::Module* p);
  void reset() override;

  //! Solve the Riemann problem
  /*!
   * upwind: vel_or_flux is the face velocity or mass flux and the upwind
   * flux is returned. plume-roe (dim 1 only) returns its Lax-Friedrichs
   * flux of the plume equations. The other kinds write the flux into
   * vel_or_flux and return it.
   *
   * central, if defined (hllc, lmars, roe, shallow-roe): the kernel writes
   * riemann.central there, the mass rows (IDN and the species) of
   * 1/2 (F(w_L) + F(w_R)) from the face states it solved; other rows are
   * left as they are.
   */
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor vel_or_flux,
                        torch::Tensor face_pressure = torch::Tensor(),
                        torch::Tensor central = torch::Tensor());

  //! hllc only. Pressure rows are perturbations about one shared face
  //! background. Returns perturbation normal-momentum flux and contact
  //! pressure; energy flux includes the total pressure. Other primitive rows
  //! retain their normal meanings. This is separate from the total-state
  //! forward contract.
  //!
  //! central, if defined: riemann.central as in forward. Its mass rows read
  //! only the density, normal velocity and species rows, which are totals
  //! here too, so it is the total-state central.
  torch::Tensor forward_perturbations(
      torch::Tensor wl, torch::Tensor wr, int dim, torch::Tensor out,
      torch::Tensor reference_pressure,
      torch::Tensor face_pressure_delta = torch::Tensor(),
      torch::Tensor central = torch::Tensor());

 protected:
  FORWARD_HAS_DEFAULT_ARGS({4, torch::nn::AnyValue(torch::Tensor())},
                           {5, torch::nn::AnyValue(torch::Tensor())})

 private:
  //! the face functions' solve: one EOS face pass, rotation, call_riemann
  torch::Tensor solve(torch::Tensor wl, torch::Tensor wr, int dim,
                      torch::Tensor flx, torch::Tensor face_pressure,
                      torch::Tensor reference_pressure, torch::Tensor central);
};
using RiemannSolver = std::shared_ptr<RiemannSolverImpl>;

//! The Python classes of the pin (pyriemann.cpp): each is the one module with
//! its kind fixed. One RiemannSolver binding in their place is FOR CHENG
//! (C.rst 3.8).
#define SNAP_RIEMANN_KIND_MODULE(Name, Kind)                               \
  struct Name##Impl : public RiemannSolverImpl {                           \
    Name##Impl() { kind = RiemannKind::Kind; }                             \
    Name##Impl(const RiemannSolverOptions& options_, torch::nn::Module* p) \
        : RiemannSolverImpl(options_, RiemannKind::Kind, p) {}             \
  }

SNAP_RIEMANN_KIND_MODULE(UpwindSolver, upwind);
SNAP_RIEMANN_KIND_MODULE(RoeSolver, roe);
SNAP_RIEMANN_KIND_MODULE(LmarsSolver, lmars);
SNAP_RIEMANN_KIND_MODULE(ShallowRoeSolver, shallow_roe);
#undef SNAP_RIEMANN_KIND_MODULE
}  // namespace snap

#undef ADD_ARG
