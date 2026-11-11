// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>

#include "riemann_solver.hpp"

namespace snap {
RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
    std::string const& filename, std::string const& section) {
  auto op = RiemannSolverOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  if (!config[section]) return op;
  if (!config[section]["riemann-solver"]) return op;
  check_keys(config["dynamics"]["riemann-solver"], "dynamics/riemann-solver",
             {"type", "dir"});
  return from_yaml(config["dynamics"]["riemann-solver"]);
}

RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
    YAML::Node const& node) {
  auto op = RiemannSolverOptionsImpl::create();

  // without the key: the type of create() (hllc), as for a deck with no
  // riemann-solver block
  op->type() = node["type"].as<std::string>(op->type());
  op->dir() = node["dir"].as<std::string>(op->dir());

  return op;
}

RiemannSolverImpl::RiemannSolverImpl(const RiemannSolverOptions& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
}

torch::Tensor RiemannSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                         int dim, torch::Tensor vel,
                                         torch::Tensor face_pressure) {
  TORCH_CHECK(
      !face_pressure.defined(),
      "Face-pressure output is not implemented for Riemann solver type ",
      options->type());
  auto ui = (vel > 0).to(torch::kInt);
  return vel * (ui * wl + (1 - ui) * wr);
}

RiemannSolver RiemannSolverImpl::create(RiemannSolverOptions const& opts,
                                        torch::nn::Module* p,
                                        std::string const& name) {
  TORCH_CHECK(p, "[RiemannSolver] parent module is nullptr");
  TORCH_CHECK(opts, "[RiemannSolver] options pointer is nullptr");

  if (opts->type() == "roe") {
    return p->register_module(name, RoeSolver(opts, p));
  } else if (opts->type() == "lmars") {
    return p->register_module(name, LmarsSolver(opts, p));
  } else if (opts->type() == "hllc") {
    return p->register_module(name, HLLCSolver(opts, p));
  } else if (opts->type() == "upwind") {
    return p->register_module(name, UpwindSolver(opts, p));
  } else if (opts->type() == "shallow-roe") {
    return p->register_module(name, ShallowRoeSolver(opts, p));
  } else if (opts->type() == "plume-roe") {
    return p->register_module(name, PlumeRoeSolver(opts, p));
  } else {
    TORCH_CHECK(false, "RiemannSolver: unknown type " + opts->type());
  }
}

}  // namespace snap
