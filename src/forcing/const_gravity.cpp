// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>

#include "forcing.hpp"

namespace snap {
ConstGravityOptions ConstGravityOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["const-gravity"]) return nullptr;

  auto node = forcing["const-gravity"];
  check_keys(node, "forcing/const-gravity",
             {"grav1", "grav2", "grav3", "non-hydrostatic", "gravity-work",
              "gravity-work-fixer"});
  auto op = ConstGravityOptionsImpl::create();

  op->grav1() = node["grav1"].as<double>(op->grav1());
  op->grav2() = node["grav2"].as<double>(op->grav2());
  op->grav3() = node["grav3"].as<double>(op->grav3());
  op->non_hydrostatic() =
      node["non-hydrostatic"].as<double>(op->non_hydrostatic());
  TORCH_CHECK(op->non_hydrostatic() >= 0. && op->non_hydrostatic() <= 1.);

  // V3 I-7 (rulings (a), (b)): face is the one gravity-work form; the cell
  // form is removed (the global fixer went in V3 X5)
  op->gravity_work() = node["gravity-work"].as<std::string>(op->gravity_work());
  TORCH_CHECK(op->gravity_work() == "face",
              "forcing/const-gravity/gravity-work: '", op->gravity_work(),
              "' is not supported: the cell form was removed (V3 I-7); use "
              "gravity-work: face");
  // the global gravity-work fixer is removed (V3 X5, gw1; Cheng): the key
  // stays readable so decks that switch it off keep working
  op->gravity_work_fixer() =
      node["gravity-work-fixer"].as<bool>(op->gravity_work_fixer());
  TORCH_CHECK(!op->gravity_work_fixer(),
              "forcing/const-gravity/gravity-work-fixer was removed in V3; "
              "set it false or drop the key");

  return op;
}

torch::Tensor ConstGravityImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, StepDt dt) {
  return forward_split(du, w, dt, w[IDN]);
}

torch::Tensor ConstGravityImpl::forward_split(torch::Tensor du, torch::Tensor w,
                                              StepDt dt, torch::Tensor rho1) {
  if (options->grav1() != 0.) {
    du[IVX] += dt * rho1 * options->grav1() * options->non_hydrostatic();
    du[IPR] +=
        dt * w[IDN] * w[IVX] * options->grav1() * options->non_hydrostatic();
  }

  if (options->grav2() != 0.) {
    du[IVY] += dt * w[IDN] * options->grav2();
    du[IPR] += dt * w[IDN] * w[IVY] * options->grav2();
  }

  if (options->grav3() != 0.) {
    du[IVZ] += dt * w[IDN] * options->grav3();
    du[IPR] += dt * w[IDN] * w[IVZ] * options->grav3();
  }

  return du;
}
}  // namespace snap
