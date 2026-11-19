// snap
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>

#include "sedimentation.hpp"

namespace snap {
SedVelImpl::SedVelImpl(SedVelOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  psed = dynamic_cast<SedHydroImpl const*>(p);
  reset();
}

void SedVelImpl::reset() {
  TORCH_CHECK(psed, "[SedVel] Parent SedHydro is null");
  TORCH_CHECK(options->radius().size() == options->density().size(),
              "[SedVel] radius and density must have the same size");

  radius = register_buffer("radius",
                           torch::tensor(options->radius(), torch::kFloat64));

  density = register_buffer("density",
                            torch::tensor(options->density(), torch::kFloat64));

  const_vsed = register_buffer(
      "const_vsed", torch::tensor(options->const_vsed(), torch::kFloat64));
}

std::shared_ptr<SedVelImpl> SedVelImpl::create(SedVelOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(opts != nullptr, "[SedVel] Options pointer is nullptr");
  TORCH_CHECK(p != nullptr, "[SedVel] Parent module is nullptr");
  return p->register_module(name, SedVel(opts, p));
}

}  // namespace snap
