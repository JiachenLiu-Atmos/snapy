// snap
#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

std::vector<std::string> HydroImpl::_register_forcings_module() {
  std::vector<std::string> names;
  // gravity first (lane I's module), then the sources in this order
  if (options->grav()) {
    pgrav = register_module("const-gravity", ConstGravity(options->grav()));
    names.push_back("const-gravity");
  }
  auto add = [&](auto m) {
    auto const& name = m->declaration().name;
    sources.add(register_module(name, m.ptr()));
    names.push_back(name);
    return m;
  };
  if (options->coriolis()) {
    // and again after the implicit solve, under the x1 mass-flux gate (#355)
    sources.add(add(Coriolis(options->coriolis(), this)).ptr(),
                SourcePhase::post_implicit);
  }
  if (options->diffusion())
    pdiffusion = add(Diffusion(options->diffusion(), this));
  if (options->bodyHeat()) add(BodyHeat(options->bodyHeat(), this));
  if (options->topCool()) add(LayerHeat(options->topCool(), this));
  if (options->botHeat()) add(LayerHeat(options->botHeat(), this));
  if (options->relaxBotComp()) add(RelaxBotComp(options->relaxBotComp(), this));
  if (options->relaxBotTemp()) add(RelaxBotTemp(options->relaxBotTemp(), this));
  if (options->relaxBotVelo()) add(RelaxBotVelo(options->relaxBotVelo(), this));
  if (options->topSpongeLyr()) add(Sponge(options->topSpongeLyr(), this));
  if (options->botSpongeLyr()) add(Sponge(options->botSpongeLyr(), this));
  if (options->eos()->type() == "plume-eos")
    add(PlumeForcing(options->plumeForcing()));
  sources.on_block = [this](SourceRegion r) {
    return r == SourceRegion::interior ||
           pmb->options->is_physical_boundary(0, 0,
                                              r == SourceRegion::top ? 1 : -1);
  };
  return names;
}
}  // namespace snap
