// C/C++
#include <algorithm>
#include <numeric>
#include <unordered_set>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

#include <snap_options/relax_bot_comp_read.inc>

RelaxBotCompOptions RelaxBotCompOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  auto op = read_forcing(forcing, "relax-bot-comp", read_relax_bot_comp_rows);
  std::unordered_set<std::string> names;
  for (int n = 0; op && n < op->species().size(); ++n) {
    TORCH_CHECK(names.insert(op->species()[n]).second,
                "RelaxBotCompOptions: duplicate species '", op->species()[n],
                "'.");
    TORCH_CHECK(op->xfrac()[n] >= 0. && op->xfrac()[n] <= 1.,
                "RelaxBotCompOptions: xfrac values must lie in [0, 1].");
  }
  TORCH_CHECK(
      !op || std::accumulate(op->xfrac().begin(), op->xfrac().end(), 0.) <= 1.,
      "RelaxBotCompOptions: configured xfrac values must sum to no "
      "more than one.");
  return op;
}

RelaxBotCompImpl::RelaxBotCompImpl(RelaxBotCompOptions op, HydroImpl const* p)
    : BottomSlab(std::move(op), p, "relax-bot-comp",
                 kMass | kSpecies | kEnergy) {
  decl_.capturable = false;  // kintera's VT->U is not capturable (section 8)
  reset();
}

void RelaxBotCompImpl::reset() {
  TORCH_CHECK(phydro, "[RelaxBotComp] Parent Hydro is null");
  auto thermo = phydro->options->eos()->thermo();
  TORCH_CHECK(thermo, "[RelaxBotComp] Thermodynamics options are required.");

  pthermo_y = register_module("thermo-y", kintera::ThermoY(thermo));
  pthermo_x = register_module("thermo-x", kintera::ThermoX(thermo));

  species_ids.clear();
  auto species = thermo->species();
  for (auto const& name : options->species()) {
    auto it = std::find(species.begin(), species.end(), name);
    TORCH_CHECK(it != species.end(), "[RelaxBotComp] Unknown species '", name,
                "'.");
    auto id = std::distance(species.begin(), it);
    TORCH_CHECK(id != 0, "[RelaxBotComp] Dry species '", name,
                "' cannot be configured directly.");
    species_ids.push_back(id);
  }
}

void RelaxBotCompImpl::add(torch::Tensor du, StageState const& s) {
  auto w = s.w, temp = s.temp;
  auto bottom =
      phydro->pmb->part({0, 0, -1}, PartOptions().exterior(false).depth(1));
  auto bottom3 = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  int ny = pthermo_y->options->species().size() - 1;

  auto wbot = w.index(bottom);
  auto rho = wbot[IDN];
  auto current_y = wbot.narrow(0, ICY, ny);
  auto target_x = pthermo_y->compute("Y->X", {current_y});
  for (int n = 0; n < species_ids.size(); ++n) {
    target_x.select(-1, species_ids[n]).fill_(options->xfrac()[n]);
  }

  auto dry = 1. - target_x.narrow(-1, 1, ny).sum(-1);
  target_x.select(-1, 0) = dry.clamp_min(0.);

  auto target_y = pthermo_x->compute("X->Y", {target_x});
  auto target_v = pthermo_y->compute("DY->V", {rho, target_y});
  auto current_v = pthermo_y->compute("DY->V", {rho, current_y});
  auto temp_bot = temp.index(bottom3);
  auto target_ie = pthermo_y->compute("VT->U", {target_v, temp_bot});
  auto current_ie = pthermo_y->compute("VT->U", {current_v, temp_bot});

  auto delta = torch::zeros_like(wbot);
  delta[IDN] = -rho * (target_y - current_y).sum(0);
  delta.narrow(0, ICY, ny) = rho * (target_y - current_y);
  delta[IPR] = target_ie - current_ie;
  du.index(bottom) += s.dt / options->tau() * delta;
}

}  // namespace snap
