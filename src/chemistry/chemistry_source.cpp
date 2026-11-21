// kintera
#include <kintera/kinetics/evolve_implicit.hpp>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "chemistry_source.hpp"

namespace snap {

ChemistrySource::ChemistrySource(kintera::KineticsOptionsImpl reactions,
                                 kintera::ThermoOptions thermo)
    : _reactions(std::make_shared<kintera::KineticsOptionsImpl>(
          std::move(reactions))) {
  kinet = kintera::Kinetics(_reactions);
  thermo_x = kintera::ThermoX(thermo);
}

void ChemistrySource::forward(MeshBlockImpl& block, Variables& vars,
                              double dt) {
  if (empty()) return;  // a dry or reaction-free deck: no update, no Jacobian
  auto& hydro_u = vars.at("hydro_u");
  auto& hydro_w = vars.at("hydro_w");
  if (kinet->stoich.device() != hydro_u.device()) {  // the block moved
    kinet->to(hydro_u.device());
    thermo_x->to(hydro_u.device());
  }
  auto peos = block.phydro->peos;
  auto thermo_y = block.thermo();
  int ny = static_cast<int>(thermo_y->options->species().size()) - 1;

  // kinetics needs the final dynamics state, including the saturation
  // adjustment (the pin: run_hydro.cpp:180)
  peos->forward(hydro_u, hydro_w);
  auto temp = peos->compute("W->T", {hydro_w});
  auto pres = hydro_w[IPR];
  auto xfrac = thermo_y->compute("Y->X", {hydro_w.narrow(0, ICY, ny)});
  auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
  auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});
  auto conc_kinet = conc.slice(-1, 1, conc.size(-1));
  auto [rate, rc_ddC, rc_ddT, rate_constants] =
      kinet->forward_with_rate_constants(temp, pres, conc_kinet);
  auto jac = kinet->jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT,
                             rate_constants);
  auto del_conc = kintera::evolve_implicit(rate, kinet->stoich, jac, dt);
  std::vector<int64_t> vec(del_conc.dim(), 1);
  vec[del_conc.dim() - 1] = -1;
  auto del_rho = del_conc / thermo_y->inv_mu.narrow(0, 1, ny).view(vec);
  // the FP64 increment on the stored dtype's species rows only (the pin's
  // Python caller asserts the same boundary for an FP32 state)
  hydro_u.narrow(0, ICY, ny) += del_rho.permute({3, 0, 1, 2});
  // the register posts (cell_amount "chem.<species>", the element stock,
  // K.rst 3.5) arrive with lane I's I-4 posts
}

}  // namespace snap
