// kintera
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "moist_base.hpp"

namespace snap {

void MoistEOSImpl::sync_dry_options_() {
  options->gammad(1. + 1. / options->thermo()->cref_R()[0]);
  options->weight(1. / pthermo->inv_mu[0].item<double>());
}

double MoistEOSImpl::species_weight(int n) const {
  return 1. / pthermo->inv_mu[n].item<double>();
}

double MoistEOSImpl::species_cv_ref(int n) const {
  auto Ri = kintera::constants::Rgas * pthermo->inv_mu[n];
  return (options->thermo()->cref_R()[n] * Ri).item<double>();
}

torch::Tensor MoistEOSImpl::species_energy(torch::Tensor prim) {
  return prim[IDN] * prim.narrow(0, ICY, nspecies()) *
         (species_enthalpy(prim) - species_pressure_share(prim));
}

bool MoistEOSImpl::compute_common_(std::string const& ab,
                                   std::vector<torch::Tensor> const& args,
                                   torch::Tensor* out) {
  if (convert_(ab, args, out)) {
    return true;
  } else if (ab == "W->I") {
    *out = _prim2intEng(args[0]);
  } else if (ab == "W->T") {
    *out = _prim2temp(args[0]);
  } else if (ab == "W->E") {
    *out = species_energy(args[0]);
  } else if (ab == "U->K") {
    *out = _cons2ke(args[0]);
  } else if (ab == "UT->I") {
    *out = _temp2intEng(args[0], args[1]);
  } else {
    return false;
  }
  return true;
}

void MoistEOSImpl::_prim2cons(torch::Tensor prim, torch::Tensor& cons) {
  apply_primitive_limiter_(prim);

  int ny = nspecies();

  // den -> den
  auto out = cons[IDN];
  torch::mul_out(out, (1. - prim.narrow(0, ICY, ny).sum(0)), prim[IDN]);

  // mixr -> den
  out = cons.narrow(0, ICY, ny);
  torch::mul_out(out, prim.narrow(0, ICY, ny), prim[IDN]);

  // vel -> mom
  _mom_from_vel_(prim, cons);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  limit_totals_(cons);
}

torch::Tensor MoistEOSImpl::_cons2prim_layout(torch::Tensor cons,
                                              torch::Tensor& prim) {
  auto pcoord = phydro->pmb->pcoord;

  int ny = nspecies();

  // den -> den
  auto out = prim[IDN];
  torch::sum_out(out, cons.narrow(0, ICY, ny), /*dim=*/0);
  out += cons[IDN];

  // den -> mixr
  out = prim.narrow(0, ICY, ny);
  torch::div_out(out, cons.narrow(0, ICY, ny), prim[IDN]);

  // mom -> vel
  out = prim.narrow(0, IVX, 3);
  torch::div_out(out, cons.narrow(0, IVX, 3), prim[IDN]);

  coord_vec_raise_(out, pcoord->cosine_cell_kj);

  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
  return cons[IPR] - ke;
}

torch::Tensor MoistEOSImpl::_cons2ke(torch::Tensor cons) {
  auto pcoord = phydro->pmb->pcoord;
  int ny = nspecies();
  auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);

  auto mom = cons.narrow(0, IVX, 3).clone();
  coord_vec_raise_(mom, pcoord->cosine_cell_kj);
  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
}

}  // namespace snap
