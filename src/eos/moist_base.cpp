// kintera
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "moist_base.hpp"
#include "species_sum.hpp"

namespace snap {

void MoistEOSImpl::sync_dry_options_() {
  options->gammad(1. + 1. / options->thermo()->cref_R()[0]);
  host_dtype_ = c10::ScalarType::Undefined;  // a new pthermo: read it again
  options->weight(species_weight(0));
}

// host copies of inv_mu's reads, taken again when to() changes its dtype,
// so no stage call reads the device (G2)
void MoistEOSImpl::host_weights_() const {
  auto const& im = pthermo->inv_mu;
  auto const& cref = options->thermo()->cref_R();
  if (im.scalar_type() == host_dtype_) return;
  host_weight_.resize(im.size(0)), host_cv_ref_.resize(cref.size());
  for (size_t n = 0; n < host_weight_.size(); ++n) {
    host_weight_[n] = 1. / im[n].item<double>();
    auto Ri = kintera::constants::Rgas * im[n];
    if (n < cref.size()) host_cv_ref_[n] = (cref[n] * Ri).item<double>();
  }
  host_dtype_ = im.scalar_type();
}

double MoistEOSImpl::species_weight(int n) const {
  return host_weights_(), host_weight_.at(n);
}

double MoistEOSImpl::species_cv_ref(int n) const {
  return host_weights_(), host_cv_ref_.at(n);
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
  // (species_total accumulates in species order, as in _prim2intEng)
  auto out = cons[IDN];
  auto yd = ny > 0 ? 1. - species_total(prim.narrow(0, ICY, ny))
                   : torch::ones_like(prim[IDN]);
  torch::mul_out(out, yd, prim[IDN]);

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
  return cons_kinetic_energy_(cons, cons[IDN], nspecies());
}

}  // namespace snap
