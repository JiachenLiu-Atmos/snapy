// kintere
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "ideal_moist.hpp"

namespace snap {

void IdealMoistImpl::reset() {
  TORCH_CHECK(options->thermo(), "[IdealMoist] thermo pointer is null");
  pthermo = kintera::ThermoYImpl::create(options->thermo(), this);

  // make gammad and weight consistent
  options->gammad(1. + 1. / options->thermo()->cref_R()[0]);
  options->weight(1. / pthermo->inv_mu[0].item<double>());

  // populate buffers
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  inv_mu_ratio_m1 =
      register_buffer("inv_mu_ratio_m1", torch::zeros({ny}, torch::kFloat64));

  for (int i = 0; i < ny; ++i) {
    inv_mu_ratio_m1[i] = pthermo->inv_mu[i + 1] / pthermo->inv_mu[0] - 1.;
  }

  cv_ratio_m1 =
      register_buffer("cv_ratio_m1", torch::zeros({ny}, torch::kFloat64));

  auto Rd = kintera::constants::Rgas * pthermo->inv_mu[0];
  for (int i = 0; i < ny; ++i) {
    auto Ri = kintera::constants::Rgas * pthermo->inv_mu[i + 1];
    cv_ratio_m1[i] = (options->thermo()->cref_R()[1 + i] * Ri) /
                         (options->thermo()->cref_R()[0] * Rd) -
                     1.;
  }

  u0 = register_buffer(
      "u0", torch::tensor(options->thermo()->uref_R(), torch::kFloat64));

  u0 *= kintera::constants::Rgas * pthermo->inv_mu;
}

double IdealMoistImpl::species_weight(int n) const {
  return 1. / pthermo->inv_mu[n].item<double>();
}

double IdealMoistImpl::species_cv_ref(int n) const {
  auto Ri = kintera::constants::Rgas * pthermo->inv_mu[n];
  return (options->thermo()->cref_R()[n] * Ri).item<double>();
}

torch::Tensor IdealMoistImpl::specific_heat_cv(torch::Tensor prim,
                                               torch::Tensor temp) {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  return species_cv_ref() * f_sig(prim.narrow(0, ICY, ny));
}

torch::Tensor IdealMoistImpl::internal_energy_offset(
    torch::Tensor hydro_like) const {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  auto offset = hydro_like[IDN] * u0[0];
  if (ny == 0) return offset;

  std::vector<int64_t> vec(hydro_like.dim(), 1);
  vec[0] = -1;
  offset +=
      (hydro_like.narrow(0, ICY, ny) * u0.narrow(0, 1, ny).view(vec)).sum(0);
  return offset;
}

IdealMoistPerturbations IdealMoistImpl::perturbations(
    torch::Tensor const& stored) const {
  TORCH_CHECK(phydro->has_reference(),
              "ideal-moist perturbations require an anomaly background");
  auto species0 = phydro->species_ref();
  TORCH_CHECK(species0.defined(),
              "ideal-moist perturbations require the original species "
              "background; this legacy restart only supports anomaly storage");
  auto const& thermo = pthermo->options;
  TORCH_CHECK(!thermo->use_nasa9_cp() && !thermo->use_h2_cp(),
              "ideal-moist perturbations require constant heat capacities");
  auto u = stored.to(torch::kFloat64);
  auto ref = phydro->u_ref_like(u);
  int ny = nvar() - ICY;
  TORCH_CHECK(stored.size(0) == nvar() && species0.size(0) == ny,
              "ideal-moist perturbations: species count mismatch");
  auto s0 = species0.expand({ny, u.size(1), u.size(2), u.size(3)});
  auto rho0 = torch::cat({ref[0].unsqueeze(0), s0}, 0);
  auto drho = torch::cat({u[IDN].unsqueeze(0), u.narrow(0, ICY, ny) - s0}, 0);

  // Recreate constants from the double-valued options, not buffers that
  // Module::to(float32) may already have rounded. Energy offsets have already
  // been referred to T=0 by ThermoY during reset.
  auto inv_mu = 1. / torch::tensor(thermo->mu(), u.options());
  auto cv = torch::tensor(thermo->cref_R(), u.options()) *
            kintera::constants::Rgas * inv_mu;
  auto offset = torch::tensor(thermo->uref_R(), u.options()) *
                kintera::constants::Rgas * inv_mu;
  auto gas = kintera::constants::Rgas * inv_mu;
  gas.narrow(0, thermo->vapor_ids().size(), thermo->cloud_ids().size()).zero_();
  auto dot = [](torch::Tensor const& r, torch::Tensor const& c) {
    return (r * c.view({-1, 1, 1, 1})).sum(0);
  };
  auto density0 = rho0.sum(0);
  auto density_delta = drho.sum(0);
  auto density = density0 + density_delta;
  auto momentum = u.narrow(0, IVX, 3);
  auto raised = momentum.clone();
  coord_vec_raise_(raised, phydro->pmb->pcoord->cosine_cell_kj);
  auto ke = 0.5 * (momentum * raised).sum(0) / density;
  auto thermal0 = ref[1] - dot(rho0, offset);
  auto dthermal = u[IPR] - ke - dot(drho, offset);
  auto gas0 = dot(rho0, gas), cv0 = dot(rho0, cv);
  auto dcv = dot(drho, cv);
  auto total_cv = cv0 + dcv;
  TORCH_CHECK((torch::isfinite(cv0) & (cv0 > 0.) & torch::isfinite(total_cv) &
               (total_cv > 0.))
                  .all()
                  .item<bool>(),
              "ideal-moist perturbations require finite positive background "
              "and total volumetric heat capacities");
  auto pressure0 = thermal0 * gas0 / cv0;
  auto pressure_delta = kintera::eval_const_cv_pressure_delta(
      thermal0, gas0, cv0, dthermal, dot(drho, gas), dcv);
  return {density0, pressure0, density_delta.to(stored.scalar_type()),
          pressure_delta.to(stored.scalar_type())};
}

torch::Tensor IdealMoistImpl::compute(std::string ab,
                                      std::vector<torch::Tensor> const& args) {
  if (ab == "W->U") {
    auto w = args[0];
    auto u = args.size() > 1 ? args[1] : torch::empty_like(w);
    _prim2cons(w, u);
    return u;
  } else if (ab == "U->W") {
    auto u = args[0];
    auto w = args.size() > 1 ? args[1] : torch::empty_like(u);
    _cons2prim(u, w);
    return w;
  } else if (ab == "W->U'") {
    // the stored representation: totals, then rows {IDN, IPR} minus the
    // reference (fp32 roadmap Step 1); totals when there is no reference
    auto w = args[0];
    auto u = args.size() > 1 ? args[1] : torch::empty_like(w);
    _prim2cons(w, u);
    phydro->sub_reference_(u);
    return u;
  } else if (ab == "U'->W") {
    // limit the stored anomalies in place (anomaly-space floors), then
    // convert the totals they stand for; `u` itself keeps anomalies
    auto u = args[0];
    auto w = args.size() > 1 ? args[1] : torch::empty_like(u);
    if (!phydro->has_reference()) {
      _cons2prim(u, w);
      return w;
    }
    apply_conserved_limiter_(u);
    _cons2prim(phydro->total_u(u), w, /*limit=*/false);
    return w;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    return _prim2temp(w);
  } else if (ab == "W->E") {
    auto w = args[0];
    return _prim2speciesEng(w);
  } else if (ab == "U->K") {
    auto u = args[0];
    return _cons2ke(u);
  } else if (ab == "UT->I") {
    auto u = args[0];
    auto temp = args[1];
    return _temp2intEng(u, temp);
  } else if (ab == "W->A") {
    auto w = args[0];
    auto gammad =
        (pthermo->options->cref_R()[0] + 1) / pthermo->options->cref_R()[0];
    int ny = pthermo->options->vapor_ids().size() +
             pthermo->options->cloud_ids().size() - 1;
    auto feps = f_eps(w.narrow(0, ICY, ny));
    auto fsig = f_sig(w.narrow(0, ICY, ny));
    return 1. + (gammad - 1.) * feps / fsig;
  } else if (ab == "WA->L") {
    auto w = args[0];
    auto gamma = args[1];
    auto dens = w[IDN];
    auto pres = w[IPR];
    return torch::sqrt(gamma * pres / dens);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void IdealMoistImpl::_prim2cons(torch::Tensor prim, torch::Tensor& cons) {
  apply_primitive_limiter_(prim);
  auto pcoord = phydro->pmb->pcoord;

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  // den -> den
  auto out = cons[IDN];
  torch::mul_out(out, (1. - prim.narrow(0, ICY, ny).sum(0)), prim[IDN]);

  // mixr -> den
  out = cons.narrow(0, ICY, ny);
  torch::mul_out(out, prim.narrow(0, ICY, ny), prim[IDN]);

  // vel -> mom
  out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  coord_vec_lower_(out, pcoord->cosine_cell_kj);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons, /*whole_column=*/false, /*totals=*/true);
}

void IdealMoistImpl::_cons2prim(torch::Tensor cons, torch::Tensor& prim,
                                bool limit) {
  if (limit) {
    apply_conserved_limiter_(cons, /*whole_column=*/false, /*totals=*/true);
  }
  auto pcoord = phydro->pmb->pcoord;

  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

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
  auto ie = cons[IPR] - ke;

  // subtract the internal energy offset
  ie -= cons[IDN] * u0[0];

  std::vector<int64_t> vec(cons.dim(), 1);
  vec[0] = -1;
  ie -= (cons.narrow(0, ICY, ny) * u0.narrow(0, 1, ny).view(vec)).sum(0);

  // eng -> pr
  auto gammad =
      (pthermo->options->cref_R()[0] + 1) / pthermo->options->cref_R()[0];

  // TODO(cli) iteration needed here
  auto yfrac = prim.narrow(0, ICY, ny);
  prim[IPR] = (gammad - 1) * ie * f_eps(yfrac) / f_sig(yfrac);

  apply_primitive_limiter_(prim);
}

torch::Tensor IdealMoistImpl::_prim2intEng(torch::Tensor prim) {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  auto yfrac = prim.narrow(0, ICY, ny);

  auto gammad =
      (pthermo->options->cref_R()[0] + 1) / pthermo->options->cref_R()[0];

  // a dry gas has f_sig = f_eps = 1 and yd = 1 exactly, and
  // the empty species matmul adds rho * 0; skip those launches
  if (ny == 0) {
    auto ie0 = prim[IPR] / (gammad - 1);
    ie0 += prim[IDN] * u0[0];
    return ie0;
  }

  // TODO(cli) iteration needed here
  auto ie = prim[IPR] * f_sig(yfrac) / f_eps(yfrac) / (gammad - 1);

  // add the internal energy offset
  auto yd = 1. - yfrac.sum(0);
  ie += prim[IDN] * yd * u0[0];
  ie +=
      prim[IDN] * yfrac.unfold(0, ny, 1).matmul(u0.narrow(0, 1, ny)).squeeze(0);
  return ie;
}

torch::Tensor IdealMoistImpl::_prim2temp(torch::Tensor prim) {
  // Recompute so each public call owns its autograd graph.
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  auto Rd = kintera::constants::Rgas / options->weight();
  auto yfrac = prim.narrow(0, ICY, ny);
  auto t = prim[IPR] / (prim[IDN] * Rd * f_eps(yfrac));
  return t;
}

torch::Tensor IdealMoistImpl::_prim2speciesEng(torch::Tensor prim) {
  auto pcoord = phydro->pmb->pcoord;
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  auto mud = options->weight();
  auto Rd = kintera::constants::Rgas / mud;
  auto cvd = pthermo->options->cref_R()[0] * Rd;

  auto yfrac = prim.narrow(0, ICY, ny);
  auto temp = prim[IPR] / (prim[IDN] * Rd * f_eps(yfrac));

  auto rhos = prim[IDN] * yfrac;

  std::vector<int64_t> vec = {ny, 1, 1, 1};
  auto ie = rhos * (u0.narrow(0, 1, ny).view(vec) +
                    (cv_ratio_m1.view(vec) + 1.) * cvd * temp);

  auto vel = prim.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  return ie + ke * rhos;
}

torch::Tensor IdealMoistImpl::species_enthalpy(torch::Tensor prim) {
  auto pcoord = phydro->pmb->pcoord;
  int nvapor = pthermo->options->vapor_ids().size() - 1;
  int ny = nvapor + pthermo->options->cloud_ids().size();

  auto Rd = kintera::constants::Rgas / options->weight();
  auto cvd = pthermo->options->cref_R()[0] * Rd;

  // the per-species split of the flux enthalpy rho*(W->I/rho + KE + p/rho):
  // u0 + cv T + KE, plus R T for a vapour (a cloud has no pressure share)
  // Recompute to avoid carrying inference tensors into tracked calls.
  auto c = (cv_ratio_m1 + 1.) * cvd;
  c.narrow(0, 0, nvapor) += (inv_mu_ratio_m1.narrow(0, 0, nvapor) + 1.) * Rd;

  auto vel = prim.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0);

  std::vector<int64_t> vec(prim.dim(), 1);
  vec[0] = ny;
  return u0.narrow(0, 1, ny).view(vec) + c.view(vec) * _prim2temp(prim) + ke;
}

torch::Tensor IdealMoistImpl::_cons2ke(torch::Tensor cons) {
  auto pcoord = phydro->pmb->pcoord;
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;
  auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);

  auto mom = cons.narrow(0, IVX, 3).clone();
  coord_vec_raise_(mom, pcoord->cosine_cell_kj);
  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
}

torch::Tensor IdealMoistImpl::_temp2intEng(torch::Tensor cons,
                                           torch::Tensor temp) {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  auto mud = options->weight();
  auto Rd = kintera::constants::Rgas / mud;
  auto cvd = pthermo->options->cref_R()[0] * Rd;
  auto cvy = (cv_ratio_m1 + 1.) * cvd;

  std::vector<int64_t> vec(cons.dim(), 1);
  vec[0] = -1;

  // Heat capacity per unit volume. The DRY channel carries cvd and dominates
  // a mostly-dry column, so leaving it out is not a small error.
  auto rho_cv =
      cons[IDN] * cvd + (cons.narrow(0, ICY, ny) * cvy.view(vec)).sum(0);

  // u0 is an energy per unit mass, not a heat capacity: the offset is ADDED to
  // the thermal energy, never scaled by temperature. This is the same offset
  // _prim2intEng adds and _cons2prim subtracts, so "UT->I" at a state's own
  // temperature now reproduces "W->I" at that state, which is the identity the
  // temperature-floor clamp in EquationOfStateImpl relies on.
  return internal_energy_offset(cons) + rho_cv * temp;
}

torch::Tensor IdealMoistImpl::f_eps(torch::Tensor const& yfrac) const {
  int nvapor = pthermo->options->vapor_ids().size() - 1;
  int ncloud = pthermo->options->cloud_ids().size();

  if (nvapor == 0 && ncloud == 0) {
    auto vec = yfrac.sizes().vec();
    vec.erase(vec.begin());
    return torch::ones(vec, yfrac.options());
  }

  auto yu = yfrac.narrow(0, 0, nvapor).unfold(0, nvapor, 1);
  return 1. + yu.matmul(inv_mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         yfrac.narrow(0, nvapor, ncloud).sum(0);
}

torch::Tensor IdealMoistImpl::f_sig(torch::Tensor const& yfrac) const {
  int ny = pthermo->options->vapor_ids().size() +
           pthermo->options->cloud_ids().size() - 1;

  if (ny == 0) {
    auto vec = yfrac.sizes().vec();
    vec.erase(vec.begin());
    return torch::ones(vec, yfrac.options());
  }

  auto yu = yfrac.unfold(0, ny, 1);
  return 1. + yu.matmul(cv_ratio_m1).squeeze(0);
}

}  // namespace snap
