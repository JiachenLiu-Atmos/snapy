// C/C++
#include <numeric>

// snap
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "scalar.hpp"

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void ScalarImpl::reset() {
  if (nvar() <= 0) return;

  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(options->riemann()->type() == "upwind",
              "[Scalar] Passive scalars currently require the upwind solver");

  // Zero is rejected rather than read as "off": it would pin every tracer to
  // zero and freeze transport, and it is far likelier to have meant "off".
  TORCH_CHECK(options->upper_bound() != 0.,
              "[Scalar] upper-bound = 0 pins the tracer to zero and freezes "
              "transport; use a negative value to disable the bound");
  TORCH_CHECK(
      options->upper_bound() <= 0. || pmb->phydro->options->eos()->limiter(),
      "[Scalar] upper-bound needs the eos limiter: it bounds r from "
      "above by the positivity of the complement, and enforcing one "
      "side alone leaves the other unbounded");

  pcoord = pmb->pcoord;
  // q and rho get the same scale-invariant weights: a uniform r stays uniform
  TORCH_CHECK(options->recon(), "[Scalar] Reconstruct options are null");
  auto recon = std::make_shared<ReconstructOptionsImpl>(*options->recon());
  recon->interp() = std::make_shared<InterpOptionsImpl>(*recon->interp());
  recon->interp()->scale() = true;
  precon = ReconstructImpl::create(recon, this);
  auto x1f =
      pcoord ? pcoord->x1_row_faces(recon->interp()->type()) : torch::Tensor();
  if (x1f.defined()) {
    precon1 = ReconstructImpl::create(recon, this, "recon1");
    precon1->set_x1_faces(x1f);
  } else {
    precon1 = precon;
  }
  priemann = RiemannSolverImpl::create(options->riemann(), this);

  if (options->thermo()) {
    pthermo = kintera::ThermoXImpl::create(options->thermo(), this);
  }
  if (options->kinetics()) {
    pkinetics = kintera::KineticsImpl::create(options->kinetics(), this);
  }

  auto nc1 = pcoord->options->nc1();
  auto nc2 = pcoord->options->nc2();
  auto nc3 = pcoord->options->nc3();

  freg.reset(nvar(), {nc3, nc2, nc1}, {nc1 > 1, nc2 > 1, nc3 > 1},
             torch::kFloat64, RegisterRequest());
  _flux1 = register_buffer("F1", freg.totals()[0]);
  _flux2 = register_buffer("F2", freg.totals()[1]);
  _flux3 = register_buffer("F3", freg.totals()[2]);
  _div = register_buffer(
      "D", torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64));
  // the census is a slot of the block's EfStatus, registered as this buffer
  register_buffer("positivity_hits",
                  pmb->pefs->bind(EfStatus::kScalarPosHits,
                                  torch::zeros({1}, torch::kInt64)));
}

torch::Tensor ScalarImpl::forward(double dt, torch::Tensor u,
                                  Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(other.count("hydro_u"),
              "[Scalar] hydro_u is required for passive scalar transport");
  auto ph = pmb->phydro;
  // the TOTAL dry density (anomaly storage adds the reference)
  auto rho = ph->dry_density(other.at("hydro_u")).to(u.options()).unsqueeze(0);
  // r = q_face / rho_face; q / rho of cell averages is second order (S177)
  auto face_r = [&](int dim) {
    auto recon = dim == DIM1 ? precon1 : precon;
    auto rf = recon->forward(rho, dim), r = u / rho;  // donor r if rf <= 0
    return torch::where(rf > 0., recon->forward(u, dim) / rf,
                        torch::stack({r.roll(1, dim), r}));
  };
  // the fluxes, by direction, and the hydro dry-mass rows they ride on: the
  // hydro register's mass.dry.limited@d, row IDN of its total as limit ends
  // (R-5; R.rst 4.2 EF c), not the total after the hydro forward
  torch::Tensor* flux[3] = {&_flux1, &_flux2, &_flux3};
  auto const& hreg = ph->flux_register();
  auto row = [&](int d) {
    if (!hreg.total(d).defined()) return torch::Tensor();
    TORCH_CHECK(hreg.keeps(d, "mass.dry.limited"), "[Scalar] the hydro ",
                "register keeps no mass.dry.limited@", d);
    return hreg.kept(d, "mass.dry.limited")[IDN];
  };
  torch::Tensor rlr[3], mass[3] = {row(1), row(2), row(3)};
  if (_flux1.defined()) {
    rlr[0] = face_r(DIM1);
    _flux1.set_data(priemann->forward(rlr[0][ILT], rlr[0][IRT], DIM1, mass[0]));
  }

  // cubed sphere: DIRECTION-SUFFIXED keys, as the hydro's, resolve the L/R
  // roles across a panel edge (an un-suffixed key swapped them at flip_flag)
  bool cubed = pmb->get_layout()->options->type() == "cubed-sphere";
  Variables send[3];
  std::vector<CommWorkPtr> works[3];
  SyncOptions so;
  so.cross_panel_only(true).interpolate(false).type(kScalar);
  for (int d : {1, 2}) {
    if (!flux[d]->defined()) continue;
    rlr[d] = face_r(3 - d);
    if (!cubed) continue;
    send[d]["scalar_wl:+"] = rlr[d][ILT];
    send[d]["scalar_wr:-"] = rlr[d][IRT];
    pmb->begin_exchange(send[d], so.dim(3 - d));
  }
  for (int d : {1, 2})
    if (cubed && flux[d]->defined())
      pmb->launch_exchange(so.dim(3 - d), works[d]);
  for (int d : {1, 2})
    if (cubed && flux[d]->defined())
      pmb->finalize_exchange(send[d], so.dim(3 - d), works[d]);

  // flux-covariance (#289): a scalar rides on the mass flux without its dry
  // row (R4) and adds its own tracer row (HydroImpl::scalar_flux_covariance)
  bool cov = false;
  for (int d : {1, 2}) {
    if (!flux[d]->defined()) continue;
    auto dcov = ph->scalar_flux_covariance(rlr[d][ILT], rlr[d][IRT], 3 - d);
    if (!dcov.defined()) {
      flux[d]->set_data(
          priemann->forward(rlr[d][ILT], rlr[d][IRT], 3 - d, mass[d]));
      continue;
    }
    cov = true;
    auto vel = mass[d] - ph->flux_covariance_dry(3 - d);
    flux[d]->set_data(priemann->forward(rlr[d][ILT], rlr[d][IRT], 3 - d, vel) +
                      dcov);
  }

  auto cells = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  std::vector<int> rows(nvar());
  std::iota(rows.begin(), rows.end(), 0);
  // P4: r <= b is positivity of the complement b*rho - s (flux g = b*F_mass -
  // F_s), a SECOND theta. It adds the CHANGE g - g*factor, never b*F_mass - g,
  // which perturbs every face and in float32 annihilates r near eps(b).
  if (options->upper_bound() > 0.) {
    auto b = options->upper_bound();
    torch::Tensor g[3];
    for (int d = 0; d < 3; ++d)
      if (flux[d]->defined()) g[d] = b * mass[d].unsqueeze(0) - *flux[d];
    auto theta =
        flux_positivity_theta(b * rho - u, g[0], g[1], g[2], pcoord, dt);
    pmb->pefs->slot(EfStatus::kScalarPosHits) +=
        (theta.index(cells) < 1.).sum();
    flux_positivity_sync_theta_(pmb, theta, "scalar_theta");
    for (int d = 0; d < 3; ++d) {
      if (!g[d].defined()) continue;
      auto factor = flux_positivity_factor(theta, g[d], 3 - d, pcoord);
      freg.add(d + 1, "scalar.upper_bound", rows,
               g[d] - (g[d] * factor).to(g[d].scalar_type()));
    }
  }

  // P5, last (the blend can export tracer a donor lacks): the shared limiter
  if (ph->options->eos()->limiter() || cov) {
    PositivityInputs in;
    in.u = u;
    in.rho_dry = rho;
    for (int d = 0; d < 3; ++d)
      if (flux[d]->defined())
        in.flux[d] = *flux[d], in.mass[d] = mass[d].unsqueeze(0);
    in.dt = dt;
    PositivityLimiter(pmb, pmb->pefs.get(), EfStatus::kScalarPosHits)
        .apply(freg, in, "scalar", "scalar.positivity.withhold");
  }

  _div.set_data(pcoord->divergence(_flux1, _flux2, _flux3));
  auto ds = torch::zeros_like(_div);
  ds.index(cells) = -dt * _div.index(cells);
  return ds;
}

std::shared_ptr<ScalarImpl> ScalarImpl::create(ScalarOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Scalar] Parent module is null");
  TORCH_CHECK(opts, "[Scalar] Options pointer is null");

  return p->register_module(name, Scalar(opts, p));
}

}  // namespace snap
