// C/C++
#include <limits>
#include <numeric>
#include <utility>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

#include "flux_positivity.hpp"
#include "flux_register.hpp"

namespace snap {

double machine_epsilon(c10::ScalarType dtype) {
  return dtype == torch::kFloat ? std::numeric_limits<float>::epsilon()
                                : std::numeric_limits<double>::epsilon();
}

double positivity_roundoff(c10::ScalarType dtype) {
  double ulp = dtype == torch::kFloat ? kPositivityRoundoffUlpFloat
                                      : kPositivityRoundoffUlp;
  return ulp * machine_epsilon(dtype);
}

double species_repair_mark(c10::ScalarType dtype) {
  double ulp = dtype == torch::kFloat ? kSpeciesRepairMarkUlpFloat
                                      : kSpeciesRepairMarkUlp;
  return ulp * machine_epsilon(dtype);
}

namespace {
// the faces lo..hi+1 the divergence consumes along tensor axis dim (3: x1)
std::pair<int, int> faces(Coordinate const& pcoord, int dim) {
  return dim == 3   ? std::pair{pcoord->il(), pcoord->iu()}
         : dim == 2 ? std::pair{pcoord->jl(), pcoord->ju()}
                    : std::pair{pcoord->kl(), pcoord->ku()};
}
}  // namespace

torch::Tensor flux_positivity_theta(torch::Tensor const& u,
                                    torch::Tensor const& flux1,
                                    torch::Tensor const& flux2,
                                    torch::Tensor const& flux3,
                                    Coordinate const& pcoord, double dt,
                                    torch::Tensor* drain_out) {
  auto out = torch::zeros_like(u);

  // outgoing A*F over the faces the divergence uses: the upper face i+1
  // drains cell i where F > 0, the lower face i where F < 0
  torch::Tensor const* flux[3] = {&flux1, &flux2, &flux3};
  for (int d = 0; d < 3; ++d) {
    if (!flux[d]->defined()) continue;
    int dim = 3 - d;
    auto [lo, hi] = faces(pcoord, dim);
    int nf = hi - lo + 2;  // faces lo..hi+1
    auto area = d == 0   ? pcoord->face_area1(lo, lo + nf)
                : d == 1 ? pcoord->face_area2(lo, lo + nf)
                         : pcoord->face_area3(lo, lo + nf);
    auto af = area * flux[d]->slice(dim, lo, lo + nf);
    out.slice(dim, lo, hi + 1) +=
        af.slice(dim, 1, nf).relu() + af.slice(dim, 0, nf - 1).neg().relu();
  }

  // 1 where nothing drains, ghosts included; stop 4096 ulp short of zero
  double margin = kPositivityRoundoffUlp * machine_epsilon(u.scalar_type());
  auto avail = u.relu() * pcoord->cell_volume() * (1. - margin);
  auto drain = out.mul_(dt);
  if (drain_out) *drain_out = drain;
  // finfo(dtype).tiny: 1e-300 was 0 in float32 (only where() discards it)
  double tiny = drain.scalar_type() == torch::kFloat
                    ? std::numeric_limits<float>::min()
                    : std::numeric_limits<double>::min();
  return torch::where(drain > 0.,
                      (avail / drain.clamp_min(tiny)).clamp_max(1.0),
                      torch::ones_like(u));
}

torch::Tensor flux_positivity_share(torch::Tensor const& q) {
  auto low = torch::minimum(q.roll(1, 3), q.roll(-1, 3));
  for (int dim : {2, 1}) {
    if (q.size(dim) > 1)
      low =
          torch::minimum(low, torch::minimum(q.roll(1, dim), q.roll(-1, dim)));
  }
  return torch::where(q > 0., (low / q).clamp(0., 1.), torch::zeros_like(q));
}

torch::Tensor flux_positivity_factor(torch::Tensor const& theta,
                                     torch::Tensor const& flux, int dim,
                                     Coordinate const& pcoord) {
  auto [lo, hi] = faces(pcoord, dim);
  auto f = flux.slice(dim, lo, hi + 2);  // faces lo..hi+1
  auto factor = torch::ones(flux.sizes(), theta.options());
  factor.slice(dim, lo, hi + 2)
      .copy_(torch::where(f > 0., theta.slice(dim, lo - 1, hi + 1),
                          theta.slice(dim, lo, hi + 2)));
  return factor;
}

void flux_positivity_carry_(torch::Tensor const& theta,
                            torch::Tensor const& hspec,
                            torch::Tensor const& vel, FluxRegister& reg,
                            Coordinate const& pcoord,
                            torch::Tensor const& fsed1,
                            torch::Tensor const& vface2,
                            torch::Tensor const& vface3) {
  int ny = theta.size(0);
  for (int d = 1; d <= 3; ++d) {
    auto flux = reg.total(d);
    if (!flux.defined()) continue;
    int dim = 4 - d;
    auto [lo, hi] = faces(pcoord, dim);
    auto const& vface = d == 2 ? vface2 : d == 3 ? vface3 : torch::Tensor();
    auto f = flux.narrow(0, ICY, ny).slice(dim, lo, hi + 2);  // faces lo..hi+1
    auto lower = [&](auto const& x) { return x.slice(dim, lo - 1, hi + 1); };
    auto upper = [&](auto const& x) { return x.slice(dim, lo, hi + 2); };
    auto vlo = vface.defined() ? upper(vface[0]) : lower(vel);
    auto vhi = vface.defined() ? upper(vface[1]) : upper(vel);
    auto share = 1. - torch::where(f > 0., lower(theta), upper(theta));
    // -c on faces lo..hi+1, -0. elsewhere, in the in-place dtype: x - c, x
    auto post = [&](std::vector<int> const& rows, torch::Tensor const& c) {
      auto shape = flux.sizes().vec();
      shape[0] = rows.size();
      auto type = torch::promote_types(flux.scalar_type(), c.scalar_type());
      auto inc = torch::full(shape, -0., c.options().dtype(type));
      inc.slice(dim, lo, hi + 2).copy_(-c);
      reg.add(d, "positivity.carry", rows, inc);
    };
    // each part of the face flux keeps the energy and momentum of its donor
    auto withhold = [&](torch::Tensor const& part) {
      auto up = part > 0.;  // the donor is the lower cell
      auto dm = share * part;
      post({IPR},
           (dm * torch::where(up, lower(hspec), upper(hspec))).sum(0, true));
      post({IVX, IVY, IVZ},
           (dm * up).sum(0) * vlo + (dm * up.logical_not()).sum(0) * vhi);
    };
    if (d == 1 && fsed1.defined()) {
      auto sed = fsed1.slice(dim, lo, hi + 2);
      withhold(f - sed);
      withhold(sed);
    } else {
      withhold(f);
    }
  }
}

torch::Tensor flux_positivity_relax(torch::Tensor const& theta,
                                    torch::Tensor const& td,
                                    torch::Tensor const& phi) {
  return torch::where((td < 1.) & (td > 0.),
                      theta + phi * ((theta / td).clamp_max(1.) - theta),
                      theta);
}

void flux_positivity_sync_theta_(MeshBlockImpl const* pmb,
                                 torch::Tensor const& theta,
                                 std::string const& key) {
  Variables tvars;
  tvars[key] = theta;
  SyncOptions topts;
  topts.interpolate(false).type(kScalar);
  pmb->exchange(tvars, topts);

  BoundaryFuncOptions bops;
  bops.nghost(pmb->pcoord->options->nghost());
  bops.type(kScalar);
  for (int i = 0; i < pmb->options->bfuncs().size(); ++i) {
    if (pmb->options->bfuncs()[i] == nullptr) continue;
    pmb->options->bfuncs()[i](theta, 3 - i / 2, bops);
  }
}

PositivityLimiter::PositivityLimiter(MeshBlockImpl const* pmb, EfStatus* status,
                                     EfStatus::Slot hits)
    : pmb_(pmb), status_(status), hits_(hits) {}

torch::Tensor PositivityLimiter::theta(PositivityInputs const& in,
                                       torch::Tensor* drain) const {
  auto const& pc = pmb_->pcoord;
  auto theta = flux_positivity_theta(in.u, in.flux[0], in.flux[1], in.flux[2],
                                     pc, in.dt, drain);
  // where the dry air drains too, drain as far as the poorer neighbour repays
  auto td = flux_positivity_theta(in.rho_dry, in.mass[0], in.mass[1],
                                  in.mass[2], pc, in.dt);
  auto phi = flux_positivity_share(in.u / in.rho_dry);
  // nothing refills settling mass: a cell it drains keeps the plain theta
  if (in.fsed1.defined()) {
    auto out = in.fsed1.clamp_min(0.).roll(-1, 3) + (-in.fsed1).clamp_min(0.);
    phi.masked_fill_(out > 0., 0.);
  }
  return flux_positivity_relax(theta, td, phi);
}

void PositivityLimiter::census_(PositivityInputs const& in,
                                torch::Tensor const& theta,
                                torch::Tensor const& drain) const {
  auto cells = pmb_->part({0, 0, 0}, PartOptions().exterior(false));
  if (hits_ != EfStatus::kPosHits) {
    status_->slot(hits_) += (theta.index(cells) < 1.).sum();
    return;
  }
  auto wide = theta.is_mps() ? torch::kFloat32 : torch::kFloat64;
  auto ti = theta.index(cells).to(wide);
  status_->slot(hits_) += (ti < 1.).sum();
  // severe: theta < 0.9 and withheld mass above round-off of the gas (#256)
  auto cell_mass = (in.rho_dry[0] + in.u.sum(0)) * pmb_->pcoord->cell_volume();
  auto withheld = ((1. - theta) * drain).index(cells).to(wide);
  auto above = withheld > positivity_roundoff(in.u.scalar_type()) *
                              cell_mass.unsqueeze(0).index(cells).to(wide);
  status_->slot(EfStatus::kPosSevere) += ((ti < 0.9) & above).sum();
  auto& tmin = status_->slot(EfStatus::kPosMin);
  torch::minimum_out(tmin, tmin, ti.min());
}

void PositivityLimiter::sync_(PositivityInputs const& in,
                              torch::Tensor const& theta,
                              std::string const& key) const {
  flux_positivity_sync_theta_(pmb_, theta, key + "_theta");
  if (in.hspec.defined())
    flux_positivity_sync_theta_(pmb_, in.hspec, key + "_hspec");
}

void PositivityLimiter::post_(FluxRegister& reg, PositivityInputs const& in,
                              torch::Tensor const& theta,
                              char const* withhold) const {
  auto const& pc = pmb_->pcoord;
  if (in.hspec.defined())
    flux_positivity_carry_(theta, in.hspec, in.vel, reg, pc, in.fsed1,
                           in.vface[0], in.vface[1]);
  std::vector<int> rows(in.u.size(0));
  std::iota(rows.begin(), rows.end(), in.first);
  for (int d = 1; d <= 3; ++d) {
    auto const& f = in.flux[d - 1];
    if (!f.defined()) continue;
    auto factor = flux_positivity_factor(theta, f, 4 - d, pc);
    if (d == 1 && hits_ == EfStatus::kPosHits) {
      // the flux theta removes (>= 0 while the ghost theta is in [0,1])
      auto cells = pmb_->part({0, 0, 0}, PartOptions().exterior(false));
      auto wide = theta.is_mps() ? torch::kFloat32 : torch::kFloat64;
      auto pre = f.abs();
      auto cut = pre - (f * factor).to(f.scalar_type()).abs();
      status_->slot(EfStatus::kLimFlux) += pre.index(cells).to(wide).sum();
      status_->slot(EfStatus::kLimCut) += cut.index(cells).to(wide).sum();
    }
    reg.scale(d, withhold, rows, factor);
  }
}

void PositivityLimiter::apply(FluxRegister& reg, PositivityInputs const& in,
                              std::string const& key,
                              char const* withhold) const {
  torch::Tensor drain;
  auto theta = this->theta(in, &drain);
  census_(in, theta, drain);
  sync_(in, theta, key);
  post_(reg, in, theta, withhold);
}

}  // namespace snap
