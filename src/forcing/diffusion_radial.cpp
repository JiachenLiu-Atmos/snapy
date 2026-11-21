#include <cmath>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {
void DiffusionImpl::begin_split(torch::Tensor u, double dt) {
  split_initial_ = u.clone();
  radial_conduction_step(u, 0.5 * dt);
}
void DiffusionImpl::finish_split(torch::Tensor u, double dt) {
  radial_conduction_step(u, 0.5 * dt);
}
// ruling A-ph3.radial_step_field: Hydro requested the x1 step field when
// radial conduction is on, so the post is unguarded
void DiffusionImpl::begin_split(torch::Tensor u, double dt, FluxRegister& reg) {
  split_initial_ = u.clone();
  auto flux = radial_conduction_step(u, 0.5 * dt);
  if (flux.defined()) reg.post_split(1, "diffusion.radial", flux, 0.5, dt);
}
void DiffusionImpl::finish_split(torch::Tensor u, double dt,
                                 FluxRegister& reg) {
  auto flux = radial_conduction_step(u, 0.5 * dt);
  if (flux.defined()) reg.post_split(1, "diffusion.radial", flux, 0.5, dt);
}
void DiffusionImpl::restore_split(torch::Tensor u) const {
  TORCH_CHECK(split_initial_.defined(),
              "[Diffusion] No split-step state to restore.");
  u.copy_(split_initial_);
}

torch::Tensor DiffusionImpl::radial_conduction_step(torch::Tensor u,
                                                    double dt) {
  if (!options->implicit_radial() || options->kappa_iso() == 0. || dt == 0.)
    return torch::Tensor();
  TORCH_CHECK(std::isfinite(dt) && dt > 0.,
              "[Diffusion] Invalid radial step dt.");
  check_profiles();
  auto c = phydro->pmb->pcoord;
  TORCH_CHECK(c->options->type() == "gnomonic-equiangle" &&
                  phydro->peos->options->type() == "ideal-gas",
              "[Diffusion] Radial implicit solve requires cubed sphere and dry "
              "ideal gas.");
  auto sl = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto w = phydro->peos->compute("U->W", {u.clone()});
  auto alltemp = phydro->peos->compute("W->T", {w});
  auto temp = alltemp.index(sl);
  auto capacity =
      (w[IDN] * phydro->peos->specific_heat_cv(w, alltemp)).index(sl);
  auto k = options->dynamic() ? torch::full_like(temp, options->kappa_iso())
                              : options->kappa_iso() * capacity;
  int n = temp.size(-1), start = c->il();
  if (kappa_scale_.defined())
    k.mul_(kappa_scale_.to(u.options())
               .slice(0, start, start + n)
               .view({1, 1, -1}));
  auto r = c->x1v.to(u.options()).slice(0, start, start + n);
  auto rf = c->x1f.to(u.options()).slice(0, start, start + n + 1);
  auto rm = rf.slice(0, 0, n), rp = rf.slice(0, 1, n + 1);
  // Use the same radial volume measure as Coordinate::cell_volume, including
  // its cancellation-safe global cell widths in a very thin shell.
  auto h = c->dx1f.to(u.options()).slice(0, start, start + n);
  auto mass = capacity * h * (rp.square() + rp * rm + rm.square()) / 3.;
  auto conductance =
      torch::zeros({temp.size(0), temp.size(1), n + 1}, u.options());
  auto rl = r.slice(0, 0, n - 1), rr = r.slice(0, 1, n);
  conductance.narrow(-1, 1, n - 1)
      .copy_(0.5 * (k.narrow(-1, 0, n - 1) + k.narrow(-1, 1, n - 1)) * rl * rr /
             (rr - rl));
  auto source = torch::zeros_like(temp);
  for (int side = 0; side < 2; ++side) {
    double tw =
        side ? options->temperature_outer() : options->temperature_inner();
    if (tw == 0.) continue;
    int near = side ? n - 1 : 0, next = side ? n - 2 : 1, face = side ? n : 0;
    // linear in position over the cell widths (#280; 1.5, -0.5 if uniform)
    auto g = h[near] / (h[near] + h[next]);
    auto kn = k.select(-1, near), kf = (1. + g) * kn - g * k.select(-1, next);
    kf = torch::where(kf >= 0., kf, kn);
    auto value = kf * rf[face] * r[near] / torch::abs(r[near] - rf[face]);
    conductance.select(-1, face).copy_(value);
    source.select(-1, near).add_(value * tw);
  }
  // SDIRK(2,2), gamma=1-1/sqrt(2): second order and L-stable.
  // Frozen density and ideal-gas cv make this an exact linear column system.
  constexpr double gamma = 1. - 0.7071067811865475244;
  double step = gamma * dt;
  auto lower = -step * conductance.narrow(-1, 0, n);
  auto upper = -step * conductance.narrow(-1, 1, n);
  auto diag = mass - lower - upper;
  // Factor once, solve both stages. Horizontal columns are batched tensors.
  auto pivot = diag.clone(), mult = torch::zeros_like(diag);
  for (int i = 1; i < n; ++i) {
    mult.select(-1, i).copy_(lower.select(-1, i) / pivot.select(-1, i - 1));
    pivot.select(-1, i).sub_(mult.select(-1, i) * upper.select(-1, i - 1));
  }
  auto solve = [&](torch::Tensor rhs_temp) {
    auto rhs = mass * rhs_temp + step * source;
    for (int i = 1; i < n; ++i)
      rhs.select(-1, i).sub_(mult.select(-1, i) * rhs.select(-1, i - 1));
    rhs.select(-1, n - 1).div_(pivot.select(-1, n - 1));
    for (int i = n - 2; i >= 0; --i)
      rhs.select(-1, i).copy_(
          (rhs.select(-1, i) - upper.select(-1, i) * rhs.select(-1, i + 1)) /
          pivot.select(-1, i));
    return rhs;
  };
  auto first = solve(temp);
  auto next = solve(temp + (1. - gamma) / gamma * (first - temp));
  TORCH_CHECK(torch::isfinite(next).all().item<bool>() &&
                  next.min().item<double>() > 0.,
              "[Diffusion] Radial SDIRK step produced nonpositive/nonfinite "
              "temperature; reduce dt.");
  u[IPR].index(sl).add_(capacity * (next - temp));
  // the step's x1 heat flux density (A.rst 3.7). Per unit solid angle the
  // step moves dt [(1 - gamma) Q(first) + gamma Q(next)] up through face f,
  // Q_f = conductance_f (T below - T above), a wall's T_w outside; a face
  // of area A_f and solid angle Omega = V / (mass / capacity) then carries
  // F_f = Omega Q_f / A_f, so -dt div(F) is the increment above
  auto q = [&](torch::Tensor const& t) {
    auto out = torch::zeros_like(conductance);
    out.narrow(-1, 1, n - 1)
        .copy_(conductance.narrow(-1, 1, n - 1) *
               (t.narrow(-1, 0, n - 1) - t.narrow(-1, 1, n - 1)));
    out.select(-1, 0).copy_(conductance.select(-1, 0) *
                            (options->temperature_inner() - t.select(-1, 0)));
    out.select(-1, n).copy_(
        conductance.select(-1, n) *
        (t.select(-1, n - 1) - options->temperature_outer()));
    return out;
  };
  auto qbar = (1. - gamma) * q(first) + gamma * q(next);
  auto vol = c->cell_volume().to(u.options()).index(sl);
  auto omega = (vol * capacity / mass).narrow(-1, 0, 1);
  int ks = c->kl(), ke = c->ku() + 1, js = c->jl(), je = c->ju() + 1;
  auto area = c->face_area1()
                  .to(u.options())
                  .slice(0, ks, ke)
                  .slice(1, js, je)
                  .slice(2, start, start + n + 1);
  auto flux =
      torch::zeros({u.size(0), u.size(1), u.size(2), u.size(3)}, u.options());
  flux[IPR]
      .slice(0, ks, ke)
      .slice(1, js, je)
      .slice(2, start, start + n + 1)
      .copy_(omega * qbar / area);
  return flux;
}
}  // namespace snap
