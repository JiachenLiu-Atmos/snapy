// C/C++
#include <utility>

// snap
#include <snap/snap.h>

#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "axial_momentum.hpp"

namespace snap {

// only guards 0/0 in a cell centred exactly on the axis (odd nx2): that cell
// holds no AAM, its correction is zero, and the AAM flux through its faces is
// the one leak of the flux form (one cell per polar panel)
static constexpr double kEps2 = 1.e-24;

AxialMomentum::AxialMomentum(HydroImpl const* hydro) : hydro_(hydro) {
  auto const& op = hydro->options;
  auto const& word = op->conserve_axial_momentum();
  TORCH_CHECK(word == "off" || word == "transport" || word == "exact",
              "[Hydro] conserve_axial_momentum must be off, transport or "
              "exact (exact replaces full and the closure key), got '",
              word, "'");
  auto pcoord = hydro->pmb ? hydro->pmb->pcoord : nullptr;
  if (word == "off" || !pcoord) return;
  pg_ = dynamic_cast<GnomonicEquiangleImpl const*>(pcoord.get());
  if (!pg_) return;  // no rotation data: the option does nothing
  auto const& c = op->coriolis();
  if (c && !(c->type() == "xyz" && c->omega2() == 0. && c->omega3() == 0.)) {
    TORCH_WARN_ONCE(
        "dynamics/conserve-axial-momentum is off for this run: it needs "
        "coriolis type xyz with omega2 = omega3 = 0 (the AAM is about the "
        "omega1 axis)");
    return;
  }
  mode_ = word == "exact" ? AxialMode::exact : AxialMode::transport;
  on_axis_ = (pg_->aam_kk_c.slice(0, pg_->kl(), pg_->ku() + 1)
                  .slice(1, pg_->jl(), pg_->ju() + 1) < 1.e-20)
                 .any()
                 .item<bool>();
  if (mode_ != AxialMode::exact) return;
  TORCH_CHECK(pcoord->options->interp_order() == 4,
              "dynamics/conserve-axial-momentum: exact needs cubic panel-edge "
              "ghosts, geometry/cells/interp_order: 4");
  TORCH_CHECK(!hydro->fp32() && !op->perturbation_flux(),
              "conserve-axial-momentum exact needs FP64 and no "
              "perturbation-flux");
  auto const& solver = op->riemann()->type();
  TORCH_CHECK(solver == "hllc" || solver == "lmars" || solver == "roe" ||
                  solver == "shallow-roe",
              "dynamics/conserve-axial-momentum: exact needs riemann.central, "
              "which riemann-solver type '",
              solver,
              "' does not provide; use hllc, lmars, roe or shallow-roe");
  // the closure would remove the viscous torque until diffusion posts to the
  // register (A.rst 3.7, D9)
  TORCH_CHECK(!op->diffusion() || op->diffusion()->nu_iso() == 0.,
              "dynamics/conserve-axial-momentum: exact is refused with "
              "forcing/diffusion nu_iso > 0: its closure would remove the "
              "viscous torque");
}

void AxialMomentum::request(RegisterRequest& req) const {
  if (mode_ != AxialMode::exact) return;
  if (pg_->options->nc2() > 1) req.keep.push_back("riemann.central@2");
  if (pg_->options->nc3() > 1) req.keep.push_back("riemann.central@3");
}

void AxialMomentum::rate_to_rows(torch::Tensor row2, torch::Tensor row3,
                                 torch::Tensor target) const {
  int si = pg_->il(), ei = pg_->iu() + 1;
  int sj = pg_->jl(), ej = pg_->ju() + 1;
  int sk = pg_->kl(), ek = pg_->ku() + 1;
  int n2 = ej - sj, n3 = ek - sk;

  auto kc = pg_->aam_k_c.slice(1, sk, ek).slice(2, sj, ej);    // [3, n3, n2, 1]
  auto klc = pg_->aam_kl_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
  auto kk = pg_->aam_kk_c.slice(0, sk, ek).slice(1, sj, ej);   // [n3, n2, 1]
  auto r = pg_->x1v.slice(0, si, ei);                          // [n1]
  auto vol =
      pg_->cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);

  // A cell centred on the axis (odd nx2, one per polar panel) has K_c = 0: a
  // cell-mean momentum there carries no AAM, since int (z x r) . m dV = m .
  // (z x int r dV) = 0, and the AAM its faces bring in is sub-cell rotation
  // the cell mean cannot represent. Its net rate is handed to its four
  // lateral neighbours in equal shares, so the total still telescopes. The
  // cell is interior to the panel (nx2 >= 3), so the shifts stay inside.
  if (on_axis_) {
    auto onaxis = (kk < 1.e-20).to(target.options());  // [n3, n2, 1]
    auto res = 0.25 * target * onaxis * vol;           // AAM rate per neighbour
    auto give = torch::zeros_like(target);
    give.slice(1, 1, n2) += res.slice(1, 0, n2 - 1);
    give.slice(1, 0, n2 - 1) += res.slice(1, 1, n2);
    give.slice(0, 1, n3) += res.slice(0, 0, n3 - 1);
    give.slice(0, 0, n3 - 1) += res.slice(0, 1, n3);
    target = target * (1. - onaxis) + give / vol;
  }

  // replace the K component of the rows: K . rows_new = target, with the
  // change along K_lower = r k_lower (a zonal acceleration)
  auto d2 = row2.slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  auto d3 = row3.slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  auto kd = r * (kc[VEL2] * d2 + kc[VEL3] * d3);
  auto lam = (target - kd) / (r * r * (kk + kEps2));
  d2 += lam * r * klc[VEL2];
  d3 += lam * r * klc[VEL3];
}

void AxialMomentum::x1_source(torch::Tensor row2, torch::Tensor row3,
                              torch::Tensor const& corr) const {
  int si = pg_->il(), ei = pg_->iu() + 1;
  int sj = pg_->jl(), ej = pg_->ju() + 1;
  int sk = pg_->kl(), ek = pg_->ku() + 1;
  int n1 = ei - si, n2 = ej - sj, n3 = ek - sk;

  auto target = torch::zeros({n3, n2, n1}, row2.options());
  if (corr.defined() && corr.numel() > 0) {
    auto kc = pg_->aam_k_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
    auto vol =
        pg_->cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
    auto c = corr.slice(1, sk, ek).slice(2, sj, ej).slice(3, si, ei);
    // T = P_lo - P_hi: k . (tangential momentum the solve gave the cell);
    // P = 0 at the bottom face, P_hi = P_lo - T up the column
    auto T = (kc[VEL2] * c[IVY] + kc[VEL3] * c[IVZ]) * vol;
    auto Phi = -T.cumsum(-1);
    auto Plo = Phi + T;
    auto r = pg_->x1v.slice(0, si, ei);
    auto rlo = pg_->x1f.slice(0, si, ei),
         rhi = pg_->x1f.slice(0, si + 1, ei + 1);
    target = ((rlo - r) * Plo - (rhi - r) * Phi) / vol;
  }
  rate_to_rows(row2, row3, target);
}

void AxialMomentum::flux_form(torch::Tensor div, torch::Tensor flux1,
                              torch::Tensor flux2, torch::Tensor flux3) const {
  int si = pg_->il(), ei = pg_->iu() + 1;
  int sj = pg_->jl(), ej = pg_->ju() + 1;
  int sk = pg_->kl(), ek = pg_->ku() + 1;
  int n1 = ei - si, n2 = ej - sj, n3 = ek - sk;

  // interior cells: rotation field, radius and volume
  auto kc = pg_->aam_k_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
  auto r = pg_->x1v.slice(0, si, ei);                        // [n1]
  auto vol =
      pg_->cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  // the exact area-weighted radius of a lateral face, 2 (r+^3 - r-^3) /
  // (3 (r+^2 - r-^2)): with the unit-radius face average of k this is the
  // exact face average of K = r k
  auto rm = pg_->x1f.slice(0, si, ei), rp = pg_->x1f.slice(0, si + 1, ei + 1);
  auto reff = 2. * (rp * rp * rp - rm * rm * rm) / (3. * (rp * rp - rm * rm));

  // divergence of the face AAM fluxes A_f F_f . K_f; the covariant face
  // fluxes of x1 faces are in the cell basis (flux2global1_), those of x2 and
  // x3 faces in the face's basis (flux2global2_, 3_). On the x1 faces K is
  // tangent, so the pressure flux carries no AAM and the cell value of k
  // serves; K_f is single-valued on every face, so the sums telescope.
  auto phidiv = torch::zeros({n3, n2, n1}, div.options());
  if (flux1.defined()) {
    auto A = pg_->face_area1().slice(0, sk, ek).slice(1, sj, ej).slice(2, si,
                                                                       ei + 1);
    auto f = flux1.slice(1, sk, ek).slice(2, sj, ej).slice(3, si, ei + 1);
    auto phi = A * pg_->x1f.slice(0, si, ei + 1) *
               (f[IVY] * kc[VEL2] + f[IVZ] * kc[VEL3]);
    phidiv += phi.slice(2, 1, n1 + 1) - phi.slice(2, 0, n1);
  }
  if (flux2.defined()) {
    auto A = pg_->face_area2()
                 .slice(0, sk, ek)
                 .slice(1, sj, ej + 1)
                 .slice(2, si, ei);
    auto kf = pg_->aam_k_f2.slice(1, sk, ek).slice(2, sj, ej + 1);
    auto f = flux2.slice(1, sk, ek).slice(2, sj, ej + 1).slice(3, si, ei);
    auto phi =
        A * reff * (f[IVX] * kf[VEL1] + f[IVY] * kf[VEL2] + f[IVZ] * kf[VEL3]);
    phidiv += phi.slice(1, 1, n2 + 1) - phi.slice(1, 0, n2);
  }
  if (flux3.defined()) {
    auto A = pg_->face_area3()
                 .slice(0, sk, ek + 1)
                 .slice(1, sj, ej)
                 .slice(2, si, ei);
    auto kf = pg_->aam_k_f3.slice(1, sk, ek + 1).slice(2, sj, ej);
    auto f = flux3.slice(1, sk, ek + 1).slice(2, sj, ej).slice(3, si, ei);
    auto phi =
        A * reff * (f[IVX] * kf[VEL1] + f[IVY] * kf[VEL2] + f[IVZ] * kf[VEL3]);
    phidiv += phi.slice(0, 1, n3 + 1) - phi.slice(0, 0, n3);
  }
  phidiv /= vol;

  rate_to_rows(div[IVY], div[IVZ], phidiv);
}

torch::Tensor AxialMomentum::rate(torch::Tensor flux2, torch::Tensor flux3,
                                  double omega, int ny) const {
  int si = pg_->il(), ei = pg_->iu() + 1;
  int sj = pg_->jl(), ej = pg_->ju() + 1;
  int sk = pg_->kl(), ek = pg_->ku() + 1;
  int n1 = ei - si, n2 = ej - sj, n3 = ek - sk;

  auto kk = pg_->aam_kk_c.slice(0, sk, ek).slice(1, sj, ej);  // cos^2(lat)
  auto r = pg_->x1v.slice(0, si, ei);
  auto vol =
      pg_->cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  auto s2c = r * r * kk;  // [n3, n2, n1]

  // total mass flux rows: dry plus the species
  auto mass = [&](torch::Tensor const& f) {
    auto m = f[IDN].clone();
    if (ny > 0) m += f.narrow(0, ICY, ny).sum(0);
    return m;
  };

  // -(Omega / V) sum_f M_f (s_f^2 - s_c^2), M_f the outward face mass flux
  auto rate =
      torch::zeros({n3, n2, n1}, (flux2.defined() ? flux2 : flux3).options());
  if (flux2.defined()) {
    auto A = pg_->face_area2()
                 .slice(0, sk, ek)
                 .slice(1, sj, ej + 1)
                 .slice(2, si, ei);
    auto M =
        A * mass(flux2.slice(1, sk, ek).slice(2, sj, ej + 1).slice(3, si, ei));
    auto s2f = r * r * pg_->aam_c2_f2.slice(0, sk, ek).slice(1, sj, ej + 1);
    rate += M.slice(1, 1, n2 + 1) * (s2f.slice(1, 1, n2 + 1) - s2c) -
            M.slice(1, 0, n2) * (s2f.slice(1, 0, n2) - s2c);
  }
  if (flux3.defined()) {
    auto A = pg_->face_area3()
                 .slice(0, sk, ek + 1)
                 .slice(1, sj, ej)
                 .slice(2, si, ei);
    auto M =
        A * mass(flux3.slice(1, sk, ek + 1).slice(2, sj, ej).slice(3, si, ei));
    auto s2f = r * r * pg_->aam_c2_f3.slice(0, sk, ek + 1).slice(1, sj, ej);
    rate += M.slice(0, 1, n3 + 1) * (s2f.slice(0, 1, n3 + 1) - s2c) -
            M.slice(0, 0, n3) * (s2f.slice(0, 0, n3) - s2c);
  }
  return -omega * rate / vol;
}

void AxialMomentum::add_torque(torch::Tensor force, torch::Tensor const& w,
                               double omega) {
  if (mode_ != AxialMode::exact || !armed_) return;
  // F_d = total(d) - kept central: the limited mass flux minus the solver's
  // pre-limiter central part (A-r1, A-r2); the x1 faces stay in cell form
  auto const& reg = hydro_->flux_register();
  auto const& op = hydro_->options;
  bool on2 = !op->disable_flux_x2() && reg.keeps(2, "riemann.central");
  bool on3 = !op->disable_flux_x3() && reg.keeps(3, "riemann.central");
  if (!on2 && !on3) return;
  auto none = torch::Tensor();
  auto target = rate(on2 ? reg.total(2) - reg.kept(2, "riemann.central") : none,
                     on3 ? reg.total(3) - reg.kept(3, "riemann.central") : none,
                     omega, ny_);
  if (closure) {
    // R = sum V (K . F_cell - rate(M_central)), rate(M_central) = rate(M)
    // - rate(M - M_central); I = sum V rho s^2 (interior, this block)
    auto in =
        hydro_->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto kj = std::vector<torch::indexing::TensorIndex>{in[0], in[1]};
    auto r = pg_->x1v.index({in[2]}), kk = pg_->aam_kk_c.index(kj);
    auto vol = pg_->cell_volume().index(in), rho = w[IDN].index(in);
    auto kf = r * (pg_->aam_k_c[VEL2].index(kj) * force[VEL2].index(in) +
                   pg_->aam_k_c[VEL3].index(kj) * force[VEL3].index(in));
    auto rate_c =
        rate(on2 ? reg.total(2) : none, on3 ? reg.total(3) : none, omega, ny_) -
        target;
    sums_ = torch::stack(
                {(vol * (kf - rate_c)).sum(), (vol * rho * r * r * kk).sum()})
                .to(torch::kFloat64);  // stays on the device
    rho_ = rho.clone();
  }
  auto z2 = torch::zeros_like(force[VEL2]), z3 = torch::zeros_like(force[VEL3]);
  rate_to_rows(z2, z3, target);  // K . z = target, along K
  force[VEL2] += z2;
  force[VEL3] += z3;
}

torch::Tensor AxialMomentum::aam(torch::Tensor const& row2,
                                 torch::Tensor const& row3) const {
  auto in = hydro_->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto kj = std::vector<torch::indexing::TensorIndex>{in[0], in[1]};
  auto wide = row2.options().dtype(torch::kFloat64);
  auto k2 = pg_->aam_k_c[VEL2].index(kj).to(wide),
       k3 = pg_->aam_k_c[VEL3].index(kj).to(wide);
  return (pg_->cell_volume().index(in).to(wide) * pg_->x1v.index({in[2]}) *
          (k2 * row2.index(in).to(wide) + k3 * row3.index(in).to(wide)))
      .sum();
}

std::vector<int> AxialClosureFixer::rows() const { return {IVY, IVZ}; }

torch::Tensor AxialClosureFixer::partial_sums() const {
  return aam_->sums_.defined()
             ? aam_->sums_.clone()
             : torch::zeros({2}, aam_->hydro_->pmb->pcoord->x1v.options().dtype(
                                     torch::kFloat64));
}

torch::Tensor AxialClosureFixer::apply(torch::Tensor u,
                                       torch::Tensor const& global_sums,
                                       double dt, double stage_weight,
                                       FluxRegister& reg) const {
  auto rho = std::exchange(aam_->rho_, torch::Tensor());
  auto part = std::exchange(aam_->sums_, torch::Tensor());
  if (!rho.defined()) return torch::Tensor();
  auto pg = aam_->pg_;
  auto g = global_sums.to(u.device(), torch::kFloat64);
  auto R = g[0], I = g[1];
  auto ok = (I > 0.).logical_and(torch::isfinite(R));
  auto in =
      aam_->hydro_->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto kj = std::vector<torch::indexing::TensorIndex>{in[0], in[1]};
  auto r = pg->x1v.index({in[2]});
  // d(rho u_east) = -rho s R dt / I with this stage's weight, along K
  auto omega = -stage_weight * dt * R / I;
  auto z2 = torch::zeros_like(u[IVY]), z3 = torch::zeros_like(u[IVZ]);
  aam_->rate_to_rows(z2, z3, omega * rho * r * r * pg->aam_kk_c.index(kj));
  auto vol_du = torch::zeros({u.size(0)}, g.options());
  auto vol = pg->cell_volume();
  for (auto [row, z] : {std::pair<int, torch::Tensor>{IVY, z2}, {IVZ, z3}}) {
    auto ur = u[row];
    ur.copy_(torch::where(ok, ur + z, ur));
    vol_du[row].copy_(torch::where(ok, (vol * z).sum(), torch::zeros_like(R)));
  }
  reg.cell_amount(name(), vol_du, 1.0, AmountKind::fixer);
  auto zero = torch::zeros_like(R);
  return torch::where(ok, -stage_weight * dt * R * part[1] / I, zero);
}

}  // namespace snap
