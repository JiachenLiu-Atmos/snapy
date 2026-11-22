// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/hydro/sweep.hpp>
#include <snap/hydro/x1_mass_covariance.hpp>
#include <snap/mesh/precision_policy.hpp>

#include "conversion.hpp"
#include "reconstruct.hpp"
#include "x1_reference.hpp"

namespace snap {

torch::Tensor WellBalancedX1::forward(torch::Tensor wx1,
                                      ReferenceFaces const& ref,
                                      ReconstructImpl& recon,
                                      RegisterMonitor& monitor) const {
  constexpr int DIM1 = 3;
  bool split = pressure0.defined();
  auto const& psf_lo = ref.psf_lo;
  auto const& pref = ref.pref;
  auto const& dsf = ref.dsf_lr;
  auto const& dref = ref.dref;
  auto fill4 = [&](torch::Tensor a, int c, bool pos, torch::Tensor r) {
    if (wall4[0] || wall4[1])  // key 2 and wall-less blocks never enter
      conv4->fill_walls(a, c, pos, r, {}, conv4_trips);
  };

  auto pressure = wx1[IPR].clone();
  auto density = wx1[IDN].clone();

  if (split)
    wx1[IPR].copy_((pressure0 - ref.cell0) + (pressure_delta - ref.cell_delta));
  else
    wx1[IPR] -= pref;
  if (policy->fp32()) wx1[IPR] -= ref.gauge;
  wx1[IDN] -= dref;

  // Even-parity ghost perturbations at the walls: p'(is-m) =
  // p'(is+m-1), rho' likewise. Only apply this at physical x1 walls.
  int ng = coord->options->nghost();
  int is = coord->il();
  int iu = coord->iu();
  for (int c : {(int)IPR, (int)IDN}) {
    if (mirror[0]) {
      wx1[c].narrow(-1, is - ng, ng).copy_(wx1[c].narrow(-1, is, ng).flip(-1));
    }

    if (mirror[1]) {
      wx1[c]
          .narrow(-1, iu + 1, ng)
          .copy_(wx1[c].narrow(-1, iu + 1 - ng, ng).flip(-1));
    }
    // key 4 rho', p' parity ghosts (wall4_rows); runs only with gravity
    fill4(wx1[c], c == IDN ? 2 : 3, true, c == IDN ? dref : pref);
  }

  torch::Tensor velocity;
  if (covariance) {
    velocity = wx1[IVX].clone();
    auto x1v = coord->x1v.to(wx1.device(), wx1.scalar_type());
    auto dx1f = coord->dx1f.to(wx1.device(), wx1.scalar_type());
    bool stretched = coord->options->stretched_x1();
    auto rho_1 = x1_cov_d1_centred(density, x1v, stretched);
    // next to a reflecting wall the even ghost density is not the
    // stratified column: one-sided second order in the first/last cell
    if (mirror[0])
      rho_1.select(-1, is).copy_(
          x1_cov_one_sided(density, x1v, is, 1, stretched));
    if (mirror[1])
      rho_1.select(-1, iu).copy_(
          x1_cov_one_sided(density, x1v, iu, -1, stretched));
    auto w_1 = x1_cov_d1_centred(velocity, x1v, stretched);
    wx1[IVX] -= dx1f * dx1f / 12. * rho_1 * w_1 / density;
    // ghosts at reflecting walls: odd mirror of the corrected interior
    if (mirror[0])
      wx1[IVX]
          .narrow(-1, is - ng, ng)
          .copy_(-wx1[IVX].narrow(-1, is, ng).flip(-1));
    if (mirror[1])
      wx1[IVX]
          .narrow(-1, iu + 1, ng)
          .copy_(-wx1[IVX].narrow(-1, iu + 1 - ng, ng).flip(-1));
  }
  if (wall4[0] || wall4[1]) {
    if (!velocity.defined()) velocity = wx1[IVX].clone();
    fill4(wx1[IVX], 1, false, {});
  }

  // floor=false: reconstruction-stage floors would clamp legitimately
  // negative perturbations.
  auto wtmp = recon.forward(
      policy->fp32() ? policy->cast(Quantity::kFlux, wx1) : wx1, DIM1,
      /*floor=*/false);
  post_face_floors(recon, monitor);

  wx1[IPR].copy_(pressure);
  wx1[IDN].copy_(density);
  if (velocity.defined()) wx1[IVX].copy_(velocity);
  // Restore full face pressure/density; floor any nonlinear-WENO overshoot
  // that would go non-positive (the references are tiny near the top) back
  // to the reference. At rest the perturbation is ~0 so the floor never
  // fires and well-balancing is preserved.
  auto pl = wtmp[ILT][IPR] + psf_lo;
  auto pr = wtmp[IRT][IPR] + psf_lo;
  if (policy->fp32()) {
    pl += ref.gauge;
    pr += ref.gauge;
  }
  // FaceFloor: the WB fallbacks below count as floors, faces is..iu+1
  auto fallback = [&](torch::Tensor const& x) {
    return (x > 0.).logical_not().narrow(-1, is, iu + 2 - is);
  };
  monitor.count("face_floor.p", fallback(pl));
  monitor.count("face_floor.p", fallback(pr));
  if (split) {
    wtmp[ILT][IPR].copy_(torch::where(
        pl > 0., wtmp[ILT][IPR] + ref.face_delta,
        policy->fp32() ? ref.face_delta - ref.gauge : ref.face_delta));
    wtmp[IRT][IPR].copy_(torch::where(
        pr > 0., wtmp[IRT][IPR] + ref.face_delta,
        policy->fp32() ? ref.face_delta - ref.gauge : ref.face_delta));
  } else {
    wtmp[ILT][IPR].copy_(torch::where(pl > 0., pl, psf_lo));
    wtmp[IRT][IPR].copy_(torch::where(pr > 0., pr, psf_lo));
  }

  auto dl = wtmp[ILT][IDN] + dsf[ILT];
  auto dr = wtmp[IRT][IDN] + dsf[IRT];
  monitor.count("face_floor.rho", fallback(dl));
  monitor.count("face_floor.rho", fallback(dr));
  // positivity fallback: the adjacent cell's density, shifted with edge
  // replication, not torch::roll (lane-c-notes.md)
  auto n1 = density.size(-1);
  auto rho_below =
      torch::cat({density.narrow(-1, 0, 1), density.narrow(-1, 0, n1 - 1)}, -1);
  wtmp[ILT][IDN].copy_(torch::where(dl > 0., dl, rho_below));
  wtmp[IRT][IDN].copy_(torch::where(dr > 0., dr, density));
  // key 4, after the floors: the wall face mirrors its owned side, v* = 0
  for (int s : {0, 1}) {
    if (!wall4[s]) continue;
    int i = s ? iu + 1 : is, o = s ? ILT : IRT;
    wtmp[1 - o].select(-1, i).copy_(wtmp[o].select(-1, i));
    wtmp[1 - o][IVX].select(-1, i).neg_();
  }
  return wtmp;
}

}  // namespace snap
