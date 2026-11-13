#pragma once

// C/C++
#include <cmath>
#include <limits>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "forward_sweep_impl.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define MASS(n, i) mass_fix[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Split form of the MS-VIC redistribution around ForwardSweep
// (forward_sweep_impl.h). Backward substitution and the donor-upwinded
// constituent transport both retain their required serial layer dependency;
// the final per-cell update is exposed separately so CUDA can parallelize it.
// The Woodbury solve (K = 2) passes its unit solves z0 and z1 too: one pass
// reads each a[i] once for all of them, with each vector's arithmetic as in a
// pass of its own.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_substitute(
    Eigen::Matrix<T, N, N>* a, Eigen::Matrix<T, N, 1>* delta, int il, int iu,
    Eigen::Matrix<T, N, 1>* z0 = nullptr,
    Eigen::Matrix<T, N, 1>* z1 = nullptr) {
  // CUDA uses the shared failed-column sentinel instead of a second status
  // buffer. Do not back-substitute a rejected forward sweep on either device;
  // each vector has its own test
  auto finite = [&](Eigen::Matrix<T, N, 1> const* v) {
    if (v == nullptr) return false;
    for (int row = 0; row < N; ++row)
      if (!std::isfinite(v[il](row))) return false;
    return true;
  };
  bool fd = finite(delta), f0 = finite(z0), f1 = finite(z1);
  for (int i = iu - 1; i >= il; --i) {
    Eigen::Matrix<T, N, N> const ai = a[i];
    if (fd) delta[i] -= ai * delta[i + 1];
    if (f0) z0[i] -= ai * z0[i + 1];
    if (f1) z1[i] -= ai * z1[i + 1];
  }
}

// Component B: implicit constituent transport in FLUX form (serial per
// column).
//
// The MS-VIC solve produces a per-cell total-mass correction
//   diffusion_fix_i = delta_i(0) - explicit_total_i          [density/step].
// The legacy species update applied it pointwise, pro-rata by the cell's own
// mixing ratio (du += diffusion_fix * y_i). That form (a) does not conserve
// species mass over the column whenever y varies with height, and (b) cannot
// advect a composition contrast (arriving mass always takes the receiving
// cell's own y). This pass converts the correction into the unique implied
// face mass transfer instead:
//
//   phi_i    = diffusion_fix_i * V_i                          [mass/step]
//   phi'_i   = phi_i - R * m_i / sum(m)   (R = sum(phi): the column residual
//              with no closed-face flux representation, removed in proportion
//              to cell mass so every constituent sees only the
//              flux-representable component)
//   M_{i+1/2} = M_{i-1/2} - phi'_i,  M_{-1/2} = 0  (closed bottom)
//              => M at the top face telescopes to exactly zero.
//
// Dry gas and every species then move by M * y_donor per face
// (donor-upwinded, compositionally active), with exact sequential availability
// clamping: a face never moves more than the donor holds after the explicit
// update. Conservation is exact by construction: every face's transfer is
// added to one cell and subtracted from its neighbour.
//
// Writes ONLY the mass_fix buffer (never du), so vic_redistribute_cell stays
// cell-parallel:
//   MASS(IDN, i)      : dry-gas increment [density/step]
//   MASS(IVX+dir, i)  : M through the face BELOW cell i  [mass/step]
//   MASS(IVX+(dir+1)%3, i) : the (clamped) DRY-gas transfer through that face
//                       [mass/step], consumed by the passive-scalar update
//   MASS(ICY+n, i)    : the species increment [density/step] that
//                       vic_redistribute_cell applies
//   MASS(IPR, i)      : 1 where pass 3a's dry-gas clamp emptied donor cell i
//   MASS(IVX+(dir+2)%3, i) : the TOTAL (dry + species) mass that face moved
//                       after the clamps [mass/step], gravity work; dir == 0
// Assumes the whole column lives on this rank (implicit is nb1 = 1 by
// decision; no distributed-column support planned).
template <typename T, int N>
void DISPATCH_MACRO vic_constituent_column(T* du, T* w, T* mass_fix,
                                           Eigen::Matrix<T, N, 1>* delta,
                                           T* vol, int nlayer, int dir, int ny,
                                           int stride1, int stride2) {
  // pass 1: per-cell mass increments and the column residual
  T R = 0, S = 0;
  for (int i = 0; i < nlayer; ++i) {
    T explicit_total = DU(IDN, i);
    for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);
    T phi = (delta[i](0) - explicit_total) * VOL(i);
    MASS(IDN, i) = 0;
    for (int n = 0; n < ny; ++n) MASS(ICY + n, i) = 0;
    MASS(IPR, i) = phi;
    R += phi;
    S += W(IDN, i) * VOL(i);
  }

  // pass 2: face transfers by prefix sum of the zero-sum part
  T M = 0;
  for (int i = 0; i < nlayer; ++i) {
    MASS(IVX + dir, i) = M;
    M -= MASS(IPR, i) - R * (W(IDN, i) * VOL(i) / S);
    MASS(IPR, i) = 0;
  }
  // (M here is the top-face transfer: exactly zero up to roundoff)

  // pass 3a: donor-upwinded dry-gas transfer with sequential availability
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, 0);
  T avail = (W(IDN, 0) * dryfrac + DU(IDN, 0)) * VOL(0);
  if (avail < 0) avail = 0;
  for (int i = 0; i + 1 < nlayer; ++i) {
    T dryfrac_up = 1;
    for (int n = 0; n < ny; ++n) dryfrac_up -= W(ICY + n, i + 1);
    T avail_up = (W(IDN, i + 1) * dryfrac_up + DU(IDN, i + 1)) * VOL(i + 1);
    if (avail_up < 0) avail_up = 0;

    T Mf = MASS(IVX + dir, i + 1);
    T q = Mf > 0 ? Mf * dryfrac : Mf * dryfrac_up;
    if (q > avail) {
      q = avail;
      MASS(IPR, i) = 1;
    }
    if (q < -avail_up) {
      q = -avail_up;
      MASS(IPR, i + 1) = 1;
    }

    MASS(IDN, i) -= q / VOL(i);
    MASS(IDN, i + 1) += q / VOL(i + 1);
    MASS(IVX + (dir + 1) % 3, i + 1) = q;
    MASS(IVX + (dir + 2) % 3, i + 1) = q;
    avail = avail_up + q;
    dryfrac = dryfrac_up;
  }

  // pass 3b: donor-upwinded species transfer with sequential availability
  // clamping; carry the running availability of the lower cell upward.
  // A donor keeps 4096 ulp of its species density, the margin
  // flux_positivity_theta holds back: rho * y from the primitives can exceed
  // the conserved density by an ulp, so a clamp to exactly zero left -1 ulp,
  // which the column repair then re-mixed with the cell below (#260).
  T const keep = 1 - T(4096) * std::numeric_limits<T>::epsilon();
  for (int n = 0; n < ny; ++n) {
    avail = (W(IDN, 0) * W(ICY + n, 0) * keep + DU(ICY + n, 0)) * VOL(0);
    if (avail < 0) avail = 0;
    for (int i = 0; i + 1 < nlayer; ++i) {
      T Mf = MASS(IVX + dir, i + 1);  // face between cells i and i+1
      T avail_up =
          (W(IDN, i + 1) * W(ICY + n, i + 1) * keep + DU(ICY + n, i + 1)) *
          VOL(i + 1);
      if (avail_up < 0) avail_up = 0;

      T q = Mf > 0 ? Mf * W(ICY + n, i) : Mf * W(ICY + n, i + 1);
      if (q > avail) q = avail;          // draining cell i upward
      if (q < -avail_up) q = -avail_up;  // draining cell i+1 downward

      MASS(ICY + n, i) -= q / VOL(i);
      MASS(ICY + n, i + 1) += q / VOL(i + 1);
      MASS(IVX + (dir + 2) % 3, i + 1) += q;
      avail = avail_up + q;
    }
  }
}

// Partial matrix only (N == 3): the solve has no tangential-momentum rows, so
// the face mass transfers of vic_constituent_column would otherwise move mass
// without its tangential momentum and change the tangential velocity of both
// cells. Carry it along at the donor (upwind) cell's velocity: every face's
// momentum leaves one cell and enters its neighbour, so the column sum of
// each tangential momentum is unchanged. Reads the TOTAL clamped transfer
// MASS(IVX+(dir+2)%3, i+1) of the face between cells i and i+1; must run
// after vic_constituent_column. A zero transfer leaves DU untouched (bitwise).
template <typename T>
void DISPATCH_MACRO vic_tangential_momentum_column(T* du, T* w, T* mass_fix,
                                                   T* vol, int nlayer, int dir,
                                                   int stride1, int stride2) {
  for (int i = 0; i + 1 < nlayer; ++i) {
    T q = MASS(IVX + (dir + 2) % 3, i + 1);
    int donor = q > 0 ? i : i + 1;
    for (int t = 1; t < 3; ++t) {
      int m = IVX + (dir + t) % 3;
      T p = q * W(m, donor);
      if (p == 0) continue;
      DU(m, i) -= p / VOL(i);
      DU(m, i + 1) += p / VOL(i + 1);
    }
  }
}

// Per-cell MS-VIC redistribution map. Reads delta and the precomputed
// constituent increments from mass_fix, then writes the final tendencies DU.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* mass_fix,
                                          Eigen::Matrix<T, N, 1>* delta, int i,
                                          int dir, int ny, int stride1,
                                          int stride2) {
  DU(IDN, i) += MASS(IDN, i);

  if constexpr (N == 3) {  // partial matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IPR, i) = delta[i](2);
  } else {  // full matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
    DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
    DU(IPR, i) = delta[i](4);
  }

  for (int n = 0; n < ny; ++n) {
    DU(ICY + n, i) += MASS(ICY + n, i);
  }
}

}  // namespace snap

#undef DU
#undef W
#undef MASS
#undef VOL
