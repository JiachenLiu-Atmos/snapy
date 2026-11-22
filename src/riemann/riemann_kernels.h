#pragma once

// C/C++
#include <cmath>
#include <type_traits>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "../eos/ideal_moist_impl.h"
#include "../eos/shallow_water_impl.h"
#include "riemann.hpp"

// The Riemann face functions (V3 C.rst 3.6): hllc, lmars, roe and
// shallow-roe, one definition each for CPU and CUDA (MPS runs the CPU loop
// through OnHost), and riemann_face<Kind>, the loop body of both loops.
// upwind and plume-roe are torch formulas (riemann.cpp).

#define SQR(x) ((x) * (x))
#define FLX(n) flx[(n) * stride_f]
#define WL(n) wl[(n) * stride_w]
#define WR(n) wr[(n) * stride_w]

namespace snap {

//! the small speed and sound-speed floor of the hllc and roe solvers
template <typename T>
DISPATCH_MACRO constexpr T riemann_tiny() {
  return T(1.0e-10);
}

template <typename T>
void DISPATCH_MACRO hllc_impl(T* flx, T* wl, T* wr, T el, T er, T gammal,
                              T gammar, T cl, T cr, int dim, int ny,
                              int stride_w, int stride_f,
                              T* face_pressure = nullptr,
                              T reference_pressure = T(0)) {
  // With a shared face background, pressure rows and momentum pressure flux
  // are perturbations. Wave speeds and energy transport still use totals.
  // Typed constants keep float instantiations on FP32 arithmetic throughout.
  bool shifted = reference_pressure != T(0);
  auto pl = shifted ? reference_pressure + WL(IPR) : WL(IPR);
  auto pr = shifted ? reference_pressure + WR(IPR) : WR(IPR);

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  //--- Step 1.  Compute kappa

  //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

  el += T(0.5) * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  er += T(0.5) * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  auto rhoa = T(0.5) * (WL(IDN) + WR(IDN));  // average density
  auto ca = T(0.5) * (cl + cr);              // average sound speed
  auto pmid = T(0.5) * (WL(IPR) + WR(IPR) + (WL(ivx) - WR(ivx)) * rhoa * ca);

  //--- Step 3.  Compute sound speed in L,R

  auto shock_l = shifted ? (pmid - WL(IPR)) / pl : pmid / WL(IPR) - T(1.0);
  auto shock_r = shifted ? (pmid - WR(IPR)) / pr : pmid / WR(IPR) - T(1.0);
  auto ql = (pmid <= WL(IPR))
                ? T(1.0)
                : std::sqrt(T(1.0) + (gammal + 1) / (2 * gammal) * shock_l);
  auto qr = (pmid <= WR(IPR))
                ? T(1.0)
                : std::sqrt(T(1.0) + (gammar + 1) / (2 * gammar) * shock_r);

  static_assert(std::is_same_v<decltype(rhoa), T> &&
                    std::is_same_v<decltype(pmid), T> &&
                    std::is_same_v<decltype(ql), T>,
                "HLLC work must retain the dispatched scalar precision");

  //--- Step 4.  Compute the max/min wave speeds based on L/R

  auto al = WL(ivx) - cl * ql;
  auto ar = WR(ivx) + cr * qr;

  auto bp = ar > T(0.0) ? ar : riemann_tiny<T>();
  auto bm = al < T(0.0) ? al : -riemann_tiny<T>();

  //--- Step 5. Compute the contact wave speed and pressure

  T vxl, vxr;
  if constexpr (std::is_same_v<T, float>) {
    // Do not recover a small acoustic speed by subtracting large velocities.
    vxl = cl * ql;
    vxr = -cr * qr;
  } else {
    // Preserve the established FP64 regression arithmetic.
    vxl = WL(ivx) - al;
    vxr = WR(ivx) - ar;
  }

  auto tl = WL(IPR) + vxl * WL(IDN) * WL(ivx);
  auto tr = WR(IPR) + vxr * WR(IDN) * WR(ivx);

  auto ml = WL(IDN) * vxl;
  auto mr = -(WR(IDN) * vxr);

  // Determine the contact wave speed...
  auto am = (tl - tr) / (ml + mr);
  // ...and the pressure at the contact surface
  auto cp = (ml * tr + mr * tl) / (ml + mr);
  cp = shifted ? (cp > -reference_pressure ? cp : -reference_pressure)
               : (cp > T(0.0) ? cp : T(0.0));
  if (face_pressure != nullptr) *face_pressure = cp;

  //--- Step 6. Compute L/R fluxes along the line bm, bp

  if constexpr (std::is_same_v<T, float>) {
    // bm/bp are clipped at zero: only their acoustic branches simplify.
    vxl = al < T(0) ? cl * ql : WL(ivx) + riemann_tiny<T>();
    vxr = ar > T(0) ? -cr * qr : WR(ivx) - riemann_tiny<T>();
  } else {
    vxl = WL(ivx) - bm;
    vxr = WR(ivx) - bp;
  }

  T rdl = T(1.), rdr = T(1.);
  for (int n = 0; n < ny; ++n) {
    rdl -= WL(ICY + n);
    rdr -= WR(ICY + n);
  }

  T fl[ICY], fr[ICY];

  fl[IDN] = WL(IDN) * vxl * rdl;
  fr[IDN] = WR(IDN) * vxr * rdr;

  fl[ivx] = WL(IDN) * WL(ivx) * vxl + WL(IPR);
  fr[ivx] = WR(IDN) * WR(ivx) * vxr + WR(IPR);

  fl[ivy] = WL(IDN) * WL(ivy) * vxl;
  fr[ivy] = WR(IDN) * WR(ivy) * vxr;

  fl[ivz] = WL(IDN) * WL(ivz) * vxl;
  fr[ivz] = WR(IDN) * WR(ivz) * vxr;

  fl[IPR] = el * vxl + pl * WL(ivx);
  fr[IPR] = er * vxr + pr * WR(ivx);

  //--- Step 8. Compute flux weights or scales

  T sl, sr, sm;
  if (am >= T(0.0)) {
    sl = am / (am - bm);
    sr = T(0.0);
    sm = -bm / (am - bm);
  } else {
    sl = T(0.0);
    sr = -am / (bp - am);
    sm = bp / (bp - am);
  }

  //--- Step 9. Compute the HLLC flux at interface, including weighted
  // contribution
  // of the flux along the contact

  FLX(IDN) = sl * fl[IDN] + sr * fr[IDN];
  FLX(ivx) = sl * fl[ivx] + sr * fr[ivx] + sm * cp;
  FLX(ivy) = sl * fl[ivy] + sr * fr[ivy];
  FLX(ivz) = sl * fl[ivz] + sr * fr[ivz];
  auto cp_total = shifted ? reference_pressure + cp : cp;
  FLX(IPR) = sl * fl[IPR] + sr * fr[IPR] + sm * cp_total * am;

  for (int n = 0; n < ny; ++n) {
    auto fln = WL(IDN) * WL(ICY + n) * vxl;
    auto frn = WR(IDN) * WR(ICY + n) * vxr;
    FLX(ICY + n) = sl * fln + sr * frn;
  }
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef SQR

#define WL(n) (wl[(n) * stride_w])
#define WR(n) (wr[(n) * stride_w])
#define FLX(n) (flx[(n) * stride_f])
#define SQR(x) ((x) * (x))

namespace snap {

template <typename T>
void DISPATCH_MACRO lmars_impl(T* flx, T* wl, T* wr, T hl, T hr, T gammal,
                               T gammar, int dim, int ny, int stride_w,
                               int stride_f, T* face_pressure = nullptr) {
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  T wli[5] = {WL(IDN), WL(ivx), WL(ivy), WL(ivz), WL(IPR)};
  T wri[5] = {WR(IDN), WR(ivx), WR(ivy), WR(ivz), WR(IPR)};

  // Enthalpies. Typed constants (V3 C9) keep a float instantiation on FP32
  // arithmetic throughout; FP64 is unchanged.
  hl += T(0.5) * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ])) +
        wli[IPR] / wli[IDN];
  hr += T(0.5) * (SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ])) +
        wri[IPR] / wri[IDN];

  // Average density, average sound speed, pressure, velocity
  T rhobar = T(0.5) * (wli[IDN] + wri[IDN]);
  T gamma_bar = T(0.5) * (gammal + gammar);
  T cbar = std::sqrt(T(0.5) * gamma_bar * (wli[IPR] + wri[IPR]) / rhobar);

  T pbar = T(0.5) * (wli[IPR] + wri[IPR]) +
           T(0.5) * (rhobar * cbar) * (wli[IVX] - wri[IVX]);
  if (face_pressure != nullptr) *face_pressure = pbar;

  T ubar = T(0.5) * (wli[IVX] + wri[IVX]) +
           T(0.5) / (rhobar * cbar) * (wli[IPR] - wri[IPR]);

  // Compute fluxes depending on the sign of ubar
  T rd = T(1.0);
  if (ubar > T(0)) {
    // Left side flux
    for (int n = 0; n < ny; n++) {
      rd -= WL(ICY + n);
    }

    FLX(IDN) = ubar * wli[IDN] * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * wli[IDN] * WL(ICY + n);
    }

    FLX(ivx) = ubar * wli[IDN] * wli[IVX] + pbar;
    FLX(ivy) = ubar * wli[IDN] * wli[IVY];
    FLX(ivz) = ubar * wli[IDN] * wli[IVZ];
    FLX(IPR) = ubar * wli[IDN] * hl;
  } else {
    // Right side flux
    for (int n = 0; n < ny; n++) {
      rd -= WR(ICY + n);
    }

    FLX(IDN) = ubar * wri[IDN] * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * wri[IDN] * WR(ICY + n);
    }

    FLX(ivx) = ubar * wri[IDN] * wri[IVX] + pbar;
    FLX(ivy) = ubar * wri[IDN] * wri[IVY];
    FLX(ivz) = ubar * wri[IDN] * wri[IVZ];
    FLX(IPR) = ubar * wri[IDN] * hr;
  }
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef SQR

#define SQR(x) ((x) * (x))
#define WL(n) (wl[(n) * stride_w])
#define WR(n) (wr[(n) * stride_w])
#define FLX(n) (flx[(n) * stride_f])

namespace snap {

template <typename T>
inline DISPATCH_MACRO T roe_max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
void DISPATCH_MACRO roe_impl(T* flx, T const* wl, T const* wr, T el, T er,
                             T gammal, T gammar, T cl, T cr, int dim, int ny,
                             bool ideal_moist, int nvapor, T gammad,
                             T const* inv_mu_ratio_m1, T const* cv_ratio_m1,
                             T const* u0, int stride_w, int stride_f,
                             T* face_pressure = nullptr, T* llf = nullptr) {
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  T etl = el + T(0.5) * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  T etr = er + T(0.5) * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  T sqrtdl = sqrt(WL(IDN));
  T sqrtdr = sqrt(WR(IDN));
  T isdlpdr = T(1) / (sqrtdl + sqrtdr);

  T rhobar = sqrtdl * sqrtdr;
  T v1 = (sqrtdl * WL(ivx) + sqrtdr * WR(ivx)) * isdlpdr;
  T v2 = (sqrtdl * WL(ivy) + sqrtdr * WR(ivy)) * isdlpdr;
  T v3 = (sqrtdl * WL(ivz) + sqrtdr * WR(ivz)) * isdlpdr;
  T hl = (etl + WL(IPR)) / WL(IDN);
  T hr = (etr + WR(IPR)) / WR(IDN);
  T h = (hl * sqrtdl + hr * sqrtdr) * isdlpdr;

  T dryl = T(1);
  T dryr = T(1);
  for (int n = 0; n < ny; ++n) {
    dryl -= WL(ICY + n);
    dryr -= WR(ICY + n);
  }

  T rhol0 = WL(IDN) * dryl;
  T rhor0 = WR(IDN) * dryr;

  T fl0 = rhol0 * WL(ivx);
  T fr0 = rhor0 * WR(ivx);

  T fl1 = WL(IDN) * WL(ivx) * WL(ivx) + WL(IPR);
  T fr1 = WR(IDN) * WR(ivx) * WR(ivx) + WR(IPR);

  T fl2 = WL(IDN) * WL(ivx) * WL(ivy);
  T fr2 = WR(IDN) * WR(ivx) * WR(ivy);

  T fl3 = WL(IDN) * WL(ivx) * WL(ivz);
  T fr3 = WR(IDN) * WR(ivx) * WR(ivz);

  T fl4 = (etl + WL(IPR)) * WL(ivx);
  T fr4 = (etr + WR(IPR)) * WR(ivx);

  T qbar[64];
  qbar[0] = (rhol0 / sqrtdl + rhor0 / sqrtdr) * isdlpdr;
  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    qbar[1 + n] = (rholn / sqrtdl + rhorn / sqrtdr) * isdlpdr;
  }

  T gamma_roe = T(0.5) * (gammal + gammar);
  T offset = T(0);
  if (ideal_moist) {
    T feps = ideal_moist_feps(qbar + 1, ny, nvapor, inv_mu_ratio_m1);
    T fsig = ideal_moist_fsig(qbar + 1, ny, cv_ratio_m1);
    gamma_roe = T(1) + (gammad - T(1)) * feps / fsig;

    offset = qbar[0] * u0[0];
    for (int n = 0; n < ny; ++n) {
      offset += qbar[1 + n] * u0[1 + n];
    }
  }

  T du1 = WR(IDN) * WR(ivx) - WL(IDN) * WL(ivx);
  T du2 = WR(IDN) * WR(ivy) - WL(IDN) * WL(ivy);
  T du3 = WR(IDN) * WR(ivz) - WL(IDN) * WL(ivz);
  T du4 = etr - etl;

  FLX(IDN) = T(0.5) * (fl0 + fr0);
  FLX(ivx) = T(0.5) * (fl1 + fr1);
  FLX(ivy) = T(0.5) * (fl2 + fr2);
  FLX(ivz) = T(0.5) * (fl3 + fr3);
  FLX(IPR) = T(0.5) * (fl4 + fr4);
  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    FLX(ICY + n) = T(0.5) * (rholn * WL(ivx) + rhorn * WR(ivx));
  }

  T vsq = v1 * v1 + v2 * v2 + v3 * v3;
  T gm1_roe = gamma_roe - T(1);
  T q = h - T(0.5) * vsq - offset;
  T cs_sq = roe_max(gm1_roe * roe_max(q, riemann_tiny<T>()), riemann_tiny<T>());
  T cs = sqrt(cs_sq);
  if (face_pressure != nullptr) {
    *face_pressure =
        T(0.5) * (WL(IPR) + WR(IPR) + rhobar * cs * (WL(ivx) - WR(ivx)));
  }

  T lam_m = v1 - cs;
  T lam_0 = v1;
  T lam_p = v1 + cs;
  T spd_m = abs(lam_m);
  T spd_0 = abs(lam_0);
  T spd_p = abs(lam_p);

  bool llf_flag = false;
  T du = WR(ivx) - WL(ivx);
  T dv = WR(ivy) - WL(ivy);
  T dw = WR(ivz) - WL(ivz);
  T dp = WR(IPR) - WL(IPR);

  T alpha_m = -T(0.5) * rhobar / cs * du + T(0.5) * dp / cs_sq;
  T alpha_p = +T(0.5) * rhobar / cs * du + T(0.5) * dp / cs_sq;
  T alpha_v = rhobar * dv;
  T alpha_w = rhobar * dw;

  T alpha0 = T(0);
  T alpha_comp0 = rhor0 - rhol0 - dp / cs_sq * qbar[0];
  T rho_first = rhol0 + alpha_m * qbar[0];
  if (rho_first < T(0)) llf_flag = true;
  T rho_second = rho_first + alpha_comp0;
  if (rho_second < T(0)) llf_flag = true;
  alpha0 += alpha_comp0;
  FLX(IDN) -= T(0.5) * (spd_m * alpha_m * qbar[0] + spd_0 * alpha_comp0 +
                        spd_p * alpha_p * qbar[0]);

  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    auto alpha_n = rhorn - rholn - dp / cs_sq * qbar[1 + n];
    auto rho1 = rholn + alpha_m * qbar[1 + n];
    if (rho1 < T(0)) llf_flag = true;
    auto rho2 = rho1 + alpha_n;
    if (rho2 < T(0)) llf_flag = true;
    alpha0 += alpha_n;
    FLX(ICY + n) -= T(0.5) * (spd_m * alpha_m * qbar[1 + n] + spd_0 * alpha_n +
                              spd_p * alpha_p * qbar[1 + n]);
  }

  FLX(ivx) -= T(0.5) * (spd_m * alpha_m * (v1 - cs) + spd_0 * (v1 * alpha0) +
                        spd_p * alpha_p * (v1 + cs));
  FLX(ivy) -= T(0.5) * (spd_m * alpha_m * v2 + spd_0 * (v2 * alpha0 + alpha_v) +
                        spd_p * alpha_p * v2);
  FLX(ivz) -= T(0.5) * (spd_m * alpha_m * v3 + spd_0 * (v3 * alpha0 + alpha_w) +
                        spd_p * alpha_p * v3);
  FLX(IPR) -=
      T(0.5) * (spd_m * alpha_m * (h - v1 * cs) +
                spd_0 * (rhobar * (hr - hl - v1 * du) + alpha0 * h - dp) +
                spd_p * alpha_p * (h + v1 * cs));

  if (lam_m > T(0)) {
    FLX(IDN) = fl0;
    FLX(ivx) = fl1;
    FLX(ivy) = fl2;
    FLX(ivz) = fl3;
    FLX(IPR) = fl4;
    for (int n = 0; n < ny; ++n) {
      FLX(ICY + n) = WL(IDN) * WL(ICY + n) * WL(ivx);
    }
  }
  if (lam_p < T(0)) {
    FLX(IDN) = fr0;
    FLX(ivx) = fr1;
    FLX(ivy) = fr2;
    FLX(ivz) = fr3;
    FLX(IPR) = fr4;
    for (int n = 0; n < ny; ++n) {
      FLX(ICY + n) = WR(IDN) * WR(ICY + n) * WR(ivx);
    }
  }

  if (llf_flag) {
    T a = T(0.5) * roe_max(abs(WL(ivx)) + cl, abs(WR(ivx)) + cr);
    FLX(IDN) = T(0.5) * (fl0 + fr0) - a * (rhor0 - rhol0);
    FLX(ivx) = T(0.5) * (fl1 + fr1) - a * du1;
    FLX(ivy) = T(0.5) * (fl2 + fr2) - a * du2;
    FLX(ivz) = T(0.5) * (fl3 + fr3) - a * du3;
    FLX(IPR) = T(0.5) * (fl4 + fr4) - a * du4;
    for (int n = 0; n < ny; ++n) {
      auto rholn = WL(IDN) * WL(ICY + n);
      auto rhorn = WR(IDN) * WR(ICY + n);
      FLX(ICY + n) =
          T(0.5) * (rholn * WL(ivx) + rhorn * WR(ivx)) - a * (rhorn - rholn);
    }
  }
  if (llf != nullptr) *llf = llf_flag ? T(1) : T(0);
}

}  // namespace snap

#undef FLX
#undef WR
#undef WL
#undef SQR

namespace snap {

#define SWL(n) (wl[(n) * stride_w])
#define SWR(n) (wr[(n) * stride_w])
#define SFLX(n) (flx[(n) * stride_f])

template <typename T>
void DISPATCH_MACRO shallow_roe_impl(T* flx, T const* wl, T const* wr, int dim,
                                     int dir_yz, int stride_w, int stride_f) {
  int ivx, ivy, ivz;
  if (dir_yz) {
    ivx = dim == 2 ? IVY : IVZ;
    ivy = dim == 2 ? IVZ : IVY;
    ivz = IVX;
  } else {
    ivx = dim == 3 ? IVX : IVY;
    ivy = dim == 3 ? IVY : IVX;
    ivz = IVZ;
  }

  T sqrtdl = sqrt(SWL(IDN));
  T sqrtdr = sqrt(SWR(IDN));
  T isdlpdr = T(1) / (sqrtdl + sqrtdr);

  T ubar = (SWL(ivx) * sqrtdl + SWR(ivx) * sqrtdr) * isdlpdr;
  T vbar = (SWL(ivy) * sqrtdl + SWR(ivy) * sqrtdr) * isdlpdr;
  T cbar = shallow_water_roe_sound_speed(SWL(IDN), SWR(IDN));

  T del0 = SWR(IDN) - SWL(IDN);
  T delx = SWR(ivx) - SWL(ivx);
  T dely = SWR(ivy) - SWL(ivy);
  T hbar = sqrt(SWL(IDN) * SWR(IDN));

  T a1 = T(0.5) * (cbar * del0 - hbar * delx) / cbar;
  T a2 = hbar * dely;
  T a3 = T(0.5) * (cbar * del0 + hbar * delx) / cbar;

  T wave0[4] = {0, 0, 0, 0};
  T wave1[4] = {0, 0, 0, 0};
  T wave2[4] = {0, 0, 0, 0};
  wave0[IDN] = a1;
  wave0[ivx] = a1 * (ubar - cbar);
  wave0[ivy] = a1 * vbar;
  wave1[ivy] = a2;
  wave2[IDN] = a3;
  wave2[ivx] = a3 * (ubar + cbar);
  wave2[ivy] = a3 * vbar;

  T speed0 = abs(ubar - cbar);
  T speed1 = abs(ubar);
  T speed2 = abs(ubar + cbar);

  SFLX(IDN) = T(0.5) * (SWL(IDN) * SWL(ivx) + SWR(IDN) * SWR(ivx));
  SFLX(ivx) =
      T(0.5) * (SWL(IDN) * SWL(ivx) * SWL(ivx) + T(0.5) * SWL(IDN) * SWL(IDN) +
                SWR(IDN) * SWR(ivx) * SWR(ivx) + T(0.5) * SWR(IDN) * SWR(IDN));
  SFLX(ivy) = T(0.5) *
              (SWL(IDN) * SWL(ivx) * SWL(ivy) + SWR(IDN) * SWR(ivx) * SWR(ivy));
  SFLX(ivz) = 0;

  for (int v = 0; v < 4; ++v) {
    SFLX(v) -=
        T(0.5) * (speed0 * wave0[v] + speed1 * wave1[v] + speed2 * wave2[v]);
  }
}

#undef SWL
#undef SWR
#undef SFLX

}  // namespace snap

namespace snap {

//! what a face function reads besides its operands (riemann_args)
template <typename T>
struct RiemannArgs {
  int dim, ny, stride;
  bool background;  // hllc: the operand after c is the face background p0
  bool ideal_moist;
  int nvapor;
  T gammad;
  T const* inv_mu_ratio_m1;
  T const* cv_ratio_m1;
  T const* u0;
  int dir_yz;
};

//! riemann.central's mass rows at one face (lane-c-notes.md)
template <RiemannKind K, typename T>
DISPATCH_MACRO void riemann_central(T* cen, T const* wl, T const* wr,
                                    RiemannArgs<T> const& a) {
  auto s = a.stride;
  if constexpr (K == RiemannKind::shallow_roe) {
    int ivx = a.dir_yz ? (a.dim == 2 ? IVY : IVZ) : (a.dim == 3 ? IVX : IVY);
    cen[IDN * s] =
        T(0.5) * (wl[IDN * s] * wl[ivx * s] + wr[IDN * s] * wr[ivx * s]);
  } else {
    int ivx = IPR - a.dim;
    auto ml = wl[IDN * s] * wl[ivx * s];
    auto mr = wr[IDN * s] * wr[ivx * s];
    T rdl = T(1), rdr = T(1);
    for (int n = 0; n < a.ny; ++n) {
      rdl -= wl[(ICY + n) * s];
      rdr -= wr[(ICY + n) * s];
      cen[(ICY + n) * s] =
          T(0.5) * (ml * wl[(ICY + n) * s] + mr * wr[(ICY + n) * s]);
    }
    cen[IDN * s] = T(0.5) * (ml * rdl + mr * rdr);
  }
}

//! one face of kind K; p holds the operands of call_riemann
//! (lane-c-notes.md)
template <RiemannKind K, bool C, bool E, typename T>
DISPATCH_MACRO void riemann_face(T* const* p, RiemannArgs<T> const& a) {
  auto s = a.stride;
  constexpr int o = K == RiemannKind::shallow_roe ? 1 : 2;
  constexpr int i = o + (C ? 1 : 0) + (E ? 1 : 0);
  if constexpr (K == RiemannKind::hllc) {
    auto p0 = a.background ? *p[i + 5] : T(0);
    hllc_impl(p[0], p[i], p[i + 1], *p[i + 2], *(p[i + 2] + s), *p[i + 3],
              *(p[i + 3] + s), *p[i + 4], *(p[i + 4] + s), a.dim, a.ny, s, s,
              p[1], p0);
  } else if constexpr (K == RiemannKind::lmars) {
    lmars_impl(p[0], p[i], p[i + 1], *p[i + 2], *(p[i + 2] + s), *p[i + 3],
               *(p[i + 3] + s), a.dim, a.ny, s, s, p[1]);
  } else if constexpr (K == RiemannKind::roe) {
    roe_impl(p[0], p[i], p[i + 1], *p[i + 2], *(p[i + 2] + s), *p[i + 3],
             *(p[i + 3] + s), *p[i + 4], *(p[i + 4] + s), a.dim, a.ny,
             a.ideal_moist, a.nvapor, a.gammad, a.inv_mu_ratio_m1,
             a.cv_ratio_m1, a.u0, s, s, p[1],
             E ? p[o + (C ? 1 : 0)] : static_cast<T*>(nullptr));
  } else if constexpr (K == RiemannKind::shallow_roe) {
    shallow_roe_impl(p[0], p[i], p[i + 1], a.dim, a.dir_yz, s, s);
  }
  if constexpr (C) riemann_central<K>(p[o], p[i], p[i + 1], a);
}

}  // namespace snap
