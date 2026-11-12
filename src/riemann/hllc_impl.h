#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define SQR(x) ((x) * (x))
#define FLX(n) flx[(n) * stride_f]
#define WL(n) wl[(n) * stride_w]
#define WR(n) wr[(n) * stride_w]

namespace snap {

template <typename T>
void DISPATCH_MACRO hllc_impl(T *flx, T *wl, T *wr, T el, T er, T gammal,
                              T gammar, T cl, T cr, int dim, int ny,
                              int stride_w, int stride_f,
                              T *face_pressure = nullptr,
                              T reference_pressure = T(0)) {
  // With a shared face background, pressure rows and momentum pressure flux
  // are perturbations. Wave speeds and energy transport still use totals.
  // Keep the original arithmetic when the background is absent.
  bool shifted = reference_pressure != T(0);
  auto pl = shifted ? reference_pressure + WL(IPR) : WL(IPR);
  auto pr = shifted ? reference_pressure + WR(IPR) : WR(IPR);
  auto TINY_NUMBER = 1.0e-10;

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  //--- Step 1.  Compute kappa

  //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

  el += 0.5 * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  er += 0.5 * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  auto rhoa = .5 * (WL(IDN) + WR(IDN));  // average density
  auto ca = .5 * (cl + cr);              // average sound speed
  auto pmid = .5 * (WL(IPR) + WR(IPR) + (WL(ivx) - WR(ivx)) * rhoa * ca);
  auto umid = .5 * (WL(ivx) + WR(ivx) + (WL(IPR) - WR(IPR)) / (rhoa * ca));

  //--- Step 3.  Compute sound speed in L,R

  auto shock_l = shifted ? (pmid - WL(IPR)) / pl : pmid / WL(IPR) - 1.0;
  auto shock_r = shifted ? (pmid - WR(IPR)) / pr : pmid / WR(IPR) - 1.0;
  auto ql = (pmid <= WL(IPR))
                ? 1.0
                : std::sqrt(1.0 + (gammal + 1) / (2 * gammal) * shock_l);
  auto qr = (pmid <= WR(IPR))
                ? 1.0
                : std::sqrt(1.0 + (gammar + 1) / (2 * gammar) * shock_r);

  //--- Step 4.  Compute the max/min wave speeds based on L/R

  auto al = WL(ivx) - cl * ql;
  auto ar = WR(ivx) + cr * qr;

  auto bp = ar > 0.0 ? ar : (TINY_NUMBER);
  auto bm = al < 0.0 ? al : -(TINY_NUMBER);

  //--- Step 5. Compute the contact wave speed and pressure

  auto vxl = WL(ivx) - al;
  auto vxr = WR(ivx) - ar;

  auto tl = WL(IPR) + vxl * WL(IDN) * WL(ivx);
  auto tr = WR(IPR) + vxr * WR(IDN) * WR(ivx);

  auto ml = WL(IDN) * vxl;
  auto mr = -(WR(IDN) * vxr);

  // Determine the contact wave speed...
  auto am = (tl - tr) / (ml + mr);
  // ...and the pressure at the contact surface
  auto cp = (ml * tr + mr * tl) / (ml + mr);
  cp = shifted ? (cp > -reference_pressure ? cp : -reference_pressure)
               : (cp > 0.0 ? cp : 0.0);
  if (face_pressure != nullptr) *face_pressure = cp;

  //--- Step 6. Compute L/R fluxes along the line bm, bp

  vxl = WL(ivx) - bm;
  vxr = WR(ivx) - bp;

  T rdl = 1., rdr = 1.;
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
  if (am >= 0.0) {
    sl = am / (am - bm);
    sr = 0.0;
    sm = -bm / (am - bm);
  } else {
    sl = 0.0;
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
