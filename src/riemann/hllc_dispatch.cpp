// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include "riemann_dispatch.hpp"

// impl
#include "hllc_impl.h"

namespace snap {

void call_hllc_cpu(at::TensorIterator& iter, int dim) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_hllc_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto face_pressure =
                reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wl = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wr = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto elr = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto glr = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            auto clr = reinterpret_cast<scalar_t*>(data[6] + i * strides[6]);
            auto p0 =
                iter.ninputs() == 6
                    ? *reinterpret_cast<scalar_t*>(data[7] + i * strides[7])
                    : scalar_t(0);
            hllc_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride),
                      *clr, *(clr + stride), dim, ny, stride, stride,
                      face_pressure, p0);
          }
        },
        grain_size);
  });
}

void call_hllc_mps(at::TensorIterator& iter, int dim) {
  auto TINY_NUMBER = 1.0e-10;

  auto flx = iter.output(0);
  auto face_pressure = iter.output(1).squeeze(0);
  auto wl = iter.input(0);
  auto wr = iter.input(1);
  auto elr = iter.input(2);
  auto glr = iter.input(3);
  auto clr = iter.input(4);
  bool shifted = iter.ninputs() == 6;
  auto p0 = shifted ? iter.input(5).squeeze(0) : torch::Tensor();
  auto pl = shifted ? p0 + wl[IPR] : wl[IPR];
  auto pr = shifted ? p0 + wr[IPR] : wr[IPR];

  auto el = elr[ILT].clone();
  auto er = elr[IRT].clone();

  auto gammal = glr[ILT];
  auto gammar = glr[IRT];

  auto cl = clr[ILT];
  auto cr = clr[IRT];

  // dim, ivx, ivy, ivz
  // 3, IVX, IVY, iVZ
  // 2, IVX + 1, IVX + 2, IVX
  // 1, IVX + 2, IVX, IVX + 1
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  el += 0.5 * wl[IDN] * wl.narrow(0, IVX, 3).square().sum(0);
  er += 0.5 * wr[IDN] * wr.narrow(0, IVX, 3).square().sum(0);

  //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

  auto rhoa = .5 * (wl[IDN] + wr[IDN]);  // average density
  auto ca = .5 * (cl + cr);              // average sound speed
  auto pmid = .5 * (wl[IPR] + wr[IPR] + (wl[ivx] - wr[ivx]) * rhoa * ca);
  auto umid = .5 * (wl[ivx] + wr[ivx] + (wl[IPR] - wr[IPR]) / (rhoa * ca));

  //--- Step 3.  Compute sound speed in L,R

  auto shock_l = shifted ? (pmid - wl[IPR]) / pl : pmid / wl[IPR] - 1.0;
  auto shock_r = shifted ? (pmid - wr[IPR]) / pr : pmid / wr[IPR] - 1.0;
  auto ql = torch::sqrt(1.0 + (gammal + 1) / (2 * gammal) * shock_l);
  ql = torch::where(pmid <= wl[IPR], 1., ql);

  auto qr = torch::sqrt(1.0 + (gammar + 1) / (2 * gammar) * shock_r);
  qr = torch::where(pmid <= wr[IPR], 1., qr);

  //--- Step 4.  Compute the max/min wave speeds based on L/R

  auto al = wl[ivx] - cl * ql;
  auto ar = wr[ivx] + cr * qr;

  auto bp = torch::where(ar > 0.0, ar, TINY_NUMBER);
  auto bm = torch::where(al < 0.0, al, -TINY_NUMBER);

  //--- Step 5. Compute the contact wave speed and pressure

  auto vxl = wl[ivx] - al;
  auto vxr = wr[ivx] - ar;

  auto tl = wl[IPR] + vxl * wl[IDN] * wl[ivx];
  auto tr = wr[IPR] + vxr * wr[IDN] * wr[ivx];

  auto ml = wl[IDN] * vxl;
  auto mr = -(wr[IDN] * vxr);

  // Determine the contact wave speed...
  auto am = (tl - tr) / (ml + mr);
  // ...and the pressure at the contact surface
  auto cp = (ml * tr + mr * tl) / (ml + mr);
  cp = shifted ? torch::maximum(cp, -p0) : torch::where(cp > 0.0, cp, 0.0);
  face_pressure.copy_(cp);

  //--- Step 6. Compute L/R fluxes along the line bm, bp
  auto fl = torch::zeros_like(wl);
  auto fr = torch::zeros_like(wr);

  vxl = wl[ivx] - bm;
  vxr = wr[ivx] - bp;

  fl[IDN] = wl[IDN] * vxl;
  fr[IDN] = wr[IDN] * vxr;

  fl.narrow(0, IVX, 3) = wl[IDN] * wl.narrow(0, IVX, 3) * vxl;
  fr.narrow(0, IVX, 3) = wr[IDN] * wr.narrow(0, IVX, 3) * vxr;

  fl[ivx] += wl[IPR];
  fr[ivx] += wr[IPR];

  fl[IPR] = el * vxl + pl * wl[ivx];
  fr[IPR] = er * vxr + pr * wr[ivx];
  if (shifted) {
    int ny = wl.size(0) - ICY;
    fl.narrow(0, ICY, ny).copy_(wl[IDN] * wl.narrow(0, ICY, ny) * vxl);
    fr.narrow(0, ICY, ny).copy_(wr[IDN] * wr.narrow(0, ICY, ny) * vxr);
    fl[IDN].mul_(1. - wl.narrow(0, ICY, ny).sum(0));
    fr[IDN].mul_(1. - wr.narrow(0, ICY, ny).sum(0));
  }

  //--- Step 8. Compute flux weights or scales

  auto ami = am >= 0.0;
  auto sl = torch::where(ami, am / (am - bm), 0.0);
  auto sr = torch::where(ami, 0.0, -am / (bp - am));
  auto sm = torch::where(ami, -bm / (am - bm), bp / (bp - am));

  //--- Step 9. Compute the HLLC flux at interface, including weighted
  // contribution
  // of the flux along the contact

  torch::add_out(flx, sl.unsqueeze(0) * fl, sr.unsqueeze(0) * fr);
  flx[ivx] += sm * cp;
  flx[IPR] += sm * (shifted ? p0 + cp : cp) * am;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_hllc);
REGISTER_ALL_CPU_DISPATCH(call_hllc, &snap::call_hllc_cpu);
REGISTER_MPS_DISPATCH(call_hllc, &snap::call_hllc_mps);

}  // namespace at::native
