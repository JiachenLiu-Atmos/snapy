// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// C/C++
#include <cmath>

// snap
#include <snap/utils/on_host.hpp>

#include "hydro_dispatch.hpp"
#include "hydro_ref_x1_impl.h"

namespace snap {

void hydro_pressure_scan_x1_cpu(torch::Tensor const& rho,
                                torch::Tensor const& dx1f,
                                torch::Tensor const& anchor,
                                torch::Tensor const& lo,
                                torch::Tensor const& hi, int iu, double grav) {
  int nc1 = rho.size(-1), columns = rho.numel() / nc1;
  AT_DISPATCH_FLOATING_TYPES(
      rho.scalar_type(), "hydro_pressure_scan_x1_cpu", [&] {
        at::parallel_for(0, columns, 0, [&](int64_t begin, int64_t end) {
          for (int64_t c = begin; c < end; ++c)
            hydro_ref_x1_pressure_scan_impl<scalar_t, false>(
                rho.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(),
                anchor.data_ptr<scalar_t>()[c], lo.data_ptr<scalar_t>(),
                hi.data_ptr<scalar_t>(), c, nc1, iu, scalar_t(grav));
        });
      });
}

void hydro_ref_x1_cpu(torch::Tensor const& w, torch::Tensor const& dx1f,
                      torch::Tensor const& anchor, torch::Tensor const& psf_lo,
                      torch::Tensor const& psf_hi, torch::Tensor const& pref,
                      torch::Tensor const& dsf, torch::Tensor const& dref,
                      int iu, double grav, int x1_grid, bool phys_in,
                      bool phys_out, bool wall_clamp,
                      torch::Tensor const& rows) {
  TORCH_CHECK(x1_grid != kX1RefStretched || rows.defined(),
              "hydro_ref_x1: a stretched x1 needs its six-face rows");
  int ncolumns = w.size(1) * w.size(2);
  int nc1 = w.size(3);
  // A trailing pair dimension selects finite-increment reference arithmetic.
  if (w.dim() == 5) {
    TORCH_CHECK(w.scalar_type() == torch::kFloat64 && w.size(4) == 2,
                "split reference requires packed FP64 pairs");
    auto ptr = [](torch::Tensor const& t) -> SRS* {
      return t.defined() ? reinterpret_cast<SRS*>(t.data_ptr<double>())
                         : nullptr;
    };
    at::parallel_for(0, ncolumns, 0, [&](int64_t begin, int64_t end) {
      for (int column = begin; column < end; ++column)
        hydro_ref_x1_impl(ptr(w), ptr(dx1f), ptr(anchor), ptr(psf_lo),
                          ptr(psf_hi), ptr(pref), ptr(dsf), ptr(dref), column,
                          ncolumns, nc1, iu, SRS(grav), x1_grid, phys_in,
                          phys_out, wall_clamp, ptr(rows));
    });
    return;
  }
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "hydro_ref_x1_cpu", [&] {
    auto anchor_ptr = anchor.defined() ? anchor.data_ptr<scalar_t>() : nullptr;
    auto rows_ptr = rows.defined() ? rows.data_ptr<scalar_t>() : nullptr;
    at::parallel_for(0, ncolumns, 0, [&](int64_t begin, int64_t end) {
      for (int64_t column = begin; column < end; ++column) {
        hydro_ref_x1_impl(w.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(),
                          anchor_ptr, psf_lo.data_ptr<scalar_t>(),
                          psf_hi.data_ptr<scalar_t>(),
                          pref.data_ptr<scalar_t>(), dsf.data_ptr<scalar_t>(),
                          dref.data_ptr<scalar_t>(), static_cast<int>(column),
                          ncolumns, nc1, iu, scalar_t(grav), x1_grid, phys_in,
                          phys_out, wall_clamp, rows_ptr);
      }
    });
  });
}

torch::Tensor hydro_ref_x1_stretched_rows(torch::Tensor const& dx1f, int iu,
                                          bool phys_in, bool phys_out,
                                          bool wall_clamp) {
  auto h = dx1f.cpu().to(torch::kFloat64).contiguous();
  int nc1 = h.size(0);
  auto hp = h.data_ptr<double>();
  auto rows = torch::zeros({nc1, 7}, torch::kFloat64);
  auto r = rows.accessor<double, 2>();
  long double const tq[3] = {0.5L - 0.1L * std::sqrt(15.0L), 0.5L,
                             0.5L + 0.1L * std::sqrt(15.0L)};
  long double const wq[3] = {5.0L / 18.0L, 8.0L / 18.0L, 5.0L / 18.0L};
  for (int i = 0; i < nc1; ++i) {
    int start =
        hydro_ref_x1_six_face_start(i, nc1, iu, phys_in, phys_out, wall_clamp);
    r[i][0] = start;
    if (start < 0) continue;
    long double t[6];
    for (int k = 0; k < 6; ++k) {
      long double o = 0.0L;
      for (int j = i; j < start + k; ++j) o += hp[j];
      for (int j = start + k; j < i; ++j) o -= hp[j];
      t[k] = o / hp[i];
    }
    for (int m = 0; m < 6; ++m) {
      long double wm = 0.0L;
      for (int q = 0; q < 3; ++q) {
        long double l = 1.0L;
        for (int k = 0; k < 6; ++k)
          if (k != m) l *= (tq[q] - t[k]) / (t[m] - t[k]);
        wm += wq[q] * l;
      }
      r[i][1 + m] = static_cast<double>(wm);
    }
  }
  return rows;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_hydro_pressure_scan_x1);
REGISTER_ALL_CPU_DISPATCH(call_hydro_pressure_scan_x1,
                          &snap::hydro_pressure_scan_x1_cpu);
REGISTER_MPS_DISPATCH(call_hydro_pressure_scan_x1,
                      &snap::OnHost<&snap::hydro_pressure_scan_x1_cpu>::run);
DEFINE_DISPATCH(call_hydro_ref_x1);
REGISTER_ALL_CPU_DISPATCH(call_hydro_ref_x1, &snap::hydro_ref_x1_cpu);
REGISTER_MPS_DISPATCH(call_hydro_ref_x1,
                      &snap::OnHost<&snap::hydro_ref_x1_cpu>::run);

}  // namespace at::native
