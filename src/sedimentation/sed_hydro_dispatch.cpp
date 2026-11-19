// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "sed_hydro_impl.h"

namespace snap {

void sedimentation_flux(torch::Tensor w, torch::Tensor temp,
                        torch::Tensor energy, torch::Tensor flux,
                        torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                        torch::Tensor radius, torch::Tensor density,
                        torch::Tensor const_vsed, torch::Tensor hydro_ids,
                        SedParams p) {
  for (auto const& t : {w, temp, energy, flux, vsed, cosine_cell_kj, radius,
                        density, const_vsed}) {
    TORCH_CHECK(t.device() == w.device() && t.is_contiguous() &&
                    t.scalar_type() == w.scalar_type(),
                "[SedHydro] kernel inputs need one device, one dtype and "
                "contiguous storage");
  }
  TORCH_CHECK(w.dim() == 4 && flux.sizes() == w.sizes() &&
                  temp.numel() == w[0].numel() &&
                  energy.numel() >= (w.size(0) - ICY) * w[0].numel() &&
                  vsed.size(0) == radius.size(0) &&
                  vsed[0].numel() == w[0].numel() &&
                  cosine_cell_kj.numel() == w.size(1) * w.size(2) &&
                  hydro_ids.scalar_type() == torch::kLong &&
                  hydro_ids.device() == w.device() && p.il >= -1 &&
                  p.il <= p.iu && p.iu <= w.size(3),
              "[SedHydro] kernel inputs have inconsistent shapes");
  at::native::call_sedimentation_flux(w.device().type(), w, temp, energy, flux,
                                      vsed, cosine_cell_kj, radius, density,
                                      const_vsed, hydro_ids, p);
}

void sedimentation_flux_cpu(torch::Tensor w, torch::Tensor temp,
                            torch::Tensor energy, torch::Tensor flux,
                            torch::Tensor vsed, torch::Tensor cos,
                            torch::Tensor radius, torch::Tensor density,
                            torch::Tensor const_vsed, torch::Tensor ids,
                            SedParams p) {
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cpu", [&] {
    auto a = sed_arrays<scalar_t>(w, temp, energy, flux, vsed, cos, radius,
                                  density, const_vsed, ids);
    at::parallel_for(0, a.nc1 * a.nc2 * a.nc3, 0, [&](int64_t b, int64_t e) {
      for (int64_t f = b; f < e; ++f) sedimentation_flux_impl(a, p, int(f));
    });
  });
}

}  // namespace snap

namespace at::native {
DEFINE_DISPATCH(call_sedimentation_flux);
REGISTER_ALL_CPU_DISPATCH(call_sedimentation_flux,
                          &snap::sedimentation_flux_cpu);
REGISTER_MPS_DISPATCH(call_sedimentation_flux,
                      &snap::OnHost<&snap::sedimentation_flux_cpu>::run);
}  // namespace at::native
