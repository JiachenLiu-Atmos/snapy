// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// C/C++
#include <cmath>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/utils/on_host.hpp>

#include "sed_hydro_dispatch.hpp"
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
                            torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                            torch::Tensor radius, torch::Tensor density,
                            torch::Tensor const_vsed, torch::Tensor hydro_ids,
                            SedParams p) {
  int nc3 = w.size(1), nc2 = w.size(2), nc1 = w.size(3);
  int nparticle = radius.size(0);
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cpu", [&] {
    using T = scalar_t;
    at::parallel_for(0, nc1 * nc2 * nc3, 0, [&](int64_t begin, int64_t end) {
      for (int64_t flat = begin; flat < end; ++flat) {
        sedimentation_flux_impl(
            w.data_ptr<T>(), temp.data_ptr<T>(), energy.data_ptr<T>(),
            flux.data_ptr<T>(), vsed.data_ptr<T>(),
            cosine_cell_kj.data_ptr<T>(), radius.data_ptr<T>(),
            density.data_ptr<T>(), const_vsed.data_ptr<T>(),
            hydro_ids.data_ptr<int64_t>(), nparticle, nc3, nc2, nc1,
            static_cast<int>(flat), p.il, p.iu, T(p.grav), T(p.gas_diameter),
            T(p.gas_epsilon_lj), T(p.gas_mass), T(p.upper_limit), T(M_PI),
            T(kintera::constants::KBoltz));
      }
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
