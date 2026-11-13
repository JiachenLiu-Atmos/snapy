// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// C/C++
#include <cmath>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>

#include "sed_hydro_dispatch.hpp"
#include "sed_hydro_impl.h"

namespace snap {
namespace {

void check_sedimentation_args(torch::Tensor w, torch::Tensor flux,
                              torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                              torch::Tensor radius, torch::Tensor density,
                              torch::Tensor const_vsed, torch::Tensor hydro_ids,
                              torch::Tensor inv_mu_ratio_m1,
                              torch::Tensor cv_ratio_m1, torch::Tensor u0,
                              int il, int iu, int ny, int nvapor) {
  TORCH_CHECK(
      w.device() == flux.device() && w.device() == vsed.device() &&
          w.device() == cosine_cell_kj.device() &&
          w.device() == radius.device() && w.device() == density.device() &&
          w.device() == const_vsed.device() &&
          w.device() == hydro_ids.device() &&
          w.device() == inv_mu_ratio_m1.device() &&
          w.device() == cv_ratio_m1.device() && w.device() == u0.device(),
      "sedimentation_flux_dispatch requires tensors on one device");
  TORCH_CHECK(w.is_contiguous() && flux.is_contiguous() &&
                  vsed.is_contiguous() && cosine_cell_kj.is_contiguous() &&
                  radius.is_contiguous() && density.is_contiguous() &&
                  const_vsed.is_contiguous() && hydro_ids.is_contiguous() &&
                  inv_mu_ratio_m1.is_contiguous() &&
                  cv_ratio_m1.is_contiguous() && u0.is_contiguous(),
              "sedimentation_flux_dispatch requires contiguous tensors");
  TORCH_CHECK(w.sizes() == flux.sizes(),
              "sedimentation_flux_dispatch requires w/flux shape match");
  TORCH_CHECK(w.dim() == 4,
              "sedimentation_flux_dispatch expects state [nvar,nc3,nc2,nc1]");
  TORCH_CHECK(
      vsed.dim() == 4 && vsed.size(1) == w.size(1) &&
          vsed.size(2) == w.size(2) && vsed.size(3) == w.size(3),
      "sedimentation_flux_dispatch expects vsed [nparticle,nc3,nc2,nc1]");
  TORCH_CHECK(cosine_cell_kj.dim() == 2 &&
                  cosine_cell_kj.size(0) == w.size(1) &&
                  cosine_cell_kj.size(1) == w.size(2),
              "sedimentation_flux_dispatch expects cosine_cell_kj [nc3,nc2]");
  TORCH_CHECK(
      radius.dim() == 1 && density.sizes() == radius.sizes() &&
          const_vsed.sizes() == radius.sizes() &&
          hydro_ids.sizes() == radius.sizes() && vsed.size(0) == radius.size(0),
      "sedimentation_flux_dispatch particle arrays must have matching sizes");
  TORCH_CHECK(inv_mu_ratio_m1.dim() == 1 && cv_ratio_m1.dim() == 1 &&
                  inv_mu_ratio_m1.size(0) >= ny && cv_ratio_m1.size(0) >= ny &&
                  u0.dim() == 1 && u0.size(0) >= ny + 1,
              "sedimentation_flux_dispatch received inconsistent EOS buffers");
  TORCH_CHECK(hydro_ids.scalar_type() == torch::kLong,
              "sedimentation_flux_dispatch expects int64 hydro_ids");
  TORCH_CHECK(w.scalar_type() == flux.scalar_type() &&
                  w.scalar_type() == vsed.scalar_type() &&
                  w.scalar_type() == cosine_cell_kj.scalar_type() &&
                  w.scalar_type() == radius.scalar_type() &&
                  w.scalar_type() == density.scalar_type() &&
                  w.scalar_type() == const_vsed.scalar_type() &&
                  w.scalar_type() == inv_mu_ratio_m1.scalar_type() &&
                  w.scalar_type() == cv_ratio_m1.scalar_type() &&
                  w.scalar_type() == u0.scalar_type(),
              "sedimentation_flux_dispatch requires matching floating dtypes");
  TORCH_CHECK(il >= -1 && il <= iu && iu <= w.size(3),
              "sedimentation_flux_dispatch received invalid active x1 bounds");
  TORCH_CHECK(nvapor >= 0 && nvapor <= ny,
              "sedimentation_flux_dispatch received invalid nvapor/ny");
}

torch::Tensor sedimentation_feps_tensor(torch::Tensor w, int ny, int nvapor,
                                        torch::Tensor inv_mu_ratio_m1) {
  auto sizes = w.sizes().vec();
  sizes.erase(sizes.begin());
  auto feps = torch::ones(sizes, w.options());
  if (nvapor > 0) {
    std::vector<int64_t> vec = {nvapor, 1, 1, 1};
    feps += (w.narrow(0, ICY, nvapor) *
             inv_mu_ratio_m1.narrow(0, 0, nvapor).view(vec))
                .sum(0);
  }
  if (ny > nvapor) {
    feps -= w.narrow(0, ICY + nvapor, ny - nvapor).sum(0);
  }
  return feps;
}

}  // namespace

void sedimentation_flux_dispatch(
    torch::Tensor w, torch::Tensor flux, torch::Tensor vsed,
    torch::Tensor cosine_cell_kj, torch::Tensor radius, torch::Tensor density,
    torch::Tensor const_vsed, torch::Tensor hydro_ids,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int il, int iu, int ny, int nvapor, double grav, double gas_constant_dry,
    double cv_dry, double gas_diameter, double gas_epsilon_lj, double gas_mass,
    double upper_limit) {
  check_sedimentation_args(w, flux, vsed, cosine_cell_kj, radius, density,
                           const_vsed, hydro_ids, inv_mu_ratio_m1, cv_ratio_m1,
                           u0, il, iu, ny, nvapor);
  at::native::call_sedimentation_flux(
      w.device().type(), w, flux, vsed, cosine_cell_kj, radius, density,
      const_vsed, hydro_ids, inv_mu_ratio_m1, cv_ratio_m1, u0, il, iu, ny,
      nvapor, grav, gas_constant_dry, cv_dry, gas_diameter, gas_epsilon_lj,
      gas_mass, upper_limit);
}

void sedimentation_flux_cpu(torch::Tensor w, torch::Tensor flux,
                            torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                            torch::Tensor radius, torch::Tensor density,
                            torch::Tensor const_vsed, torch::Tensor hydro_ids,
                            torch::Tensor inv_mu_ratio_m1,
                            torch::Tensor cv_ratio_m1, torch::Tensor u0, int il,
                            int iu, int ny, int nvapor, double grav,
                            double gas_constant_dry, double cv_dry,
                            double gas_diameter, double gas_epsilon_lj,
                            double gas_mass, double upper_limit) {
  int nvar = w.size(0);
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int ncells = nc1 * nc2 * nc3;
  int nparticle = radius.size(0);

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cpu", [&] {
    at::parallel_for(0, ncells, 0, [&](int64_t begin, int64_t end) {
      for (int64_t flat = begin; flat < end; ++flat) {
        sedimentation_flux_impl(
            w.data_ptr<scalar_t>(), flux.data_ptr<scalar_t>(),
            vsed.data_ptr<scalar_t>(), cosine_cell_kj.data_ptr<scalar_t>(),
            radius.data_ptr<scalar_t>(), density.data_ptr<scalar_t>(),
            const_vsed.data_ptr<scalar_t>(), hydro_ids.data_ptr<int64_t>(),
            inv_mu_ratio_m1.data_ptr<scalar_t>(),
            cv_ratio_m1.data_ptr<scalar_t>(), u0.data_ptr<scalar_t>(),
            nparticle, ny, nvapor, nvar, nc3, nc2, nc1, static_cast<int>(flat),
            il, iu, scalar_t(grav), scalar_t(gas_constant_dry),
            scalar_t(cv_dry), scalar_t(gas_diameter), scalar_t(gas_epsilon_lj),
            scalar_t(gas_mass), scalar_t(upper_limit), scalar_t(M_PI),
            scalar_t(kintera::constants::KBoltz));
      }
    });
  });
}

void sedimentation_flux_mps(torch::Tensor w, torch::Tensor flux,
                            torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                            torch::Tensor radius, torch::Tensor density,
                            torch::Tensor const_vsed, torch::Tensor hydro_ids,
                            torch::Tensor inv_mu_ratio_m1,
                            torch::Tensor cv_ratio_m1, torch::Tensor u0, int il,
                            int iu, int ny, int nvapor, double grav,
                            double gas_constant_dry, double cv_dry,
                            double gas_diameter, double gas_epsilon_lj,
                            double gas_mass, double upper_limit) {
  using namespace kintera::constants;

  std::vector<int64_t> vec(w.dim(), 1);
  vec[0] = -1;
  auto feps = sedimentation_feps_tensor(w, ny, nvapor, inv_mu_ratio_m1);
  auto temp = w[IPR] / (w[IDN] * gas_constant_dry * feps);

  auto eta = (5.0 / 16.0) * std::sqrt(M_PI * KBoltz) * std::sqrt(gas_mass) *
             torch::sqrt(temp) *
             torch::pow(KBoltz / gas_epsilon_lj * temp, 0.16) /
             (M_PI * gas_diameter * gas_diameter * 1.22);
  auto lambda = eta * torch::sqrt(M_PI * KBoltz * temp) /
                (w[IPR] * std::sqrt(2.0 * gas_mass));
  auto kn = lambda / radius.view(vec);
  vsed.copy_(1.0 + kn * (1.256 + 0.4 * torch::exp(-1.1 / kn)));
  vsed *= 2.0 * radius.view(vec) * radius.view(vec) * grav *
          (density.view(vec) - w[IDN]);
  vsed /= 9.0 * eta;
  auto cvsed = const_vsed.view(vec);
  vsed.copy_(torch::where(cvsed != 0., cvsed, vsed));
  vsed.clamp_(-upper_limit, upper_limit);
  auto vcell = vsed.clone();

  auto vel = w.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, cosine_cell_kj.unsqueeze(-1));
  auto ke = 0.5 * (w.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  auto species_ids = hydro_ids - ICY;
  auto rhos = w[IDN] * w.index_select(0, hydro_ids);
  auto rhos_vsed = rhos * vcell;
  auto species_energy =
      rhos * (u0.index_select(0, species_ids + 1).view(vec) +
              (cv_ratio_m1.index_select(0, species_ids).view(vec) + 1.) *
                  cv_dry * temp +
              ke);

  // each particle is carried from its upwind cell; walls are sealed
  vsed.copy_(sedimentation_upwind(vcell, vcell, il, iu));
  flux.index_add_(0, hydro_ids, sedimentation_upwind(rhos_vsed, vcell, il, iu));
  flux.narrow(0, IVX, 3) +=
      sedimentation_upwind(vel.unsqueeze(1) * rhos_vsed, vcell, il, iu).sum(1);
  flux[IPR] +=
      sedimentation_upwind(vcell * species_energy, vcell, il, iu).sum(0);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_sedimentation_flux);

REGISTER_ALL_CPU_DISPATCH(call_sedimentation_flux,
                          &snap::sedimentation_flux_cpu);
REGISTER_MPS_DISPATCH(call_sedimentation_flux, &snap::sedimentation_flux_mps);

}  // namespace at::native
