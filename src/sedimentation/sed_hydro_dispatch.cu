// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/constants.h>

// snap
#include "sed_hydro_dispatch.hpp"
#include "sed_hydro_impl.h"

namespace snap {

template <typename T>
__global__ void call_sedimentation_flux_cuda(
    T const* w, T const* temp, T const* energy, T* flux, T* vsed,
    T const* cosine_cell_kj, T const* radius, T const* density,
    T const* const_vsed, int64_t const* hydro_ids, int nparticle, int nc3,
    int nc2, int nc1, SedParams p) {
  int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= nc1 * nc2 * nc3) return;
  sedimentation_flux_impl(w, temp, energy, flux, vsed, cosine_cell_kj, radius,
                          density, const_vsed, hydro_ids, nparticle, nc3, nc2,
                          nc1, flat, p.il, p.iu, T(p.grav), T(p.gas_diameter),
                          T(p.gas_epsilon_lj), T(p.gas_mass), T(p.upper_limit),
                          T(M_PI), T(kintera::constants::KBoltz));
}

void sedimentation_flux_cuda(torch::Tensor w, torch::Tensor temp,
                             torch::Tensor energy, torch::Tensor flux,
                             torch::Tensor vsed, torch::Tensor cosine_cell_kj,
                             torch::Tensor radius, torch::Tensor density,
                             torch::Tensor const_vsed, torch::Tensor hydro_ids,
                             SedParams p) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nc3 = w.size(1), nc2 = w.size(2), nc1 = w.size(3);
  int threads = 128, blocks = (nc1 * nc2 * nc3 + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cuda", [&] {
    using T = scalar_t;
    call_sedimentation_flux_cuda<T>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            w.data_ptr<T>(), temp.data_ptr<T>(), energy.data_ptr<T>(),
            flux.data_ptr<T>(), vsed.data_ptr<T>(),
            cosine_cell_kj.data_ptr<T>(), radius.data_ptr<T>(),
            density.data_ptr<T>(), const_vsed.data_ptr<T>(),
            hydro_ids.data_ptr<int64_t>(), radius.size(0), nc3, nc2, nc1, p);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_sedimentation_flux,
                       &snap::sedimentation_flux_cuda);

}  // namespace at::native
