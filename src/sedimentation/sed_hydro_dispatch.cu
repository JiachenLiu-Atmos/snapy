// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "sed_hydro_impl.h"

namespace snap {

template <typename T>
__global__ void call_sedimentation_flux_cuda(SedArrays<T> a, SedParams p) {
  int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat < a.nc1 * a.nc2 * a.nc3) sedimentation_flux_impl(a, p, flat);
}

void sedimentation_flux_cuda(torch::Tensor w, torch::Tensor temp,
                             torch::Tensor energy, torch::Tensor flux,
                             torch::Tensor vsed, torch::Tensor cos,
                             torch::Tensor radius, torch::Tensor density,
                             torch::Tensor const_vsed, torch::Tensor ids,
                             SedParams p) {
  at::cuda::CUDAGuard device_guard(w.device());
  int threads = 128, blocks = (w[0].numel() + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "sedimentation_flux_cuda", [&] {
    auto a = sed_arrays<scalar_t>(w, temp, energy, flux, vsed, cos, radius,
                                  density, const_vsed, ids);
    call_sedimentation_flux_cuda<scalar_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(a, p);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

namespace at::native {
REGISTER_CUDA_DISPATCH(call_sedimentation_flux,
                       &snap::sedimentation_flux_cuda);
}  // namespace at::native
