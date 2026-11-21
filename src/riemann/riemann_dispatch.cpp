// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/utils/on_host.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_kernels.h"

namespace snap {
namespace {

// One CPU loop for every face-function kind: each face is
// riemann_face<K, C, E>, C when the iterator carries riemann.central, E when
// it carries roe's LLF flag.
template <RiemannKind K, bool C, bool E>
void riemann_cpu_loop(at::TensorIterator& iter, int dim,
                      RiemannParams const& params,
                      torch::Tensor const& inv_mu_ratio_m1,
                      torch::Tensor const& cv_ratio_m1,
                      torch::Tensor const& u0) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "riemann_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    bool moist = params.ideal_moist;
    RiemannArgs<scalar_t> a{
        dim,
        static_cast<int>(nhydro - ICY),
        static_cast<int>(stride),
        iter.ninputs() == 6,
        moist,
        params.nvapor,
        scalar_t(params.gammad),
        moist ? inv_mu_ratio_m1.data_ptr<scalar_t>() : nullptr,
        moist ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr,
        moist ? u0.data_ptr<scalar_t>() : nullptr,
        params.dir_yz};
    int nt = iter.ntensors();

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          scalar_t* p[(C ? 9 : 8) + (E ? 1 : 0)];
          for (int64_t i = 0; i < n; ++i) {
            for (int k = 0; k < nt; ++k)
              p[k] = reinterpret_cast<scalar_t*>(data[k] + i * strides[k]);
            riemann_face<K, C, E>(p, a);
          }
        },
        grain_size);
  });
}

}  // namespace

void riemann_cpu(at::TensorIterator& iter, int kind, int dim,
                 RiemannParams const& params,
                 torch::Tensor const& inv_mu_ratio_m1,
                 torch::Tensor const& cv_ratio_m1, torch::Tensor const& u0) {
  auto k = static_cast<RiemannKind>(kind);
  bool c = params.central;
#define SNAP_RIEMANN_CPU(K)                                           \
  return c ? riemann_cpu_loop<K, true, false>(                        \
                 iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0) \
           : riemann_cpu_loop<K, false, false>(                       \
                 iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0)
  switch (k) {
    case RiemannKind::hllc:
      SNAP_RIEMANN_CPU(RiemannKind::hllc);
    case RiemannKind::lmars:
      SNAP_RIEMANN_CPU(RiemannKind::lmars);
    case RiemannKind::roe:
      if (params.llf)
        return c ? riemann_cpu_loop<RiemannKind::roe, true, true>(
                       iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0)
                 : riemann_cpu_loop<RiemannKind::roe, false, true>(
                       iter, dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
      SNAP_RIEMANN_CPU(RiemannKind::roe);
    case RiemannKind::shallow_roe:
      SNAP_RIEMANN_CPU(RiemannKind::shallow_roe);
    default:
      TORCH_CHECK(false, "[call_riemann] kind ", kind,
                  " is a torch formula, not a face function");
  }
#undef SNAP_RIEMANN_CPU
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_riemann);
REGISTER_ALL_CPU_DISPATCH(call_riemann, &snap::riemann_cpu);
REGISTER_MPS_DISPATCH(call_riemann, &snap::OnHost<&snap::riemann_cpu>::run);

}  // namespace at::native
