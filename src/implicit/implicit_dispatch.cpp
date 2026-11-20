// eigen
#include <Eigen/Dense>

// torch
#include <ATen/TensorIterator.h>
#include <torch/torch.h>

// snap
#include <snap/utils/on_host.hpp>

#include "implicit_dispatch.hpp"
#include "vic_kernels.h"

namespace snap {

//! vic_stage's launcher on a CPU (and MPS, through OnHost): the columns in
//! parallel, each one serially, its cells in order
struct CpuColumns {
  template <int NOPS, int NT = 0, typename F>
  static void columns(at::TensorIterator& iter, F const& f) {
    int grain_size = iter.numel() / at::get_num_threads();
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          char* p[NOPS];
          for (int64_t col = 0; col < n; ++col) {
            for (int k = 0; k < NOPS; ++k) p[k] = data[k] + col * strides[k];
            f(p);
          }
        },
        grain_size);
  }

  template <int NOPS, typename F>
  static void cells(at::TensorIterator& iter, int nlayer, F const& f) {
    columns<NOPS>(iter, [&](char* const* p) {
      for (int i = 0; i < nlayer; ++i) f(p, i);
    });
  }

  template <int NOPS, typename F, typename G>
  static void redistribute(at::TensorIterator& iter, int nlayer,
                           F const& column, G const& cell) {
    columns<NOPS>(iter, [&](char* const* p) {
      column(p);
      for (int i = 0; i < nlayer; ++i) cell(p, i);
    });
  }
};

template void vic_stage<CpuColumns>(at::TensorIterator&, VicParams const&);

// the donor sweep of MeshBlock's implicit tracer transfer, skipped when no
// donor can overdraw
void donor_cap_cpu(at::Tensor& P, at::Tensor const& held, int is, int ie) {
  int nf = ie - is;
  if (nf <= 0) return;
  auto qf = P.narrow(-1, is + 1, nf);
  auto cap = held.narrow(-1, is, nf).clone();
  cap.narrow(-1, 1, nf - 1) += P.narrow(-1, is + 1, nf - 1);
  auto binds = (qf > cap) | (qf < -held.narrow(-1, is + 1, nf));
  if (binds.any().item<bool>()) {
    auto avail = held.select(-1, is);
    for (int i = is + 1; i <= ie; ++i) {  // q: the face between i-1 and i
      auto q = P.select(-1, i);
      auto avail_up = held.select(-1, i);
      q.copy_(torch::maximum(torch::minimum(q, avail), -avail_up));
      avail = avail_up + q;
    }
  }
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(vic_stage);
DEFINE_DISPATCH(donor_cap);

REGISTER_ALL_CPU_DISPATCH(vic_stage, &snap::vic_stage<snap::CpuColumns>);
REGISTER_ALL_CPU_DISPATCH(donor_cap, &snap::donor_cap_cpu);

REGISTER_MPS_DISPATCH(vic_stage,
                      &snap::OnHost<&snap::vic_stage<snap::CpuColumns>>::run);
REGISTER_MPS_DISPATCH(donor_cap, &snap::donor_cap_cpu);  // plain torch ops

}  // namespace at::native
