#pragma once

#include <torch/torch.h>

#include "hydro_dispatch.hpp"

namespace snap {

//! Outputs of the existing well-balanced reference kernel. The caller owns
//! seam exchange and the optional wb_ref4 correction, in that order.
struct HydroReferenceX1 {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;

  explicit HydroReferenceX1(torch::Tensor const& w) {
    auto shape = w.sizes().slice(1);
    psf_lo = torch::empty(shape, w.options());
    psf_hi = torch::empty(shape, w.options());
    pref = torch::empty(shape, w.options());
    dsf = torch::empty(shape, w.options());
    dref = torch::empty(shape, w.options());
  }
};

//! Shared grid setup and evaluation for runtime hydro and balance_column.
//! Explicit stretched grids always use quadrature. The legacy runtime
//! nonuniform branch is retained; initialization opts into stretched rows for
//! unequal widths. These are different operators, not accuracy settings.
class HydroReferenceX1Operator {
 public:
  HydroReferenceX1Operator(torch::Tensor const& dx1f, int iu, bool stretched,
                           bool phys_in, bool phys_out, bool wall_clamp,
                           X1RefGrid unequal = kX1RefNonUniform)
      : dx1f_(dx1f),
        version_(version(dx1f)),
        dtype_(dx1f.scalar_type()),
        device_(dx1f.device()),
        iu_(iu),
        stretched_(stretched),
        phys_in_(phys_in),
        phys_out_(phys_out),
        wall_clamp_(wall_clamp) {
    auto d = dx1f.to(torch::kCPU, torch::kFloat64);
    grid_ = stretched ? kX1RefStretched
                      : ((d.max() - d.min()).item<double>() <
                                 1.e-10 * d.mean().item<double>()
                             ? kX1RefUniform
                             : unequal);
    if (grid_ == kX1RefStretched) {
      rows_ =
          hydro_ref_x1_stretched_rows(dx1f, iu, phys_in, phys_out, wall_clamp);
    }
  }

  bool matches(torch::Tensor const& dx1f, int iu, bool stretched, bool phys_in,
               bool phys_out, bool wall_clamp) const {
    // Module::to preserves identity and version but changes dtype/device.
    return dx1f_.is_same(dx1f) && version_ == version(dx1f) &&
           dtype_ == dx1f.scalar_type() && device_ == dx1f.device() &&
           iu_ == iu && stretched_ == stretched && phys_in_ == phys_in &&
           phys_out_ == phys_out && wall_clamp_ == wall_clamp;
  }

  int grid() const { return grid_; }

  void evaluate(torch::Tensor const& w, double grav,
                torch::Tensor const& anchor, HydroReferenceX1& out) const {
    TORCH_CHECK(w.dim() == 4 && w.size(-1) == dx1f_.numel(),
                "hydro reference: state and grid shapes disagree");
    auto dx = dx1f_.to(w.options()).contiguous();
    auto rows =
        rows_.defined() ? rows_.to(w.options()).contiguous() : torch::Tensor();
    auto a = anchor.defined() ? anchor.to(w.options()).contiguous()
                              : torch::Tensor();
    at::native::call_hydro_ref_x1(w.device().type(), w.contiguous(), dx, a,
                                  out.psf_lo, out.psf_hi, out.pref, out.dsf,
                                  out.dref, iu_, grav, grid_, phys_in_,
                                  phys_out_, wall_clamp_, rows);
  }

 private:
  static int64_t version(torch::Tensor const& t) {
    return t.is_inference() ? 0 : t._version();
  }
  torch::Tensor dx1f_, rows_;
  int64_t version_;
  c10::ScalarType dtype_;
  torch::Device device_;
  int iu_, grid_;
  bool stretched_, phys_in_, phys_out_, wall_clamp_;
};

}  // namespace snap
