#pragma once

// C/C++
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/hydro/hydro_dispatch.hpp>

#include "x1_rows.hpp"

namespace snap {

class CoordinateImpl;
class InterpImpl;
class MeshBlockImpl;
class PrecisionPolicy;

//! Outputs of the existing well-balanced reference kernel. The caller owns
//! seam exchange and the optional wb_ref4 correction, in that order.
struct HydroReferenceX1 {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;

  HydroReferenceX1() = default;
  explicit HydroReferenceX1(torch::Tensor const& w) {
    for (auto* t : {&psf_lo, &psf_hi, &pref, &dsf, &dref})
      *t = torch::empty(w.sizes().slice(1), w.options());
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
    auto d = dx1f.cpu().to(torch::kFloat64);
    grid_ = stretched ? kX1RefStretched
                      : ((d.max() - d.min()).item<double>() <
                                 1.e-10 * d.mean().item<double>()
                             ? kX1RefUniform
                             : unequal);
    if (grid_ == kX1RefStretched) {
      rows_ = six_face_rows(dx1f, iu, phys_in, phys_out, wall_clamp);
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

  //! Linear pressure scan anchored at the upper face of cell iu; returns
  //! {lower, upper} face pressure (lane-c-notes.md)
  std::pair<torch::Tensor, torch::Tensor> scan_pressure(
      torch::Tensor const& rho, double grav,
      torch::Tensor const& anchor) const {
    TORCH_CHECK(rho.dim() == 3 && rho.numel() > 0 && dx1f_.dim() == 1 &&
                    rho.size(-1) == dx1f_.numel() && iu_ >= 0 &&
                    iu_ < rho.size(-1),
                "hydro pressure scan: density and grid shapes disagree");
    TORCH_CHECK(rho.scalar_type() == torch::kFloat32 ||
                    rho.scalar_type() == torch::kFloat64,
                "hydro pressure scan: density must be float32 or float64");
    TORCH_CHECK(
        anchor.defined() && anchor.dim() == 3 &&
            anchor.size(0) == rho.size(0) && anchor.size(1) == rho.size(1) &&
            anchor.size(2) == 1,
        "hydro pressure scan: explicit anchor must have shape [nc3,nc2,1]");
    auto lo = torch::empty(rho.sizes(), rho.options());
    auto hi = torch::empty_like(lo);
    at::native::call_hydro_pressure_scan_x1(
        rho.device().type(), rho.contiguous(),
        dx1f_.to(rho.options()).contiguous(),
        anchor.to(rho.options()).contiguous(), lo, hi, iu_, grav);
    return {lo, hi};
  }

  //! Packed [base, finite increment] evaluation of the SAME nonlinear kernel.
  //! Geometry alone is cached. Branch-dependent reference values are rebuilt
  //! each sweep, avoiding stale guards after heating, restart or regridding.
  void evaluate_split(torch::Tensor const& w0, torch::Tensor const& dw,
                      double grav, torch::Tensor const& anchor,
                      HydroReferenceX1& out) const {
    TORCH_CHECK(w0.dim() == 4 && w0.sizes() == dw.sizes() &&
                    w0.size(-1) == dx1f_.numel(),
                "split hydro reference: incompatible state shapes");
    TORCH_CHECK(w0.device().is_cpu() || w0.device().is_cuda(),
                "split hydro reference requires FP64 on CPU or CUDA");
    TORCH_CHECK(!anchor.defined() ||
                    (anchor.dim() == 4 && anchor.size(0) == w0.size(1) &&
                     anchor.size(1) == w0.size(2) && anchor.size(2) == 1 &&
                     anchor.size(3) == 2),
                "split hydro reference: anchor must be [nc3,nc2,1,2]");
    auto op = w0.options().dtype(torch::kFloat64);
    auto pair = [&](torch::Tensor const& a) {
      auto d = a.to(op);
      return torch::stack({d, torch::zeros_like(d)}, -1).contiguous();
    };
    auto w = torch::stack({w0.to(op), dw.to(op)}, -1).contiguous();
    out = HydroReferenceX1(w);
    at::native::call_hydro_ref_x1(
        w.device().type(), w, pair(dx1f_),
        anchor.defined() ? anchor.to(op).contiguous() : torch::Tensor(),
        out.psf_lo, out.psf_hi, out.pref, out.dsf, out.dref, iu_, grav, grid_,
        phys_in_, phys_out_, wall_clamp_,
        rows_.defined() ? pair(rows_) : torch::Tensor());
  }

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

//! total path: delta undefined. Split path (perturbation flux): base is the
//! FP64 background, delta the perturbation; the kernel's nonlinear decisions
//! read the FP64 total, its arithmetic keeps the increment
struct ReferenceInput {
  torch::Tensor base, delta;
};

//! the well-balanced x1 reference of one evaluation
struct ReferenceFaces {
  //! totals: face pressure (lower), cell pressure, cell density
  torch::Tensor psf_lo, pref, dref;
  //! (2, ...): the face density on the ILT and IRT side of each face
  torch::Tensor dsf_lr;
  //! split path only: the background and increment parts of psf_lo and pref;
  //! gauge (FP32 runs) the FP64 face gauge
  torch::Tensor face0, face_delta, cell0, cell_delta, gauge;
};

//! the x1 block column a block belongs to: the anchor relay down the column
//! and the reference ghost rows. Absent (nullptr) for a single column.
struct X1ColumnLink {
  //! the column is split along x1 (a neighbour below or above)
  virtual bool split() const = 0;
  //! the running face value from the block above; undefined at the top
  virtual torch::Tensor take_anchor(torch::Tensor const& like) = 0;
  //! true when a block below takes this block's bottom-face value
  virtual bool passes_anchor() const = 0;
  virtual void pass_anchor(torch::Tensor const& face) = 0;
  //! replace the ng ghost rows of each field by the neighbours' owned rows
  virtual void exchange_ghost_rows(std::vector<torch::Tensor> rows, int ng) = 0;
  virtual ~X1ColumnLink() = default;
};

//! X1ColumnLink over the layout's Exchange relays (rows 6-8 of the stage
//! schedule), with their tags
class ExchangeX1Link : public X1ColumnLink {
 public:
  explicit ExchangeX1Link(MeshBlockImpl const* pmb);
  bool split() const override { return below_ >= 0 || above_ >= 0; }
  torch::Tensor take_anchor(torch::Tensor const& like) override;
  bool passes_anchor() const override { return below_ >= 0; }
  void pass_anchor(torch::Tensor const& face) override;
  void exchange_ghost_rows(std::vector<torch::Tensor> rows, int ng) override;

 private:
  MeshBlockImpl const* pmb_;
  int below_ = -1, above_ = -1;
};

struct X1ReferenceConfig {
  CoordinateImpl const* coord = nullptr;
  PrecisionPolicy const* policy = nullptr;
  bool phys_in = false, phys_out = false, wall_clamp = false;
  //! reflecting x1 walls (inner, outer): one face density on both sides
  std::array<bool, 2> reflecting{false, false};
  //! 4 adds the face-density rows (wb_ref4)
  int conversion_order = 2;
  //! the x1 reconstruction: cp5 face-density rows for weno5/cp5, with the
  //! reconstruction's own x1 coefficients if it has them
  std::string interp;
  bool interp_x1_coeffs = false;
};

//! one evaluator of the well-balanced x1 reference; it owns the caches of the
//! kernel operator, the wb_ref4 rows and the cp5 face-density rows
class X1Reference {
 public:
  //! one evaluation; order: anchor take, kernel, wb_ref4 flag, anchor pass,
  //! ghost-row exchange, wb_ref4 faces, per-side face density
  ReferenceFaces evaluate(X1ReferenceConfig const& cfg,
                          ReferenceInput const& in, double grav,
                          X1ColumnLink* link) const;

 private:
  mutable std::shared_ptr<HydroReferenceX1Operator> op_;
  //! conversion-order 4 face-density rows, rebuilt when their key changes
  mutable std::shared_ptr<FaceDensityRows> wb_ref4_;
  //! the x1 reconstruction's optimal-linear rows (cp5) and their key
  mutable std::shared_ptr<InterpImpl> dsf_lin_;
  mutable X1RowsKey dsf_lin_key_;
  //! key 4: (side, face) rows of the reflecting x1 walls that keep wb_ref4's
  //! face density, and the block layout they were built for
  mutable torch::Tensor wb4_lr_side_, wb4_lr_col_;
  mutable std::array<int64_t, 3> wb4_lr_at_{-1, -1, -1};
};

}  // namespace snap
