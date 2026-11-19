#pragma once

// C/C++
#include <array>
#include <memory>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include "x1_rows.hpp"

namespace snap {

class CoordinateImpl;
class InterpImpl;
class MeshBlockImpl;
class PrecisionPolicy;
class HydroReferenceX1Operator;

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
