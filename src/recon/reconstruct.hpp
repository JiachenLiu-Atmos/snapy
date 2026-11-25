#pragma once

// C/C++
#include <memory>
#include <utility>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/snap.h>

#include <snap/utils/tensor_cast_cache.hpp>

#include "recon_point.h"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct InterpOptionsImpl {
  static std::shared_ptr<InterpOptionsImpl> create() {
    return std::make_shared<InterpOptionsImpl>();
  }

  InterpOptionsImpl() = default;
  explicit InterpOptionsImpl(std::string name) { type(name); }
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* scale = " << (scale() ? "true" : "false") << "\n";
  }

  ADD_ARG(std::string, type) = "dc";
  ADD_ARG(bool, scale) = false;
};
using InterpOptions = std::shared_ptr<InterpOptionsImpl>;

//! the scheme named by a reconstruct `type`
Scheme parse_scheme(std::string const& type);

//! the scheme of the velocity and pressure group: weno5 reconstructs them
//! with cp5 and weno3 with cp3; every other scheme is its own partner
Scheme velocity_pressure_scheme(Scheme s);

//! One reconstruction module for every scheme. dc, plm and ppm are tensor
//! code (ppm throws "not implemented"); cp3, weno3, cp5 and weno5 run the
//! stencil kernel (recon_dispatch) on their table, or on per-face x1 rows
//! after set_x1_coeffs (cp5 and weno5 only).
class InterpImpl : public torch::nn::Cloneable<InterpImpl> {
 public:
  //! Create and register an `Interp` module of scheme s
  static std::shared_ptr<InterpImpl> create(InterpOptions const& options,
                                            Scheme s, torch::nn::Module* p,
                                            std::string const& name);

  //! options with which this `Interp` was constructed
  InterpOptions options;
  Scheme scheme = Scheme::dc;

  //! uniform tables of the stencil schemes; per-face x1 rows (cp5, weno5)
  torch::Tensor cm, cp;
  torch::Tensor cm_x1, cp_x1;

  TensorCastCache cm_cast, cp_cast, cm_x1_cast, cp_x1_cast;

  InterpImpl() : InterpImpl(InterpOptionsImpl::create()) {}
  //! the scheme is options->type()
  explicit InterpImpl(InterpOptions const& options_);
  InterpImpl(InterpOptions const& options_, Scheme s);
  void reset() override;

  int stencils() const;

  void set_x1_coeffs(torch::Tensor const& x1f);
  bool has_x1_coeffs() const { return cm_x1.defined() && cm_x1.numel() > 0; }

  std::pair<torch::Tensor, torch::Tensor> forward(
      torch::Tensor w, int dim,
      torch::optional<torch::Tensor> wl = torch::nullopt,
      torch::optional<torch::Tensor> wr = torch::nullopt);

  void left(torch::Tensor w, int dim, torch::Tensor const& out);
  void right(torch::Tensor w, int dim, torch::Tensor const& out);

 private:
  void apply(torch::Tensor const& w, int dim, torch::Tensor const& out,
             torch::Tensor const& table, TensorCastCache& cache,
             torch::Tensor const& rows, TensorCastCache& rows_cache);
};
using Interp = std::shared_ptr<InterpImpl>;

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

struct ReconstructOptionsImpl {
  static std::shared_ptr<ReconstructOptionsImpl> create() {
    auto op = std::make_shared<ReconstructOptionsImpl>();
    op->interp() = InterpOptionsImpl::create();
    return op;
  }
  static std::shared_ptr<ReconstructOptionsImpl> from_yaml(
      std::string const& filename, std::string const& section);
  static std::shared_ptr<ReconstructOptionsImpl> from_yaml(
      const YAML::Node& node, std::string const& section);
  //! one reconstruct section node (shock, type, scale) at YAML path `path`;
  //! a missing key keeps its ADD_ARG default
  static std::shared_ptr<ReconstructOptionsImpl> from_node(
      const YAML::Node& node, std::string const& path);

  ReconstructOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- reconstruction options --\n";
    interp()->report(os);
    os << "* shock = " << (shock() ? "true" : "false") << "\n";
  }

  //! configure options; false as in the YAML reader (V3 C10)
  ADD_ARG(bool, shock) = false;

  //! abstract submodules
  ADD_ARG(InterpOptions, interp) = nullptr;
};
using ReconstructOptions = std::shared_ptr<ReconstructOptionsImpl>;

class HydroImpl;

class ReconstructImpl : public torch::nn::Cloneable<ReconstructImpl> {
 public:
  //! Create the module and register it as submodule `name` of `p`
  static std::shared_ptr<ReconstructImpl> create(
      ReconstructOptions const& opts, torch::nn::Module* p,
      std::string const& name = "recon");

  //! options with which this `Reconstruction` was constructed
  ReconstructOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  //! concrete submodules
  Interp pinterp1 = nullptr;
  Interp pinterp2 = nullptr;

  //! Constructor to initialize the layers
  ReconstructImpl() : options(ReconstructOptionsImpl::create()) {}
  explicit ReconstructImpl(const ReconstructOptions& options_,
                           torch::nn::Module* p = nullptr);
  void reset() override;

  void set_x1_faces(torch::Tensor const& x1f);

  //! w -> [wl, wr]
  // floor=false skips the reconstruction-stage density/pressure floors (the
  // caller reconstructs perturbation variables that are legitimately
  // negative and enforces positivity itself); the scalar floor is kept.
  torch::Tensor forward(torch::Tensor w, int dim, bool floor = true);

  //! FaceFloor's counts of the last forward (C.rst 3.7): (row, mask) pairs,
  //! row 0 density, 1 pressure, 2 species (a face counts once for any
  //! species row), mask a bool tensor over each side of the faces il..iu+1
  //! the reconstruction computes, true where the floor changed the face;
  //! the monitor reduces the stage's masks together (V3 S4). Only a floor
  //! that ran posts a pair, so a forward with the floors off launches
  //! nothing for the census. Empty without a parent Hydro.
  std::vector<std::pair<int, torch::Tensor>> floored;
};

TORCH_MODULE(Reconstruct);

class Conversion;
class CoordinateImpl;
class PrecisionPolicy;
class RegisterMonitor;
struct ReferenceFaces;

//! x1 well-balanced reconstruction: w' = w - ref, parity ghosts, recon of w',
//! restore ref at faces with the reference fallbacks, wall mirror. The
//! reference (X1Reference) and the split path's pressure_base1 are the
//! caller's.
struct WellBalancedX1 {
  CoordinateImpl const* coord = nullptr;
  PrecisionPolicy const* policy = nullptr;
  //! physical, non-outflow x1 walls (inner, outer): even-parity ghosts
  bool mirror[2] = {false, false};
  //! conversion order 4: the reflecting walls whose ghosts conv4 fills
  //! (wall4_rows) and whose wall face mirrors its owned side; trips counts
  bool wall4[2] = {false, false};
  Conversion const* conv4 = nullptr;
  torch::Tensor conv4_trips;
  //! dynamics/x1-mass-covariance: the velocity correction before the recon
  bool covariance = false;
  //! split path: the cell pressure background and its increment
  torch::Tensor pressure0, pressure_delta;

  //! the x1 face states [wl, wr] of wx1, which is changed and restored in
  //! place; recon (floors off) and the fallbacks count into monitor
  torch::Tensor forward(torch::Tensor wx1, ReferenceFaces const& ref,
                        ReconstructImpl& recon, RegisterMonitor& monitor) const;
};
}  // namespace snap

#undef ADD_ARG
