#pragma once

// C/C++
#include <cstdint>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

class HydroImpl;
class GnomonicEquiangleImpl;
class FluxRegister;
struct RegisterRequest;

//! dynamics/conserve-axial-momentum (A.rst 3.2): exact is transport plus the
//! flux-form Coriolis torque and its closure (the old full + closure key)
enum class AxialMode : uint8_t { off, transport, exact };

//! Lane A's AAM consumer of the flux register (A.rst 3.1-3.3, 3.8). It reads
//! total(d) and kept(d, "riemann.central") and never evaluates a flux; the
//! rotation data (k, its face averages, cos^2 lat) stays with coord. Owned by
//! Hydro; mode() is off where the option does not act: another grid, or a
//! Coriolis force that is not of type xyz about omega1.
class AxialMomentum {
 public:
  //! reads the deck word and makes reset()'s refusals (A.rst 3.2, 10.3)
  explicit AxialMomentum(HydroImpl const* hydro);
  AxialMode mode() const { return mode_; }
  //! test hook (A.rst 3.2): exact without its closure, the old full
  bool closure = true;
  bool closes() const { return mode_ == AxialMode::exact && closure; }
  //! exact keeps riemann.central@2/@3 (A-r2)
  void request(RegisterRequest& req) const;

  //! divergence phase: the K component of div's momentum rows becomes the
  //! divergence of the face AAM fluxes A F . K_f of total(d) (A-r1), so the
  //! AAM leaving a cell enters its neighbour; a null flux is skipped
  void flux_form(torch::Tensor div, torch::Tensor flux1, torch::Tensor flux2,
                 torch::Tensor flux3) const;
  //! Hydro arms the torque for the pre_implicit Coriolis call of a stage
  //! (ny species rows) and disarms it after: once per stage
  void arm(int ny) { ny_ = ny, armed_ = true; }
  void disarm() { armed_ = false; }
  //! exact, pre_implicit: along K, the flux-form torque of F_d = total(d) -
  //! kept central on the x2/x3 faces, added to the cell-form force; with the
  //! closure also this block's {R, I} and density
  void add_torque(torch::Tensor force, torch::Tensor const& w, double omega);
  //! post_implicit: the K component of the x1 metric source rows puts the
  //! tangential momentum the implicit solve moved (corr, may be undefined)
  //! at the x1 face radius
  void x1_source(torch::Tensor row2, torch::Tensor row3,
                 torch::Tensor const& corr) const;
  //! K . rows = target (an AAM rate per volume, interior) with the change
  //! along K; a cell on the axis hands its target to its four neighbours
  void rate_to_rows(torch::Tensor row2, torch::Tensor row3,
                    torch::Tensor target) const;
  //! sum V r K . m over the interior of two covariant momentum rows: the AAM
  //! L of u's rows, or a source's share of it from its increment (A.rst 5.1)
  torch::Tensor aam(torch::Tensor const& row2, torch::Tensor const& row3) const;

 private:
  friend class AxialClosureFixer;
  HydroImpl const* hydro_;
  GnomonicEquiangleImpl const* pg_ = nullptr;
  AxialMode mode_ = AxialMode::off;
  bool on_axis_ = false;  //!< a cell centred on the axis, found at reset
  int ny_ = 0;            //!< species rows of the armed stage
  bool armed_ = false;
  torch::Tensor sums_;  //!< {R, I}, FP64 on the device
  torch::Tensor rho_;   //!< interior density the increment scales with
  //! rule 1 (#363): -(Omega / V) sum_f M_f (s_f^2 - s_c^2), s = r cos(lat)
  torch::Tensor rate(torch::Tensor flux2, torch::Tensor flux3, double omega,
                     int ny) const;
};

//! The closure as a fixer (A.rst 3.3, 5.2), with the interface of lane R's
//! Fixer (R:104-116), which it derives from once that class exists. Per
//! stage the residual R = sum V (K . F_cell - rate(central)) leaves as one
//! solid-body increment d(rho u_east) = -w2 dt R / I rho s, I = sum V rho s^2.
class AxialClosureFixer {
 public:
  explicit AxialClosureFixer(AxialMomentum* aam) : aam_(aam) {}
  std::string name() const { return "aam.closure"; }
  std::vector<int> rows() const;
  //! this block's {R, I}, FP64 on the device; zeros before the first torque
  torch::Tensor partial_sums() const;
  //! global_sums: {R, I} over all blocks; stage_weight: w2 of the stage just
  //! combined. No host read: a device mask skips I <= 0 or a non-finite R.
  //! Posts cell_amount("aam.closure", V du, 1.0, fixer); returns this
  //! block's AAM change, -w2 dt R I_b / I (the "aam" row's amount).
  torch::Tensor apply(torch::Tensor u, torch::Tensor const& global_sums,
                      double dt, double stage_weight, FluxRegister& reg) const;

 private:
  AxialMomentum* aam_;
};

}  // namespace snap
