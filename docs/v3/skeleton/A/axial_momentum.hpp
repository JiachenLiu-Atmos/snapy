// V3 lane A skeleton: AAM as a register consumer, its torque rule and the
// closure fixer (A.rst 3.1-3.3, 5.2, 6.2). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap {

//! dynamics/conserve-axial-momentum (A.rst 3.2, settled). "full" and the
//! conserve-axial-momentum-closure key are refused with a message naming exact
enum class AxialMode : uint8_t { off, transport, exact };

//! register reads (A.rst 3.1, 4.1); views valid to the next begin_stage
//! (R:9-12, 20); undefined = direction inactive
struct AxialFluxView {
  torch::Tensor total[3];    //!< A-r1: reg.total(d), after the limiter carry
  torch::Tensor central[3];  //!< A-r2: reg.kept(d, "riemann.central"), x2/x3
  torch::Tensor momentum_t;  //!< A-r5: vic.momentum_t[2], kept per face (R-4)
};

//! flux-form torque of F_d per volume. Rule 1: #363's -(Omega/V) sum_f M_d,f
//! (s_f^2 - s_c^2); rule 2: the side study's, if it passes (A.rst 3.2)
class FaceTorqueRule {
 public:
  virtual ~FaceTorqueRule() = default;
  virtual torch::Tensor rate(AxialFluxView const& f, RotationGeometry const& g,
                             double omega) const = 0;
  virtual bool closes_globally() const = 0;  //!< true: no closure fixer
};

//! lane-A consumer. Owned by Hydro; holds no flux state between stages
class AxialMomentum {
 public:
  AxialMomentum(AxialMode mode, double omega,
                std::unique_ptr<FaceTorqueRule> rule);
  AxialMode mode() const;
  //! parts lane A asks R to keep (R:56-60): riemann.central@2/@3 in exact,
  //! vic.momentum_t@1 (R-4); vic.mass_total is not needed (A.rst 3.4)
  RegisterRequest request() const;
  //! reset checks: z axis (others refused, FOR CHENG); exact needs FP64, no
  //! perturbation-flux (until G10, A.rst 6.2), nu_iso == 0 (until phase 3)
  void check(StageState const& s, bool fp32, bool perturbation_flux,
             double nu_iso) const;
  //! divergence phase: rewrite the K component of du's divergence from
  //! total(d) (today's conserve_axial_momentum_, same arithmetic)
  void flux_form(torch::Tensor div, AxialFluxView const& f,
                 RotationGeometry const& g) const;
  //! exact, pre_implicit: add the torque of F_d = total - central along K,
  //! and record this block's closure sums {R, I} on the device
  void add_torque(torch::Tensor du, torch::Tensor force_cell,
                  AxialFluxView const& f, StageState const& s);
  //! post_implicit: today's axial_x1_source_, from vic.momentum_t
  void x1_source(torch::Tensor du, AxialFluxView const& f,
                 RotationGeometry const& g) const;
  //! step end: K-weighted x1-wall reduction of reg.step_field(1) (R:91),
  //! posted as cell_amount("aam.x1_wall") (R-4: lane A owns it)
  void wall_torque(FluxRegister& reg, RotationGeometry const& g) const;
  torch::Tensor closure_sums() const;  //!< {R, I}, FP64 device; exact only
 private:
  AxialMode mode_;
  double omega_;
  std::unique_ptr<FaceTorqueRule> rule_;
  torch::Tensor sums_;  //!< {R, I}, written by add_torque
  torch::Tensor rho_;   //!< interior density the increment scales with
};

//! the closure as a registered fixer (A.rst 3.3, 5.2), R:106-116; the
//! registry is lane X's. {R, I} FP64 on device; X's global phase reduces
//! them via host (never captured; X.rst:98, 187)
class AxialClosureFixer : public Fixer {
 public:
  explicit AxialClosureFixer(AxialMomentum const* aam);  //!< non-owning
  std::string name() const override;          //!< "aam.closure"
  std::vector<int> rows() const override;     //!< {IVY, IVZ}
  torch::Tensor partial_sums() const override;  //!< {R, I}, FP64
  //! -w2 dt R / I rho s^2 along K on device (where-masked); stage_weight
  //! must be the stage's own w2 (pin meshblock.cpp:1027), not c_s (R-6);
  //! posts reg.cell_amount("aam.closure", V du) (R:83)
  void apply(torch::Tensor u, torch::Tensor const& global_sums, double dt,
             double stage_weight, FluxRegister& reg) const override;
 private:
  AxialMomentum const* aam_;
};

}  // namespace snap
