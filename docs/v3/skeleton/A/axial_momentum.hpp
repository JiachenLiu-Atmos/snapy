// V3 lane A skeleton: AAM as a register consumer, its torque rule and the
// closure fixer (A.rst 3.1-3.3, 5.2, 6.2). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap::v3 {

class Fixer;  // assumed †, fixer base class (lane R list; registry X-1)

//! dynamics/conserve-axial-momentum (A.rst 3.2, settled). "full" and the
//! conserve-axial-momentum-closure key are refused with a message naming exact
enum class AxialMode : uint8_t { off, transport, exact };

//! register views AAM reads (A.rst 3.1); undefined = direction inactive
struct AxialFluxView {
  torch::Tensor flux[3];     //!< A-r1: total face flux, all rows
  torch::Tensor central[3];  //!< A-r2: riemann.central† (one summed row)
  torch::Tensor x1_transfer;  //!< A-r3 + vic.mass_total (lane I), a rate
  torch::Tensor x1_tangential_corr;  //!< vic.momentum_t[2] (lane I)
  int ny = 0;                          //!< species rows after IDN
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
  //! reset checks: an axis exists; exact needs FP64, no perturbation-flux
  //! (until G10, A.rst 6.2) and nu_iso == 0 (until phase 3, A.rst 3.7)
  void check(StageState const& s, bool fp32, bool perturbation_flux,
             double nu_iso) const;
  //! step 2: rewrite the K component of the register's divergence from
  //! the face fluxes (today's conserve_axial_momentum_, same arithmetic)
  void flux_form(torch::Tensor div, AxialFluxView const& f,
                 RotationGeometry const& g) const;
  //! exact, pre-implicit: add the torque of F_d = F - F_c along K, and
  //! record this block's closure sums {R, I} on the device
  void add_torque(torch::Tensor du, torch::Tensor force_cell,
                  AxialFluxView const& f, StageState const& s);
  //! post-implicit: today's axial_x1_source_ (implicit transfer at r_face)
  void x1_source(torch::Tensor du, AxialFluxView const& f,
                 RotationGeometry const& g) const;
  torch::Tensor closure_sums() const;  //!< {R, I}, FP64 device; exact only
 private:
  AxialMode mode_;
  double omega_;
  std::unique_ptr<FaceTorqueRule> rule_;
  torch::Tensor sums_;  //!< {R, I}, written by add_torque
  torch::Tensor rho_;   //!< interior density the increment scales with
};

//! the closure as a registered fixer (A.rst 5.2): kind fixer, rows {IVY, IVZ},
//! moves_ke_to_ie, AAM change -w2 dt R; partial sums {R, I} FP64 on device.
//! X's global phase reduces them via host (never captured; X.rst:98, 187)
class AxialClosureFixer /* : public Fixer (assumed) */ {
 public:
  explicit AxialClosureFixer(AxialMomentum const* aam);  //!< non-owning
  SourceDeclaration const& declaration() const;
  torch::Tensor partial_sums() const;  //!< into the stage's one collective
  //! apply -w2 dt R / I rho s^2 along K from the reduced sums, on device;
  //! I <= 0 or non-finite R leaves u unchanged (torch::where)
  void apply(torch::Tensor u, torch::Tensor const& global_sums, double dt,
             double stage_weight) const;
 private:
  AxialMomentum const* aam_;
};

}  // namespace snap::v3
