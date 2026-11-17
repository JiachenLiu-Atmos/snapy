// V3 lane A skeleton: AAM as a register consumer, its torque rule and the
// closure fixer (design A.rst, sections 3-5). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap::v3 {

class Fixer;  // assumed, lane X / register owner (Q X-2)

//! dynamics/conserve-axial-momentum (A.rst 4.2). "full" and the
//! conserve-axial-momentum-closure key are refused with a message naming exact
enum class AxialMode : uint8_t { off, transport, exact };

//! register views AAM reads (A.rst 3.2); undefined = direction inactive
struct AxialFluxView {
  torch::Tensor flux[3];     //!< A-r1: total face flux, all rows
  torch::Tensor central[3];  //!< A-r2: part riemann.central, x2/x3 only
  torch::Tensor x1_transfer;           //!< A-r3 + A-r4: mass per time
  torch::Tensor x1_tangential_corr;    //!< A-r5: implicit correction
  int ny = 0;                          //!< species rows after IDN
};

//! flux-form torque of F_d per volume. Rule 1: #363's -(Omega/V) sum_f M_d,f
//! (s_f^2 - s_c^2); rule 2: the side study's, if it passes (A.rst 4.3)
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
  //! (until G5, A.rst 7) and nu_iso == 0 (until phase 3, A.rst 9.4)
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

//! the closure as a registered fixer (A.rst 5). Declares to the monitor:
//! kind fixer, rows {IVY, IVZ}, moves_ke_to_ie, AAM change -w2 dt R,
//! reduction {R, I} (FP64, global, every stage). Sits at the stage's last
//! momentum change, before boundaries; no host sync
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
