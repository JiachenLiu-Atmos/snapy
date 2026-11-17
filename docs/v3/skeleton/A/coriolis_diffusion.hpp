// V3 lane A skeleton: one Coriolis/metric source (A.rst 6) and diffusion
// as a register contributor (A.rst 9). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap::v3 {

class FluxContributor;  // assumed, register owner (Q R-2)
class AxialMomentum;    // axial_momentum.hpp

//! -2 Omega x m, m = (m1, rho v2, rho v3), and -m1 v_h / r. Registered twice:
//! pre_implicit (m1 = 0; through the scheme-9 solve), post_implicit (m1 rows).
class Coriolis : public SourceTerm {
 public:
  Coriolis(double omega1, double omega2, double omega3, bool traditional,
           SourcePhase phase, AxialMomentum* aam /* non-owning, may be null */);
  SourceDeclaration const& declaration() const override;
  void add(torch::Tensor du, StageState const& s) override;
 private:
  SourceDeclaration decl_;
  double omega_[3];
  bool traditional_;
  AxialMomentum* aam_;   //!< exact: adds the F_d torque in pre_implicit
  torch::Tensor omega_local_;  //!< from RotationGeometry, built at reset
};

//! shared by both evaluators; x1 profiles frozen after build (Q O-1)
struct DiffusionCoefficients {
  double nu_iso = 0.0, kappa_iso = 0.0;
  bool dynamic = false;
  torch::Tensor nu_scale, kappa_scale;  //!< per x1 cell, or undefined
  double temperature_inner = 0.0, temperature_outer = 0.0;
  bool no_slip = false;
};

//! phase 3 (A.rst 9.4): viscous momentum flux and energy flux -k grad T . n
//! - v_f . tau . n into the register; no divergence of its own
class DiffusionFlux /* : public FluxContributor (assumed) */ {
 public:
  explicit DiffusionFlux(DiffusionCoefficients c);
  //! posts the named increment "diffusion" (after riemann, covariance)
  void contribute(FluxRegister& reg, int dir, StageState const& s) const;
  double max_time_step(StageState const& s) const;
 private:
  DiffusionCoefficients c_;
  //! two-point (diffusion.cpp) and cubed-sphere 4-point evaluators; the
  //! latter projects traction to the face basis (Q S-3), keeps trace exchange
  torch::Tensor cartesian_face_flux(int dir, StageState const& s) const;
  torch::Tensor cubed_sphere_face_flux(int dir, StageState const& s) const;
  void exchange_traces(StageState const& s) const;
};

//! phases 1-2: same evaluators, own divergence, budget declared (physical)
class DiffusionSource : public SourceTerm {
 public:
  explicit DiffusionSource(std::shared_ptr<DiffusionFlux const> flux);
  SourceDeclaration const& declaration() const override;
  void add(torch::Tensor du, StageState const& s) override;
  double max_time_step(StageState const& s) const override;
 private:
  SourceDeclaration decl_;
  std::shared_ptr<DiffusionFlux const> flux_;
};

//! Strang-split implicit radial conduction (diffusion_radial.cpp, kept);
//! reports its wall heat flux to the monitor as a boundary energy flux
class RadialConduction : public SourceTerm {
 public:
  explicit RadialConduction(DiffusionCoefficients c);
  SourceDeclaration const& declaration() const override;  //!< phase split
  void add(torch::Tensor du, StageState const& s) override;  //!< no-op
  void begin_split(torch::Tensor u, double dt) override;
  void finish_split(torch::Tensor u, double dt) override;
  void restore_split(torch::Tensor u) const override;
 private:
  SourceDeclaration decl_;
  DiffusionCoefficients c_;
  torch::Tensor split_initial_;
};

//! merged modules (A.rst 8.2), deck keys unchanged
class Sponge;      // top-/bot-sponge-lyr: Sponge{side}
class LayerHeat;   // top-cool / bot-heat: LayerHeat{side}
class BottomSlab;  // base of relax-bot-temp / -velo / -comp

}  // namespace snap::v3
