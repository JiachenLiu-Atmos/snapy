// V3 lane A skeleton: one Coriolis/metric source (A.rst 3.4) and diffusion
// as a register contributor (A.rst 3.7). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap {

class AxialMomentum;  // axial_momentum.hpp

//! -2 Omega x m, m = (m1, rho v2, rho v3); -m1 v_h/r stays in coord (J1). Twice:
//! pre_implicit (m1 = 0; through the scheme-9 solve), post_implicit (m1 rows,
//! R:21). m1 = phi_lo * lo + phi_hi * hi, phi = A x mass rows of total(1),
//! which holds the vic.* posts in post_implicit (R:20-21; R.rst:521-523)
class Coriolis : public SourceTerm {
 public:
  Coriolis(double omega1, double omega2, double omega3, bool traditional,
           RegisterPhase phase, AxialMomentum* aam /* non-owning, or null */);
  SourceDeclaration const& declaration() const override;
  void add(torch::Tensor du, StageState const& s) override;
 private:
  SourceDeclaration decl_;
  double omega_[3];
  bool traditional_;
  AxialMomentum* aam_;   //!< exact: adds the F_d torque in pre_implicit
  torch::Tensor omega_local_;  //!< lane A (S-1), from coord's rotation data
  torch::Tensor m1_lo_, m1_hi_;  //!< lane A (S-1); lo/hi hold 1/(2rV), no /dt
};

//! shared by both evaluators; x1 profiles frozen after build (lane O schema)
struct DiffusionCoefficients {
  double nu_iso = 0.0, kappa_iso = 0.0;
  bool dynamic = false;
  torch::Tensor nu_scale, kappa_scale;  //!< per x1 cell, or undefined
  double temperature_inner = 0.0, temperature_outer = 0.0;
  bool no_slip = false;
};

//! phase 3 (A.rst 3.7): posts "diffusion" (viscous momentum flux, energy flux
//! -k grad T . n - v_f . tau . n) with reg.add (R:57-58); no divergence of its
//! own. Its part is an FP64 monitor amount only (R.rst:452-454)
class DiffusionFlux {
 public:
  explicit DiffusionFlux(DiffusionCoefficients c);
  void contribute(FluxRegister& reg, int dir, StageState const& s) const;
  double max_time_step(StageState const& s) const;
 private:
  DiffusionCoefficients c_;
  //! two-point (diffusion.cpp) and cubed-sphere 4-point evaluators; traction
  //! goes to the panel face basis by coord's rotation (S-2, S-3)
  torch::Tensor cartesian_face_flux(int dir, StageState const& s) const;
  torch::Tensor cubed_sphere_face_flux(int dir, StageState const& s) const;
  void exchange_traces(StageState const& s) const;  //!< own exchange (S-3)
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

//! Strang-split radial conduction (diffusion_radial.cpp, kept); each half
//! step posts its x1 heat flux with reg.post_split(1, "diffusion.radial",
//! flux, 0.5, dt) (R:80-81), after begin_step (R:6, 48)
class RadialConduction : public SourceTerm {
 public:
  explicit RadialConduction(DiffusionCoefficients c);
  SourceDeclaration const& declaration() const override;  //!< split = true
  void add(torch::Tensor du, StageState const& s) override;  //!< no-op
  void begin_split(torch::Tensor u, double dt, FluxRegister& reg) override;
  void finish_split(torch::Tensor u, double dt, FluxRegister& reg) override;
  void restore_split(torch::Tensor u) const override;
 private:
  SourceDeclaration decl_;
  DiffusionCoefficients c_;
  torch::Tensor split_initial_;
  torch::Tensor failed_;  //!< reg.flag_retry("diffusion.radial", failed_) (R:78)
};

//! merged modules (A.rst 3.6), deck keys unchanged; BottomSlab: relax-bot-*
class Sponge; class LayerHeat; class BottomSlab;

}  // namespace snap
