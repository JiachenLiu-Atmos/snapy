// V3 lane A skeleton: one Coriolis/metric source (A.rst 3.4) and diffusion
// as a register contributor (A.rst 3.7). Declarations only.
#pragma once

#include "source_term.hpp"

namespace snap {

class AxialMomentum;  // axial_momentum.hpp

//! H3: m1 = phi_lo lo + phi_hi hi, phi = A x mass rows of total(1) (R:20-21);
//! built under #355's gate; held by Hydro (R-7), read by Coriolis and S (S-8)
class X1Momentum {
 public:
  torch::Tensor m1(FluxRegister const& reg) const;  //!< [nk, nj, ni]
 private:
  torch::Tensor lo_, hi_;  //!< lane A (S-1); hold 1/(2rV), no /dt
};

//! -2 Omega x m, m = (m1, rho v2, rho v3); -m1 v_h/r stays in coord (J1).
//! Twice: pre_implicit (m1 = 0), post_implicit (m1 rows from X1Momentum, R:21)
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
  torch::Tensor omega_local_;  //!< lane A (S-1); centre frame from S (S-6)
};

//! shared by both evaluators; x1 profiles frozen after build (lane O schema)
struct DiffusionCoefficients {
  double nu_iso = 0.0, kappa_iso = 0.0;
  bool dynamic = false;
  torch::Tensor nu_scale, kappa_scale;  //!< per x1 cell, or undefined
  double temperature_inner = 0.0, temperature_outer = 0.0;
  bool no_slip = false;
};

//! phase 3 (A.rst 3.7): posts the "diffusion" energy flux with reg.add
//! (R:57-58); momentum stays Cartesian-channel until gate G9b (H9)
class DiffusionFlux {
 public:
  explicit DiffusionFlux(DiffusionCoefficients c);
  void contribute(FluxRegister& reg, int dir, StageState const& s) const;
  double max_time_step(StageState const& s) const;
 private:
  DiffusionCoefficients c_;
  //! two-point (diffusion.cpp) and cubed-sphere 4-point evaluators; traction
  //! goes to coord's flux2global<d>_ frame (S-2, S-3)
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

//! Strang-split radial conduction (A.rst 3.7): post_split(1, "diffusion.radial",
//! flux, 0.5, dt) (R:80-81); failure aborts before the increment, as at the pin
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
};

//! merged modules (A.rst 3.6), deck keys unchanged; BottomSlab: relax-bot-*
class Sponge; class LayerHeat; class BottomSlab;

}  // namespace snap
