#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

#include <vector>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/implicit/implicit_hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>
#include <snap/sedimentation/sedimentation.hpp>
#include <snap/utils/fp64_store.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;
struct WbRef4Stencils;
class HydroReferenceX1Operator;
struct X1PlainMeanStencils;
struct X1PressureSourceStencils;

struct HydroOptionsImpl {
  static std::shared_ptr<HydroOptionsImpl> create() {
    return std::make_shared<HydroOptionsImpl>();
  }
  static std::shared_ptr<HydroOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  HydroOptionsImpl() = default;
  std::shared_ptr<HydroOptionsImpl> clone() const;
  void report(std::ostream& os) const {
    os << "-- hydro options --\n";
    os << "* verbose = " << verbose() << "\n"
       << "* disable_flux_x1 = " << disable_flux_x1() << "\n"
       << "* disable_flux_x2 = " << disable_flux_x2() << "\n"
       << "* disable_flux_x3 = " << disable_flux_x3() << "\n"
       << "* flux_covariance = " << flux_covariance() << "\n"
       << "* x1_mass_covariance = " << x1_mass_covariance() << "\n";
  }

  //! verbose
  ADD_ARG(bool, verbose) = false;

  ADD_ARG(bool, disable_flux_x1) = false;
  ADD_ARG(bool, disable_flux_x2) = false;
  ADD_ARG(bool, disable_flux_x3) = false;

  //! Keep the well-balanced x1 reference's stencils off the wall ghosts
  ADD_ARG(bool, wb_wall_clamp) = true;

  //! order of the x1 cell-average <-> point conversion of the primitives: 2
  //! (W of the averages) or 4 (hydro_forward.cpp); x2/x3 stay second order
  ADD_ARG(int, conversion_order) = 2;
  //! store hydro_u rows IDN and IPR as anomalies about a static fp64
  //! reference (fp32 roadmap, Step 1). YAML `dynamics: {anomaly: true}`,
  //! overridden by SNAP_ANOMALY=0|1. Off: every path is unchanged.
  ADD_ARG(bool, anomaly) = false;

  //! dynamics/flux-covariance: the O(dx1^2) x1 covariance term of the x2/x3
  //! face fluxes (#289). A scheme choice: every block of a run must make the
  //! same one, or the shared faces stop being single-valued; blocks built from
  //! one deck do. Off restores the plain face product.
  ADD_ARG(bool, flux_covariance) = true;

  //! dynamics/x1-mass-covariance: subtract the x1 rho-w covariance
  //! dz^2/12 rho_1 w_1 / rho from the reconstructed cell velocity; needs at
  //! least 3 x1 cells per block. Same every-block rule as flux_covariance.
  ADD_ARG(bool, x1_mass_covariance) = false;

  //! forcing options
  ADD_ARG(ConstGravityOptions, grav) = nullptr;
  ADD_ARG(CoriolisOptions, coriolis) = nullptr;
  ADD_ARG(DiffusionOptions, diffusion) = nullptr;
  ADD_ARG(BodyHeatOptions, bodyHeat) = nullptr;
  ADD_ARG(BotHeatOptions, botHeat) = nullptr;
  ADD_ARG(TopCoolOptions, topCool) = nullptr;
  ADD_ARG(RelaxBotCompOptions, relaxBotComp) = nullptr;
  ADD_ARG(RelaxBotTempOptions, relaxBotTemp) = nullptr;
  ADD_ARG(RelaxBotVeloOptions, relaxBotVelo) = nullptr;
  ADD_ARG(TopSpongeLyrOptions, topSpongeLyr) = nullptr;
  ADD_ARG(BotSpongeLyrOptions, botSpongeLyr) = nullptr;
  ADD_ARG(PlumeForcingOptions, plumeForcing) = nullptr;

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos) = nullptr;

  ADD_ARG(ReconstructOptions, recon1) = nullptr;
  ADD_ARG(ReconstructOptions, recon23) = nullptr;
  ADD_ARG(RiemannSolverOptions, riemann) = nullptr;

  ADD_ARG(ImplicitOptions, icorr) = nullptr;
  ADD_ARG(SedHydroOptions, sed) = nullptr;
};

using HydroOptions = std::shared_ptr<HydroOptionsImpl>;
using Variables = std::map<std::string, torch::Tensor>;

class HydroImpl : public torch::nn::Cloneable<HydroImpl> {
 public:
  //! \brief Create and register a `Hydro` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Hydro` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Hydro` module
   */
  static std::shared_ptr<HydroImpl> create(HydroOptions const& opts,
                                           torch::nn::Module* p,
                                           std::string const& name = "hydro");

  //! options with which this `Hydro` was constructed
  HydroOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const* pmb = nullptr;

  //! owning submodules
  EquationOfState peos = nullptr;
  RiemannSolver priemann = nullptr;

  Reconstruct precon1 = nullptr;
  Reconstruct precon23 = nullptr;

  ImplicitHydro picorr = nullptr;

  SedHydro psed = nullptr;

  //! forcings
  std::vector<torch::nn::AnyModule> forcings;
  Diffusion pdiffusion = nullptr;

  //! Constructor to initialize the layers
  HydroImpl() = default;
  explicit HydroImpl(const HydroOptions& options_,
                     torch::nn::Module* p = nullptr);
  void reset() override;

  double max_time_step(torch::Tensor hydro_w,
                       torch::Tensor solid = torch::Tensor()) const;

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor hydro_u,
                        Variables const& other);

  torch::Tensor flux1() const { return _flux1; }
  torch::Tensor flux2() const { return _flux2; }
  torch::Tensor flux3() const { return _flux3; }

  //! dynamics/flux-covariance (#289): the face-flux covariance rows a passive
  //! scalar carries on this stage's x2 (dim 2) or x3 (dim 1) faces, the rows a
  //! hydro tracer with the same mass fraction gets in _flux_covariance, built
  //! from the same face states
  /*!
   * \param[in] rl,rr  the scalar's reconstructed ratio to the DRY density on
   *                   each side of the face, (nscalar, nc3, nc2, nc1)
   * \param[in] dim    the solver's direction tag: 2 for x2, 1 for x3
   * \return           the scalar flux increments, shaped like rl; undefined
   *                   when this stage's faces carried no covariance
   */
  torch::Tensor scalar_flux_covariance(torch::Tensor const& rl,
                                       torch::Tensor const& rr, int dim) const;
  //! the dry-mass row of this stage's covariance increment on the x2 (dim 2)
  //! or x3 (dim 1) faces, which flux2()/flux3()[IDN] carry; undefined when
  //! those faces carried none
  torch::Tensor flux_covariance_dry(int dim) const {
    return _cov_dry[dim == 2 ? 0 : 1];
  }
  torch::Tensor face_pressure1() const { return _face_pressure1; }
  torch::Tensor implicit_mass_correction() const;

  //! dry-density increment the forcings added this stage (du[IDN])
  torch::Tensor forcing_dry_increment() const { return _forcing_dry; }

  //! cumulative count of (cell, species) entries with positivity theta < 1
  //! (diagnostic; accumulated when the EOS limiter is enabled)
  torch::Tensor positivity_hits() const { return _positivity_hits; }
  //! how HARD the limiter bites, not just how often
  torch::Tensor positivity_severe() const { return _positivity_severe; }
  torch::Tensor positivity_min() const { return _positivity_min; }
  //! x1 face flux the limiter removes, and the total offered (lifetime sums)
  torch::Tensor lim_cut() const { return _lim_cut; }
  torch::Tensor lim_flux() const { return _lim_flux; }

  //! gravity-work: cell with gravity-work-fixer on (and grav1 != 0)
  bool gravity_work_fixer() const;
  //! x1 face f (0 inner, 1 outer) of this block is a physical, non-periodic
  //! boundary (a wall whatever installed it, or an open boundary)
  bool is_x1_wall(int f) const;
  //! gravity-work: face with an implicit scheme: the implicit
  //! operator books the face work of the mass it moves (#283)
  bool face_work_in_operator() const;
  //! add the O(dx1^2) covariance term to the x2/x3 face fluxes (#289)?
  //! dynamics/flux-covariance, on by default
  bool flux_covariance() const { return options->flux_covariance(); }
  //! subtract the x1 rho-w covariance dz^2/12 rho_1 w_1 / rho from the
  //! reconstructed cell velocity? dynamics/x1-mass-covariance, off by default
  bool x1_mass_covariance() const { return options->x1_mass_covariance(); }
  //! the x1 gravity work of the corrected potential energy is booked here
  //! (derivation curved-gravity-work-weight.md sec 7): gravity-work: face on a
  //! Cartesian, a spherical-polar or a gnomonic-equiangle grid, grav1 != 0.
  //! E+P is then the conserved invariant: gravity_work_defect() and E+PE_d
  //! checks measure the wrong one
  bool radial_exact_work() const;
  //! the x1 neighbour blocks {below, above} across a split, non-periodic x1
  //! column, -1 where there is none (as the W reference relay pairs them)
  std::pair<int, int> x1_neighbors() const;
  //! this block's E+PE defect of the dynamics in the current step (J),
  //! accumulated over the stages with their weight in the step
  torch::Tensor gravity_work_defect() const { return _gwfix_d; }
  //! energy the fixer has added in this run (J, global), accepted steps only;
  //! like the other run-to-date meters it is not carried across a restart
  torch::Tensor gravity_work_fix() const {
    return _gwfix_total + _gwfix_pending;
  }
  //! the current step's fixer energy: committed when the step is accepted
  //! (check_redo, or the next step's start), dropped when it is redone
  void add_gravity_work_fix(double e) { _gwfix_pending += e; }
  void commit_gravity_work_fix() {
    _gwfix_total += _gwfix_pending;
    _gwfix_pending.zero_();
  }
  void drop_gravity_work_fix() { _gwfix_pending.zero_(); }
  //! mass through this block's physical x1 boundary faces in the current
  //! step (kg, summed |flux|); the fixer needs it at round-off
  torch::Tensor gravity_work_wall_mass() const { return _gwfix_wall; }

  //! RK stage currently being advanced, published by
  //! MeshBlockImpl::advance_local. The vertical implicit correction needs
  //! it because that solve is nonlinear in dt (see hydro_forward.cpp).
  //! -1 means "not set". Reaching the correction in that state means running
  //! the FULL-dt operator -- the one this fix replaces -- so it warns; it does
  //! not abort, because tests/test_forcing.cpp drives HydroImpl::forward
  //! directly, outside the stage loop, and must keep working.
  int rk_stage = -1;

  //// ---------- anomaly storage (fp32 roadmap, Step 1) ---------- ////
  //! fp64 holder of the reference state, registered as "ref" when anomaly
  //! mode is on (null otherwise). Buffer "u0": rows {IDN, IPR} of the
  //! conserved reference, shape (2, n3|1, n2|1, nc1) including ghosts.
  Fp64Store pref = nullptr;

  //! anomaly storage is on for this block
  bool anomaly() const { return options->anomaly(); }
  //! the reference has been built (anomaly on and initialization done)
  bool has_reference() const { return pref && pref->has("u0"); }
  //! rows {IDN, IPR} of the conserved reference (undefined if none)
  torch::Tensor u_ref() const {
    return pref ? pref->get("u0") : torch::Tensor();
  }
  //! Initial species densities for thermodynamic/mixture-density perturbations.
  //! Undefined for legacy restarts that did not retain composition.
  torch::Tensor species_ref() const {
    auto r = pref ? pref->get("species0") : torch::Tensor();
    return r.defined() && r.dim() == 4 ? r : torch::Tensor();
  }
  //! `t` has the full block shape the reference covers (nvar, n3, n2, n1)
  bool is_full_block(torch::Tensor const& t) const {
    if (!has_reference() || t.dim() != 4 || t.size(0) <= IPR) return false;
    auto r = u_ref();
    return (r.size(1) == 1 || t.size(1) == r.size(1)) &&
           (r.size(2) == 1 || t.size(2) == r.size(2)) && t.size(3) == r.size(3);
  }
  //! rows {IDN, IPR} of the reference expanded to the block shape of `u`
  //! (nvar, n3, n2, n1), retaining FP64; checks that `u` is a full block
  torch::Tensor u_ref_like(torch::Tensor const& u) const;
  //! store the reference from a full-row conserved state `u_full`
  //! (velocities already zero). Keeps a broadcast column when `u_full` is
  //! bit-identical along x2 and x3.
  void set_reference(torch::Tensor const& u_full);
  //! u[{IDN, IPR}] += ref, in place, in u's dtype; no-op without a reference
  void add_reference_(torch::Tensor u) const;
  //! u[{IDN, IPR}] -= ref, in place; no-op without a reference
  void sub_reference_(torch::Tensor u) const;
  //! total dry density u[IDN] (+ ref) of a full-block stored state
  torch::Tensor dry_density(torch::Tensor const& u) const {
    return has_reference() ? u[IDN] + u_ref_like(u)[0] : u[IDN];
  }
  //! FP64 totals from the stored representation (new tensor; `u` if no ref)
  torch::Tensor total_u(torch::Tensor const& u) const;

 protected:
  void _revise_x1inner_ghost(torch::Tensor const& w);
  void _revise_x1outer_ghost(torch::Tensor const& w);

  void _revise_x1inner_lr(torch::Tensor const& wl, torch::Tensor const& wr);
  void _revise_x1outer_lr(torch::Tensor const& wl, torch::Tensor const& wt);

  //! O(dx1^2) covariance correction to one horizontal energy flux (#289)
  /*!
   * \param[in] wl,wr  reconstructed states at the face, BEFORE the Riemann
   *                   solver projects them into the face-local frame
   * \param[in] dim    the solver's direction tag: 2 for x2, 1 for x3
   * \return           the flux increment, shaped like the flux; in Cartesian
   *                   only its nonzero rows on the x1-interior cells, stacked
   *                   (IPR, ICY.., IDN); undefined when the x1 axis is too
   *                   thin to difference
   */
  torch::Tensor _flux_covariance(torch::Tensor const& wl,
                                 torch::Tensor const& wr, int dim) const;

  // Per-column hydrostatic references for the well-balanced x1
  // reconstruction: {psf_lo (face pressure), pref (cell pressure), dsf (face
  // density, (2, ...) = its ILT and IRT values), dref (cell density)},
  // rebuilt from the current field on every call.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  _hydro_ref_x1(torch::Tensor const& w) const;
  //! overwrite the m x1 ghost rows of f next to each x1 neighbour with the
  //! neighbour's own rows of the same cells (faces: the same faces past the
  //! seam face), so a split column reads the values of one block
  void _x1_ghost_rows(torch::Tensor f, int m, bool faces, int tag) const;
  torch::Tensor _apply_implicit_correction(torch::Tensor& du,
                                           torch::Tensor const& w, double dt,
                                           Variables const& other);

 private:
  //! Register all forcing modules
  std::vector<std::string> _register_forcings_module();

  mutable std::shared_ptr<HydroReferenceX1Operator> x1_reference_;

  void _check_stretched_x1(torch::Tensor const& u) const;
  //! conversion-order 4 face-density stencils for this block's x1 grid, built
  //! on first use
  mutable std::shared_ptr<WbRef4Stencils> wb_ref4_;
  //! the x1 reconstruction's optimal-linear rows (cp5) for the face density
  mutable std::shared_ptr<Center5InterpImpl> dsf_lin_;
  //! SNAP_X1_CENTROID_EXACT r^2 -> plain-mean stencils, built on first use
  mutable std::shared_ptr<X1PlainMeanStencils> x1pm_;
  //! and its pressure-force stencils, for the hydrostatic correction
  mutable std::shared_ptr<X1PressureSourceStencils> x1src_;
  //! conversion-order 4 x1 weights {kp, km}, built on first use, and the x1f
  //! (storage, version) they were built from
  mutable torch::Tensor conv4_x1_[2];
  mutable std::pair<void*, int64_t> conv4_x1_at_;
  mutable torch::Tensor wall4_E_;  // reflecting-wall ghost weights, key 4
  //! corrected-PE grid-only work factors, built on first use
  CorrectedPEFactors _pe_work;
  //! face_moment2_x1() for this block's x1 grid, built on first use. Holds
  //! x1f, so the key tensor cannot be freed and its address reused
  struct FaceMoment2 {
    torch::Tensor x1f, val;
    int64_t ver = -1;
  };
  mutable FaceMoment2 _s2;
  //! cell_volume() for this block's grid, built on first use by the tracer
  //! positivity limiter. Holds the three face tensors, so a key tensor
  //! cannot be freed and its address reused
  struct CellVolume {
    torch::Tensor f1, f2, f3, val;
    int64_t v1 = -1, v2 = -1, v3 = -1;
  };
  CellVolume _cell_vol;
  //! x1 mass-flux w (forward step 8): the grid-only factors (x1 face area and
  //! the split weights (r^2 - r_lo^2) / (2 r V), (r_hi^2 - r^2) / (2 r V) of
  //! the interior cells) and the stage buffers it reuses: phi (face transfer
  //! rates) and the Coriolis inputs wc1 = (rho, 0, v2, v3) before the
  //! implicit solve and wc2 = (1, m1, 0, 0) after it, whose row IVX is m1.
  //! Rebuilt when the x1 faces, the dtype, the device or the block shape
  //! change
  struct MassW {
    torch::Tensor x1f, area, lo, hi, phi, wc1, wc2;
    int64_t ver = -1;
  };
  MassW _mass_w;

  torch::Tensor _flux1, _flux2, _flux3, _face_pressure1, _div, _forcing_dry;
  torch::Tensor _positivity_hits, _positivity_severe, _positivity_min;
  //! conversion-order 4 cells that kept the second-order w (buffer
  //! "conversion_trips", registered only with the key at 4)
  torch::Tensor _conv4_trips;
  torch::Tensor _lim_cut, _lim_flux;
  //! this stage's x2 [0] / x3 [1] face states the covariance was built from
  //! and the dry-mass row it added (undefined when the switch is off)
  torch::Tensor _cov_wl[2], _cov_wr[2], _cov_dry[2];
  torch::Tensor _gwfix_d, _gwfix_total, _gwfix_pending, _gwfix_wall;
};

TORCH_MODULE(Hydro);

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3);
void check_eos(torch::Tensor w, int nghost);*/
}  // namespace snap

#undef ADD_ARG
