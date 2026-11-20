#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>
#include <cstdlib>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "gravity_work_radial.hpp"
#include "hydro.hpp"
#include "hydro_dispatch.hpp"

namespace snap {
HydroImpl::HydroImpl(const HydroOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void HydroImpl::reset() {
  _forcing_work = _forcing_cast = torch::Tensor();
  TORCH_CHECK(pmb, "[Hydro] Parent MeshBlock is null");
  TORCH_CHECK(
      options->conversion_order() == 2 || options->conversion_order() == 4,
      "[Hydro] dynamics/conversion-order must be 2 or 4, got ",
      options->conversion_order());
  TORCH_CHECK(options->precision() == "float64" || fp32(),
              "dynamics/precision must be float64 or float32");
  // conserve-axial-momentum: lane A's consumer makes the mode's refusals
  paam = std::make_shared<AxialMomentum>(this);
  TORCH_CHECK(!fp32() || options->perturbation_flux(),
              "dynamics/precision float32 requires perturbation-flux");

  //// ---- (1) set up equation-of-state model ---- ////
  peos = EquationOfStateImpl::create(options->eos(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "EOS type: " << peos->options->type() << "\n";
  }

  //// ---- (3) set up reconstruction-x1 model ---- ////
  precon1 = ReconstructImpl::create(options->recon1(), this, "recon1");
  if (pmb->pcoord) {
    auto const& co = pmb->pcoord->options;
    // SNAP_X1_CENTROID_EXACT hands this module plain means (hydro_forward.cpp)
    bool plain = x1_centroid_exact_enabled() && co->type() == "spherical-polar";
    auto x1f =
        !plain ? pmb->pcoord->x1_row_faces(options->recon1()->interp()->type())
        : co->stretched_x1() ? pmb->pcoord->x1f
                             : torch::Tensor();
    if (x1f.defined()) precon1->set_x1_faces(x1f);
  }
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x1 type: "
                 << precon1->pinterp1->options->type() << "\n";
  }

  //// ---- (4) set up reconstruction-x23 model ---- ////
  precon23 = ReconstructImpl::create(options->recon23(), this, "recon23");
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x2/x3 type: "
                 << precon23->pinterp1->options->type() << "\n";
  }

  //// ---- (5) set up riemann-solver model ---- ////
  priemann = RiemannSolverImpl::create(options->riemann(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "Riemann solver type: " << priemann->options->type()
                 << "\n";
  }

  // gravity-work options, checked here so YAML and Python construction agree
  if (options->grav()) {
    auto const& gw = options->grav()->gravity_work();
    // V3 I-7 (rulings (a), (b)): the cell form is removed; face is the one
    // gravity-work form
    TORCH_CHECK(gw == "face", "const-gravity gravity-work: '", gw,
                "' is not supported: the cell form was removed (V3 I-7); use "
                "gravity-work: face");
    // the global gravity-work fixer is removed (V3 X5, gw1; Cheng)
    TORCH_CHECK(!options->grav()->gravity_work_fixer(),
                "const-gravity gravity-work-fixer was removed in V3; "
                "set it false or drop the key");
    // the corrected PE needs a single-valued potential; -grav1 * x1 jumps at
    // a periodic x1 face: warn, no change. The layout's x1 wrap is read from
    // its topology: a Python bfuncs() call clears the names.
    auto const& nm = pmb->options->bcnames();
    bool wrap = pmb->options->layout()->periodic_z();
    if (options->grav()->grav1() != 0. && gw == "face") {
      bool periodic_x1 = wrap;
      for (int f : {0, 1})
        periodic_x1 = periodic_x1 || (f < static_cast<int>(nm.size()) &&
                                      nm[f].compare(0, 8, "periodic") == 0);
      if (periodic_x1)
        TORCH_WARN_ONCE(
            "gravity-work: face with periodic x1: -grav1*x1 jumps at the "
            "periodic face, so the corrected PE its work books is not defined "
            "there; use x1 walls");
    }
    // the corrected-PE work has no form on other grids: plain face work there
    if (options->grav()->grav1() != 0. && gw == "face" && pmb && pmb->pcoord) {
      auto const& type = pmb->pcoord->options->type();
      if (x1_measure(type) == X1Measure::none)
        TORCH_WARN_ONCE(
            "the corrected-PE work of gravity-work: face has no form on a '",
            type, "' grid: it books the plain face work there");
    }
    // the conversion-order 4 resolution flag (wb_ref4.cpp) reads scan pressures
    // three cells away, so x1 seam sides agree only with nghost >= 3;
    // SNAP_X1_CENTROID_EXACT keeps the requirement it had while it switched
    // that flag on
    if ((x1_centroid_exact_enabled() || options->conversion_order() == 4) &&
        options->grav()->grav1() != 0. && pmb && pmb->pcoord) {
      int ng = pmb->pcoord->options->nghost();
      TORCH_CHECK(ng >= 3,
                  "SNAP_X1_CENTROID_EXACT (or dynamics/conversion-order 4) "
                  "needs nghost >= 3, got ",
                  ng);
    }
    // the one-sided rho_1 at a wall spans three cells: fewer reads the ghost
    if (x1_mass_covariance() && options->grav()->grav1() != 0. && pmb &&
        pmb->pcoord) {
      int nx1 = pmb->pcoord->options->nx1();
      TORCH_CHECK(nx1 >= 3,
                  "dynamics/x1-mass-covariance needs at least 3 x1 cells "
                  "per block, got ",
                  nx1, "; set it to false");
    }
  }

  //// ---- (5b) anomaly storage (fp32 roadmap, Step 1) ---- ////
  if (options->anomaly()) {
    TORCH_CHECK(options->eos()->type() == "ideal-moist",
                "dynamics: anomaly is implemented for the ideal-moist "
                "equation of state only, got '",
                options->eos()->type(), "'; unset it (or SNAP_ANOMALY)");
    // curved grids interpolate ghosts (cubed-sphere seams): the reference's
    // ghosts would not match the interpolated anomaly's, so on/off would no
    // longer agree to round-off. Step 1 is cartesian only.
    TORCH_CHECK(!pmb->pcoord || pmb->pcoord->options->type() == "cartesian",
                "dynamics: anomaly supports cartesian grids only in Step 1, "
                "got '",
                pmb->pcoord ? pmb->pcoord->options->type() : std::string(),
                "'");
    pref = register_module("ref", Fp64Store());
  }

  if (options->perturbation_flux()) {
    TORCH_CHECK(options->anomaly() && options->riemann()->type() == "hllc",
                "dynamics/perturbation-flux requires anomaly storage and HLLC");
  }

  //// ---- (6) set up implicit solver ---- ////
  if (options->icorr()) {
    picorr = ImplicitHydroImpl::create(options->icorr(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Implicit correction type: " << picorr->options->type()
                   << "\n";
    }
  }

  //// ---- (7) set up sedimentation ---- ////
  if (options->sed() != nullptr) {
    psed = SedHydroImpl::create(options->sed(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Sedimentation particle ids: "
                   << fmt::format("{}", psed->options->sedvel()->particle_ids())
                   << "\n";
    }
  }

  //// ---- (8) set up forcings ---- ////
  auto forcing_names = _register_forcings_module();
  if (options->verbose()) {
    SINFO(Hydro) << "Forcings: " << fmt::format("{}", forcing_names) << "\n";
  }

  //// ---- (10) populate buffers ---- ////
  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nvar = peos->nvar();

  // the flux register (R-1): F1/F2/F3 are its stage totals T_d, so every
  // producer still writes them in place; the monitor's tensors are buffers
  // too, so module moves and clones reach them
  RegisterRequest req;
  // EF's names (EF.rst section 4, R.rst 4.5): EfStatus's slots (EF-2) move
  // to the monitor in R-1's second part; nothing writes these yet
  req.counters = {"ef.floor_patch",    "ef.nan",      "ef.vapor_fail",
                  "ef.heat_capacity",  "ef.pos_hits", "ef.pos_severe",
                  "ef.scalar_pos_hits"};
  // C's FaceFloor counts (C.rst 3.7; names adopted in R.rst 2.6): faces a
  // floor changed, which register counts, not flux
  for (auto name : {"face_floor.rho", "face_floor.p", "face_floor.species",
                    "riemann.pstar_clamp", "riemann.llf"})
    req.counters.push_back(name);
  // V3 I-6: the retried columns (R.rst 4.5), last, so the counters above
  // keep their indices
  req.counters.push_back("implicit.fp64_columns");
  req.minima = {"ef.pos_min"};
  // V3 I-5: lane I's cause (R.rst 4.5); X maps it to its cause bit 5
  req.retry = {"ef.floor_patch", "ef.nan", "ef.vapor_fail", "ef.heat_capacity",
               "implicit.column"};
  // conserve-axial-momentum exact: A's riemann.central on the x2/x3 faces
  // (R-2), kept until the next stage
  paam->request(req);
  // R-5: lane I's kept x1 parts, posted by the implicit solve (I-4)
  if (picorr && nc1 > 1)
    for (auto const* key :
         {"vic.mass_total@1", "vic.mass_dry@1", "vic.momentum_t@1"})
      req.keep.push_back(key);
  // R-5: the scalars' dry-mass flux, row IDN of each total as limit ends
  // (R.rst 4.2 EF c); hydro requests it, the Scalar module does not exist yet
  auto sopts = pmb->options->scalar();
  if (sopts && sopts->nvar() > 0) {
    std::array<int, 3> nc = {nc1, nc2, nc3};
    for (int d = 1; d <= 3; ++d)
      if (nc[d - 1] > 1)
        req.keep.push_back("mass.dry.limited@" + std::to_string(d));
  }
  // R-4's coordinate reset: replace and apply_prescribed (S step (4)) book
  // with the face areas A_f; kFlux is float32 under precision float32, else
  // float64, as before
  freg.reset(pmb->pcoord.get(), PrecisionPolicy(options->precision()), nvar,
             req);
  _flux1 = register_buffer("F1", freg.totals()[0]);
  _flux2 = register_buffer("F2", freg.totals()[1]);
  _flux3 = register_buffer("F3", freg.totals()[2]);
  register_buffer("monitor_counters", freg.monitor().counters);
  register_buffer("monitor_minima", freg.monitor().minima);
  register_buffer("monitor_retry", freg.monitor().retry_flags);
  for (auto& [key, part] : freg.parts()) {  // "riemann.central@2" -> K_..._2
    std::string name = "K_" + key;
    std::replace(name.begin(), name.end(), '.', '_');
    std::replace(name.begin(), name.end(), '@', '_');
    register_buffer(name, part);
  }
  _face_pressure1 = register_buffer(
      "P1", nc1 > 1 ? torch::zeros({nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());

  _div = register_buffer("D",
                         torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));

  // the positivity meters live in the block's EfStatus; the buffers are the
  // same tensors, so they keep their names and move with Module::to
  auto& status = *pmb->pefs;
  register_buffer(
      "positivity_hits",
      status.bind(EfStatus::kPosHits, torch::zeros({1}, torch::kInt64)));
  register_buffer(
      "positivity_severe",
      status.bind(EfStatus::kPosSevere, torch::zeros({1}, torch::kInt64)));
  register_buffer(
      "positivity_min",
      status.bind(EfStatus::kPosMin, torch::ones({1}, torch::kFloat64)));
  if (options->conversion_order() == 4)
    _conv4_trips =
        register_buffer("conversion_trips", torch::zeros({1}, torch::kInt64));
  register_buffer("lim_cut", status.bind(EfStatus::kLimCut,
                                         torch::zeros({1}, torch::kFloat64)));
  register_buffer("lim_flux", status.bind(EfStatus::kLimFlux,
                                          torch::zeros({1}, torch::kFloat64)));
}

torch::Tensor HydroImpl::positivity_hits() const {
  return pmb->pefs->slot(EfStatus::kPosHits);
}
torch::Tensor HydroImpl::positivity_severe() const {
  return pmb->pefs->slot(EfStatus::kPosSevere);
}
torch::Tensor HydroImpl::positivity_min() const {
  return pmb->pefs->slot(EfStatus::kPosMin);
}
torch::Tensor HydroImpl::lim_cut() const {
  return pmb->pefs->slot(EfStatus::kLimCut);
}
torch::Tensor HydroImpl::lim_flux() const {
  return pmb->pefs->slot(EfStatus::kLimFlux);
}

bool HydroImpl::is_x1_wall(int f) const {
  // a boundary function that is not a periodic copy (S step (2): FaceKind)
  auto k = Exchange(pmb).kind(f);
  return k == FaceKind::wall || k == FaceKind::outflow || k == FaceKind::custom;
}

void HydroImpl::_check_stretched_x1(torch::Tensor const& u) const {
  auto pcoord = pmb->pcoord;
  if (!pcoord || !pcoord->options->stretched_x1()) return;
  TORCH_CHECK(u.device().type() != torch::kMPS,
              "a stretched x1 (geometry/x1f or cells/x1rat) is not supported "
              "on MPS (#280); run it on CPU or CUDA");
  auto const& bf = pmb->options->bfuncs();
  bool wrap = pmb->options->layout() && pmb->options->layout()->periodic_z();
  for (int f : {0, 1}) {
    bool phys = pmb->options->is_physical_boundary(0, 0, f == 0 ? -1 : 1);
    TORCH_CHECK(
        !wrap && (!phys || is_reflecting(bf[f])),
        "a stretched x1 (geometry/x1f or cells/x1rat) needs reflecting ",
        "x1 boundaries (#280): its ghost faces mirror the interior ",
        "widths; the x1-", f == 0 ? "inner" : "outer",
        " boundary is not reflecting");
  }
}

bool HydroImpl::radial_exact_work() const {
  auto g = options->grav();
  auto const& type = pmb->pcoord->options->type();
  return g && g->grav1() != 0. && g->gravity_work() == "face" &&
         x1_measure(type) != X1Measure::none;
}

std::pair<int, int> HydroImpl::x1_neighbors() const {
  return Exchange(pmb).x1_neighbors();
}

double HydroImpl::max_time_step(torch::Tensor w, torch::Tensor solid) const {
  auto sub3 = pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));

  torch::Tensor cs;
  if (options->eos()->type() == "aneos") {
    cs = peos->compute("W->L", {w});
  } else {
    auto gamma = peos->compute("W->A", {w});
    cs = peos->compute("WA->L", {w, gamma});
  }

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  auto dt_min = torch::tensor({1.e9, 1.e9, 1.e9},
                              torch::dtype(w.dtype()).device(w.device()));

  auto icorr = options->icorr();

  if (icorr) {
    auto adv = pmb->pintg->options->cfl() / icorr->advection_cfl();
    if (cs.size(2) > 1) {
      auto denom1 =
          (!(icorr->scheme() & 1) || (cs.size(0) == 1 && cs.size(1) == 1))
              ? (w[IVX].abs() + cs)
              : w[IVX].abs() * adv;
      dt_min[0] = (pmb->pcoord->center_width1() / denom1).index(sub3).min();
    }

    if (cs.size(1) > 1) {
      auto denom2 = (!((icorr->scheme() >> 1) & 1)) ? (w[IVY].abs() + cs)
                                                    : w[IVY].abs() * adv;
      dt_min[1] = (pmb->pcoord->center_width2() / denom2).index(sub3).min();
    }

    if (cs.size(0) > 1) {
      auto denom3 = (!((icorr->scheme() >> 2) & 1)) ? (w[IVZ].abs() + cs)
                                                    : w[IVZ].abs() * adv;
      dt_min[2] = (pmb->pcoord->center_width3() / denom3).index(sub3).min();
    }

    // A horizontal wind that jumps by more than cs across an x1 face
    if (icorr->shear_cfl() > 0. && cs.size(2) > 1) {
      auto n1 = cs.size(2);
      auto csf = 0.5 * (cs.slice(2, 0, n1 - 1) + cs.slice(2, 1, n1));
      auto scale = icorr->shear_cfl() / pmb->pintg->options->cfl();
      auto face_bound = [&](int iv, torch::Tensor width) {
        auto lo = w[iv].slice(2, 0, n1 - 1);
        auto hi = w[iv].slice(2, 1, n1);
        auto prod = (lo.abs() * hi.abs()).clamp_min(1.e-30);
        auto dts = scale * csf * width.slice(2, 0, n1 - 1) / prod;
        return torch::where((hi - lo).abs() >= csf, dts, 1.e9)
            .index(sub3)
            .min();
      };
      if (cs.size(1) > 1) {
        dt_min[0] = torch::minimum(
            dt_min[0], face_bound(IVY, pmb->pcoord->center_width2()));
      }
      if (cs.size(0) > 1) {
        dt_min[0] = torch::minimum(
            dt_min[0], face_bound(IVZ, pmb->pcoord->center_width3()));
      }
    }
  } else {
    if (cs.size(2) > 1) {
      dt_min[0] = (pmb->pcoord->center_width1() / (w[IVX].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(1) > 1) {
      dt_min[1] = (pmb->pcoord->center_width2() / (w[IVY].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(0) > 1) {
      dt_min[2] = (pmb->pcoord->center_width3() / (w[IVZ].abs() + cs))
                      .index(sub3)
                      .min();
    }
  }

  double dt = dt_min.min().item<double>();
  dt = std::min(dt, sources.max_time_step({w}));
  return dt;
}

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

torch::Tensor HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                                    torch::Tensor const& w,
                                                    double dt,
                                                    Variables const& other,
                                                    double post_dt) {
  if (!picorr) return torch::Tensor();
  // V3 I-4: the solve posts its face transfers into total(1), which the
  // Riemann slot rewrites each stage; with the x1 flux off nothing does, so
  // nothing is posted and m1 keeps the solve's own transfer
  FluxRegister* reg = options->disable_flux_x1() ? nullptr : &freg;

  // Implicit x1 solve has no cross-rank coupling, so nb1 > 1 would silently
  // solve each rank's own sub-column. Full column <=> both x1 faces physical.
  TORCH_CHECK(pmb->options->is_physical_boundary(0, 0, -1) &&
                  pmb->options->is_physical_boundary(0, 0, 1),
              "[Hydro] implicit scheme requires nb1 = 1 (no x1 decomposition): "
              "the vertical solve has no cross-rank coupling.");

  torch::Tensor wi;
  if (other.count("solid")) {
    wi = torch::where(other.at("solid").unsqueeze(0).expand_as(w),
                      other.at("fill_solid_hydro_w"), w);
    du.masked_fill_(other.at("solid").unsqueeze(0).expand_as(du), 0.0);
  } else {
    wi = w;
  }

  auto destination = du;
  bool solve_float = options->icorr()->precision() == "float32" ||
                     (options->icorr()->precision() == "auto" && fp32());
  TORCH_CHECK(
      !solve_float || (options->eos()->type() == "ideal-moist" &&
                       pmb->pcoord->options->type() == "cartesian"),
      "FP32 vertical implicit correction requires Cartesian ideal-moist");
  bool boundary = fp32() || solve_float;
  if (boundary) du = du.to(torch::kFloat64);
  du[IPR].sub_(peos->internal_energy_offset(du));
  auto rhs64 = solve_float ? du.clone() : torch::Tensor();
  auto wi64 = wi;
  if (solve_float) {
    du = du.to(torch::kFloat32);
    wi = wi.to(torch::kFloat32);
  }
  auto gamma_for = [&](torch::Tensor const& state) {
    if (options->eos()->type() == "aneos") {
      auto cs = peos->compute("W->L", {w});
      return peos->compute("WL->A", {w, cs});
    }
    return peos->compute("W->A", {state});
  };
  auto solid = other.count("solid") ? other.at("solid") : torch::Tensor();
  auto correction =
      picorr->forward_masked(du, wi, gamma_for(wi), dt, solid, reg, post_dt);
  // V3 I-6: the predicated FP64 stage, always run with the FP32 solve and
  // its fallback: only the columns whose FP32 solve failed are solved again,
  // from rhs64 and wi64, and no host read decides it; the other columns keep
  // their FP32 solution
  torch::Tensor retry;
  if (solve_float && options->icorr()->fallback_fp64()) {
    retry = picorr->stage_failed();
    picorr->forward_masked(rhs64, wi64, gamma_for(wi64), dt, solid, reg,
                           post_dt, retry);
    correction = picorr->merge_fp64(retry);
    picorr->count_fp64_columns(retry);
    freg.monitor().count("implicit.fp64_columns", retry);
  }
  // V3 I-5: the column status OR after the whole-block FP64 fallback (until
  // I-6 retries per column); slot 5 of the redo still reads
  // solve_failed_tensor(), the same OR, until lane X reads retry_flags
  freg.monitor().flag_retry("implicit.column", picorr->status());
  // Restore energy offsets in FP64, then write the increment to state dtype.
  if (boundary) {
    du = du.to(torch::kFloat64);
    correction = correction.to(torch::kFloat64);
  }
  // the retried columns' FP64 solution (rhs64 now holds it), in place of
  // the FP32 stage's (I-6)
  if (retry.defined())
    du = torch::where(retry.unsqueeze(0).unsqueeze(-1), rhs64, du);
  du[IPR].add_(peos->internal_energy_offset(du));
  correction[IPR].add_(peos->internal_energy_offset(correction));
  if (boundary) {
    destination.copy_(du);
    du = destination;
    return correction.to(du.options());
  }
  return correction;
}

void HydroImpl::_revise_x1inner_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int is = pmb->pcoord->il();
  wl[IPR].narrow(-1, is, 1) = wr[IPR].narrow(-1, is, 1);
  wl[IDN].narrow(-1, is, 1) = wr[IDN].narrow(-1, is, 1);
}

void HydroImpl::_revise_x1outer_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int ie = pmb->pcoord->iu();
  wr[IPR].narrow(-1, ie + 1, 1) = wl[IPR].narrow(-1, ie + 1, 1);
  wr[IDN].narrow(-1, ie + 1, 1) = wl[IDN].narrow(-1, ie + 1, 1);
}

ReferenceFaces HydroImpl::_hydro_ref_x1(torch::Tensor const& w,
                                        torch::Tensor const& dw) const {
  auto pcoord = pmb->pcoord;
  X1ReferenceConfig cfg;
  cfg.coord = pcoord.get();
  cfg.policy = &pmb->policy;
  cfg.phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  cfg.phys_out = pmb->options->is_physical_boundary(0, 0, 1);
  cfg.wall_clamp = options->wb_wall_clamp();
  auto const& bf = pmb->options->bfuncs();
  for (int f : {0, 1})
    cfg.reflecting[f] = is_x1_wall(f) && is_reflecting(bf[f]);
  cfg.conversion_order = options->conversion_order();
  cfg.interp = options->recon1()->interp()->type();
  cfg.interp_x1_coeffs = precon1->pinterp1->has_x1_coeffs();
  ExchangeX1Link link(pmb);
  // downward magnitude (>0)
  return x1_reference_.evaluate(cfg, {w, dw}, -options->grav()->grav1(), &link);
}

std::shared_ptr<HydroImpl> HydroImpl::create(HydroOptions const& opts,
                                             torch::nn::Module* p,
                                             std::string const& name) {
  TORCH_CHECK(p, "[Hydro] Parent module is null");
  TORCH_CHECK(opts, "[Hydro] Options pointer is null");

  return p->register_module(name, Hydro(opts, p));
}

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3) {
  auto interior =
      get_interior(wlr.sizes(), nghost, extend_x1, extend_x2, extend_x3);

  int dim = extend_x1 == 1 ? 1 : (extend_x2 == 1 ? 2 : 3);
  TORCH_CHECK(wlr.index(interior).select(1, IDN).min().item<double>() > 0.,
              "Negative density detected after reconstruction in dimension ",
              dim);
  TORCH_CHECK(wlr.index(interior).select(1, IPR).min().item<double>() > 0.,
              "Negative pressure detected after reconstruction in dimension ",
              dim);
}

void check_eos(torch::Tensor w, int nghost) {
  auto interior = get_interior(w.sizes(), nghost);
  TORCH_CHECK(w.index(interior)[IDN].min().item<double>() > 0.,
              "Negative density detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number;",
              " 2) Activate EOS limiter and set the density floor");
  TORCH_CHECK(w.index(interior)[IPR].min().item<double>() > 0.,
              "Negative pressure detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number; ",
              " 2) Activate EOS limiter and set the pressure floor");
}*/

void HydroImpl::set_reference(torch::Tensor const& u_full) {
  TORCH_CHECK(pref, "Hydro::set_reference: anomaly mode is off");
  TORCH_CHECK(u_full.dim() == 4 && u_full.size(0) > IPR,
              "Hydro::set_reference: expected (nvar, n3, n2, n1), got ",
              u_full.sizes());
  auto idx =
      torch::tensor({static_cast<int64_t>(IDN), static_cast<int64_t>(IPR)},
                    torch::dtype(torch::kLong).device(u_full.device()));
  auto r = u_full.index_select(0, idx).to(torch::kFloat64);
  // a horizontally uniform reference is kept as one broadcast column; the
  // test is bit equality, so a column is used only when it changes nothing
  auto col = r.narrow(1, 0, 1).narrow(2, 0, 1);
  if (torch::equal(r, col.expand_as(r))) r = col;
  pref->set("u0", r);
  auto species =
      u_full.narrow(0, ICY, u_full.size(0) - ICY).to(torch::kFloat64);
  auto species_col = species.narrow(1, 0, 1).narrow(2, 0, 1);
  if (torch::equal(species, species_col.expand_as(species)))
    species = species_col;
  pref->set("species0", species);
}

torch::Tensor HydroImpl::u_ref_like(torch::Tensor const& u) const {
  TORCH_CHECK(has_reference(), "Hydro: no anomaly reference");
  auto r = u_ref();
  TORCH_CHECK(u.dim() == 4 && u.size(0) > IPR && u.size(1) >= r.size(1) &&
                  (r.size(1) == 1 || u.size(1) == r.size(1)) &&
                  (r.size(2) == 1 || u.size(2) == r.size(2)) &&
                  u.size(3) == r.size(3),
              "Hydro: the anomaly reference ", r.sizes(),
              " needs a full-block state (nvar, n3, n2, n1), got ", u.sizes(),
              "; a sub-array cannot be converted with the anomaly keys");
  // Keep the master precision when reading totals, including dry_density().
  // In-place encoding helpers retain their destination dtype; callers that
  // need exact totals use total_u(), which promotes before addition.
  return r.expand({2, u.size(1), u.size(2), u.size(3)});
}

void HydroImpl::add_reference_(torch::Tensor u) const {
  if (!has_reference()) return;
  auto r = u_ref_like(u);
  u[IDN].add_(r[0]);
  u[IPR].add_(r[1]);
}

void HydroImpl::sub_reference_(torch::Tensor u) const {
  if (!has_reference()) return;
  auto r = u_ref_like(u);
  u[IDN].sub_(r[0]);
  u[IPR].sub_(r[1]);
}

torch::Tensor HydroImpl::total_u(torch::Tensor const& u) const {
  if (!has_reference()) return u;
  // Promote BEFORE adding the background: casting the rounded total back to
  // double cannot recover the small increment retained by anomaly storage.
  auto t =
      u.scalar_type() == torch::kFloat32 ? u.to(torch::kFloat64) : u.clone();
  add_reference_(t);
  return t;
}

}  // namespace snap
