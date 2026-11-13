#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>
#include <cctype>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "gravity_work_radial.hpp"
#include "hydro.hpp"
#include "hydro_dispatch.hpp"
#include "hydro_reference.hpp"
#include "wb_ref4.hpp"

namespace snap {
HydroImpl::HydroImpl(const HydroOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void HydroImpl::reset() {
  TORCH_CHECK(pmb, "[Hydro] Parent MeshBlock is null");
  TORCH_CHECK(
      options->conversion_order() == 2 || options->conversion_order() == 4,
      "[Hydro] dynamics/conversion-order must be 2 or 4, got ",
      options->conversion_order());

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
    TORCH_CHECK(gw == "cell" || gw == "face",
                "const-gravity gravity-work must be cell or face, got '", gw,
                "'");
    // the fixer runs with gravity-work: cell only; record the effective value
    // (a face form set from Python keeps the default true otherwise)
    if (gw != "cell") options->grav()->gravity_work_fixer() = false;
    TORCH_CHECK(!gravity_work_fixer() || (options->grav()->grav2() == 0. &&
                                          options->grav()->grav3() == 0.),
                "const-gravity gravity-work-fixer needs grav2 = grav3 = 0 (its "
                "potential is -grav1 * x1); set it false otherwise");
    // mass that wraps across a periodic x1 face jumps by grav1 * Lx1 in that
    // potential, which the E+PE defect would book as a defect. The layout's
    // x1 wrap is read from its topology: a Python bfuncs() call clears names.
    auto const& nm = pmb->options->bcnames();
    bool wrap = pmb->options->layout()->periodic_z();
    for (int f : {0, 1}) {
      TORCH_CHECK(!gravity_work_fixer() ||
                      !(wrap || (f < static_cast<int>(nm.size()) &&
                                 nm[f].compare(0, 8, "periodic") == 0)),
                  "const-gravity gravity-work-fixer needs non-periodic x1 "
                  "boundaries (its potential -grav1 * x1 jumps across a "
                  "periodic x1 face); set it false otherwise");
    }
    // the corrected PE needs a single-valued potential; -grav1 * x1 jumps at
    // a periodic x1 face (read as for the fixer above): warn, no change
    if (options->grav()->grav1() != 0. && gw == "face") {
      bool periodic_x1 = wrap;
      for (int f : {0, 1})
        periodic_x1 = periodic_x1 || (f < static_cast<int>(nm.size()) &&
                                      nm[f].compare(0, 8, "periodic") == 0);
      if (periodic_x1)
        TORCH_WARN_ONCE(
            "gravity-work: face with periodic x1: -grav1*x1 jumps at the "
            "periodic face, so the corrected PE its work books is not defined "
            "there; use x1 walls or gravity-work: cell");
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
                  "SNAP_X1_MASS_COVARIANCE needs at least 3 x1 cells per "
                  "block, got ",
                  nx1, "; unset it");
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

  //// ---- (6) set up implicit solver ---- ////
  if (options->icorr()) {
    picorr = ImplicitHydroImpl::create(options->icorr(), this);
    if (options->grav() && options->grav()->grav1() != 0. &&
        options->grav()->gravity_work() != "cell" && !face_work_in_operator()) {
      TORCH_WARN(
          "gravity-work: ", options->grav()->gravity_work(),
          " with an implicit scheme: the face work "
          "sits outside the implicit operator, which linearises the cell "
          "work; a tall column at acoustic Courant ~66 blows up "
          "(chengcli/snapy#283). gravity-work: cell is consistent with it.");
    }
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

  //// ---- (9) register all forcings ---- ////
  for (auto i = 0; i < forcings.size(); i++) {
    register_module(forcing_names[i], forcings[i].ptr());
  }

  //// ---- (10) populate buffers ---- ////
  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nvar = peos->nvar();

  if (nc1 > 1) {
    _flux1 = register_buffer(
        "F1", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
    _face_pressure1 =
        register_buffer("P1", torch::zeros({nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux1 = register_buffer("F1", torch::Tensor());
    _face_pressure1 = register_buffer("P1", torch::Tensor());
  }

  if (nc2 > 1) {
    _flux2 = register_buffer(
        "F2", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux2 = register_buffer("F2", torch::Tensor());
  }

  if (nc3 > 1) {
    _flux3 = register_buffer(
        "F3", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux3 = register_buffer("F3", torch::Tensor());
  }

  _div = register_buffer("D",
                         torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));

  _positivity_hits =
      register_buffer("positivity_hits", torch::zeros({1}, torch::kInt64));
  _positivity_severe =
      register_buffer("positivity_severe", torch::zeros({1}, torch::kInt64));
  _positivity_min =
      register_buffer("positivity_min", torch::ones({1}, torch::kFloat64));
  if (options->conversion_order() == 4)
    _conv4_trips =
        register_buffer("conversion_trips", torch::zeros({1}, torch::kInt64));
  _lim_cut = register_buffer("lim_cut", torch::zeros({1}, torch::kFloat64));
  _lim_flux = register_buffer("lim_flux", torch::zeros({1}, torch::kFloat64));
  _gwfix_d = register_buffer("gwfix_d", torch::zeros({1}, torch::kFloat64));
  _gwfix_total =
      register_buffer("gwfix_total", torch::zeros({1}, torch::kFloat64));
  _gwfix_pending =
      register_buffer("gwfix_pending", torch::zeros({1}, torch::kFloat64));
  _gwfix_wall =
      register_buffer("gwfix_wall", torch::zeros({1}, torch::kFloat64));
}

bool HydroImpl::is_x1_wall(int f) const {
  auto const& nm = pmb->options->bcnames();
  return pmb->options->is_physical_boundary(0, 0, f == 0 ? -1 : 1) &&
         !(f < static_cast<int>(nm.size()) &&
           nm[f].compare(0, 8, "periodic") == 0);
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

bool HydroImpl::gravity_work_fixer() const {
  auto g = options->grav();
  return g && g->grav1() != 0. && g->gravity_work() == "cell" &&
         g->gravity_work_fixer();
}

bool HydroImpl::flux_covariance() {
  // read once: the term is a scheme choice for the whole run, and every block
  // must make the same one or the shared faces stop being single-valued; on
  // unless SNAP_FLUX_COVARIANCE is set to 0/false/off/no or empty
  static const bool on = [] {
    auto v = get_env("SNAP_FLUX_COVARIANCE", "1");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  return on;
}

bool HydroImpl::x1_mass_covariance() {
  static const bool on = [] {
    auto v = get_env("SNAP_X1_MASS_COVARIANCE", "0");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  return on;
}

bool HydroImpl::radial_exact_work() const {
  auto g = options->grav();
  auto const& type = pmb->pcoord->options->type();
  return g && g->grav1() != 0. && g->gravity_work() == "face" &&
         x1_measure(type) != X1Measure::none;
}

std::pair<int, int> HydroImpl::x1_neighbors() const {
  auto layout = pmb->get_layout();
  if (!layout || layout->options->periodic_z() || layout->options->pz() <= 1)
    return {-1, -1};
  auto iloc = layout->loc_of(layout->options->rank());
  return {layout->neighbor_rank(iloc, {0, 0, -1}),
          layout->neighbor_rank(iloc, {0, 0, 1})};
}

void HydroImpl::_x1_ghost_rows(torch::Tensor f, int m, bool faces,
                               int tag) const {
  auto [below, above] = x1_neighbors();
  if (below < 0 && above < 0) return;
  auto layout = pmb->get_layout();
  int is = pmb->pcoord->il(), iu = pmb->pcoord->iu(), o = faces ? 1 : 0;
  std::vector<std::vector<torch::Tensor>> sbufs;
  std::vector<CommWorkPtr> sends;
  sbufs.reserve(2);
  // same board / process-group pairing as the W ghost-row exchange
  auto post = [&](torch::Tensor rows, int to, int t) {
    if (layout->is_local_block(to)) {
      layout->post_to_local_block(to, rows, t);
    } else {
      sbufs.push_back({rows.contiguous()});
      sends.push_back(layout->send_to_block(sbufs.back(), to, t));
    }
  };
  auto take = [&](int from, int t) {
    if (layout->is_local_block(from))
      return layout->take_from_local_block(from, t);
    std::vector<torch::Tensor> rbuf = {
        torch::empty_like(f.narrow(-1, 0, m).contiguous())};
    layout->recv_from_block(rbuf, from, t)->wait();
    return rbuf[0];
  };
  // up: my top rows are the above block's lower ghosts; down: my bottom rows
  // (past the seam face, for faces) are the below block's upper ghosts
  if (above >= 0) post(f.narrow(-1, iu + 1 - m, m), above, tag);
  if (below >= 0) post(f.narrow(-1, is + o, m), below, tag + 1);
  if (below >= 0) f.narrow(-1, is - m, m).copy_(take(below, tag));
  if (above >= 0) f.narrow(-1, iu + 1 + o, m).copy_(take(above, tag + 1));
  for (auto& sw : sends) sw->wait();
}

bool HydroImpl::face_work_in_operator() const {
  auto g = options->grav();
  return g && g->grav1() != 0. && g->gravity_work() == "face" && picorr &&
         picorr->options->scheme() != 0;
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
                              torch::dtype(torch::kFloat64).device(w.device()));

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
  if (pdiffusion) dt = std::min(dt, pdiffusion->max_time_step(w));
  return dt;
}

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

torch::Tensor HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                                    torch::Tensor const& w,
                                                    double dt,
                                                    Variables const& other) {
  if (!picorr) return torch::Tensor();

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

  du[IPR].sub_(peos->internal_energy_offset(du));

  torch::Tensor gamma;
  if (options->eos()->type() == "aneos") {
    auto cs = peos->compute("W->L", {w});
    gamma = peos->compute("WL->A", {w, cs});
  } else {
    gamma = peos->compute("W->A", {wi});
  }
  auto correction = picorr->forward_masked(
      du, wi, gamma, dt,
      other.count("solid") ? other.at("solid") : torch::Tensor());
  du[IPR].add_(peos->internal_energy_offset(du));
  // picorr measured its delta after removing the EOS reference energy.
  // Diagnostics expose a conserved-state delta, so restore that reference
  // contribution using the corrected density and species tendencies.
  correction[IPR].add_(peos->internal_energy_offset(correction));

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

void HydroImpl::_revise_x1inner_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, is, 1) / w[IDN].narrow(-1, is, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[is - n - 1];
    auto h =
        w[IPR].narrow(-1, is - n, 1).pow(a) + a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, is - n - 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, is - n - 1, 1) =
        (w[IPR].narrow(-1, is - n - 1, 1) / K).pow(1. / gamma);
  }
}

void HydroImpl::_revise_x1outer_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int ie = pcoord->iu();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, ie, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, ie, 1) / w[IDN].narrow(-1, ie, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[ie + n];
    auto h =
        w[IPR].narrow(-1, ie + n, 1).pow(a) - a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, ie + n + 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, ie + n + 1, 1) =
        (w[IPR].narrow(-1, ie + n + 1, 1) / K).pow(1. / gamma);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
HydroImpl::_hydro_ref_x1(torch::Tensor const& w) const {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  int iu = pcoord->iu();
  double g = -options->grav()->grav1();  // downward magnitude (>0)

  // Global x1 reference across a vertical (nb1>1) decomposition: make the
  // hydrostatic reference continuous over the x1 process seams. The block
  // owning x1-outer anchors at the true domain top; every block below receives
  // the running seam-face pressure from the block above and passes on that
  // value plus its own interior hydrostatic drop (= its bottom-face pressure).
  // A serial top->bottom scan along the x1 block column, whether the blocks
  // sit on other processes or on other threads of this one (#254). nb1 == 1
  // (no x1 neighbor) => an empty anchor tells the backend to compute the
  // local top anchor.
  constexpr int kWbRefTag = 0x7715;
  torch::Tensor anchor;
  int below = -1;
  int above = -1;
  auto layout = pmb->get_layout();
  // pz==nb1: relay only across a SPLIT x1 column (else unmatched send/recv at
  // nb1=1)
  bool x1_split =
      layout && !layout->options->periodic_z() && layout->options->pz() > 1;
  if (x1_split) {
    auto iloc = layout->loc_of(layout->options->rank());
    above = layout->neighbor_rank(iloc, {0, 0, 1});   // toward x1-outer
    below = layout->neighbor_rank(iloc, {0, 0, -1});  // toward x1-inner
    anchor = layout->take_x1_anchor(
        above, torch::empty({w.size(1), w.size(2), 1}, w.options()), kWbRefTag);
  }

  bool stretched = pcoord->options->stretched_x1();
  bool phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = pmb->options->is_physical_boundary(0, 0, 1);
  bool clamp = options->wb_wall_clamp();
  if (!x1_reference_ || !x1_reference_->matches(pcoord->dx1f, iu, stretched,
                                                phys_in, phys_out, clamp)) {
    x1_reference_ = std::make_shared<HydroReferenceX1Operator>(
        pcoord->dx1f, iu, stretched, phys_in, phys_out, clamp);
  }
  HydroReferenceX1 reference(w);
  x1_reference_->evaluate(w, g, anchor, reference);
  auto& [psf_lo, psf_hi, pref, dsf, dref] = reference;

  // dynamics/conversion-order 4 (#289): the only fourth-order part is the
  // face density below, the face value of the kernel's own cell dref -- that
  // is what makes the O(dz^2) face offset vanish, whatever dref's own accuracy
  // (wb-ref4.md section 2), and replacing dref broke the theta bound of a
  // density current at a wall. psf is never changed, so the rest balance is
  // the kernel's. Only the resolution flag is wanted here; the face part
  // follows the seam exchange, so a column split along x1 gets one block's
  // faces.
  torch::Tensor wb4_flag;
  if (options->conversion_order() == 4) {
    if (!wb_ref4_ || wb_ref4_->stale(w)) {
      bool clamp = options->wb_wall_clamp();
      wb_ref4_ = std::make_shared<WbRef4Stencils>(
          wb_ref4_stencils(pcoord->x1f, is, iu, clamp && phys_in,
                           clamp && phys_out, w.options()));
    }
    wb4_flag = wb_ref4_flag(*wb_ref4_, psf_lo, psf_hi);
  }

  if (below >= 0) {
    layout->pass_x1_anchor(below, psf_lo.narrow(-1, is, 1), kWbRefTag);
  }

  // Ghost-row reference exchange across x1 seams: overwrite this block's
  // ghost rows of (pref, dref) with the NEIGHBOR'S interior rows for the same
  // physical cells. The block-local ghost quadrature (w6e edge rows, and the
  // [lo,hi] guard whose accept/fallback decision is stencil-dependent) does
  // NOT reproduce the owner's interior values -- near a physical wall the
  // mismatch reaches ~1e2 Pa (measured), which enters
  // w' = w - ref as a spurious seam perturbation every step and makes the
  // dynamics nb1-dependent. After this exchange the perturbation field seen
  // by the reconstruction is identical on both sides of every seam, so the
  // seam-face states agree and the single-valued seam flux average becomes a
  // no-op. nb1=1: no seams, bit-unchanged. A same-process neighbor goes
  // through a board, a remote one through the process group (#254).
  if (x1_split && (below >= 0 || above >= 0)) {
    constexpr int kWbGhostUpTag = 0x7717;
    constexpr int kWbGhostDnTag = 0x7718;
    int ng = is;  // il() == nghost
    std::vector<std::vector<torch::Tensor>> sbufs;
    std::vector<CommWorkPtr> sends;
    auto post = [&](torch::Tensor rows, int to, int tag) {
      if (layout->is_local_block(to)) {
        layout->post_to_local_block(to, rows, tag);
      } else {
        sbufs.push_back({rows.contiguous()});
        sends.push_back(layout->send_to_block(sbufs.back(), to, tag));
      }
    };
    auto take = [&](int from, int tag) {
      if (layout->is_local_block(from))
        return layout->take_from_local_block(from, tag);
      std::vector<torch::Tensor> rbuf = {
          torch::empty({w.size(1), w.size(2), 2 * ng}, w.options())};
      layout->recv_from_block(rbuf, from, tag)->wait();
      return rbuf[0];
    };
    sbufs.reserve(2);
    if (above >= 0) {  // my top interior rows are the above block's lower
                       // ghosts
      post(torch::cat({pref.narrow(-1, iu - ng + 1, ng),
                       dref.narrow(-1, iu - ng + 1, ng)},
                      -1),
           above, kWbGhostUpTag);
    }
    if (below >= 0) {  // my bottom interior rows are the below block's upper
                       // ghosts
      post(torch::cat({pref.narrow(-1, is, ng), dref.narrow(-1, is, ng)}, -1),
           below, kWbGhostDnTag);
    }
    if (below >= 0) {  // receive my lower ghost rows from below's top interior
      auto rb = take(below, kWbGhostUpTag);
      pref.narrow(-1, 0, ng).copy_(rb.narrow(-1, 0, ng));
      dref.narrow(-1, 0, ng).copy_(rb.narrow(-1, ng, ng));
    }
    if (above >= 0) {  // receive my upper ghost rows from above's bottom
                       // interior
      auto ra = take(above, kWbGhostDnTag);
      pref.narrow(-1, iu + 1, ng).copy_(ra.narrow(-1, 0, ng));
      dref.narrow(-1, iu + 1, ng).copy_(ra.narrow(-1, ng, ng));
    }
    for (auto& sw : sends) sw->wait();
  }

  // conversion-order 4: the face density is the fourth-order face value of
  // the (exchanged) cell density reference, so a split column gets the same
  // faces as one block
  if (wb4_flag.defined()) wb_ref4_faces(*wb_ref4_, dref, dsf, wb4_flag);

  // the face density on each side of a face is the x1 reconstruction's own
  // optimal-linear row applied to dref (cp5 = the weno5 candidates combined
  // with their linear weights), so rho_f = R_lin(rho) + N(rho') whatever
  // dref is. Read after the seam exchange; a face whose value leaves
  // [min, max] of its two cells' dref keeps the face density above, which is
  // also what any other reconstruction gets on both sides.
  auto dsf_lr = torch::stack({dsf, dsf});
  auto const& itype = options->recon1()->interp()->type();
  int nc1 = dref.size(-1);
  if ((itype == "weno5" || itype == "cp5") && nc1 >= 6) {
    if (!dsf_lin_ || dsf_lin_->cm.device() != w.device() ||
        dsf_lin_->cm.scalar_type() != w.scalar_type()) {
      auto op = InterpOptionsImpl::create();
      op->type("cp5");
      dsf_lin_ = std::make_shared<Center5InterpImpl>(op);
      if (precon1->pinterp1->has_x1_coeffs()) {
        auto const& co = pcoord->options;
        bool plain =
            x1_centroid_exact_enabled() && co->type() == "spherical-polar";
        auto x1f = !plain               ? pcoord->x1_row_faces(itype)
                   : co->stretched_x1() ? pcoord->x1f
                                        : torch::Tensor();
        if (x1f.defined()) dsf_lin_->set_x1_coeffs(x1f);
      }
      dsf_lin_->to(w.device(), w.scalar_type());
    }
    constexpr int D1 = 3;
    int il = 3, iu5 = nc1 - 3;
    auto lin = torch::stack({dsf, dsf}).unsqueeze(1);
    dsf_lin_->forward(dref.unsqueeze(0), D1,
                      lin[IRT].slice(D1, il - 1, iu5 + 1),
                      lin[ILT].slice(D1, il, iu5 + 2));
    lin = lin.squeeze(1);
    auto dl = dref.narrow(-1, 0, nc1 - 1), dr = dref.narrow(-1, 1, nc1 - 1);
    for (int s : {(int)ILT, (int)IRT}) {
      auto face = lin[s].narrow(-1, il - 1, iu5 + 3 - il);
      auto lo = torch::minimum(dl, dr).narrow(-1, il - 2, iu5 + 3 - il);
      auto hi = torch::maximum(dl, dr).narrow(-1, il - 2, iu5 + 3 - il);
      auto d = dsf_lr[s].narrow(-1, il - 1, iu5 + 3 - il);
      d.copy_(torch::where((face >= lo) & (face <= hi), face, d));
    }
    // a reflecting x1 wall face is a mirror: one face density on both sides,
    // the mean of the two rows. Two values there give a Riemann solver whose
    // wave speeds read the face density (hllc, roe) a nonzero contact speed,
    // which moves mass through the wall.
    auto const& bf = pmb->options->bfuncs();
    for (int f : {0, 1}) {
      if (!is_x1_wall(f) || !is_reflecting(bf[f])) continue;
      int i = f == 0 ? pcoord->il() : pcoord->iu() + 1;
      auto mean = 0.5 * (dsf_lr[ILT].select(-1, i) + dsf_lr[IRT].select(-1, i));
      for (int s : {(int)ILT, (int)IRT}) dsf_lr[s].select(-1, i).copy_(mean);
    }
  }

  return {psf_lo, pref, dsf_lr, dref};
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
