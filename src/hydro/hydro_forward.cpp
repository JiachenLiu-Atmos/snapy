// C/C++
#include <chrono>
#include <limits>
#include <numeric>
#include <string>
#include <typeinfo>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/species_sum.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/layout/seam_register.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/conversion.hpp>
#include <snap/utils/log.hpp>

#include "flux_positivity.hpp"
#include "gravity_work.hpp"
#include "gravity_work_radial.hpp"
#include "hydro.hpp"
#include "hydro_dispatch.hpp"
#include "sweep.hpp"
#include "x1_mass_covariance.hpp"

namespace snap {

// #289: the x1 derivative of the pressure for the cells with both x1
// neighbours, centred, but one-sided at the first and last interior cell so
// that no x1 ghost row is read. The pressure part of the centroid term and the
// geometric source must use the SAME difference for the rest balance to stay
// exact, and x2/x3 face states in the x1 ghost rows are not reliable next to a
// cubed-sphere panel edge.
static torch::Tensor d1_pressure(torch::Tensor const& p,
                                 torch::Tensor const& x1v, int is, int ie) {
  int n1 = p.size(-1);
  auto dx1 = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
  auto d = (p.narrow(-1, 2, n1 - 2) - p.narrow(-1, 0, n1 - 2)) / dx1;
  d.select(-1, is - 1) =
      (p.select(-1, is + 1) - p.select(-1, is)) / (x1v[is + 1] - x1v[is]);
  d.select(-1, ie - 1) =
      (p.select(-1, ie) - p.select(-1, ie - 1)) / (x1v[ie] - x1v[ie - 1]);
  return d;
}

// #289 x1 face-centroid shift, as `like`; undefined where identically zero
static torch::Tensor centroid_shift_x1(CoordinateImpl const& coord,
                                       torch::Tensor const& like) {
  if (coord.face_centroid_shift_zero()) return torch::Tensor();
  return coord.face_centroid_shift_x1()
      .to(like.device(), like.scalar_type())
      .unsqueeze(0)
      .unsqueeze(1);
}

torch::Tensor HydroImpl::_flux_covariance(torch::Tensor const& wl,
                                          torch::Tensor const& wr,
                                          int dim) const {
  // #289: the finite-volume flux through an x2/x3 face is the FACE AVERAGE of
  // the point flux, but the solver evaluates the flux FROM the cell values.
  // Expanding about the face's AREA centroid, with r_v the centroid of the
  // CELL measure and r_c that of the FACE measure, the enthalpy flux is left
  // short by TWO terms of the same order:
  //
  //   dF = sigma1^2 rho h_1 (u_n)_1 - (r_v - r_c) d_1( rho h u_n ) ,
  //
  // the first a covariance, the second a centroid offset that vanishes
  // identically in Cartesian and is of relative size ~L/r on a curved grid.
  //
  // where sigma1^2 is the x1 second central moment of the face's own area
  // measure (dx1^2/12 in cartesian, face_moment2_x1() in general) and u_n is
  // the face-normal velocity, and h = (W->I + p)/rho is the solver's own
  // specific enthalpy (every species' energy offset included). Differencing h
  // is exact for any p = rho Pi(e, y), and offset-invariant together with the
  // tracer rows below; for a dry ideal gas with zero offset it equals the
  // earlier gamma/(gamma-1) p [ln(p/rho)]_1 form.
  //
  // The covariance along the OTHER in-face coordinate is dropped: only x1
  // carries an O(1) background gradient, so the horizontal one is quadratic in
  // the perturbation.
  //
  // dF is zero at rest by construction: every term below carries a factor of
  // the face-normal velocity or its x1 difference, both exactly zero there, so
  // a balanced state keeps its exact zero tendency bitwise. The COVARIANCE
  // term is also zero for an isothermal state; the CENTROID term is not, so
  // that second exact zero holds in Cartesian only. dF is added to the FACE
  // FLUX, so both terms telescope in the x2/x3 sums exactly like the flux they
  // correct.
  auto pcoord = pmb->pcoord;
  int n1 = wl.size(-1);
  // a centred x1 difference needs both neighbours; nghost >= 1 gives them to
  // every interior cell of a resolved x1 axis, and an unresolved one has no
  // vertical gradient to correct for
  if (n1 < 3) return torch::Tensor();
  // shallow water carries no internal energy row to correct
  if (peos->options->type() == "shallow-water") return torch::Tensor();

  auto wbar = 0.5 * (wl + wr);
  auto p = wbar[IPR];
  // rho h: the solver's own enthalpy density, W->I (with every species'
  // energy offset) plus p
  auto enth = peos->compute("W->I", {wbar}) + p;

  // the face-normal velocity in the face-local orthonormal frame -- the
  // component the energy flux actually carries. Projected on our own copy:
  // whether the Riemann solver projects its inputs in place is the solver's
  // business (plume_roe does not), so do not read that side effect.
  if (dim == 2) {
    pcoord->prim2local2_(wbar);
  } else {
    pcoord->prim2local3_(wbar);
  }
  auto un = wbar[dim == 2 ? IVY : IVZ];

  auto x1v = pcoord->x1v.to(p.device(), p.scalar_type());
  // face_moment2_x1() is grid-only; cache it per block, keyed on the x1f
  // tensor it is built from, its version, the device and the dtype
  auto const& x1f_ = pcoord->x1f;
  if (!(_s2.x1f.defined() && _s2.x1f.is_same(x1f_) &&
        _s2.ver == X1GridKey::version(x1f_) && _s2.val.defined() &&
        _s2.val.device() == p.device() &&
        _s2.val.scalar_type() == p.scalar_type())) {
    _s2.val = pcoord->face_moment2_x1()
                  .to(p.device(), p.scalar_type())
                  .unsqueeze(0)
                  .unsqueeze(1);
    _s2.x1f = x1f_;
    _s2.ver = X1GridKey::version(x1f_);
  }
  auto s2 = _s2.val;
  // Second O(dx1^2) term, nonzero on curved grids only: a cell stores the
  // average over the CELL measure, whose x1 centroid is r_v, while the flux
  // needs the average over the FACE measure, whose centroid is r_c. The
  // offset r_v - r_c costs (r_v - r_c) d_1(enthalpy flux). It is identically
  // zero in Cartesian and is NOT small on a curved grid -- its ratio to the
  // covariance term is ~L/r with L the local gradient scale. See
  // docs/derivations/289-covariance-x3-curved.md.
  // Cartesian (face_centroid_shift_zero): the shift is identically zero, so
  // every centroid part below is an exact zero and is not computed
  auto ds = centroid_shift_x1(*pcoord, p);
  bool centroid = ds.defined();
  auto rho = wbar[IDN];
  auto dx1 = (x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2));

  // ---- every row (#289 all-rows, see the derivation, section 4B) -----------
  // Cell values are DENSITY-WEIGHTED: q = (rho q)/rho, u = m/rho
  // (ideal_moist_impl.h:40,44), so the face-average gap of a flux rho u_n q is
  // sigma1^2 rho (u_n)_1 q_1 and no pair contains rho'; the total mass flux m_n
  // is linear in the stored state and has no covariance at all. Every row also
  // keeps its centroid part -delta d_1(F). The covariance parts:
  //   - total mass: 0;
  //   - tracer n: sigma1^2 rho D1[u_n] D1[q_n];
  //   - dry mass: minus the sum of the tracer parts;
  //   - energy: sigma1^2 rho D1[h] D1[u_n], h = (W->I + p)/rho (no KE);
  //   - momentum: none; the centroid part is taken in the face-local frame,
  //     and its pressure part p* = p - delta d_1 p is mirrored in the
  //     geometric source (forward(), step 5), so rest stays exact.
  // The velocity-squared parts of the momentum and energy gaps are the scheme's
  // ordinary second-order truncation and are left out, as is the pressure's J
  // (Hessian of p/rho along the gradient; ~1e-3 of the energy term).
  int ny = wl.size(0) - ICY;  // species rows, dry mass fraction is 1 - sum
  auto cell = [&](torch::Tensor const& q) { return q.narrow(-1, 1, n1 - 2); };
  auto d1 = [&](torch::Tensor const& q) {
    return (q.narrow(-1, 2, n1 - 2) - q.narrow(-1, 0, n1 - 2)) / dx1;
  };
  // sigma1^2 rho a_1 b_1 : the covariance of a density-weighted product;
  // sigma1^2 rho and D1[u_n] are shared by every row, so formed once
  auto s2rho = cell(s2) * cell(rho);
  auto dun = d1(un);
  // -delta d_1(F) : the centroid part, the same rule for every row
  auto ctr = [&](torch::Tensor const& f) { return -cell(ds) * d1(f); };

  auto h = enth / rho;
  auto ys = wbar.narrow(0, ICY, ny);

  // Cartesian: only the IPR row and, with species, the ICY.. and IDN rows are
  // nonzero, and only on the x1-interior cells; write just those, stacked as
  // (IPR, ICY.., IDN), for forward()'s add_cov (no zero-filled full flux)
  if (!centroid) {
    auto rows = torch::empty(
        {ny > 0 ? ny + 2 : 1, rho.size(0), rho.size(1), n1 - 2}, rho.options());
    auto ce = rows[0];
    torch::mul_out(ce, s2rho * d1(h), dun);
    if (ny > 0) {
      // tracer rows, all species at once, each with its own q
      auto cy = rows.narrow(0, 1, ny);
      torch::mul_out(cy, s2rho * dun, d1(ys));
      // the dry row carries minus the sum of the tracer rows
      auto dry = rows[ny + 1];
      torch::neg_out(dry, cy[0]);
      for (int n = 1; n < ny; ++n) torch::sub_out(dry, dry, cy[n]);
    }
    return rows;
  }

  auto dflx = torch::zeros_like(wl);
  // energy row
  dflx[IPR].narrow(-1, 1, n1 - 2) = s2rho * d1(h) * dun + ctr(enth * un);
  // tracer rows, all species at once; the dry row carries minus their sum
  auto mflx = rho * un;
  // dry mass fraction: LMARS carries FLX(IDN) = ubar * rho * rd
  auto rd = torch::ones_like(rho);
  for (int n = 0; n < ny; ++n) rd = rd - wbar[ICY + n];
  auto dry = ctr(mflx * rd);
  if (ny > 0) {
    auto cy = s2rho * dun * d1(ys);
    dflx.narrow(0, ICY, ny).narrow(-1, 1, n1 - 2) = cy + ctr(mflx * ys);
    for (int n = 0; n < ny; ++n) dry = dry - cy[n];
  }
  dflx[IDN].narrow(-1, 1, n1 - 2) = dry;

  // the momentum rows are centroid parts only: momentum rows in the
  // face-local frame, then to the solver's global frame like the Riemann flux
  // (lmars.cpp): normal rho u_n^2 + p, tangential rho u_n u_t
  int ivn = dim == 2 ? IVY : IVZ;
  for (int v = IVX; v <= IVZ; ++v) {
    dflx[v].narrow(-1, 1, n1 - 2) = ctr(mflx * wbar[v]);
  }
  // the pressure part: p* - p = -delta d_1 p, with the difference step 5 of
  // forward() also uses for the geometric source
  dflx[ivn].narrow(-1, 1, n1 - 2) -=
      cell(ds) * d1_pressure(p, x1v, pcoord->il(), pcoord->iu());
  if (dim == 2) {
    pcoord->flux2global2_(dflx);
  } else {
    pcoord->flux2global3_(dflx);
  }
  return dflx;
}

torch::Tensor HydroImpl::scalar_flux_covariance(torch::Tensor const& rl,
                                                torch::Tensor const& rr,
                                                int dim) const {
  // A passive scalar s = rho_dry r is a tracer like the hydro species: its
  // x2/x3 face flux needs the tracer row of _flux_covariance,
  // sigma1^2 rho D1[u_n] D1[q] - delta d_1(rho u_n q), with q = s/rho its mass
  // fraction of the total density. On each side of the face q = r rd, rd the
  // dry fraction of the hydro state on that side, and the face value is their
  // mean, as wbar is for a species. Every other factor is the hydro's own,
  // from the same face states and in the same order, so a scalar and the same
  // tracer carried as a species get the same row.
  int k = dim == 2 ? 0 : 1;
  if (!_cov_wl[k].defined()) return torch::Tensor();
  auto const& wl = _cov_wl[k];
  auto const& wr = _cov_wr[k];
  auto pcoord = pmb->pcoord;
  int n1 = wl.size(-1);
  int ny = wl.size(0) - ICY;
  auto dry_fraction = [&](torch::Tensor const& w) {
    return species_complement(w[IDN], w.narrow(0, ICY, ny));
  };
  // past an x1 seam wl/wr hold the neighbour's rows (x1_seam_rows in
  // forward()); the scalar's own face states take the same rows
  // both sides travel in one x1 message (S's merge M5)
  auto sl = rl, sr = rr;
  Exchange ex(pmb);
  auto [below, above] = ex.x1_neighbors();
  if (below >= 0 || above >= 0) {
    sl = rl.clone();
    sr = rr.clone();
    ex.x1_rows({sl, sr}, 1, false, kTagScalarCov + 4 * k);
  }
  auto q = 0.5 * (sl * dry_fraction(wl) + sr * dry_fraction(wr));

  // the face-normal velocity and the geometry, as in _flux_covariance
  auto wbar = 0.5 * (wl + wr);
  if (dim == 2) {
    pcoord->prim2local2_(wbar);
  } else {
    pcoord->prim2local3_(wbar);
  }
  auto un = wbar[dim == 2 ? IVY : IVZ];
  auto x1v = pcoord->x1v.to(un.device(), un.scalar_type());
  auto s2 = pcoord->face_moment2_x1()
                .to(un.device(), un.scalar_type())
                .unsqueeze(0)
                .unsqueeze(1);
  auto ds = centroid_shift_x1(*pcoord, un);
  bool centroid = ds.defined();
  auto rho = wbar[IDN];
  auto dx1 = (x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2));
  auto cell = [&](torch::Tensor const& a) { return a.narrow(-1, 1, n1 - 2); };
  auto d1 = [&](torch::Tensor const& a) {
    return (a.narrow(-1, 2, n1 - 2) - a.narrow(-1, 0, n1 - 2)) / dx1;
  };
  auto cov = [&](torch::Tensor const& a, torch::Tensor const& b) {
    return cell(s2) * cell(rho) * d1(a) * d1(b);
  };
  auto ctr = [&](torch::Tensor const& f) { return -cell(ds) * d1(f); };

  auto dflx = torch::zeros_like(q);
  auto c = cov(un, q);
  dflx.narrow(-1, 1, n1 - 2) = centroid ? c + ctr(rho * un * q) : c;
  return dflx;
}

torch::Tensor HydroImpl::forward(StepDt dt, torch::Tensor u,
                                 Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  bool has_solid = other.count("solid");
  auto start = std::chrono::high_resolution_clock::now();
  // verbose: the time since the last report, then restart the clock
  auto report_time = [&](std::string const& label) {
    if (!options->verbose()) return;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << label << " time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  };
  _check_stretched_x1(u);

  auto playout = pmb->get_layout();
  for (int k = 0; k < 2; ++k) {
    _cov_wl[k] = _cov_wr[k] = _cov_dry[k] = torch::Tensor();
  }
  // S step (4), boundary.prescribed: the walls and solid faces S hands to
  // R's register; R applies them after reconcile and after the implicit step.
  // A wall whose boundary prescribes a temperature (the diffusion's wall
  // temperature, or a fixed-temperature boundary function) has no mass flux
  // but conducts: its energy row stays free
  std::array<std::vector<int>, 2> x1_wall_rows;
  for (int lu = 0; lu < 2; ++lu) {
    double wall_t =
        lu == 0 ? (pdiffusion ? pdiffusion->options->temperature_inner() : 0.)
                : (pdiffusion ? pdiffusion->options->temperature_outer() : 0.);
    auto const& names = pmb->options->bcnames();
    bool fixed = names.size() > static_cast<size_t>(lu) &&
                 names[lu] == (lu == 0 ? "fixed_temperature_inner"
                                       : "fixed_temperature_outer");
    if (wall_t > 0. || fixed) x1_wall_rows[lu] = {IPR};
  }
  supply_prescribed(freg, pmb, has_solid ? other.at("solid") : torch::Tensor(),
                    {IVX, IVX + 1, IVX + 2}, x1_wall_rows);

  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  // conversion-order 4: x1 ghosts at a reflecting wall from the owned cells
  // (wall4_rows, constraint c); a rho or p ghost whose total (ref + a) is
  // not positive keeps the mirror and counts as a conversion trip
  int i0 = pmb->pcoord->il(), i1 = pmb->pcoord->iu();
  bool wall4[2];
  for (int f : {0, 1}) {
    wall4[f] = options->conversion_order() == 4 && u.size(DIM1) > 1 &&
               is_x1_wall(f) && is_reflecting(pmb->options->bfuncs()[f]);
    if (wall4[f] && i1 - i0 < 3)
      TORCH_WARN_ONCE("fewer than 4 x1 cells: wall ghosts stay mirrored");
    wall4[f] = wall4[f] && i1 - i0 >= 3;
  }
  bool split_flux = options->perturbation_flux();
  IdealMoistPerturbations perturb;
  torch::Tensor raw_primitive;
  if (split_flux) {
    TORCH_CHECK(!has_solid,
                "perturbation-flux does not support immersed solids");
    TORCH_CHECK(
        w.scalar_type() == torch::kFloat64,
        "perturbation-flux currently requires FP64 compute (M4 enables FP32)");
    perturb = dynamic_cast<IdealMoistImpl&>(*peos).perturbations(
        u, /*defer_check=*/true);
    if (options->conversion_order() == 4) raw_primitive = w.clone();
  }
  // x1 point values u - L u, the primitives there, and back to cell averages:
  // w = W(u - L u) + L W(u), fourth order where W(u) is second (Conversion)
  if (options->conversion_order() == 4 && u.size(DIM1) > 1) {
    auto const& mo = pmb->options;
    int ng = pmb->pcoord->options->nghost();
    auto const& x1f = pmb->pcoord->x1f;  // rebuilt on a new or rewritten grid
    if (!conv4_ || conv4_->key != X1RowsKey(x1f, w.options()) ||
        conv4_->wall(0) != wall4[0] || conv4_->wall(1) != wall4[1])
      conv4_ = std::make_shared<Conversion>(x1f, i0, i1, ng, wall4, u.size(0),
                                            w.options());
    // Conversion acts on total cell averages, including the static background.
    // Differencing only stored anomalies omits the background curvature.
    // Trips count cells per stage, ghosts included
    conv4_->forward(has_reference() ? total_u(u) : u, w, *peos,
                    peos->options->limiter(),
                    pmb->pefs->slot(EfStatus::kConvTrips));
    // every ghost from the converted owned cells: the block's exchange, the
    // x1 rows again across all x2/x3 columns (the corners past a seam), then
    // the boundary conditions on every ghost layer (outflow from the saved
    // background, as the block's own outflow)
    Variables gv{{"hydro_w", w}};
    pmb->exchange(gv, SyncOptions().interpolate(true).type(kPrimitive));
    Exchange(pmb).x1_rows({w}, ng, false, kTagPrimGhosts);
    BoundaryFuncOptions bop;
    bop.nghost(ng);
    bop.type(kPrimitive);
    if (other.count("boundary_reference_w")) {
      bop.reference = other.at("boundary_reference_w");
      bop.eos = peos.get();
      bop.coord = pmb->pcoord.get();
    }
    for (int f = 0; f < mo->bfuncs().size(); ++f)
      if (mo->bfuncs()[f] &&
          (bop.reference.defined() || !is_outflow(mo->bfuncs()[f])))
        mo->bfuncs()[f](w, 3 - f / 2, bop);
  }
  report_time("EOS");

  if (split_flux && raw_primitive.defined()) {
    // Conversion and its admissibility/ghost decisions remain FP64. Apply its
    // finite correction to the algebraic EOS increment, not to rounded totals.
    perturb.pressure_delta =
        perturb.pressure_delta + (w[IPR] - raw_primitive[IPR]);
    perturb.density_delta =
        perturb.density_delta + (w[IDN] - raw_primitive[IDN]);
  }

  if (has_solid) {
    pmb->pib->mark_prim_solid_(w, other.at("solid"));
  }

  // hydrostatic pressure correction
  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);
  torch::Tensor rho_grav_delta;
  torch::Tensor pressure_base1, pressure_base1_before_seam;
  torch::Tensor pressure_base2, pressure_base3;
  torch::Tensor background_force1;
  auto face_gradient = [&](torch::Tensor const& p, int dim) {
    auto out = torch::zeros_like(w[IDN]);
    int axis = dim - 1, n = p.size(axis) - 1;
    auto dx = dim == DIM1   ? pmb->pcoord->dx1f
              : dim == DIM2 ? pmb->pcoord->dx2f
                            : pmb->pcoord->dx3f;
    std::vector<int64_t> shape(3, 1);
    shape[axis] = n;
    out.narrow(axis, 0, n)
        .copy_((p.narrow(axis, 1, n) - p.narrow(axis, 0, n)) /
               dx.narrow(0, 0, n).view(shape));
    return out;
  };
  int ny = u.size(0) - ICY;
  // the settling part of the x1 species flux, for the positivity carry: the
  // species rows of sedimentation's own P1 buffer (R2)
  torch::Tensor fsed1;

  // FaceFloor (C.rst 3.7): the faces a floor changed go to the monitor
  auto& monitor = freg.monitor();
  // the split path's background and increments for the face sweeps
  IdealMoistPerturbations const* split = split_flux ? &perturb : nullptr;
  // a disabled direction's hydro flux is zero: its total is rewritten each
  // stage too, so the sources' posts (diffusion energy) never accumulate
  bool const off[3] = {options->disable_flux_x1(), options->disable_flux_x2(),
                       options->disable_flux_x3()};
  for (int d = 1; d <= 3; ++d)
    if (off[d - 1] && freg.total(d).defined()) {
      freg.slot(d, "riemann").zero_();
      freg.post_slot(d, "riemann");
    }
  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  if (u.size(DIM1) > 1) {
    // Hydrostatic wall revision is a PHYSICAL-boundary operation: it copies
    // the interior side's pressure and density to the ghost side of the x1
    // wall face (_revise_x1*_lr below). On a domain decomposed in x1
    // (cubed nb1>1), a block's local il/iu at an internal seam is NOT a
    // physical boundary — the neighbor's data has already been exchanged
    // there — so applying the wall extrapolation would clobber the true
    // neighbor state. Gate each face on whether it is actually physical.
    // (Slab / single-rank: every block owns the whole x1 column, so both
    // faces are physical and behavior is unchanged — bit-identical.)
    bool grav1 = options->grav() && (options->grav()->grav1() != 0);
    bool phys_x1inner = pmb->options->is_physical_boundary(0, 0, -1);
    bool phys_x1outer = pmb->options->is_physical_boundary(0, 0, 1);

    // Well-balanced x1 reconstruction: decompose pressure and density into a
    // discretely hydrostatic reference + perturbation, reconstruct only the
    // perturbations, restore the reference at the faces. The restored faces
    // satisfy psf_lo(i) - psf_hi(i) = g*rho(i)*dx1f(i) identically, so a
    // resting stratification generates zero flux residual regardless of the
    // reconstruction. Engaged whenever gravity is on and the scheme is
    // defined: the state carries a pressure row, and the block owns the full
    // x1 column, OR it spans a vertical (nb1>1) decomposition -- in which case
    // _hydro_ref_x1 makes the reference continuous across the x1 block seams
    // with a distributed scan, so WB now engages under x1 decomposition too.
    bool wb_x1 =
        grav1 && w.size(0) > IPR && options->eos()->type() != "shallow-water";

    // SNAP_X1_CENTROID_EXACT (spherical-polar): a cell value is the r^2 dr
    // average, but the reconstruction, the hydrostatic scan and the reference
    // below are formulas for plain averages. They read the plain means instead
    // (x1_centroid.hpp, docs/derivations/x1-centroid-spherical.md); the
    // primitives themselves are not changed.
    torch::Tensor wx1 = w;
    if (x1_centroid_exact_enabled() &&
        pmb->pcoord->options->type() == "spherical-polar") {
      if (!x1pm_ || x1pm_->stale(w)) {
        x1pm_ = std::make_shared<X1PlainMeanStencils>(x1_plain_mean_stencils(
            pmb->pcoord->x1f, pmb->pcoord->il(), pmb->pcoord->iu(),
            phys_x1inner, phys_x1outer, w.options()));
      }
      wx1 =
          x1_plain_means(*x1pm_, w, IVX, !is_outflow(pmb->options->bfuncs()[0]),
                         !is_outflow(pmb->options->bfuncs()[1]));
      // seam ghosts: the neighbour's centred plain means, not this block's
      // off-centre ones
      Exchange(pmb).x1_rows({wx1}, pmb->pcoord->il(), false, kTagPlainMeans);
    }
    torch::Tensor wtmp;
    if (wb_x1) {
      auto base = wx1;
      torch::Tensor delta;
      if (split_flux) {
        base = wx1.clone();
        base[IPR].copy_(perturb.pressure0);
        base[IDN].copy_(perturb.density0);
        delta = torch::zeros_like(base);
        delta[IPR].copy_(perturb.pressure_delta);
        delta[IDN].copy_(perturb.density_delta);
      }
      auto ref = _hydro_ref_x1(base, delta);
      if (split_flux) {
        pressure_base1 = ref.face0;
        // The unclamped scan satisfies grad(p0)=g*rho0 analytically. Retain
        // the exceptional floor residual, rather than imposing rest balance.
        auto grad = face_gradient(pressure_base1, DIM1);
        auto floor = pressure_base1 <= std::numeric_limits<double>::min();
        auto clamped = floor | torch::roll(floor, {-1}, {-1});
        background_force1 = torch::where(
            clamped, options->grav()->grav1() * perturb.density0 - grad,
            torch::zeros_like(grad));
        if (pmb->policy.fp32()) pressure_base1 = pressure_base1 + ref.gauge;
        pressure_base1_before_seam = pressure_base1.clone();
      }

      WellBalancedX1 wb;
      wb.coord = pmb->pcoord.get();
      wb.policy = &pmb->policy;
      for (int f : {0, 1}) {
        wb.mirror[f] = (f ? phys_x1outer : phys_x1inner) &&
                       !is_outflow(pmb->options->bfuncs()[f]);
        wb.wall4[f] = wall4[f];
      }
      if (wall4[0] || wall4[1]) {
        wb.conv4 = conv4_.get();
        wb.conv4_trips = pmb->pefs->slot(EfStatus::kConvTrips);
      }
      wb.covariance = x1_mass_covariance();
      if (split_flux) {
        wb.pressure0 = perturb.pressure0;
        wb.pressure_delta = perturb.pressure_delta;
      }
      wtmp = wb.forward(wx1, ref, *precon1, monitor);
    } else {
      if (pmb->policy.fp32()) {
        wtmp = reconstruct_faces(*this, {DIM1, 0, options->disable_flux_x1()},
                                 w, split, pressure_base1);
      } else {
        wtmp = precon1->forward(wx1, DIM1);
        post_face_floors(*precon1, monitor);
      }
      if (grav1) {
        if (phys_x1inner) _revise_x1inner_lr(wtmp[ILT], wtmp[IRT]);
        if (phys_x1outer) _revise_x1outer_lr(wtmp[ILT], wtmp[IRT]);
      }
    }

    auto wlr1 =
        has_solid ? pmb->pib->forward(wtmp, DIM1, other.at("solid")) : wtmp;

    // Compute hydrostatic pressure correction
    if (options->grav() && (options->grav()->grav1() != 0) &&
        (options->grav()->non_hydrostatic() < 1.)) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      rho_grav.slice(2, is, ie) = (wlr1[ILT][IPR].slice(2, is + 1, ie + 1) -
                                   wlr1[IRT][IPR].slice(2, is, ie)) /
                                  pmb->pcoord->dx1f.slice(0, is, ie);
    }

    if (pressure_base1.defined()) {
      rho_grav_delta = rho_grav.clone();
      rho_grav += face_gradient(pressure_base1, DIM1);
    }

    // riemann solver
    if (!options->disable_flux_x1()) {
      auto face_pressure1 = options->eos()->type() == "shallow-water"
                                ? torch::Tensor()
                                : _face_pressure1;
      riemann_solve(*this, wlr1, DIM1, freg.slot(1, "riemann"), face_pressure1,
                    pressure_base1, split);
      freg.post_slot(1, "riemann");
      report_time("Flux-x1");
    }

    // sedimentation flux (P1); skipped when x1 flux is off (_flux1 not
    // rewritten). Its settling part is read wherever step 4.C can limit
    // species: with the eos limiter, or with the covariance rows that switch
    // it on (cov2, cov3)
    if (psed && !options->disable_flux_x1() && psed->settles()) {
      auto const& sed = psed->post(freg, fp32() ? w.to(torch::kFloat32) : w);
      if ((options->eos()->limiter() || flux_covariance()) && ny > 0)
        fsed1 = sed.narrow(0, ICY, ny);
    }

    // Make internal x1 seam fluxes single-valued (row 9). The two blocks
    // sharing an internal x1 face each compute its flux from their own
    // reconstruction; when the two are not bit-identical the shared face
    // carries two values, a spurious source at the seam. S moves the
    // seam-face slab and R's register sets both sides to the mean
    // (replace(1, faces, peer, SeamRule::mean), S step (4)): averaging is
    // symmetric, so both blocks hold bit-identical values and the conserved
    // sums telescope across the seam. On every x1 seam, process or same
    // process; nothing at nb1 = 1. The companions (face pressure,
    // pressure base) are not register rows: S averages them alongside.
    if (!options->disable_flux_x1() && playout && playout->options->nb(1) > 1) {
      Exchange ex(pmb);
      auto [below, above] = ex.x1_seam_peers(/*process_only=*/false);
      if (above >= 0 || below >= 0) {
        int il = pmb->pcoord->il();
        int iu = pmb->pcoord->iu();
        // R-4's peer rows: the total's, then each kept part's
        std::vector<torch::Tensor> regrows = {_flux1};
        for (auto const& part : freg.kept_parts(1))
          regrows.push_back(freg.kept(1, part));
        int nreg = 0;
        for (auto const& t : regrows) nreg += t.size(0);
        bool has_fp = _face_pressure1.defined() && _face_pressure1.numel() > 0;

        // ONE contiguous tensor per direction: some backends restrict
        // send/recv to a single tensor. torch::cat allocates fresh storage,
        // so the payload cannot alias the flux buffers while it travels.
        auto pack = [&](int face) {
          std::vector<torch::Tensor> rows;
          for (auto const& t : regrows) rows.push_back(t.select(-1, face));
          if (has_fp)
            rows.push_back(_face_pressure1.select(-1, face).unsqueeze(0));
          if (pressure_base1.defined())
            rows.push_back(pressure_base1.select(-1, face).unsqueeze(0));
          return torch::cat(rows, 0);
        };
        torch::Tensor up_mine, dn_mine;
        if (above >= 0) up_mine = pack(iu + 1);
        if (below >= 0) dn_mine = pack(il);
        auto [from_below, from_above] =
            ex.x1_seam_slab(dn_mine, up_mine, kTagSeamFlux, false);

        auto peer = torch::cat(regrows, 0);
        auto faces = torch::zeros(_flux1.sizes().slice(1),
                                  _flux1.options().dtype(torch::kBool));
        auto companions = [&](int face, torch::Tensor const& mine,
                              torch::Tensor const& theirs) {
          peer.select(-1, face).copy_(theirs.narrow(0, 0, nreg));
          faces.select(-1, face).fill_(true);
          int row = nreg;
          if (has_fp)
            _face_pressure1.select(-1, face).copy_(0.5 *
                                                   (mine[row] + theirs[row]));
          if (has_fp) ++row;
          if (pressure_base1.defined())
            pressure_base1.select(-1, face).copy_(0.5 *
                                                  (mine[row] + theirs[row]));
        };
        if (above >= 0) companions(iu + 1, up_mine, from_above);
        if (below >= 0) companions(il, dn_mine, from_below);
        freg.replace(1, faces, peer, SeamRule::mean);
      }
    }
    // reconcile, then boundary.prescribed on the x1 faces (R applies S's mask)
    if (!options->disable_flux_x1()) freg.apply_prescribed(1);

    // SNAP_X1_CENTROID_EXACT: the pressure source's six-face window reaches
    // two faces past a seam; they are the neighbour's
    if (wx1.data_ptr() != w.data_ptr() && _face_pressure1.defined() &&
        _face_pressure1.numel() > 0 &&
        options->eos()->type() != "shallow-water") {
      Exchange(pmb).x1_rows({_face_pressure1}, std::min(2, pmb->pcoord->il()),
                            true, kTagFacePressure);
    }

    // SNAP_X1_CENTROID_EXACT: the pressure force is the r^2-weighted
    // (A p*|)/V - (2/V) int r p~ dr (coord/spherical_polar.cpp), so the
    // hydrostatic correction is the same operator on the cell's own face
    // states; at rest (p_L = p_R = p*) the two cancel as the plain
    // differences do without the switch
    if (wx1.data_ptr() != w.data_ptr() && options->grav() &&
        options->grav()->grav1() != 0 &&
        options->grav()->non_hydrostatic() < 1. &&
        !options->disable_flux_x1() && _face_pressure1.defined() &&
        _face_pressure1.numel() > 0) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      if (!x1src_ || x1src_->stale(w)) {
        auto [below, above] = x1_neighbors();
        x1src_ = std::make_shared<X1PressureSourceStencils>(
            x1_pressure_source_stencils(pmb->pcoord->x1f, is, ie - 1, below < 0,
                                        above < 0, w.options()));
      }
      if (x1src_->usable) {
        auto area1 = pmb->pcoord->face_area1();
        auto volume = pmb->pcoord->cell_volume();
        rho_grav.slice(2, is, ie) =
            (area1.slice(-1, is + 1, ie + 1) *
                 wlr1[ILT][IPR].slice(2, is + 1, ie + 1) -
             area1.slice(-1, is, ie) * wlr1[IRT][IPR].slice(2, is, ie)) /
                volume.slice(-1, is, ie) -
            x1_pressure_source(*x1src_, _face_pressure1);
      }
    }
  }

  //// ------------ (3.A) Calculate dimension 2 LR states ------------ ////
  // conserve-axial-momentum: exact: lane A reads the solver's dissipative
  // x2/x3 mass flux M - M_central; the solve writes M_central = 1/2 (F(wL) +
  // F(wR)), riemann.central, in the kernel (C5)
  bool const c2 = paam->mode() == AxialMode::exact;
  torch::Tensor wtmp2, wtmp3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);
  std::vector<CommWorkPtr> works2, works3;
  Variables send_vars2, send_vars3;
  FaceDir const dir2{DIM2, kTagFaceX2, options->disable_flux_x2()};
  FaceDir const dir3{DIM3, kTagFaceX3, options->disable_flux_x3()};

  if (u.size(DIM2) > 1) {
    wtmp2 = reconstruct_faces(*this, dir2, w, split, pressure_base2);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->is_cubed_sphere()) {
      send_vars2["hydro_wl:+"] = wtmp2[ILT];
      send_vars2["hydro_wr:-"] = wtmp2[IRT];
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  //// ------------ (3.B) Calculate dimension 3 LR states ------------ ////
  if (u.size(DIM3) > 1) {
    wtmp3 = reconstruct_faces(*this, dir3, w, split, pressure_base3);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->is_cubed_sphere()) {
      send_vars3["hydro_wl:+"] = wtmp3[ILT];
      send_vars3["hydro_wr:-"] = wtmp3[IRT];
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->is_cubed_sphere()) {
    bool exchange_dim2 = u.size(DIM2) > 1;
    bool exchange_dim3 = u.size(DIM3) > 1;
    if (exchange_dim2) {
      pmb->launch_exchange(sync_opts.dim(DIM2), works2);
    }
    if (exchange_dim3) {
      pmb->launch_exchange(sync_opts.dim(DIM3), works3);
    }
    if (exchange_dim2) {
      pmb->finalize_exchange(send_vars2, sync_opts.dim(DIM2), works2);
    }
    if (exchange_dim3) {
      pmb->finalize_exchange(send_vars3, sync_opts.dim(DIM3), works3);
    }
  }

  //// ------------ (4.A) Calculate dimension 2 flux ------------ ////
  // #289: whether the x2 / x3 face fluxes carry the covariance correction; the
  // geometric pressure source then takes the same shifted pressure (step 5)
  bool cov2 = false, cov3 = false;
  bool has_scalars = pmb->pscalar && pmb->pscalar->nvar() > 0;
  // #44: the x1 differences of _flux_covariance read one row past an x1 seam;
  // there a split column's face states must be the neighbour's own (its
  // interior row), so a seam cell takes the stencil of one block. Physical x1
  // walls and an unsplit column keep their rows and the same tensor.
  // the left and right face states travel in one x1 message (S's merge M4,
  // per direction)
  auto x1_seam_rows = [&](torch::Tensor const& fl, torch::Tensor const& fr,
                          int tag) {
    Exchange ex(pmb);
    auto [below, above] = ex.x1_neighbors();
    if (below < 0 && above < 0) return std::make_pair(fl, fr);
    auto gl = fl.clone(), gr = fr.clone();
    ex.x1_rows({gl, gr}, 1, false, tag);
    return std::make_pair(gl, gr);
  };
  // the covariance increment of _flux_covariance into a face flux: the whole
  // flux on a curved grid, only the nonzero rows' x1 interior in Cartesian
  bool cov_rows = pmb->pcoord->face_centroid_shift_zero();
  auto add_cov = [&](torch::Tensor& flx, torch::Tensor const& dcov) {
    if (!cov_rows) {
      flx += dcov;
      return;
    }
    auto f = flx.narrow(-1, 1, flx.size(-1) - 2);
    f[IPR] += dcov[0];
    if (ny > 0) {
      f.narrow(0, ICY, ny) += dcov.narrow(0, 1, ny);
      f[IDN] += dcov[ny + 1];
    }
  };
  // its dry-mass row, shaped like one flux row, for the passive scalars
  auto cov_dry = [&](torch::Tensor const& dcov, torch::Tensor const& like) {
    if (!cov_rows) return dcov[IDN].clone();
    auto d = torch::zeros_like(like[IDN]);
    if (ny > 0) d.narrow(-1, 1, d.size(-1) - 2).copy_(dcov[ny + 1]);
    return d;
  };
  // the x2 / x3 sweep after reconstruction (solve_faces); #289: the
  // covariance is built from the face states BEFORE the solver may project
  // them into the face-local frame (its before_solve hook), added to the flux
  // after. On the x3 faces it is taken along x1 too, because x1 is the
  // stratified direction whichever horizontal face it crosses
  auto sweep = [&](FaceDir const& dir, torch::Tensor const& faces,
                   torch::Tensor& background, torch::Tensor& flux, int k) {
    torch::Tensor dcov;
    FaceHook covariance;
    if (flux_covariance()) {
      covariance = [&](torch::Tensor const& wlr, int dim) {
        auto [cwl, cwr] = x1_seam_rows(wlr[ILT], wlr[IRT], dir.seam_tag);
        dcov = _flux_covariance(cwl, cwr, dim);
        // the passive scalars' rows are built from the same face states
        // (scalar_flux_covariance), taken before the solver may project them
        if (dcov.defined() && has_scalars) {
          _cov_wl[k] = cwl.clone();
          _cov_wr[k] = cwr.clone();
          _cov_dry[k] = cov_dry(dcov, cwl);
        }
      };
    }
    solve_faces(*this, dir, faces, other, split, background, c2, covariance);
    if (dir.disabled) return false;
    if (dcov.defined()) add_cov(flux, dcov);
    report_time("Flux-x" + std::to_string(4 - dir.dim));
    return dcov.defined();
  };
  if (u.size(DIM2) > 1) cov2 = sweep(dir2, wtmp2, pressure_base2, _flux2, 0);

  //// ------------ (4.B) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) cov3 = sweep(dir3, wtmp3, pressure_base3, _flux3, 1);

  // S step (4), row 16r: the panel seams of the cubed sphere take the owning
  // panel's face flux (replace(d, faces, peer, SeamRule::owner_copy)), then
  // boundary.prescribed on the x2/x3 faces
  reconcile_panel_seams(freg, pmb, kTagPanelSeams);
  for (int d : {2, 3})
    if (freg.total(d).defined()) freg.apply_prescribed(d);

  //// ------- (4.C) Tracer flux positivity limiter (flux_positivity.hpp) ----
  ///////
  // Scale each face's species flux by the donor cell's theta so no cell can be
  // drained below zero by transport in this stage (covers advected species AND
  // the sedimentation flux added into _flux1 above). Runs after the x1 seam
  // averaging, so seam-face fluxes are already single-valued; theta's ghost
  // layer is then filled exactly like conserved-variable ghosts (exchange at
  // internal seams, boundary functions at physical faces) so the donor factor
  // of every shared face is identical on both ranks and conservation stays
  // exact. Positivity of the full multi-stage update follows from the SSP
  // structure of the integrators (see flux_positivity.hpp).
  if ((options->eos()->limiter() || cov2 || cov3) && ny > 0) {
    PositivityInputs in;
    in.u = u.narrow(0, ICY, ny);
    in.rho_dry = dry_density(u).unsqueeze(0);
    for (int d = 1; d <= 3; ++d) {
      auto t = d == 1 ? _flux1 : d == 2 ? _flux2 : _flux3;
      if (!t.defined()) continue;
      in.flux[d - 1] = t.narrow(0, ICY, ny);
      in.mass[d - 1] = t.narrow(0, IDN, 1);
    }
    in.fsed1 = fsed1;
    in.first = ICY;
    in.dt = dt;
    // the energy each species carries, from w; the withheld mass keeps it
    in.hspec = peos->species_enthalpy(w);
    if (in.hspec.defined()) {
      in.vel = u.narrow(0, IVX, 3) / w[IDN];
      // a lateral face's momentum flux is covariant in the face's own basis
      // (flux2global2_, 3_), and each block of a panel seam holds its own copy
      // of the face: carry each donor's velocity as a face state, synced
      // across the seam like them and taken to that basis, so both copies
      // withhold the same momentum
      if (playout && playout->options->is_cubed_sphere()) {
        std::vector<int> dims;
        if (_flux2.defined() && u.size(DIM2) > 1) dims.push_back(DIM2);
        if (_flux3.defined() && u.size(DIM3) > 1) dims.push_back(DIM3);
        std::vector<Variables> fv(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) {
          int n = w.size(dims[d]);
          auto lo = torch::zeros_like(w), hi = torch::zeros_like(w);
          lo.narrow(0, IVX, 3).narrow(dims[d], 1, n - 1) =
              w.narrow(0, IVX, 3).narrow(dims[d], 0, n - 1);
          hi.narrow(0, IVX, 3) = w.narrow(0, IVX, 3);
          fv[d]["hydro_vlo:+"] = lo;
          fv[d]["hydro_vhi:-"] = hi;
        }
        // a seam's x2 edge may be its neighbour's x3 edge: both directions
        // are in flight together, as for the face states
        SyncOptions fo;
        fo.cross_panel_only(true).interpolate(false).type(kPrimitive);
        std::vector<std::vector<CommWorkPtr>> works(dims.size());
        for (size_t d = 0; d < dims.size(); ++d)
          pmb->begin_exchange(fv[d], fo.dim(dims[d]));
        for (size_t d = 0; d < dims.size(); ++d)
          pmb->launch_exchange(fo.dim(dims[d]), works[d]);
        for (size_t d = 0; d < dims.size(); ++d) {
          pmb->finalize_exchange(fv[d], fo.dim(dims[d]), works[d]);
          for (auto const& t : {fv[d]["hydro_vlo:+"], fv[d]["hydro_vhi:-"]}) {
            if (dims[d] == DIM2) {
              pmb->pcoord->prim2local2_(t);
              pmb->pcoord->flux2global2_(t);
            } else {
              pmb->pcoord->prim2local3_(t);
              pmb->pcoord->flux2global3_(t);
            }
          }
          in.vface[dims[d] == DIM2 ? 0 : 1] =
              torch::stack({fv[d]["hydro_vlo:+"].narrow(0, IVX, 3),
                            fv[d]["hydro_vhi:-"].narrow(0, IVX, 3)});
        }
      }
    }
    PositivityLimiter(pmb, pmb->pefs.get(), EfStatus::kPosHits)
        .apply(freg, in, "hydro", "positivity.withhold");

    report_time("Positivity");
  }

  // R-5: limit ends; the Scalar module's dry-mass flux is row IDN of each
  // total now, before the implicit posts enter total(1)
  for (int d = 1; d <= 3; ++d)
    if (freg.keeps(d, "mass.dry.limited"))
      freg.keep(d, "mass.dry.limited", freg.total(d)[IDN]);

  //// ------------ (5) Calculate flux divergence ------------ ////
  // divergence starts with the sources' face-flux posts (ruling
  // A-ph3.diffusion_phase): after limit, so EF's scale never sees them, and
  // before the divergence of the totals below
  sources.post_fluxes(freg, {w, torch::Tensor(), dt});
  // R-5: the posts write the prescribed faces too (diffusion adds its IPR
  // row on every face); the divergence reads the totals next, so the faces
  // are prescribed again here, before any reader, and the removal is booked
  for (int d = 1; d <= 3; ++d)
    if (freg.total(d).defined()) freg.apply_prescribed(d);
  // #289: the normal-momentum face flux carries p* = p - delta d_1 p (its
  // centroid part), so the geometric pressure source must see the same p*, or
  // a hydrostatic rest state is no longer balanced (derivation 4A.3.2). Only
  // the lateral sources read the cell pressure here; the x1 source uses the
  // face pressure and is untouched. The shift is gated per direction: the x2
  // source (the IVY row) takes p* exactly when the x2 faces are corrected, the
  // x3 source (IVZ) exactly when the x3 faces are, so e.g. disable_flux_x3 with
  // nx3 > 1 keeps the x2 balance.
  bool any2 = u.size(DIM2) > 1, any3 = u.size(DIM3) > 1;
  // a zero centroid shift (Cartesian) leaves p* == p, the plain source
  if ((!cov2 && !cov3) || pmb->pcoord->face_centroid_shift_zero()) {
    _div.set_data(
        pmb->pcoord->forward(w, _flux1, _flux2, _flux3, _face_pressure1));
  } else {
    int n1 = w.size(-1);
    auto dsh = pmb->pcoord->face_centroid_shift_x1()
                   .to(w.device(), w.scalar_type())
                   .narrow(0, 1, n1 - 2);
    auto x1v = pmb->pcoord->x1v.to(w.device(), w.scalar_type());
    auto wsrc = w.clone();
    wsrc[IPR].narrow(-1, 1, n1 - 2) -=
        dsh * d1_pressure(w[IPR], x1v, pmb->pcoord->il(), pmb->pcoord->iu());
    auto div =
        pmb->pcoord->forward(wsrc, _flux1, _flux2, _flux3, _face_pressure1);
    // a resolved direction whose faces are NOT corrected keeps the plain p
    if ((any2 && !cov2) || (any3 && !cov3)) {
      auto plain =
          pmb->pcoord->forward(w, _flux1, _flux2, _flux3, _face_pressure1);
      if (any2 && !cov2) div[IVY].copy_(plain[IVY]);
      if (any3 && !cov3) div[IVZ].copy_(plain[IVZ]);
    }
    _div.set_data(div);
  }
  // axial angular momentum in flux form (#314): the rotation component of the
  // momentum rows from the same face fluxes, so it telescopes across faces
  bool aam = axial_flux_form();
  if (aam) {
    paam->flux_form(_div,
                    options->disable_flux_x1() || u.size(DIM1) == 1
                        ? torch::Tensor()
                        : _flux1,
                    options->disable_flux_x2() ? torch::Tensor() : _flux2,
                    options->disable_flux_x3() ? torch::Tensor() : _flux3);
  }
  report_time("Divergence");

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  // only a block carrying tracers needs the forcings' dry-density increment
  bool track_dry =
      pmb->pscalar && pmb->pscalar->nvar() > 0 && (pgrav || !sources.empty());
  auto dry_before = track_dry ? du[IDN].clone() : torch::Tensor();
  // x1 mass-flux w (step 8): the Coriolis rows that read rho w wait for the
  // stage's final x1 mass transfer, so Coriolis sees w = 0 here
  auto ptype = pmb->pcoord->options->type();
  bool mass_w = px1m->applies();
  auto wc1 = mass_w ? px1m->before_solve(w) : torch::Tensor();
  paam->arm(ny);  // exact: the torque of F_d, in this Coriolis call only
  auto forcing_du = du;
  if (fp32()) {
    // Stage-local contents in reusable storage. Keep accumulation in FP64 and
    // the explicit rounding to FP32 before addition to du unchanged.
    if (!_forcing_work.defined() || _forcing_work.sizes() != du.sizes() ||
        _forcing_work.device() != du.device()) {
      _forcing_work =
          torch::empty(du.sizes(), du.options().dtype(torch::kFloat64));
      _forcing_cast = torch::empty_like(du);
    }
    forcing_du = _forcing_work.zero_();
  }
  bool split_gravity =
      background_force1.defined() && !options->disable_flux_x1();
  if (split_gravity && pgrav)
    pgrav->forward_split(forcing_du, w, dt,
                         perturb.density_delta.to(w.options()));
  else if (pgrav)
    pgrav->forward(forcing_du, w, temp, dt);
  StageState stage{w, temp, dt, wc1};
  for (auto const& r : sources.runs(SourcePhase::pre_implicit))
    sources.run(r, forcing_du, stage);
  if (fp32()) {
    _forcing_cast.copy_(forcing_du);
    du += _forcing_cast;
  }
  paam->disarm();  // the step-8 Coriolis call must not apply it again
  if (split_flux) {
    auto force = torch::zeros_like(du);
    if (!options->disable_flux_x1() && pressure_base1.defined()) {
      if (split_gravity) {
        force[IVX] =
            background_force1 * options->grav()->non_hydrostatic() -
            face_gradient(pressure_base1 - pressure_base1_before_seam, DIM1);
      } else
        force[IVX] = -face_gradient(pressure_base1, DIM1);
    }
    if (pressure_base2.defined())
      force[IVY] = -face_gradient(pressure_base2, DIM2);
    if (pressure_base3.defined())
      force[IVZ] = -face_gradient(pressure_base3, DIM3);
    du.index(interior) += dt * force.index(interior);
  }
  _forcing_dry = track_dry ? (fp32() ? forcing_du[IDN].to(du.options())
                                     : du[IDN] - dry_before)
                           : torch::Tensor();

  // Face work uses the x1 mass flux after positivity limiting and
  // sedimentation. Implicit face mode consumes it before the solve; the
  // other modes add it after the solve.
  torch::Tensor gravity_energy_correction;
  if (options->grav() && options->grav()->grav1() != 0. && _flux1.defined() &&
      !options->disable_flux_x1()) {
    auto grav1 = options->grav()->grav1();
    auto non_hydrostatic = options->grav()->non_hydrostatic();
    auto vertical_mass_flux1 = _flux1[IDN].clone();
    if (ny > 0) {
      vertical_mass_flux1 += _flux1.narrow(0, ICY, ny).sum(0);
    }

    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    auto area1 = pmb->pcoord->face_area1();
    auto volume = pmb->pcoord->cell_volume();
    auto phi_cell = -grav1 * pmb->pcoord->x1v;
    // the x1 term of CoordinateImpl::divergence over the interior cells
    auto x1div = [&](torch::Tensor const& f) {
      return (area1.slice(-1, is + 1, ie + 1) * f.slice(-1, is + 1, ie + 1) -
              area1.slice(-1, is, ie) * f.slice(-1, is, ie)) /
             volume.slice(-1, is, ie);
    };
    auto vertical_mass_div = x1div(vertical_mass_flux1);
    torch::Tensor face_gravity_work;
    if (pmb->pcoord->options->type() == "cartesian") {
      auto face_mass = area1.narrow(-1, 0, vertical_mass_flux1.size(-1)) *
                       vertical_mass_flux1;
      face_gravity_work = dt * gravity_work_from_face_mass(
                                   face_mass, pmb->pcoord->x1f,
                                   pmb->pcoord->x1v, volume, is, ie, grav1);
    } else {
      auto phi_face = -grav1 * pmb->pcoord->x1f;
      auto potential_flux1 =
          vertical_mass_flux1 *
          phi_face.narrow(0, 0, vertical_mass_flux1.size(-1));
      auto potential_flux_div = x1div(potential_flux1);
      face_gravity_work = dt * (phi_cell.slice(0, is, ie) * vertical_mass_div -
                                potential_flux_div);
    }

    // gravity-work: face adds the work of the corrected potential energy for
    // this stage's x1 density change (derivation sec 7), where the grid has a
    // form for it
    bool radial_exact = radial_exact_work();
    if (radial_exact) {
      auto drho = -dt * vertical_mass_div;
      auto measure = x1_measure(pmb->pcoord->options->type());
      // limit: x1_neighbors() is {-1, -1} on a periodic x1 column, so its
      // block edges keep the one-sided slope; a gravitating periodic x1
      // column has no single-valued potential (-g1 x1 jumps at the periodic
      // face), so the corrected PE is not defined there in any case
      auto [below, above] = x1_neighbors();
      if (below < 0 && above < 0) {
        face_gravity_work +=
            corrected_pe_work(drho, pmb->pcoord->x1f, pmb->pcoord->x1v, is, ie,
                              grav1, measure, &_pe_work);
      } else {
        // x1 seam: the neighbour's drho one cell past it, so a seam cell
        // takes the centred slope of one block (#303)
        auto shape = drho.sizes().vec();
        shape.back() = pmb->pcoord->x1v.size(0);
        auto full = torch::zeros(shape, drho.options());
        full.slice(-1, is, ie).copy_(drho);
        Exchange(pmb).x1_rows({full}, 1, false, kTagGravityWork);
        face_gravity_work += corrected_pe_work_seam(
            full, pmb->pcoord->x1f, pmb->pcoord->x1v, is, ie, grav1, measure,
            below >= 0, above >= 0, &_pe_work);
      }
    }

    auto original_gravity_work = dt * w[IDN].slice(-1, is, ie) *
                                 w[IVX].slice(-1, is, ie) * grav1 *
                                 non_hydrostatic;
    if (non_hydrostatic < 1.) {
      original_gravity_work += dt * w[IVX].slice(-1, is, ie) *
                               rho_grav.slice(-1, is, ie) *
                               (1. - non_hydrostatic);
    }
    gravity_energy_correction = face_gravity_work - original_gravity_work;
  }

  // apply hydrostatic correction
  if (options->grav() && (options->grav()->non_hydrostatic() < 1.)) {
    du[IVX] += dt * (split_gravity ? rho_grav_delta : rho_grav) *
               (1. - options->grav()->non_hydrostatic());
    du[IPR] +=
        dt * w[IVX] * rho_grav * (1. - options->grav()->non_hydrostatic());
  }

  report_time("Forcing");

  //// ------------ (7) Perform implicit correction ------------ ////
  if (picorr) {
    // The implicit correction is NONLINEAR in dt, so the RK stage weight must
    // be applied INSIDE the solve, not to the solve's result:
    //   correct form   : delta = (I + b*dt*J')^-1 * (b*dt*L),  u += delta
    //   snapy (before)  : delta = (I +   dt*J')^-1 * (  dt*L),  u += b*delta
    // The two RHS scalings are equivalent because the solve is linear in its
    // RHS; the OPERATOR is not. At stage 1 of rk3, b = 1/4, so the I/dt
    // regularisation was 4x too weak.
    //
    // Scale ONLY the dt handed to the correction. du, the flux divergence and
    // the forcings stay at the full dt, or this becomes a different operator.
    // Applied for integrators of 3+ stages (rk3, rk3s4), matching the behaviour
    // already in production; generalising would change rk1/rk2 results.
    // V3 C4: dt_corr and the posts' dt are StepDt; in a captured stage they
    // are the block's device dt (scaled on the device, bitwise the host
    // product), so the solve reads no host dt and a dt change replays
    StepDt dt_corr = dt;
    if (pmb->pintg->stages.size() >= 3) {
      if (rk_stage >= 0 && rk_stage < pmb->pintg->stages.size()) {
        dt_corr = dt.scaled(pmb->pintg->stages[rk_stage].wght2());
      } else {
        // Loud, but not fatal: some callers drive HydroImpl::forward directly
        // without the stage loop (tests/test_forcing.cpp), and those must keep
        // working. A silent fallback here would restore the full-dt operator
        // this commit exists to remove, so say so.
        TORCH_WARN_ONCE(
            "[Hydro] rk_stage was not published before the implicit "
            "correction, so it is running with the FULL dt -- the operator "
            "this fix replaces. MeshBlockImpl::advance_local publishes it; a "
            "caller invoking HydroImpl::forward directly will see this.");
      }
    }
    // face work in the operator: the solve also sees the explicit face
    // correction, so the energy it inverts is the face-form energy
    if (gravity_energy_correction.defined()) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      du[IPR].slice(-1, is, ie) += gravity_energy_correction;
      gravity_energy_correction = torch::Tensor();
    }
    _apply_implicit_correction(du, w, dt_corr, other, dt);

    report_time("Implicit");
  }

  if (gravity_energy_correction.defined()) {
    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    du[IPR].slice(-1, is, ie) += gravity_energy_correction;
  }

  // R-5, post_implicit starts: S's prescribed x1 faces on the total and the
  // kept x1 parts, before any reader (a no-op until S sets a mask). After the
  // pre-divergence apply the total is written again by the implicit
  // correction (directly) or through the register's add, scale and replace,
  // which mark the direction. The apply runs when the implicit correction
  // ran or the mark is set; otherwise the faces are already prescribed
  // (main's reporting bitwise)
  if (freg.total(1).defined() && (picorr || freg.written_since_prescribed(1)))
    freg.apply_prescribed(1);

  //// ------------ (8) x1 mass-flux w in Coriolis and metric sources -------
  // -2 Omega w cos(lat) and -u w / r take rho w from the stage's final x1
  // face mass transfers: lane A's m1 (X1Momentum), built once here and read
  // by the post_implicit Coriolis and coord's x1 metric source
  if (mass_w) {
    auto pc = pmb->pcoord;
    auto const& wc2 = px1m->after_solve(_flux1, ny);
    auto m1 = px1m->m1();
    for (auto const& r : sources.runs(SourcePhase::post_implicit))
      sources.run(r, du, {w, temp, dt, wc2});
    if (!aam) {
      pc->x1_mass_source_(du, w, m1, dt);
    } else {
      // The explicit x1 faces are already in the flux form of step 5 (their
      // AAM flux carries r_f, which is -rho w u / r for the K component), so
      // the K component of this source only has to put the tangential
      // momentum the implicit solve moved between cells i and i+1 at the face
      // radius: with P the transfer through a face (k . p, per unit radius),
      //   r k . src V = (r_lo - r) P_lo - (r_hi - r) P_hi .
      // The component orthogonal to K keeps -m1 v / r.
      auto src = torch::zeros_like(du);
      pc->x1_mass_source_(src, w, m1, dt);
      paam->x1_source(src[IVY], src[IVZ],
                      picorr ? picorr->correction() : torch::Tensor());
      du.narrow(0, IVY, 2) += src.narrow(0, IVY, 2);
    }
  } else if (ptype == "gnomonic-equiangle") {
    // no x1 mass-flux w (nc1 == 1, shallow water): m1 = rho w. Only the
    // gnomonic grid has an x1_mass_source_; the others add nothing
    auto src = torch::zeros_like(du);
    pmb->pcoord->x1_mass_source_(src, w, w[IDN] * w[IVX], dt);
    // conserve-axial-momentum: no K component, as in the branch above
    if (aam) paam->x1_source(src[IVY], src[IVZ], torch::Tensor());
    du.index(interior) += src.index(interior);
  }

  return fp32() ? du.to(torch::kFloat32) : du;
}

}  // namespace snap
