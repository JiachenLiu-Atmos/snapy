// C/C++
#include <chrono>
#include <typeinfo>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "flux_positivity.hpp"
#include "gravity_work.hpp"
#include "gravity_work_radial.hpp"
#include "hydro.hpp"
#include "hydro_dispatch.hpp"
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
  bool centroid = !pcoord->face_centroid_shift_zero();
  auto ds = centroid ? pcoord->face_centroid_shift_x1()
                           .to(p.device(), p.scalar_type())
                           .unsqueeze(0)
                           .unsqueeze(1)
                     : torch::Tensor();
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
    auto rd = torch::ones_like(w[IDN]);
    for (int n = 0; n < ny; ++n) rd = rd - w[ICY + n];
    return rd;
  };
  // past an x1 seam wl/wr hold the neighbour's rows (x1_seam_rows in
  // forward()); the scalar's own face states take the same rows
  auto seam = [&](torch::Tensor const& f, int tag) {
    auto [below, above] = x1_neighbors();
    if (below < 0 && above < 0) return f;
    auto g = f.clone();
    _x1_ghost_rows(g, 1, false, tag);
    return g;
  };
  auto q = 0.5 * (seam(rl, 0x7740 + 4 * k) * dry_fraction(wl) +
                  seam(rr, 0x7742 + 4 * k) * dry_fraction(wr));

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
  bool centroid = !pcoord->face_centroid_shift_zero();
  auto ds = centroid ? pcoord->face_centroid_shift_x1()
                           .to(un.device(), un.scalar_type())
                           .unsqueeze(0)
                           .unsqueeze(1)
                     : torch::Tensor();
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

// dynamics/conversion-order 4: the x1 weights of L a = h^2/24 q''(x_i), with q
// the quadratic whose averages over cells i-1, i, i+1 are a. k = {kp, km}.
static void conv4_x1_weights(torch::Tensor x1f, int n,
                             torch::TensorOptions opts, torch::Tensor* k) {
  TORCH_CHECK(n >= 4 && x1f.size(0) == n + 1,
              "dynamics/conversion-order 4 needs nc1 >= 4 cells and nc1 + 1 "
              "x1 faces, got ",
              n, " and ", x1f.size(0));
  x1f = x1f.cpu().to(torch::kFloat64);
  auto f = [&](int s) { return x1f.narrow(0, s, n - 2); };
  auto c = (f(2) - f(1)).square() / 4. / (f(3) - f(0));
  k[0] = (c / (f(3) - f(1))).to(opts);
  k[1] = (c / (f(2) - f(0))).to(opts);
}

// conversion-order 4 at a reflecting x1 wall: ghost averages of the polynomial
// through the four owned averages next to the wall, plus q(0) = 0 (c = 1) or
// q'(0) = 0 (c = 2); E[side][c] maps those four (narrow order) to the ghosts
torch::Tensor wall4_weights(torch::Tensor x1f, int is, int iu, int ng) {
  x1f = x1f.to(torch::kCPU, torch::kFloat64);
  auto E = torch::zeros({2, 3, ng, 4}, torch::kFloat64);
  for (int s : {0, 1}) {
    int w = s ? iu + 1 : is, d = s ? -1 : 1;
    auto y = (x1f - x1f[w]) / (x1f[w + d] - x1f[w]);
    auto m = torch::arange(5, torch::kFloat64) + 1.;
    auto row = [&](int j) {
      return (y[j + 1].pow(m) - y[j].pow(m)) / m / (y[j + 1] - y[j]);
    };
    std::vector<torch::Tensor> a, b;
    for (int k = 0; k < 4; ++k) a.push_back(row(s ? iu - 3 + k : is + k));
    for (int k = 0; k < ng; ++k) b.push_back(row(s ? iu + 1 + k : is - ng + k));
    for (int c : {0, 1, 2}) {
      int n = 4 + (c > 0);
      auto A = torch::zeros({n, n}, torch::kFloat64);
      A.narrow(0, 0, 4) = torch::stack(a).narrow(1, 0, n);
      if (c) A[4][c - 1] = 1.;
      E[s][c] =
          torch::stack(b).narrow(1, 0, n).matmul(A.inverse()).narrow(1, 0, 4);
    }
  }
  return E;
}

torch::Tensor wall4_fill(torch::Tensor a, torch::Tensor const& E, int c, int i0,
                         int i1, bool const* wall, bool pos,
                         torch::Tensor ref) {
  auto trips = torch::zeros({}, a.options().dtype(torch::kLong));
  for (int s : {0, 1}) {
    if (!wall[s]) continue;
    int g = E.size(2);
    auto old = a.narrow(-1, s ? i1 + 1 : i0 - g, g);
    auto x = a.narrow(-1, s ? i1 - 3 : i0, 4).matmul(E[s][c].t());
    auto t = ref.defined() ? x + ref.narrow(-1, s ? i1 + 1 : i0 - g, g) : x;
    auto bad = pos ? t <= 0. : torch::zeros_like(t, torch::kBool);
    trips += bad.sum();
    old.copy_(torch::where(bad, old, x));
  }
  return trips;
}

torch::Tensor conv4_admissible(torch::Tensor const& a, bool prim,
                               EquationOfStateImpl* eos, bool limiter) {
  double df = limiter ? eos->options->density_floor() : 0.;
  double pf = limiter ? eos->options->pressure_floor() : 0.;
  auto good = torch::isfinite(a).all(0) & (a[IDN] > df);
  if (limiter && a.size(0) > ICY)
    good &= (a.narrow(0, ICY, a.size(0) - ICY) >= 0.).all(0);
  return prim && a.size(0) > IPR ? good & (a[IPR] > pf) : good;
}

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  auto wide = u.is_mps() ? torch::kFloat32 : torch::kFloat64;  // MPS: no f64
  bool has_solid = other.count("solid");
  auto start = std::chrono::high_resolution_clock::now();
  _check_stretched_x1(u);

  auto playout = pmb->get_layout();
  for (int k = 0; k < 2; ++k) {
    _cov_wl[k] = _cov_wr[k] = _cov_dry[k] = torch::Tensor();
  }

  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  // conversion-order 4: x1 ghosts at a reflecting wall from the owned cells
  // (wall4_weights, constraint c); a rho or p ghost whose total (ref + a) is
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
  auto fill4 = [&](torch::Tensor a, int c, bool pos, torch::Tensor ref) {
    if (wall4[0] || wall4[1])  // key 2 and wall-less blocks never enter
      _conv4_trips += wall4_fill(a, wall4_E_, c, i0, i1, wall4, pos, ref);
  };
  // x1 point values u - L u, the primitives there, and back to cell averages:
  // w = W(u - L u) + L W(u), fourth order where W(u) is second
  if (options->conversion_order() == 4 && u.size(DIM1) > 1) {
    auto const& mo = pmb->options;
    int ng = pmb->pcoord->options->nghost();
    auto* k = conv4_x1_;
    auto const& x1f = pmb->pcoord->x1f;  // rebuilt on a new or rewritten grid
    std::pair<void*, int64_t> at{x1f.data_ptr(), x1f._version()};
    if (!k[0].defined() || at != conv4_x1_at_ || k[0].device() != u.device() ||
        k[0].scalar_type() != u.scalar_type()) {
      conv4_x1_weights(x1f, u.size(DIM1), u.options(), k), conv4_x1_at_ = at;
      wall4_E_ = wall4_weights(x1f, i0, i1, ng).to(u.options());
    }
    int n = u.size(DIM1) - 2;
    // centred everywhere, reading the u ghosts the boundary condition or the
    // exchange filled; the converted ghosts are refilled below
    auto lap = [&](torch::Tensor const& a) {
      auto out = torch::zeros_like(a);
      auto am = a.narrow(-1, 0, n), a0 = a.narrow(-1, 1, n);
      out.narrow(-1, 1, n) =
          k[0] * (a.narrow(-1, 2, n) - a0) - k[1] * (a0 - am);
      return out;
    };
    // per cell, keep the second-order w where u - L u or its primitives are
    // not admissible: non-finite, rho or p not above the floor (zero with the
    // limiter off), or repaired by the limiter. Those cells hand u to the
    // EOS; this call's redo marks and vapor failures are dropped. Trips count
    // cells per stage, ghosts included
    bool lim = peos->options->limiter();
    auto ok = [&](torch::Tensor const& a, bool p) {
      return conv4_admissible(a, p, peos.get(), lim);
    };
    auto w0 = w.clone();
    // Conversion acts on total cell averages, including the static background.
    // Differencing only stored anomalies omits the background curvature.
    auto averages = has_reference() ? total_u(u) : u;
    // the wall cell's row reads one-sided ghosts: m(0) = 0; W(u) has no
    // wall condition (m/rho = v + O(h^2), not zero at the wall)
    if (wall4[0] || wall4[1]) {
      averages = averages.clone();
      for (int c = 0; c < u.size(0); ++c) {
        fill4(averages[c], c == IVX, c == IDN || c == IPR, {});
        fill4(w0[c], 0, c == IDN || c == IPR, {});
      }
    }
    auto up = averages - lap(averages);
    auto good = ok(up, false);
    auto tmp = torch::where(good, up, averages);
    auto keep = [](torch::Tensor const& a) {
      return a.defined() ? a.clone() : a;
    };
    auto marks = keep(peos->limiter_marks());
    auto fails = keep(peos->vapor_failures());
    auto tmp0 = lim ? tmp.clone() : torch::Tensor();
    peos->compute("U->W", {tmp, w});
    if (lim) {
      good &= (tmp == tmp0).all(0);
      if (marks.defined()) peos->limiter_marks().copy_(marks);
      if (peos->vapor_failures().defined())
        fails.defined() ? peos->vapor_failures().copy_(fails)
                        : peos->vapor_failures().zero_();
    }
    good &= ok(w, true);
    w.add_(lap(w0));
    w.copy_(torch::where(good, w, w0));
    _conv4_trips += good.logical_not().sum();
    // every ghost from the converted owned cells: the block's exchange, the
    // x1 rows again across all x2/x3 columns (the corners past a seam), then
    // the boundary conditions on every ghost layer (outflow from the saved
    // background, as the block's own outflow)
    Variables gv{{"hydro_w", w}};
    pmb->exchange(gv, SyncOptions().interpolate(true).type(kPrimitive));
    _x1_ghost_rows(w, ng, false, 0x772a);
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
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "EOS time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  if (has_solid) {
    pmb->pib->mark_prim_solid_(w, other.at("solid"));
  }

  // hydrostatic pressure correction
  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);
  int ny = u.size(0) - ICY;
  // the settling part of the x1 species flux, for the positivity carry
  torch::Tensor fsed1;
  // gravity-work: cell -- the Riemann (background) x1 mass flux F^R; the mass
  // that sedimentation and the positivity limiter add (F - F^R) keeps its
  // face-form gravity work
  torch::Tensor bflux1;
  bool gw_cell = options->grav() && options->grav()->grav1() != 0. &&
                 options->grav()->gravity_work() == "cell";

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  if (u.size(DIM1) > 1) {
    // Hydrostatic wall revision is a PHYSICAL-boundary operation: it rebuilds
    // the x1 boundary in isentropic balance. On a domain decomposed in x1
    // (cubed nb1>1), a block's local il/iu at an internal seam is NOT a
    // physical boundary — the neighbor's data has already been exchanged
    // there — so applying the wall extrapolation would clobber the true
    // neighbor state. Gate each face on whether it is actually physical.
    // (Slab / single-rank: every block owns the whole x1 column, so both
    // faces are physical and behavior is unchanged — bit-identical.)
    bool grav1 = options->grav() && (options->grav()->grav1() != 0);
    bool phys_x1inner = pmb->options->is_physical_boundary(0, 0, -1);
    bool phys_x1outer = pmb->options->is_physical_boundary(0, 0, 1);

    /*if (grav1) {
      if (phys_x1inner) _revise_x1inner_ghost(w);
      if (phys_x1outer) _revise_x1outer_ghost(w);
    }*/

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
      _x1_ghost_rows(wx1, pmb->pcoord->il(), false, 0x7724);
    }
    torch::Tensor wtmp;
    if (wb_x1) {
      auto [psf_lo, pref, dsf, dref] = _hydro_ref_x1(wx1);
      auto pressure = wx1[IPR].clone();
      auto density = wx1[IDN].clone();

      wx1[IPR] -= pref;
      wx1[IDN] -= dref;

      // Even-parity ghost perturbations at the walls: p'(is-m) =
      // p'(is+m-1), rho' likewise. Only apply this at physical x1 walls.
      int ng = pmb->pcoord->options->nghost();
      int is = pmb->pcoord->il();
      int iu = pmb->pcoord->iu();
      for (int c : {(int)IPR, (int)IDN}) {
        if (phys_x1inner && !is_outflow(pmb->options->bfuncs()[0])) {
          wx1[c]
              .narrow(-1, is - ng, ng)
              .copy_(wx1[c].narrow(-1, is, ng).flip(-1));
        }

        if (phys_x1outer && !is_outflow(pmb->options->bfuncs()[1])) {
          wx1[c]
              .narrow(-1, iu + 1, ng)
              .copy_(wx1[c].narrow(-1, iu + 1 - ng, ng).flip(-1));
        }
        // key 4 one-sided rho', p', v ghosts: this path runs only with gravity
        fill4(wx1[c], c == IPR ? 2 : 0, true, c == IPR ? pref : dref);
      }

      torch::Tensor velocity;
      if (x1_mass_covariance()) {
        velocity = wx1[IVX].clone();
        auto x1v = pmb->pcoord->x1v.to(w.device(), w.scalar_type());
        auto dx1f = pmb->pcoord->dx1f.to(w.device(), w.scalar_type());
        bool stretched = pmb->pcoord->options->stretched_x1();
        auto rho_1 = x1_cov_d1_centred(density, x1v, stretched);
        // next to a reflecting wall the even ghost density is not the
        // stratified column: one-sided second order in the first/last cell
        if (phys_x1inner && !is_outflow(pmb->options->bfuncs()[0]))
          rho_1.select(-1, is).copy_(
              x1_cov_one_sided(density, x1v, is, 1, stretched));
        if (phys_x1outer && !is_outflow(pmb->options->bfuncs()[1]))
          rho_1.select(-1, iu).copy_(
              x1_cov_one_sided(density, x1v, iu, -1, stretched));
        auto w_1 = x1_cov_d1_centred(velocity, x1v, stretched);
        wx1[IVX] -= dx1f * dx1f / 12. * rho_1 * w_1 / density;
        // ghosts at reflecting walls: odd mirror of the corrected interior
        if (phys_x1inner && !is_outflow(pmb->options->bfuncs()[0]))
          wx1[IVX]
              .narrow(-1, is - ng, ng)
              .copy_(-wx1[IVX].narrow(-1, is, ng).flip(-1));
        if (phys_x1outer && !is_outflow(pmb->options->bfuncs()[1]))
          wx1[IVX]
              .narrow(-1, iu + 1, ng)
              .copy_(-wx1[IVX].narrow(-1, iu + 1 - ng, ng).flip(-1));
      }
      if (wall4[0] || wall4[1]) {
        if (!velocity.defined()) velocity = wx1[IVX].clone();
        fill4(wx1[IVX], 1, false, {});
      }

      // floor=false: reconstruction-stage floors would clamp legitimately
      // negative perturbations.
      wtmp = precon1->forward(wx1, DIM1, /*floor=*/false);

      wx1[IPR].copy_(pressure);
      wx1[IDN].copy_(density);
      if (velocity.defined()) wx1[IVX].copy_(velocity);
      // Restore full face pressure/density; floor any nonlinear-WENO overshoot
      // that would go non-positive (the references are tiny near the top) back
      // to the reference. At rest the perturbation is ~0 so the floor never
      // fires and well-balancing is preserved.
      auto pl = wtmp[ILT][IPR] + psf_lo;
      auto pr = wtmp[IRT][IPR] + psf_lo;
      wtmp[ILT][IPR].copy_(torch::where(pl > 0., pl, psf_lo));
      wtmp[IRT][IPR].copy_(torch::where(pr > 0., pr, psf_lo));

      auto dl = wtmp[ILT][IDN] + dsf[ILT];
      auto dr = wtmp[IRT][IDN] + dsf[IRT];
      // Positivity fallback = the adjacent cell's density, not dsf: a
      // reference that overestimates density aloft (a bottom-anchored
      // isentrope did, by orders of magnitude) turns every floor event at a
      // reflecting top wall into a large spurious wall impedance.
      // Shift by one along x1 with EDGE REPLICATION, not torch::roll: roll is
      // circular, so at index 0 it would substitute the density from the TOP
      // of the column. That word is not consumed today (physical faces run
      // il..iu+1 with il = nghost), but a wraparound inside a positivity
      // fallback becomes live the moment a caller changes the range.
      auto n1 = density.size(-1);
      auto rho_below = torch::cat(
          {density.narrow(-1, 0, 1), density.narrow(-1, 0, n1 - 1)}, -1);
      wtmp[ILT][IDN].copy_(torch::where(dl > 0., dl, rho_below));
      wtmp[IRT][IDN].copy_(torch::where(dr > 0., dr, density));
      // key 4, after the floors: the wall face mirrors its owned side, v* = 0
      for (int s : {0, 1}) {
        if (!wall4[s]) continue;
        int i = s ? iu + 1 : is, o = s ? ILT : IRT;
        wtmp[1 - o].select(-1, i).copy_(wtmp[o].select(-1, i));
        wtmp[1 - o][IVX].select(-1, i).neg_();
      }
    } else {
      wtmp = precon1->forward(wx1, DIM1);
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

    // riemann solver
    if (!options->disable_flux_x1()) {
      auto face_pressure1 = options->eos()->type() == "shallow-water"
                                ? torch::Tensor()
                                : _face_pressure1;
      priemann->forward(wlr1[ILT], wlr1[IRT], DIM1, _flux1, face_pressure1);
      if (gw_cell) {
        bflux1 = _flux1[IDN].clone();
        if (ny > 0) bflux1 += _flux1.narrow(0, ICY, ny).sum(0);
      }
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x1 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    // sedimentation flux; skipped when x1 flux is off (_flux1 not rewritten).
    // Its settling part is kept wherever step 4.C can limit species: with the
    // eos limiter, or with the covariance rows that switch it on (cov2, cov3)
    if (psed && !options->disable_flux_x1()) {
      bool carry = (options->eos()->limiter() || flux_covariance()) && ny > 0;
      auto fadv = carry ? _flux1.narrow(0, ICY, ny).clone() : torch::Tensor();
      psed->forward(w, _flux1);
      if (carry) fsed1 = _flux1.narrow(0, ICY, ny) - fadv;
    }

    // Make internal x1 seam fluxes single-valued. The two ranks sharing an
    // internal x1 face each compute the face flux from their own
    // reconstruction. When the two are not bit-identical (e.g. a reference
    // rebuilt per block, as the well-balanced reconstruction does once it
    // spans seams), the shared face carries two different flux values --
    // a spurious source/sink of mass, momentum, and energy at the seam.
    // Exchange the seam-face slab with the x1 neighbors and set both sides
    // to the average: averaging is symmetric, so both ranks hold
    // bit-identical values and the conserved sums telescope exactly across
    // the seam, independent of reconstruction details (same principle as
    // flux correction at mesh-refinement boundaries). The scalar advective
    // flux upwinds by this mass flux afterwards and inherits the property.
    // No exchange, no behavior change at nb1 = 1. Across process seams only,
    // as with one block per process; a same-process seam is not averaged.
    if (!options->disable_flux_x1() && playout &&
        playout->has_process_group() && playout->options->pz() > 1) {
      auto iloc = playout->loc_of(playout->options->rank());
      int above = playout->neighbor_rank(iloc, {0, 0, 1});
      int below = playout->neighbor_rank(iloc, {0, 0, -1});
      if (above >= 0 && playout->is_local_block(above)) above = -1;
      if (below >= 0 && playout->is_local_block(below)) below = -1;
      if (above >= 0 || below >= 0) {
        constexpr int kSeamFluxUpTag = 0x7720;  // payload travels upward
        constexpr int kSeamFluxDnTag = 0x7721;  // payload travels downward
        int il = pmb->pcoord->il();
        int iu = pmb->pcoord->iu();
        int nv = _flux1.size(0);
        bool has_fp = _face_pressure1.defined() && _face_pressure1.numel() > 0;

        // Pack the flux slab and the face pressure into ONE contiguous tensor
        // per direction: some backends restrict send/recv to a single tensor,
        // and one message per seam is cheaper anyway. torch::cat allocates
        // fresh storage, so the payload cannot alias the flux buffers while a
        // send is in flight.
        bool has_bf = bflux1.defined();  // F^R is averaged with the flux
        auto pack = [&](int face) {
          std::vector<torch::Tensor> rows = {_flux1.select(-1, face)};
          if (has_fp)
            rows.push_back(_face_pressure1.select(-1, face).unsqueeze(0));
          if (has_bf) rows.push_back(bflux1.select(-1, face).unsqueeze(0));
          if (rows.size() == 1) return rows[0].clone();
          return torch::cat(rows, 0);
        };
        auto unpack = [&](int face, torch::Tensor const& avg) {
          _flux1.select(-1, face).copy_(avg.narrow(0, 0, nv));
          int row = nv;
          if (has_fp) _face_pressure1.select(-1, face).copy_(avg[row++]);
          if (has_bf) bflux1.select(-1, face).copy_(avg[row]);
        };

        std::vector<CommWorkPtr> seam_sends;
        std::vector<torch::Tensor> up_mine, dn_mine;
        if (above >= 0) {
          up_mine = {pack(iu + 1)};
          seam_sends.push_back(
              playout->send_to_block(up_mine, above, kSeamFluxUpTag));
        }
        if (below >= 0) {
          dn_mine = {pack(il)};
          seam_sends.push_back(
              playout->send_to_block(dn_mine, below, kSeamFluxDnTag));
        }
        if (above >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(up_mine[0])};
          playout->recv_from_block(theirs, above, kSeamFluxDnTag)->wait();
          unpack(iu + 1, 0.5 * (up_mine[0] + theirs[0]));
        }
        if (below >= 0) {
          std::vector<torch::Tensor> theirs = {torch::empty_like(dn_mine[0])};
          playout->recv_from_block(theirs, below, kSeamFluxUpTag)->wait();
          unpack(il, 0.5 * (dn_mine[0] + theirs[0]));
        }
        for (auto& sw : seam_sends) sw->wait();
      }
    }

    // SNAP_X1_CENTROID_EXACT: the pressure source's six-face window reaches
    // two faces past a seam; they are the neighbour's
    if (wx1.data_ptr() != w.data_ptr() && _face_pressure1.defined() &&
        _face_pressure1.numel() > 0 &&
        options->eos()->type() != "shallow-water") {
      _x1_ghost_rows(_face_pressure1, std::min(2, pmb->pcoord->il()), true,
                     0x7722);
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
  torch::Tensor wtmp2, wtmp3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);
  std::vector<CommWorkPtr> works2, works3;
  Variables send_vars2, send_vars3;

  if (u.size(DIM2) > 1) {
    wtmp2 = precon23->forward(w, DIM2);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars2["hydro_wl:+"] = wtmp2[ILT];
      send_vars2["hydro_wr:-"] = wtmp2[IRT];
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  //// ------------ (3.B) Calculate dimension 3 LR states ------------ ////
  if (u.size(DIM3) > 1) {
    wtmp3 = precon23->forward(w, DIM3);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars3["hydro_wl:+"] = wtmp3[ILT];
      send_vars3["hydro_wr:-"] = wtmp3[IRT];
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->type() == "cubed-sphere") {
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
  auto x1_seam_rows = [&](torch::Tensor const& f, int tag) {
    auto [below, above] = x1_neighbors();
    if (below < 0 && above < 0) return f;
    auto g = f.clone();
    _x1_ghost_rows(g, 1, false, tag);
    return g;
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
  if (u.size(DIM2) > 1) {
    auto wlr2 =
        has_solid ? pmb->pib->forward(wtmp2, DIM2, other.at("solid")) : wtmp2;
    if (!options->disable_flux_x2()) {
      // #289: built from the face states BEFORE the solver may project them
      // into the face-local frame, added to the flux after
      torch::Tensor cwl, cwr;
      if (flux_covariance()) {
        cwl = x1_seam_rows(wlr2[ILT], 0x7730);
        cwr = x1_seam_rows(wlr2[IRT], 0x7732);
      }
      auto dcov = flux_covariance() ? _flux_covariance(cwl, cwr, DIM2)
                                    : torch::Tensor();
      // the passive scalars' rows are built from the same face states
      // (scalar_flux_covariance), taken before the solver may project them
      if (dcov.defined() && has_scalars) {
        _cov_wl[0] = cwl.clone();
        _cov_wr[0] = cwr.clone();
        _cov_dry[0] = cov_dry(dcov, cwl);
      }
      priemann->forward(wlr2[ILT], wlr2[IRT], DIM2, _flux2);
      if (dcov.defined()) add_cov(_flux2, dcov);
      cov2 = dcov.defined();
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x2 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (4.B) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wlr3 =
        has_solid ? pmb->pib->forward(wtmp3, DIM3, other.at("solid")) : wtmp3;
    if (!options->disable_flux_x3()) {
      // #289: the same term on the x3 faces -- the covariance is taken along
      // x1 there too, because x1 is the stratified direction whichever
      // horizontal face it crosses
      torch::Tensor cwl, cwr;
      if (flux_covariance()) {
        cwl = x1_seam_rows(wlr3[ILT], 0x7734);
        cwr = x1_seam_rows(wlr3[IRT], 0x7736);
      }
      auto dcov = flux_covariance() ? _flux_covariance(cwl, cwr, DIM3)
                                    : torch::Tensor();
      if (dcov.defined() && has_scalars) {
        _cov_wl[1] = cwl.clone();
        _cov_wr[1] = cwr.clone();
        _cov_dry[1] = cov_dry(dcov, cwl);
      }
      priemann->forward(wlr3[ILT], wlr3[IRT], DIM3, _flux3);
      if (dcov.defined()) add_cov(_flux3, dcov);
      cov3 = dcov.defined();
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x3 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

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
    auto uy = u.narrow(0, ICY, ny);
    auto f1 = _flux1.defined() ? _flux1.narrow(0, ICY, ny) : torch::Tensor();
    auto f2 = _flux2.defined() ? _flux2.narrow(0, ICY, ny) : torch::Tensor();
    auto f3 = _flux3.defined() ? _flux3.narrow(0, ICY, ny) : torch::Tensor();

    torch::Tensor drain;
    auto theta = flux_positivity_theta(uy, f1, f2, f3, pmb->pcoord, dt, &drain);
    // the eos limiter does not limit the mass flux either: where it drains
    // the dry air too (Courant>1), the neighbours (implicit x1 refill, inflow
    // along x2, x3) bring their q back, so a species may drain past its
    // content only as far as the poorer neighbour repays: all of it at a
    // uniform q, none beside an empty cell
    {
      auto m = [&](torch::Tensor const& f) {
        return f.defined() ? f.narrow(0, IDN, 1) : torch::Tensor();
      };
      auto rho = dry_density(u).unsqueeze(0);
      auto td = flux_positivity_theta(rho, m(_flux1), m(_flux2), m(_flux3),
                                      pmb->pcoord, dt);
      auto phi = flux_positivity_share(uy / rho);
      // settling mass is not dry air: nothing refills it, so a cell it drains
      // keeps the plain theta
      if (fsed1.defined()) {
        auto out = fsed1.clamp_min(0.).roll(-1, DIM1) + (-fsed1).clamp_min(0.);
        phi.masked_fill_(out > 0., 0.);
      }
      theta = flux_positivity_relax(theta, td, phi);
    }
    // census of interior (cell, species) entries, before the ghost fill
    auto cells = pmb->part({0, 0, 0}, PartOptions().exterior(false));
    auto ti = theta.index(cells).to(wide);
    _positivity_hits += (ti < 1.).sum();
    // severe: theta < 0.9 AND the withheld mass is above round-off of the
    // cell's gas mass. An empty cell drained by a round-off face flux has
    // theta = 0 at any dt; its withheld mass is ~1e-20 of the cell's (#256).
    // the cell volume depends on the grid only
    auto& vc = _cell_vol;
    auto pc = pmb->pcoord;
    if (!(vc.f1.defined() && vc.f1.is_same(pc->x1f) && vc.f2.is_same(pc->x2f) &&
          vc.f3.is_same(pc->x3f) && vc.v1 == pc->x1f._version() &&
          vc.v2 == pc->x2f._version() && vc.v3 == pc->x3f._version() &&
          vc.val.defined() && vc.val.device() == u.device() &&
          vc.val.scalar_type() == u.scalar_type())) {
      vc.val = pc->cell_volume();
      vc.f1 = pc->x1f;
      vc.f2 = pc->x2f;
      vc.f3 = pc->x3f;
      vc.v1 = pc->x1f._version();
      vc.v2 = pc->x2f._version();
      vc.v3 = pc->x3f._version();
    }
    auto cell_mass = (dry_density(u) + uy.sum(0)) * vc.val;
    auto withheld = ((1. - theta) * drain).index(cells).to(wide);
    auto above = withheld > positivity_roundoff(u.scalar_type()) *
                                cell_mass.unsqueeze(0).index(cells).to(wide);
    _positivity_severe += ((ti < 0.9) & above).sum();
    torch::minimum_out(_positivity_min, _positivity_min, ti.min());

    // the energy each species carries, from w, ghosts included: exchanging it
    // with theta changed no bit, so it is not exchanged (#238)
    auto hspec = peos->species_enthalpy(w);
    flux_positivity_sync_theta_(pmb, theta, "hydro_theta");

    // theta's depth is not its consequence: measure the flux it removes
    auto f1_pre = f1.defined() ? f1.abs() : torch::Tensor();

    // the withheld species mass keeps its energy and momentum in the donor
    if (hspec.defined()) {
      flux_positivity_carry_(theta, hspec, u.narrow(0, IVX, 3) / w[IDN], _flux1,
                             _flux2, _flux3, pmb->pcoord, fsed1);
    }
    flux_positivity_scale_(theta, f1, f2, f3, pmb->pcoord);

    if (f1_pre.defined()) {
      // non-negative while every bfunc keeps the ghost theta in [0,1]
      auto cut = f1_pre - f1.abs();
      _lim_flux += f1_pre.index(cells).sum().to(wide);
      _lim_cut += cut.index(cells).sum().to(wide);
    }

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(Hydro) << "Positivity time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  //// ------------ (5) Calculate flux divergence ------------ ////
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
    _div.set_(pmb->pcoord->forward(w, _flux1, _flux2, _flux3, _face_pressure1));
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
    _div.set_(div);
  }
  // axial angular momentum in flux form (#314): the rotation component of the
  // momentum rows from the same face fluxes, so it telescopes across faces
  bool aam = axial_flux_form();
  if (aam) {
    pmb->pcoord->conserve_axial_momentum_(
        _div,
        options->disable_flux_x1() || u.size(DIM1) == 1 ? torch::Tensor()
                                                        : _flux1,
        options->disable_flux_x2() ? torch::Tensor() : _flux2,
        options->disable_flux_x3() ? torch::Tensor() : _flux3);
  }
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Divergence time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  // only a block carrying tracers needs the forcings' dry-density increment
  bool track_dry =
      pmb->pscalar && pmb->pscalar->nvar() > 0 && !forcings.empty();
  auto dry_before = track_dry ? du[IDN].clone() : torch::Tensor();
  // x1 mass-flux w (step 8): the Coriolis rows that read rho w wait for the
  // stage's final x1 mass transfer, so Coriolis sees w = 0 here
  auto ptype = pmb->pcoord->options->type();
  bool mass_w = peos->options->type() != "shallow-water" &&
                pmb->pcoord->options->nc1() > 1 &&
                (ptype == "gnomonic-equiangle" ||
                 (ptype == "spherical-polar" && options->coriolis()));
  auto coriolis = [](torch::nn::AnyModule& f) {
    return f.type_info() == typeid(CoriolisXYZImpl) ||
           f.type_info() == typeid(Coriolis123Impl);
  };
  auto& mw = _mass_w;
  if (mass_w) {
    auto pc = pmb->pcoord;
    auto shape = w[IDN].sizes().vec();
    if (!(mw.x1f.defined() && mw.x1f.is_same(pc->x1f) &&
          mw.ver == pc->x1f._version() && mw.wc1.device() == w.device() &&
          mw.wc1.scalar_type() == w.scalar_type() &&
          mw.wc1[IDN].sizes() == c10::IntArrayRef(shape))) {
      int is = pc->il(), ie = pc->iu() + 1;
      auto r = pc->x1v.slice(0, is, ie);
      mw.x1f = pc->x1f;
      mw.ver = pc->x1f._version();
      mw.area = pc->face_area1();
      auto den = 2. * r * pc->cell_volume().slice(-1, is, ie);
      mw.lo = (r.square() - pc->x1f.slice(0, is, ie).square()) / den;
      mw.hi = (pc->x1f.slice(0, is + 1, ie + 1).square() - r.square()) / den;
      mw.phi = torch::empty_like(w[IDN]);
      // only rows IDN..IVZ are read by the Coriolis forcings
      mw.wc1 =
          torch::zeros({IVZ + 1, shape[0], shape[1], shape[2]}, w.options());
      mw.wc2 = torch::zeros_like(mw.wc1);
      mw.wc2[IDN].fill_(1.);
    }
    // Coriolis sees w = 0 here; rows IDN, IVY, IVZ are w's, row IVX stays 0
    mw.wc1[IDN].copy_(w[IDN]);
    mw.wc1.narrow(0, IVY, 2).copy_(w.narrow(0, IVY, 2));
  }
  for (auto& f : forcings)
    f.forward(du, mass_w && coriolis(f) ? mw.wc1 : w, temp, dt);
  _forcing_dry = track_dry ? du[IDN] - dry_before : torch::Tensor();

  // Face work uses the x1 mass flux after positivity limiting and
  // sedimentation. Implicit face mode consumes it before the solve; the
  // other modes keep their existing cell or post-solve work.
  torch::Tensor gravity_energy_correction;
  // gravity-work-fixer: this stage's E+PE change of the dynamics (J)
  torch::Tensor gwfix_stage;
  if (options->grav() && options->grav()->grav1() != 0. && _flux1.defined() &&
      !options->disable_flux_x1()) {
    auto grav1 = options->grav()->grav1();
    auto non_hydrostatic = options->grav()->non_hydrostatic();
    auto vertical_mass_flux1 = _flux1[IDN].clone();
    if (ny > 0) {
      vertical_mass_flux1 += _flux1.narrow(0, ICY, ny).sum(0);
    }
    auto total_mass_flux1 =
        gw_cell ? vertical_mass_flux1.clone() : torch::Tensor();
    // cell: the cell work rho g v (const-gravity forcing) stays; only F - F^R
    // is booked as face work
    if (gw_cell) {
      vertical_mass_flux1 = bflux1.defined()
                                ? vertical_mass_flux1 - bflux1
                                : torch::zeros_like(vertical_mass_flux1);
    }

    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    auto area1 = pmb->pcoord->face_area1();
    auto volume = pmb->pcoord->cell_volume();
    auto phi_cell = -grav1 * pmb->pcoord->x1v;
    auto vertical_mass_div =
        (area1.slice(-1, is + 1, ie + 1) *
             vertical_mass_flux1.slice(-1, is + 1, ie + 1) -
         area1.slice(-1, is, ie) * vertical_mass_flux1.slice(-1, is, ie)) /
        volume.slice(-1, is, ie);
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
      auto potential_flux_div =
          (area1.slice(-1, is + 1, ie + 1) *
               potential_flux1.slice(-1, is + 1, ie + 1) -
           area1.slice(-1, is, ie) * potential_flux1.slice(-1, is, ie)) /
          volume.slice(-1, is, ie);
      face_gravity_work = dt * (phi_cell.slice(0, is, ie) * vertical_mass_div -
                                potential_flux_div);
    }

    // gravity-work: face adds the work of the corrected potential energy for
    // this stage's x1 density change (derivation sec 7), where the grid has a
    // form for it
    bool radial_exact = !gw_cell && radial_exact_work();
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
        _x1_ghost_rows(full, 1, false, 0x7726);
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
    if (gw_cell) {
      gravity_energy_correction = face_gravity_work;
      if (gravity_work_fixer()) {
        // the PE change of the mass the x1 fluxes move (x1 wall faces dropped:
        // the fixer runs with sealed walls only) plus the gravity work booked
        // into E; x2/x3 fluxes do not move mass across geopotential surfaces
        auto fw = total_mass_flux1.clone();
        if (is_x1_wall(0)) fw.select(-1, is).zero_();
        if (is_x1_wall(1)) fw.select(-1, ie).zero_();
        auto dm =
            -dt *
            (area1.slice(-1, is + 1, ie + 1) * fw.slice(-1, is + 1, ie + 1) -
             area1.slice(-1, is, ie) * fw.slice(-1, is, ie)) /
            volume.slice(-1, is, ie);
        auto e = (original_gravity_work + face_gravity_work +
                  phi_cell.slice(0, is, ie) * dm) *
                 volume.slice(-1, is, ie);
        int js = pmb->pcoord->jl(), je = pmb->pcoord->ju() + 1;
        int ks = pmb->pcoord->kl(), ke = pmb->pcoord->ku() + 1;
        gwfix_stage = e.slice(-2, js, je).slice(-3, ks, ke).sum().to(wide);
        // the fixer drops these faces, so it checks that no mass crossed them
        for (int f : {0, 1}) {
          if (!is_x1_wall(f)) continue;
          int i = f == 0 ? is : ie;
          auto m = area1.select(-1, i) * total_mass_flux1.select(-1, i).abs();
          _gwfix_wall +=
              dt * m.slice(-1, js, je).slice(-2, ks, ke).sum().to(wide);
        }
      }
    } else {
      gravity_energy_correction = face_gravity_work - original_gravity_work;
    }
  }

  // apply hydrostatic correction
  if (options->grav() && (options->grav()->non_hydrostatic() < 1.)) {
    du[IVX] += dt * rho_grav * (1. - options->grav()->non_hydrostatic());
    du[IPR] +=
        dt * w[IVX] * rho_grav * (1. - options->grav()->non_hydrostatic());
  }

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Forcing time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

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
    double dt_corr = dt;
    if (pmb->pintg->stages.size() >= 3) {
      if (rk_stage >= 0 && rk_stage < pmb->pintg->stages.size()) {
        dt_corr *= pmb->pintg->stages[rk_stage].wght2();
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
    // gravity-work-fixer: the implicit block's E+PE change (its mass transfer
    // and the cell gravity work it linearises; its other energy flux sums to
    // zero over a sealed column)
    auto epe = [&](torch::Tensor const& d) {
      auto m = d[IDN].clone();
      if (ny > 0) m += d.narrow(0, ICY, ny).sum(0);
      auto phi = -options->grav()->grav1() * pmb->pcoord->x1v;
      auto in3 = pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
      return ((d[IPR] + phi * m) * pmb->pcoord->cell_volume())
          .index(in3)
          .sum()
          .to(wide);
    };
    torch::Tensor epe0;
    if (gwfix_stage.defined()) epe0 = epe(du);
    // face work in the operator: the solve also sees the explicit face
    // correction, so the energy it inverts is the face-form energy
    if (gravity_energy_correction.defined() && face_work_in_operator()) {
      int is = pmb->pcoord->il();
      int ie = pmb->pcoord->iu() + 1;
      du[IPR].slice(-1, is, ie) += gravity_energy_correction;
      gravity_energy_correction = torch::Tensor();
    }
    _apply_implicit_correction(du, w, dt_corr, other);
    if (gwfix_stage.defined()) gwfix_stage = gwfix_stage + epe(du) - epe0;

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(Hydro) << "Implicit time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  if (gravity_energy_correction.defined()) {
    int is = pmb->pcoord->il();
    int ie = pmb->pcoord->iu() + 1;
    du[IPR].slice(-1, is, ie) += gravity_energy_correction;
  }

  //// ------------ (8) x1 mass-flux w in Coriolis and metric sources -------
  // -2 Omega w cos(lat) and -u w / r take rho w from the stage's final x1
  // face mass transfers Phi (explicit flux + implicit transfer), split as
  //   m1 = [Phi_lo (r^2 - r_lo^2) + Phi_hi (r_hi^2 - r^2)] / (2 r V dt),
  // so a column's Coriolis torque equals the planetary angular momentum
  // its x1 transfers move, and a wall face that moves no mass exerts none
  // (the cell-centre w of a wall cell moves no mass but did exert a torque)
  if (mass_w) {
    auto pc = pmb->pcoord;
    int is = pc->il(), ie = pc->iu() + 1;
    // phi: mass transfer rate through face i (below i), explicit flux plus
    // the implicit transfer per unit time
    auto& phi = mw.phi;
    if (_flux1.defined() && !options->disable_flux_x1()) {
      // no species rows: the dry flux is the whole mass flux
      auto f1 =
          ny > 0 ? _flux1[IDN] + _flux1.narrow(0, ICY, ny).sum(0) : _flux1[IDN];
      torch::mul_out(phi, mw.area.narrow(-1, 0, f1.size(-1)), f1);
    } else {
      phi.zero_();
    }
    auto mc = implicit_mass_correction();
    if (mc.defined() && mc.numel() > 0) phi.add_(mc[IVZ], 1. / dt);
    // m1 lives in row IVX of the Coriolis input wc2 = (1, m1, 0, 0), x1
    // ghosts stay 0: only the rows driven by rho w are added here, the
    // w-free rows went in before the implicit solve
    auto m1 = mw.wc2[IVX];
    auto m1i = m1.slice(-1, is, ie);
    torch::mul_out(m1i, phi.slice(-1, is, ie), mw.lo);
    m1i.addcmul_(phi.slice(-1, is + 1, ie + 1), mw.hi);
    for (auto& f : forcings)
      if (coriolis(f)) f.forward(du, mw.wc2, temp, dt);
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
      pc->axial_x1_source_(src[IVY], src[IVZ],
                           picorr ? picorr->correction() : torch::Tensor());
      du.narrow(0, IVY, 2) += src.narrow(0, IVY, 2);
    }
  } else if (ptype == "gnomonic-equiangle") {
    // no x1 mass-flux w (nc1 == 1, shallow water): m1 = rho w. Only the
    // gnomonic grid has an x1_mass_source_; the others add nothing
    auto src = torch::zeros_like(du);
    pmb->pcoord->x1_mass_source_(src, w, w[IDN] * w[IVX], dt);
    // conserve-axial-momentum: no K component, as in the branch above
    if (aam) pmb->pcoord->axial_x1_source_(src[IVY], src[IVZ], torch::Tensor());
    du.index(interior) += src.index(interior);
  }

  if (gwfix_stage.defined()) {
    // weight of this stage's du in the step's update u0 -> u1 (stage s:
    // u <- w0 u0 + w1 u + w2 du): w2_s * prod_{t > s} w1_t (rk3: 1/6 1/6 2/3)
    double cw = 1.;
    auto const& st = pmb->pintg->stages;
    if (rk_stage >= 0 && rk_stage < static_cast<int>(st.size())) {
      cw = st[rk_stage].wght2();
      for (int t = rk_stage + 1; t < static_cast<int>(st.size()); ++t)
        cw *= st[t].wght1();
    }
    _gwfix_d += cw * gwfix_stage.to(_gwfix_d.device());
  }

  return du;
}

}  // namespace snap
