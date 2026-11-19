#pragma once

// C/C++
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

//! Work from area-integrated face mass fluxes (or mass increments). The result
//! has the interior cell shape [is, ie); callers multiply by dt only for
//! fluxes. Difference geometry BEFORE casting to the working dtype, so a large
//! origin cannot erase a short cell width when FP64 master coordinates are
//! supplied.
inline torch::Tensor gravity_work_from_face_mass(torch::Tensor const& face_mass,
                                                 torch::Tensor const& x1f,
                                                 torch::Tensor const& x1v,
                                                 torch::Tensor const& volume,
                                                 int is, int ie, double grav1) {
  auto centres = x1v.slice(0, is, ie);
  auto hi = (grav1 * (x1f.slice(0, is + 1, ie + 1) - centres))
                .to(face_mass.options());
  auto lo = (grav1 * (x1f.slice(0, is, ie) - centres)).to(face_mass.options());
  return (hi * face_mass.slice(-1, is + 1, ie + 1) -
          lo * face_mass.slice(-1, is, ie)) /
         volume.slice(-1, is, ie).to(face_mass.options());
}

//! the x1 cell measure the corrected-PE work integrates over, and where x1v
//! sits in it (docs/derivations/curved-gravity-work-weight.md, secs 7, 12)
enum class X1Measure {
  none,       //!< no form for the corrected-PE work on this grid
  plain,      //!< dx1 (cartesian); x1v is the face midpoint, the centroid
  radial,     //!< r^2 dr (spherical-polar); x1v is the r^2 centroid
  radial_mid  //!< r^2 dr (gnomonic-equiangle); x1v is the face midpoint
};

inline X1Measure x1_measure(std::string const& type) {
  if (type == "cartesian") return X1Measure::plain;
  if (type == "spherical-polar") return X1Measure::radial;
  if (type == "gnomonic-equiangle") return X1Measure::radial_mid;
  return X1Measure::none;
}

//! <(x1 - r_c)^2> over each cell, r_c the centroid of the cell's own x1
//! measure, from its x1 faces (n + 1 of them): dx1^2/12 on a Cartesian grid,
//! the r^2 measure on a radial one (spherical-polar or gnomonic-equiangle;
//! r_c is x1v except on gnomonic-equiangle, docs/derivations/
//! curved-gravity-work-weight.md, secs 7, 12).
//! The r^2 form is written about the face midpoint so it does not cancel at
//! large r.
inline torch::Tensor x1_variance(torch::Tensor const& x1f, bool spherical) {
  int n = x1f.size(0) - 1;
  auto h = x1f.narrow(0, 1, n) - x1f.narrow(0, 0, n);
  auto h2 = h * h;
  if (!spherical) return h2 / 12.;
  auto rb = .5 * (x1f.narrow(0, 1, n) + x1f.narrow(0, 0, n));
  auto vol = rb * rb * h + h * h2 / 12.;  // (r+^3 - r-^3) / 3
  auto shift = rb * h * h2 / (6. * vol);  // r_c - rb
  return (rb * rb * h * h2 / 12. + h * h2 * h2 / 80.) / vol - shift * shift;
}

//! centroid_slope as a three-point stencil of x: s_i = wt[0]_i q[idx[0]_i] +
//! wt[1]_i q[idx[1]_i] + wt[2]_i q[idx[2]_i], summed in that order. Built once
//! for a grid, a slope is 3 gathers, 3 products and 2 sums (#291). No weights
//! (a zero slope) for fewer than three cells.
struct CentroidSlopeStencil {
  torch::Tensor idx[3], wt[3];
  torch::Tensor operator()(torch::Tensor const& q) const {
    if (!wt[0].defined()) return torch::zeros_like(q);
    auto s = wt[0] * q.index_select(-1, idx[0]);
    s += wt[1] * q.index_select(-1, idx[1]);
    s += wt[2] * q.index_select(-1, idx[2]);
    return s.to(q.scalar_type());
  }
  //! the weight of cell i + off in the slope of cell i, for each cell i (zero
  //! where the stencil does not reach): one diagonal of the slope as a matrix,
  //! summed as operator() sums it
  torch::Tensor band(int off) const {
    auto at = torch::arange(idx[0].size(0), idx[0].options()) + off;
    auto s = wt[0] * (idx[0] == at).to(wt[0].scalar_type());
    s += wt[1] * (idx[1] == at).to(wt[1].scalar_type());
    s += wt[2] * (idx[2] == at).to(wt[2].scalar_type());
    return s;
  }
};

//! d q / d x1 at each x along the last dimension (n = x.size(0) cells): the
//! slope at x_i of the quadratic through cells i - 1, i, i + 1, and through
//! the first or last three cells at the two ends (at an internal x1 seam the
//! caller extends q by the neighbour's cell, corrected_pe_work_seam, so the
//! seam cell takes the centred slope; derivation sec 7). Zero for n < 3.
//! A subtracted term's minus sits in its weight: a - w q == a + (-w) q exactly.
inline CentroidSlopeStencil centroid_slope_stencil(torch::Tensor const& x) {
  CentroidSlopeStencil st;
  int n = x.size(0);
  if (n < 3) return st;
  auto hm = x.narrow(0, 1, n - 2) - x.narrow(0, 0, n - 2);
  auto hp = x.narrow(0, 2, n - 2) - x.narrow(0, 1, n - 2);
  auto a = x.narrow(0, 1, 1) - x.narrow(0, 0, 1);
  auto b = x.narrow(0, 2, 1) - x.narrow(0, 1, 1);
  auto c = x.narrow(0, n - 2, 1) - x.narrow(0, n - 3, 1);
  auto d = x.narrow(0, n - 1, 1) - x.narrow(0, n - 2, 1);
  st.wt[0] = torch::cat({-(2. * a + b) / (a * (a + b)), -hp / (hm * (hm + hp)),
                         d / (c * (c + d))});
  st.wt[1] = torch::cat(
      {(a + b) / (a * b), (hp - hm) / (hm * hp), -((c + d) / (c * d))});
  st.wt[2] = torch::cat({-(a / ((a + b) * b)), hm / (hp * (hm + hp)),
                         (c + 2. * d) / ((c + d) * d)});
  auto j =
      torch::arange(n, x.options().dtype(torch::kLong)).clamp(1, n - 2) - 1;
  st.idx[0] = j;
  st.idx[1] = j + 1;
  st.idx[2] = j + 2;
  return st;
}

inline torch::Tensor centroid_slope(torch::Tensor const& q,
                                    torch::Tensor const& x) {
  return centroid_slope_stencil(x)(q);
}

//! r_c - x1v for each cell, from its x1 faces (n + 1) and x1v (n): the
//! offset of the r^2 centroid r_c = rb + rb h^3 / (6 V_r) from x1v, written
//! about the face midpoint rb so it does not cancel at large r (sec 12)
inline torch::Tensor x1_centroid_offset(torch::Tensor const& x1f,
                                        torch::Tensor const& x1v) {
  int n = x1f.size(0) - 1;
  auto h = x1f.narrow(0, 1, n) - x1f.narrow(0, 0, n);
  auto h3 = h * h * h;
  auto rb = .5 * (x1f.narrow(0, 1, n) + x1f.narrow(0, 0, n));
  auto vol = rb * rb * h + h3 / 12.;  // (r+^3 - r-^3) / 3
  return rb * h3 / (6. * vol) + (rb - x1v);
}

//! the grid, x1 range and gravity a cached x1 factor was built for (#291); an
//! in-place write to x1f or x1v bumps its version (inference tensors keep none)
struct X1GridKey {
  torch::Tensor x1f, x1v;
  int64_t vf = -1, vv = -1;
  int is = 0, ie = 0, lo = 0, hi = 0;
  double grav1 = 0.;
  X1Measure measure = X1Measure::none;

  X1GridKey() = default;
  X1GridKey(torch::Tensor const& x1f_, torch::Tensor const& x1v_, int is_,
            int ie_, int lo_, int hi_, double grav1_, X1Measure measure_)
      : x1f(x1f_),
        x1v(x1v_),
        vf(version(x1f_)),
        vv(version(x1v_)),
        is(is_),
        ie(ie_),
        lo(lo_),
        hi(hi_),
        grav1(grav1_),
        measure(measure_) {}

  static int64_t version(torch::Tensor const& t) {
    return t.is_inference() ? 0 : t._version();
  }
  bool operator==(X1GridKey const& o) const {
    return x1f.defined() && x1f.is_same(o.x1f) && x1v.is_same(o.x1v) &&
           vf == o.vf && vv == o.vv && is == o.is && ie == o.ie && lo == o.lo &&
           hi == o.hi && grav1 == o.grav1 && measure == o.measure;
  }
};

//! the grid-only factors of corrected_pe_work_seam for the cells [is, ie) of
//! a q given over [lo, hi): the slope stencil on x1v[lo, hi), grav1 <(x1 -
//! r_c)^2> and, on radial_mid, grav1 (r_c - x1v)
struct CorrectedPEFactors {
  X1GridKey key;
  CentroidSlopeStencil slope;
  torch::Tensor gvar, goff;

  torch::Tensor operator()(torch::Tensor const& q) const {
    int i0 = key.is - key.lo, n = key.ie - key.is;
    auto work = gvar * slope(q).narrow(-1, i0, n);
    if (goff.defined()) work += goff * q.narrow(-1, i0, n);
    return work;
  }
};

inline CorrectedPEFactors corrected_pe_factors(X1GridKey const& key) {
  CorrectedPEFactors f;
  f.key = key;
  auto faces = key.x1f.slice(0, key.is, key.ie + 1);
  f.slope = centroid_slope_stencil(key.x1v.slice(0, key.lo, key.hi));
  f.gvar = key.grav1 * x1_variance(faces, key.measure != X1Measure::plain);
  if (key.measure == X1Measure::radial_mid)
    f.goff =
        key.grav1 * x1_centroid_offset(faces, key.x1v.slice(0, key.is, key.ie));
  return f;
}

//! the factors for key, from cache when it holds them (else rebuilt into it).
//! Module::to keeps the x1f identity and version, so check dtype and device
inline CorrectedPEFactors const& corrected_pe_factors(
    X1GridKey const& key, CorrectedPEFactors* cache) {
  if (!(cache->key == key) ||
      cache->gvar.scalar_type() != key.x1f.scalar_type() ||
      cache->gvar.device() != key.x1f.device())
    *cache = corrected_pe_factors(key);
  return *cache;
}

//! x1 gravity work of the corrected potential energy beyond the face form,
//! per unit volume: grav1 [<(x1 - r_c)^2> d(drho)/dx1 + (r_c - x1v) drho],
//! for a density change drho of the interior cells [is, ie) (sec 7, eq. 7;
//! sec 12, eq. 12.5). r_c is the centroid of the cell measure; it is x1v
//! except on radial_mid, so the second term is booked only there. Booking it
//! makes E + P exact, P = sum V [rho phi(r_c) - grav1 <(x1 - r_c)^2> drho/dx1].
//! A cache keeps the grid-only factors between calls.
inline torch::Tensor corrected_pe_work(torch::Tensor const& drho,
                                       torch::Tensor const& x1f,
                                       torch::Tensor const& x1v, int is, int ie,
                                       double grav1, X1Measure measure,
                                       CorrectedPEFactors* cache = nullptr) {
  X1GridKey key(x1f, x1v, is, ie, is, ie, grav1, measure);
  if (cache) return corrected_pe_factors(key, cache)(drho);
  return corrected_pe_factors(key)(drho);
}

//! corrected_pe_work of the interior cells [is, ie) when q, given over the
//! whole x1 array (..., nc1), also holds the neighbour's value one cell past an
//! internal x1 seam (seam_lo: cell is - 1, seam_hi: cell ie). The slope of a
//! seam cell is then the centred one that a single block takes, so the work
//! and P do not depend on the x1 decomposition; at a physical wall the slope
//! stays one-sided (sec 7 item 3, #303).
inline torch::Tensor corrected_pe_work_seam(
    torch::Tensor const& q, torch::Tensor const& x1f, torch::Tensor const& x1v,
    int is, int ie, double grav1, X1Measure measure, bool seam_lo, bool seam_hi,
    CorrectedPEFactors* cache = nullptr) {
  int lo = seam_lo ? is - 1 : is, hi = seam_hi ? ie + 1 : ie;
  X1GridKey key(x1f, x1v, is, ie, lo, hi, grav1, measure);
  auto ql = q.slice(-1, lo, hi);
  if (cache) return corrected_pe_factors(key, cache)(ql);
  return corrected_pe_factors(key)(ql);
}

//! The x1 gravity work of lane I (V3 I-3): G[M] = W[M] + K[drho] (I.rst
//! section 4). One object per vertical implicit solve holds every grid-only
//! coefficient the solve uses, built once per grid:
//!   * the face weights of W for the operator's energy row, lower / upper =
//!     0.5 A (x1v - x1f) / V on each side of a cell ("rows from the
//!     coefficients"; on a uniform Cartesian grid both are 0.25), with the
//!     face areas, cell volumes, cell potentials and face-minus-cell
//!     potentials the post-solve remainder reads;
//!   * the corrected-PE rows of K on the density column: three bands and the
//!     two wall third points of the exact 3-point slope (Woodbury), and the
//!     stencil and work factors corrected_pe_work reads.
//! The explicit stage (hydro_forward.cpp) still calls the free functions
//! above; its call site moves with lane R.
class GravityWork {
 public:
  //! the grid, device, dtype, shape and gravity the solve's factors are for
  struct Key {
    c10::TensorImpl const *f1 = nullptr, *f2 = nullptr, *f3 = nullptr;
    int64_t v1 = -1, v2 = -1, v3 = -1;
    torch::Device dev = torch::kCPU;
    c10::ScalarType st = torch::kDouble;
    std::vector<int64_t> shape;
    double grav1 = 0.;
    bool operator==(Key const& o) const {
      return f1 == o.f1 && f2 == o.f2 && f3 == o.f3 && v1 == o.v1 &&
             v2 == o.v2 && v3 == o.v3 && dev == o.dev && st == o.st &&
             shape == o.shape && grav1 == o.grav1;
    }
  };

  //! rebuild the face weights and potentials when the grid, device, dtype,
  //! shape (of gamma) or grav1 changed
  template <class Coord>
  void update(Coord const& pc, torch::Tensor const& w,
              torch::Tensor const& gamma, double grav1) {
    Key key;
    key.f1 = pc.x1f.unsafeGetTensorImpl();
    key.f2 = pc.x2f.unsafeGetTensorImpl();
    key.f3 = pc.x3f.unsafeGetTensorImpl();
    key.v1 = pc.x1f._version();
    key.v2 = pc.x2f._version();
    key.v3 = pc.x3f._version();
    key.dev = w.device();
    key.st = w.scalar_type();
    key.shape = gamma.sizes().vec();
    key.grav1 = grav1;
    if (_key == key && _area.defined()) return;
    _area = pc.face_area1().to(w.options()).contiguous();
    _volume = pc.cell_volume().to(w.options()).contiguous();
    int nc1g = w.size(-1);
    _face_lo = .5 * _area.narrow(-1, 0, nc1g) *
               (pc.x1v - pc.x1f.narrow(0, 0, nc1g)) / _volume;
    _face_hi = .5 * _area.narrow(-1, 1, nc1g) *
               (pc.x1f.narrow(0, 1, nc1g) - pc.x1v) / _volume;
    _face_lo = _face_lo.to(w.options()).unsqueeze(0).contiguous();
    _face_hi = _face_hi.to(w.options()).unsqueeze(0).contiguous();
    _mask0 = torch::zeros_like(gamma);
    int isg = pc.il(), ieg = pc.iu() + 1;
    _phi = -grav1 * pc.x1v.slice(0, isg, ieg);
    auto zg = pc.x1v.slice(0, isg, ieg);
    _dp_lo = -grav1 * (pc.x1f.slice(0, isg, ieg) - zg);
    _dp_hi = -grav1 * (pc.x1f.slice(0, isg + 1, ieg + 1) - zg);
    _key = key;
  }

  torch::Tensor const& area() const { return _area; }
  torch::Tensor const& volume() const { return _volume; }
  //! W's face weights for the operator's energy row, [1, ..., nc1]
  torch::Tensor const& face_lo() const { return _face_lo; }
  torch::Tensor const& face_hi() const { return _face_hi; }
  //! a zero solid mask of gamma's shape
  torch::Tensor const& mask0() const { return _mask0; }
  //! cell potentials and face-minus-cell potentials of the interior x1 range
  torch::Tensor const& phi() const { return _phi; }
  torch::Tensor const& dp_lo() const { return _dp_lo; }
  torch::Tensor const& dp_hi() const { return _dp_hi; }

  //! rebuild K's rows for the operator (the exact 3-point slope on a whole
  //! column, one-sided only at its two walls) when the grid key, dtype or
  //! device changed
  void update_rows(X1GridKey const& key, torch::Tensor const& like, int nx1) {
    if (_rows_key == key && _k_mid.scalar_type() == like.scalar_type() &&
        _k_mid.device() == like.device())
      return;
    auto opts = like.options();
    auto const& f = corrected_pe_factors(key, &_pe);
    auto band = [&](int off) { return f.slope.band(off).to(opts); };
    auto gv = f.gvar.to(opts);
    auto mid = gv * band(0);
    // gnomonic-equiangle: x1v is not the r^2 centroid, and the work carries
    // g1 (r_c - x1v) drho as well (sec 12), a diagonal term
    if (f.goff.defined()) mid += f.goff.to(opts);
    auto lo = torch::zeros_like(mid);
    auto hi = torch::zeros_like(mid);
    lo.narrow(0, 1, nx1 - 1).copy_(band(-1).narrow(0, 1, nx1 - 1));
    hi.narrow(0, 0, nx1 - 1).copy_(band(1).narrow(0, 0, nx1 - 1));
    lo *= gv;
    hi *= gv;
    _rows_key = key;
    _k_lo = lo;
    _k_mid = mid;
    _k_hi = hi;
    // the one-sided wall slopes' third points
    _k_t0 = gv[0] * band(2)[0];
    _k_t1 = gv[nx1 - 1] * band(-2)[nx1 - 1];
  }

  //! K's rows: three bands on the density column and two wall third points
  torch::Tensor const& k_lo() const { return _k_lo; }
  torch::Tensor const& k_mid() const { return _k_mid; }
  torch::Tensor const& k_hi() const { return _k_hi; }
  torch::Tensor const& k_t0() const { return _k_t0; }
  torch::Tensor const& k_t1() const { return _k_t1; }

  //! K's rows applied to q [nx3, nx2, nx1] (the full corrected-PE operator,
  //! no lumping); fluid (1 or 0 per cell) masks the rows of solid cells
  torch::Tensor apply_rows(torch::Tensor const& q, torch::Tensor const& fluid,
                           int nx1) const {
    auto s = _k_mid * q;
    s.narrow(-1, 1, nx1 - 1) +=
        _k_lo.narrow(-1, 1, nx1 - 1) * q.narrow(-1, 0, nx1 - 1);
    s.narrow(-1, 0, nx1 - 1) +=
        _k_hi.narrow(-1, 0, nx1 - 1) * q.narrow(-1, 1, nx1 - 1);
    s.select(-1, 0) += _k_t0 * q.select(-1, 2);
    s.select(-1, nx1 - 1) += _k_t1 * q.select(-1, nx1 - 3);
    return fluid.defined() ? fluid * s : s;  // 1 * s == s
  }

  //! the stencil and work factors of corrected_pe_work, shared with the rows
  CorrectedPEFactors* pe_factors() { return &_pe; }

 private:
  Key _key;
  torch::Tensor _area, _volume, _face_lo, _face_hi, _mask0, _phi, _dp_lo,
      _dp_hi;
  X1GridKey _rows_key;
  torch::Tensor _k_lo, _k_mid, _k_hi, _k_t0, _k_t1;
  CorrectedPEFactors _pe;
};

}  // namespace snap
