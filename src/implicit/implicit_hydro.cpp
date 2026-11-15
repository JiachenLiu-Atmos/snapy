// yaml
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "implicit_dispatch.hpp"
#include "implicit_hydro.hpp"

namespace snap {

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename,
                                               bool /*verbose*/) {
  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return nullptr;
  auto intg = config["integration"];
  // pyharp's integrator reads this block too, so its keys are listed by name
  check_keys(intg, "integration",
             {"implicit-scheme", "implicit-advection-cfl", "shear-cfl",
              "implicit-precision", "implicit-fallback-fp64"},
             "pyharp",
             {"type", "cfl", "tlim", "nlim", "ncycle_out", "verbose"});
  auto op =
      intg["implicit-scheme"] ? from_yaml(intg["implicit-scheme"]) : nullptr;
  if (!op) {
    TORCH_CHECK(!intg["implicit-advection-cfl"] && !intg["shear-cfl"] &&
                    !intg["implicit-precision"] &&
                    !intg["implicit-fallback-fp64"],
                "integration/implicit-advection-cfl and shear-cfl bound an "
                "implicit direction: set implicit-scheme or remove them");
    return nullptr;
  }
  op->precision() = intg["implicit-precision"].as<std::string>("auto");
  op->fallback_fp64() = intg["implicit-fallback-fp64"].as<bool>(true);
  TORCH_CHECK(
      op->precision() == "auto" || op->precision() == "float32" ||
          op->precision() == "float64",
      "integration/implicit-precision must be auto, float32 or float64");
  {
    auto read_finite = [&](char const* key, double fallback) {
      if (!intg[key]) return fallback;
      auto const value = intg[key];
      TORCH_CHECK(value.IsScalar(), "integration/", key,
                  " must be a finite number.");
      double parsed = 0.;
      try {
        parsed = value.as<double>();
      } catch (YAML::Exception const&) {
        TORCH_CHECK(false, "integration/", key,
                    " must be a finite number, got '", value.Scalar(), "'.");
      }
      TORCH_CHECK(std::isfinite(parsed), "integration/", key,
                  " must be a finite number, got ", parsed, ".");
      return parsed;
    };
    op->advection_cfl(read_finite("implicit-advection-cfl", 1.0));
    TORCH_CHECK(op->advection_cfl() > 0.,
                "integration/implicit-advection-cfl must be > 0, got ",
                op->advection_cfl(), " (use a large value to relax the bound)");
    op->shear_cfl(read_finite("shear-cfl", 0.0));
    TORCH_CHECK(op->shear_cfl() >= 0.,
                "integration/shear-cfl must be >= 0, got ", op->shear_cfl(),
                " (0 switches the bound off)");
  }
  return op;
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const YAML::Node& node) {
  int s = node.as<int>();
  // scheme 0 == "none": an implicit object that does nothing. Treat it as if
  // the `implicit-scheme` key were absent (return nullptr) so `implicit-scheme:
  // 0` is a true explicit spelling that also runs at nb1>1, instead of tripping
  // the nb1 guard on a phantom no-op object. picorr != null now faithfully
  // means "implicit is active".
  if (s == 0) return nullptr;
  auto op = ImplicitOptionsImpl::create();
  op->scheme(s);
  return op;
}

std::string ImplicitOptionsImpl::type() const {
  switch (scheme()) {
    case 0:
      return "none";
      break;
    case 1:
      return "vic-partial";
      break;
    case 9:
      return "vic-full";
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }
}

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions const& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ImplicitHydroImpl::reset() {
  TORCH_CHECK(phydro, "[ImplicitHydro] Parent Hydro is null");
  TORCH_CHECK(options, "[ImplicitHydro] options is null");
  options->type();  // throws on an unsupported scheme
  _fp64_fallbacks =
      register_buffer("fp64_fallbacks", torch::zeros({}, torch::kInt64));
  _a = register_buffer("a", torch::empty({0}, torch::kFloat64));
  _b = register_buffer("b", torch::empty({0}, torch::kFloat64));
  _c = register_buffer("c", torch::empty({0}, torch::kFloat64));
  _delta = register_buffer("delta", torch::empty({0}, torch::kFloat64));
  _du0 = register_buffer("du0", torch::empty({0}, torch::kFloat64));
  _corr = register_buffer("corr", torch::empty({0}, torch::kFloat64));
  _mass_corr = register_buffer("mass_corr", torch::empty({0}, torch::kFloat64));
  _clamp_residual =
      register_buffer("clamp_residual", torch::zeros({1}, torch::kFloat64));
  _dry_clamp_step =
      register_buffer("dry_clamp_step", torch::zeros({1}, torch::kFloat64));
}

void ImplicitHydroImpl::ensure_workspace(torch::Tensor const& w) {
  auto pcoord = phydro->pmb->pcoord;
  int nx1 = pcoord->options->nx1();
  int nx2 = pcoord->options->nx2();
  int nx3 = pcoord->options->nx3();
  int m = options->size();

  auto abc_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m * m};
  auto delta_shape = std::vector<int64_t>{1, nx3, nx2, nx1 * m};

  auto needs_reset = [&](torch::Tensor const& t,
                         std::vector<int64_t> const& shape) {
    return !t.defined() || t.sizes().vec() != shape ||
           t.scalar_type() != w.scalar_type() || t.device() != w.device();
  };

  auto maybe_resize = [&](torch::Tensor& t, std::vector<int64_t> const& shape) {
    if (needs_reset(t, shape)) {
      t.set_data(torch::empty(shape, w.options()));
    }
  };

  maybe_resize(_a, abc_shape);
  maybe_resize(_b, abc_shape);
  maybe_resize(_c, abc_shape);
  maybe_resize(_delta, delta_shape);
  maybe_resize(_du0, w.sizes().vec());
  maybe_resize(_corr, w.sizes().vec());
  maybe_resize(_mass_corr, w.sizes().vec());
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  return forward_masked(du, w, gamma, dt, torch::Tensor());
}

torch::Tensor ImplicitHydroImpl::forward_masked(torch::Tensor du,
                                                torch::Tensor w,
                                                torch::Tensor gamma, double dt,
                                                torch::Tensor solid) {
  if (options->scheme() == 0) {  // null operation
    if (_corr.sizes() != du.sizes() ||
        _corr.scalar_type() != du.scalar_type() ||
        _corr.device() != du.device()) {
      _corr.set_data(torch::zeros_like(du));
    } else {
      _corr.zero_();
    }
    return _corr;
  }

  // Once a stage fails, later stages cannot turn the step into a success.
  if (_solve_failed) {
    _corr.zero_();
    _mass_corr.zero_();
    return _corr;
  }

  TORCH_CHECK(phydro->options->grav(),
              "[ImplicitHydro] forcing does not have const-gravity");

  auto pcoord = phydro->pmb->pcoord;
  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered before implicit solve");
  }*/

  ensure_workspace(w);
  if (!_failed_dev.defined() || _failed_dev.device() != du.device())
    _failed_dev = torch::zeros({}, du.options().dtype(torch::kBool));
  _du0.copy_(du);
  _mass_corr.zero_();

  // restore exactly if this solve is rejected; the solve changes only these
  // two rows of w in place (the frame projection)
  auto w0 = w.narrow(0, IVY, 2).clone();

  auto reject = [&](torch::Tensor const& bad_columns) {
    _solve_failed = true;
    // no columns: the previous stage failed after its solve
    auto columns = bad_columns.defined() ? bad_columns.nonzero().cpu()
                                         : torch::zeros({0, 3}, torch::kInt64);
    auto index = columns.accessor<int64_t, 2>();
    for (int64_t n = 0; n < columns.size(0); ++n)
      std::cerr << "[ImplicitHydro] rank=" << get_rank()
                << " VIC singular/near-singular or nonfinite value: column=("
                << index[n][1] + pcoord->kl() << ","
                << index[n][2] + pcoord->jl()
                << ") step=" << phydro->pmb->cycle + 1
                << " stage=" << phydro->rk_stage
                << " retry=" << phydro->pmb->pintg->current_redo
                << "; check_redo will restore step input." << std::endl;
    du.copy_(_du0);
    w.narrow(0, IVY, 2).copy_(w0);
    _corr.zero_();
    _mass_corr.zero_();
    return _corr;
  };
  auto finite_columns = [&](torch::Tensor const& values) {
    return torch::isfinite(values.index(interior)).all(0).all(-1).unsqueeze(0);
  };
  // one device-to-host read per stage (#291): the sum of each checked tensor
  // is nonfinite if any of its values is; only then are the columns found,
  // in the order the checks would have met them. The read also carries the
  // previous stage's post-solve failure, latched on the device.
  std::vector<torch::Tensor> screen = {
      du.index(interior).to(torch::kFloat64).sum(),
      w.index(interior).to(torch::kFloat64).sum(),
      gamma.unsqueeze(0).index(interior).to(torch::kFloat64).sum(),
      torch::zeros({}, du.options())
          .masked_fill(_failed_dev, std::numeric_limits<double>::quiet_NaN())};

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  pcoord->prim2local1_(du);

  // the geometry below depends on the grid only
  bool no_solid = !solid.defined();
  {
    GeoKey key;
    key.f1 = pcoord->x1f.unsafeGetTensorImpl();
    key.f2 = pcoord->x2f.unsafeGetTensorImpl();
    key.f3 = pcoord->x3f.unsafeGetTensorImpl();
    key.v1 = pcoord->x1f._version();
    key.v2 = pcoord->x2f._version();
    key.v3 = pcoord->x3f._version();
    key.dev = w.device();
    key.st = w.scalar_type();
    key.shape = gamma.sizes().vec();
    key.grav1 = phydro->options->grav()->grav1();
    if (!(_g_key == key) || !_g_area.defined()) {
      _g_area = pcoord->face_area1().to(w.options()).contiguous();
      _g_volume = pcoord->cell_volume().to(w.options()).contiguous();
      int nc1g = w.size(-1);
      _g_work_lo = .5 * _g_area.narrow(-1, 0, nc1g) *
                   (pcoord->x1v - pcoord->x1f.narrow(0, 0, nc1g)) / _g_volume;
      _g_work_hi = .5 * _g_area.narrow(-1, 1, nc1g) *
                   (pcoord->x1f.narrow(0, 1, nc1g) - pcoord->x1v) / _g_volume;
      _g_work_lo = _g_work_lo.to(w.options()).unsqueeze(0).contiguous();
      _g_work_hi = _g_work_hi.to(w.options()).unsqueeze(0).contiguous();
      _g_mask0 = torch::zeros_like(gamma);
      int isg = pcoord->il(), ieg = pcoord->iu() + 1;
      double g1 = key.grav1;
      _g_phi = -g1 * pcoord->x1v.slice(0, isg, ieg);
      auto zg = pcoord->x1v.slice(0, isg, ieg);
      _g_dp_lo = -g1 * (pcoord->x1f.slice(0, isg, ieg) - zg);
      _g_dp_hi = -g1 * (pcoord->x1f.slice(0, isg + 1, ieg + 1) - zg);
      _g_key = key;
    }
  }
  auto mask = solid.defined() ? solid.to(w.options()).contiguous() : _g_mask0;

  auto area = _g_area;
  auto volume = _g_volume;
  int nc1 = w.size(-1);

  //// -------- Solve block-tridiagonal matrix --------- ////
  // one config for the solve iterators; Woodbury appends its z0, z1
  auto vic_config = [&]() {
    at::TensorIteratorConfig cfg;
    cfg.resize_outputs(false)
        .check_all_same_dtype(true)
        .declare_static_shape(du.index(interior).sizes(),
                              /*squash_dims=*/{0, 3})
        .add_owned_output(du.index(interior))
        .add_owned_output(_mass_corr.index(interior))
        .add_owned_input(w.index(interior))
        .add_owned_input(gamma.unsqueeze(0).index(interior))
        .add_owned_input(area.unsqueeze(0).index(interior))
        .add_owned_input(volume.unsqueeze(0).index(interior))
        .add_input(_a)
        .add_input(_b)
        .add_input(_c)
        .add_input(_delta)
        .add_owned_input(mask.unsqueeze(0).index(interior))
        .add_owned_input(_g_work_lo.index(interior))
        .add_owned_input(_g_work_hi.index(interior));
    return cfg;
  };
  auto iter = vic_config().build();

  // Linearize the FULL gravity: du always carries it (body force + rho_grav
  // sum to grav1); scaling by non_hydrostatic() drops the gravity coupling
  // and destabilizes the solve at dt >> dt_acoustic whenever nh < 1.
  auto grav1 = phydro->options->grav()->grav1();
  // gravity-work: face books the metric-weighted work in the energy row
  bool face_work = phydro->face_work_in_operator();
  int adir = face_work ? kVicFaceWork : 0;
  if (face_work && pcoord->options->type() == "cartesian")
    adir |= kVicCartesianFaceWork;
  bool diffusive_work =
      grav1 != 0. && phydro->options->grav()->gravity_work() == "cell";
  if (diffusive_work) adir |= kVicDiffusiveCell;

  // gravity-work: face: the energy row also carries the corrected
  // potential energy's work g1 sigma^2 s[drho] of the mass this solve moves
  // (derivation sec 7, eq. 7). Booked only after the solve, that work is
  // explicit in a fixed fraction of the face work of a grid-scale density
  // change, and a rest column grows from round-off at vertical acoustic
  // Courant ~200 (#296). The matrix holds the exact 3-point slope s: its
  // tridiagonal part in the blocks and the two wall third points by Woodbury
  // (below), coupled to the solved minus the explicit density change; the
  // post-solve term books the rest, so the total stays g1 sigma^2 s[du - du0].
  bool radial_exact = phydro->radial_exact_work();
  auto in3 =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int nx1 = pcoord->options->nx1();
  torch::Tensor rx_lo, rx_mid, rx_hi, rx_t0, rx_t1, rx_fluid, rx_mass0;
  // the full corrected-PE operator: tridiagonal part plus the two wall
  // third points of the exact 3-point slope (no lumping)
  auto rx_tri = [&](torch::Tensor const& q) {  // q: [nx3, nx2, nx1]
    auto s = rx_mid * q;
    s.narrow(-1, 1, nx1 - 1) +=
        rx_lo.narrow(-1, 1, nx1 - 1) * q.narrow(-1, 0, nx1 - 1);
    s.narrow(-1, 0, nx1 - 1) +=
        rx_hi.narrow(-1, 0, nx1 - 1) * q.narrow(-1, 1, nx1 - 1);
    s.select(-1, 0) += rx_t0 * q.select(-1, 2);
    s.select(-1, nx1 - 1) += rx_t1 * q.select(-1, nx1 - 3);
    return no_solid ? s : rx_fluid * s;  // 1 * s == s
  };
  auto couple_radial_exact = [&]() {
    if (!radial_exact || nx1 < 3) return;
    int is = pcoord->il(), ie = pcoord->iu() + 1, m = options->size();
    auto measure = x1_measure(pcoord->options->type());
    // the rows depend on the grid and gravity only: build them once (#291)
    X1GridKey key(pcoord->x1f, pcoord->x1v, is, ie, is, ie, grav1, measure);
    if (!(_rx_key == key) || _rx_mid.scalar_type() != w.scalar_type() ||
        _rx_mid.device() != w.device()) {
      // the stencil and work factors of corrected_pe_work, which the
      // post-solve term below reads from the same cache. The slope is
      // one-sided only at the column's two physical walls: an implicit scheme
      // refuses an x1 split (HydroImpl::_apply_implicit_correction), so a VIC
      // block has no internal x1 seam and this per-block slope is the
      // one-block slope
      auto const& f = corrected_pe_factors(key, &_rx_work);
      auto band = [&](int off) { return f.slope.band(off).to(w.options()); };
      auto gv = f.gvar.to(w.options());
      rx_mid = gv * band(0);
      // gnomonic-equiangle: x1v is not the r^2 centroid, and the work carries
      // g1 (r_c - x1v) drho as well (sec 12), a diagonal term
      if (f.goff.defined()) rx_mid += f.goff.to(w.options());
      rx_lo = torch::zeros_like(rx_mid);
      rx_hi = torch::zeros_like(rx_mid);
      rx_lo.narrow(0, 1, nx1 - 1).copy_(band(-1).narrow(0, 1, nx1 - 1));
      rx_hi.narrow(0, 0, nx1 - 1).copy_(band(1).narrow(0, 0, nx1 - 1));
      rx_lo *= gv;
      rx_hi *= gv;
      _rx_key = key;
      _rx_lo = rx_lo;
      _rx_mid = rx_mid;
      _rx_hi = rx_hi;
      // the one-sided wall slopes' third points
      _rx_t0 = gv[0] * band(2)[0];
      _rx_t1 = gv[nx1 - 1] * band(-2)[nx1 - 1];
    }
    rx_lo = _rx_lo;
    rx_mid = _rx_mid;
    rx_hi = _rx_hi;
    rx_t0 = _rx_t0;
    rx_t1 = _rx_t1;
    if (!no_solid) rx_fluid = (mask.index(in3) == 0).to(w.options());
    // row IPR, column 0 (the total mass) of each cell's blocks; Eigen stores
    // them column-major
    int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
    auto entry = [&](torch::Tensor const& t) {
      return t.view({nx3, nx2, nx1, m, m}).select(-2, 0).select(-1, m - 1);
    };
    if (no_solid) {  // 1 * x == x
      entry(_a).sub_(rx_mid / dt);
      entry(_b).sub_(rx_lo / dt);
      entry(_c).sub_(rx_hi / dt);
    } else {
      entry(_a).sub_(rx_fluid * rx_mid / dt);
      entry(_b).sub_(rx_fluid * rx_lo / dt);
      entry(_c).sub_(rx_fluid * rx_hi / dt);
    }
    rx_mass0 = _du0[IDN].index(in3).clone();
    if (du.size(0) > ICY)
      rx_mass0 += _du0.narrow(0, ICY, du.size(0) - ICY).sum(0).index(in3);
    du[IPR].index(in3).sub_(rx_tri(rx_mass0));
  };

  bool full = (options->scheme() >> 3) & 1;
  if (full) {
    at::native::vic_assemble_full(du.device().type(), iter, dt, grav1, adir);
  } else {
    // Match the full-VIC pipeline: assemble coefficients, run the column
    // solve + reductions, then apply the per-cell redistribution map.
    at::native::vic_assemble_partial(du.device().type(), iter, dt, grav1, adir);
  }
  couple_radial_exact();
  // Woodbury: the wall third points (E row of cells 0 and n-1, mass column of
  // cells 2 and n-3) are a rank-2 update U V^T of the block-tridiagonal T:
  // x = y - Z (I + V^T Z)^{-1} V^T y, T y = r, T Z = U. One factorisation of
  // T per column solves y and both columns of Z (they depend on T only)
  bool woodbury = radial_exact && nx1 >= 3;
  if (!woodbury) {
    if (full)
      at::native::vic_solve_full(du.device().type(), iter, dt, grav1, 0);
    else
      at::native::vic_solve_partial(du.device().type(), iter, dt, grav1, 0);
  } else {
    int m = options->size(), nx2 = pcoord->options->nx2(),
        nx3 = pcoord->options->nx3();
    auto z0 = torch::empty_like(_delta), z1 = torch::empty_like(_delta);
    auto it = vic_config().add_input(z0).add_input(z1).build();
    if (full)
      at::native::vic_solve_woodbury_full(du.device().type(), it, dt, grav1, 0);
    else
      at::native::vic_solve_woodbury_partial(du.device().type(), it, dt, grav1,
                                             0);
    auto y = _delta.view({nx3, nx2, nx1, m});
    z0 = z0.view({nx3, nx2, nx1, m}), z1 = z1.view({nx3, nx2, nx1, m});
    auto k0 = -rx_t0 / dt, k1 = -rx_t1 / dt;
    if (!no_solid) {
      k0 = k0 * rx_fluid.select(-1, 0), k1 = k1 * rx_fluid.select(-1, nx1 - 1);
    }
    auto pick = [&](torch::Tensor const& v, int cell) {
      return v.select(-2, cell).select(-1, 0);
    };
    auto m00 = 1. + k0 * pick(z0, 2), m01 = k0 * pick(z1, 2);
    auto m10 = k1 * pick(z0, nx1 - 3), m11 = 1. + k1 * pick(z1, nx1 - 3);
    auto r0 = k0 * pick(y, 2), r1 = k1 * pick(y, nx1 - 3);
    auto det = m00 * m11 - m01 * m10;
    auto al0 = (m11 * r0 - m01 * r1) / det, al1 = (m00 * r1 - m10 * r0) / det;
    y.sub_(z0 * al0.unsqueeze(-1).unsqueeze(-1))
        .sub_(z1 * al1.unsqueeze(-1).unsqueeze(-1));
  }

  // Both dispatches share the same rejection sentinel. This also catches
  // nonfinite backward-substitution results, before redistribution sees them.
  screen.push_back(_delta.to(torch::kFloat64).sum());
  if (!std::isfinite(torch::stack(screen).sum().item<double>())) {
    // a stage after a failed one stops here
    if (_failed_dev.item<bool>()) return reject(torch::Tensor());
    auto w_in = w.clone();  // the inputs, then the solve
    w_in.narrow(0, IVY, 2).copy_(w0);
    auto bad_inputs =
        torch::logical_not(finite_columns(_du0) & finite_columns(w_in) &
                           finite_columns(gamma.unsqueeze(0)));
    if (bad_inputs.any().item<bool>()) return reject(bad_inputs);
    auto bad_columns = torch::logical_not(torch::isfinite(_delta).all(-1));
    if (bad_columns.any().item<bool>()) return reject(bad_columns);
  }
  if ((options->scheme() >> 3) & 1)
    at::native::vic_redistribute_full(du.device().type(), iter, dt, grav1, 0);
  else
    at::native::vic_redistribute_partial(du.device().type(), iter, dt, grav1,
                                         0);

  // only the availability clamp breaks sum_ch MASS*VOL == M(i) - M(i+1);
  // the meters are committed below, unless the stage fails
  torch::Tensor clamp_next;
  {
    int is = pcoord->il();
    int ie = pcoord->iu() + 1;
    int nyc = du.size(0) - ICY;
    // no copy when there is nothing to add
    auto cell = nyc > 0 ? _mass_corr[IDN].clone() : _mass_corr[IDN];
    for (int n = 0; n < nyc; ++n) cell += _mass_corr[ICY + n];
    auto M = _mass_corr[IVX];
    auto Ml = M.slice(-1, is, ie);
    auto Mu = M.slice(-1, is + 1, ie + 1);
    auto lhs = (cell * volume).slice(-1, is, ie);  // cached
    auto rhs = Ml - Mu;
    // per cell: a column max would hide a binding in a low-flux layer
    // Keep the double floor unchanged; 1e-300 is zero in float32.
    const double floor = Ml.scalar_type() == torch::kFloat32
                             ? std::numeric_limits<float>::min()
                             : 1.e-300;
    auto scale = torch::maximum(Ml.abs(), Mu.abs()).clamp_min(floor);
    clamp_next = torch::maximum(_clamp_residual,
                                ((lhs - rhs).abs() / scale).max().detach());
  }
  auto dry_next =
      torch::maximum(_dry_clamp_step, _mass_corr[IPR].max().detach());

  if (face_work || diffusive_work) {
    int is = pcoord->il(), ie = pcoord->iu() + 1;
    auto in3 =
        phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto volume = _g_volume.index(in3);  // cached
    auto requested = _mass_corr[IVX];
    auto moved = _mass_corr[IVZ] - requested;
    auto projected =
        (requested.slice(-1, is, ie) - requested.slice(-1, is + 1, ie + 1))
            .slice(-2, pcoord->jl(), pcoord->ju() + 1)
            .slice(-3, pcoord->kl(), pcoord->ku() + 1) /
        volume;
    auto raw_mass = _delta
                        .view({pcoord->options->nx3(), pcoord->options->nx2(),
                               pcoord->options->nx1(), options->size()})
                        .select(-1, 0);
    auto explicit_mass = _du0[IDN].clone();
    if (w.size(0) > ICY)
      explicit_mass += _du0.narrow(0, ICY, w.size(0) - ICY).sum(0);
    auto phi = _g_phi;  // cached grid-only potentials
    auto projection_work =
        phi * (raw_mass - explicit_mass.index(in3) - projected);
    auto dp_lo = _g_dp_lo;
    auto dp_hi = _g_dp_hi;
    auto clamp_work = -(dp_hi * moved.slice(-1, is + 1, ie + 1) -
                        dp_lo * moved.slice(-1, is, ie))
                           .slice(-2, pcoord->jl(), pcoord->ju() + 1)
                           .slice(-3, pcoord->kl(), pcoord->ku() + 1) /
                      volume;
    if (no_solid)  // where(true, x, 0) == x
      du[IPR].index(in3).add_(projection_work + clamp_work);
    else
      du[IPR].index(in3).add_(
          torch::where(mask.index(in3) == 0, projection_work + clamp_work, 0.));
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  pcoord->flux2global1_(du);

  // gravity-work: face: the matrix books the face form for the
  // mass this solve moved and the corrected potential energy's remainder
  // (derivation sec 7) for its raw density change, with the exact 3-point
  // slope s (tridiagonal part in the blocks, the two wall third points by
  // Woodbury); add the rest, so the remainder is g1 sigma^2 s[du - du0] and
  // E + P closes over the explicit and implicit parts
  if (radial_exact) {
    int is = pcoord->il(), ie = pcoord->iu() + 1;
    auto moved = du[IDN] - _du0[IDN];
    if (du.size(0) > ICY)
      moved += (du.narrow(0, ICY, du.size(0) - ICY) -
                _du0.narrow(0, ICY, du.size(0) - ICY))
                   .sum(0);
    // the whole column (no x1 split under an implicit scheme, see above), so
    // the slope is one-sided only at the two walls
    auto work = corrected_pe_work(
        moved.slice(-1, is, ie), pcoord->x1f, pcoord->x1v, is, ie, grav1,
        x1_measure(pcoord->options->type()), &_rx_work);
    if (rx_mass0.defined()) {
      int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
      auto raw = _delta.view({nx3, nx2, nx1, options->size()}).select(-1, 0);
      work.slice(-2, pcoord->jl(), pcoord->ju() + 1)
          .slice(-3, pcoord->kl(), pcoord->ku() + 1)
          .sub_(rx_tri(raw - rx_mass0));
    }
    // solid cells get none: their slope stencil reads fluid neighbours
    if (no_solid)
      du[IPR].slice(-1, is, ie) += work;
    else
      du[IPR].slice(-1, is, ie) +=
          torch::where(mask.slice(-1, is, ie) == 0, work, 0.);
  }

  _corr.copy_(du);
  _corr.sub_(_du0);

  auto results =
      torch::stack({du.index(interior).to(torch::kFloat64).sum(),
                    _mass_corr.index(interior).to(torch::kFloat64).sum(),
                    _corr.index(interior).to(torch::kFloat64).sum()});
  // no host read: a nonfinite result is latched on the device, undone here
  // with masks (on success they select the values as they are) and stops
  // the next stage at its screen; check_redo reads the latch
  auto fail = torch::isfinite(results.sum()).logical_not();
  _failed_dev.logical_or_(fail);
  du.copy_(torch::where(fail, _du0, du));
  w.narrow(0, IVY, 2).copy_(torch::where(fail, w0, w.narrow(0, IVY, 2)));
  _corr.masked_fill_(fail, 0.);
  _mass_corr.masked_fill_(fail, 0.);
  // a failed stage leaves the meters as they were
  _clamp_residual.copy_(torch::where(fail, _clamp_residual, clamp_next));
  _dry_clamp_step.copy_(torch::where(fail, _dry_clamp_step, dry_next));
  return _corr;
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p != nullptr, "[ImplicitHydro] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr, "[ImplicitHydro] Options pointer is nullptr");

  return p->register_module(name, ImplicitHydro(opts, p));
}

}  // namespace snap
