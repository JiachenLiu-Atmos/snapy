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
// screen sums in FP64, except on MPS, which has no FP64
static torch::ScalarType acc_type(torch::Tensor const& t) {
  return t.is_mps() ? torch::kFloat32 : torch::kFloat64;
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename,
                                               bool /*verbose*/) {
  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return nullptr;
  auto intg = config["integration"];
  // pyharp's integrator reads this block too, so its keys are listed by name
  check_keys(intg, "integration",
             {"implicit-scheme", "implicit-advection-cfl", "shear-cfl",
              "implicit-precision", "implicit-fallback-fp64", "execution",
              "execution-strict"},
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
  // V3 I-6: the FP64 stage's own scratch (unregistered), so the FP32
  // stage's keeps its dtype and address (I.rst section 7, fixed storage)
  for (auto* t :
       {&_a64, &_b64, &_c64, &_delta64, &_du064, &_corr64, &_mass_corr64})
    *t = torch::empty({0}, torch::kFloat64);
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
  // V3 I-5/I-6: the stage's VicStatusBit per column, in the solve's dtype
  // (the kernel operand), and the step's status, int32, which survives a
  // change of dtype (the FP64 stage) or device
  auto st_shape = std::vector<int64_t>{w.size(1), w.size(2)};
  if (needs_reset(_stage_status, st_shape))  // unregistered scratch
    _stage_status = torch::empty(st_shape, w.options());
  auto i32 = w.options().dtype(torch::kInt32);
  if (!_status.defined() || _status.sizes().vec() != st_shape)
    _status = torch::zeros(st_shape, i32);
  else if (_status.device() != w.device())
    _status = _status.to(i32);
}

torch::Tensor ImplicitHydroImpl::col_status() const { return _status; }

torch::Tensor ImplicitHydroImpl::status() const {
  if (!_status.defined()) return torch::zeros({}, torch::kBool);
  // V3 I-6: a column whose last solve failed: the FP64 retry failed, or a
  // failure that no FP64 retry followed (an FP64 solve, or the fallback off)
  auto failed = (_status & kVicFailBits) != 0;
  auto retried = (_status & kVicFp64Retried) != 0;
  auto fp64_failed = (_status & kVicFp64Failed) != 0;
  return (fp64_failed | (failed & retried.logical_not())).any();
}

void ImplicitHydroImpl::reset_status() {
  if (_status.defined()) _status.zero_();
  if (_stage_failed.defined()) _stage_failed.zero_();
}

void ImplicitHydroImpl::swap_scratch_() {
  std::swap(_a, _a64);
  std::swap(_b, _b64);
  std::swap(_c, _c64);
  std::swap(_delta, _delta64);
  std::swap(_du0, _du064);
  std::swap(_corr, _corr64);
  std::swap(_mass_corr, _mass_corr64);
  std::swap(_stage_status, _stage_status64);
}

torch::Tensor ImplicitHydroImpl::merge_fp64(torch::Tensor const& retry) {
  // the FP64 stage's correction in the retried columns, the FP32 stage's
  // elsewhere, in the FP32 stage's buffers and dtype (I-6)
  auto r = retry.unsqueeze(-1);
  _corr.copy_(torch::where(r, _corr64.to(_corr.scalar_type()), _corr));
  _mass_corr.copy_(
      torch::where(r, _mass_corr64.to(_mass_corr.scalar_type()), _mass_corr));
  return _corr;
}

void ImplicitHydroImpl::count_fp64_columns(torch::Tensor const& retry) {
  _fp64_fallbacks.add_(retry.sum().to(_fp64_fallbacks.device()));
}

void ImplicitHydroImpl::report_failure(std::ostream& os) const {
  if (!_status.defined()) return;
  auto pcoord = phydro->pmb->pcoord;
  auto st = _status.to(torch::kCPU);
  auto columns = st.nonzero();
  auto a = st.accessor<int32_t, 2>();
  auto index = columns.accessor<int64_t, 2>();
  for (int64_t n = 0; n < columns.size(0); ++n) {
    int64_t k = index[n][0], j = index[n][1];
    int32_t bits = a[k][j];
    os << "[ImplicitHydro] rank=" << get_rank() << " VIC column=(" << k << ","
       << j
       << ") failed:" << (bits & kVicInputNonfinite ? " nonfinite input" : "")
       << (bits & kVicSingular ? " singular/near-singular" : "")
       << (bits & kVicResultNonfinite ? " nonfinite result" : "")
       << " step=" << phydro->pmb->cycle + 1
       << " retry=" << phydro->pmb->pintg->current_redo
       << "; check_redo will restore step input." << std::endl;
  }
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  return forward_masked(du, w, gamma, dt, torch::Tensor());
}

torch::Tensor ImplicitHydroImpl::forward_masked(
    torch::Tensor du, torch::Tensor w, torch::Tensor gamma, double dt,
    torch::Tensor solid, FluxRegister* reg, double post_dt,
    torch::Tensor const& retry) {
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

  TORCH_CHECK(phydro->options->grav(),
              "[ImplicitHydro] forcing does not have const-gravity");

  auto pcoord = phydro->pmb->pcoord;
  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered before implicit solve");
  }*/

  struct Swap {  // the FP64 stage's scratch for this call only
    ImplicitHydroImpl* self;
    bool on;
    ~Swap() {
      if (on) self->swap_scratch_();
    }
  } swap{this, retry.defined()};
  if (swap.on) swap_scratch_();
  ensure_workspace(w);
  _du0.copy_(du);
  _mass_corr.zero_();

  // restore exactly if this solve is rejected; the solve changes only these
  // two rows of w in place (the frame projection)
  auto w0 = w.narrow(0, IVY, 2).clone();

  // V3 I-5: no host read in the solve. Each column's thread sets its
  // VicStatusBit (input screen, forward sweep, solution); a failed stage, or
  // one after a failed stage, is undone below with device masks, and
  // check_redo reads the status OR (solve_failed_tensor); the failed columns
  // are logged on a redo only (report_failure)
  // V3 I-6: the predicated FP64 stage (retry defined) solves only the
  // columns in retry; every other column carries kVicSkipColumn, returns at
  // once in each kernel and is undone below. It runs on its own scratch.
  if (retry.defined())
    _stage_status.copy_(torch::where(retry, 0., double(kVicSkipColumn)));
  else
    _stage_status.zero_();
  auto status_in = _stage_status.slice(0, pcoord->kl(), pcoord->ku() + 1)
                       .slice(1, pcoord->jl(), pcoord->ju() + 1);
  auto per_column = [&](torch::Tensor const& t) {  // one value per column
    return t.unsqueeze(0).unsqueeze(-1).expand(du.index(interior).sizes());
  };

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  pcoord->prim2local1_(du);

  // the geometry below depends on the grid only
  bool no_solid = !solid.defined();
  _gw.update(*pcoord, w, gamma, phydro->options->grav()->grav1());
  auto mask =
      solid.defined() ? solid.to(w.options()).contiguous() : _gw.mask0();

  auto area = _gw.area();
  auto volume = _gw.volume();
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
        .add_owned_input(_gw.face_lo().index(interior))
        .add_owned_input(_gw.face_hi().index(interior))
        .add_owned_input(per_column(status_in));  // written per column
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
    return _gw.apply_rows(q, no_solid ? torch::Tensor() : rx_fluid, nx1);
  };
  auto couple_radial_exact = [&]() {
    if (!radial_exact || nx1 < 3) return;
    int is = pcoord->il(), ie = pcoord->iu() + 1, m = options->size();
    auto measure = x1_measure(pcoord->options->type());
    // the rows depend on the grid and gravity only: build them once (#291)
    X1GridKey key(pcoord->x1f, pcoord->x1v, is, ie, is, ie, grav1, measure);
    // the stencil and work factors of corrected_pe_work, which the
    // post-solve remainder reads from the same object. The slope is
    // one-sided only at the column's two physical walls: an implicit scheme
    // refuses an x1 split (HydroImpl::_apply_implicit_correction), so a VIC
    // block has no internal x1 seam and this per-block slope is the
    // one-block slope
    _gw.update_rows(key, w, nx1);
    rx_lo = _gw.k_lo();
    rx_mid = _gw.k_mid();
    rx_hi = _gw.k_hi();
    rx_t0 = _gw.k_t0();
    rx_t1 = _gw.k_t1();
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

  // assemble the coefficients, run the column solve, then apply the
  // per-cell redistribution map: one vic_stage call each
  auto stage = [&](VicStage s, int dir, bool woodbury = false) {
    VicParams prm;
    prm.stage = s;
    prm.n = options->size();
    prm.woodbury = woodbury;
    prm.dt = dt;
    prm.grav = grav1;
    prm.dir = dir;
    return prm;
  };
  at::native::vic_stage(du.device().type(), iter,
                        stage(VicStage::assemble, adir));
  couple_radial_exact();
  // Woodbury: the wall third points (E row of cells 0 and n-1, mass column of
  // cells 2 and n-3) are a rank-2 update U V^T of the block-tridiagonal T:
  // x = y - Z (I + V^T Z)^{-1} V^T y, T y = r, T Z = U. One factorisation of
  // T per column solves y and both columns of Z (they depend on T only)
  bool woodbury = radial_exact && nx1 >= 3;
  if (!woodbury) {
    at::native::vic_stage(du.device().type(), iter, stage(VicStage::solve, 0));
  } else {
    // the 2x2 correction runs in the kernel, after the back substitution of
    // each column (V3 I-5), with its wall coefficients one value per column
    int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
    auto z0 = torch::empty_like(_delta), z1 = torch::empty_like(_delta);
    auto one = torch::ones({}, w.options());
    auto it =
        vic_config()
            .add_input(z0)
            .add_input(z1)
            .add_owned_input(per_column(rx_t0.expand({nx3, nx2})))
            .add_owned_input(per_column(rx_t1.expand({nx3, nx2})))
            .add_owned_input(per_column(no_solid ? one.expand({nx3, nx2})
                                                 : rx_fluid.select(-1, 0)))
            .add_owned_input(per_column(no_solid
                                            ? one.expand({nx3, nx2})
                                            : rx_fluid.select(-1, nx1 - 1)))
            .build();
    at::native::vic_stage(du.device().type(), it,
                          stage(VicStage::solve, 0, /*woodbury=*/true));
  }

  at::native::vic_stage(du.device().type(), iter,
                        stage(VicStage::redistribute, 0));

  // V3 I-4: the face transfers go to the register. A face i of total(1) is
  // the face below cell i, as MASS(IVX+dir, i); the posts are M_f / (A_f dt)
  // with the step's dt (I.rst section 4, "Frame and units")
  bool post = reg != nullptr && post_dt > 0. && reg->keeps(1, "vic.mass_total");
  bool partial = options->matrix() == VicMatrix::partial;
  torch::Tensor kdt, ptan;
  if (post) {
    kdt = pcoord->face_area1().narrow(-1, 0, nc1).to(_mass_corr.options()) *
          post_dt;
    // partial matrix: the donor-cell tangential transfers of
    // vic_tangential_momentum_column, q w_t(donor), in the local frame
    if (partial) {
      auto q = _mass_corr[IVZ];
      auto wt = w.narrow(0, IVY, 2);
      ptan = q * torch::where(q > 0., wt.roll(1, -1), wt);
    }
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  pcoord->flux2global1_(du);

  // the post-solve gravity-work remainder, one call (V3 I-3); I-4: it reads
  // the moved total mass from the kept "vic.mass_total"
  torch::Tensor moved_total = _mass_corr[IVZ], energy_solved, kept_before;
  if (post) {
    // the FP64 stage merges its columns into the FP32 stage's kept total
    if (retry.defined())
      kept_before = reg->kept(1, "vic.mass_total")[IDN].clone();
    reg->keep(1, "vic.mass_total", _mass_corr[IVZ] / kdt);
    moved_total =
        reg->kept(1, "vic.mass_total")[IDN].to(_mass_corr.options()) * kdt;
    energy_solved = du[IPR].clone();  // vic.cell: without the remainder
  }
  book_remainder_(du, w.size(0), mask, no_solid, face_work, radial_exact, grav1,
                  rx_mass0, rx_fluid, nx1, moved_total);

  _corr.copy_(du);
  _corr.sub_(_du0);

  // a nonfinite result after the solve (redistribution, remainder) sets the
  // column's bit too; the stage fails if any column has a bit, this stage's
  // or an earlier one's: undone with masks (on success they select the
  // values as they are)
  auto bad = [&](torch::Tensor const& t) {
    return torch::isfinite(t.index(interior)).logical_not().any(0).any(-1);
  };
  auto result_bad = bad(du) | bad(_mass_corr) | bad(_corr);
  status_in.copy_(torch::where(
      result_bad,
      (status_in.to(torch::kInt32) | kVicResultNonfinite).to(w.options()),
      status_in));
  // V3 I-6: undone per column, a column that failed in this stage or that
  // the FP64 stage skipped; the others keep their solution (on success the
  // masks select the values as they are)
  auto sbits = _stage_status.to(torch::kInt32);
  auto failed = (sbits & kVicFailBits) != 0;
  auto drop = failed | ((sbits & kVicSkipColumn) != 0);
  auto drop3 = drop.unsqueeze(-1);
  du.copy_(torch::where(drop3, _du0, du));
  w.narrow(0, IVY, 2).copy_(torch::where(drop3, w0, w.narrow(0, IVY, 2)));
  _corr.masked_fill_(drop3, 0.);
  _mass_corr.masked_fill_(drop3, 0.);
  // only the availability clamp breaks sum_ch MASS*VOL == M(i) - M(i+1);
  // the meters read the kept columns only: an undone column's mass
  // correction is zero (I-6; before, a failed stage left them as they were)
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
  _clamp_residual.copy_(clamp_next);
  _dry_clamp_step.copy_(dry_next);
  // the step's status: the FP32 (or only) stage's bits; the FP64 stage marks
  // its columns retried and, if they failed again, FP64-failed
  if (retry.defined()) {
    auto fp64 = torch::where(failed, (sbits | kVicFp64Failed) & ~kVicSkipColumn,
                             torch::zeros_like(sbits)) |
                kVicFp64Retried;
    _status.bitwise_or_(torch::where(retry, fp64, torch::zeros_like(sbits)));
  } else {
    _status.bitwise_or_(sbits);
    _stage_failed = failed;
  }
  if (post)
    post_(*reg, du, drop3, kdt, ptan, energy_solved, volume, kept_before);
  return _corr;
}

void ImplicitHydroImpl::post_(FluxRegister& reg, torch::Tensor const& du,
                              torch::Tensor const& drop,
                              torch::Tensor const& kdt,
                              torch::Tensor const& ptan,
                              torch::Tensor const& energy_solved,
                              torch::Tensor const& volume,
                              torch::Tensor const& kept_before) {
  auto pcoord = phydro->pmb->pcoord;
  int is = pcoord->il(), ie = pcoord->iu() + 1;
  int ny = du.size(0) - ICY;
  // an undone column posts -0: x + (-0) is x for every x, so a column the
  // FP64 stage skipped leaves the FP32 stage's total as it is, bit for bit
  auto ok = [&](torch::Tensor const& t) {
    return torch::where(drop, torch::full({}, -0., t.options()), t);
  };
  auto interior_jk = [&](torch::Tensor t) {
    return t.slice(-2, pcoord->jl(), pcoord->ju() + 1)
        .slice(-3, pcoord->kl(), pcoord->ku() + 1);
  };

  // (1) the posts, in declared order; each add also fills its kept part
  reg.add(1, "vic.mass_dry", {IDN}, ok(_mass_corr[IVY] / kdt).unsqueeze(0));
  if (ny > 0) {
    // the species face transfers of pass 3b, from their cell increments:
    // q_{i+1} = q_i - MASS(ICY+n, i) VOL(i), q = 0 at the bottom wall; the
    // top wall moves nothing
    auto q = torch::zeros_like(_mass_corr.narrow(0, ICY, ny));
    if (ie - is > 1) {
      auto s = (_mass_corr.narrow(0, ICY, ny) * volume).slice(-1, is, ie - 1);
      q.slice(-1, is + 1, ie) = -s.cumsum(-1);
    }
    std::vector<int> rows(ny);
    for (int n = 0; n < ny; ++n) rows[n] = ICY + n;
    reg.add(1, "vic.mass_species", rows, ok(q / kdt));
  }
  // tangential momenta, in the explicit x1 flux's global components
  torch::Tensor pt;
  if (ptan.defined()) {  // partial matrix: donor transfers, local frame
    auto tmp = torch::zeros_like(du);
    tmp.narrow(0, IVY, 2).copy_(ok(ptan));
    pcoord->flux2global1_(tmp);
    pt = tmp.narrow(0, IVY, 2);
  } else {  // full matrix: the prefix sum of the cell increments from the
            // bottom wall, top face included (gnomonic axial_x1_source_)
    pt = torch::zeros_like(du.narrow(0, IVY, 2));
    auto T = interior_jk((_corr.narrow(0, IVY, 2) * volume).slice(-1, is, ie));
    interior_jk(pt).slice(-1, is + 1, ie + 1).copy_(-T.cumsum(-1));
  }
  reg.add(1, "vic.momentum_t", {IVY, IVZ}, ok(pt / kdt));

  // (2) the kept total, from the (undone: zeroed) moved masses; the FP64
  // stage keeps the FP32 stage's value in the columns it did not solve
  auto total = _mass_corr[IVZ] / kdt;
  if (kept_before.defined())
    total = torch::where(drop, kept_before.to(total.options()), total);
  reg.keep(1, "vic.mass_total", total);

  // (3) the amounts, V du of this stage's du: weight w2_s
  double w2 = reg.weight2();
  int nvar = du.size(0);
  auto in = [&](torch::Tensor t) { return interior_jk(t.slice(-1, is, ie)); };
  auto cell = torch::zeros_like(du);
  cell[IVX] = du[IVX] - _du0[IVX];
  cell[IPR] = ok(energy_solved - _du0[IPR]);
  if (!ptan.defined()) cell.narrow(0, IVY, 2) = _corr.narrow(0, IVY, 2);
  reg.cell_amount("vic.cell", in(cell * volume), w2, AmountKind::cell_source);
  // V div(moved - requested) of the total mass, on row IDN
  auto d = _mass_corr[IVZ] - _mass_corr[IVX];
  auto clamped = torch::zeros({nvar}, du.options());
  clamped[IDN] =
      interior_jk(d.slice(-1, is + 1, ie + 1) - d.slice(-1, is, ie)).sum();
  reg.cell_amount("vic.mass_clamped", clamped, w2, AmountKind::flux_correction);
  if (!ptan.defined()) {  // the top face of the prefix sum (scheme 9)
    auto top = torch::zeros({nvar}, du.options());
    top.narrow(0, IVY, 2) = interior_jk(pt.narrow(-1, ie, 1)).sum({-3, -2, -1});
    reg.cell_amount("vic.momentum_t.top", top, w2, AmountKind::diagnostic);
  }
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p != nullptr, "[ImplicitHydro] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr, "[ImplicitHydro] Options pointer is nullptr");

  return p->register_module(name, ImplicitHydro(opts, p));
}

void ImplicitHydroImpl::book_remainder_(torch::Tensor du, int wsize,
                                        torch::Tensor const& mask,
                                        bool no_solid, bool face_work,
                                        bool radial_exact, double grav1,
                                        torch::Tensor const& rx_mass0,
                                        torch::Tensor const& rx_fluid, int nx1,
                                        torch::Tensor const& moved_total) {
  auto pcoord = phydro->pmb->pcoord;
  // du[IPR] is a scalar row: the frame de-projection before this call
  // (flux2global1_) leaves it as it was, so the order of the two terms below
  // is the order they had when the first was booked before it

  // the projection's and the availability clamp's work, from the face masses
  // the solve requested and moved
  if (face_work) {
    int is = pcoord->il(), ie = pcoord->iu() + 1;
    auto in3 =
        phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto volume = _gw.volume().index(in3);  // cached
    auto requested = _mass_corr[IVX];
    auto moved = moved_total - requested;
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
    if (wsize > ICY) explicit_mass += _du0.narrow(0, ICY, wsize - ICY).sum(0);
    auto phi = _gw.phi();  // cached grid-only potentials
    auto projection_work =
        phi * (raw_mass - explicit_mass.index(in3) - projected);
    auto dp_lo = _gw.dp_lo();
    auto dp_hi = _gw.dp_hi();
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
        x1_measure(pcoord->options->type()), _gw.pe_factors());
    if (rx_mass0.defined()) {
      int nx2 = pcoord->options->nx2(), nx3 = pcoord->options->nx3();
      auto raw = _delta.view({nx3, nx2, nx1, options->size()}).select(-1, 0);
      work.slice(-2, pcoord->jl(), pcoord->ju() + 1)
          .slice(-3, pcoord->kl(), pcoord->ku() + 1)
          .sub_(_gw.apply_rows(raw - rx_mass0,
                               no_solid ? torch::Tensor() : rx_fluid, nx1));
    }
    // solid cells get none: their slope stencil reads fluid neighbours
    if (no_solid)
      du[IPR].slice(-1, is, ie) += work;
    else
      du[IPR].slice(-1, is, ie) +=
          torch::where(mask.slice(-1, is, ie) == 0, work, 0.);
  }
}

}  // namespace snap
