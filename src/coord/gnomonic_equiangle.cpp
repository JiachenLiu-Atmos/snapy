// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "coord_dispatch.hpp"
#include "coord_utils.hpp"
#include "cubed_sphere_utils.hpp"
#include "gnomonic_equiangle.hpp"

namespace snap {

//! contravariant components, in the unit (alpha, beta) basis, of the
//! unit-radius rotation field k = z_hat x n at the points (alpha, beta) of
//! face face_id; returns [3, alpha.sizes()...], radial component zero
static torch::Tensor cs_axial_rotation_field(torch::Tensor alpha,
                                             torch::Tensor beta, int face_id) {
  std::vector<int64_t> shape = {3};
  for (auto s : alpha.sizes()) shape.push_back(s);
  // unit position n: the radial unit vector, in Cartesian (z, x, y)
  auto n = torch::zeros(shape, alpha.options());
  n[VEL1].fill_(1.);
  cs_contra_to_cart_(n, alpha, beta, face_id);
  // z_hat x n = (-n_y, n_x, 0) in (x, y, z), ordered (z, x, y) here
  auto k = torch::zeros_like(n);
  k[VEL2] = -n[VEL3];
  k[VEL3] = n[VEL2];
  cs_cart_to_contra_(k, alpha, beta, face_id);
  return k;
}

void GnomonicEquiangleImpl::reset() {
  TORCH_CHECK(pmb, "[GnomonicEquiangle] Parent MeshBlock is null");

  auto const& op = options;
  TORCH_CHECK(op->nx2() == op->nx3(),
              "GnomonicEquiangleImpl::reset(): nx2 must equal nx3");

  TORCH_CHECK(op->interp_order() == 2,
              "Only 2nd order ghost interpolation is supported");
  TORCH_CHECK(
      op->interp_order() <= 2 * op->nghost(),
      "Ghost zone size must be at least half of the interpolation order");

  // dimension 1
  auto [df, xv, dv] = x1_metrics_();
  dx1f = register_buffer("dx1f", df);
  x1v = register_buffer("x1v", xv);
  dx1v = register_buffer("dx1v", dv);

  // dimension 2
  auto dx = op->dx2();
  dx2f = register_buffer("dx2f", torch::ones(op->nc2(), torch::kFloat64) * dx);
  x2v = register_buffer("x2v", 0.5 * (x2f.slice(0, 0, op->nc2()) +
                                      x2f.slice(0, 1, op->nc2() + 1)));
  dx2v = register_buffer("dx2v", torch::ones(op->nc2(), torch::kFloat64) * dx);

  // dimension 3
  dx = op->dx3();
  dx3f = register_buffer("dx3f", torch::ones(op->nc3(), torch::kFloat64) * dx);
  x3v = register_buffer("x3v", 0.5 * (x3f.slice(0, 0, op->nc3()) +
                                      x3f.slice(0, 1, op->nc3() + 1)));
  dx3v = register_buffer("dx3v", torch::ones(op->nc3(), torch::kFloat64) * dx);

  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);

  auto mesh = torch::meshgrid({x3v, x2v}, "ij");
  auto rank = pmb->options->layout()->rank();
  auto face_id = pmb->options->layout()->type() == "cubed-sphere"
                     ? std::get<2>(pmb->get_layout()->loc_of(rank))
                     : 0;
  auto [longitude, latitude] =
      cs_ab_to_lonlat(CS_FACE_NAMES[face_id], mesh[1], mesh[0]);
  register_buffer("longitude", longitude.unsqueeze(-1));
  register_buffer("latitude", latitude.unsqueeze(-1));

  // populate and register geometry data
  auto x = x2v.tan().unsqueeze(0).unsqueeze(-1);
  auto xf = x2f.tan().unsqueeze(0).unsqueeze(-1);
  auto y = x3v.tan().unsqueeze(-1).unsqueeze(-1);
  auto yf = x3f.tan().unsqueeze(-1).unsqueeze(-1);

  auto C = (1.0 + x * x).sqrt();
  auto Cf = (1.0 + xf * xf).sqrt();
  auto D = (1.0 + y * y).sqrt();
  auto Df = (1.0 + yf * yf).sqrt();

  cosine_cell_kj = -x * y / (C * D);
  sine_cell_kj = (1.0 + x * x + y * y).sqrt() / (C * D);

  cosine_face2_kj = -xf * y / (Cf * D);
  sine_face2_kj = (1.0 + xf * xf + y * y).sqrt() / (Cf * D);

  cosine_face3_kj = -x * yf / (C * Df);
  sine_face3_kj = (1.0 + x * x + yf * yf).sqrt() / (C * Df);

  register_buffer("cosine_cell_kj", cosine_cell_kj);
  register_buffer("sine_cell_kj", sine_cell_kj);
  register_buffer("cosine_face2_kj", cosine_face2_kj);
  register_buffer("sine_face2_kj", sine_face2_kj);
  register_buffer("cosine_face3_kj", cosine_face3_kj);
  register_buffer("sine_face3_kj", sine_face3_kj);

  // the angle between the panel points (e, a) and (e, b) (tangents, e held
  // fixed) seen from the centre, written once: a seam face is then the same
  // number on both panels, whichever of x2/x3 is held fixed on each side
  auto arc = [](torch::Tensor const& e, torch::Tensor const& a,
                torch::Tensor const& b) {
    auto e2 = e * e;
    return ((1. + e2 + a * b) /
            ((1. + e2 + a * a).sqrt() * (1. + e2 + b * b).sqrt()))
        .acos();
  };

  auto x1 = x2f.slice(0, 0, op->nc2()).tan().unsqueeze(0).unsqueeze(-1);
  auto x2 = x2f.slice(0, 1, op->nc2() + 1).tan().unsqueeze(0).unsqueeze(-1);

  dx2f_ang_kj = arc(y, x1, x2);
  dx2f_ang_face3_kj = arc(yf, x1, x2);

  register_buffer("dx2f_ang_kj", dx2f_ang_kj);
  register_buffer("dx2f_ang_face3_kj", dx2f_ang_face3_kj);

  auto y1 = x3f.slice(0, 0, op->nc3()).tan().unsqueeze(-1).unsqueeze(-1);
  auto y2 = x3f.slice(0, 1, op->nc3() + 1).tan().unsqueeze(-1).unsqueeze(-1);

  dx3f_ang_kj = arc(x, y1, y2);
  dx3f_ang_face2_kj = arc(xf, y1, y2);

  register_buffer("dx3f_ang_kj", dx3f_ang_kj);
  register_buffer("dx3f_ang_face2_kj", dx3f_ang_face2_kj);

  // exact solid angle of a cell's x1 face: the cube-face rectangle
  // [tan x2f_j, tan x2f_{j+1}] x [tan x3f_k, tan x3f_{k+1}] seen from the
  // centre, the corner sum of atan(x y / sqrt(1 + x^2 + y^2)); the six panels
  // sum to 4 pi
  auto xc = x2f.tan().unsqueeze(0);
  auto yc = x3f.tan().unsqueeze(-1);
  auto corner = (xc * yc / (1. + xc * xc + yc * yc).sqrt()).atan();
  int nc2 = op->nc2(), nc3 = op->nc3();
  solid_angle_kj = (corner.slice(0, 1, nc3 + 1).slice(1, 1, nc2 + 1) -
                    corner.slice(0, 1, nc3 + 1).slice(1, 0, nc2) -
                    corner.slice(0, 0, nc3).slice(1, 1, nc2 + 1) +
                    corner.slice(0, 0, nc3).slice(1, 0, nc2))
                       .unsqueeze(-1);
  register_buffer("solid_angle_kj", solid_angle_kj);

  auto fx = face_area2() * sine_face2_kj;
  auto fy = face_area3() * sine_face3_kj;

  x_ov_rD_kji = (fx.slice(1, 1, op->nc2() + 1) - fx.slice(1, 0, op->nc2())) /
                cell_volume();
  y_ov_rC_kji = (fy.slice(0, 1, op->nc3() + 1) - fy.slice(0, 0, op->nc3())) /
                cell_volume();

  register_buffer("x_ov_rD_kji", x_ov_rD_kji);
  register_buffer("y_ov_rC_kji", y_ov_rC_kji);

  // axial rotation field (see the header and conserve_axial_momentum_)
  {
    auto c = torch::meshgrid({x3v, x2v}, "ij");
    aam_k_c = cs_axial_rotation_field(c[1], c[0], face_id).unsqueeze(-1);
    aam_kl_c = aam_k_c.clone();
    coord_vec_lower_(aam_kl_c, cosine_cell_kj);
    aam_kk_c = (aam_k_c * aam_kl_c).sum(0);
    // Exact face averages of the unit-radius field on the x2 and x3 faces.
    // Each such face is a planar sector (the gnomonic plane through the
    // centre) between two corner directions p1, p2 at arc angle dth, so
    //   (1/A) int k dA = [2 sin(dth/2) / dth] k(p_mid),  p_mid = (p1 + p2)/|.|
    // per unit radius; conserve_axial_momentum_ supplies the radial factor.
    // With this average the pressure flux p n A of the four lateral faces
    // sums to p times the exact flux of k through the closed cell, zero.
    auto corner = torch::meshgrid({x3f, x2f}, "ij");
    auto pc = torch::zeros({3, op->nc3() + 1, op->nc2() + 1}, x1v.options());
    pc[VEL1].fill_(1.);
    cs_contra_to_cart_(pc, corner[1], corner[0], face_id);  // unit positions
    auto face_avg = [&](torch::Tensor p1, torch::Tensor p2, torch::Tensor dth,
                        torch::Tensor alpha, torch::Tensor beta) {
      auto pm = p1 + p2;
      pm = pm / pm.square().sum(0, true).sqrt();
      auto k = torch::zeros_like(pm);  // z_hat x p_mid, ordered (z, x, y)
      k[VEL2] = -pm[VEL3];
      k[VEL3] = pm[VEL2];
      cs_cart_to_contra_(k, alpha, beta, face_id);
      auto fac = 2. * (0.5 * dth).sin() / dth;
      return (k * fac.unsqueeze(0)).unsqueeze(-1);
    };
    auto f2 = torch::meshgrid({x3v, x2f}, "ij");
    aam_k_f2 =
        face_avg(pc.slice(1, 0, op->nc3()), pc.slice(1, 1, op->nc3() + 1),
                 dx3f_ang_face2_kj.squeeze(-1), f2[1], f2[0]);
    auto f3 = torch::meshgrid({x3f, x2v}, "ij");
    aam_k_f3 =
        face_avg(pc.slice(2, 0, op->nc2()), pc.slice(2, 1, op->nc2() + 1),
                 dx2f_ang_face3_kj.squeeze(-1), f3[1], f3[0]);
    register_buffer("aam_k_c", aam_k_c);
    register_buffer("aam_kl_c", aam_kl_c);
    register_buffer("aam_kk_c", aam_kk_c);
    register_buffer("aam_k_f2", aam_k_f2);
    register_buffer("aam_k_f3", aam_k_f3);
    // only guards 0/0 in a cell centred exactly on the axis (odd nx2): that
    // cell holds no AAM, its correction is zero, and the AAM flux through its
    // faces is the one leak of the flux form (one cell per polar panel)
    _aam_eps2 = 1.e-24;
  }

  // register metric data (placeholder, overwritten later)
  auto vol = cell_volume();
  g11 = register_buffer("g11", torch::ones_like(vol));
  g22 = register_buffer("g22", torch::ones_like(vol));
  g33 = register_buffer("g33", torch::ones_like(vol));
  gi11 = register_buffer("gi11", torch::ones_like(vol));
  gi22 = register_buffer("gi22", torch::ones_like(vol));
  gi33 = register_buffer("gi33", torch::ones_like(vol));
  g12 = register_buffer("g12", torch::zeros_like(vol));
  g13 = register_buffer("g13", torch::zeros_like(vol));
  g23 = register_buffer("g23", torch::zeros_like(vol));

  // build global ghost cell usrc
  int N, offset_x, offset_y;
  torch::Tensor usrc;
  N = op->nx2() * pmb->options->layout()->px();
  usrc = torch::empty({op->nghost(), N}, torch::kFloat64);
  cs_build_ghost_usrc(usrc.data_ptr<double>(), N, op->nghost());

  int my_rank = pmb->options->layout()->rank();
  auto [rx, ry, _] = pmb->get_layout()->loc_of(my_rank);
  offset_x = op->nx2() * rx;
  offset_y = op->nx3() * ry;

  // register local ghost cell usrc.
  // The strip these indices address is widened by cs_interp_margin() on each
  // side (see CubedSphereLayout::{serialize,deserialize} and interp_ghost
  // below), so rebase by that -- NOT by interp_order/2, which only ever covered
  // a seam at beta = 0.
  int margin = cs_interp_margin(op->nghost());
  // interp_order > 2 reaches u0-1, one cell beyond what the exchange
  // carries, at every px -- so px == 1 is not an escape.
  TORCH_CHECK(options->interp_order() <= 2,
              "GnomonicEquiangle: interp_order > 2 needs more tangential "
              "margin than the cross-panel exchange carries; use "
              "interp_order: 2.");

  // Keep the GLOBAL source coordinate and carry the block's shift as an
  // INTEGER: folding `margin - offset` into the double makes the stored value
  // depend on the decomposition, since `offset` differs per block.
  usrc_BT =
      register_buffer("usrc_BT", usrc.narrow(-1, offset_x, op->nx2()).clone());
  usrc_shift_BT = margin - offset_x;

  usrc_LR = register_buffer(
      "usrc_LR", usrc.narrow(-1, offset_y, op->nx3()).transpose(0, 1).clone());
  usrc_shift_LR = margin - offset_y;
}

torch::Tensor GnomonicEquiangleImpl::center_width2() const {
  return x1v.unsqueeze(0).unsqueeze(1) * dx2f_ang_kj;
}

torch::Tensor GnomonicEquiangleImpl::center_width3() const {
  return x1v.unsqueeze(0).unsqueeze(1) * dx3f_ang_kj;
}

torch::Tensor GnomonicEquiangleImpl::face_area1() const {
  return (x1f * x1f).unsqueeze(0).unsqueeze(1) * solid_angle_kj;
}

torch::Tensor GnomonicEquiangleImpl::face_area2() const {
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx3f_ang_face2_kj;
}

torch::Tensor GnomonicEquiangleImpl::face_area3() const {
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx2f_ang_face3_kj;
}

torch::Tensor GnomonicEquiangleImpl::cell_volume() const {
  // exact: solid angle times (rp^3 - rm^3) / 3, written as
  // dr (rp^2 + rp rm + rm^2) / 3 so that thin shells do not cancel
  auto rm = x1f.slice(0, 0, options->nc1());
  auto rp = x1f.slice(0, 1, options->nc1() + 1);
  auto radial =
      (dx1f * (rp * rp + rp * rm + rm * rm) / 3.0).unsqueeze(0).unsqueeze(1);
  return radial * solid_angle_kj;
}

torch::Tensor GnomonicEquiangleImpl::face_moment2_x1() const {
  // face_area2 = (x1v * dx1f) * (angular arc) = 0.5 (rp^2 - rm^2) * arc, and
  // face_area3 likewise: both are the r-weighted radial measure times an
  // angular factor that does not vary along x1 on one face. The angular factor
  // carries the non-orthogonal metric and sqrt(g) and cancels in the average.
  return radial_face_moment2_(x1f, options->nc1());
}

torch::Tensor GnomonicEquiangleImpl::face_centroid_shift_x1() const {
  // Same two geometric measures as spherical-polar: the cell is r^2 dr times
  // an angular factor and an x2/x3 face is r dr times one, so the shift is
  // the same. cell_volume() above is the exact radial integral (rp^3-rm^3)/3
  // times the same solid angle as face_area1().
  return radial_face_centroid_shift_(x1f, options->nc1());
}

void GnomonicEquiangleImpl::interp_ghost(
    torch::Tensor var, std::tuple<int, int, int> const& offset) const {
  auto [dy, dx, dz] = offset;
  auto sub = pmb->part(offset, PartOptions().exterior(true).ndim(var.dim()));
  // must match the widening done by CubedSphereLayout::{serialize,deserialize}
  auto margin = cs_interp_margin(options->nghost());

  if (dy != 0 && dx == 0) {
    auto sub1 = pmb->part(
        offset, PartOptions().exterior(true).extend_x2(margin).ndim(var.dim()));
    var.index(sub) = _interp_ghost_BT(var.index(sub1), dy > 0);
  }

  if (dx != 0 && dy == 0) {
    auto sub1 = pmb->part(
        offset, PartOptions().exterior(true).extend_x3(margin).ndim(var.dim()));
    var.index(sub) = _interp_ghost_LR(var.index(sub1), dx > 0);
  }
}

// TODO(cli):: CHECK
void GnomonicEquiangleImpl::_set_face2_metric() const {
  auto cos_theta = cosine_face2_kj.narrow(1, 0, options->nc2());
  auto sin_theta = sine_face2_kj.narrow(1, 0, options->nc2());

  g11.set_(torch::ones_like(cos_theta));
  g22.set_(torch::ones_like(cos_theta));
  g23.set_(cos_theta);
  g33.set_(torch::ones_like(cos_theta));

  gi11.set_(torch::ones_like(cos_theta));
  gi22.set_(1. / (sin_theta * sin_theta));
  gi33.set_(1. / (sin_theta * sin_theta));
}

// TODO(cli):: CHECK
void GnomonicEquiangleImpl::_set_face3_metric() const {
  auto cos_theta = cosine_face3_kj.narrow(0, 0, options->nc3());
  auto sin_theta = sine_face3_kj.narrow(0, 0, options->nc3());

  g11.set_(torch::ones_like(cos_theta));
  g22.set_(torch::ones_like(cos_theta));
  g23.set_(cos_theta);
  g33.set_(torch::ones_like(cos_theta));

  gi11.set_(torch::ones_like(cos_theta));
  gi22.set_(1. / (sin_theta * sin_theta));
  gi33.set_(1. / (sin_theta * sin_theta));
}

void GnomonicEquiangleImpl::prim2local1_(torch::Tensor const& w) const {
  auto cos_theta = cosine_cell_kj;
  auto sin_theta = sine_cell_kj;

  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;
}

void GnomonicEquiangleImpl::prim2local2_(torch::Tensor const& w) const {
  _set_face2_metric();

  // Extract global projected 4-velocities
  // auto uu1 = w[IVX].clone();
  auto uu2 = w[IVY];
  auto uu3 = w[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.;
  auto T22 = 1.0 / gi22.sqrt();
  auto T32 = g23 / g33.sqrt();
  auto T33 = g33.sqrt();

  // Transform projected velocities
  // w[IVX] = T11 * uu1;
  w[IVZ] = T32 * uu2 + T33 * uu3;
  w[IVY] = T22 * uu2;
}

void GnomonicEquiangleImpl::prim2local3_(torch::Tensor const& w) const {
  _set_face3_metric();

  // Extract global projected 4-velocities
  // auto uu1 = w[IVX].clone();
  auto uu2 = w[IVY];
  auto uu3 = w[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.;
  auto T22 = g22.sqrt();
  auto T23 = g23 / g22.sqrt();
  auto T33 = 1.0 / gi33.sqrt();

  // Transform projected velocities
  // w[IVX] = T11 * uu1;
  w[IVY] = T22 * uu2 + T23 * uu3;
  w[IVZ] = T33 * uu3;
}

// de-orthonormal and transforms to covariant form
void GnomonicEquiangleImpl::flux2global1_(torch::Tensor const& flux) const {
  auto cos_theta = cosine_cell_kj;
  auto sin_theta = sine_cell_kj;

  // Extract contravariant fluxes
  auto tz = flux[IVZ] / sin_theta;
  auto ty = flux[IVY] - tz * cos_theta;

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cos_theta;
  flux[IVZ] = tz + ty * cos_theta;
}

// de-orthonormal and transforms to covariant form
void GnomonicEquiangleImpl::flux2global2_(torch::Tensor const& flux) const {
  _set_face2_metric();

  // Extract local conserved quantities and fluxes
  // auto txx = flux[IVX].clone();
  auto txy = flux[IVY];
  auto txz = flux[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.0;
  auto T22 = gi22.sqrt();
  auto T32 = -gi22.sqrt() * g23 / g33;
  auto T33 = 1.0 / g33.sqrt();

  // Set fluxes
  // flux[IVX] = T11 * txx;
  flux[IVZ] = T32 * txy + T33 * txz;
  flux[IVY] = T22 * txy;

  // Extract contravariant fluxes
  auto ty = flux[IVY].clone();
  auto tz = flux[IVZ].clone();

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cosine_face2_kj.narrow(1, 0, options->nc2());
  flux[IVZ] = tz + ty * cosine_face2_kj.narrow(1, 0, options->nc2());
}

// de-orthonormal and transforms to covariant form
void GnomonicEquiangleImpl::flux2global3_(torch::Tensor const& flux) const {
  _set_face3_metric();

  // Extract local conserved quantities and fluxes
  // auto txx = flux[IVX];
  auto txy = flux[IVY];
  auto txz = flux[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.0;
  auto T22 = 1.0 / g22.sqrt();
  auto T23 = -g23 / g22 * gi33.sqrt();
  auto T33 = gi33.sqrt();

  // Set fluxes
  // flux[IVX] = T11 * txx;
  flux[IVY] = T22 * txy + T23 * txz;
  flux[IVZ] = T33 * txz;

  // Extract contravariant fluxes
  auto ty = flux[IVY].clone();
  auto tz = flux[IVZ].clone();

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cosine_face3_kj.narrow(0, 0, options->nc3());
  flux[IVZ] = tz + ty * cosine_face3_kj.narrow(0, 0, options->nc3());
}

void GnomonicEquiangleImpl::axial_rate_to_rows_(torch::Tensor row2,
                                                torch::Tensor row3,
                                                torch::Tensor target) const {
  int si = il(), ei = iu() + 1;
  int sj = jl(), ej = ju() + 1;
  int sk = kl(), ek = ku() + 1;
  int n2 = ej - sj, n3 = ek - sk;

  auto kc = aam_k_c.slice(1, sk, ek).slice(2, sj, ej);    // [3, n3, n2, 1]
  auto klc = aam_kl_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
  auto kk = aam_kk_c.slice(0, sk, ek).slice(1, sj, ej);   // [n3, n2, 1]
  auto r = x1v.slice(0, si, ei);                          // [n1]
  auto vol = cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);

  // A cell centred on the axis (odd nx2, one per polar panel) has K_c = 0: a
  // cell-mean momentum there carries no AAM, since int (z x r) . m dV = m .
  // (z x int r dV) = 0, and the AAM its faces bring in is sub-cell rotation
  // the cell mean cannot represent. Its net rate is handed to its four
  // lateral neighbours in equal shares, so the total still telescopes. The
  // cell is interior to the panel (nx2 >= 3), so the shifts stay inside.
  auto onaxis = (kk < 1.e-20).to(target.options());  // [n3, n2, 1]
  if (onaxis.sum().item<double>() > 0.) {
    auto res = 0.25 * target * onaxis * vol;  // AAM rate per neighbour
    auto give = torch::zeros_like(target);
    give.slice(1, 1, n2) += res.slice(1, 0, n2 - 1);
    give.slice(1, 0, n2 - 1) += res.slice(1, 1, n2);
    give.slice(0, 1, n3) += res.slice(0, 0, n3 - 1);
    give.slice(0, 0, n3 - 1) += res.slice(0, 1, n3);
    target = target * (1. - onaxis) + give / vol;
  }

  // replace the K component of the rows: K . rows_new = target, with the
  // change along K_lower = r k_lower (a zonal acceleration)
  auto d2 = row2.slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  auto d3 = row3.slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  auto kd = r * (kc[VEL2] * d2 + kc[VEL3] * d3);
  auto lam = (target - kd) / (r * r * (kk + _aam_eps2));
  d2 += lam * r * klc[VEL2];
  d3 += lam * r * klc[VEL3];
}

void GnomonicEquiangleImpl::axial_x1_source_(torch::Tensor row2,
                                             torch::Tensor row3,
                                             torch::Tensor const& corr) const {
  int si = il(), ei = iu() + 1;
  int sj = jl(), ej = ju() + 1;
  int sk = kl(), ek = ku() + 1;
  int n1 = ei - si, n2 = ej - sj, n3 = ek - sk;

  auto target = torch::zeros({n3, n2, n1}, row2.options());
  if (corr.defined() && corr.numel() > 0) {
    auto kc = aam_k_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
    auto vol = cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
    auto c = corr.slice(1, sk, ek).slice(2, sj, ej).slice(3, si, ei);
    // T = P_lo - P_hi: k . (tangential momentum the solve gave the cell);
    // P = 0 at the bottom face, P_hi = P_lo - T up the column
    auto T = (kc[VEL2] * c[IVY] + kc[VEL3] * c[IVZ]) * vol;
    auto Phi = -T.cumsum(-1);
    auto Plo = Phi + T;
    auto r = x1v.slice(0, si, ei);
    auto rlo = x1f.slice(0, si, ei), rhi = x1f.slice(0, si + 1, ei + 1);
    target = ((rlo - r) * Plo - (rhi - r) * Phi) / vol;
  }
  axial_rate_to_rows_(row2, row3, target);
}

void GnomonicEquiangleImpl::conserve_axial_momentum_(
    torch::Tensor div, torch::Tensor flux1, torch::Tensor flux2,
    torch::Tensor flux3) const {
  int si = il(), ei = iu() + 1;
  int sj = jl(), ej = ju() + 1;
  int sk = kl(), ek = ku() + 1;
  int n1 = ei - si, n2 = ej - sj, n3 = ek - sk;

  // interior cells: rotation field, radius and volume
  auto kc = aam_k_c.slice(1, sk, ek).slice(2, sj, ej);  // [3, n3, n2, 1]
  auto r = x1v.slice(0, si, ei);                        // [n1]
  auto vol = cell_volume().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei);
  // the exact area-weighted radius of a lateral face, 2 (r+^3 - r-^3) /
  // (3 (r+^2 - r-^2)): with the unit-radius face average of k this is the
  // exact face average of K = r k
  auto rm = x1f.slice(0, si, ei), rp = x1f.slice(0, si + 1, ei + 1);
  auto reff = 2. * (rp * rp * rp - rm * rm * rm) / (3. * (rp * rp - rm * rm));

  // divergence of the face AAM fluxes A_f F_f . K_f; the covariant face
  // fluxes of x1 faces are in the cell basis (flux2global1_), those of x2 and
  // x3 faces in the face's basis (flux2global2_, 3_). On the x1 faces K is
  // tangent, so the pressure flux carries no AAM and the cell value of k
  // serves; K_f is single-valued on every face, so the sums telescope.
  auto phidiv = torch::zeros({n3, n2, n1}, div.options());
  if (flux1.defined()) {
    auto A =
        face_area1().slice(0, sk, ek).slice(1, sj, ej).slice(2, si, ei + 1);
    auto f = flux1.slice(1, sk, ek).slice(2, sj, ej).slice(3, si, ei + 1);
    auto phi =
        A * x1f.slice(0, si, ei + 1) * (f[IVY] * kc[VEL2] + f[IVZ] * kc[VEL3]);
    phidiv += phi.slice(2, 1, n1 + 1) - phi.slice(2, 0, n1);
  }
  if (flux2.defined()) {
    auto A =
        face_area2().slice(0, sk, ek).slice(1, sj, ej + 1).slice(2, si, ei);
    auto kf = aam_k_f2.slice(1, sk, ek).slice(2, sj, ej + 1);
    auto f = flux2.slice(1, sk, ek).slice(2, sj, ej + 1).slice(3, si, ei);
    auto phi =
        A * reff * (f[IVX] * kf[VEL1] + f[IVY] * kf[VEL2] + f[IVZ] * kf[VEL3]);
    phidiv += phi.slice(1, 1, n2 + 1) - phi.slice(1, 0, n2);
  }
  if (flux3.defined()) {
    auto A =
        face_area3().slice(0, sk, ek + 1).slice(1, sj, ej).slice(2, si, ei);
    auto kf = aam_k_f3.slice(1, sk, ek + 1).slice(2, sj, ej);
    auto f = flux3.slice(1, sk, ek + 1).slice(2, sj, ej).slice(3, si, ei);
    auto phi =
        A * reff * (f[IVX] * kf[VEL1] + f[IVY] * kf[VEL2] + f[IVZ] * kf[VEL3]);
    phidiv += phi.slice(0, 1, n3 + 1) - phi.slice(0, 0, n3);
  }
  phidiv /= vol;

  axial_rate_to_rows_(div[IVY], div[IVZ], phidiv);
}

torch::Tensor GnomonicEquiangleImpl::forward(torch::Tensor prim,
                                             torch::Tensor flux1,
                                             torch::Tensor flux2,
                                             torch::Tensor flux3,
                                             torch::Tensor face_pressure1) {
  std::string eos_type = pmb->phydro->peos->options->type();
  auto div = CoordinateImpl::forward(prim, flux1, flux2, flux3);

  auto cosine = cosine_cell_kj;
  auto sine2 = sine_cell_kj.square();

  // General variables
  auto v2 = prim[IVY];
  auto v3 = prim[IVZ];
  auto radius = x1v.unsqueeze(0).unsqueeze(1);

  torch::Tensor pr, rho;

  auto v_2 = v2 + v3 * cosine;
  auto v_3 = v3 + v2 * cosine;

  if (eos_type == "shallow-water") {
    pr = 0.5 * prim[IDN].square();
    rho = prim[IDN];
  } else {
    pr = prim[IPR];
    rho = prim[IDN];
    auto src1 = rho * (v2 * v_2 + v3 * v_3) / radius;
    if (face_pressure1.defined()) {
      int si = il();
      int ei = iu() + 1;
      auto pressure_gradient = (face_pressure1.slice(-1, si + 1, ei + 1) -
                                face_pressure1.slice(-1, si, ei)) /
                               dx1f.slice(0, si, ei);
      src1.slice(-1, si, ei) += (CoordinateImpl::face_area1(si + 1, ei + 1) *
                                     face_pressure1.slice(-1, si + 1, ei + 1) -
                                 CoordinateImpl::face_area1(si, ei) *
                                     face_pressure1.slice(-1, si, ei)) /
                                    cell_volume().slice(-1, si, ei) -
                                pressure_gradient;
    } else {
      src1 += 2.0 * pr / radius;
    }
    div[IVX] -= src1;
  }

  // Update flux 2
  // the -rho v1 v_2 / r part is x1_mass_source_
  auto src2 = x_ov_rD_kji * (pr + rho * v3 * v3 * sine2);
  div[IVY] -= src2;

  // Update flux 3
  auto src3 = y_ov_rC_kji * (pr + rho * v2 * v2 * sine2);
  div[IVZ] -= src3;

  return div;
}

void GnomonicEquiangleImpl::x1_mass_source_(torch::Tensor du,
                                            torch::Tensor const& prim,
                                            torch::Tensor const& m1,
                                            double dt) const {
  // du[v_i] -= dt m1 v_i / r with the covariant v_2 = v2 + c v3, v_3 = v3 + c
  // v2
  auto c = cosine_cell_kj;
  auto mr = m1 / x1v.unsqueeze(0).unsqueeze(1);
  du[IVY].addcmul_(mr, torch::addcmul(prim[IVY], prim[IVZ], c), -dt);
  du[IVZ].addcmul_(mr, torch::addcmul(prim[IVZ], prim[IVY], c), -dt);
}

torch::Tensor GnomonicEquiangleImpl::_interp_ghost(torch::Tensor buf,
                                                   torch::Tensor usrc_t,
                                                   int shift, int dim) const {
  auto vec = usrc_t.sizes().vec();
  vec.push_back(1);
  for (int n = 0; n < buf.dim() - 3; n++) {
    vec.insert(vec.begin(), 1);
  }

  // weight from the GLOBAL coordinate; block shift applied to the integer index
  // only
  auto u0f = usrc_t.floor();
  auto x = (usrc_t - u0f).view(vec);
  auto u0 = u0f.to(torch::kInt64).view(vec) + shift;
  auto u1 = u0 + 1;

  // set the correct output dimensions
  vec.back() = buf.size(-1);
  for (int n = 0; n < buf.dim() - 3; n++) {
    vec[n] = buf.size(n);
  }

  auto buf0 = buf.gather(dim, u0.expand(vec));
  auto buf1 = buf.gather(dim, u1.expand(vec));

  // linear, written about buf0 so that equal neighbours give buf0 exactly
  // (interp_order 2; reset() refuses any other order)
  return buf0 + x * (buf1 - buf0);
}

torch::Tensor GnomonicEquiangleImpl::_interp_ghost_LR(torch::Tensor buf,
                                                      bool flip) const {
  return _interp_ghost(buf, flip ? usrc_LR : usrc_LR.flip(1), usrc_shift_LR,
                       -3);
}

torch::Tensor GnomonicEquiangleImpl::_interp_ghost_BT(torch::Tensor buf,
                                                      bool flip) const {
  return _interp_ghost(buf, flip ? usrc_BT : usrc_BT.flip(0), usrc_shift_BT,
                       -2);
}

}  // namespace snap
