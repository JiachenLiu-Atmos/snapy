// C/C++
#include <cmath>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "spherical_polar.hpp"

namespace snap {

namespace {

torch::Tensor radial_centers(torch::Tensor const& x1f, int nc1) {
  auto rm = x1f.slice(0, 0, nc1);
  auto rp = x1f.slice(0, 1, nc1 + 1);
  return 0.75 * (rp.pow(4) - rm.pow(4)) / (rp.pow(3) - rm.pow(3));
}

torch::Tensor polar_centers(torch::Tensor const& x2f, int nc2) {
  auto tm = x2f.slice(0, 0, nc2);
  auto tp = x2f.slice(0, 1, nc2 + 1);
  return ((tp.sin() - tp * tp.cos()) - (tm.sin() - tm * tm.cos())) /
         (tm.cos() - tp.cos());
}

torch::Tensor center_distances(torch::Tensor const& xv,
                               torch::Tensor const& dxf) {
  auto dxv = dxf.clone();
  if (xv.size(0) > 1) {
    dxv.slice(0, 1, xv.size(0))
        .copy_(xv.slice(0, 1, xv.size(0)) - xv.slice(0, 0, xv.size(0) - 1));
  }
  return dxv;
}

}  // namespace

void SphericalPolarImpl::reset() {
  TORCH_CHECK(pmb, "[SphericalPolar] Parent MeshBlock is null");

  auto const& op = options;
  TORCH_CHECK(op->nx2() <= 1 || (op->x2min() >= 0.0 && op->x2max() <= M_PI),
              "Spherical-polar coordinates require x2 in [0, pi], got [",
              op->x2min(), ", ", op->x2max(), "]");

  auto opts = torch::TensorOptions().dtype(torch::kFloat64);

  cosine_cell_kj = register_buffer(
      "cosine_cell_kj", torch::zeros({op->nc3(), op->nc2(), 1}, opts));
  cosine_face2_kj = register_buffer(
      "cosine_face2_kj", torch::zeros({op->nc3(), op->nc2() + 1, 1}, opts));
  cosine_face3_kj = register_buffer(
      "cosine_face3_kj", torch::zeros({op->nc3() + 1, op->nc2(), 1}, opts));

  dx1f = register_buffer(
      "dx1f", x1f.slice(0, 1, op->nc1() + 1) - x1f.slice(0, 0, op->nc1()));
  x1v = register_buffer("x1v", radial_centers(x1f, op->nc1()));
  dx1v = register_buffer("dx1v", center_distances(x1v, dx1f));

  dx2f = register_buffer(
      "dx2f", x2f.slice(0, 1, op->nc2() + 1) - x2f.slice(0, 0, op->nc2()));
  x2v = register_buffer("x2v", op->nc2() == 1
                                   ? 0.5 * (x2f.slice(0, 0, op->nc2()) +
                                            x2f.slice(0, 1, op->nc2() + 1))
                                   : polar_centers(x2f, op->nc2()));
  dx2v = register_buffer("dx2v", center_distances(x2v, dx2f));

  dx3f = register_buffer(
      "dx3f", x3f.slice(0, 1, op->nc3() + 1) - x3f.slice(0, 0, op->nc3()));
  x3v = register_buffer("x3v", 0.5 * (x3f.slice(0, 0, op->nc3()) +
                                      x3f.slice(0, 1, op->nc3() + 1)));
  dx3v = register_buffer("dx3v", center_distances(x3v, dx3f));

  auto rm = x1f.slice(0, 0, op->nc1());
  auto rp = x1f.slice(0, 1, op->nc1() + 1);
  auto radial_volume = shell_volume_(rm, rp);
  auto radial_area23 = shell_area_(rm, rp);

  auto theta_m = x2f.slice(0, 0, op->nc2());
  auto theta_p = x2f.slice(0, 1, op->nc2() + 1);
  auto sin_m = theta_m.sin().abs();
  auto sin_p = theta_p.sin().abs();
  auto polar_volume = polar_measure_(theta_m, theta_p);

  coord_src1_i = register_buffer(
      "coord_src1_i",
      (radial_area23 / radial_volume).unsqueeze(0).unsqueeze(0));
  coord_src2_i = register_buffer(
      "coord_src2_i",
      (dx1f / ((rm + rp) * radial_volume)).unsqueeze(0).unsqueeze(0));
  coord_src1_j = register_buffer(
      "coord_src1_j",
      ((sin_p - sin_m) / polar_volume).unsqueeze(0).unsqueeze(-1));
  coord_src2_j = register_buffer(
      "coord_src2_j", ((sin_p - sin_m) / ((sin_m + sin_p) * polar_volume))
                          .unsqueeze(0)
                          .unsqueeze(-1));
  coord_src3_j = register_buffer(
      "coord_src3_j",
      ((sin_p - sin_m) / polar_volume).unsqueeze(0).unsqueeze(-1));

  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);
}

torch::Tensor SphericalPolarImpl::center_width2() const {
  return x1v.unsqueeze(0).unsqueeze(1) * dx2f.unsqueeze(0).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::center_width3() const {
  return x1v.unsqueeze(0).unsqueeze(1) *
         x2v.sin().abs().unsqueeze(0).unsqueeze(2) *
         dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::center_distance2() const {
  return x1v.unsqueeze(0).unsqueeze(1) * dx2v.unsqueeze(0).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::center_distance3() const {
  return x1v.unsqueeze(0).unsqueeze(1) *
         x2v.sin().abs().unsqueeze(0).unsqueeze(2) *
         dx3v.unsqueeze(1).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::face_area1() const {
  auto polar = polar_measure_(x2f.slice(0, 0, options->nc2()),
                              x2f.slice(0, 1, options->nc2() + 1));
  return x1f.square().unsqueeze(0).unsqueeze(1) *
         polar.unsqueeze(0).unsqueeze(2) * dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::face_area2() const {
  auto radial = shell_area_(x1f.slice(0, 0, options->nc1()),
                            x1f.slice(0, 1, options->nc1() + 1));
  return radial.unsqueeze(0).unsqueeze(1) *
         x2f.sin().abs().unsqueeze(0).unsqueeze(2) *
         dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::face_area3() const {
  auto radial = shell_area_(x1f.slice(0, 0, options->nc1()),
                            x1f.slice(0, 1, options->nc1() + 1));
  return (radial.unsqueeze(0).unsqueeze(1) * dx2f.unsqueeze(0).unsqueeze(2))
      .expand({x3f.size(0), -1, -1});
}

torch::Tensor SphericalPolarImpl::cell_volume() const {
  auto radial = shell_volume_(x1f.slice(0, 0, options->nc1()),
                              x1f.slice(0, 1, options->nc1() + 1))
                    .unsqueeze(0)
                    .unsqueeze(1);
  auto polar = polar_measure_(x2f.slice(0, 0, options->nc2()),
                              x2f.slice(0, 1, options->nc2() + 1))
                   .unsqueeze(0)
                   .unsqueeze(2);
  return radial * polar * dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor SphericalPolarImpl::face_moment2_x1() const {
  // an x2 face is r sin(theta) dr dphi and an x3 face is r dr dtheta, so both
  // carry the same radial weight w(r) = r
  return radial_face_moment2_(x1f, options->nc1());
}

torch::Tensor SphericalPolarImpl::face_centroid_shift_x1() const {
  // cell measure r^2 dr (cell_volume above), x2/x3 face measure r dr
  return radial_face_centroid_shift_(x1f, options->nc1());
}

torch::Tensor SphericalPolarImpl::forward(torch::Tensor prim,
                                          torch::Tensor flux1,
                                          torch::Tensor flux2,
                                          torch::Tensor flux3,
                                          torch::Tensor face_pressure1) {
  std::string eos_type = pmb->phydro->peos->options->type();

  enum { DIM1 = 2, DIM2 = 1 };

  auto div = CoordinateImpl::forward(prim, flux1, flux2, flux3);
  bool use_x2_fluxes = options->nx2() > 1;

  int si = il();
  int ei = iu() + 1;
  int sj = jl();
  int ej = ju() + 1;

  auto vol = cell_volume();

  auto src1 =
      coord_src1_i * prim[IDN] * (prim[IVY].square() + prim[IVZ].square());
  if (eos_type != "shallow-water") {
    // SNAP_X1_CENTROID_EXACT: with the face pressures in the flux, the pressure
    // force is -(1/V) int r^2 d_r p~ dr when the source is (2/V) int r p~ dr,
    // p~ the quintic through the six nearest faces: the r^2 average of the
    // gradient, which a hydrostatic column of r^2 means balances against
    // g rho. The plain difference below balances g <rho> instead.
    bool exact_x1 = false;
    if (face_pressure1.defined() && x1_centroid_exact_enabled()) {
      if (!x1src_ || x1src_->stale(face_pressure1)) {
        // ghost faces at a seam are the neighbour's (hydro_forward.cpp)
        auto [below, above] = pmb->phydro->x1_neighbors();
        x1src_ = std::make_shared<X1PressureSourceStencils>(
            x1_pressure_source_stencils(x1f, si, ei - 1, below < 0, above < 0,
                                        face_pressure1.options()));
      }
      exact_x1 = x1src_->usable;
    }
    if (exact_x1) {
      src1.slice(-1, si, ei) += x1_pressure_source(*x1src_, face_pressure1);
    } else if (face_pressure1.defined()) {
      auto pressure_gradient = (face_pressure1.slice(-1, si + 1, ei + 1) -
                                face_pressure1.slice(-1, si, ei)) /
                               dx1f.slice(0, si, ei);
      src1.slice(-1, si, ei) += (CoordinateImpl::face_area1(si + 1, ei + 1) *
                                     face_pressure1.slice(-1, si + 1, ei + 1) -
                                 CoordinateImpl::face_area1(si, ei) *
                                     face_pressure1.slice(-1, si, ei)) /
                                    vol.slice(-1, si, ei) -
                                pressure_gradient;
    } else {
      src1 += 2.0 * coord_src1_i * prim[IPR];
    }
  }
  div[IVX] -= src1;

  auto m_pp = prim[IDN] * prim[IVZ].square();
  if (eos_type != "shallow-water") {
    m_pp += prim[IPR];
  }
  div[IVY] -= coord_src1_i * coord_src1_j * m_pp;
  div[IVY].slice(-1, si, ei) +=
      coord_src2_i.slice(-1, si, ei) *
      (CoordinateImpl::face_area1(si, ei) * flux1[IVY].slice(DIM1, si, ei) +
       CoordinateImpl::face_area1(si + 1, ei + 1) *
           flux1[IVY].slice(DIM1, si + 1, ei + 1));

  div[IVZ].slice(-1, si, ei) +=
      coord_src2_i.slice(-1, si, ei) *
      (CoordinateImpl::face_area1(si, ei) * flux1[IVZ].slice(DIM1, si, ei) +
       CoordinateImpl::face_area1(si + 1, ei + 1) *
           flux1[IVZ].slice(DIM1, si + 1, ei + 1));
  if (use_x2_fluxes) {
    div[IVZ].slice(DIM2, sj, ej).slice(-1, si, ei) +=
        coord_src1_i.slice(-1, si, ei) * coord_src2_j.slice(DIM2, sj, ej) *
        (CoordinateImpl::face_area2(sj, ej).slice(-1, si, ei) *
             flux2[IVZ].slice(DIM2, sj, ej).slice(-1, si, ei) +
         CoordinateImpl::face_area2(sj + 1, ej + 1).slice(-1, si, ei) *
             flux2[IVZ].slice(DIM2, sj + 1, ej + 1).slice(-1, si, ei));
  } else {
    auto m_ph = prim[IDN] * prim[IVZ] * prim[IVY];
    div[IVZ] += coord_src1_i * coord_src3_j * m_ph;
  }

  return div;
}

}  // namespace snap
