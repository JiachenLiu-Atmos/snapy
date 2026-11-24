#include "coord_rotation.hpp"

// C/C++
#include <vector>

// snap
#include "coord_utils.hpp"
#include "cubed_sphere_utils.hpp"

namespace snap {

namespace {

// Cartesian components are stored (z, x, y) in VEL1..VEL3, as
// cs_contra_to_cart_ writes them. The local frame at a point with unit
// position (x, y, z), s = sqrt(x^2 + y^2): r = (x, y, z),
// theta = (x z / s, y z / s, -s), phi = (-y / s, x / s, 0); at a pole
// (s = 0) phi is taken as 0, as atan2(+0, +0) gives.
struct LocalFrame {
  torch::Tensor x, y, z, cphi, sphi, s;
};

LocalFrame local_frame(int panel, torch::Tensor alpha, torch::Tensor beta) {
  // b_1, the first covariant basis vector, is the unit radial vector
  auto shape = alpha.sizes().vec();
  shape.insert(shape.begin(), 3);
  auto up = torch::zeros(shape, alpha.options());
  up[0].fill_(1.);
  cs_contra_to_cart_(up, alpha, beta, panel);
  LocalFrame f;
  f.z = up[0];
  f.x = up[1];
  f.y = up[2];
  f.s = (f.x * f.x + f.y * f.y).sqrt();
  auto pole = f.s == 0.;
  auto safe = torch::where(pole, torch::ones_like(f.s), f.s);
  f.cphi = torch::where(pole, torch::ones_like(f.s), f.x / safe);
  f.sphi = torch::where(pole, torch::zeros_like(f.s), f.y / safe);
  return f;
}

void cart_to_local_(torch::Tensor vel, LocalFrame const& f) {
  auto vz = vel[0].clone(), vx = vel[1].clone(), vy = vel[2].clone();
  auto horiz = vx * f.cphi + vy * f.sphi;  // along the meridian plane
  vel[0] = horiz * f.s + vz * f.z;         // v_r
  vel[1] = horiz * f.z - vz * f.s;         // v_theta
  vel[2] = -vx * f.sphi + vy * f.cphi;     // v_phi
}

void local_to_cart_(torch::Tensor vel, LocalFrame const& f) {
  auto vr = vel[0].clone(), vt = vel[1].clone(), vp = vel[2].clone();
  auto horiz = vr * f.s + vt * f.z;       // in the meridian plane, away from z
  vel[1] = horiz * f.cphi - vp * f.sphi;  // v_x
  vel[2] = horiz * f.sphi + vp * f.cphi;  // v_y
  vel[0] = vr * f.z - vt * f.s;           // v_z
}

}  // namespace

void PanelRotation::contra_to_cart_(torch::Tensor vel, int panel,
                                    torch::Tensor alpha,
                                    torch::Tensor beta) const {
  cs_contra_to_cart_(vel, alpha, beta, panel);
}

void PanelRotation::cart_to_contra_(torch::Tensor vel, int panel,
                                    torch::Tensor alpha,
                                    torch::Tensor beta) const {
  cs_cart_to_contra_(vel, alpha, beta, panel);
}

PanelRotation::FaceOperands PanelRotation::face_operands(
    std::vector<int64_t> const& shape, torch::Tensor alpha, torch::Tensor beta,
    torch::Tensor cosine) {
  return {cs_metric_at(alpha.expand(shape), beta.expand(shape)),
          cosine.expand(shape).contiguous()};
}

void PanelRotation::face_rows_to_cart_(torch::Tensor rows, int panel,
                                       torch::Tensor alpha, torch::Tensor beta,
                                       torch::Tensor cosine) const {
  auto shape = rows[0].sizes().vec();  // own the sizes: rows[0] is a temporary
  face_rows_to_cart_(rows, panel, face_operands(shape, alpha, beta, cosine));
}

void PanelRotation::cart_to_face_rows_(torch::Tensor rows, int panel,
                                       torch::Tensor alpha, torch::Tensor beta,
                                       torch::Tensor cosine) const {
  auto shape = rows[0].sizes().vec();  // own the sizes: rows[0] is a temporary
  cart_to_face_rows_(rows, panel, face_operands(shape, alpha, beta, cosine));
}

void PanelRotation::face_rows_to_cart_(torch::Tensor rows, int panel,
                                       FaceOperands const& op) const {
  coord_vec_raise_(rows, op.cosine);
  cs_contra_to_cart_at_(rows, op.metric, panel);
}

void PanelRotation::cart_to_face_rows_(torch::Tensor rows, int panel,
                                       FaceOperands const& op) const {
  cs_cart_to_contra_at_(rows, op.metric, panel);
  coord_vec_lower_(rows, op.cosine);
}

torch::Tensor PanelRotation::matrix(int panel, torch::Tensor alpha,
                                    torch::Tensor beta, bool to_global,
                                    torch::Tensor cosine) const {
  TORCH_CHECK(alpha.sizes() == beta.sizes(),
              "PanelRotation: coordinates must have matching shapes");
  TORCH_CHECK(alpha.scalar_type() == beta.scalar_type() &&
                  alpha.device() == beta.device(),
              "PanelRotation: coordinates must match in dtype and device");
  if (cosine.defined()) {
    TORCH_CHECK(cosine.sizes() == alpha.sizes(),
                "PanelRotation: cosine has an incompatible shape");
    TORCH_CHECK(cosine.scalar_type() == alpha.scalar_type() &&
                    cosine.device() == alpha.device(),
                "PanelRotation: cosine must match the coordinates' dtype and "
                "device");
  }

  auto spatial = alpha.sizes().vec();
  auto vector_shape = spatial;
  vector_shape.insert(vector_shape.begin(), 3);
  auto matrix_shape = vector_shape;
  matrix_shape.insert(matrix_shape.begin() + 1, 3);
  auto m = torch::empty(matrix_shape, alpha.options());

  // the local geographic frame at the point, from its Cartesian position
  // alone (no atan2): up = r, south = theta, east = phi
  auto frame = local_frame(panel, alpha, beta);

  for (int input = 0; input < 3; ++input) {
    auto basis = torch::zeros(vector_shape, alpha.options());
    basis[input].fill_(1.);
    if (to_global) {
      if (cosine.defined()) coord_vec_raise_(basis, cosine);
      contra_to_cart_(basis, panel, alpha, beta);
      cart_to_local_(basis, frame);
    } else {
      local_to_cart_(basis, frame);
      cart_to_contra_(basis, panel, alpha, beta);
      if (cosine.defined()) coord_vec_lower_(basis, cosine);
    }
    m.select(1, input).copy_(basis);
  }
  return m;
}

}  // namespace snap
