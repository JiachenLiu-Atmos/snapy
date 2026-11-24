// C/C++
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

// snap
#include <snap/snap.h>

#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "coord_dispatch.hpp"
#include "coord_utils.hpp"
#include "cubed_sphere_utils.hpp"
#include "spherical_utils.hpp"

namespace snap {
namespace {

int cs_face_id(char const *face) {
  for (int n = 0; n < 6; ++n) {
    if (strcmp(face, CS_FACE_NAMES[n]) == 0) return n;
  }
  throw std::runtime_error("invalid cubed-sphere face name");
}

}  // namespace

/*!
 * beta   ^ (T,3)
 *        |------
 *  (L,0) |  X  | (R,1)
 *        --------> alpha
 *            (B,2)
 * Convert angle on a face to a fractional center index (for 1-D interp).
 * beta: varies in Y (L/R)
 * alpha: varies in X (B/T)
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
static inline double cs_angle_to_center_u(double angle, int N) {
  double d = M_PI / (2.0 * (double)N);
  // centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5
  return (angle + M_PI / 4.0) / d - 0.5;
}

Vec3 cs_ab_to_xyz(char const *face, double alpha, double beta) {
  double a = tan(alpha), b = tan(beta);
  double x, y, z;
  cs_face_xyz_from_tan(cs_face_id(face), a, b, &x, &y, &z);
  return unit_vec3(x, y, z);
}

void cs_xyz_to_ab(char const *face, Vec3 v, double *alpha, double *beta) {
  if (strcmp(face, "+X") == 0) {
    *alpha = atan2(v.y, v.x);
    *beta = atan2(v.z, v.x);
  } else if (strcmp(face, "-X") == 0) {
    *alpha = atan2(-v.y, -v.x);
    *beta = atan2(v.z, -v.x);
  } else if (strcmp(face, "+Y") == 0) {
    *alpha = atan2(-v.x, v.y);
    *beta = atan2(v.z, v.y);
  } else if (strcmp(face, "-Y") == 0) {
    *alpha = atan2(v.x, -v.y);
    *beta = atan2(v.z, -v.y);
  } else if (strcmp(face, "+Z") == 0) {
    *alpha = atan2(v.y, v.z);
    *beta = atan2(-v.x, v.z);
  } else if (strcmp(face, "-Z") == 0) {
    *alpha = atan2(v.y, -v.z);
    *beta = atan2(v.x, -v.z);
  } else {
    throw std::runtime_error("cs_xyz_to_ab: invalid face name");
  }
}

void cs_build_ghost_usrc(double *usrc, int N, int nghost, int face_t,
                         int side_t) {
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];

  // which angle varies on target face
  const int src_ax = cs_side_axis(emap.nside);
  for (int depth = 1; depth <= nghost; ++depth) {
    for (int j = 0; j < N; ++j) {
      // 1) Target ghost center
      double alpha_t, beta_t;
      cs_equ_ghost_center(side_t, N, j, depth, &alpha_t, &beta_t);

      // 2) To cartesian
      Vec3 v = cs_ab_to_xyz(CS_FACE_NAMES[face_t], alpha_t, beta_t);

      // 3) Re-express on the source face from connectivity
      double alpha_s, beta_s;
      cs_xyz_to_ab(CS_FACE_NAMES[emap.nface], v, &alpha_s, &beta_s);

      // 4) Fractional abscissa along source edge’s interior line
      double u_src =
          cs_angle_to_center_u((src_ax == AX_Y) ? beta_s : alpha_s, N);

      // 5) Flip per connectivity flag
      if (emap.rev) u_src = (double)(N - 1) - u_src;

      // 6) Write out
      usrc[(depth - 1) * N + j] = u_src;
    }
  }
}

std::pair<torch::Tensor, torch::Tensor> cs_ab_to_lonlat(char const *face,
                                                        torch::Tensor alpha,
                                                        torch::Tensor beta) {
  auto x = alpha.tan();
  auto y = beta.tan();
  auto r = (x * x + y * y + 1).sqrt();

  torch::Tensor lon, lat;

  if (strcmp(face, "+X") == 0) {
    lon = alpha.clone();
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "+Y") == 0) {
    lon = alpha + 0.5 * M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "-X") == 0) {
    lon = alpha + M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "-Y") == 0) {
    lon = alpha + 1.5 * M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "+Z") == 0) {
    lon = torch::atan2(x, -y);
    lat = torch::asin(1. / r);
  } else if (strcmp(face, "-Z") == 0) {
    lon = torch::atan2(x, y);
    lat = -torch::asin(1. / r);
  }

  // Map to the interval [0, 2 pi]
  lon += torch::where(lon < 0.0, 2.0 * M_PI, torch::zeros_like(lon));

  return {lon, lat};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cs_ab_to_xyz_tensors(char const *face, torch::Tensor const &alpha,
                     torch::Tensor const &beta) {
  auto a = alpha.tan();
  auto b = beta.tan();
  auto one = torch::ones_like(a);

  torch::Tensor x, y, z;
  if (strcmp(face, "+X") == 0) {
    x = one;
    y = a;
    z = b;
  } else if (strcmp(face, "+Y") == 0) {
    x = -a;
    y = one;
    z = b;
  } else if (strcmp(face, "-X") == 0) {
    x = -one;
    y = -a;
    z = b;
  } else if (strcmp(face, "+Z") == 0) {
    x = -b;
    y = a;
    z = one;
  } else if (strcmp(face, "-Y") == 0) {
    x = a;
    y = -one;
    z = b;
  } else if (strcmp(face, "-Z") == 0) {
    x = b;
    y = a;
    z = -one;
  } else {
    TORCH_CHECK(false, "invalid cubed-sphere face name: ", face);
  }

  auto r = (x * x + y * y + z * z).sqrt();
  return {x / r, y / r, z / r};
}

static std::pair<torch::Tensor, torch::Tensor> cs_ab_to_theta_phi(
    torch::Tensor const &alpha, torch::Tensor const &beta, int face_id) {
  auto [x, y, z] = cs_ab_to_xyz_tensors(CS_FACE_NAMES[face_id], alpha, beta);
  auto theta = z.clamp(-1.0, 1.0).acos();
  auto phi = torch::atan2(y, x);
  return {theta, phi};
}

//! gnomonic metric factors at panel tangents (x, y): delta = sqrt(x^2 + y^2 +
//! 1), C = sqrt(1 + x^2), D = sqrt(1 + y^2)
static std::array<torch::Tensor, 3> cs_metric(torch::Tensor const &x,
                                              torch::Tensor const &y) {
  return {sqrt(x * x + y * y + 1), sqrt(1 + x * x), sqrt(1 + y * y)};
}

CsMetric cs_metric_at(torch::Tensor const &alpha, torch::Tensor const &beta) {
  auto x = alpha.tan();
  auto y = beta.tan();
  auto [delta, C, D] = cs_metric(x, y);
  return {x, y, delta, C, D};
}

//! Transform global cartesian velocity to local panel contravariant velocity
void cs_cart_to_contra_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta, int face_id) {
  cs_cart_to_contra_at_(vel, cs_metric_at(alpha, beta), face_id);
}

void cs_cart_to_contra_at_(torch::Tensor const &vel, CsMetric const &m,
                           int face_id) {
  auto x = m.x, y = m.y, delta = m.delta, C = m.C, D = m.D;

  std::array<torch::Tensor, 3> local_vel;

  auto const g2l = CS_CART_TO_LOCAL_VEL;
  auto f = face_id;

  local_vel[g2l[f][VEL1].idx] = g2l[f][VEL1].sgn * vel[VEL1];
  local_vel[g2l[f][VEL2].idx] = g2l[f][VEL2].sgn * vel[VEL2];
  local_vel[g2l[f][VEL3].idx] = g2l[f][VEL3].sgn * vel[VEL3];

  auto vz = local_vel[VEL1];
  auto vx = local_vel[VEL2];
  auto vy = local_vel[VEL3];

  vel[VEL1] = (vz + x * vx + y * vy) / delta;
  vel[VEL2] = D / delta * (vx - x * vz);
  vel[VEL3] = C / delta * (vy - y * vz);
}

void cs_contra_to_cart_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta, int face_id) {
  cs_contra_to_cart_at_(vel, cs_metric_at(alpha, beta), face_id);
}

void cs_contra_to_cart_at_(torch::Tensor const &vel, CsMetric const &m,
                           int face_id) {
  auto x = m.x, y = m.y, delta = m.delta, C = m.C, D = m.D;

  auto const l2g = CS_LOCAL_TO_CART_VEL;
  auto f = face_id;

  auto vz = vel[VEL1].clone();
  auto vx = vel[VEL2].clone();
  auto vy = vel[VEL3].clone();

  vel[l2g[f][VEL1].idx] =
      l2g[f][VEL1].sgn * (vz - vx * x / D - vy * y / C) / delta;
  vel[l2g[f][VEL2].idx] =
      l2g[f][VEL2].sgn * (vz * x + vx * D - (vy * x * y) / C) / delta;
  vel[l2g[f][VEL3].idx] =
      l2g[f][VEL3].sgn * (vz * y + vy * C - (vx * x * y) / D) / delta;
}

void cs_contra_to_sph_(torch::Tensor const &vel, torch::Tensor alpha,
                       torch::Tensor beta, int face_id) {
  auto [theta, phi] = cs_ab_to_theta_phi(alpha, beta, face_id);
  cs_contra_to_cart_(vel, alpha, beta, face_id);
  sph_cart_to_contra_(vel, theta, phi);
}

void cs_sph_to_contra_(torch::Tensor const &vel, torch::Tensor alpha,
                       torch::Tensor beta, int face_id) {
  auto [theta, phi] = cs_ab_to_theta_phi(alpha, beta, face_id);
  sph_contra_to_cart_(vel, theta, phi);
  cs_cart_to_contra_(vel, alpha, beta, face_id);
}

torch::Tensor cs_velocity_transform_matrix(torch::Tensor alpha,
                                           torch::Tensor beta, int face_id,
                                           bool to_spherical,
                                           torch::Tensor cosine) {
  TORCH_CHECK(alpha.sizes() == beta.sizes(),
              "cubed-sphere transform coordinates must have matching shapes");
  TORCH_CHECK(alpha.scalar_type() == beta.scalar_type() &&
                  alpha.device() == beta.device(),
              "cubed-sphere transform coordinates must have matching dtype "
              "and device");
  if (cosine.defined()) {
    TORCH_CHECK(cosine.sizes() == alpha.sizes(),
                "cubed-sphere transform cosine has incompatible shape");
    TORCH_CHECK(cosine.scalar_type() == alpha.scalar_type() &&
                    cosine.device() == alpha.device(),
                "cubed-sphere transform cosine must match coordinate dtype "
                "and device");
  }

  auto spatial = alpha.sizes().vec();
  auto vector_shape = spatial;
  vector_shape.insert(vector_shape.begin(), 3);
  auto matrix_shape = vector_shape;
  matrix_shape.insert(matrix_shape.begin() + 1, 3);
  auto matrix = torch::empty(matrix_shape, alpha.options());

  for (int input = 0; input < 3; ++input) {
    auto basis = torch::zeros(vector_shape, alpha.options());
    basis[input].fill_(1.);
    if (to_spherical) {
      if (cosine.defined()) coord_vec_raise_(basis, cosine);
      cs_contra_to_sph_(basis, alpha, beta, face_id);
    } else {
      cs_sph_to_contra_(basis, alpha, beta, face_id);
      if (cosine.defined()) coord_vec_lower_(basis, cosine);
    }
    matrix.select(1, input).copy_(basis);
  }
  return matrix;
}

void cs_apply_velocity_transform_(torch::Tensor const &vel,
                                  torch::Tensor const &matrix) {
  TORCH_CHECK(vel.dim() >= 2 && vel.size(0) == 3,
              "cubed-sphere velocity must have shape [3,...], got ",
              vel.sizes());
  TORCH_CHECK(matrix.dim() == vel.dim() + 1 && matrix.size(0) == 3 &&
                  matrix.size(1) == 3,
              "cubed-sphere matrix must have shape [3,3,...], got ",
              matrix.sizes());
  TORCH_CHECK(matrix.scalar_type() == vel.scalar_type() &&
                  matrix.device() == vel.device(),
              "cubed-sphere matrix must match velocity dtype and device");
  for (int dim = 1; dim < vel.dim(); ++dim) {
    TORCH_CHECK(
        matrix.size(dim + 1) == 1 || matrix.size(dim + 1) == vel.size(dim),
        "cubed-sphere matrix spatial shape ", matrix.sizes(),
        " cannot broadcast to velocity shape ", vel.sizes());
  }

  auto spatial = vel[0];
  at::TensorIteratorConfig config;
  config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(spatial.sizes())
      .add_owned_output(vel[0])
      .add_owned_output(vel[1])
      .add_owned_output(vel[2]);
  for (int output = 0; output < 3; ++output) {
    for (int input = 0; input < 3; ++input) {
      config.add_owned_input(matrix[output][input].expand_as(spatial));
    }
  }
  auto iter = config.build();
  at::native::call_cs_matrix_vec(vel.device().type(), iter);
}

}  // namespace snap
