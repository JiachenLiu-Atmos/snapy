// PanelRotation (V3 lane S step (1), S.rst rev 2.4 section 3.1): the one
// rotation between cubed-sphere panels goes through the global Cartesian
// frame into the local geographic frame at the point. Three properties:
//  - round trip: contravariant -> Cartesian -> contravariant is the identity
//    to round-off on every panel, with and without the metric (cosine);
//  - same components as the pin's spherical route: every matrix equals
//    cs_velocity_transform_matrix to round-off, and for a point seen from two
//    adjacent panels sender-to-receiver equals the pin's route;
//  - the matrix is a function of the point alone (H16, gate T1): the same
//    point gives the same bits wherever it sits in the strip.

// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <tuple>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_rotation.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>

using namespace snap;

namespace {

auto f64() { return torch::TensorOptions().dtype(torch::kFloat64); }

// [n] points spread over a panel, a little past its edges (ghost cells)
std::pair<torch::Tensor, torch::Tensor> panel_points(int n, double reach) {
  auto a = torch::linspace(-reach, reach, n, f64());
  auto g = torch::meshgrid({a, a}, "ij");
  return {g[1].reshape({-1}), g[0].reshape({-1})};  // alpha, beta
}

// apply a [3, 3, n] matrix to a [3, n] vector
torch::Tensor apply(torch::Tensor const& m, torch::Tensor const& v) {
  return torch::einsum("ijn,jn->in", {m, v});
}

}  // namespace

TEST(PanelRotation, round_trip_is_identity_on_every_panel) {
  PanelRotation rot;
  auto [alpha, beta] = panel_points(23, 0.25 * M_PI + 0.1);
  torch::manual_seed(1);
  // the equiangular metric: cos of the angle between the x2 and x3 lines
  auto X = alpha.tan(), Y = beta.tan();
  auto cosine = -X * Y / ((1. + X * X) * (1. + Y * Y)).sqrt();
  for (int panel = 0; panel < 6; ++panel) {
    auto v0 = torch::randn({3, alpha.size(0)}, f64());
    auto v = v0.clone();
    rot.contra_to_cart_(v, panel, alpha, beta);
    rot.cart_to_contra_(v, panel, alpha, beta);
    EXPECT_LE((v - v0).abs().max().item<double>(), 4e-15) << "panel " << panel;

    // the cached matrices, with the metric composed (conserved momentum)
    auto to = rot.matrix(panel, alpha, beta, true, cosine);
    auto from = rot.matrix(panel, alpha, beta, false, cosine);
    auto w = apply(from, apply(to, v0));
    EXPECT_LE((w - v0).abs().max().item<double>(), 4e-15) << "panel " << panel;
  }
}

TEST(PanelRotation, components_equal_the_spherical_route_across_a_seam) {
  PanelRotation rot;
  double worst = 0.;
  // adjacent pairs (+X,+Y), (+X,+Z), (+Y,-Z), (-X,-Y), (+Z,-Y), (-Z,-X)
  int const pairs[6][2] = {{0, 1}, {0, 3}, {1, 5}, {2, 4}, {3, 4}, {5, 2}};
  for (auto const& pr : pairs) {
    int pa = pr[0], pb = pr[1];
    // points just around the shared edge: the midpoint of the two face
    // centres' directions, swept along the edge and a little to each side
    auto ca = cs_ab_to_xyz(CS_FACE_NAMES[pa], 0., 0.);
    auto cb = cs_ab_to_xyz(CS_FACE_NAMES[pb], 0., 0.);
    Vec3 axis{ca.y * cb.z - ca.z * cb.y, ca.z * cb.x - ca.x * cb.z,
              ca.x * cb.y - ca.y * cb.x};  // along the edge
    int n = 0;
    std::vector<double> aa, ba, ab, bb;
    for (int s = -5; s <= 5; ++s) {
      for (int t = -3; t <= 3; ++t) {
        double u = 0.12 * s, e = 0.02 * t;
        auto p =
            unit_vec3(0.5 * (ca.x + cb.x) * (1 + e) + u * axis.x + e * ca.x,
                      0.5 * (ca.y + cb.y) * (1 + e) + u * axis.y + e * ca.y,
                      0.5 * (ca.z + cb.z) * (1 + e) + u * axis.z + e * ca.z);
        double a1, b1, a2, b2;
        cs_xyz_to_ab(CS_FACE_NAMES[pa], p, &a1, &b1);
        cs_xyz_to_ab(CS_FACE_NAMES[pb], p, &a2, &b2);
        aa.push_back(a1);
        ba.push_back(b1);
        ab.push_back(a2);
        bb.push_back(b2);
        ++n;
      }
    }
    auto alpha_a = torch::tensor(aa, f64()), beta_a = torch::tensor(ba, f64());
    auto alpha_b = torch::tensor(ab, f64()), beta_b = torch::tensor(bb, f64());
    auto cart =
        torch::matmul(rot.matrix(pb, alpha_b, beta_b, false).permute({2, 0, 1}),
                      rot.matrix(pa, alpha_a, beta_a, true).permute({2, 0, 1}));
    auto sph =
        torch::matmul(cs_velocity_transform_matrix(alpha_b, beta_b, pb, false)
                          .permute({2, 0, 1}),
                      cs_velocity_transform_matrix(alpha_a, beta_a, pa, true)
                          .permute({2, 0, 1}));
    double d = (cart - sph).abs().max().item<double>();
    // each side on its own, too: the components carried between the panels
    // are the pin's spherical ones
    for (auto [p, al, be] : {std::make_tuple(pa, alpha_a, beta_a),
                             std::make_tuple(pb, alpha_b, beta_b)}) {
      for (bool to : {true, false}) {
        d = std::max(d, (rot.matrix(p, al, be, to) -
                         cs_velocity_transform_matrix(al, be, p, to))
                            .abs()
                            .max()
                            .item<double>());
      }
    }
    worst = std::max(worst, d);
    EXPECT_LE(d, 1e-14) << "panels " << pa << " -> " << pb << ", " << n
                        << " points";
  }
  std::printf("PanelRotation vs spherical route, max |difference| = %.3e\n",
              worst);
}

TEST(PanelRotation, matrix_depends_on_the_point_alone) {
  PanelRotation rot;
  // the same points evaluated as part of strips of different lengths and
  // offsets (as the exchange does for nb = 1 and nb = 4) give the same bits
  int const n = 67;
  auto a = torch::linspace(-0.85, 0.85, n, f64());
  auto b = 0.3 * torch::sin(3. * a);
  for (int panel = 0; panel < 6; ++panel) {
    for (bool to : {true, false}) {
      auto full = rot.matrix(panel, a, b, to);
      for (int off : {1, 3, 7, 16, 33}) {
        for (int len : {1, 5, 8, 17, n - off}) {
          if (off + len > n) continue;
          auto part = rot.matrix(panel, a.narrow(0, off, len).clone(),
                                 b.narrow(0, off, len).clone(), to);
          EXPECT_TRUE(torch::equal(part, full.narrow(2, off, len)))
              << "panel " << panel << " to_global " << to << " offset " << off
              << " length " << len;
        }
      }
    }
  }
}
