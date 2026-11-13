// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/coord/spherical_polar.hpp>
#include <snap/coord/spherical_utils.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_type.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

const char* gnomonic_radial_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4

geometry:
  type: gnomonic-equiangle
  cells: {nx1: 6, nx2: 6, nx3: 6, nghost: 2}
  bounds:
    x1min: 1.
    x1max: 2.
    x2min_pi: -0.25
    x2max_pi: 0.25
    x3min_pi: -0.25
    x3max_pi: 0.25

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: custom
    x2-outer: custom
    x3-inner: custom
    x3-outer: custom
)";

const char* spherical_polar_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4

distribute:
  layout: slab
  nb2: 1
  nb3: 1
  verbose: false

geometry:
  type: spherical-polar
  cells: {nx1: 4, nx2: 4, nx3: 4, nghost: 1}
  bounds:
    x1min: 1.
    x1max: 2.
    x2min_pi: 0.25
    x2max_pi: 0.75
    x3min_pi: -0.5
    x3max_pi: 0.5

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)";

std::string write_temp_config(char const* config) {
  char fname[] = "/tmp/test-coordinate-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);

  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();
  return fname;
}

}  // namespace

TEST(GnomonicEquiangle, area_vol) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);

  auto pcoord = block->pcoord;
  for (auto const& spacing : {pcoord->dx1f, pcoord->dx1v, pcoord->dx2f,
                              pcoord->dx2v, pcoord->dx3f, pcoord->dx3v}) {
    EXPECT_EQ(spacing.scalar_type(), torch::kFloat64);
  }

  auto area1 = pcoord->face_area1();
  std::cout << "area1 = \n" << area1 << std::endl;

  auto area2 = pcoord->face_area2();
  std::cout << "area2 = \n" << area2 << std::endl;

  auto area3 = pcoord->face_area3();
  std::cout << "area3 = \n" << area3 << std::endl;

  auto vol = pcoord->cell_volume();
  std::cout << "volume = \n" << vol << std::endl;
}

TEST(SphericalPolar, geometry_matches_athena_reference_formulas) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  std::remove(fname.c_str());

  auto pcoord = std::dynamic_pointer_cast<SphericalPolarImpl>(block->pcoord);
  ASSERT_TRUE(pcoord != nullptr);

  auto area1 = pcoord->face_area1();
  auto area2 = pcoord->face_area2();
  auto area3 = pcoord->face_area3();
  auto vol = pcoord->cell_volume();

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();

  auto x1m = pcoord->x1f.slice(0, 0, nc1);
  auto x1p = pcoord->x1f.slice(0, 1, nc1 + 1);
  auto x2m = pcoord->x2f.slice(0, 0, nc2);
  auto x2p = pcoord->x2f.slice(0, 1, nc2 + 1);

  auto expected_x1v =
      0.75 * (x1p.pow(4) - x1m.pow(4)) / (x1p.pow(3) - x1m.pow(3));
  auto expected_x2v =
      ((x2p.sin() - x2p * x2p.cos()) - (x2m.sin() - x2m * x2m.cos())) /
      (x2m.cos() - x2p.cos());
  EXPECT_TRUE(torch::allclose(pcoord->x1v, expected_x1v, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(pcoord->x2v, expected_x2v, 1.e-12, 1.e-12));

  auto radial_area23 = 0.5 * (x1p.square() - x1m.square());
  auto radial_volume = (x1p.pow(3) - x1m.pow(3)) / 3.0;
  auto polar_area1 = torch::abs(x2m.cos() - x2p.cos());
  auto sin_face = torch::abs(pcoord->x2f.sin());

  auto expected_area1 = pcoord->x1f.square().unsqueeze(0).unsqueeze(1) *
                        polar_area1.unsqueeze(0).unsqueeze(2) *
                        pcoord->dx3f.unsqueeze(1).unsqueeze(2);
  auto expected_area2 = radial_area23.unsqueeze(0).unsqueeze(1) *
                        sin_face.unsqueeze(0).unsqueeze(2) *
                        pcoord->dx3f.unsqueeze(1).unsqueeze(2);
  auto expected_area3 = (radial_area23.unsqueeze(0).unsqueeze(1) *
                         pcoord->dx2f.unsqueeze(0).unsqueeze(2))
                            .expand({pcoord->x3f.size(0), -1, -1});
  auto expected_vol = radial_volume.unsqueeze(0).unsqueeze(1) *
                      polar_area1.unsqueeze(0).unsqueeze(2) *
                      pcoord->dx3f.unsqueeze(1).unsqueeze(2);

  EXPECT_TRUE(torch::allclose(area1, expected_area1, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(area2, expected_area2, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(area3, expected_area3, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(vol, expected_vol, 1.e-12, 1.e-12));

  auto sin_m = torch::abs(x2m.sin());
  auto sin_p = torch::abs(x2p.sin());
  auto expected_src1_i =
      (radial_area23 / radial_volume).unsqueeze(0).unsqueeze(0);
  auto expected_src2_i =
      (pcoord->dx1f / ((x1m + x1p) * radial_volume)).unsqueeze(0).unsqueeze(0);
  auto expected_src1_j =
      ((sin_p - sin_m) / polar_area1).unsqueeze(0).unsqueeze(-1);
  auto expected_src2_j = ((sin_p - sin_m) / ((sin_m + sin_p) * polar_area1))
                             .unsqueeze(0)
                             .unsqueeze(-1);

  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src1_i, expected_src1_i, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src2_i, expected_src2_i, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src1_j, expected_src1_j, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src2_j, expected_src2_j, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src3_j, expected_src1_j, 1.e-12, 1.e-12));
}

// The flux covariance skips its centroid parts when face_centroid_shift_zero()
// is true, a host constant of the coordinate type that is never checked
// against the device. It must agree with face_centroid_shift_x1() for every
// coordinate type that builds the shift (cylindrical builds none and is not
// zero-shift), on a curved-radius grid and after a dtype change, so a new
// type that overrides the shift without the flag fails here.
TEST(Coordinate, face_centroid_shift_zero_matches_the_shift) {
  struct {
    std::string type;
    char const* config;
    bool zero;
  } cases[] = {{"cartesian", spherical_polar_config, true},
               {"cylindrical", spherical_polar_config, false},
               {"spherical-polar", spherical_polar_config, false},
               {"gnomonic-equiangle", gnomonic_radial_config, false}};
  for (auto const& c : cases) {
    std::string config = c.config;
    std::string from = "type: spherical-polar";
    auto at = config.find(from);
    if (at != std::string::npos)
      config.replace(at, from.size(), "type: " + c.type);
    auto fname = write_temp_config(config.c_str());
    auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(fname));
    std::remove(fname.c_str());
    auto pcoord = block->pcoord;
    ASSERT_EQ(pcoord->options->type(), c.type);

    for (auto dtype : {torch::kFloat64, torch::kFloat32}) {
      block->to(dtype);
      EXPECT_EQ(pcoord->face_centroid_shift_zero(), c.zero) << c.type;
      // cylindrical builds no dx1f, so it has no shift to compare with
      if (c.type == "cylindrical") continue;
      auto shift = pcoord->face_centroid_shift_x1();
      ASSERT_EQ(shift.scalar_type(), dtype) << c.type;
      EXPECT_EQ(shift.eq(0).all().item<bool>(), c.zero) << c.type;
    }
  }
}

TEST(Coordinate, refuses_a_block_narrower_than_the_ghost_zone) {
  std::string config = spherical_polar_config;
  auto replace = [&](std::string const& from, std::string const& to) {
    auto at = config.find(from);
    ASSERT_NE(at, std::string::npos) << from;
    config.replace(at, from.size(), to);
  };
  replace("type: spherical-polar", "type: cartesian");
  replace("{nx1: 4, nx2: 4, nx3: 4, nghost: 1}",
          "{nx1: 4, nx2: 8, nx3: 4, nghost: 3}");

  // one block holds all 8 x2 cells
  auto fname = write_temp_config(config.c_str());
  EXPECT_NO_THROW(CoordinateOptionsImpl::from_yaml(fname));
  std::remove(fname.c_str());

  // four blocks hold 2 x2 cells each, fewer than the 3 ghost cells
  replace("nb2: 1", "nb2: 4");
  fname = write_temp_config(config.c_str());
  EXPECT_THROW(CoordinateOptionsImpl::from_yaml(fname), c10::Error);
  std::remove(fname.c_str());

  // eight blocks hold 1 x2 cell each: a split axis, not a collapsed one
  replace("nb2: 4", "nb2: 8");
  fname = write_temp_config(config.c_str());
  EXPECT_THROW(CoordinateOptionsImpl::from_yaml(fname), c10::Error);
  std::remove(fname.c_str());
}

TEST(GnomonicEquiangle, l2g) {
  CSVel l2g[6][3];
  populate_cs_l2g_vel(l2g);

  for (int f = 0; f < 6; ++f)
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(l2g[f][i].idx, CS_LOCAL_TO_CART_VEL[f][i].idx);
      EXPECT_EQ(l2g[f][i].sgn, CS_LOCAL_TO_CART_VEL[f][i].sgn);
    }
}

TEST_P(DeviceTest, vec_lower_raise) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel = torch::ones({3, nc3, nc2, nc1},
                         torch::TensorOptions().dtype(dtype).device(device));

  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  coord_vec_raise_(vel, pcoord->cosine_cell_kj);

  EXPECT_TRUE(torch::allclose(vel, torch::ones_like(vel)));
}

TEST_P(DeviceTest, contra_cart) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel_cart = torch::ones(
      {3, nc3, nc2, nc1}, torch::TensorOptions().dtype(dtype).device(device));

  auto vel = vel_cart.clone();
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");

  cs_cart_to_contra_(vel, mesh[0], mesh[1], 0);
  std::cout << "vel contravariant = \n" << vel << std::endl;
  cs_contra_to_cart_(vel, mesh[0], mesh[1], 0);

  EXPECT_TRUE(torch::allclose(vel, vel_cart));
}

TEST_P(DeviceTest, contra_sph) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_contra = torch::randn({3, nc3, nc2, nc1}, opts) +
                    torch::full({3, nc3, nc2, nc1}, 0.5, opts);

  for (int face = 0; face < 6; ++face) {
    auto vel = vel_contra.clone();
    cs_contra_to_sph_(vel, alpha, beta, face);
    cs_sph_to_contra_(vel, alpha, beta, face);
    EXPECT_TRUE(torch::allclose(vel, vel_contra, 1.e-4, 1.e-5))
        << "face " << face;
  }
}

TEST_P(DeviceTest, cached_cubed_sphere_velocity_matrices_match_direct) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];
  auto cosine = pcoord->cosine_cell_kj.expand({nc3, nc2, nc1});

  for (int face = 0; face < 6; ++face) {
    for (bool conserved : {false, true}) {
      auto contra = torch::randn({3, nc3, nc2, nc1}, opts);
      auto expected_sph = contra.clone();
      if (conserved) coord_vec_raise_(expected_sph, cosine);
      cs_contra_to_sph_(expected_sph, alpha, beta, face);

      auto alpha_plane = alpha.narrow(-1, 0, 1);
      auto beta_plane = beta.narrow(-1, 0, 1);
      auto metric_plane = conserved ? cosine.narrow(-1, 0, 1) : torch::Tensor();
      auto to_sph = cs_velocity_transform_matrix(alpha_plane, beta_plane, face,
                                                 true, metric_plane);
      EXPECT_EQ(to_sph.size(-1), 1);
      auto actual_sph = contra.clone();
      cs_apply_velocity_transform_(actual_sph, to_sph);
      EXPECT_TRUE(torch::allclose(actual_sph, expected_sph, 1.e-4, 1.e-5))
          << "to spherical face=" << face << " conserved=" << conserved;

      auto sph = torch::randn({3, nc3, nc2, nc1}, opts);
      auto expected_contra = sph.clone();
      cs_sph_to_contra_(expected_contra, alpha, beta, face);
      if (conserved) coord_vec_lower_(expected_contra, cosine);

      auto from_sph = cs_velocity_transform_matrix(alpha_plane, beta_plane,
                                                   face, false, metric_plane);
      auto actual_contra = sph.clone();
      cs_apply_velocity_transform_(actual_contra, from_sph);
      EXPECT_TRUE(torch::allclose(actual_contra, expected_contra, 1.e-4, 1.e-5))
          << "from spherical face=" << face << " conserved=" << conserved;
    }
  }
}

TEST_P(DeviceTest, contra_sph_matches_cartesian_composition) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_contra = torch::randn({3, nc3, nc2, nc1}, opts) +
                    torch::full({3, nc3, nc2, nc1}, 0.5, opts);

  for (int face = 0; face < 6; ++face) {
    auto expected = vel_contra.clone();
    cs_contra_to_cart_(expected, alpha, beta, face);
    auto lonlat = cs_ab_to_lonlat(CS_FACE_NAMES[face], alpha, beta);
    auto theta = 0.5 * M_PI - lonlat.second;
    sph_cart_to_contra_(expected, theta, lonlat.first);

    auto actual = vel_contra.clone();
    cs_contra_to_sph_(actual, alpha, beta, face);

    EXPECT_TRUE(torch::allclose(actual, expected, 1.e-4, 1.e-5))
        << "face " << face;
  }
}

TEST_P(DeviceTest, uniform_radial_spherical_velocity_round_trips) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_sph = torch::zeros({3, nc3, nc2, nc1}, opts);
  vel_sph[VEL1].fill_(1.2345);

  for (int face = 0; face < 6; ++face) {
    auto vel = vel_sph.clone();
    cs_sph_to_contra_(vel, alpha, beta, face);
    cs_contra_to_sph_(vel, alpha, beta, face);
    EXPECT_TRUE(torch::allclose(vel, vel_sph, 1.e-4, 1.e-5)) << "face " << face;
  }
}

TEST_P(DeviceTest, usrc) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = std::dynamic_pointer_cast<GnomonicEquiangleImpl>(block->pcoord);
  std::cout << "usrc_LR = \n" << pcoord->usrc_LR << std::endl;
  std::cout << "usrc_BT = \n" << pcoord->usrc_BT << std::endl;
}

TEST_P(DeviceTest, interpolate_LR) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto var = torch::zeros({nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));

  // interior
  auto sub = block->part({0, 0, 0}, PartOptions().exterior(true).ndim(3));
  var.index(sub).fill_(1.0);

  // left
  sub = block->part({-1, 0, 0}, PartOptions().exterior(true).ndim(3));
  auto buf = torch::ones_like(var.index(sub)) * 2.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = j;

  std::cout << "var before = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  var.index_put_(sub, buf);
  pcoord->interp_ghost(var, {-1, 0, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // right
  sub = block->part({1, 0, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 3.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = j;

  var.index_put_(sub, buf);
  pcoord->interp_ghost(var, {1, 0, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // bottom
  sub = block->part({0, -1, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 4.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = k;

  var.index_put_(sub, buf);
  pcoord->interp_ghost(var, {0, -1, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // top
  sub = block->part({0, 1, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 5.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = k;

  var.index_put_(sub, buf);
  pcoord->interp_ghost(var, {0, 1, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;
}

TEST_P(DeviceTest, flux_projection1) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local1_(prim);
  pcoord->flux2global1_(prim);

  coord_vec_raise_(prim.narrow(0, IVX, 3), pcoord->cosine_cell_kj);
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, flux_projection2) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local2_(prim);
  pcoord->flux2global2_(prim);

  auto xf = pcoord->x2f.tan().unsqueeze(0).unsqueeze(-1);
  auto y = pcoord->x3v.tan().unsqueeze(-1).unsqueeze(-1);
  auto Cf = torch::sqrt(1. + xf * xf);
  auto D = torch::sqrt(1. + y * y);
  auto cthf = -xf * y / Cf / D;

  coord_vec_raise_(prim.narrow(0, IVX, 3), cthf.narrow(1, 0, nc2));
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, flux_projection3) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local3_(prim);
  pcoord->flux2global3_(prim);

  auto x = pcoord->x2v.tan().unsqueeze(0).unsqueeze(-1);
  auto yf = pcoord->x3f.tan().unsqueeze(-1).unsqueeze(-1);
  auto C = torch::sqrt(1. + x * x);
  auto Df = torch::sqrt(1. + yf * yf);
  auto cthf = -x * yf / C / Df;

  coord_vec_raise_(prim.narrow(0, IVX, 3), cthf.narrow(0, 0, nc3));
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, radial_source_uses_face_pressure_in_x1_momentum) {
  auto fname = write_temp_config(gnomonic_radial_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim_lo = torch::zeros({5, nc3, nc2, nc1}, opts);
  auto prim_hi = torch::zeros_like(prim_lo);
  prim_lo[IDN].fill_(1.0);
  prim_hi[IDN].fill_(1.0);
  prim_lo[IPR].fill_(3.0);
  prim_hi[IPR].fill_(11.0);

  auto flux1 = torch::zeros_like(prim_lo);
  auto face_pressure = torch::full({nc3, nc2, nc1}, 5.0, opts);

  auto div_lo = pcoord->forward(prim_lo, flux1, torch::Tensor(),
                                torch::Tensor(), face_pressure);
  auto div_hi = pcoord->forward(prim_hi, flux1, torch::Tensor(),
                                torch::Tensor(), face_pressure);

  EXPECT_TRUE(torch::allclose(div_lo[IVX], div_hi[IVX], 1.e-8, 1.e-8));
}

TEST_P(DeviceTest, radial_source_preserves_face_pressure_gradient) {
  auto fname = write_temp_config(gnomonic_radial_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int si = pcoord->il();
  int ei = pcoord->iu() + 1;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim = torch::zeros({5, nc3, nc2, nc1}, opts);
  prim[IDN].fill_(1.0);
  auto face_pressure = pcoord->x1f.slice(0, 0, nc1)
                           .to(opts)
                           .view({1, 1, nc1})
                           .expand({nc3, nc2, nc1});
  auto flux1 = torch::zeros_like(prim);
  flux1[IVX].copy_(face_pressure);

  auto div = pcoord->forward(prim, flux1, torch::Tensor(), torch::Tensor(),
                             face_pressure);
  auto radial_div = div[IVX].slice(-1, si, ei);
  auto expected = (face_pressure.slice(-1, si + 1, ei + 1) -
                   face_pressure.slice(-1, si, ei)) /
                  pcoord->dx1f.slice(0, si, ei);

  EXPECT_TRUE(torch::allclose(radial_div, expected, 1.e-6, 1.e-6))
      << "radial_div=" << radial_div << "\nexpected=" << expected;
}

TEST_P(DeviceTest,
       spherical_polar_radial_source_preserves_face_pressure_gradient) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int si = pcoord->il();
  int ei = pcoord->iu() + 1;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim = torch::zeros({5, nc3, nc2, nc1}, opts);
  prim[IDN].fill_(1.0);
  auto face_pressure = pcoord->x1f.slice(0, 0, nc1)
                           .to(opts)
                           .view({1, 1, nc1})
                           .expand({nc3, nc2, nc1});
  auto flux1 = torch::zeros_like(prim);
  flux1[IVX].copy_(face_pressure);
  auto flux2 = torch::zeros_like(prim);
  auto flux3 = torch::zeros_like(prim);

  auto div = pcoord->forward(prim, flux1, flux2, flux3, face_pressure);
  auto radial_div = div[IVX].slice(-1, si, ei);
  auto expected = (face_pressure.slice(-1, si + 1, ei + 1) -
                   face_pressure.slice(-1, si, ei)) /
                  pcoord->dx1f.slice(0, si, ei);

  EXPECT_TRUE(torch::allclose(radial_div, expected, 1.e-6, 1.e-6))
      << "radial_div=" << radial_div << "\nexpected=" << expected;
}

TEST_P(DeviceTest,
       spherical_polar_radial_source_uses_face_pressure_in_x1_momentum) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim_lo = torch::zeros({5, nc3, nc2, nc1}, opts);
  auto prim_hi = torch::zeros_like(prim_lo);
  prim_lo[IDN].fill_(1.0);
  prim_hi[IDN].fill_(1.0);
  prim_lo[IPR].fill_(3.0);
  prim_hi[IPR].fill_(11.0);

  auto flux1 = torch::zeros_like(prim_lo);
  auto flux2 = torch::zeros_like(prim_lo);
  auto flux3 = torch::zeros_like(prim_lo);
  auto face_pressure = torch::full({nc3, nc2, nc1}, 5.0, opts);

  auto div_no_face_lo = pcoord->forward(prim_lo, flux1, flux2, flux3);
  auto div_no_face_hi = pcoord->forward(prim_hi, flux1, flux2, flux3);
  auto div_face_lo =
      pcoord->forward(prim_lo, flux1, flux2, flux3, face_pressure);
  auto div_face_hi =
      pcoord->forward(prim_hi, flux1, flux2, flux3, face_pressure);

  EXPECT_FALSE(
      torch::allclose(div_no_face_lo[IVX], div_no_face_hi[IVX], 1.e-6, 1.e-6));
  EXPECT_TRUE(
      torch::allclose(div_face_lo[IVX], div_face_hi[IVX], 1.e-6, 1.e-6));
}

//! ---- a coordinate built from block-local options alone -------------------
//! A block constructed programmatically carries no global grid. The yaml path
//! never reaches this; nothing in tests/ covered it.
//! Assertions are per-face VALUES and per-element SPACINGS, not shapes: an
//! implementation that repairs the face count while leaving dxN() dividing by
//! an unresolved global count is wrong by (nxN)x on every width, area and
//! volume, and a shape-only gate passes it.
//! Eleven of the twelve tests below gate the sentinel rather than the
//! global-grid design; only (J) fails a revert to block-local face
//! arithmetic.

namespace {
// distinct span AND distinct count per axis, so an axis mix-up cannot pass
CoordinateOptions a3_opts() {
  auto op = CoordinateOptionsImpl::create();
  op->nx1(3).x1min(-2.).x1max(4.);
  op->nx2(4).x2min(10.).x2max(30.);
  op->nx3(6).x3min(-1.).x3max(2.);
  op->nghost(3);
  return op;
}
void expect_axis(torch::Tensor xf, double lo, double hi, int nx, int ng) {
  double dx = (hi - lo) / nx;
  ASSERT_EQ(xf.size(0), nx + 2 * ng + 1);
  EXPECT_DOUBLE_EQ(xf[ng].item<double>(), lo);
  EXPECT_DOUBLE_EQ(xf[ng + nx].item<double>(), hi);
  EXPECT_DOUBLE_EQ(xf[0].item<double>(), lo - ng * dx);
  EXPECT_DOUBLE_EQ(xf[nx + 2 * ng].item<double>(), hi + ng * dx);
  auto d = xf.narrow(0, 1, xf.size(0) - 1) - xf.narrow(0, 0, xf.size(0) - 1);
  EXPECT_TRUE(torch::isfinite(d).all().item<bool>());
  EXPECT_LT((d - dx).abs().max().item<double>(), 1.e-15 * std::abs(dx));
}
}  // namespace

// (A) faces carry the caller's bounds on every axis
TEST(CoordinateProgrammatic, faces_span_the_callers_bounds_on_every_axis) {
  Cartesian coord(a3_opts());
  expect_axis(coord->x1f, -2., 4., 3, 3);
  expect_axis(coord->x2f, 10., 30., 4, 3);
  expect_axis(coord->x3f, -1., 2., 6, 3);
}

// (B) the SPACING buffers -- what kills a shape-only-correct fix
TEST(CoordinateProgrammatic, spacings_are_finite_and_match_the_callers_grid) {
  Cartesian coord(a3_opts());
  struct {
    torch::Tensor f, v;
    double dx;
  } ax[] = {{coord->dx1f, coord->dx1v, 6. / 3.},
            {coord->dx2f, coord->dx2v, 20. / 4.},
            {coord->dx3f, coord->dx3v, 3. / 6.}};
  for (auto const& a : ax) {
    ASSERT_TRUE(torch::isfinite(a.f).all().item<bool>());  // +inf mode
    ASSERT_TRUE(torch::isfinite(a.v).all().item<bool>());
    EXPECT_LT((a.f - a.dx).abs().max().item<double>(), 1.e-15 * a.dx);
    EXPECT_LT((a.v - a.dx).abs().max().item<double>(), 1.e-15 * a.dx);
  }
}

// (C) the metric built on those spacings
TEST(CoordinateProgrammatic, cell_volume_is_the_product_of_the_callers_widths) {
  Cartesian coord(a3_opts());
  double want = (6. / 3.) * (20. / 4.) * (3. / 6.);
  auto vol = coord->cell_volume();
  ASSERT_TRUE(torch::isfinite(vol).all().item<bool>());
  EXPECT_LT((vol - want).abs().max().item<double>(), 1.e-14 * want);
}

// (D) the MIXED shape a real call site uses (test_exchange.py: nx1 == 1 beside
// live x2/x3)
TEST(CoordinateProgrammatic, a_degenerate_axis_beside_live_ones) {
  auto op = CoordinateOptionsImpl::create();
  op->nx1(1).x1min(5.).x1max(10.);
  op->nx2(4).x2min(0.).x2max(2.);
  op->nx3(6).x3min(0.).x3max(3.);
  op->nghost(3);
  Cartesian coord(op);

  ASSERT_EQ(coord->x1f.size(0), 2);
  EXPECT_DOUBLE_EQ(coord->x1f[0].item<double>(), 5.);
  EXPECT_DOUBLE_EQ(coord->x1f[1].item<double>(), 10.);
  expect_axis(coord->x2f, 0., 2., 4, 3);
  expect_axis(coord->x3f, 0., 3., 6, 3);
  EXPECT_TRUE(torch::isfinite(coord->dx1f).all().item<bool>());
  EXPECT_LT((coord->dx1f - 5.).abs().max().item<double>(), 1.e-14);
}

// (E) resolving must not corrupt the caller's options, and must be idempotent
TEST(CoordinateProgrammatic, two_coordinates_from_one_options_object_agree) {
  auto op = a3_opts();
  Cartesian first(op);
  Cartesian second(op);
  EXPECT_TRUE(torch::equal(first->x1f, second->x1f));
  EXPECT_TRUE(torch::equal(first->x2f, second->x2f));
  EXPECT_TRUE(torch::equal(first->x3f, second->x3f));
}

// the SILENT variant: ix resolves to 0, so the block quietly gets the global
// default [0, 1] back instead of the caller's [0, 10]. No abort, wrong grid.
TEST(CoordinateProgrammatic,
     a_degenerate_axis_is_not_silently_the_global_default) {
  auto op = CoordinateOptionsImpl::create();
  op->nx1(1).x1min(0.).x1max(10.);
  op->nx2(1).nx3(1).nghost(3);
  Cartesian coord(op);

  ASSERT_EQ(coord->x1f.size(0), 2);
  EXPECT_DOUBLE_EQ(coord->x1f[0].item<double>(), 0.);
  EXPECT_DOUBLE_EQ(coord->x1f[1].item<double>(), 10.);
}

// (F) the yaml path must STAND DOWN: globals stay as the card declared them.
// This gates the SENTINEL, not the design: it reads the options only, so a
// revert to block-local face arithmetic would leave it green. The one test
// here that such a revert fails is (J), below.
TEST(CoordinateProgrammatic, the_yaml_path_keeps_its_declared_global_grid) {
  auto block =
      MeshBlock(MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml"));
  auto op = block->pcoord->options;
  EXPECT_GT(op->global_nx1(), 0);
  EXPECT_GT(op->global_nx2(), 0);
  EXPECT_GT(op->global_nx3(), 0);
  EXPECT_EQ(op->global_nx1(), op->nx1());
  EXPECT_EQ(op->global_nx2(), op->nx2());
  EXPECT_EQ(op->global_nx3(), op->nx3());
}

// (I) a card that declares `bounds:` but no `cells:` must keep its declared
// domain
TEST(CoordinateProgrammatic, a_bounds_only_card_keeps_its_declared_domain) {
  auto op =
      CoordinateOptionsImpl::from_yaml("test_coordinate_bounds_only.yaml");
  Cartesian coord(op);

  EXPECT_DOUBLE_EQ(op->global_x1min(), 100.);
  EXPECT_DOUBLE_EQ(op->global_x1max(), 400.);
  EXPECT_DOUBLE_EQ(op->global_x2min(), -5.);
  EXPECT_DOUBLE_EQ(op->global_x2max(), 5.);
  EXPECT_DOUBLE_EQ(op->global_x3min(), 0.);
  EXPECT_DOUBLE_EQ(op->global_x3max(), 2.);

  ASSERT_EQ(coord->x1f.size(0), 2);
  EXPECT_DOUBLE_EQ(coord->x1f[0].item<double>(), 100.);
  EXPECT_DOUBLE_EQ(coord->x1f[1].item<double>(), 400.);
  EXPECT_DOUBLE_EQ(coord->dx1f.abs().max().item<double>(), 300.);
}

namespace {
// x2 in [0, 1] with 6 cells: dx = 1/6 is NOT exactly representable, which is
// the whole point -- on a grid of exact binary values every arithmetic path
// agrees by luck and the test below would pass even for the block-local
// construction it exists to forbid.
CoordinateOptions declared_six_cells_in_x2() {
  auto op = CoordinateOptionsImpl::create();
  op->global_x1min(0.).global_x1max(1.).global_nx1(1);
  op->global_x2min(0.).global_x2max(1.).global_nx2(6);
  op->global_x3min(0.).global_x3max(1.).global_nx3(1);
  op->nghost(2);
  return op;
}
LayoutOptions split_x2(int nb2, int rank) {
  auto lo = LayoutOptionsImpl::create();
  lo->px(nb2).world_size(nb2).rank(rank);
  return lo;
}
}  // namespace

// (J) THE invariant the global grid exists for: decomposing the domain must not
// move a single face by a single bit, in any decomposition. Every other test in
// this file is single-block, so without this one a revert to block-local
// arithmetic stays green.
TEST(CoordinateDecomposition, blocks_match_the_undecomposed_grid_bitwise) {
  auto whole = declared_six_cells_in_x2();
  whole->x1min(0.).x1max(1.).nx1(1);
  whole->x2min(0.).x2max(1.).nx2(6);
  whole->x3min(0.).x3max(1.).nx3(1);
  Cartesian ref(whole);

  int ng = 2;
  for (int nb2 : {2, 3}) {
    int seen = 0;
    for (int rank = 0; rank < nb2; ++rank) {
      auto op = declared_six_cells_in_x2();
      op->repartition(split_x2(nb2, rank));
      Cartesian blk(op);

      ASSERT_EQ(op->nx2(), 6 / nb2) << "nb2=" << nb2 << " rank=" << rank;
      ASSERT_EQ(op->ix2(), rank * (6 / nb2))
          << "nb2=" << nb2 << " rank=" << rank;
      seen += op->nx2();

      auto got = blk->x2f.narrow(0, ng, op->nx2() + 1);
      auto want = ref->x2f.narrow(0, ng + op->ix2(), op->nx2() + 1);
      EXPECT_TRUE(
          torch::equal(got, want))  // BITWISE. allclose would not see 1 ULP.
          << "nb2=" << nb2 << " rank=" << rank << "\n  got  " << got
          << "\n  want " << want;
    }
    EXPECT_EQ(seen, 6) << "nb2=" << nb2;
  }
}

// (K) an output slice must be checked against the domain the CALLER declared.
// The output loop in MeshBlockImpl's ctor reads the globals directly, long
// before pcoord resolves them, and the dangerous direction is not the rejection
// -- it is the silent ACCEPTANCE.
TEST(CoordinateProgrammatic,
     an_output_slice_is_validated_against_the_callers_domain) {
  auto mk = [](double slice) {
    // the same card the rest of this binary uses: kintera's species tables are
    // process-global and the first card wins, so a moist card here would not
    // find its own `vapor`.
    auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
    op->hydro()->eos()->type() = "ideal-gas";
    op->hydro()->riemann()->type() = "lmars";
    auto co = CoordinateOptionsImpl::create();
    co->nx1(4).x1min(10.).x1max(30.).nghost(2);
    op->coord(co);
    auto out = OutputOptionsImpl::create();
    out->file_type("restart");
    out->x1_slice({slice});
    op->outputs(std::vector<OutputOptions>{out});
    return op;
  };
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(mk(20.)));  // inside [10, 30)
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(mk(0.5)));  // outside it
}

namespace {
// the text of the abort, or "" when the coordinate built
std::string build_error(CoordinateOptions const& op) {
  try {
    Cartesian coord(op);
  } catch (std::exception const& exc) {
    return std::string(exc.what());
  }
  return "";
}
CoordinateOptions one_live_axis() {
  auto op = CoordinateOptionsImpl::create();
  op->nx1(1).x1min(0.).x1max(1.);
  op->nx2(4).x2min(0.).x2max(2.);
  op->nx3(1).x3min(0.).x3max(1.);
  op->nghost(2);
  return op;
}
}  // namespace

// (L) a block that cannot be a grid must be refused where the grid is formed.
// Without the fix nx2 <= 0 aborts inside dx2() blaming the resolve, and a zero
// span builds silently with zero cell spacing, so the message is asserted too.
TEST(CoordinateProgrammatic,
     a_block_that_cannot_be_the_global_grid_is_refused) {
  EXPECT_EQ(build_error(one_live_axis()), "");  // the control still builds

  auto refused = [](CoordinateOptions const& op) {
    auto msg = build_error(op);
    EXPECT_NE(msg.find("cannot adopt this block as the global x2 grid"),
              std::string::npos)
        << msg;
  };
  auto no_cells = one_live_axis();
  no_cells->nx2(0);
  refused(no_cells);
  auto flat = one_live_axis();
  flat->x2max(0.);  // zero span
  refused(flat);
}

// (M) one options object, two blocks: the second inherits the first's grid, so
// it must be checked against it instead of silently building on it.
TEST(CoordinateProgrammatic, a_block_outside_its_declared_grid_is_refused) {
  auto op = one_live_axis();
  op->nx2(6).x2min(0.).x2max(1.);
  Cartesian first(op);  // adopts [0, 1] / 6 as the global x2 grid

  op->nx2(3).x2min(2.).x2max(3.);  // a different block entirely
  auto msg = build_error(op);
  EXPECT_NE(msg.find("lies outside the declared global x2 grid"),
            std::string::npos)
      << msg;
}

// (N) inside the global interval is not enough. block_faces_ slices by the
// rounded start and the local nx, so a block of [0, 0.3] with nx 2 on a
// [0, 1] grid of 10 cells used to be accepted and its interior faces ended
// at 0.2, not 0.3.
TEST(CoordinateProgrammatic,
     a_declared_block_must_span_exactly_nx_global_faces) {
  auto grid = [](double lo, double hi, int nx) {
    auto op = CoordinateOptionsImpl::create();
    op->global_x1min(0.).global_x1max(1.).global_nx1(1);
    op->global_x2min(0.).global_x2max(1.).global_nx2(10);
    op->global_x3min(0.).global_x3max(1.).global_nx3(1);
    op->x1min(0.).x1max(1.).nx1(1);
    op->x2min(lo).x2max(hi).nx2(nx);
    op->x3min(0.).x3max(1.).nx3(1);
    op->nghost(2);
    return op;
  };

  auto bad = grid(0., 0.3, 2);
  auto msg = build_error(bad);
  EXPECT_NE(msg.find("not exactly 2 cells of the declared global x2 grid"),
            std::string::npos)
      << msg;

  auto off = grid(0., 0.25, 2);  // inside [0, 1], on no face
  msg = build_error(off);
  EXPECT_NE(msg.find("not exactly 2 cells of the declared global x2 grid"),
            std::string::npos)
      << msg;

  auto none = grid(0., 0.2, 0);
  msg = build_error(none);
  EXPECT_NE(msg.find("need nx2 > 0"), std::string::npos) << msg;

  double dx = 1. / 10.;
  auto good = grid(0., 2. * dx, 2);
  EXPECT_EQ(build_error(good), "");
  Cartesian blk(good);
  EXPECT_EQ(good->ix2(), 0);
  EXPECT_NEAR(blk->x2f[2 + 2].item<double>(), 2. * dx, 1.e-15);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}
