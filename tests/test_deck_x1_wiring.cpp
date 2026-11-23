#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/x1_centroid.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_rows.hpp>
#include <snap/scalar/scalar.hpp>
#include <string>
#include <utility>
#include <vector>

// tests
#include "make_interp.hpp"

using namespace snap;

namespace {
constexpr int kNx1 = 16, kNghost = 3;
constexpr double kX1min = 0., kX1max = 16.;
enum { DIM1 = 3, DIM2 = 2 };
std::string deck(std::string const& cells_extra,
                 std::string const& geom_extra = "") {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: %g, x1max: %g, x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 8, nx3: 1, nghost: %d%s}
%sdynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars
scalar:
  nvar: 1
  names: [tracer]
  reconstruct: {type: weno5, shock: false}
integration:
  type: rk3
  cfl: 0.4
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic
)",
                kX1min, kX1max, kNx1, kNghost, cells_extra.c_str(),
                geom_extra.c_str());
  return buf;
}
Mesh make_mesh(std::string const& config) {
  char fname[] = "/tmp/test-deck-x1-wiring-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);
  return Mesh(mesh_opts);
}
std::string face_list(double q) {
  double lq = std::log(q), den = std::expm1(kNx1 * lq);
  std::string s = "  x1f: [";
  char b[32];
  for (int k = 0; k < kNx1; ++k) {
    std::snprintf(b, sizeof(b), "%.17g",
                  kX1min + (kX1max - kX1min) * std::expm1(k * lq) / den);
    s += (k ? ", " : "") + std::string(b);
  }
  std::snprintf(b, sizeof(b), "%.17g", kX1max);
  return s + ", " + b + "]\n";
}
using Tables = std::pair<torch::Tensor, torch::Tensor>;
Tables installed(Interp const& p) {
  bool rows = p->scheme == Scheme::weno5 || p->scheme == Scheme::cp5;
  EXPECT_TRUE(rows) << "a weno5 deck holds weno5 + cp5";
  return rows ? Tables{p->cm_x1, p->cp_x1} : Tables{};
}
Tables from_faces(Interp const& p, torch::Tensor const& x1f) {
  return p->scheme == Scheme::weno5 ? weno5_x1_coeffs(x1f) : cp5_x1_coeffs(x1f);
}
void expect_faces_installed(ReconstructImpl const& recon,
                            torch::Tensor const& x1f) {
  for (auto const& p : {recon.pinterp1, recon.pinterp2}) {
    EXPECT_TRUE(p->has_x1_coeffs())
        << p->options->type()
        << " on recon1 has no x1 coefficients: the deck's stretched x1 never "
           "reached the reconstructor";
    auto got = installed(p), want = from_faces(p, x1f);
    ASSERT_EQ(got.first.numel(), want.first.numel()) << p->options->type();
    EXPECT_TRUE(torch::equal(got.first, want.first))
        << "installed cm rows are not the ones this block's x1f gives";
    EXPECT_TRUE(torch::equal(got.second, want.second))
        << "installed cp rows are not the ones this block's x1f gives";
  }
}
void expect_no_faces(ReconstructImpl const& recon, char const* what) {
  for (auto const& p : {recon.pinterp1, recon.pinterp2}) {
    EXPECT_FALSE(p->has_x1_coeffs()) << what << " must hold no x1 table";
    auto got = installed(p);
    EXPECT_EQ(got.first.numel(), 0) << what;
    EXPECT_EQ(got.second.numel(), 0) << what;
  }
}
std::shared_ptr<ReconstructImpl> scalar_x1(Scalar const& ps) {
  for (auto const& item : ps->named_children()) {
    if (item.key() == "recon1")
      return std::dynamic_pointer_cast<ReconstructImpl>(item.value());
  }
  return ps->precon.ptr();
}
void expect_forward_runs(MeshBlock pmb) {
  auto coord = pmb->pcoord;
  int nc1 = coord->options->nc1(), nc2 = coord->options->nc2(),
      nc3 = coord->options->nc3();
  auto w = torch::ones({pmb->phydro->peos->nvar(), nc3, nc2, nc1},
                       coord->x1v.options());
  auto wlr = pmb->phydro->precon1->forward(w, DIM1);
  EXPECT_EQ(wlr.size(DIM1 + 1), nc1);
  auto r =
      torch::ones({pmb->pscalar->nvar(), nc3, nc2, nc1}, coord->x1v.options());
  EXPECT_EQ(scalar_x1(pmb->pscalar)->forward(r, DIM1).size(DIM1 + 1), nc1);
  EXPECT_EQ(pmb->pscalar->precon->forward(r, DIM2).size(DIM2 + 1), nc2);
}
void check_stretched(std::string const& config) {
  auto mesh = make_mesh(config);
  auto pmb = mesh->blocks.front();
  ASSERT_TRUE(pmb->pcoord->options->stretched_x1())
      << "the deck's grid key did not reach the coordinate";
  expect_faces_installed(*pmb->phydro->precon1, pmb->pcoord->x1f);
  expect_faces_installed(*scalar_x1(pmb->pscalar), pmb->pcoord->x1f);
  expect_no_faces(*pmb->phydro->precon23, "recon23");
  expect_no_faces(*pmb->pscalar->precon, "scalar x2/x3");
  expect_forward_runs(pmb);
}
}  // namespace

TEST(DeckX1Wiring, x1rat_deck_installs_the_block_faces_on_recon1) {
  check_stretched(deck(", x1rat: 1.05"));
}

TEST(DeckX1Wiring, face_list_deck_installs_the_block_faces_on_recon1) {
  check_stretched(deck("", face_list(1.05)));
}

// a weno5 deck on a spherical-polar x1 in [1, 2] with nx1 cells
Mesh spherical_mesh(int nx1) {
  auto config = deck("");
  for (auto [from, to] : {std::pair<std::string, std::string>{
                              "type: cartesian", "type: spherical-polar"},
                          {"x1min: 0, x1max: 16", "x1min: 1, x1max: 2"},
                          {"nx1: 16", "nx1: " + std::to_string(nx1)}}) {
    auto at = config.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    if (at != std::string::npos) config.replace(at, from.size(), to);
  }
  return make_mesh(config);
}
// the r^3/3 table of a single block, ghost widths mirrored at both walls
torch::Tensor r3_table(torch::Tensor const& r, int ng) {
  auto v = r.pow(3) / 3.;
  int n = v.size(0);
  v.slice(0, 0, ng).copy_(2. * v[ng] - v.slice(0, ng + 1, 2 * ng + 1).flip(0));
  v.slice(0, n - ng).copy_(2. * v[n - 1 - ng] -
                           v.slice(0, n - 1 - 2 * ng, n - 1 - ng).flip(0));
  return v;
}

// the scalar reconstructs the r^2 dr averages themselves, so its x1 rows are
// the r^3/3 table's with or without SNAP_X1_CENTROID_EXACT (ctest entry
// test_deck_x1_wiring_x1_centroid); hydro's are too without it, and with it
// hydro reconstructs plain means and holds none
TEST(DeckX1Wiring, spherical_scalar_x1_faces_keep_the_r3_table) {
  auto mesh = spherical_mesh(16);
  auto pmb = mesh->blocks.front();
  ASSERT_EQ(pmb->pcoord->options->type(), "spherical-polar");
  auto v = r3_table(pmb->pcoord->x1f, pmb->pcoord->options->nghost());
  expect_faces_installed(*scalar_x1(pmb->pscalar), v);
  if (x1_centroid_exact_enabled())
    expect_no_faces(*pmb->phydro->precon1, "hydro x1 on plain means");
  else
    expect_faces_installed(*pmb->phydro->precon1, v);
  expect_forward_runs(pmb);
}

// Jiheng's probe: cells holding the r^2 dr averages of e^r on a
// spherical-polar x1 in [1, 2]. The scalar's weno5 x1 faces are no less
// accurate than with its parent's r^3/3 table, at each resolution, on the
// faces whose stencil lies in the owned cells and on all owned faces (the
// ghosts hold the averages over the ghost cells, which the wall-mirrored
// table does not assume, so the wall faces are second order)
TEST(DeckX1Wiring, spherical_scalar_x1_face_error_no_worse_than_r3_table) {
  for (int nx1 : {16, 32, 64, 128}) {
    auto mesh = spherical_mesh(nx1);
    auto pmb = mesh->blocks.front();
    auto const& r = pmb->pcoord->x1f;
    int ng = pmb->pcoord->options->nghost();
    auto prim = [](torch::Tensor x) { return x.exp() * (x * x - 2. * x + 2.); };
    auto avg = (prim(r.slice(0, 1)) - prim(r.slice(0, 0, -1))) /
               ((r.slice(0, 1).pow(3) - r.slice(0, 0, -1).pow(3)) / 3.);
    auto q = avg.view({1, 1, 1, -1})
                 .expand({1, pmb->pcoord->options->nc3(),
                          pmb->pcoord->options->nc2(), -1});
    auto scalar = scalar_x1(pmb->pscalar);
    auto ref = std::make_shared<ReconstructImpl>(scalar->options);
    ref->set_x1_faces(r3_table(r, ng));
    // |face - e^r| on both states of the owned faces ng .. ng + nx1
    auto exact = r.slice(0, ng, ng + nx1 + 1).exp();
    auto error = [&](ReconstructImpl& recon) {
      auto lr = recon.forward(q.clone(), DIM1).select(1, 0).select(1, 0);
      auto e = (lr.select(1, 0).slice(-1, ng, ng + nx1 + 1) - exact).abs();
      return std::pair{e.slice(-1, 3, nx1 - 2).max().item<double>(),
                       e.max().item<double>()};
    };
    auto [got_in, got_all] = error(*scalar);
    auto [want_in, want_all] = error(*ref);
    std::printf(
        "nx1 %3d SNAP_X1_CENTROID_EXACT=%d interior: scalar %.4e "
        "r3 %.4e; all faces: scalar %.4e r3 %.4e\n",
        nx1, (int)x1_centroid_exact_enabled(), got_in, want_in, got_all,
        want_all);
    EXPECT_LE(got_in, want_in * (1. + 1e-6))
        << "nx1=" << nx1 << ": interior x1 face error of the scalar";
    EXPECT_LE(got_all, want_all * (1. + 1e-6))
        << "nx1=" << nx1 << ": x1 face error of the scalar";
  }
}

TEST(DeckX1Wiring, uniform_deck_installs_nothing) {
  for (char const* cells : {"", ", x1rat: 1."}) {
    auto mesh = make_mesh(deck(cells));
    auto pmb = mesh->blocks.front();
    EXPECT_FALSE(pmb->pcoord->options->stretched_x1())
        << "a uniform deck must not look stretched";
    expect_no_faces(*pmb->phydro->precon1, "recon1 on a uniform deck");
    expect_no_faces(*scalar_x1(pmb->pscalar), "scalar x1 on a uniform deck");
    expect_no_faces(*pmb->phydro->precon23, "recon23 on a uniform deck");
    expect_no_faces(*pmb->pscalar->precon, "scalar x2/x3 on a uniform deck");
    expect_forward_runs(pmb);
  }
}
