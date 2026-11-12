// gtest
#include <gtest/gtest.h>

// C/C++
#include <fstream>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

constexpr int N = 16;

//! six N x N panels, one block each, all in this process
Mesh make_sphere() {
  std::string path = "test_cubed_sphere_seams.yaml";
  std::ofstream(path)
      << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species: [{name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, "
         "cv_R: 2.5}]\n"
         "dynamics: {equation-of-state: {type: ideal-gas, gammad: 1.4}}\n"
         "distribute: {layout: cubed-sphere, nb2: 1, nb3: 1, "
         "blocks_per_process: 6}\n"
         "geometry:\n"
         "  type: gnomonic-equiangle\n"
         "  cells: {nx1: 4, nx2: "
      << N << ", nx3: " << N << ", nghost: 3}\n"
      << "  bounds: {x1min: 6.4e6, x1max: 6.41e6, x2min_pi: -0.25, "
         "x2max_pi: 0.25, x3min_pi: -0.25, x3max_pi: 0.25}\n"
         "boundary-condition: {external: {x1-inner: reflecting, x1-outer: "
         "reflecting, x2-inner: custom, x2-outer: custom, x3-inner: custom, "
         "x3-outer: custom}}\n"
         "integration: {type: rk3, cfl: 0.4, implicit-scheme: 0}\n";
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(MeshBlockOptionsImpl::from_yaml(path));
  mesh_opts->blocks_per_process(6);
  return Mesh(mesh_opts);
}

// A seam face is one physical face: both panels hold the same area, bit for
// bit (with an x2 arc on one side and an x3 arc on the other, some differed).
TEST(CubedSphereSeams, seam_face_areas_are_single_valued) {
  auto g = make_sphere()->blocks[0]->pcoord;
  int ng = g->options->nghost(), is = g->il(), ie = g->iu() + 1;
  auto a2 = g->face_area2(), a3 = g->face_area3();
  auto side = [&](int s, int p) {
    if (s == SIDE_L || s == SIDE_R)
      return a2.select(1, s == SIDE_L ? ng : ng + N).select(0, ng + p);
    return a3.select(0, s == SIDE_B ? ng : ng + N).select(0, ng + p);
  };
  int differ = 0;
  for (int f = 0; f < 6; ++f)
    for (int s = 0; s < 4; ++s) {
      auto e = CS_FACE_EDGES[f][s];
      for (int p = 0; p < N; ++p)
        differ += !torch::equal(
            side(s, p).slice(0, is, ie),
            side(e.nside, e.rev ? N - 1 - p : p).slice(0, is, ie));
    }
  EXPECT_EQ(differ, 0) << "seam face pairs that differ";
}

//! a cubed-sphere layout whose every neighbour is the block itself
struct SelfNeighbourLayout : CubedSphereLayoutImpl {
  using CubedSphereLayoutImpl::CubedSphereLayoutImpl;
  int neighbor_rank(std::tuple<int, int, int>,
                    std::tuple<int, int, int>) const override {
    return options->rank();
  }
};

// A self-send across a seam would fill the wrong ghosts; it is refused.
TEST(CubedSphereSeams, self_send_across_a_seam_is_refused) {
  auto mesh = make_sphere();
  auto pmb = mesh->blocks[0].get();
  SelfNeighbourLayout layout(pmb->options->layout(), pmb);
  std::vector<CommWorkPtr> works;
  try {
    layout.exchange_remote(pmb, SyncOptions(), works);
    FAIL() << "a cubed-sphere self-send was not refused";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("its own neighbour"),
              std::string::npos);
  }
}
