// The shared moist-EOS base, the ideal-gas card check and the array container
// that output reads tensors through:
//  - ideal-moist and moist-mixture agree on the same state (shared layout code,
//    each type's own thermodynamics);
//  - an ideal-gas EOS refuses a card with vapor, cloud or reactions at
//    construction (it used to fail in the first limiter or saturation step);
//  - a float32 tensor copied into AthenaArray<double> keeps its values (the
//    blob over the double buffer used to carry the tensor's dtype, so float32
//    bytes were written into double storage);
//  - the jupiter_gcm deck still builds IdealMoistImpl, the type the closed-form
//    sedimentation kernel and the Roe fast path dispatch on.
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cstdlib>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/moist_mixture.hpp>
#include <snap/interface/athena_arrays.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <string>

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block(std::string const& type) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  options->hydro()->eos()->type() = type;
  return std::make_shared<MeshBlockImpl>(options);
}

double rel(torch::Tensor const& a, torch::Tensor const& b) {
  return ((a - b).abs().max() / b.abs().max()).item<double>();
}

}  // namespace

TEST(MoistEOS, ideal_moist_and_moist_mixture_agree) {
  auto im = make_block("ideal-moist");
  auto mm = make_block("moist-mixture");
  auto e1 = im->phydro->peos, e2 = mm->phydro->peos;
  ASSERT_EQ(e1->nvar(), e2->nvar());

  auto pcoord = im->pcoord;
  torch::manual_seed(7);
  auto w = torch::empty({e1->nvar(), pcoord->options->nc3(),
                         pcoord->options->nc2(), pcoord->options->nc1()},
                        torch::kFloat64);
  w[IDN].uniform_(0.5, 1.5);
  w.narrow(0, IVX, 3).uniform_(-10., 10.);
  w[IPR].uniform_(0.5e5, 1.5e5);
  w.narrow(0, ICY, e1->nvar() - ICY).uniform_(0., 0.02);

  // measured on this card: W->I 3.3e-12, W->U 1.2e-12, W->E 5.4e-13, the rest
  // below 1e-13 (same w, CPU)
  double const tol = 1.e-11;
  EXPECT_LE(
      rel(e1->compute("W->T", {w.clone()}), e2->compute("W->T", {w.clone()})),
      tol);
  EXPECT_LE(
      rel(e1->compute("W->I", {w.clone()}), e2->compute("W->I", {w.clone()})),
      tol);
  auto u1 = e1->compute("W->U", {w.clone()});
  auto u2 = e2->compute("W->U", {w.clone()});
  EXPECT_LE(rel(u1, u2), tol);
  EXPECT_LE(
      rel(e1->compute("U->W", {u1.clone()}), e2->compute("U->W", {u1.clone()})),
      tol);
  EXPECT_LE(
      rel(e1->compute("U->K", {u1.clone()}), e2->compute("U->K", {u1.clone()})),
      tol);
  auto g1 = e1->compute("W->A", {w.clone()});
  auto g2 = e2->compute("W->A", {w.clone()});
  EXPECT_LE(rel(g1, g2), tol);
  EXPECT_LE(rel(e1->compute("WA->L", {w.clone(), g1}),
                e2->compute("WA->L", {w.clone(), g2})),
            tol);
  EXPECT_LE(
      rel(e1->compute("W->E", {w.clone()}), e2->compute("W->E", {w.clone()})),
      tol);
  EXPECT_LE(
      rel(e1->species_enthalpy(w.clone()), e2->species_enthalpy(w.clone())),
      tol);
  EXPECT_EQ(e1->species_weight(1), e2->species_weight(1));
  EXPECT_EQ(e1->species_cv_ref(1), e2->species_cv_ref(1));
}

TEST(MoistEOS, ideal_gas_refuses_a_moist_card) {
  try {
    make_block("ideal-gas");
    ADD_FAILURE() << "an ideal-gas EOS accepted a card with vapor, cloud and "
                     "reactions";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("ideal-gas has no vapor"),
              std::string::npos)
        << e.what();
  }
}

TEST(AthenaArray, float32_tensor_keeps_its_values) {
  torch::manual_seed(3);
  auto t4 = torch::rand({3, 2, 4, 5}, torch::kFloat32) + 1.;
  AthenaArray<double> a4;
  a4.InitFromTensor(t4, 4);
  auto back4 = torch::from_blob(a4.data(), {3, 2, 4, 5}, torch::kFloat64);
  EXPECT_EQ((back4 - t4.to(torch::kFloat64)).abs().max().item<double>(), 0.);

  auto t3 = torch::rand({2, 4, 5}, torch::kFloat32) + 1.;
  AthenaArray<double> a3;
  a3.CopyFromTensor(t3);
  auto back3 = torch::from_blob(a3.data(), {2, 4, 5}, torch::kFloat64);
  EXPECT_EQ((back3 - t3.to(torch::kFloat64)).abs().max().item<double>(), 0.);

  // float64 keeps its bytes
  auto d3 = torch::rand({2, 4, 5}, torch::kFloat64);
  AthenaArray<double> b3;
  b3.CopyFromTensor(d3);
  auto backd = torch::from_blob(b3.data(), {2, 4, 5}, torch::kFloat64);
  EXPECT_TRUE(torch::equal(backd, d3));
}

TEST(MoistEOS, jupiter_gcm_deck_builds_ideal_moist) {
  char const* dir = std::getenv("SNAPY_EXAMPLES_DIR");
  if (dir == nullptr) GTEST_SKIP() << "SNAPY_EXAMPLES_DIR is not set";
  auto op =
      MeshBlockOptionsImpl::from_yaml(std::string(dir) + "/jupiter_gcm.yaml");
  op->outputs().clear();
  // the EOS type does not depend on the resolution: a small grid suffices
  auto c = op->coord();
  for (auto* n : {&c->nx1(), &c->nx2(), &c->nx3()}) *n = 4;
  for (auto* n : {&c->global_nx1(), &c->global_nx2(), &c->global_nx3()})
    if (*n > 0) *n = 4;
  auto mo = MeshOptionsImpl::create();
  mo->block(op);
  mo->blocks_per_process(6);
  auto mesh = Mesh(mo);
  ASSERT_EQ(mesh->blocks.size(), 6u);
  for (auto& blk : mesh->blocks) {
    auto peos = blk->phydro->peos;
    EXPECT_EQ(peos->options->type(), "ideal-moist");
    EXPECT_NE(dynamic_cast<IdealMoistImpl*>(peos.get()), nullptr);
    EXPECT_EQ(dynamic_cast<MoistMixtureImpl*>(peos.get()), nullptr);
  }
}
