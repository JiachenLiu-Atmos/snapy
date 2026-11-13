// An ideal-gas EOS refuses a card with vapor, cloud or reactions at
// construction (it used to fail in the first limiter or saturation step), and
// a float32 tensor copied into AthenaArray<double> keeps its values.
#include <gtest/gtest.h>
#include <snap/snap.h>

#include <snap/interface/athena_arrays.hpp>
#include <snap/mesh/meshblock.hpp>
#include <string>

using namespace snap;

TEST(MoistEOS, ideal_gas_refuses_a_moist_card) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  op->hydro()->eos()->type() = "ideal-gas";
  try {
    std::make_shared<MeshBlockImpl>(op);
    ADD_FAILURE() << "ideal-gas accepted a moist card";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("ideal-gas has no vapor"),
              std::string::npos)
        << e.what();
  }
}

TEST(MoistEOS, ideal_gas_accepts_a_card_without_species) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_eos_temp2inteng.yaml");
  op->hydro()->eos()->type() = "ideal-gas";
  op->hydro()->eos()->thermo() = kintera::ThermoOptionsImpl::create();
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(op));
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
