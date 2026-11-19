// An ideal-gas EOS refuses a card with vapor, cloud or reactions at
// construction (it used to fail in the first limiter or saturation step), and
// a float32 field cast to the host by Frame::to_host() keeps its values.
#include <gtest/gtest.h>
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/output/writer_core.hpp>
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

TEST(Frame, to_host_keeps_float32_values) {
  torch::manual_seed(3);
  auto t4 = torch::rand({3, 2, 4, 5}, torch::kFloat32) + 1.;
  auto t3 = torch::rand({2, 4, 5}, torch::kFloat32) + 1.;
  auto d3 = torch::rand({2, 4, 5}, torch::kFloat64);
  Frame fr;
  fr.fields = {{"a", t4}, {"b", t3}, {"c", d3}};
  fr.to_host();
  for (auto const& f : fr.fields) {
    EXPECT_EQ(f.data.scalar_type(), torch::kFloat64);
    EXPECT_EQ(f.data.dim(), 4);
  }
  auto back4 = fr.fields[0].data;
  EXPECT_EQ((back4 - t4.to(torch::kFloat64)).abs().max().item<double>(), 0.);
  auto back3 = fr.fields[1].data.view({2, 4, 5});
  EXPECT_EQ((back3 - t3.to(torch::kFloat64)).abs().max().item<double>(), 0.);
  // float64 keeps its bytes, in a copy of its own
  auto backd = fr.fields[2].data.view({2, 4, 5});
  EXPECT_TRUE(torch::equal(backd, d3));
  EXPECT_NE(backd.data_ptr(), d3.data_ptr());
}
