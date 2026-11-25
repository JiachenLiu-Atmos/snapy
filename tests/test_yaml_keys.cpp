// C/C++
#include <fstream>
#include <string>

// gtest
#include <gtest/gtest.h>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

// kintera's species table is process-global, so one card for the whole binary
constexpr char const* kCard = "test_cycle_diagnostics.yaml";

// Load `card` as a run does; the error message, or "" when it loads.
std::string load_error(YAML::Node const& card, std::string const& name) {
  {
    std::ofstream out(name);
    out << card;
  }
  try {
    MeshBlockOptionsImpl::from_yaml(name);
  } catch (std::exception const& e) {
    return e.what();
  }
  return "";
}

// A misspelled key must be refused, and the message must name it by path.
// A part of `block` that names a list item is its index.
void expect_refused(YAML::Node card, std::string const& block,
                    std::string const& key) {
  auto node = card;
  std::string rest = block;
  for (auto pos = rest.find('/'); !rest.empty(); pos = rest.find('/')) {
    auto part = rest.substr(0, pos);
    if (node.IsSequence()) {
      node.reset(node[std::stoi(part)]);
    } else {
      node.reset(node[part]);
    }
    rest = pos == std::string::npos ? "" : rest.substr(pos + 1);
  }
  node[key] = 1;
  auto msg = load_error(card, "test_yaml_keys_card.yaml");
  EXPECT_NE(msg.find("unknown key '" + block + "/" + key + "'"),
            std::string::npos)
      << "loaded or failed otherwise: '" << msg << "'";
}

void expect_refused(std::string const& block, std::string const& key) {
  expect_refused(YAML::LoadFile(kCard), block, key);
}

}  // namespace

TEST(yaml_keys, the_card_itself_loads) {
  EXPECT_EQ(load_error(YAML::LoadFile(kCard), "test_yaml_keys_card.yaml"), "");
}

TEST(yaml_keys, const_gravity_refuses_an_unknown_key) {
  expect_refused("forcing/const-gravity", "garv1");
}

TEST(yaml_keys, geometry_cells_refuses_an_unknown_key) {
  expect_refused("geometry/cells", "nx4");
}

TEST(yaml_keys, external_boundaries_refuse_an_unknown_key) {
  expect_refused("boundary-condition/external", "x1-iner");
}

TEST(yaml_keys, dynamics_reconstruct_refuses_an_unknown_key) {
  expect_refused("dynamics/reconstruct", "horizonal");
}

TEST(yaml_keys, dynamics_reconstruct_vertical_refuses_an_unknown_key) {
  expect_refused("dynamics/reconstruct/vertical", "tpye");
}

TEST(yaml_keys, dynamics_reconstruct_horizontal_refuses_an_unknown_key) {
  expect_refused("dynamics/reconstruct/horizontal", "schock");
}

TEST(yaml_keys, dynamics_riemann_solver_refuses_an_unknown_key) {
  expect_refused("dynamics/riemann-solver", "tpye");
}

TEST(yaml_keys, coriolis_refuses_an_unknown_key) {
  expect_refused("forcing/coriolis", "omega_3");
}

TEST(yaml_keys, diffusion_refuses_an_unknown_key) {
  expect_refused("forcing/diffusion", "nu");
}

TEST(yaml_keys, body_heat_refuses_an_unknown_key) {
  expect_refused("forcing/body-heat", "dtdt");
}

TEST(yaml_keys, top_cool_refuses_an_unknown_key) {
  expect_refused("forcing/top-cool", "flx");
}

TEST(yaml_keys, bot_heat_refuses_an_unknown_key) {
  expect_refused("forcing/bot-heat", "flx");
}

TEST(yaml_keys, relax_bot_comp_refuses_an_unknown_key) {
  expect_refused("forcing/relax-bot-comp", "xfrc");
}

TEST(yaml_keys, relax_bot_temp_refuses_an_unknown_key) {
  expect_refused("forcing/relax-bot-temp", "at_face");
}

TEST(yaml_keys, relax_bot_velo_refuses_an_unknown_key) {
  expect_refused("forcing/relax-bot-velo", "bv_x");
}

TEST(yaml_keys, top_sponge_lyr_refuses_an_unknown_key) {
  expect_refused("forcing/top-sponge-lyr", "widht");
}

TEST(yaml_keys, bot_sponge_lyr_refuses_an_unknown_key) {
  expect_refused("forcing/bot-sponge-lyr", "widht");
}

TEST(yaml_keys, plume_forcing_refuses_an_unknown_key) {
  // read only under the plume EOS: same card, same species
  auto card = YAML::LoadFile(kCard);
  card["dynamics"]["equation-of-state"]["type"] = "plume-eos";
  expect_refused(card, "forcing/plume-forcing", "n2");
}

TEST(yaml_keys, geometry_refuses_an_unknown_key) {
  expect_refused("geometry", "bound");
}

TEST(yaml_keys, geometry_bounds_refuses_an_unknown_key) {
  expect_refused("geometry/bounds", "x1_max");
}

TEST(yaml_keys, outputs_refuses_an_unknown_key) {
  auto card = YAML::LoadFile(kCard);
  card["outputs"].push_back(YAML::Load("{type: netcdf}"));
  expect_refused(card, "outputs/0", "DT");
}

TEST(yaml_keys, boundary_condition_refuses_an_unknown_key) {
  expect_refused("boundary-condition", "internl");
}

TEST(yaml_keys, boundary_condition_internal_refuses_an_unknown_key) {
  expect_refused("boundary-condition/internal", "max_iter");
}

TEST(yaml_keys, scalar_refuses_an_unknown_key) {
  expect_refused("scalar", "upper_bound");
}

TEST(yaml_keys, scalar_reconstruct_refuses_an_unknown_key) {
  expect_refused("scalar/reconstruct", "tpye");
}

TEST(yaml_keys, scalar_reconstruct_refuses_scale) {
  // the scalar always reconstructs scale-invariant: no scale key
  expect_refused("scalar/reconstruct", "scale");
}

TEST(yaml_keys, scalar_riemann_solver_refuses_an_unknown_key) {
  expect_refused("scalar/riemann-solver", "tpye");
}

TEST(yaml_keys, sedimentation_refuses_an_unknown_key) {
  expect_refused("sedimentation", "upper_limit");
}

TEST(yaml_keys, distribute_refuses_an_unknown_key) {
  expect_refused("distribute", "nb_2");
}

TEST(yaml_keys, integration_refuses_an_unknown_key) {
  expect_refused("integration", "CFL");
}
