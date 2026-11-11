// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/input/check_keys.hpp>

#include "sedimentation.hpp"

namespace snap {

SedHydroOptions SedHydroOptionsImpl::from_yaml(std::string const& filename) {
  auto config = YAML::LoadFile(filename);
  if (!config["sedimentation"]) return nullptr;

  auto op = SedHydroOptionsImpl::create();

  op->sedvel() = SedVelOptionsImpl::from_yaml(config["sedimentation"]);

  auto eos = EquationOfStateOptionsImpl::from_yaml(filename);

  // check all precipitating particles are in the clouds
  std::unordered_set<int> cloud_set(eos->thermo()->cloud_ids().begin(),
                                    eos->thermo()->cloud_ids().end());

  auto particle_ids = op->sedvel()->particle_ids();
  auto pass = std::all_of(particle_ids.begin(), particle_ids.end(),
                          [&](int x) { return cloud_set.count(x); });

  TORCH_CHECK(pass, "Missing sedimentation particles in the clouds.");

  // setup hydro ids
  auto hydro_species = eos->thermo()->species();
  for (auto const& p : op->sedvel()->species()) {
    auto it = std::find(hydro_species.begin(), hydro_species.end(), p);
    op->hydro_ids().push_back(ICY - 1 + it - hydro_species.begin());
  }

  return op;
}

SedVelOptions SedVelOptionsImpl::from_yaml(YAML::Node const& node) {
  check_keys(node, "sedimentation",
             {"radius", "density", "const-vsed", "a-diameter", "a-epsilon-LJ",
              "a-mass", "upper-limit"});
  auto op = SedVelOptionsImpl::create();

  // get all sedimentation particles
  std::set<std::string> particle_names;
  for (auto r : node["radius"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  for (auto r : node["density"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  for (auto r : node["const-vsed"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  // get particle ids
  for (auto& name : particle_names) {
    auto it = std::find(kintera::species_names.begin(),
                        kintera::species_names.end(), name);
    TORCH_CHECK(it != kintera::species_names.end(), "Sedimentation particle '",
                name, "' is not a valid species.");
    int id = it - kintera::species_names.begin();
    op->particle_ids().push_back(id);
    op->particle_names().push_back(name);
  }

  op->radius().resize(op->particle_ids().size(), 0.);
  op->density().resize(op->particle_ids().size(), 0.);
  op->const_vsed().resize(op->particle_ids().size(), 0.);

  // read particle radius
  auto species = op->species();
  for (auto r : node["radius"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    auto radius = r.second.as<double>();
    TORCH_CHECK(radius > 0., "Sedimentation radius must be positive.");
    op->radius()[it - species.begin()] = radius;
  }

  // read particle density
  for (auto r : node["density"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    auto density = r.second.as<double>();
    TORCH_CHECK(density > 0., "Sedimentation density must be positive.");
    op->density()[it - species.begin()] = density;
  }

  // read particle constant sedimentation velocity
  for (auto r : node["const-vsed"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    op->const_vsed()[it - species.begin()] = r.second.as<double>();
  }

  // every particle needs a positive radius (Stokes) or a non-zero const-vsed
  // (prescribed velocity); r = 0 in the Stokes branch is a NaN (kn = inf)
  auto names = op->species();
  for (size_t i = 0; i < op->particle_ids().size(); ++i) {
    TORCH_CHECK(op->radius()[i] > 0. || op->const_vsed()[i] != 0.,
                "Sedimentation particle '", names[i],
                "' needs radius > 0 or a non-zero const-vsed.");
  }

  // the defaults are the ADD_ARG values in sedimentation.hpp
  op->a_diameter() = node["a-diameter"].as<double>(op->a_diameter());
  op->a_epsilon_LJ() = node["a-epsilon-LJ"].as<double>(op->a_epsilon_LJ());
  op->a_mass() = node["a-mass"].as<double>(op->a_mass());
  op->upper_limit() = node["upper-limit"].as<double>(op->upper_limit());

  return op;
}

std::vector<std::string> SedVelOptionsImpl::species() const {
  if (particle_names().size() == particle_ids().size()) {
    return particle_names();
  }

  // options built by hand carry ids only: resolve them in the table as it is
  // now, as kintera does for a thermo not built from a card
  std::vector<std::string> species_list;
  for (int id : particle_ids()) {
    species_list.push_back(kintera::species_names[id]);
  }

  return species_list;
}

}  // namespace snap
