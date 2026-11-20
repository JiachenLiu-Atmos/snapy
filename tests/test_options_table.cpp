// Gate G-3 (lane O, O4): src/options/*.yaml give 730b3500's members, defaults
// and report text; lane S's tables read what its parsers read on every deck.

// C/C++
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// gtest
#include <gtest/gtest.h>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/output/output_type.hpp>

// generated from src/options/{coord,layout,internal_boundary}.yaml
#include <snap_options/coord_rows.hpp>
#include <snap_options/internal_boundary_rows.hpp>
#include <snap_options/layout_rows.hpp>

using namespace snap;
namespace fs = std::filesystem;

namespace {

std::vector<fs::path> in_tree_decks() {
  std::vector<fs::path> decks;
  for (auto dir : {"examples", "tests"}) {
    for (auto const& e :
         fs::directory_iterator(fs::path(SNAPY_SOURCE_DIR) / dir))
      if (e.path().extension() == ".yaml") decks.push_back(e.path());
  }
  std::sort(decks.begin(), decks.end());
  return decks;
}

// a deck that is not a YAML map is not a deck (e.g. a list fixture)
bool load(fs::path const& p, YAML::Node* config) {
  try {
    *config = YAML::LoadFile(p.string());
  } catch (std::exception const&) {
    return false;
  }
  return config->IsMap();
}

}  // namespace

// The pin's members and defaults (730b3500), default-constructed.
TEST(options_table, pin_defaults) {
  OutputOptionsImpl o;
  EXPECT_EQ(o.fid(), 0);
  EXPECT_EQ(o.dt(), 0.);
  EXPECT_FALSE(o.output_sumx1());
  EXPECT_FALSE(o.output_sumx2());
  EXPECT_FALSE(o.output_sumx3());
  EXPECT_FALSE(o.include_ghost_zones());
  EXPECT_FALSE(o.cartesian_vector());
  EXPECT_TRUE(o.x1_slice().empty());
  EXPECT_TRUE(o.x2_slice().empty());
  EXPECT_TRUE(o.x3_slice().empty());
  EXPECT_EQ(o.file_type(), "");
  EXPECT_EQ(o.data_format(), "");
  EXPECT_TRUE(o.variables().empty());
  EXPECT_FALSE(o.double_precision());
  EXPECT_TRUE(o.combine());
  EXPECT_FALSE(o.verbose());
  EXPECT_FALSE(o.super_resolution());

  HydroOptionsImpl h;
  EXPECT_FALSE(h.verbose());
  EXPECT_FALSE(h.disable_flux_x1());
  EXPECT_FALSE(h.disable_flux_x2());
  EXPECT_FALSE(h.disable_flux_x3());
  EXPECT_TRUE(h.wb_wall_clamp());
  EXPECT_EQ(h.conversion_order(), 2);
  EXPECT_FALSE(h.anomaly());
  EXPECT_FALSE(h.perturbation_flux());
  EXPECT_EQ(h.precision(), "float64");
  EXPECT_TRUE(h.flux_covariance());
  EXPECT_FALSE(h.x1_mass_covariance());
  EXPECT_EQ(h.conserve_axial_momentum(), "off");
  EXPECT_EQ(h.grav(), nullptr);
  EXPECT_EQ(h.eos(), nullptr);
}

// The pin's report text for the defaults (730b3500).
TEST(options_table, pin_report) {
  std::stringstream so;
  OutputOptionsImpl().report(so);
  EXPECT_EQ(so.str(),
            "-- output options --\n* fid = 0\n* dt = 0\n* output_sumx1 = 0\n"
            "* output_sumx2 = 0\n* output_sumx3 = 0\n"
            "* include_ghost_zones = 0\n* cartesian_vector = 0\n"
            "* double_precision = 0\n* x1_slice = []\n* x2_slice = []\n"
            "* x3_slice = []\n* file_type = \n* data_format = \n"
            "* variables = []\n* combine = 1\n* verbose = 0\n");
  std::stringstream sh;
  HydroOptionsImpl().report(sh);
  EXPECT_EQ(sh.str(),
            "-- hydro options --\n* verbose = 0\n* precision = float64\n"
            "* disable_flux_x1 = 0\n* disable_flux_x2 = 0\n"
            "* disable_flux_x3 = 0\n* flux_covariance = 1\n"
            "* x1_mass_covariance = 0\n* conserve_axial_momentum = off\n");
}

// SNAP_ANOMALY keeps its words and its refusal.
TEST(options_table, hydro_env_row) {
  auto deck = fs::temp_directory_path() / "test_options_table_hydro.yaml";
  {
    std::ofstream out(deck);
    out << "dynamics:\n  anomaly: false\n  conserve-axial-momentum: yes\n";
  }
  ::setenv("SNAP_ANOMALY", "on", 1);
  auto h = HydroOptionsImpl::from_yaml(deck.string());
  EXPECT_TRUE(h->anomaly());
  EXPECT_EQ(h->conserve_axial_momentum(), "transport");
  ::setenv("SNAP_ANOMALY", "", 1);
  EXPECT_FALSE(HydroOptionsImpl::from_yaml(deck.string())->anomaly());
  ::setenv("SNAP_ANOMALY", "maybe", 1);
  EXPECT_THROW(HydroOptionsImpl::from_yaml(deck.string()), c10::Error);
  ::unsetenv("SNAP_ANOMALY");
}

// A-ph2: exact replaces full and the closure key (A.rst 3.2, 10.3).
TEST(options_table, axial_mode_words) {
  auto deck = fs::temp_directory_path() / "test_options_table_axial.yaml";
  auto read = [&](std::string const& body) {
    {
      std::ofstream out(deck);
      out << "dynamics:\n" << body;
    }
    return HydroOptionsImpl::from_yaml(deck.string());
  };
  EXPECT_EQ(
      read("  conserve-axial-momentum: exact\n")->conserve_axial_momentum(),
      "exact");
  try {
    read("  conserve-axial-momentum: full\n");
    ADD_FAILURE() << "full was read";
  } catch (std::exception const& e) {
    EXPECT_NE(std::string(e.what()).find("exact replaces full"),
              std::string::npos)
        << e.what();
  }
  try {
    read(
        "  conserve-axial-momentum: exact\n"
        "  conserve-axial-momentum-closure: true\n");
    ADD_FAILURE() << "the closure key was read";
  } catch (std::exception const& e) {
    EXPECT_NE(std::string(e.what()).find("conserve-axial-momentum-closure"),
              std::string::npos)
        << e.what();
  }
}

// Output checks keep their exception type and messages.
TEST(options_table, output_checks) {
  auto node = YAML::Load("{type: netcdf, x2_slice: [1.], output_sumx2: true}");
  try {
    OutputOptionsImpl::from_yaml(node, 3);
    FAIL() << "slice and sum accepted";
  } catch (std::invalid_argument const& e) {
    EXPECT_STREQ(e.what(), "Cannot request both slice and sum along x2");
  }
  try {
    OutputOptionsImpl::from_yaml(YAML::Load("{dt: 1.}"), 0);
    FAIL() << "no type accepted";
  } catch (std::invalid_argument const& e) {
    EXPECT_STREQ(e.what(),
                 "OutputOptions::from_yaml: output file type must be "
                 "specified");
  }
  EXPECT_THROW(
      OutputOptionsImpl::from_yaml(YAML::Load("{type: netcdf, dtt: 1}"), 2),
      c10::Error);
}

// Lane S's three parsers against the generated rows, deck by deck.
TEST(options_table, lane_s_rows_match_parsers) {
  int ndeck = 0, ngeom = 0, nlayout = 0, ninternal = 0;
  for (auto const& p : in_tree_decks()) {
    YAML::Node config;
    if (!load(p, &config)) continue;
    ++ndeck;
    SCOPED_TRACE(p.filename().string());

    // geometry: the old parser also builds the grid; compare when it loads
    CoordinateOptions co;
    try {
      co = CoordinateOptionsImpl::from_yaml(p.string());
    } catch (std::exception const&) {
      co = nullptr;
    }
    if (co) {
      auto g = GeometryRows::read(config);
      EXPECT_EQ(g.present(), static_cast<bool>(config["geometry"]));
      if (g.present()) {
        ++ngeom;
        EXPECT_EQ(g.type(), co->type());
        bool cells = static_cast<bool>(config["geometry"]["cells"]);
        double x1lo = g.x1f() ? g.x1f()->front() : g.x1min();
        double x1hi = g.x1f() ? g.x1f()->back() : g.x1max();
        EXPECT_EQ(x1lo, co->global_x1min());
        EXPECT_EQ(x1hi, co->global_x1max());
        EXPECT_EQ(g.x2min(), co->global_x2min());
        EXPECT_EQ(g.x2max(), co->global_x2max());
        EXPECT_EQ(g.x3min(), co->global_x3min());
        EXPECT_EQ(g.x3max(), co->global_x3max());
        EXPECT_EQ(g.nx1(), co->global_nx1());
        EXPECT_EQ(g.nx2(), co->global_nx2());
        EXPECT_EQ(g.nx3(), co->global_nx3());
        if (cells) {
          EXPECT_EQ(g.nghost(), co->nghost());
          EXPECT_EQ(g.interp_order(), co->interp_order());
          bool stretched =
              g.x1f().has_value() || (g.x1rat() && *g.x1rat() != 1.);
          EXPECT_EQ(stretched, co->stretched_x1());
          if (g.x1f()) EXPECT_EQ(*g.x1f(), co->x1f_interior());
        }
      }
    }

    auto lo = LayoutOptionsImpl::from_yaml(p.string());
    auto d = DistributeRows::read(config);
    EXPECT_EQ(d.present(), static_cast<bool>(config["distribute"]));
    if (d.present()) {
      ++nlayout;
      EXPECT_EQ(d.type(), lo->type());
      EXPECT_EQ(d.px(), lo->px());
      EXPECT_EQ(d.py(), lo->py());
      EXPECT_EQ(d.pz(), lo->pz());
      EXPECT_EQ(d.verbose(), lo->verbose());
      EXPECT_EQ(d.backend(), lo->backend());
      EXPECT_EQ(d.device(), lo->device());
    }

    auto io = InternalBoundaryOptionsImpl::from_yaml(p.string());
    auto r = InternalBoundaryRows::read(config);
    if (r.present()) ++ninternal;
    EXPECT_EQ(r.max_iter(), io->max_iter());
    EXPECT_EQ(r.solid_density(), io->solid_density());
    EXPECT_EQ(r.solid_pressure(), io->solid_pressure());
  }
  std::cout << "decks " << ndeck << ": geometry " << ngeom << ", distribute "
            << nlayout << ", boundary-condition/internal " << ninternal
            << std::endl;
  EXPECT_GT(ngeom, 50);
  EXPECT_GT(nlayout, 20);
  EXPECT_GT(ninternal, 0);
}

// The cross-option checks of the geometry table refuse what lane S refuses.
TEST(options_table, geometry_checks) {
  auto refused = [](char const* deck) {
    try {
      GeometryRows::read(YAML::Load(deck));
    } catch (c10::Error const&) {
      return true;
    }
    return false;
  };
  EXPECT_TRUE(
      refused("geometry: {cells: {nx1: 2, x1rat: 1.1}, x1f: [0, 1, 2]}"));
  EXPECT_TRUE(refused("geometry: {x1f: [0, 1, 2]}"));
  EXPECT_TRUE(refused("geometry: {cells: {nx1: 2, x1rat: -1}}"));
  EXPECT_TRUE(refused("geometry: {cells: {nx1: 3}, x1f: [0, 1, 2]}"));
  EXPECT_TRUE(refused(
      "geometry: {bounds: {x1min: 1}, cells: {nx1: 2}, x1f: [0, 1, 2]}"));
  EXPECT_TRUE(refused(
      "geometry: {bounds: {x1max: 1}, cells: {nx1: 2}, x1f: [0, 1, 2]}"));
  EXPECT_TRUE(refused("geometry: {bounds: {x1mid: 1}}"));
  EXPECT_FALSE(refused("geometry: {cells: {nx1: 2}, x1f: [0, 1, 2]}"));
  auto g = GeometryRows::read(
      YAML::Load("geometry: {bounds: {x2max_pi: 0.5, x2max: 7}}"));
  EXPECT_EQ(g.x2max(), 0.5 * M_PI);
}

// The distribute backend row follows default_backend() (ucx on Linux with
// UCX, else gloo), as LayoutOptionsImpl does; BACKEND overrides both.
TEST(options_table, distribute_backend_default) {
  char const* saved = std::getenv("BACKEND");
  std::string keep = saved ? saved : "";
  auto deck = YAML::Load("distribute: {layout: slab}");
  ::unsetenv("BACKEND");
  EXPECT_EQ(DistributeRows::read(deck).backend(), default_backend());
  EXPECT_EQ(DistributeRows().backend(), default_backend());
  ::setenv("BACKEND", "gloo", 1);
  EXPECT_EQ(DistributeRows::read(deck).backend(), "gloo");
  if (saved)
    ::setenv("BACKEND", keep.c_str(), 1);
  else
    ::unsetenv("BACKEND");
}
