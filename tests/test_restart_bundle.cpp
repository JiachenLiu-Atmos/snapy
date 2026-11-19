// Gate G-4 (O.rst section 5), restart after the merge of writer, bundle and
// reader (O7): a 4-block bundle reads back each block's variables, a broken
// bundle throws (it used to print and return an empty map), and the #367
// metadata checks still refuse a mismatch.

// C/C++
#include <filesystem>
#include <fstream>

// external
#include <gtest/gtest.h>

// base
#include <configure.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/output/output_type.hpp>
#include <snap/output/restart.hpp>

using namespace snap;
namespace fs = std::filesystem;

namespace {

struct Bundle {
  fs::path file;
  std::vector<Variables> blocks;  //!< what each block wrote, on the CPU
};

// a 4-block mesh writes one restart bundle
Bundle write_bundle(std::string const& tag) {
  auto dir = fs::temp_directory_path() / ("snapy_g4_" + tag);
  if (fs::exists(dir)) {
    for (auto const& e : fs::directory_iterator(dir)) fs::remove(e.path());
  }
  auto bo = MeshBlockOptionsImpl::from_yaml(std::string(SNAPY_SOURCE_DIR) +
                                            "/examples/shallow_xy_mesh4.yaml");
  auto rst = OutputOptionsImpl::create();
  rst->file_type("restart");
  bo->outputs({rst});
  bo->output_dir(dir.string()).basename("g4");
  auto mo = MeshOptionsImpl::create();
  mo->block(bo);
  mo->blocks_per_process(4);
  auto mesh = Mesh(mo);
  torch::manual_seed(11);
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto co = mesh->blocks[i]->pcoord->options;
    int nvar = mesh->blocks[i]->phydro->peos->nvar();
    vars[i]["hydro_w"] =
        1. +
        torch::rand({nvar, co->nc3(), co->nc2(), co->nc1()}, torch::kFloat64);
  }
  mesh->initialize(vars);
  mesh->make_outputs(vars, 0.);
  Bundle b{dir / "g4.00000.restart", {}};
  for (auto& v : vars) b.blocks.push_back(v);
  return b;
}

// the bundle cut to `keep` bytes
fs::path cut(fs::path const& file, std::uintmax_t keep) {
  auto out = file.parent_path() / "cut.restart";
  std::ifstream in(file, std::ios::binary);
  std::vector<char> bytes(keep);
  in.read(bytes.data(), keep);
  std::ofstream(out, std::ios::binary).write(bytes.data(), keep);
  return out;
}

}  // namespace

TEST(RestartBundle, four_blocks_read_back_their_own_variables) {
  auto b = write_bundle("roundtrip");
  ASSERT_TRUE(fs::exists(b.file));
  for (int r = 0; r < 4; ++r) {
    auto data = load_restart(b.file.string(), r);
    for (auto key : {"hydro_u", "hydro_w"}) {
      ASSERT_TRUE(data.count(key)) << key << " in block " << r;
      EXPECT_TRUE(torch::equal(data.at(key), b.blocks[r].at(key).cpu()))
          << key << " of block " << r;
    }
  }
}

// RED before O7: the reader printed to std::cerr and returned an empty map
TEST(RestartBundle, a_truncated_bundle_throws) {
  auto b = write_bundle("truncated");
  auto size = fs::file_size(b.file);
  auto file = cut(b.file, size - 100);
  EXPECT_THROW(load_restart(file.string(), 3), std::runtime_error);
  file = cut(b.file, 40);  // inside the index
  EXPECT_THROW(load_restart(file.string(), 0), std::runtime_error);
}

TEST(RestartBundle, a_block_missing_from_the_bundle_throws) {
  auto b = write_bundle("missing");
  EXPECT_THROW(load_restart(b.file.string(), 7), std::runtime_error);
}

TEST(RestartBundle, metadata_mismatches_are_still_refused) {
  auto x = torch::linspace(0., 1., 5, torch::kFloat64);
  auto u = torch::zeros({5, 1, 1, 4}, torch::kFloat64);
  Variables data = {{"hydro_u", u},
                    {"hydro_u_stored", u.to(torch::kFloat32)},
                    {"hydro_ref", u},
                    {kHydroRestartMetadata,
                     torch::tensor({int64_t(kHydroRestartVersion), int64_t(32)},
                                   torch::kInt64)},
                    {kHydroReferenceGrid[0], x},
                    {kHydroReferenceGrid[1], x},
                    {kHydroReferenceGrid[2], x}};
  std::array<torch::Tensor, 3> grid = {x, x, x};
  EXPECT_NO_THROW(check_restart_metadata(data, grid, false));
  // another grid of the same shape
  EXPECT_THROW(check_restart_metadata(data, {x * 2., x, x}, false), c10::Error);
  // stored bits that disagree with the metadata
  auto wrong = data;
  wrong["hydro_u_stored"] = u;
  EXPECT_THROW(check_restart_metadata(wrong, grid, false), c10::Error);
  // perturbation-flux needs the species reference
  EXPECT_THROW(check_restart_metadata(data, grid, true), c10::Error);
}
