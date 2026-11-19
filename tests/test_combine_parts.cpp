// Gate G-2 (O.rst section 5): combine_parts against mppnccombine. Each case
// writes its block files with `combine: false`, merges the same parts once
// with mppnccombine -r (as combine_netcdf.cpp calls it) and once with
// combine_parts, and requires every variable's bytes and the whole header
// (format, dimensions, variables, attributes, in order) to be equal.

// C/C++
#include <cstdio>
#include <filesystem>
#include <functional>
#include <sstream>

// external
#include <gtest/gtest.h>

// base
#include <configure.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/output/output_type.hpp>
#include <snap/output/writer_core.hpp>

#ifdef NETCDFOUTPUT
#include <netcdf.h>

int mppnccombine(int argc, char* argv[]);

using namespace snap;
namespace fs = std::filesystem;

namespace {

// format, dimensions, variables with their attributes, global attributes
std::string header(int id) {
  std::ostringstream os;
  int fmt, ndims, nvars, ngatts, unlim;
  nc_inq_format(id, &fmt);
  nc_inq(id, &ndims, &nvars, &ngatts, &unlim);
  os << "format " << fmt << " unlimited " << unlim << "\n";
  char name[NC_MAX_NAME + 1];
  auto atts = [&](int v, int n) {
    for (int a = 0; a < n; ++a) {
      nc_type t;
      size_t len, size;
      nc_inq_attname(id, v, a, name);
      nc_inq_att(id, v, name, &t, &len);
      nc_inq_type(id, t, nullptr, &size);
      std::vector<unsigned char> raw(len * size);
      nc_get_att(id, v, name, raw.data());
      os << "  att " << name << " type " << t << " =";
      for (auto b : raw) os << ' ' << int(b);
      os << "\n";
    }
  };
  for (int d = 0; d < ndims; ++d) {
    size_t len;
    nc_inq_dim(id, d, name, &len);
    os << "dim " << name << " " << len << "\n";
  }
  for (int v = 0; v < nvars; ++v) {
    nc_type t;
    int nd, dims[NC_MAX_VAR_DIMS], natts;
    nc_inq_var(id, v, name, &t, &nd, dims, &natts);
    os << "var " << name << " type " << t << " dims";
    for (int d = 0; d < nd; ++d) os << ' ' << dims[d];
    os << "\n";
    atts(v, natts);
  }
  atts(NC_GLOBAL, ngatts);
  return os.str();
}

// every variable's values as bytes
std::vector<std::vector<char>> values(int id) {
  int nvars;
  nc_inq_nvars(id, &nvars);
  std::vector<std::vector<char>> out;
  for (int v = 0; v < nvars; ++v) {
    nc_type t;
    int nd, dims[NC_MAX_VAR_DIMS];
    size_t size, n = 1, len;
    nc_inq_var(id, v, nullptr, &t, &nd, dims, nullptr);
    nc_inq_type(id, t, nullptr, &size);
    for (int d = 0; d < nd; ++d) nc_inq_dimlen(id, dims[d], &len), n *= len;
    out.emplace_back(n * size);
    nc_get_var(id, v, out.back().data());
  }
  return out;
}

// remove() entry by entry: libtorch's own remove_all can crash on a
// non-empty directory (see test_output_barrier.cpp)
void clear(fs::path const& dir) {
  if (!fs::exists(dir)) return;
  for (auto const& e : fs::directory_iterator(dir))
    e.is_directory() ? clear(e.path()) : (void)fs::remove(e.path());
  fs::remove(dir);
}

std::vector<std::string> parts_in(fs::path const& dir) {
  std::vector<std::string> parts;
  for (auto const& e : fs::directory_iterator(dir))
    if (e.path().filename().string().find(".block") != std::string::npos)
      parts.push_back(e.path().string());
  std::sort(parts.begin(), parts.end());
  return parts;
}

struct Case {
  std::string deck;  //!< under examples/ or tests/
  int blocks;        //!< all blocks in this process
  std::function<void(OutputOptionsImpl&)> tweak = [](OutputOptionsImpl&) {};
  std::function<void(MeshBlockOptionsImpl&)> block = [](MeshBlockOptionsImpl&) {
  };
};

// the block files of one output of the deck, with combine: false
fs::path write_parts(Case const& c, std::string const& tag) {
  auto dir = fs::temp_directory_path() / ("snapy_g2_" + tag);
  clear(dir);
  auto bo = MeshBlockOptionsImpl::from_yaml(c.deck);
  auto nc = OutputOptionsImpl::create();
  nc->file_type("netcdf").variables({"prim"}).combine(false);
  c.tweak(*nc);
  c.block(*bo);
  bo->outputs({nc});
  bo->output_dir((dir / "a").string()).basename("g2");
  auto mo = MeshOptionsImpl::create();
  mo->block(bo);
  mo->blocks_per_process(c.blocks);
  auto mesh = Mesh(mo);
  torch::manual_seed(7);
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto co = mesh->blocks[i]->pcoord->options;
    int nvar = mesh->blocks[i]->phydro->peos->nvar();
    vars[i]["hydro_w"] =
        1. + 0.5 * torch::rand({nvar, co->nc3(), co->nc2(), co->nc1()},
                               torch::kFloat64);
  }
  mesh->initialize(vars);
  mesh->make_outputs(vars, 0.);
  // the same parts for the second merge
  fs::create_directories(dir / "b");
  for (auto const& p : parts_in(dir / "a"))
    fs::copy_file(p, dir / "b" / fs::path(p).filename());
  return dir;
}

void expect_parity(Case const& c, std::string const& tag) {
  auto dir = write_parts(c, tag);
  auto pa = parts_in(dir / "a"), pb = parts_in(dir / "b");
  ASSERT_GT(pa.size(), 1u) << tag << ": nothing to combine";
  ASSERT_EQ(pa.size(), pb.size());
  auto out_a = (dir / "mpp.nc").string(), out_b = (dir / "parts.nc").string();

  std::vector<std::string> args = {"CombineBlocks", "-r", out_a};
  args.insert(args.end(), pa.begin(), pa.end());
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  ASSERT_EQ(mppnccombine(argv.size(), argv.data()), 0) << tag;
  ASSERT_EQ(combine_parts(out_b, pb), NC_NOERR) << tag;
  EXPECT_TRUE(parts_in(dir / "a").empty() && parts_in(dir / "b").empty())
      << tag << ": -r leaves no parts";

  int ia, ib;
  ASSERT_EQ(nc_open(out_a.c_str(), NC_NOWRITE, &ia), NC_NOERR);
  ASSERT_EQ(nc_open(out_b.c_str(), NC_NOWRITE, &ib), NC_NOERR);
  EXPECT_EQ(header(ia), header(ib)) << tag;
  auto va = values(ia), vb = values(ib);
  ASSERT_EQ(va.size(), vb.size()) << tag;
  for (size_t v = 0; v < va.size(); ++v)
    EXPECT_TRUE(va[v] == vb[v]) << tag << ": variable " << v << " differs";
  nc_close(ia);
  nc_close(ib);
  if (!std::getenv("SNAPY_G2_KEEP")) clear(dir);  // keep for ncdump/cmp
}

std::string example(char const* name) {
  return std::string(SNAPY_SOURCE_DIR) + "/examples/" + name;
}

}  // namespace

// one rank, several blocks, Cartesian; and the options of the #350 matrix
TEST(CombineParts, straka_mesh2) {
  expect_parity({example("straka_mesh2.yaml"), 2}, "straka_mesh2");
}
TEST(CombineParts, straka_mesh2_ghost_zones) {
  expect_parity({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.include_ghost_zones(true); }},
                "straka_mesh2_ghost");
}
TEST(CombineParts, straka_mesh2_double_precision) {
  expect_parity({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.double_precision(true); }},
                "straka_mesh2_double");
}
TEST(CombineParts, straka_mesh2_x2_slice) {
  expect_parity({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.x2_slice({3.e3, 1.2e4, 2.e4}); }},
                "straka_mesh2_x2_slice");
}
TEST(CombineParts, shallow_xy_mesh4) {
  expect_parity({example("shallow_xy_mesh4.yaml"), 4}, "shallow_xy_mesh4");
}
// one rank, cubed sphere: six panels
TEST(CombineParts, shallow_splash_mesh6) {
  expect_parity({example("shallow_splash_mesh6.yaml"), 6},
                "shallow_splash_mesh6");
}
// the several-rank decks' decompositions, every block in this process
TEST(CombineParts, shallow_xy_proc2_mesh2_decomposition) {
  expect_parity({example("shallow_xy_proc2_mesh2.yaml"), 4},
                "shallow_xy_proc2_mesh2");
}
TEST(CombineParts, shallow_splash_proc2_mesh3_decomposition) {
  expect_parity({example("shallow_splash_proc2_mesh3.yaml"), 6},
                "shallow_splash_proc2_mesh3");
}
// test_user_output's uneven slice lists, split over two blocks in x2
TEST(CombineParts, uneven_slice_lists) {
  expect_parity(
      {std::string(SNAPY_SOURCE_DIR) + "/tests/test_forcing_3d.yaml", 2,
       [](OutputOptionsImpl& o) {
         o.x1_slice({0., 2., 5.}).x2_slice({1., 2., 3.});
       },
       [](MeshBlockOptionsImpl& b) {
         b.layout()->px(2);
         b.coord()->repartition(b.layout());
       }},
      "uneven_slice_lists");
}
TEST(CombineParts, straka_proc2_decomposition) {
  expect_parity({example("straka_proc2.yaml"), 2}, "straka_proc2");
}
#endif  // NETCDFOUTPUT
