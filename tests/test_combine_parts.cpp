// combine_parts on the cases of gate G-2 (O.rst section 5). G-2 compared
// combine_parts with mppnccombine (O6a) before O6b deleted mppnccombine; each
// case now writes its block files with `combine: false`, merges them with
// combine_parts and requires every part's values, byte for byte, at the part's
// offset in the merged file (domain_decomposition), and the parts removed.

// C/C++
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>

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

using namespace snap;
namespace fs = std::filesystem;

namespace {

// one part's variables: name, values as bytes, and where they sit in the
// merged file (a coordinate's domain_decomposition {first, last, start, end},
// 1-based, splits its dimension)
struct PartVar {
  std::string name;
  std::vector<size_t> start, count;
  std::vector<char> bytes;
};

std::vector<PartVar> read_part(std::string const& path) {
  int id, ndims, nvars;
  EXPECT_EQ(nc_open(path.c_str(), NC_NOWRITE, &id), NC_NOERR) << path;
  nc_inq(id, &ndims, &nvars, nullptr, nullptr);
  std::vector<size_t> offset(ndims, 0);
  char name[NC_MAX_NAME + 1];
  for (int v = 0, d, dd[4]; v < nvars; ++v) {
    nc_inq_varname(id, v, name);
    if (nc_inq_dimid(id, name, &d) == NC_NOERR &&
        nc_get_att_int(id, v, "domain_decomposition", dd) == NC_NOERR)
      offset[d] = dd[2] - dd[0];
  }
  std::vector<PartVar> out;
  for (int v = 0; v < nvars; ++v) {
    nc_type t;
    int nd, dims[NC_MAX_VAR_DIMS];
    size_t size, n = 1;
    PartVar pv;
    nc_inq_var(id, v, name, &t, &nd, dims, nullptr);
    nc_inq_type(id, t, nullptr, &size);
    pv.name = name;
    for (int i = 0; i < nd; ++i) {
      size_t len;
      nc_inq_dimlen(id, dims[i], &len);
      pv.start.push_back(offset[dims[i]]);
      pv.count.push_back(len);
      n *= len;
    }
    pv.bytes.resize(n * size);
    nc_get_var(id, v, pv.bytes.data());
    out.push_back(std::move(pv));
  }
  nc_close(id);
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
  return dir;
}

void expect_merged(Case const& c, std::string const& tag) {
  auto dir = write_parts(c, tag);
  auto parts = parts_in(dir / "a");
  ASSERT_GT(parts.size(), 1u) << tag << ": nothing to combine";
  std::vector<std::vector<PartVar>> in;
  for (auto const& p : parts) in.push_back(read_part(p));
  auto out = (dir / "parts.nc").string();
  ASSERT_EQ(combine_parts(out, parts), NC_NOERR) << tag;
  EXPECT_TRUE(parts_in(dir / "a").empty()) << tag << ": -r leaves no parts";

  // ghost zones overlap: an element holds the last part that covers it, so
  // walk the parts backwards and check each element once
  int id;
  ASSERT_EQ(nc_open(out.c_str(), NC_NOWRITE, &id), NC_NOERR);
  std::map<std::string, std::vector<bool>> seen;
  for (size_t f = in.size(); f-- > 0;)
    for (auto const& pv : in[f]) {
      int v, nd, dims[NC_MAX_VAR_DIMS];
      ASSERT_EQ(nc_inq_varid(id, pv.name.c_str(), &v), NC_NOERR) << pv.name;
      nc_inq_var(id, v, nullptr, nullptr, &nd, dims, nullptr);
      std::vector<size_t> whole(nd);
      size_t n = 1, nwhole = 1, esize = pv.bytes.size();
      for (int i = 0; i < nd; ++i) {
        nc_inq_dimlen(id, dims[i], &whole[i]);
        n *= pv.count[i];
        nwhole *= whole[i];
      }
      esize /= n;
      std::vector<char> got(pv.bytes.size());
      ASSERT_EQ(
          nc_get_vara(id, v, pv.start.data(), pv.count.data(), got.data()),
          NC_NOERR)
          << tag << ": " << pv.name;
      auto& mark = seen[pv.name];
      mark.resize(nwhole, false);
      size_t bad = 0;
      for (size_t k = 0; k < n; ++k) {
        size_t rem = k, at = 0, stride = 1;
        for (int i = nd; i-- > 0;) {
          at += (rem % pv.count[i] + pv.start[i]) * stride;
          rem /= pv.count[i];
          stride *= whole[i];
        }
        if (mark[at]) continue;
        mark[at] = true;
        bad += !std::equal(&got[k * esize], &got[(k + 1) * esize],
                           &pv.bytes[k * esize]);
      }
      EXPECT_EQ(bad, 0u) << tag << ": part " << f << " variable " << pv.name;
    }
  nc_close(id);
  if (!std::getenv("SNAPY_G2_KEEP")) clear(dir);  // keep for ncdump/cmp
}

std::string example(char const* name) {
  return std::string(SNAPY_SOURCE_DIR) + "/examples/" + name;
}

}  // namespace

// one rank, several blocks, Cartesian; and the options of the #350 matrix
TEST(CombineParts, straka_mesh2) {
  expect_merged({example("straka_mesh2.yaml"), 2}, "straka_mesh2");
}
TEST(CombineParts, straka_mesh2_ghost_zones) {
  expect_merged({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.include_ghost_zones(true); }},
                "straka_mesh2_ghost");
}
TEST(CombineParts, straka_mesh2_double_precision) {
  expect_merged({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.double_precision(true); }},
                "straka_mesh2_double");
}
TEST(CombineParts, straka_mesh2_x2_slice) {
  expect_merged({example("straka_mesh2.yaml"), 2,
                 [](OutputOptionsImpl& o) { o.x2_slice({3.e3, 1.2e4, 2.e4}); }},
                "straka_mesh2_x2_slice");
}
TEST(CombineParts, shallow_xy_mesh4) {
  expect_merged({example("shallow_xy_mesh4.yaml"), 4}, "shallow_xy_mesh4");
}
// one rank, cubed sphere: six panels
TEST(CombineParts, shallow_splash_mesh6) {
  expect_merged({example("shallow_splash_mesh6.yaml"), 6},
                "shallow_splash_mesh6");
}
// the several-rank decks' decompositions, every block in this process
TEST(CombineParts, shallow_xy_proc2_mesh2_decomposition) {
  expect_merged({example("shallow_xy_proc2_mesh2.yaml"), 4},
                "shallow_xy_proc2_mesh2");
}
TEST(CombineParts, shallow_splash_proc2_mesh3_decomposition) {
  expect_merged({example("shallow_splash_proc2_mesh3.yaml"), 6},
                "shallow_splash_proc2_mesh3");
}
// test_user_output's uneven slice lists, split over two blocks in x2
TEST(CombineParts, uneven_slice_lists) {
  expect_merged(
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
  expect_merged({example("straka_proc2.yaml"), 2}, "straka_proc2");
}
#endif  // NETCDFOUTPUT
