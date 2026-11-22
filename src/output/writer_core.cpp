// C/C++ headers
#include <glob.h>

#include <cctype>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_utils.hpp"
#include "writer_core.hpp"

namespace snap {

namespace {
// the non-empty tokens of s between any of the characters in delim (strtok)
std::vector<std::string> split(std::string const& s, char const* delim) {
  std::vector<std::string> out;
  for (size_t b = 0, e; b < s.size(); b = e + 1) {
    e = s.find_first_of(delim, b);
    if (e == std::string::npos) e = s.size();
    if (e > b) out.push_back(s.substr(b, e - b));
  }
  return out;
}

// the tensor dimension of physical axis dim (1-3) in a field of grid kind
// `grid`; -1 when the field has no such axis (reduced grids)
int axis_of(std::string const& grid, int dim) {
  if (grid.size() != 3 || grid[0] != '-') return 4 - dim;
  if (grid == "---") return -1;
  if (grid[1] == '-') return dim == 1 ? 3 : -1;
  return dim == 1 ? -1 : 5 - dim;
}
}  // namespace

std::string Field::grid() const {
  auto first = split(name, ";");
  return MetadataTable::GetInstance()->GetGridType(first.empty() ? ""
                                                                 : first[0]);
}

int Field::nvar() const {
  auto g = grid();
  if (g == "--C" || g == "--F") return data.size(2);
  if (g == "-CC" || g == "-CF" || g == "-FC" || g == "-FF") return data.size(1);
  return g == "---" ? data.size(3) : data.size(0);
}

std::vector<std::string> Field::names() const {
  auto names = split(name, ";");
  int nvar = this->nvar();
  if (names.size() >= nvar) return {names.begin(), names.begin() + nvar};
  // one name for all components: '?' (or the end) takes the number
  names.clear();
  auto pos = std::min(name.find('?'), name.size());
  auto tail = pos < name.size() ? name.substr(pos + 1) : "";
  for (int n = 0; n < nvar; ++n)
    names.push_back(name.substr(0, pos) +
                    (nvar == 1 ? "" : std::to_string(n + 1)) + tail);
  return names;
}

void Frame::to_host() {
  for (auto& f : fields) {
    auto t = f.data.to(torch::kCPU, torch::kFloat64, false, true).contiguous();
    std::vector<int64_t> d4(4, 1);  // [nx4, nx3, nx2, nx1], left-padded
    for (int i = 0; i < t.dim(); ++i) d4[4 - t.dim() + i] = t.size(i);
    f.data = t.view(d4);
  }
}

bool Frame::transform(MeshBlockImpl const* pmb, OutputOptionsImpl const& op) {
  auto pc = pmb->pcoord;
  auto co = pc->options;
  int* lo[3] = {&is, &js, &ks};
  int* hi[3] = {&ie, &je, &ke};
  for (int dim = 3; dim >= 1; --dim) {
    auto const& want = dim == 1   ? op.x1_slice()
                       : dim == 2 ? op.x2_slice()
                                  : op.x3_slice();
    if (want.empty()) continue;
    double lower = dim == 1   ? co->x1min()
                   : dim == 2 ? co->x2min()
                              : co->x3min();
    double upper = dim == 1   ? co->x1max()
                   : dim == 2 ? co->x2max()
                              : co->x3max();
    int begin = dim == 1 ? pc->il() : dim == 2 ? pc->jl() : pc->kl();
    int end = dim == 1 ? pc->iu() : dim == 2 ? pc->ju() : pc->ku();
    auto faces = (dim == 1   ? pc->x1f
                  : dim == 2 ? pc->x2f
                             : pc->x3f)
                     .to(torch::kCPU, torch::kFloat64)
                     .contiguous();
    auto const* xf = faces.data_ptr<double>();
    auto& sel = slice[dim - 1];
    sel.clear();
    slice_offset[dim - 1] = 0;
    for (double x : want) {
      if (x < lower) ++slice_offset[dim - 1];
      if (x < lower || x >= upper) continue;
      for (int i = begin; i <= end; ++i)
        if (xf[i + 1] > x) {
          sel.push_back(i);
          break;
        }
    }
    if (sel.empty()) return false;
    int count = sel.size();
    for (auto& f : fields) {
      int ax = axis_of(f.grid(), dim);
      if (ax < 0) continue;
      // a single cell along the axis is repeated for every selected plane
      auto idx =
          f.data.size(ax) > 1
              ? torch::tensor(std::vector<int64_t>(sel.begin(), sel.end()))
              : torch::zeros({count}, torch::kInt64);
      f.data = f.data.index_select(ax, idx);
    }
    *lo[dim - 1] = 0;
    *hi[dim - 1] = count - 1;
  }
  bool sum[3] = {op.output_sumx1(), op.output_sumx2(), op.output_sumx3()};
  for (int dim = 3; dim >= 1; --dim) {
    if (!sum[dim - 1]) continue;
    for (auto& f : fields) {
      int ax = axis_of(f.grid(), dim);
      if (ax < 0) continue;
      // in index order from zero, as the cell loop did
      auto acc = torch::zeros_like(f.data.narrow(ax, 0, 1));
      for (int k = *lo[dim - 1]; k <= *hi[dim - 1]; ++k)
        acc += f.data.narrow(ax, k, 1);
      f.data = acc;
    }
    *lo[dim - 1] = *hi[dim - 1] = 0;
  }
  return true;
}

char Frame::kind(Field const& f) const {
  auto g = f.grid();
  // a face grid writes its faces only along an axis with cells or a slice
  if (g == "CCF" && (ie > is || !slice[0].empty())) return '1';
  if (g == "CFC" && (je > js || !slice[1].empty())) return '2';
  if (g == "FCC" && (ke > ks || !slice[2].empty())) return '3';
  if (g == "--C" || g == "-CC" || g == "--F" || g == "---")
    return g == "--C" ? 'z' : g == "-CC" ? 'p' : g == "--F" ? 'f' : 's';
  return 'c';
}

torch::Tensor Frame::pack(Field const& f, int n) const {
  using torch::indexing::Slice;
  auto const& d = f.data;
  // without a slice, a face axis holds one more point than its cells
  int e1 = slice[0].empty(), e2 = slice[1].empty(), e3 = slice[2].empty();
  switch (char k = kind(f)) {
    case 'z':
    case 'f':
      return d.reshape({-1, d.size(3)})[n].slice(0, is,
                                                 ie + 1 + (k == 'f') * e1);
    case 'p':
      return d.reshape({-1, d.size(2), d.size(3)})[n].index(
          {Slice(ks, ke + 1), Slice(js, je + 1)});
    case 's':
      return d.reshape({-1})[n].reshape({1});
    default:  // cells or faces: i outermost, then k, then j
      return d[n]
          .index({Slice(ks, ke + 1 + (k == '3') * e3),
                  Slice(js, je + 1 + (k == '2') * e2),
                  Slice(is, ie + 1 + (k == '1') * e1)})
          .permute({2, 0, 1})
          .contiguous();
  }
}

std::string sanitize_netcdf_name(std::string name) {
  for (char& c : name)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) c = '_';
  if (name.empty()) return "_";
  if (!(std::isalpha(static_cast<unsigned char>(name.front())) ||
        name.front() == '_'))
    name.insert(name.begin(), '_');
  return name;
}

GriddedWriter::GriddedWriter(OutputOptions const& options_,
                             std::shared_ptr<Sink> sink)
    : OutputType(options_), sink_(std::move(sink)) {
  if (!sink_)
    sink_ = options->file_type() == "pnetcdf" ? make_pnetcdf_sink(*options)
                                              : make_netcdf_sink();
}

void GriddedWriter::write_output_file(MeshBlockImpl* pmb, Variables const& vars,
                                      double time, bool final_write) {
  if (final_write) return;
  auto fr = LoadOutputData(pmb, vars);
  std::tie(fr->is, fr->ie, fr->js, fr->je, fr->ks, fr->ke) =
      output_range(pmb, options->include_ghost_zones());
  fr->to_host();
  if (fr->transform(pmb, *options))
    sink_->write(pmb, *fr, *options, file_number, time);
  frame_.reset();
  if (options->combine()) combine_blocks(pmb, final_write);
}

void GriddedWriter::combine_blocks(MeshBlockImpl* pmb, bool) {
  sink_->combine(pmb, *options, file_number);
}

std::tuple<int, int, int, int, int, int> output_range(
    MeshBlockImpl const* pmb, bool include_ghost_zones) {
  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nghost = pmb->options->coord()->nghost();

  // set start/end array indices depending on whether ghost zones are included
  int is = nc1 > 1 ? nghost : 0;
  int ie = nc1 > 1 ? nc1 - nghost - 1 : 0;
  int js = nc2 > 1 ? nghost : 0;
  int je = nc2 > 1 ? nc2 - nghost - 1 : 0;
  int ks = nc3 > 1 ? nghost : 0;
  int ke = nc3 > 1 ? nc3 - nghost - 1 : 0;

  // FIXME: include_ghost zones probably doesn't work with grids other than
  // CCC
  if (include_ghost_zones) {
    if (is != ie) {
      is -= nghost;
      ie += nghost;
    }
    if (js != je) {
      js -= nghost;
      je += nghost;
    }
    if (ks != ke) {
      ks -= nghost;
      ke += nghost;
    }
  }
  return {is, ie, js, je, ks, ke};
}

void make_output_dir(MeshBlockImpl const* pmb) {
  std::error_code ec;
  std::filesystem::create_directories(pmb->options->output_dir(), ec);
  if (ec) {
    throw std::runtime_error("Failed to create output directory '" +
                             pmb->options->output_dir() + "': " + ec.message());
  }
}

OutputBlockLocation output_block_location(Layout const& layout, int rank) {
  OutputBlockLocation loc;
  loc.level = 0;
  std::tie(loc.lx2, loc.lx3, loc.lx1) = layout->loc_of(rank);
  loc.nb1 = layout->options->nb(1);
  loc.nb2 = layout->options->nb(2);
  loc.nb3 = layout->options->nb(3);

  loc.face = 0;
  if (layout->options->type() == "cubed-sphere") {
    loc.face = loc.lx1;
    loc.lx1 = 0;
    loc.lx2 += (loc.face % 3) * loc.nb2;
    loc.lx3 += (loc.face / 3) * loc.nb3;
    loc.nb2 *= 3;
    loc.nb3 *= 2;
  }
  return loc;
}

namespace {
std::mutex combine_mutex;
std::map<std::string, int> combine_counts;

bool ready_to_combine(Layout const& layout, std::string const& key) {
  std::lock_guard<std::mutex> lock(combine_mutex);
  int& count = combine_counts[key];
  count += 1;
  if (count < layout->options->blocks_per_process()) {
    return false;
  }
  combine_counts.erase(key);
  return true;
}
}  // namespace

void combine_on_root(Layout const& layout, std::string const& key,
                     std::function<void()> const& root_work) {
  if (!ready_to_combine(layout, key)) {
    return;
  }

  if (layout->has_process_group()) {
    layout->comm->barrier();
  }

  // Mirror of the barrier above, in a scope guard: no rank may return before
  // the combined file exists, and the root's throw below must still release the
  // others instead of leaving them in a collective it never reaches.
  struct ReleaseOnExit {
    Layout mirror;
    ~ReleaseOnExit() {
      if (mirror->has_process_group()) mirror->comm->barrier();
    }
  } release{layout};

  if (layout->options->process_rank() == layout->options->process_root_rank()) {
    root_work();
  }
}

std::vector<std::string> glob_block_files(std::string const& pattern,
                                          char const* caller) {
  glob_t glob_result;
  int err = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (err != 0) {
    globfree(&glob_result);
    std::stringstream msg;
    msg << "### FATAL ERROR in function [" << caller << "]" << std::endl
        << "glob() failed with error " << err << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  std::vector<std::string> files(glob_result.gl_pathv,
                                 glob_result.gl_pathv + glob_result.gl_pathc);
  globfree(&glob_result);
  return files;
}

}  // namespace snap
