// C/C++ headers
#include <glob.h>

#include <filesystem>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

// snap
#include <snap/mesh/meshblock.hpp>

#include "writer_core.hpp"

namespace snap {

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
  loc.nb1 = layout->options->pz();
  loc.nb2 = layout->options->px();
  loc.nb3 = layout->options->py();

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
