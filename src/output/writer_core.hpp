#pragma once

// C/C++
#include <functional>
#include <string>
#include <tuple>
#include <vector>

// snap
#include <snap/layout/layout.hpp>

namespace snap {

class MeshBlockImpl;

//! interior index range [is, ie] x [js, je] x [ks, ke] of a block's output,
//! widened by the ghost zones in every dimension with more than one cell when
//! include_ghost_zones is set (only checked for cell-centred grids)
std::tuple<int, int, int, int, int, int> output_range(MeshBlockImpl const* pmb,
                                                      bool include_ghost_zones);

//! create the block's output directory; throws with the filesystem error
void make_output_dir(MeshBlockImpl const* pmb);

//! where a block sits in the combined output grid
struct OutputBlockLocation {
  int level, lx1, lx2, lx3;  //!< refinement level, block indices
  int nb1, nb2, nb3;         //!< blocks per direction
  int face;                  //!< cubed-sphere panel (0 otherwise)
};

//! the location of `rank`; cubed-sphere panels are laid out 3 x 2 in (x2, x3)
OutputBlockLocation output_block_location(Layout const& layout, int rank);

//! the per-process combine protocol shared by the block writers: every block
//! of a process counts in under `key` and only the last one combines. All
//! ranks of the process group meet before root_work runs on the root rank and
//! again after it, also when it throws, so no rank returns before the combined
//! file exists.
void combine_on_root(Layout const& layout, std::string const& key,
                     std::function<void()> const& root_work);

//! the files matching `pattern` (glob, GLOB_TILDE); throws, naming `caller`,
//! if glob() fails
std::vector<std::string> glob_block_files(std::string const& pattern,
                                          char const* caller);

}  // namespace snap
