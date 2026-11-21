// torch
#include <torch/torch.h>

// snap
#include "mesh_functions.hpp"

namespace snap {
//! \brief wrapper fn to compute float logical position for either
//!        [0., 1.] or [-0.5, 0.5]\n
//!        real cell ranges for meshgen functions (default/user vs. uniform)
torch::Tensor compute_logical_position(torch::Tensor index, int64_t nrange,
                                       bool sym_interval) {
  // index is typically 0, ... nrange for non-ghost boundaries
  if (!sym_interval) {
    // to map to fractional logical position [0.0, 1.0], simply divide by # of
    // faces
    return 1. * index / nrange;
  } else {
    // to map to a [-0.5, 0.5] range, rescale int indices around 0 before FP
    // conversion if nrange is even, there is an index at center x=0.0; map it
    // to (int) 0 if nrange is odd, the center x=0.0 is between two indices; map
    // them to -1, 1
    auto noffset = index - (nrange) / 2;
    auto noffset_ceil =
        index - (nrange + 1) / 2;  // = noffset if nrange is even
    //  average the (possibly) biased integer indexing
    return (noffset + noffset_ceil) / (2.0 * nrange);
  }
}
}  // namespace snap
