#pragma once

// C/C+
#include <functional>

// torch
#include <torch/torch.h>

namespace snap {
//! Compute the logical position of a cell given its index and the number of
//! cells in the range.
torch::Tensor compute_logical_position(torch::Tensor index, int64_t nrange,
                                       bool sym_interval);

using MeshGenerator = std::function<torch::Tensor(torch::Tensor, float, float)>;
}  // namespace snap
