#pragma once

// C/C++
#include <array>
#include <map>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/layout/layout.hpp>

#include "restart_metadata.hpp"

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

//! the restart variables of block `block_rank`, from a bundle or from a
//! single part file; throws if the file cannot be read, the bundle is
//! malformed or truncated, or it holds no part for the block
Variables load_restart(std::string const& path, int block_rank = get_rank());

//! the #367 checks: the hydro metadata, the stored dtype and shape, and the
//! reference grid against this block's faces x1f, x2f, x3f; perturbation-flux
//! needs the original reference
void check_restart_metadata(Variables const& data,
                            std::array<torch::Tensor, 3> const& grid,
                            bool perturbation_flux);

}  // namespace snap
