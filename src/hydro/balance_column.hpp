#pragma once

// C/C++
#include <string>
#include <tuple>

// torch
#include <torch/torch.h>

namespace snap {

class EquationOfStateImpl;

//! Project a ghost-free x1 column onto the well-balanced scheme's discrete
//! hydrostatic balance at fixed temperature. Returns (w, residual, sweeps)
//! and throws if it does not converge. Parameters, the caller's four
//! obligations and the free gauge: docs/derivations/lane-c-notes.md.
std::tuple<torch::Tensor, double, int> balance_column(
    torch::Tensor const& w, torch::Tensor const& dx1f, double grav,
    bool wall_clamp = true, double rtol = 1.e-10, int max_iter = 120,
    std::string const& geometry = "", bool stretched = false,
    int conversion_order = 2, EquationOfStateImpl* eos = nullptr);

}  // namespace snap
