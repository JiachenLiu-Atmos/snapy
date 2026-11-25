#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {

//! the tendency over dt of a diffusion on block's hydro as a stage takes it
//! (A.rst 3.7): add() gives every row but energy, whose only path is the
//! register (post_fluxes, then -dt times the divergence of the totals)
inline torch::Tensor diffuse(DiffusionImpl& diffusion, MeshBlockImpl& block,
                             torch::Tensor du, torch::Tensor w,
                             torch::Tensor temp, double dt) {
  auto& reg = block.phydro->flux_register();
  for (int d = 1; d <= 3; ++d)
    if (reg.total(d).defined()) reg.total(d).zero_();
  diffusion.post_fluxes(reg, {w, temp, dt});
  diffusion.forward(du, w, temp, dt);
  auto div = block.pcoord->divergence(reg.total(1), reg.total(2), reg.total(3));
  du[IPR] -= dt * div[IPR].to(du.scalar_type());
  return du;
}

//! the block's own diffusion
inline torch::Tensor diffuse(MeshBlockImpl& block, torch::Tensor du,
                             torch::Tensor w, torch::Tensor temp, double dt) {
  return diffuse(*block.phydro->pdiffusion, block, du, w, temp, dt);
}

}  // namespace snap
