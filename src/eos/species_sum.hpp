#pragma once

// torch
#include <torch/torch.h>

namespace snap {

// Internal to the EOS sources: sums over the leading (species) axis that do
// not depend on where the element sits in memory.

// sum_k y[k] * c[k] over the leading (species) axis, accumulated one species
// at a time with elementwise ops. A sum(0) or matmul over that axis of a
// block's strided (nvar, nc3, nc2, nc1) view rounds by memory position, so
// identical states in different columns could come out 1 ulp apart.
inline torch::Tensor species_dot(torch::Tensor const& y,
                                 torch::Tensor const& c) {
  auto acc = y[0] * c[0];
  for (int64_t k = 1; k < y.size(0); ++k) acc = acc + y[k] * c[k];
  return acc;
}

// sum_k y[k] over the leading (species) axis, accumulated the same way
inline torch::Tensor species_total(torch::Tensor const& y) {
  auto acc = y[0];
  for (int64_t k = 1; k < y.size(0); ++k) acc = acc + y[k];
  return acc;
}

// 1 - sum_k y[k] over the leading (species) axis, subtracted one species at a
// time like the kernels' dry_fraction (ideal_moist_impl.h); `like` sets the
// shape and dtype of the 1. Not 1 - species_total(y): that associates
// differently.
inline torch::Tensor species_complement(torch::Tensor const& like,
                                        torch::Tensor const& y) {
  auto rd = torch::ones_like(like);
  for (int64_t k = 0; k < y.size(0); ++k) rd = rd - y[k];
  return rd;
}

}  // namespace snap
