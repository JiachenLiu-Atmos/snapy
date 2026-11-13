#pragma once

// C/C++
#include <cstdint>
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

//! the per-grid x1 stencil caches (x1_centroid.cpp, wb_ref4.cpp): rows of k
//! cell or face indices and weights, built on the CPU, applied on any device

//! (rows, k) int64 indices, a short row padded with 0
inline torch::Tensor x1_stencil_index(
    std::vector<std::vector<int64_t>> const& rows, int k) {
  auto t = torch::zeros({(int64_t)rows.size(), k}, torch::kInt64);
  auto a = t.accessor<int64_t, 2>();
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < rows[i].size(); ++j) a[i][j] = rows[i][j];
  return t;
}

//! (rows, k) float64 weights, a short row padded with 0
inline torch::Tensor x1_stencil_weight(
    std::vector<std::vector<double>> const& rows, int k) {
  auto t = torch::zeros({(int64_t)rows.size(), k}, torch::kFloat64);
  auto a = t.accessor<double, 2>();
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < rows[i].size(); ++j) a[i][j] = rows[i][j];
  return t;
}

//! sum_k v[..., idx[n, k]] * wt[n, k] over k: (..., n)
inline torch::Tensor x1_stencil_apply(torch::Tensor const& v,
                                      torch::Tensor const& idx,
                                      torch::Tensor const& wt) {
  auto sizes = v.sizes().vec();
  sizes.back() = idx.size(0);
  sizes.push_back(idx.size(1));
  return (v.index_select(-1, idx.flatten()).view(sizes) * wt).sum(-1);
}

}  // namespace snap
