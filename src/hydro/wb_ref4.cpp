// C/C++
#include <algorithm>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/coord/x1_stencil.hpp>
#include <snap/layout/layout.hpp>

#include "wb_ref4.hpp"

namespace snap {

namespace {

// relative margin of the range guards: on a flat profile the two bounds
// coincide, and an exact test would flip on round-off between a split and a
// one-block column
constexpr double kMargin = 1.e-10;
// |ln(p_lo/p_hi)| above this: fewer than two cells per pressure scale height
constexpr double kMaxLogDrop = 0.5;

// d/dx of the Lagrange basis polynomial j through nodes X, at x
double lagrange_deriv(std::vector<double> const& X, int j, double x) {
  double sum = 0.;
  int n = X.size();
  for (int l = 0; l < n; ++l) {
    if (l == j) continue;
    double prod = 1. / (X[j] - X[l]);
    for (int m = 0; m < n; ++m) {
      if (m == j || m == l) continue;
      prod *= (x - X[m]) / (X[j] - X[m]);
    }
    sum += prod;
  }
  return sum;
}

torch::Tensor in_range(torch::Tensor const& v, torch::Tensor const& lo,
                       torch::Tensor const& hi) {
  return (v >= lo - kMargin * lo.abs()) & (v <= hi + kMargin * hi.abs());
}

}  // namespace

WbRef4Stencils wb_ref4_stencils(torch::Tensor const& x1f, int is, int iu,
                                bool clamp_in, bool clamp_out,
                                torch::TensorOptions const& options) {
  WbRef4Stencils st;
  auto xt = x1f.cpu().to(torch::kFloat64).contiguous();
  int nc1 = xt.size(0) - 1;
  std::vector<double> xf(xt.data_ptr<double>(),
                         xt.data_ptr<double>() + nc1 + 1);
  int nown = iu - is + 1;
  st.is = is;
  st.iu = iu;
  st.nc1 = nc1;
  st.clamp_in = clamp_in;
  st.clamp_out = clamp_out;
  // the clamped stencils need four owned cells next to the wall; a block
  // between two seams reads its (exchanged) ghost cells instead
  st.usable = (nown >= 4 || !(clamp_in || clamp_out)) && nc1 >= 5;
  if (!st.usable) return st;
  auto clampi = [](int j, int lo, int hi) {
    return std::max(lo, std::min(hi, j));
  };

  // 1. face density at faces is..iu+1: P'(x_f), P the quartic through the
  //    primitive P(x_{s+j}) = sum_{k<j} d_{s+k} dz_{s+k}, j = 0..4, window
  //    s = f-2 kept inside the owned cells at a clamped wall. The weight of
  //    cell s+k is dz_{s+k} sum_{j>k} L_j'(x_f). Uniform interior:
  //    (-1, 7, 7, -1)/12.
  std::vector<std::vector<int64_t>> ai;
  std::vector<std::vector<double>> aw;
  for (int f = is; f <= iu + 1; ++f) {
    int s = f - 2;
    if (clamp_in) s = std::max(s, is);
    if (clamp_out) s = std::min(s, iu - 3);
    s = clampi(s, 0, nc1 - 4);
    double h = xf[f + (f < nc1 ? 1 : 0)] - xf[f - (f < nc1 ? 0 : 1)];
    std::vector<double> X(5);
    for (int j = 0; j < 5; ++j) X[j] = (xf[s + j] - xf[f]) / h;
    std::vector<int64_t> idx(4);
    std::vector<double> wt(4, 0.);
    for (int k = 0; k < 4; ++k) {
      idx[k] = s + k;
      double dz = (xf[s + k + 1] - xf[s + k]) / h;
      for (int j = k + 1; j < 5; ++j) wt[k] += dz * lagrange_deriv(X, j, 0.);
    }
    ai.push_back(idx);
    aw.push_back(wt);
  }
  st.aidx = x1_stencil_index(ai, 4).to(options.device());
  st.awt = x1_stencil_weight(aw, 4).to(options);

  // 2. ghost cells past a clamped wall do not count for the resolution flag:
  //    the stencils there never read them, and the scan pressure falls
  //    steeply past a top wall
  auto counted = torch::ones({nc1}, torch::kBool);
  if (clamp_in) counted.narrow(0, 0, is).fill_(false);
  if (clamp_out) counted.narrow(0, iu + 1, nc1 - iu - 1).fill_(false);
  st.counted = counted.to(options.device());
  return st;
}

torch::Tensor wb_ref4_flag(WbRef4Stencils const& st,
                           torch::Tensor const& psf_lo,
                           torch::Tensor const& psf_hi) {
  if (!st.usable) return torch::ones_like(psf_lo, torch::kBool);
  // resolution flag, dilated by two cells
  auto bad = ((psf_lo / psf_hi).log().abs() > kMaxLogDrop) & st.counted;
  auto sizes = bad.sizes();
  return torch::max_pool1d(bad.to(psf_lo.dtype()).reshape({-1, 1, st.nc1}), 5,
                           1, 2)
             .reshape(sizes) > 0.;
}

void wb_ref4_faces(WbRef4Stencils const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag) {
  if (!st.usable) return;
  int nf = st.iu - st.is + 2;
  auto face = x1_stencil_apply(dref, st.aidx, st.awt);
  auto dl = dref.narrow(-1, st.is - 1, nf), dr = dref.narrow(-1, st.is, nf);
  auto ok = in_range(face, torch::minimum(dl, dr), torch::maximum(dl, dr)) &
            ~flag.narrow(-1, st.is - 1, nf) & ~flag.narrow(-1, st.is, nf);
  auto d = dsf.narrow(-1, st.is, nf);
  d.copy_(torch::where(ok, face, d));
}

}  // namespace snap
