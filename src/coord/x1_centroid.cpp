// C/C++
#include <algorithm>
#include <cctype>
#include <cmath>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/layout/layout.hpp>

#include "x1_centroid.hpp"
#include "x1_stencil.hpp"

namespace snap {

namespace {

// four-point Gauss-Legendre on [-1, 1]: exact to degree 7, and every
// integrand below (x^n r^2 with n <= 4, r L_j with L_j a quintic) is
// of degree <= 6
constexpr double kGx[4] = {-0.8611363115940526, -0.3399810435848563,
                           0.3399810435848563, 0.8611363115940526};
constexpr double kGw[4] = {0.3478548451374538, 0.6521451548625461,
                           0.6521451548625461, 0.3478548451374538};

// solve the dense n x n system A x = b in place (partial pivoting)
std::vector<double> solve(std::vector<std::vector<double>> A,
                          std::vector<double> b) {
  int n = b.size();
  for (int c = 0; c < n; ++c) {
    int p = c;
    for (int r = c + 1; r < n; ++r)
      if (std::abs(A[r][c]) > std::abs(A[p][c])) p = r;
    std::swap(A[c], A[p]);
    std::swap(b[c], b[p]);
    for (int r = c + 1; r < n; ++r) {
      double f = A[r][c] / A[c][c];
      for (int k = c; k < n; ++k) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  std::vector<double> x(n);
  for (int r = n - 1; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < n; ++k) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}

double lagrange(std::vector<double> const& X, int j, double x) {
  double prod = 1.;
  for (int m = 0; m < (int)X.size(); ++m) {
    if (m != j) prod *= (x - X[m]) / (X[j] - X[m]);
  }
  return prod;
}

std::vector<double> faces_of(torch::Tensor const& x1f) {
  auto xt = x1f.to(torch::kCPU, torch::kFloat64).contiguous();
  return std::vector<double>(xt.data_ptr<double>(),
                             xt.data_ptr<double>() + xt.size(0));
}

}  // namespace

bool x1_centroid_exact_enabled() {
  // read once from the environment (`SNAP_X1_CENTROID_EXACT` is not a deck
  // key): every block must make the same choice or the x1 seam faces stop
  // agreeing
  static const bool on = [] {
    auto v = get_env("SNAP_X1_CENTROID_EXACT", "0");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  return on;
}

X1PlainMeanStencils x1_plain_mean_stencils(
    torch::Tensor const& x1f, int is, int iu, bool clamp_in, bool clamp_out,
    torch::TensorOptions const& options) {
  X1PlainMeanStencils st;
  auto xf = faces_of(x1f);
  int nc1 = xf.size() - 1;
  st.is = is;
  st.iu = iu;
  st.nc1 = nc1;
  st.clamp_in = clamp_in;
  st.clamp_out = clamp_out;
  int lo = clamp_in ? is : 0, hi = clamp_out ? iu : nc1 - 1;
  // the mirrored ghosts need as many owned cells as ghosts
  st.usable = hi - lo + 1 >= 5 && (!clamp_in || iu - is + 1 >= is) &&
              (!clamp_out || iu - is + 1 >= nc1 - 1 - iu);
  if (!st.usable) return st;

  std::vector<std::vector<int64_t>> ci;
  std::vector<std::vector<double>> cw;
  for (int i = 0; i < nc1; ++i) {
    if (i < lo || i > hi) {  // past a clamped wall: filled by mirroring
      ci.push_back({i, i, i, i, i});
      cw.push_back({1., 0., 0., 0., 0.});
      continue;
    }
    int s = std::max(lo, std::min(hi - 4, i - 2));
    double c = 0.5 * (xf[i] + xf[i + 1]), h = xf[i + 1] - xf[i];
    // M[n][k]: r^2 mean of x^n over cell s+k, x = (r - c)/h
    std::vector<std::vector<double>> M(5, std::vector<double>(5, 0.));
    for (int k = 0; k < 5; ++k) {
      double a = xf[s + k], b = xf[s + k + 1], den = 0.;
      std::vector<double> num(5, 0.);
      for (int q = 0; q < 4; ++q) {
        double r = 0.5 * (a + b) + 0.5 * (b - a) * kGx[q];
        double wq = kGw[q] * r * r, x = (r - c) / h, xn = 1.;
        den += wq;
        for (int n = 0; n < 5; ++n, xn *= x) num[n] += wq * xn;
      }
      for (int n = 0; n < 5; ++n) M[n][k] = num[n] / den;
    }
    // plain mean of x^n over cell i
    std::vector<double> rhs = {1., 0., 1. / 12., 0., 1. / 80.};
    auto wgt = solve(M, rhs);
    std::vector<int64_t> idx(5);
    for (int k = 0; k < 5; ++k) idx[k] = s + k;
    ci.push_back(idx);
    cw.push_back(wgt);
  }
  st.idx = x1_stencil_index(ci, 5).to(options.device());
  st.wt = x1_stencil_weight(cw, 5).to(options);
  return st;
}

torch::Tensor x1_plain_means(X1PlainMeanStencils const& st,
                             torch::Tensor const& w, int ivx, bool odd_in,
                             bool odd_out) {
  if (!st.usable) return w.clone();
  auto out = x1_stencil_apply(w, st.idx, st.wt);
  auto parity = [&](bool odd) {
    std::vector<int64_t> shape(w.dim(), 1);
    shape[0] = w.size(0);
    auto par = torch::ones(shape, w.options());
    if (odd) par[ivx].fill_(-1.);
    return par;
  };
  int ng_in = st.is, ng_out = st.nc1 - 1 - st.iu;
  if (st.clamp_in && ng_in > 0) {
    auto d =
        (out.narrow(-1, st.is, ng_in) - w.narrow(-1, st.is, ng_in)).flip(-1);
    out.narrow(-1, 0, ng_in).copy_(w.narrow(-1, 0, ng_in) + parity(odd_in) * d);
  }
  if (st.clamp_out && ng_out > 0) {
    int m0 = st.iu + 1 - ng_out;
    auto d = (out.narrow(-1, m0, ng_out) - w.narrow(-1, m0, ng_out)).flip(-1);
    out.narrow(-1, st.iu + 1, ng_out)
        .copy_(w.narrow(-1, st.iu + 1, ng_out) + parity(odd_out) * d);
  }
  return out;
}

X1PressureSourceStencils x1_pressure_source_stencils(
    torch::Tensor const& x1f, int is, int iu, bool clamp_in, bool clamp_out,
    torch::TensorOptions const& options) {
  X1PressureSourceStencils st;
  auto xf = faces_of(x1f);
  st.is = is;
  st.iu = iu;
  st.usable = iu - is + 2 >= 6;
  if (!st.usable) return st;
  // a seam keeps the window centred on the neighbour's faces, as one block does
  int lo = clamp_in ? is : std::max(0, is - 2);
  int hi = clamp_out ? iu + 1 : std::min((int)xf.size() - 1, iu + 3);
  std::vector<std::vector<int64_t>> si;
  std::vector<std::vector<double>> sw;
  for (int i = is; i <= iu; ++i) {
    int s = std::max(lo, std::min(hi - 5, i - 2));
    double c = 0.5 * (xf[i] + xf[i + 1]), h = xf[i + 1] - xf[i];
    std::vector<double> X(6);
    for (int j = 0; j < 6; ++j) X[j] = (xf[s + j] - c) / h;
    // the radial cell volume exactly as cell_volume() forms it, so that a
    // constant pressure still exerts no force against the face-area flux
    double vol = (std::pow(xf[i + 1], 3) - std::pow(xf[i], 3)) / 3.;
    std::vector<int64_t> idx(6);
    std::vector<double> wt(6, 0.);
    for (int j = 0; j < 6; ++j) {
      idx[j] = s + j;
      for (int q = 0; q < 4; ++q) {
        double r = c + 0.5 * h * kGx[q];
        wt[j] += 0.5 * h * kGw[q] * r * lagrange(X, j, 0.5 * kGx[q]);
      }
      wt[j] *= 2. / vol;
    }
    si.push_back(idx);
    sw.push_back(wt);
  }
  st.idx = x1_stencil_index(si, 6).to(options.device());
  st.wt = x1_stencil_weight(sw, 6).to(options);
  return st;
}

torch::Tensor x1_pressure_source(X1PressureSourceStencils const& st,
                                 torch::Tensor const& face_pressure1) {
  return x1_stencil_apply(face_pressure1, st.idx, st.wt);
}

}  // namespace snap
