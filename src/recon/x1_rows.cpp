#include "x1_rows.hpp"

#include <ATen/native/ReduceOpsUtils.h>

#include <algorithm>
#include <cmath>
#include <vector>

// snap
#include <snap/hydro/hydro_ref_x1_impl.h>

#include <snap/coord/x1_stencil.hpp>

namespace snap {

// lp[j] = L_j'(x), the Lagrange basis through z; the weight of cell k is its
// width times the sum of lp over the faces above it
void face_value_row(double const* z, int n, double x, double* w) {
  TORCH_CHECK(n >= 1 && n <= 5, "[face_value_row] 1 to 5 cells, got ", n);
  double lp[6];
  for (int j = 0; j <= n; ++j) {
    double s = 0.;
    for (int i = 0; i <= n; ++i) {
      if (i == j) continue;
      double term = 1. / (z[j] - z[i]);
      for (int l = 0; l <= n; ++l) {
        if (l == j || l == i) continue;
        term *= (x - z[l]) / (z[j] - z[l]);
      }
      s += term;
    }
    lp[j] = s;
  }

  for (int k = 0; k < n; ++k) {
    double acc = 0.;
    for (int j = k + 1; j <= n; ++j) acc += lp[j];
    w[k] = (z[k + 1] - z[k]) * acc;
  }
}

namespace {

void quad_inv_rows(double const* z, double* r1, double* r2) {
  double V[3][3];
  for (int k = 0; k < 3; ++k) {
    double p = z[k], q = z[k + 1];
    V[k][0] = 1.;
    V[k][1] = .5 * (p + q);
    V[k][2] = (p * p + p * q + q * q) / 3.;
  }

  double cof[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      cof[i][j] = V[i1][j1] * V[i2][j2] - V[i1][j2] * V[i2][j1];
    }
  }
  double det = V[0][0] * cof[0][0] + V[0][1] * cof[0][1] + V[0][2] * cof[0][2];
  TORCH_CHECK(det != 0., "[x1_coeffs] degenerate WENO5 stencil");

  for (int j = 0; j < 3; ++j) {
    r1[j] = cof[j][1] / det;
    r2[j] = cof[j][2] / det;
  }
}

void ptype_row(double const* y, double* row) {
  double h2 = y[3] - y[2];
  double z[6];
  for (int k = 0; k < 6; ++k) z[k] = (y[k] - y[3]) / h2;

  const int k0[3] = {2, 1, 0};
  const double bsign[3] = {1., 1., -1.};

  for (int i = 0; i < 45; ++i) row[i] = 0.;

  for (int r = 0; r < 3; ++r) {
    double const* nodes = z + k0[r];

    double w[3];
    face_value_row(nodes, 3, 0., w);

    double r1[3], r2[3];
    quad_inv_rows(nodes, r1, r2);

    for (int n = 0; n < 3; ++n) {
      int col = k0[r] + n;
      row[r * 5 + col] = w[n];
      row[(3 + 2 * r) * 5 + col] = 2. * r2[n];
      row[(4 + 2 * r) * 5 + col] = bsign[r] * (2. * r2[n] - 2. * r1[n]);
    }
  }

  double a = -z[0], b = -z[1], c = z[4], e = z[5];
  row[45] = a * b / ((a + e) * (b + e));
  row[46] = a * e * (a + b + c + e) / ((a + e) * (b + e) * (a + c));
  row[47] = c * e / ((a + c) * (a + e));
}

void window_weno5(double const* y, double* cm, double* cp) {
  ptype_row(y, cp);

  double ym[6];
  for (int k = 0; k < 6; ++k) ym[k] = -y[5 - k];

  double tmp[kWeno5NuCoeff];
  ptype_row(ym, tmp);
  for (int r = 0; r < 9; ++r) {
    for (int n = 0; n < 5; ++n) cm[r * 5 + n] = tmp[r * 5 + (4 - n)];
  }
  cm[45] = tmp[45];
  cm[46] = tmp[46];
  cm[47] = tmp[47];
}

void combine_optimal(double const* row48, double* out5) {
  for (int n = 0; n < 5; ++n) {
    out5[n] = row48[45] * row48[n] + row48[46] * row48[5 + n] +
              row48[47] * row48[10 + n];
  }
}

void window_cp5(double const* y, double* cm, double* cp) {
  double rm[kWeno5NuCoeff], rp[kWeno5NuCoeff];
  window_weno5(y, rm, rp);
  combine_optimal(rm, cm);
  combine_optimal(rp, cp);
}

torch::Tensor check_faces(torch::Tensor const& x1f) {
  TORCH_CHECK(x1f.defined(), "[x1_coeffs] face tensor is undefined");
  TORCH_CHECK(x1f.dim() == 1, "[x1_coeffs] expected a 1-D face vector, got ",
              x1f.dim(), " dimensions");
  TORCH_CHECK(x1f.numel() >= 6,
              "[x1_coeffs] a WENO5/cp5 window needs at least 6 faces, got ",
              x1f.numel());

  auto f = x1f.cpu().to(torch::kFloat64).contiguous();  // MPS has no float64
  TORCH_CHECK(torch::isfinite(f).all().item<bool>(),
              "[x1_coeffs] face vector has a non-finite entry");

  auto d = f.slice(0, 1) - f.slice(0, 0, f.numel() - 1);
  TORCH_CHECK(d.min().item<double>() > 0.,
              "[x1_coeffs] face vector must be strictly increasing");
  return f;
}

template <int NCoeff, void (*Window)(double const*, double*, double*)>
std::pair<torch::Tensor, torch::Tensor> build_x1_coeffs(
    torch::Tensor const& x1f) {
  auto f = check_faces(x1f);
  int64_t nout = f.numel() - 5;
  auto cm = torch::empty({nout, NCoeff}, torch::kFloat64);
  auto cp = torch::empty({nout, NCoeff}, torch::kFloat64);
  auto const* y = f.data_ptr<double>();
  auto* pm = cm.data_ptr<double>();
  auto* pp = cp.data_ptr<double>();
  for (int64_t m = 0; m < nout; ++m)
    Window(y + m, pm + m * NCoeff, pp + m * NCoeff);
  return {cm, cp};
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f) {
  return build_x1_coeffs<kWeno5NuCoeff, window_weno5>(x1f);
}

std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(
    torch::Tensor const& x1f) {
  return build_x1_coeffs<kPoly5NuCoeff, window_cp5>(x1f);
}

void x1_reset_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                     torch::Tensor& cp_x1, int ncoeff) {
  bool installed = cm_x1.defined() && cm_x1.numel() > 0;

  cm_x1 = torch::zeros({0, ncoeff}, torch::kFloat64);
  cp_x1 = torch::zeros({0, ncoeff}, torch::kFloat64);

  if (installed) {
    cm_x1 = m.register_buffer("cm_x1", cm_x1);
    cp_x1 = m.register_buffer("cp_x1", cp_x1);
  }
}

torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim) {
  TORCH_CHECK(c.dim() == 2 && c.is_contiguous(),
              "[x1_coeffs] row view needs a contiguous 2-D table");
  TORCH_CHECK(dim >= 0 && dim < out.dim(), "[x1_coeffs] bad dim ", dim);
  TORCH_CHECK(c.size(0) == out.size(dim), "[x1_coeffs] table has ", c.size(0),
              " rows but the output has ", out.size(dim), " faces along dim ",
              dim);

  auto sizes = out.sizes().vec();
  std::vector<int64_t> strides(sizes.size(), 0);
  strides[dim] = c.size(1);
  return c.as_strided(sizes, strides, c.storage_offset());
}

at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c) {
  TORCH_CHECK(dim >= 0 && dim < out.dim(), "[x1_coeffs] bad dim ", dim);
  TORCH_CHECK(c.scalar_type() == w.scalar_type() && c.device() == w.device(),
              "[x1_coeffs] coefficient table is ", c.scalar_type(), " on ",
              c.device(), " but the state is ", w.scalar_type(), " on ",
              w.device());

  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto config = at::TensorIteratorConfig();
  config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(out.sizes(), squash_dim)
      .add_output(out)
      .add_input(w);

  torch::Tensor rows;
  if (!w.device().is_cuda()) {
    rows = x1_coeffs_row_view(c, out, dim);
    config.add_input(rows);
  }

  return config.build();
}

void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what) {
  TORCH_CHECK(c.dim() == 2 && c.size(1) == ncoeff, "[", what,
              "] expected a (nout, ", ncoeff, ") coefficient table, got ",
              c.sizes());
  TORCH_CHECK(c.is_contiguous(), "[", what,
              "] coefficient table must be contiguous");
  int64_t nout = at::native::ensure_nonempty_size(iter.output(), dim);
  TORCH_CHECK(c.size(0) == nout, "[", what, "] coefficient table has ",
              c.size(0), " rows but the output has ", nout,
              " faces along dimension ", dim);
}

std::pair<torch::Tensor, torch::Tensor> conv4_rows(torch::Tensor const& x1f,
                                                   int n) {
  TORCH_CHECK(n >= 4 && x1f.size(0) == n + 1,
              "dynamics/conversion-order 4 needs nc1 >= 4 cells and nc1 + 1 "
              "x1 faces, got ",
              n, " and ", x1f.size(0));
  auto x = x1f.cpu().to(torch::kFloat64);
  auto f = [&](int s) { return x.narrow(0, s, n - 2); };
  auto c = (f(2) - f(1)).square() / 4. / (f(3) - f(0));
  return {c / (f(3) - f(1)), c / (f(2) - f(0))};
}

torch::Tensor wall4_rows(torch::Tensor const& x1f, int is, int iu, int ng) {
  TORCH_CHECK(ng <= 4, "key-4 wall ghosts mirror at most 4 cells");
  auto x = x1f.to(torch::kCPU, torch::kFloat64);
  auto E = torch::zeros({2, 4, ng, 4}, torch::kFloat64);
  for (int s : {0, 1}) {
    int w = s ? iu + 1 : is, d = s ? -1 : 1;
    auto y = (x - x[w]) / (x[w + d] - x[w]);
    auto m = torch::arange(5, torch::kFloat64) + 1.;
    auto row = [&](int j) {
      return (y[j + 1].pow(m) - y[j].pow(m)) / m / (y[j + 1] - y[j]);
    };
    std::vector<torch::Tensor> a, b;
    for (int k = 0; k < 4; ++k) a.push_back(row(s ? iu - 3 + k : is + k));
    for (int k = 0; k < ng; ++k) b.push_back(row(s ? iu + 1 + k : is - ng + k));
    auto A = torch::stack(a), Cr = A.narrow(1, 0, 4).inverse();
    E[s][0] = torch::stack(b).narrow(1, 0, 4).matmul(Cr);
    auto Cv = at::linalg_pinv(A.narrow(1, 1, 3)), P = A.narrow(0, s, 3);
    auto Cp = P.index_select(1, torch::tensor({0, 2, 3})).inverse();
    for (int k = 0; k < ng; ++k) {
      int m = s ? 3 - k : ng - 1 - k;  // the ghost's mirror cell
      E[s][1][k][m] = -1., E[s][2][k][m] = E[s][3][k][m] = 1.;
      E[s][1][k] += 2. * a[m][2] * Cv[1];
      E[s][2][k] -= 2. * (a[m][1] * Cr[1] + a[m][3] * Cr[3]);
      E[s][3][k].narrow(0, s, 3) -= 2. * a[m][3] * Cp[2];
    }
  }
  return E;
}

FaceDensityRows face_density_rows(torch::Tensor const& x1f, int is, int iu,
                                  bool clamp_in, bool clamp_out,
                                  torch::TensorOptions const& options) {
  FaceDensityRows st;
  st.key = X1RowsKey(x1f, options);
  auto xt = x1f.cpu().to(torch::kFloat64).contiguous();
  int nc1 = xt.size(0) - 1;
  auto const* xf = xt.data_ptr<double>();
  int nown = iu - is + 1;
  st.is = is;
  st.iu = iu;
  st.nc1 = nc1;
  st.clamp_in = clamp_in;
  st.clamp_out = clamp_out;
  // the clamped windows need four owned cells next to the wall; a block
  // between two seams reads its (exchanged) ghost cells instead
  st.usable = (nown >= 4 || !(clamp_in || clamp_out)) && nc1 >= 5;
  if (!st.usable) return st;

  // window s = f - 2 kept inside the owned cells at a clamped wall, in units
  // of the face's cell width. Uniform interior: (-1, 7, 7, -1) / 12
  std::vector<std::vector<int64_t>> ai;
  std::vector<std::vector<double>> aw;
  for (int f = is; f <= iu + 1; ++f) {
    int s = f - 2;
    if (clamp_in) s = std::max(s, is);
    if (clamp_out) s = std::min(s, iu - 3);
    s = std::max(0, std::min(nc1 - 4, s));
    double h = xf[f + (f < nc1 ? 1 : 0)] - xf[f - (f < nc1 ? 0 : 1)];
    double z[5];
    for (int j = 0; j < 5; ++j) z[j] = (xf[s + j] - xf[f]) / h;
    std::vector<double> wt(4);
    face_value_row(z, 4, 0., wt.data());
    ai.push_back({s, s + 1, s + 2, s + 3});
    aw.push_back(wt);
  }
  st.idx = x1_stencil_index(ai, 4).to(options.device());
  st.wt = x1_stencil_weight(aw, 4).to(options);

  // ghost cells past a clamped wall do not count for the resolution flag:
  // the windows there never read them, and the scan pressure falls steeply
  // past a top wall
  auto counted = torch::ones({nc1}, torch::kBool);
  if (clamp_in) counted.narrow(0, 0, is).fill_(false);
  if (clamp_out) counted.narrow(0, iu + 1, nc1 - iu - 1).fill_(false);
  st.counted = counted.to(options.device());
  return st;
}

torch::Tensor six_face_rows(torch::Tensor const& dx1f, int iu, bool phys_in,
                            bool phys_out, bool wall_clamp) {
  auto h = dx1f.cpu().to(torch::kFloat64).contiguous();
  int nc1 = h.size(0);
  auto hp = h.data_ptr<double>();
  auto rows = torch::zeros({nc1, 7}, torch::kFloat64);
  auto r = rows.accessor<double, 2>();
  long double const tq[3] = {0.5L - 0.1L * std::sqrt(15.0L), 0.5L,
                             0.5L + 0.1L * std::sqrt(15.0L)};
  long double const wq[3] = {5.0L / 18.0L, 8.0L / 18.0L, 5.0L / 18.0L};
  for (int i = 0; i < nc1; ++i) {
    int start =
        hydro_ref_x1_six_face_start(i, nc1, iu, phys_in, phys_out, wall_clamp);
    r[i][0] = start;
    if (start < 0) continue;
    long double t[6];
    for (int k = 0; k < 6; ++k) {
      long double o = 0.0L;
      for (int j = i; j < start + k; ++j) o += hp[j];
      for (int j = start + k; j < i; ++j) o -= hp[j];
      t[k] = o / hp[i];
    }
    for (int m = 0; m < 6; ++m) {
      long double wm = 0.0L;
      for (int q = 0; q < 3; ++q) {
        long double l = 1.0L;
        for (int k = 0; k < 6; ++k)
          if (k != m) l *= (tq[q] - t[k]) / (t[m] - t[k]);
        wm += wq[q] * l;
      }
      r[i][1 + m] = static_cast<double>(wm);
    }
  }
  return rows;
}

}  // namespace snap
