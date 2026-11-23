// V3 C9 (C.rst section 10 row C9, :1828-1830; findings :470-476): the WENO
// point functions and LMARS compute a float instantiation in FP32
// throughout, and the uniform weno5 rows hold their linear weights d in the
// table's dtype. Each case compares the library's float result, bit for bit,
// with a reference written here in float arithmetic only (every constant is
// the double literal rounded to float once). Before C9 the float paths
// promoted to double through untyped constants, so the bits differ on many
// faces. The FP64 cases check that the double results are the pre-C9
// formulas, bit for bit.
//
// Bits compare only if the library does not fuse multiply-adds: GCC with
// -ffp-contract=fast on an FMA target (e.g. a toolchain that defaults to
// x86-64-v3) fuses each translation unit by its own choice, so the library
// and an identical formula compiled here can differ in the last bits. The
// test asks the library itself (the cp3 probe below). If it fuses, a WENO
// case must be within kUlps of the reference on every face, and a float
// result must also differ from the FP64 result rounded to float on at least
// a quarter of the faces, which an FP64 computation cannot do. An unfused
// build (the bitwise-cpu CI job) keeps the bit-for-bit check.

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/riemann/riemann_kernels.h>

#include <snap/recon/reconstruct.hpp>

#include "make_interp.hpp"

using namespace snap;

namespace {

template <typename T>
T vdot(T const* a, T const* b, int n) {
  T out = T(0);
  for (int i = 0; i < n; ++i) out += a[i] * b[i];
  return out;
}

template <typename T>
T sq(T x) {
  return x * x;
}

// weno3 with the four 3-entry rows of c; typed constants
template <typename T>
T ref_weno3(T const* s, T const* c, bool scale) {
  T vscale = scale ? (std::fabs(s[0]) + std::fabs(s[1]) + std::fabs(s[2])) /
                         static_cast<T>(3.0)
                   : static_cast<T>(1.0);
  if (vscale == T(0)) return T(0);
  T phi[3];
  for (int k = 0; k < 3; ++k) phi[k] = s[k] / vscale;
  T p0 = vdot(phi, c, 3), p1 = vdot(phi, c + 3, 3);
  T b0 = sq(vdot(phi, c + 6, 3)), b1 = sq(vdot(phi, c + 9, 3));
  T a0 = static_cast<T>(2.0 / 3.0) / sq(b0 + static_cast<T>(1e-6));
  T a1 = static_cast<T>(1.0 / 3.0) / sq(b1 + static_cast<T>(1e-6));
  return (a0 * p0 + a1 * p1) / (a0 + a1) * vscale;
}

// weno5 with the nine 5-entry rows of c and the weights d = c[45..47]
template <typename T>
T ref_weno5(T const* s, T const* c, bool scale) {
  T vscale = scale ? (std::fabs(s[0]) + std::fabs(s[1]) + std::fabs(s[2]) +
                      std::fabs(s[3]) + std::fabs(s[4])) /
                         static_cast<T>(5.0)
                   : static_cast<T>(1.0);
  if (vscale == T(0)) return T(0);
  T phi[5];
  for (int k = 0; k < 5; ++k) phi[k] = s[k] / vscale;
  T p[3], b[3];
  for (int r = 0; r < 3; ++r) {
    p[r] = vdot(phi, c + 5 * r, 5);
    b[r] = static_cast<T>(13. / 12.) * sq(vdot(phi, c + 15 + 10 * r, 5)) +
           static_cast<T>(.25) * sq(vdot(phi, c + 20 + 10 * r, 5));
  }
  T a[3];
  for (int r = 0; r < 3; ++r)
    a[r] = c[45 + r] / sq(b[r] + static_cast<T>(1e-6));
  return vscale * (a[0] * p[0] + a[1] * p[1] + a[2] * p[2]) /
         (a[0] + a[1] + a[2]);
}

// the pre-C9 double weno5 (literal weights .3 .6 .1), for the FP64 guard
double old_weno5(double const* s, double const* c, bool scale) {
  double vscale =
      scale ? (fabs(s[0]) + fabs(s[1]) + fabs(s[2]) + fabs(s[3]) + fabs(s[4])) /
                  5.0
            : 1.0;
  if (vscale == 0.0) return 0.0;
  double phi[5];
  for (int k = 0; k < 5; ++k) phi[k] = s[k] / vscale;
  double p[3], b[3];
  for (int r = 0; r < 3; ++r) {
    p[r] = vdot(phi, c + 5 * r, 5);
    b[r] = 13. / 12. * sq(vdot(phi, c + 15 + 10 * r, 5)) +
           .25 * sq(vdot(phi, c + 20 + 10 * r, 5));
  }
  double a0 = .3 / sq(b[0] + 1e-6), a1 = .6 / sq(b[1] + 1e-6),
         a2 = .1 / sq(b[2] + 1e-6);
  return vscale * (a0 * p[0] + a1 * p[1] + a2 * p[2]) / (a0 + a1 + a2);
}

// the double rows of the uniform tables, written out independently
std::vector<double> weno3_left() {
  return {1. / 2., 1. / 2., 0., 0., 3. / 2., -1. / 2.,
          1.,      -1.,     0., 0., 1.,      -1.};
}

std::vector<double> weno5_left() {
  std::vector<double> c = {
      -1. / 6., 5. / 6., 1. / 3., 0.,  0.,       0.,       1. / 3., 5. / 6.,
      -1. / 6., 0.,      0.,      0.,  11. / 6., -7. / 6., 1. / 3., 1.,
      -2.,      1.,      0.,      0.,  1.,       -4.,      3.,      0.,
      0.,       0.,      1.,      -2., 1.,       0.,       0.,      -1.,
      0.,       1.,      0.,      0.,  0.,       1.,       -2.,     1.,
      0.,       0.,      3.,      -4., 1.};
  c.insert(c.end(), {.3, .6, .1});
  return c;
}

// the right face table: each stencil row reversed, d unchanged
std::vector<double> flip_rows(std::vector<double> c, int nrow, int width) {
  for (int r = 0; r < nrow; ++r)
    std::reverse(c.begin() + r * width, c.begin() + (r + 1) * width);
  return c;
}

template <typename T>
std::vector<T> cast(std::vector<double> const& v) {
  return std::vector<T>(v.begin(), v.end());
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
bool same_bits(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// position on the number line: neighbouring values differ by 1
int64_t ord(float x) {
  int32_t i;
  std::memcpy(&i, &x, 4);
  return i < 0 ? -int64_t(i & 0x7fffffff) : i;
}
int64_t ord(double x) {
  int64_t i;
  std::memcpy(&i, &x, 8);
  return i < 0 ? -(i & INT64_MAX) : i;
}

constexpr int64_t kUlps = 8;  // fused vs unfused WENO: at most 5 measured

torch::Tensor interp_out(std::string const& type, int width, bool scale,
                         torch::Tensor const& x, bool left) {
  auto opts = std::make_shared<InterpOptionsImpl>(type);
  opts->scale() = scale;
  auto interp = std::make_shared<InterpImpl>(opts);
  interp->to(torch::kCPU, x.scalar_type());
  auto out = torch::zeros({x.size(0) - (width - 1)}, x.options());
  left ? interp->left(x, 0, out) : interp->right(x, 0, out);
  return out;
}

// Does the library fuse multiply-adds? cp3 is a 3-term dot product with the
// same typed table, so its bits are either the unfused sum or the one fma
// chain; the library must match one of the two on every face.
template <typename T>
bool library_fuses(torch::Tensor const& x) {
  auto out = interp_out("cp3", 3, false, x, true);
  auto c = cast<T>({1. / 3., 5. / 6., -1. / 6.});
  auto const* px = x.template data_ptr<T>();
  int n = out.size(0), plain = 0, fused = 0;
  for (int i = 0; i < n; ++i) {
    volatile T acc = c[0] * px[i];  // volatile: no fusing here
    volatile T p1 = c[1] * px[i + 1];
    acc = acc + p1;
    volatile T p2 = c[2] * px[i + 2];
    acc = acc + p2;
    T f = std::fma(c[2], px[i + 2], std::fma(c[1], px[i + 1], c[0] * px[i]));
    T lib = out.template data_ptr<T>()[i];
    plain += same_bits(lib, static_cast<T>(acc));
    fused += same_bits(lib, f);
  }
  EXPECT_TRUE(plain == n || fused == n)
      << "cp3 is neither the unfused nor the fused sum: " << plain << ", "
      << fused << " of " << n;
  return plain != n;
}

// Interp(type) on w against ref(window, table), both faces. Bit for bit; if
// the library fuses multiply-adds, within kUlps instead, and a float result
// must differ from the FP64 one (ref64 on the same inputs and float table,
// rounded to float) on at least a quarter of the faces.
template <typename T, typename Ref, typename Ref64>
void expect_matches(std::string const& type, int width, bool scale,
                    std::vector<double> const& left, int nrow, Ref ref,
                    Ref64 ref64, torch::Tensor const& w) {
  auto dtype = std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;
  auto x = w.to(dtype).contiguous();
  auto const* px = x.template data_ptr<T>();
  auto cl = cast<T>(left);
  auto cr = cast<T>(flip_rows(left, nrow, width));
  int nface = 0, bad = 0, unlike64 = 0;
  int64_t gap = 0;
  for (bool is_left : {true, false}) {
    auto out = interp_out(type, width, scale, x, is_left);
    auto const& c = is_left ? cl : cr;
    std::vector<double> c64(c.begin(), c.end());
    for (int i = 0; i < out.size(0); ++i, ++nface) {
      T lib = out.template data_ptr<T>()[i], r = ref(px + i, c.data(), scale);
      bad += !same_bits(lib, r);
      gap = std::max(gap, std::abs(ord(lib) - ord(r)));
      double s64[5];
      for (int k = 0; k < width; ++k) s64[k] = px[i + k];
      unlike64 +=
          !same_bits(lib, static_cast<T>(ref64(s64, c64.data(), scale)));
    }
  }
  std::string what = type + " scale=" + std::to_string(scale);
  if (bad == 0 || !library_fuses<T>(x)) {
    EXPECT_EQ(bad, 0) << what;
    return;
  }
  std::printf("[ fused    ] %s: %d of %d faces differ, at most %lld ulp\n",
              what.c_str(), bad, nface, static_cast<long long>(gap));
  EXPECT_LE(gap, kUlps) << what;
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_GE(4 * unlike64, nface)
        << what << ": FP64 result on " << nface - unlike64 << " faces";
  }
}

torch::Tensor wavy(int n) {
  torch::manual_seed(20261120);
  auto r = torch::randn({n}, torch::kFloat64);
  auto i = torch::arange(n, torch::kFloat64);
  // smooth parts, jumps and sign changes, so every weight regime appears
  return 1.e5 + 3.e3 * torch::sin(0.37 * i) + 50. * r +
         2.e3 * (i.remainder(97) < 40).to(torch::kFloat64);
}

}  // namespace

TEST(Fp32TypedConstants, uniform_weno5_table_holds_its_weights) {
  auto interp = make_interp("weno5");
  ASSERT_EQ(interp->cm.numel(), SchemeTraits<Scheme::weno5>::ncoeff);
  ASSERT_EQ(interp->cm.numel(), 48);
  auto d = interp->cm.narrow(0, 45, 3);
  EXPECT_TRUE(torch::equal(d, torch::tensor({.3, .6, .1}, torch::kFloat64)));
  EXPECT_TRUE(torch::equal(interp->cp.narrow(0, 45, 3), d));
}

TEST(Fp32TypedConstants, weno5_float_computes_in_fp32) {
  auto w = wavy(4004);
  for (bool scale : {false, true}) {
    expect_matches<float>("weno5", 5, scale, weno5_left(), 9, ref_weno5<float>,
                          ref_weno5<double>, w);
  }
}

TEST(Fp32TypedConstants, weno3_float_computes_in_fp32) {
  auto w = wavy(4002);
  for (bool scale : {false, true}) {
    expect_matches<float>("weno3", 3, scale, weno3_left(), 4, ref_weno3<float>,
                          ref_weno3<double>, w);
  }
}

TEST(Fp32TypedConstants, weno_double_is_unchanged) {
  auto w = wavy(4004);
  for (bool scale : {false, true}) {
    expect_matches<double>("weno5", 5, scale, weno5_left(), 9, old_weno5,
                           old_weno5, w);
    expect_matches<double>("weno3", 3, scale, weno3_left(), 4,
                           ref_weno3<double>, ref_weno3<double>, w);
  }
}

namespace {

// LMARS with ny = 0 along IVX (dim 3), every constant typed T. The kinetic
// terms are inline products, the expression shape of lmars_impl's SQR macro,
// so a compiler that contracts one into FMAs contracts the other the same way
template <typename T>
void ref_lmars(T* f, T const* l, T const* r, T hl, T hr, T gl, T gr) {
  T const h = static_cast<T>(0.5);
  hl += h * (l[1] * l[1] + l[2] * l[2] + l[3] * l[3]) + l[4] / l[0];
  hr += h * (r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) + r[4] / r[0];
  T rhobar = h * (l[0] + r[0]);
  T gbar = h * (gl + gr);
  T cbar = std::sqrt(h * gbar * (l[4] + r[4]) / rhobar);
  T pbar = h * (l[4] + r[4]) + h * (rhobar * cbar) * (l[1] - r[1]);
  T ubar = h * (l[1] + r[1]) + h / (rhobar * cbar) * (l[4] - r[4]);
  T const* s = ubar > T(0) ? l : r;
  T hs = ubar > T(0) ? hl : hr;
  T rd = T(1);
  f[0] = ubar * s[0] * rd;
  f[1] = ubar * s[0] * s[1] + pbar;
  f[2] = ubar * s[0] * s[2];
  f[3] = ubar * s[0] * s[3];
  f[4] = ubar * s[0] * hs;
}

template <typename T>
int lmars_differing(int n) {
  torch::manual_seed(77);
  auto u = torch::rand({n, 14}, torch::kFloat64);
  auto a = u.accessor<double, 2>();
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    T wl[5] = {T(0.5 + a[i][0]), T(40. * a[i][1] - 20.), T(10. * a[i][2]),
               T(-10. * a[i][3]), T(1.e5 * (0.5 + a[i][4]))};
    T wr[5] = {T(0.5 + a[i][5]), T(40. * a[i][6] - 20.), T(10. * a[i][7]),
               T(-10. * a[i][8]), T(1.e5 * (0.5 + a[i][9]))};
    T hl = T(1.e5 * a[i][10]), hr = T(1.e5 * a[i][11]);
    T gl = T(1.3 + 0.2 * a[i][12]), gr = T(1.3 + 0.2 * a[i][13]);
    T flx[5], ref[5];
    lmars_impl<T>(flx, wl, wr, hl, hr, gl, gr, /*dim=*/3, /*ny=*/0, 1, 1);
    ref_lmars<T>(ref, wl, wr, hl, hr, gl, gr);
    for (int k = 0; k < 5; ++k) bad += !same_bits(flx[k], ref[k]);
  }
  return bad;
}

}  // namespace

TEST(Fp32TypedConstants, lmars_float_computes_in_fp32) {
  EXPECT_EQ(lmars_differing<float>(20000), 0);
}

TEST(Fp32TypedConstants, lmars_double_matches_its_formula) {
  EXPECT_EQ(lmars_differing<double>(20000), 0);
}
