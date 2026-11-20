#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "flux_decomposition_impl.h"
#include "implicit_dispatch.hpp"
#include "vic_gravity_work_impl.h"

#define GAMMA(n) gamma[(n) * stride2]
#define AREA(n) area[(n) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Assemble the block-tridiagonal coefficients a[i], b[i], c[i] of ONE vertical
// cell `i` of a column, once for both matrices (V3 lane I, I-1). Every cell is
// independent, so one GPU thread can own one cell; the interface Jacobians it
// shares with a neighbour are recomputed from the same inputs.
//
// The blocks are always built 5x5 from the Roe |A| at the two faces and the
// flux Jacobians of cells i-1, i, i+1. N = 5 keeps them; N = 3 keeps rows and
// columns {rho, m1, E}, entry by entry the same expressions.
//
// a, b, c point to the start of the column's coefficient arrays (indexed [i]),
// matching the layout used by ForwardSweep and the split redistribution
// helpers.
//
// N = 3 with du and rhs_corr set: the tangential momenta m_t (not unknowns of
// the partial matrix) are no longer held fixed: the increment of cell j is
// written dm_t,j = v_t,j drho_j + q_t,j, so each block's rho column gains its
// m_t columns times the velocity of the cell it multiplies (the |A| and flux
// Jacobian columns alike), and q_t,j = du_t,j - v_t,j du_rho,j (explicit
// increments) moves to the right-hand side:
//   rhs_corr[i] = aT q_i + bT q_{i-1} + cT q_{i+1}   (rows rho, m1, E)
// ForwardSweep subtracts it. A face merged into a[i] by the wall mirror (or a
// solid neighbour) carries cell i's v_t and q_t, as the mirror does. Exact
// zeros (rest state) leave a, b, c and the right-hand side unchanged bit for
// bit. Without du / rhs_corr the assembly is the frozen-m_t one.
template <typename T, int N>
void DISPATCH_MACRO vic_assemble_impl(
    Eigen::Matrix<T, N, N>* a, Eigen::Matrix<T, N, N>* b,
    Eigen::Matrix<T, N, N>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block, bool periodic,
    bool solid_lower, bool solid_upper, T* du = nullptr,
    Eigen::Matrix<T, N, 1>* rhs_corr = nullptr) {
  static_assert(N == 3 || N == 5, "VIC blocks are 3x3 or 5x5");
  bool face_work = dir & kVicFaceWork;
  bool diffusive_cell = dir & kVicDiffusiveCell;
  dir &= ~(kVicFaceWork | kVicDiffusiveCell);
  // eigenvectors, eigenvalues, inverse matrix of eigenvectors.
  Eigen::Matrix<T, 5, 5> Rmat, Rimat;
  Eigen::Matrix<T, 5, 1> Lambda;

  // reduced diffusion matrix |A_{i-1/2}|, |A_{i+1/2}|
  Eigen::Matrix<T, 5, 5> Am, Ap;
  Eigen::Matrix<T, 5, 5> dfdq_prev, dfdq_curr, dfdq_next;

  Eigen::Matrix<T, 5, 5> Phi;
  Eigen::Matrix<T, 5, 1> Dt, Bnd;

  Phi.setZero();
  Phi(IVX + dir, IDN) = grav;
  Phi(IPR, IVX + dir) = grav;

  Dt.setConstant(1. / dt);

  Bnd.setConstant(1.);
  Bnd(IVX + dir) = -1;

  T prim[5];       // Roe averaged primitive variables of cell i-1/2
  T wl[5], wr[5];  // left/right primitive variables of cell i-1 and i
  T gm1, cs;

  // Interface i-1/2 and the Jacobians in cells i-1 and i.
  CopyPrimitives(wl, wr, w, i, stride1, stride2, ny);
  if (solid_lower) {
    for (int n = 0; n < 5; ++n) wl[n] = wr[n];
    wl[IVX + dir] = -wr[IVX + dir];
  }
  gm1 = GAMMA(solid_lower ? i : i - 1) - 1.;
  FluxJacobian(dfdq_prev, gm1, wl, dir);
  gm1 = GAMMA(i) - 1.;
  FluxJacobian(dfdq_curr, gm1, wr, dir);

  gm1 = 0.5 * (GAMMA(solid_lower ? i : i - 1) + GAMMA(i)) - 1.;
  RoeAverage(prim, gm1, wl, wr);

  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

  Am.noalias() = Rmat * Lambda.asDiagonal() * Rimat;

  // Interface i+1/2 and the Jacobian in cell i+1.
  CopyPrimitives(wl, wr, w, i + 1, stride1, stride2, ny);
  if (solid_upper) {
    for (int n = 0; n < 5; ++n) wr[n] = wl[n];
    wr[IVX + dir] = -wl[IVX + dir];
  }
  gm1 = GAMMA(solid_upper ? i : i + 1) - 1.;
  FluxJacobian(dfdq_next, gm1, wr, dir);

  gm1 = 0.5 * (GAMMA(i) + GAMMA(solid_upper ? i : i + 1)) - 1.;
  RoeAverage(prim, gm1, wl, wr);

  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

  Ap.noalias() = Rmat * Lambda.asDiagonal() * Rimat;

  T const& area_i = AREA(i);
  T const& area_ip1 = AREA(i + 1);
  T half_inv_vol = 0.5 / VOL(i);

  // The block arithmetic runs on N x N operands: for N = 3 the rows and
  // columns {rho, m1, E} are copied out first, so every entry is the same
  // expression on the same matrix shape as before (on CUDA, FMA contraction
  // follows the expression shape); for N = 5 the copies are the 5x5 blocks.
  int const rows[3] = {IDN, IVX, IPR};
  auto take = [&](Eigen::Matrix<T, 5, 5> const& m) {
    Eigen::Matrix<T, N, N> out;
    if constexpr (N == 5) {
      out = m;
    } else {
      for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s) out(r, s) = m(rows[r], rows[s]);
    }
    return out;
  };
  Eigen::Matrix<T, N, N> AmN = take(Am), ApN = take(Ap), PhiN = take(Phi);
  Eigen::Matrix<T, N, N> dprev = take(dfdq_prev), dcurr = take(dfdq_curr),
                         dnext = take(dfdq_next);
  Eigen::Matrix<T, N, 1> DtN, BndN;
  if constexpr (N == 5) {
    DtN = Dt;
    BndN = Bnd;
  } else {
    for (int r = 0; r < 3; ++r) {
      DtN(r) = Dt(rows[r]);
      BndN(r) = Bnd(rows[r]);
    }
  }
  int const e = N == 5 ? IPR : 2;         // the energy row
  int const m1 = N == 5 ? IVX + dir : 1;  // the x1 momentum column

  // Set up diagonals a, b, c, and the forcing-function Jacobian.
  a[i] = (AmN * area_i + ApN * area_ip1 + (area_ip1 - area_i) * dcurr) *
             half_inv_vol -
         PhiN;
  a[i].diagonal() += DtN;
  b[i] = -(AmN + dprev) * area_i * half_inv_vol;
  c[i] = -(ApN - dnext) * area_ip1 * half_inv_vol;

  vic_gravity_work_rows<T, N>(a[i], b[i], c[i], AmN.row(0), ApN.row(0), e, m1,
                              grav, face_work, diffusive_cell,
                              work_lo[i * stride2], work_hi[i * stride2]);

  // Fix boundary conditions for the cells at the ends of the column.
  bool merged_lower = (i == is || solid_lower) && first_block && !periodic;
  bool merged_upper = (i == ie || solid_upper) && last_block && !periodic;
  if (merged_lower) a[i] += b[i] * BndN.asDiagonal();
  if (merged_upper) a[i] += c[i] * BndN.asDiagonal();

  if constexpr (N == 3) {
    if (du == nullptr || rhs_corr == nullptr) return;
    rhs_corr[i].setZero();

    // the m_t columns (t = dir+1, dir+2) of the same rows rho, m1, E: the
    // same terms as a[i], b[i], c[i] (Phi and the 1/dt diagonal have none
    // there; face work adds only the mass rows)
    int const it[2] = {IVX + (dir + 1) % 3, IVX + (dir + 2) % 3};
    Eigen::Matrix<T, 3, 2> AmT, ApT, dfp_T, dfc_T, dfn_T;
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < 2; ++k) {
        AmT(r, k) = Am(rows[r], it[k]);
        ApT(r, k) = Ap(rows[r], it[k]);
        dfp_T(r, k) = dfdq_prev(rows[r], it[k]);
        dfc_T(r, k) = dfdq_curr(rows[r], it[k]);
        dfn_T(r, k) = dfdq_next(rows[r], it[k]);
      }
    Eigen::Matrix<T, 3, 2> aT, bT, cT;
    aT = (AmT * area_i + ApT * area_ip1 + (area_ip1 - area_i) * dfc_T) *
         half_inv_vol;
    bT = -(AmT + dfp_T) * area_i * half_inv_vol;
    cT = -(ApT - dfn_T) * area_ip1 * half_inv_vol;
    if (face_work || diffusive_cell) {
      T lower = work_lo[i * stride2], upper = work_hi[i * stride2];
      aT.row(2) -= grav * (upper * ApT.row(0) - lower * AmT.row(0));
      bT.row(2) -= grav * lower * AmT.row(0);
      cT.row(2) += grav * upper * ApT.row(0);
    }
    // the wall mirror keeps m_t (Bnd is 1 there): a merged face is cell i's
    if (merged_lower) aT += bT;
    if (merged_upper) aT += cT;

    auto fold_cell = [&](Eigen::Matrix<T, 3, 3>& blk,
                         Eigen::Matrix<T, 3, 2> const& tb, int j) {
      T drho = du[IDN * stride1 + j * stride2];
      for (int n = 0; n < ny; ++n)
        drho += du[(ICY + n) * stride1 + j * stride2];
      for (int k = 0; k < 2; ++k) {
        T v = w[it[k] * stride1 + j * stride2];
        T q = du[it[k] * stride1 + j * stride2] - v * drho;
        if (v != 0) blk.col(0) += tb.col(k) * v;
        if (q != 0) rhs_corr[i] += tb.col(k) * q;
      }
    };
    fold_cell(a[i], aT, i);
    // a neighbour outside the fluid segment keeps the frozen coupling
    if (!merged_lower && !solid_lower && i > is) fold_cell(b[i], bT, i - 1);
    if (!merged_upper && !solid_upper && i < ie) fold_cell(c[i], cT, i + 1);
  }
}

//! the full (5x5) matrix
template <typename T>
void DISPATCH_MACRO vic_assemble_full_impl(
    Eigen::Matrix<T, 5, 5>* a, Eigen::Matrix<T, 5, 5>* b,
    Eigen::Matrix<T, 5, 5>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block, bool periodic,
    bool solid_lower = false, bool solid_upper = false) {
  vic_assemble_impl<T, 5>(a, b, c, w, gamma, area, vol, work_lo, work_hi, i, is,
                          ie, dt, grav, dir, ny, stride1, stride2, first_block,
                          last_block, periodic, solid_lower, solid_upper);
}

//! the partial (3x3) matrix, with the m_t fold when du and rhs_corr are set
template <typename T>
void DISPATCH_MACRO vic_assemble_partial_impl(
    Eigen::Matrix<T, 3, 3>* a, Eigen::Matrix<T, 3, 3>* b,
    Eigen::Matrix<T, 3, 3>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block,
    bool solid_lower = false, bool solid_upper = false, T* du = nullptr,
    Eigen::Matrix<T, 3, 1>* rhs_corr = nullptr) {
  vic_assemble_impl<T, 3>(a, b, c, w, gamma, area, vol, work_lo, work_hi, i, is,
                          ie, dt, grav, dir, ny, stride1, stride2, first_block,
                          last_block, /*periodic=*/false, solid_lower,
                          solid_upper, du, rhs_corr);
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
