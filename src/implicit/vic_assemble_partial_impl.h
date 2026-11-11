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

// Assemble the block-tridiagonal coefficients a[i], b[i], c[i] for a SINGLE
// vertical cell `i` of one column. This is the per-cell-parallel form of the
// assembly loop in vic_solve_partial_impl(): every cell is independent, so one
// GPU thread can own one cell. Shared interface Jacobians are recomputed from
// the same inputs, so the result matches the serial assembly. Kept in lockstep
// with vic_solve_partial_impl() so the GPU (split) and CPU (fused) paths agree.
//
// a, b, c point to the start of the column's coefficient arrays (indexed [i]),
// matching the layout used by ForwardSweep and the split redistribution
// helpers.
//
// With du and rhs_corr set, the tangential momenta m_t (not unknowns of the
// partial matrix) are no longer held fixed: the increment of cell j is
// written dm_t,j = v_t,j drho_j + q_t,j, so each block's rho column gains its
// m_t columns times the velocity of the cell it multiplies (the |A| and flux
// Jacobian columns alike), and q_t,j = du_t,j - v_t,j du_rho,j (explicit
// increments) moves to the right-hand side:
//   rhs_corr[i] = aT q_i + bT q_{i-1} + cT q_{i+1}   (rows rho, m1, E)
// ForwardSweep subtracts it. A face merged into a[i] by the wall mirror (or a
// solid neighbour) carries cell i's v_t and q_t, as the mirror does. Exact
// zeros (rest state) leave a, b, c and the right-hand side unchanged bit for
// bit. Without du / rhs_corr the assembly is the frozen-m_t one.
template <typename T>
void DISPATCH_MACRO vic_assemble_partial_impl(
    Eigen::Matrix<T, 3, 3>* a, Eigen::Matrix<T, 3, 3>* b,
    Eigen::Matrix<T, 3, 3>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block,
    bool solid_lower = false, bool solid_upper = false, T* du = nullptr,
    Eigen::Matrix<T, 3, 1>* rhs_corr = nullptr) {
  bool face_work = dir & kVicFaceWork;
  bool diffusive_cell = dir & kVicDiffusiveCell;
  bool cartesian_face = dir & kVicCartesianFaceWork;
  dir &= ~(kVicFaceWork | kVicDiffusiveCell | kVicCartesianFaceWork);
  Eigen::Matrix<T, 5, 5> Rmat, Rimat, Am, Ap, dfdqf;
  Eigen::Matrix<T, 5, 1> Lambda;
  Eigen::Matrix<T, 3, 3> Am2, Ap2, dfdq_prev, dfdq_curr, dfdq_next;
  // the m_t columns (t = dir+1, dir+2) of the same rows rho, m1, E
  bool fold = du != nullptr && rhs_corr != nullptr;
  int const it[2] = {IVX + (dir + 1) % 3, IVX + (dir + 2) % 3};
  Eigen::Matrix<T, 3, 2> AmT, ApT, dfp_T, dfc_T, dfn_T;
  auto tcols = [&](Eigen::Matrix<T, 3, 2>& out,
                   Eigen::Matrix<T, 5, 5> const& m) {
    int const rows[3] = {IDN, IVX, IPR};
    for (int r = 0; r < 3; ++r)
      for (int k = 0; k < 2; ++k) out(r, k) = m(rows[r], it[k]);
  };

  Eigen::Matrix<T, 3, 3> Phi;
  Eigen::Matrix<T, 3, 1> Dt, Bnd;
  Phi << 0., 0., 0.,  //
      grav, 0., 0.,   //
      0., grav, 0.;
  Dt << 1. / dt, 1. / dt, 1. / dt;
  Bnd << 1., -1, 1.;

  T prim[5];       // Roe averaged primitive variables at an interface
  T wl[5], wr[5];  // left/right primitive variables
  T gm1, cs;

  // ---- interface i-1/2 (cells i-1, i); also dfdq at cells i-1 and i ----
  CopyPrimitives(wl, wr, w, i, stride1, stride2,
                 ny);  // wl = cell i-1, wr = cell i

  if (solid_lower) {
    for (int n = 0; n < 5; ++n) wl[n] = wr[n];
    wl[IVX + dir] = -wr[IVX + dir];
  }
  gm1 = GAMMA(solid_lower ? i : i - 1) - 1.;
  FluxJacobian(dfdqf, gm1, wl, dir);
  if (fold) tcols(dfp_T, dfdqf);
  dfdq_prev << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = GAMMA(i) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  if (fold) tcols(dfc_T, dfdqf);
  dfdq_curr << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(solid_lower ? i : i - 1) + GAMMA(i)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Am.noalias() = Rmat * Lambda.asDiagonal() * Rimat;
  if (fold) tcols(AmT, Am);
  Am2 << Am(IDN, IDN), Am(IDN, IVX), Am(IDN, IPR),  //
      Am(IVX, IDN), Am(IVX, IVX), Am(IVX, IPR),     //
      Am(IPR, IDN), Am(IPR, IVX), Am(IPR, IPR);

  // ---- interface i+1/2 (cells i, i+1); also dfdq at cell i+1 ----
  CopyPrimitives(wl, wr, w, i + 1, stride1, stride2,
                 ny);  // wl = cell i, wr = cell i+1

  if (solid_upper) {
    for (int n = 0; n < 5; ++n) wr[n] = wl[n];
    wr[IVX + dir] = -wl[IVX + dir];
  }
  gm1 = GAMMA(solid_upper ? i : i + 1) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  if (fold) tcols(dfn_T, dfdqf);
  dfdq_next << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(i) + GAMMA(solid_upper ? i : i + 1)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Ap.noalias() = Rmat * Lambda.asDiagonal() * Rimat;
  if (fold) tcols(ApT, Ap);
  Ap2 << Ap(IDN, IDN), Ap(IDN, IVX), Ap(IDN, IPR),  //
      Ap(IVX, IDN), Ap(IVX, IVX), Ap(IVX, IPR),     //
      Ap(IPR, IDN), Ap(IPR, IVX), Ap(IPR, IPR);

  // ---- set up diagonals a, b, c (matches vic_solve_partial_impl) ----
  T const& area_i = AREA(i);
  T const& area_ip1 = AREA(i + 1);
  T half_inv_vol = 0.5 / VOL(i);

  a[i] = (Am2 * area_i + Ap2 * area_ip1 + (area_ip1 - area_i) * dfdq_curr) *
             half_inv_vol -
         Phi;
  a[i].diagonal() += Dt;
  b[i] = -(Am2 + dfdq_prev) * area_i * half_inv_vol;
  c[i] = -(Ap2 - dfdq_next) * area_ip1 * half_inv_vol;

  vic_gravity_work_rows<T, 3>(a[i], b[i], c[i], Am2.row(0), Ap2.row(0), 2, 1,
                              grav, face_work, diffusive_cell, cartesian_face,
                              work_lo[i * stride2], work_hi[i * stride2]);

  // ---- boundary condition. Bnd = diag(1, -1, 1). ----
  if ((i == is || solid_lower) && first_block) a[i] += b[i] * Bnd.asDiagonal();
  if ((i == ie || solid_upper) && last_block) a[i] += c[i] * Bnd.asDiagonal();

  if (!fold) return;
  rhs_corr[i].setZero();

  // the same terms as a[i], b[i], c[i] above, for the m_t columns (Phi and
  // the 1/dt diagonal have none there; face work adds only the mass rows)
  Eigen::Matrix<T, 3, 2> aT, bT, cT;
  aT = (AmT * area_i + ApT * area_ip1 + (area_ip1 - area_i) * dfc_T) *
       half_inv_vol;
  bT = -(AmT + dfp_T) * area_i * half_inv_vol;
  cT = -(ApT - dfn_T) * area_ip1 * half_inv_vol;
  if (face_work && cartesian_face) {
    aT.row(2) -= 0.25 * grav * (ApT.row(0) - AmT.row(0));
    bT.row(2) -= 0.25 * grav * AmT.row(0);
    cT.row(2) += 0.25 * grav * ApT.row(0);
  } else if (face_work || diffusive_cell) {
    T lower = work_lo[i * stride2], upper = work_hi[i * stride2];
    aT.row(2) -= grav * (upper * ApT.row(0) - lower * AmT.row(0));
    bT.row(2) -= grav * lower * AmT.row(0);
    cT.row(2) += grav * upper * ApT.row(0);
  }
  // the wall mirror keeps m_t (Bnd is 1 there): a merged face is cell i's
  bool merged_lower = (i == is || solid_lower) && first_block;
  bool merged_upper = (i == ie || solid_upper) && last_block;
  if (merged_lower) aT += bT;
  if (merged_upper) aT += cT;

  auto fold_cell = [&](Eigen::Matrix<T, 3, 3>& blk,
                       Eigen::Matrix<T, 3, 2> const& tb, int j) {
    T drho = du[IDN * stride1 + j * stride2];
    for (int n = 0; n < ny; ++n) drho += du[(ICY + n) * stride1 + j * stride2];
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

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
