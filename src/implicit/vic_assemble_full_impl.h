#pragma once

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

template <typename T>
void DISPATCH_MACRO vic_assemble_full_impl(
    Eigen::Matrix<T, 5, 5>* a, Eigen::Matrix<T, 5, 5>* b,
    Eigen::Matrix<T, 5, 5>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block, bool periodic,
    bool solid_lower = false, bool solid_upper = false) {
  bool face_work = dir & kVicFaceWork;
  bool diffusive_cell = dir & kVicDiffusiveCell;
  bool cartesian_face = dir & kVicCartesianFaceWork;
  dir &= ~(kVicFaceWork | kVicDiffusiveCell | kVicCartesianFaceWork);
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

  // Set up diagonals a, b, c, and the forcing-function Jacobian.
  a[i] = (Am * area_i + Ap * area_ip1 + (area_ip1 - area_i) * dfdq_curr) *
             half_inv_vol -
         Phi;
  a[i].diagonal() += Dt;
  b[i] = -(Am + dfdq_prev) * area_i * half_inv_vol;
  c[i] = -(Ap - dfdq_next) * area_ip1 * half_inv_vol;

  vic_gravity_work_rows<T, 5>(a[i], b[i], c[i], Am.row(IDN), Ap.row(IDN), IPR,
                              IVX + dir, grav, face_work, diffusive_cell,
                              cartesian_face, work_lo[i * stride2],
                              work_hi[i * stride2]);

  // Fix boundary conditions for the cells at the ends of the column.
  if ((i == is || solid_lower) && first_block && !periodic)
    a[i] += b[i] * Bnd.asDiagonal();
  if ((i == ie || solid_upper) && last_block && !periodic)
    a[i] += c[i] * Bnd.asDiagonal();
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
