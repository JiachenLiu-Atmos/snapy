#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

namespace snap {

//! The x1 gravity work in the energy row of cell i's blocks a, b, c, written
//! once for both VIC matrices (partial 3x3, full 5x5). e is the energy row, m
//! the x1 momentum column, am / ap the mass rows of |A| at the lower / upper
//! face.
//! The face work of the linearised mass flux with lower / upper, the
//! face-to-centre potential weights of the mass update, which are
//! GravityWork's face weights (0.25 each on a uniform Cartesian grid); always
//! booked (V3 I-8: no kVicFaceWork switch; with grav 0 the rows add zeros).
//! lower and upper are taken by value: as local copies, the CUDA build
//! rounds these rows as the two copies this replaces did.
template <typename T, int N>
DISPATCH_MACRO void vic_gravity_work_rows(Eigen::Matrix<T, N, N>& a,
                                          Eigen::Matrix<T, N, N>& b,
                                          Eigen::Matrix<T, N, N>& c,
                                          Eigen::Matrix<T, 1, N> const& am,
                                          Eigen::Matrix<T, 1, N> const& ap,
                                          int e, int m, double grav, T lower,
                                          T upper) {
  // The same face-to-centre potential weights as the mass update.
  a.row(e) -= grav * (upper * ap - lower * am);
  b.row(e) -= grav * lower * am;
  c.row(e) += grav * upper * ap;
  a(e, m) += grav * (1. - lower - upper);
  b(e, m) -= grav * lower;
  c(e, m) -= grav * upper;
}

}  // namespace snap
