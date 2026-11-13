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
//!   face_work && cartesian_face: the face work of the linearised mass flux
//!     with midpoint face potentials;
//!   face_work: the same with lower / upper, the face-to-centre potential
//!     weights of the mass update;
//!   diffusive_cell: the work of the Roe mass diffusion only (the cell work
//!     stays in Phi).
//! lower and upper are taken by value: as local copies, the CUDA build
//! rounds these rows as the two copies this replaces did.
template <typename T, int N>
DISPATCH_MACRO void vic_gravity_work_rows(
    Eigen::Matrix<T, N, N>& a, Eigen::Matrix<T, N, N>& b,
    Eigen::Matrix<T, N, N>& c, Eigen::Matrix<T, 1, N> const& am,
    Eigen::Matrix<T, 1, N> const& ap, int e, int m, double grav, bool face_work,
    bool diffusive_cell, bool cartesian_face, T lower, T upper) {
  if (face_work && cartesian_face) {
    Eigen::Matrix<T, 1, N> em;
    em.setZero();
    em(m) = 1.;
    a(e, m) += grav;
    a.row(e) -= 0.5 * grav * (em + 0.5 * (ap - am));
    b.row(e) -= 0.5 * grav * (0.5 * em + 0.5 * am);
    c.row(e) -= 0.5 * grav * (0.5 * em - 0.5 * ap);
  } else if (face_work || diffusive_cell) {
    // The same face-to-centre potential weights as the mass update.
    a.row(e) -= grav * (upper * ap - lower * am);
    b.row(e) -= grav * lower * am;
    c.row(e) += grav * upper * ap;
    if (face_work) {
      a(e, m) += grav * (1. - lower - upper);
      b(e, m) -= grav * lower;
      c(e, m) -= grav * upper;
    }
  }
}

}  // namespace snap
