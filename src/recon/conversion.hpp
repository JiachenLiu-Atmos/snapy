#pragma once

// torch
#include <torch/torch.h>

// snap
#include "x1_rows.hpp"

namespace snap {

class EquationOfStateImpl;

//! dynamics/conversion-order 4's per-cell fallback test: true where `a` is
//! admissible -- finite, rho above the density floor and, with `limiter` on,
//! no species below zero (u - L u, prim false), and p above the pressure
//! floor too (its primitives, prim true). The floors are zero with `limiter`
//! off.
torch::Tensor conv4_admissible(torch::Tensor const& a, bool prim,
                               EquationOfStateImpl* eos, bool limiter);

//! Cell averages to primitives along x1 at conversion order 4, the one stage
//! of the solver (HydroImpl::forward) and of balance_column, so that the key-4
//! rest state balance_column finds is the solver's own fixed point:
//! w = W(u - L u) + L W(u), fourth order where W(u) is second, and W(u) in a
//! cell where u - L u or its primitives are not admissible. At a reflecting
//! x1 wall the ghosts come from the owned cells (wall4_rows).
class Conversion {
 public:
  Conversion() = default;
  //! x1f: the nc1 + 1 faces of the arrays converted; is, iu: the first and
  //! last owned cell; ng: ghost cells per side; walls: the reflecting x1 walls
  //! that get key-4 ghosts; nvar: rows of the state; options: its dtype and
  //! device
  Conversion(torch::Tensor const& x1f, int is, int iu, int ng,
             bool const walls[2], int nvar,
             torch::TensorOptions const& options);

  //! the x1 rows' cache key at construction
  X1RowsKey key;

  bool wall(int s) const { return walls_[s]; }
  bool any_wall() const { return walls_[0] || walls_[1]; }

  //! the ghosts of `a` at the flagged walls from E[side][c] . (4 owned
  //! cells), c the wall4_rows constraint; with pos, a ghost whose total
  //! (ref + ghost) is not positive keeps the mirror (only on the `rows` mask
  //! if given) and is added to `trips`
  void fill_walls(torch::Tensor a, int c, bool pos, torch::Tensor ref,
                  torch::Tensor rows = {}, torch::Tensor trips = {}) const;

  //! L a = h^2/24 q'' on cells 1..nc1-2, zero on the two end cells; reads the
  //! end cells as they are
  torch::Tensor lap(torch::Tensor const& a) const;

  //! averages: the total cell averages, ghosts filled; w: W(u) in, the
  //! converted primitives out (ghosts included, for the caller to refill).
  //! The fallback cells hand u to the EOS; that call's limiter marks and vapor
  //! failures are dropped. trips (if defined) counts the fallback cells and
  //! the guarded wall ghosts
  void forward(torch::Tensor averages, torch::Tensor const& w,
               EquationOfStateImpl& eos, bool limiter,
               torch::Tensor const& trips) const;

 private:
  int is_ = 0, iu_ = -1;
  bool walls_[2] = {false, false};
  //! {kp, km} of L; the wall ghost weights; the rows with a positivity guard
  torch::Tensor kp_, km_, E_, pos_;
};

}  // namespace snap
