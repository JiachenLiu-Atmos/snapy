// C/C++
#include <limits>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/mesh/meshblock.hpp>

#include "balance_column.hpp"
#include "hydro.hpp"
#include "hydro_reference.hpp"

namespace snap {

std::tuple<torch::Tensor, double, int> balance_column(
    torch::Tensor const& w, torch::Tensor const& dx1f, double grav,
    bool wall_clamp, double rtol, int max_iter, std::string const& geometry,
    bool stretched, int conversion_order, EquationOfStateImpl* eos) {
  TORCH_CHECK(conversion_order == 2 || conversion_order == 4,
              "balance_column: conversion_order must be 2 or 4, got ",
              conversion_order);
  TORCH_CHECK(grav > 0., "balance_column: grav is a downward magnitude (> 0)");
  TORCH_CHECK(w.dim() == 4, "balance_column: w must be (nvar, nc3, nc2, nx1)");
  TORCH_CHECK(w.size(0) > IPR, "balance_column: w has no pressure channel");
  TORCH_CHECK(dx1f.dim() == 1 && dx1f.size(0) == w.size(-1),
              "balance_column: dx1f must be (nx1,) and match w's last dim");
  TORCH_CHECK(
      dx1f.scalar_type() == w.scalar_type() && dx1f.device() == w.device(),
      "balance_column: dx1f and w must share dtype and device");
  TORCH_CHECK(rtol > 0. && max_iter > 0,
              "balance_column: rtol and max_iter must be positive");
  // without the clamp the reference's wall rows quadrature faces below the
  // wall, which a ghost-free column does not have
  TORCH_CHECK(wall_clamp,
              "balance_column needs dynamics/wb-wall-clamp: without it the "
              "reference reads faces outside the column at each wall, so the "
              "balance found here is not the one the solver would enforce.");
  // no r^2 -> plain-mean conversion here (balance_column.hpp)
  TORCH_CHECK(!x1_centroid_exact_enabled() || geometry == "cartesian",
              "balance_column: under SNAP_X1_CENTROID_EXACT only a column "
              "declared geometry='cartesian' is balanced here, got '",
              geometry,
              "'; a spherical-polar column's rest state is the solver's own "
              "reference");

  int nc1 = w.size(-1);
  // below five the kernel's own thin-block fallback takes over and the answer,
  // though self-consistent, is not the one any real block would enforce; at
  // nc1 == 1 the gauge cell is the only cell and this would return "already
  // balanced" having done nothing
  TORCH_CHECK(nc1 >= 5, "balance_column: a column needs at least 5 cells, got ",
              nc1);
  TORCH_CHECK((w[IPR].min() > 0. & w[IDN].min() > 0.).item<bool>(),
              "balance_column: p and rho must be positive everywhere");
  if (conversion_order == 4) {
    TORCH_CHECK(eos && eos->phydro && eos->phydro->pmb,
                "balance_column: conversion_order 4 needs the run's eos");
    // its W <-> U applies its block's x2/x3 metric, (nc3, nc2) with ghosts
    auto const& co = eos->phydro->pmb->pcoord->options;
    TORCH_CHECK(co->nc2() == 1 && co->nc3() == 1,
                "balance_column: conversion_order 4 needs the eos of a block "
                "with nx2 = nx3 = 1 (a single column), got nx2 = ",
                co->nx2(), ", nx3 = ", co->nx3());
    int n = w.size(-1);
    // a reflecting wall mirrors the end cell and its width, and v1 with odd
    // parity (src/bc/bc_func.cpp, reflecting_inner and reflecting_outer)
    auto pad = [](torch::Tensor const& a) {
      return torch::cat({a.narrow(-1, 0, 1), a, a.narrow(-1, -1, 1)}, -1);
    };
    auto f =
        torch::cat({torch::zeros({1}, dx1f.options()), pad(dx1f).cumsum(0)});
    auto x = [&](int s) { return f.narrow(0, s, n); };
    auto c = (x(2) - x(1)).square() / 4. / (x(3) - x(0));
    auto kp = c / (x(3) - x(1)), km = c / (x(2) - x(0));
    // L a = h^2/24 q'' on the owned cells, hydro_forward.cpp's lap
    auto const& bf = eos->phydro->pmb->options->bfuncs();  // the column's BCs
    bool walls[2] = {n > 3 && is_reflecting(bf[0]),
                     n > 3 && is_reflecting(bf[1])};
    auto E = walls[0] || walls[1] ? wall4_weights(f, 1, n, 1).to(w.options())
                                  : torch::Tensor();  // the solver's ghosts
    auto lap = [&](torch::Tensor const& a) {
      auto a0 = a.narrow(-1, 1, n);
      return kp * (a.narrow(-1, 2, n) - a0) - km * (a0 - a.narrow(-1, 0, n));
    };
    // the limiter works on a block's interior, not on this column, so a cell
    // the solver's limiter would repair keeps w4 here (balance_column.hpp)
    bool lim = eos->options->limiter();
    eos->options->limiter(false);
    std::shared_ptr<void> restore(nullptr,
                                  [&](void*) { eos->options->limiter(lim); });
    auto wb = w.contiguous().clone();
    for (int pass = 0; pass < max_iter; ++pass) {
      auto wp = pad(wb);
      wp[IVX].narrow(-1, 0, 1).neg_(), wp[IVX].narrow(-1, -1, 1).neg_();
      auto u = eos->compute("W->U", {wp});
      for (int c = 0; c < u.size(0); ++c)
        wall4_fill(u[c], E, c == IVX, 1, n, walls, c == IDN || c == IPR, {});
      auto up = (u.narrow(-1, 1, n) - lap(u)).contiguous();
      // the solver's fallback: W(u) where u - L u or its W is not admissible
      auto good = conv4_admissible(up, false, eos, lim);
      auto w4 = eos->compute(
          "U->W", {torch::where(good, up, u.narrow(-1, 1, n)).contiguous()});
      good &= conv4_admissible(w4, true, eos, lim);
      auto w2 = eos->compute("U->W", {u.clone()});
      for (int c = 0; c < w2.size(0); ++c)
        wall4_fill(w2[c], E, 0, 1, n, walls, c == IDN || c == IPR, {});
      w4 = torch::where(good, w4 + lap(w2), w2.narrow(-1, 1, n));
      auto [b, err, sweeps] = balance_column(w4, dx1f, grav, wall_clamp, rtol,
                                             max_iter, geometry, stretched);
      if (sweeps == 0) return {wb, err, pass};
      wb[IDN] += b[IDN] - w4[IDN];
      wb[IPR] += b[IPR] - w4[IPR];
    }
    TORCH_CHECK(false, "balance_column: conversion_order 4 not balanced after ",
                max_iter, " passes");
  }
  auto wb = w.contiguous().clone();
  auto dxf = dx1f.contiguous();
  auto rho = wb[IDN];
  auto prs = wb[IPR];
  auto rt = (prs / rho).clone();  // p/rho per cell, the invariant, taken ONCE

  HydroReferenceX1 reference(w);
  auto& pref = reference.pref;
  HydroReferenceX1Operator ref_operator(dxf, nc1 - 1, stretched,
                                        /*phys_in=*/true, /*phys_out=*/true,
                                        wall_clamp, kX1RefStretched);
  torch::Tensor anchor;  // undefined: the kernel builds its own top anchor

  auto wgt = grav * dxf;

  // check, then update: the residual therefore describes the state returned,
  // and a column already at the fixed point comes back after zero updates
  double err = std::numeric_limits<double>::infinity();
  int sweeps = 0;
  for (; sweeps <= max_iter; ++sweeps) {
    ref_operator.evaluate(wb, grav, anchor, reference);
    auto pp = prs - pref;
    auto c = pp.narrow(-1, nc1 - 1, 1);  // one gauge per column, at its top
    err = ((pp - c).abs() / (rho * wgt)).max().item<double>();
    if (err < rtol || sweeps == max_iter) break;
    prs.copy_(pref + c);
    rho.copy_(prs / rt);
    // a diverging column walks its top cells to vacuum, and once rho goes
    // negative the residual above goes negative with it and reads as converged
    TORCH_CHECK(
        prs.min().item<double>() > 0.,
        "balance_column: the projection drove the pressure non-positive "
        "on sweep ",
        sweeps, " -- the column is diverging, not converging.");
  }

  TORCH_CHECK(err < rtol, "balance_column: no discrete balance after ", sweeps,
              " sweeps; residual bound on |a|/g is ", err, " > ", rtol);
  return {wb, err, sweeps};
}

}  // namespace snap
