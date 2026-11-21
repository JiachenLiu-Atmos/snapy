#include "conversion.hpp"

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>

namespace snap {

torch::Tensor conv4_admissible(torch::Tensor const& a, bool prim,
                               EquationOfStateImpl* eos, bool limiter) {
  double df = limiter ? eos->options->density_floor() : 0.;
  double pf = limiter ? eos->options->pressure_floor() : 0.;
  auto good = torch::isfinite(a).all(0) & (a[IDN] > df);
  if (limiter && a.size(0) > ICY)
    good &= (a.narrow(0, ICY, a.size(0) - ICY) >= 0.).all(0);
  return prim && a.size(0) > IPR ? good & (a[IPR] > pf) : good;
}

Conversion::Conversion(torch::Tensor const& x1f, int is, int iu, int ng,
                       bool const walls[2], int nvar,
                       torch::TensorOptions const& options)
    : key(x1f, options), is_(is), iu_(iu), walls_{walls[0], walls[1]} {
  auto k = conv4_rows(x1f, x1f.size(0) - 1);
  kp_ = k.first.to(options);
  km_ = k.second.to(options);
  if (!any_wall()) return;
  E_ = wall4_rows(x1f, is, iu, ng).to(options);
  pos_ = torch::zeros({nvar, 1, 1, 1}, options.dtype(torch::kBool));
  pos_[IDN].fill_(true), pos_[IPR].fill_(true);
}

void Conversion::fill_walls(torch::Tensor a, int c, bool pos, torch::Tensor ref,
                            torch::Tensor rows, torch::Tensor trips) const {
  for (int s : {0, 1}) {
    if (!walls_[s]) continue;
    int g = E_.size(2);
    auto old = a.narrow(-1, s ? iu_ + 1 : is_ - g, g);
    auto x = a.narrow(-1, s ? iu_ - 3 : is_, 4).matmul(E_[s][c].t());
    if (!pos) {  // no positivity guard: the ghosts are the polynomial's
      old.copy_(x);
      continue;
    }
    auto t = ref.defined() ? x + ref.narrow(-1, s ? iu_ + 1 : is_ - g, g) : x;
    auto bad = rows.defined() ? (t <= 0.) & rows : t <= 0.;
    if (trips.defined()) trips.add_(bad.sum());  // on the device
    old.copy_(torch::where(bad, old, x));
  }
}

// centred everywhere, reading the ghosts the boundary condition, the exchange
// or fill_walls left
torch::Tensor Conversion::lap(torch::Tensor const& a) const {
  int n = kp_.size(0);
  TORCH_CHECK(a.size(-1) == n + 2, "conversion order 4: rows built for ", n + 2,
              " x1 cells, got ", a.size(-1));
  auto out = torch::zeros_like(a);
  auto am = a.narrow(-1, 0, n), a0 = a.narrow(-1, 1, n);
  out.narrow(-1, 1, n) = kp_ * (a.narrow(-1, 2, n) - a0) - km_ * (a0 - am);
  return out;
}

void Conversion::forward(torch::Tensor averages, torch::Tensor const& w,
                         EquationOfStateImpl& eos, bool limiter,
                         torch::Tensor const& trips) const {
  // per cell, keep the second-order w where u - L u or its primitives are
  // not admissible: non-finite, rho or p not above the floor (zero with the
  // limiter off), or repaired by the limiter
  auto ok = [&](torch::Tensor const& a, bool p) {
    return conv4_admissible(a, p, &eos, limiter);
  };
  auto w0 = w.clone();
  // the wall cell's row reads one-sided ghosts, m odd parity (m(0) = 0); W(u)
  // has no wall condition (m/rho = v + O(h^2), not zero at the wall)
  if (any_wall()) {
    averages = averages.clone();  // all rows at once, then m on its slot
    fill_walls(averages, 0, true, {}, pos_, trips);
    fill_walls(averages.narrow(0, IVX, 1), 1, false, {});
    fill_walls(w0, 0, true, {}, pos_, trips);
  }
  auto up = averages - lap(averages);
  auto good = ok(up, false);
  auto tmp = torch::where(good, up, averages);
  auto keep = [](torch::Tensor const& a) {
    return a.defined() ? a.clone() : a;
  };
  auto marks = keep(eos.limiter_marks());
  auto fails = keep(eos.vapor_failures());
  auto tmp0 = limiter ? tmp.clone() : torch::Tensor();
  eos.compute("U->W", {tmp, w});
  if (limiter) {
    good &= (tmp == tmp0).all(0);
    if (marks.defined()) eos.limiter_marks().copy_(marks);
    if (eos.vapor_failures().defined())
      fails.defined() ? eos.vapor_failures().copy_(fails)
                      : eos.vapor_failures().zero_();
  }
  good &= ok(w, true);
  w.add_(lap(w0));
  w.copy_(torch::where(good, w, w0));
  if (trips.defined()) trips.add_(good.logical_not().sum());
}

}  // namespace snap
