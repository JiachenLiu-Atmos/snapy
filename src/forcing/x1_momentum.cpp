// snap
#include "x1_momentum.hpp"

#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {

X1Momentum::X1Momentum(HydroImpl const* hydro) : hydro_(hydro) {
  auto pc = hydro->pmb ? hydro->pmb->pcoord : nullptr;
  if (!pc || !hydro->peos) return;
  auto ptype = pc->options->type();
  applies_ = hydro->peos->options->type() != "shallow-water" &&
             pc->options->nc1() > 1 &&
             (ptype == "gnomonic-equiangle" ||
              (ptype == "spherical-polar" && hydro->options->coriolis()));
}

torch::Tensor const& X1Momentum::before_solve(torch::Tensor const& w) {
  auto pc = hydro_->pmb->pcoord;
  auto shape = w[IDN].sizes().vec();
  if (!(x1f_.defined() && x1f_.is_same(pc->x1f) && ver_ == pc->x1f._version() &&
        wc1_.device() == w.device() && wc1_.scalar_type() == w.scalar_type() &&
        wc1_[IDN].sizes() == c10::IntArrayRef(shape))) {
    int is = pc->il(), ie = pc->iu() + 1;
    auto r = pc->x1v.slice(0, is, ie);
    x1f_ = pc->x1f;
    ver_ = pc->x1f._version();
    area_ = pc->face_area1();
    auto den = 2. * r * pc->cell_volume().slice(-1, is, ie);
    lo_ = (r.square() - pc->x1f.slice(0, is, ie).square()) / den;
    hi_ = (pc->x1f.slice(0, is + 1, ie + 1).square() - r.square()) / den;
    phi_ = torch::empty_like(w[IDN]);
    // only rows IDN..IVZ are read by the Coriolis forcings
    wc1_ = torch::zeros({IVZ + 1, shape[0], shape[1], shape[2]}, w.options());
    wc2_ = torch::zeros_like(wc1_);
    wc2_[IDN].fill_(1.);
  }
  // Coriolis sees w = 0 here; rows IDN, IVY, IVZ are w's, row IVX stays 0
  wc1_[IDN].copy_(w[IDN]);
  wc1_.narrow(0, IVY, 2).copy_(w.narrow(0, IVY, 2));
  return wc1_;
}

torch::Tensor const& X1Momentum::after_solve(torch::Tensor const& flux1, int ny,
                                             torch::Tensor const& mc,
                                             StepDt dt) {
  auto pc = hydro_->pmb->pcoord;
  int is = pc->il(), ie = pc->iu() + 1;
  // phi: mass transfer rate through face i (below i), explicit flux plus
  // the implicit transfer per unit time
  // V3 I-4: A_f total(1) on the mass rows; total(1) holds the solve's
  // posts by now (G-R11), so no separate mc / dt term
  if (flux1.defined()) {
    // no species rows: the dry flux is the whole mass flux
    auto f1 =
        ny > 0 ? flux1[IDN] + flux1.narrow(0, ICY, ny).sum(0) : flux1[IDN];
    torch::mul_out(phi_, area_.narrow(-1, 0, f1.size(-1)), f1);
  } else {  // the x1 flux is off: the solve posts nothing (hydro.cpp)
    phi_.zero_();
    // the x1 flux is off only with an implicit solve, which captures since
    // V3 C4: a device dt's 1/dt is formed on the device (the same FP64
    // quotient) and added as add's alpha (axpy_), so no host dt is read
    if (mc.defined() && mc.numel() > 0) {
      StepDt rate = dt.on_device() ? StepDt::device(dt.tensor().reciprocal())
                                   : StepDt(1. / dt.value());
      axpy_(phi_, mc[IVZ], rate);
    }
  }
  // m1 lives in row IVX of the Coriolis input wc2 = (1, m1, 0, 0), x1
  // ghosts stay 0: only the rows driven by rho w are added here, the
  // w-free rows went in before the implicit solve
  auto m1i = wc2_[IVX].slice(-1, is, ie);
  torch::mul_out(m1i, phi_.slice(-1, is, ie), lo_);
  m1i.addcmul_(phi_.slice(-1, is + 1, ie + 1), hi_);
  return wc2_;
}

torch::Tensor X1Momentum::m1() const { return wc2_[IVX]; }

}  // namespace snap
