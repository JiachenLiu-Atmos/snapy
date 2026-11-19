#include "wb_ref4.hpp"

// snap
#include <snap/snap.h>

#include <snap/coord/x1_stencil.hpp>

namespace snap {

namespace {

// relative margin of the range guards: on a flat profile the two bounds
// coincide, and an exact test would flip on round-off between a split and a
// one-block column
constexpr double kMargin = 1.e-10;
// |ln(p_lo/p_hi)| above this: fewer than two cells per pressure scale height
constexpr double kMaxLogDrop = 0.5;

torch::Tensor in_range(torch::Tensor const& v, torch::Tensor const& lo,
                       torch::Tensor const& hi) {
  return (v >= lo - kMargin * lo.abs()) & (v <= hi + kMargin * hi.abs());
}

}  // namespace

torch::Tensor wb_ref4_flag(FaceDensityRows const& st,
                           torch::Tensor const& psf_lo,
                           torch::Tensor const& psf_hi) {
  if (!st.usable) return torch::ones_like(psf_lo, torch::kBool);
  // resolution flag, dilated by two cells
  auto bad = ((psf_lo / psf_hi).log().abs() > kMaxLogDrop) & st.counted;
  auto sizes = bad.sizes();
  return torch::max_pool1d(bad.to(psf_lo.dtype()).reshape({-1, 1, st.nc1}), 5,
                           1, 2)
             .reshape(sizes) > 0.;
}

void wb_ref4_faces(FaceDensityRows const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag) {
  if (!st.usable) return;
  int nf = st.iu - st.is + 2;
  auto face = x1_stencil_apply(dref, st.idx, st.wt);
  auto dl = dref.narrow(-1, st.is - 1, nf), dr = dref.narrow(-1, st.is, nf);
  auto ok = in_range(face, torch::minimum(dl, dr), torch::maximum(dl, dr)) &
            ~flag.narrow(-1, st.is - 1, nf) & ~flag.narrow(-1, st.is, nf);
  auto d = dsf.narrow(-1, st.is, nf);
  d.copy_(torch::where(ok, face, d));
}

}  // namespace snap
