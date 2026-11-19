#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! The one rotation between cubed-sphere panels (V3 lane S, step (1); S.rst
//! rev 2.4 section 3.1, "One rotation"). A vector goes from one panel's
//! contravariant components through the global Cartesian frame to the
//! components every panel shares, the local geographic frame (up, south,
//! east) at the point. That frame is built from the point's Cartesian position
//! alone, with no atan2, so a matrix is a function of the point alone (H16,
//! gate T1). Exchanged ghost values are carried, and interpolated, in that
//! frame: a field that is uniform in it (a radial wind, a solid-body wind
//! along the parallels) stays uniform across a seam, which Cartesian
//! components would not (they vary with the point).
class PanelRotation {
 public:
  //! panel contravariant components -> global Cartesian, in place; vel is
  //! [3, ...] (radial, x2, x3) at the points (alpha, beta) of `panel`
  void contra_to_cart_(torch::Tensor vel, int panel, torch::Tensor alpha,
                       torch::Tensor beta) const;
  //! global Cartesian -> panel contravariant components, in place
  void cart_to_contra_(torch::Tensor vel, int panel, torch::Tensor alpha,
                       torch::Tensor beta) const;

  //! the pointwise [3, 3, ...] matrix the exchange caches: to_global maps the
  //! panel's components to the local geographic frame (v_r, v_theta, v_phi),
  //! else that frame to the panel's. With
  //! cosine defined (conserved momentum), the metric raise before (to_global)
  //! or the lower after (from global) is composed into the matrix
  torch::Tensor matrix(int panel, torch::Tensor alpha, torch::Tensor beta,
                       bool to_global,
                       torch::Tensor cosine = torch::Tensor()) const;
};

}  // namespace snap
