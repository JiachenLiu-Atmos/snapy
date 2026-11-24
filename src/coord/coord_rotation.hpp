#pragma once

// torch
#include <torch/torch.h>

// snap
#include "cubed_sphere_utils.hpp"

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

  //! row 16r (S step (4)): the momentum rows of a face flux, covariant in
  //! `panel`'s basis on a face of metric `cosine`, to the global Cartesian
  //! frame (raise, then contra_to_cart_) and back (cart_to_contra_, then
  //! lower). Both panels of a seam face evaluate the same physical points,
  //! so owner -> Cartesian -> receiver joins the two covariant face bases;
  //! rows is [3, ...], (alpha, beta, cosine) broadcast to rows[0]
  void face_rows_to_cart_(torch::Tensor rows, int panel, torch::Tensor alpha,
                          torch::Tensor beta, torch::Tensor cosine) const;
  void cart_to_face_rows_(torch::Tensor rows, int panel, torch::Tensor alpha,
                          torch::Tensor beta, torch::Tensor cosine) const;

  //! the grid-only operands of the two above at one face plane: the metric at
  //! (alpha, beta) and the face cosine, both expanded to rows[0]'s shape
  struct FaceOperands {
    CsMetric metric;
    torch::Tensor cosine;
  };
  static FaceOperands face_operands(std::vector<int64_t> const& shape,
                                    torch::Tensor alpha, torch::Tensor beta,
                                    torch::Tensor cosine);
  //! the same as the two above, from operands built once (the same bits)
  void face_rows_to_cart_(torch::Tensor rows, int panel,
                          FaceOperands const& op) const;
  void cart_to_face_rows_(torch::Tensor rows, int panel,
                          FaceOperands const& op) const;

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
