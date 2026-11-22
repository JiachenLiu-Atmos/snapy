#pragma once

// snap
#include "coordinate.hpp"

namespace snap {

class GnomonicEquiangleImpl
    : public torch::nn::Cloneable<GnomonicEquiangleImpl>,
      public CoordinateImpl {
 public:
  // geometry data
  torch::Tensor x_ov_rD_kji, y_ov_rC_kji;
  torch::Tensor dx2f_ang_kj, dx3f_ang_kj;
  torch::Tensor dx2f_ang_face3_kj, dx3f_ang_face2_kj;
  torch::Tensor solid_angle_kj;

  // metric data
  torch::Tensor g11, g22, g33, gi11, gi22, gi33, g12, g13, g23;

  // axial angular momentum (AAM) about the global z axis, the axis of the
  // Coriolis option omega1. k = z_hat x n is the unit-radius rotation field
  // (n the unit position); at radius r the AAM density of momentum m is
  // r k . m. Components are in the unit (alpha, beta) basis of each location,
  // [3, nk, nj, 1] with the radial one zero:
  //   aam_k_c, aam_kl_c  contravariant and covariant k at the cell centres
  //   aam_kk_c           k . k = cos^2(lat) at the cell centres, [nk, nj, 1]
  //   aam_k_f2, aam_k_f3 the exact face average of k on the x2 faces (x2f,
  //                      x3v) and on the x3 faces (x2v, x3f), contravariant
  //                      in each face's own basis, the basis of its covariant
  //                      face fluxes
  torch::Tensor aam_k_c, aam_kl_c, aam_kk_c, aam_k_f2, aam_k_f3;
  //   aam_c2_f2, aam_c2_f3 cos^2(lat) at those faces, [nk, nj(+1), 1]
  torch::Tensor aam_c2_f2, aam_c2_f3;

  // local ghost cell usrc -- GLOBAL source coordinates, NOT rebased onto this
  // block. The block offset is applied to the integer index instead; see
  // _interp_ghost_BT.
  torch::Tensor usrc_LR, usrc_BT;

  //! this block's index shift for usrc, in whole cells (margin - offset)
  int usrc_shift_BT = 0, usrc_shift_LR = 0;

  GnomonicEquiangleImpl() = default;
  explicit GnomonicEquiangleImpl(const CoordinateOptions& options_,
                                 torch::nn::Module* p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream& stream) const override {
    stream << "GnomonicEquiangle coordinate:" << std::endl;
    print(stream);
  }

  torch::Tensor center_width2() const override;
  torch::Tensor center_width3() const override;

  torch::Tensor face_area1() const override;
  torch::Tensor face_area2() const override;
  torch::Tensor face_area3() const override;

  torch::Tensor cell_volume() const override;
  torch::Tensor face_moment2_x1() const override;
  torch::Tensor face_centroid_shift_x1() const override;
  bool face_centroid_shift_zero() const override { return false; }

  void interp_ghost(torch::Tensor var,
                    std::tuple<int, int, int> const& offset) const override;

  void prim2local1_(torch::Tensor const& wlr) const override;
  void prim2local2_(torch::Tensor const& wlr) const override;
  void prim2local3_(torch::Tensor const& wlr) const override;

  void flux2global1_(torch::Tensor const& flux) const override;
  void flux2global2_(torch::Tensor const& flux) const override;
  void flux2global3_(torch::Tensor const& flux) const override;

  torch::Tensor forward(
      torch::Tensor prim, torch::Tensor flux1, torch::Tensor flux2,
      torch::Tensor flux3,
      torch::Tensor face_pressure1 = torch::Tensor()) override;
  void x1_mass_source_(torch::Tensor du, torch::Tensor const& prim,
                       torch::Tensor const& m1, double dt) const override;

 private:
  torch::Tensor _interp_ghost_LR(torch::Tensor buf, bool flip) const;
  torch::Tensor _interp_ghost_BT(torch::Tensor buf, bool flip) const;
  //! global cells along a panel edge (nx2 * px), for the cubic ghost rule
  int usrc_ncell = 0;
  //! interpolation of a received strip along the edge (dim -3 or -2): linear
  //! (interp_order 2) or 4-point Lagrange (interp_order 4)
  torch::Tensor _interp_ghost(torch::Tensor buf, torch::Tensor usrc_t,
                              int shift, int dim) const;

  //! set metric terms at face 2
  void _set_face2_metric() const;

  //! set metric terms at face 3
  void _set_face3_metric() const;
};
TORCH_MODULE(GnomonicEquiangle);

}  // namespace snap
