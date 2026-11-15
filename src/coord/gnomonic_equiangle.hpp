#pragma once

// snap
#include "coordinate.hpp"

namespace snap {

class GnomonicEquiangleImpl
    : public torch::nn::Cloneable<GnomonicEquiangleImpl>,
      public CoordinateImpl {
 public:
  // geometry data
  torch::Tensor sine_cell_kj, sine_face2_kj, sine_face3_kj;
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

  void axial_x1_source_(torch::Tensor row2, torch::Tensor row3,
                        torch::Tensor const& corr) const override;

  //! axial angular momentum in flux form (#314)
  /*!
   * The momentum rows of `div` are the face-flux divergence minus the
   * geometric (metric) sources, both at cell centres. Contracting them with
   * the rotation field K = r k of each cell does not telescope across faces:
   * the cell-centred metric sources and the face fluxes miss the discrete
   * identity K . (div F - S) = div (F . K) by a second-order error, a
   * spurious torque. Here the K component of every interior cell's momentum
   * rows is replaced by the divergence of the face AAM fluxes A F . K_f, with
   * K_f taken at each face in the face's own basis, so the AAM leaving a cell
   * through a face is the AAM entering its neighbour and the total changes
   * only through the physical boundary faces. The component of `div`
   * orthogonal to K is unchanged. A cell with K -> 0 (on the axis) holds no
   * AAM; its correction is regularised and fades there.
   */
  void conserve_axial_momentum_(torch::Tensor div, torch::Tensor flux1,
                                torch::Tensor flux2,
                                torch::Tensor flux3) const override;

  //! set the K component of a pair of covariant horizontal rows (full-shape
  //! [nc3, nc2, nc1] views of the x2 and x3 rows of a rate) to `target`, the
  //! AAM rate per volume on the interior; the component orthogonal to K is
  //! unchanged. A cell centred on the axis passes its target to its four
  //! lateral neighbours (see the source).
  void axial_rate_to_rows_(torch::Tensor row2, torch::Tensor row3,
                           torch::Tensor target) const override;

 private:
  //! guards K . K = 0 in a cell centred exactly on the rotation axis
  double _aam_eps2 = 0.;

  torch::Tensor _interp_ghost_LR(torch::Tensor buf, bool flip) const;
  torch::Tensor _interp_ghost_BT(torch::Tensor buf, bool flip) const;
  //! linear interpolation of a received strip along the edge (dim -3 or -2)
  torch::Tensor _interp_ghost(torch::Tensor buf, torch::Tensor usrc_t,
                              int shift, int dim) const;

  //! set metric terms at face 2
  void _set_face2_metric() const;

  //! set metric terms at face 3
  void _set_face3_metric() const;
};
TORCH_MODULE(GnomonicEquiangle);

}  // namespace snap
