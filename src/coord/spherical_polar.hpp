#pragma once

// snap
#include "coordinate.hpp"
#include "x1_centroid.hpp"

namespace snap {

class SphericalPolarImpl : public torch::nn::Cloneable<SphericalPolarImpl>,
                           public CoordinateImpl {
 public:
  torch::Tensor coord_src1_i, coord_src1_j;
  torch::Tensor coord_src2_i, coord_src2_j;
  torch::Tensor coord_src3_j;

  SphericalPolarImpl() = default;
  explicit SphericalPolarImpl(const CoordinateOptions& options_,
                              torch::nn::Module* p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream& stream) const override {
    stream << "SphericalPolar coordinate:" << std::endl;
    print(stream);
  }

  torch::Tensor center_width2() const override;
  torch::Tensor center_width3() const override;
  torch::Tensor center_distance2() const override;
  torch::Tensor center_distance3() const override;
  torch::Tensor face_area1() const override;
  torch::Tensor face_area2() const override;
  torch::Tensor face_area3() const override;
  torch::Tensor cell_volume() const override;
  torch::Tensor face_moment2_x1() const override;
  torch::Tensor face_centroid_shift_x1() const override;
  bool face_centroid_shift_zero() const override { return false; }

  torch::Tensor forward(
      torch::Tensor prim, torch::Tensor flux1, torch::Tensor flux2,
      torch::Tensor flux3,
      torch::Tensor face_pressure1 = torch::Tensor()) override;

 private:
  //! SNAP_X1_CENTROID_EXACT: the (2/V) int r p~ dr weights, built on first use
  mutable std::shared_ptr<X1PressureSourceStencils> x1src_;
};
TORCH_MODULE(SphericalPolar);

}  // namespace snap
