// snap
#include <snap/hydro/hydro.hpp>

#include "coordinate.hpp"

namespace snap {

void CartesianImpl::reset() {
  auto const &op = options;

  // metric
  auto [cc, c2, c3] = zero_cosines_();
  cosine_cell_kj = register_buffer("cosine_cell_kj", cc);
  cosine_face2_kj = register_buffer("cosine_face2_kj", c2);
  cosine_face3_kj = register_buffer("cosine_face3_kj", c3);

  // dimension 1
  auto [df, xv, dv] = x1_metrics_();
  dx1f = register_buffer("dx1f", df);
  x1v = register_buffer("x1v", xv);
  dx1v = register_buffer("dx1v", dv);

  // dimensions 2 and 3 (uniform)
  auto [df2, xv2, dv2] = uniform_axis_metrics_(2);
  dx2f = register_buffer("dx2f", df2);
  x2v = register_buffer("x2v", xv2);
  dx2v = register_buffer("dx2v", dv2);
  auto [df3, xv3, dv3] = uniform_axis_metrics_(3);
  dx3f = register_buffer("dx3f", df3);
  x3v = register_buffer("x3v", xv3);
  dx3v = register_buffer("dx3v", dv3);

  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);
}
}  // namespace snap
