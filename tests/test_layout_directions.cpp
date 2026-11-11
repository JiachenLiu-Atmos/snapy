// gtest
#include <gtest/gtest.h>

// C/C++
#include <cstdlib>
#include <tuple>
#include <vector>

// snap
#include <snap/layout/layout.hpp>

using namespace snap;

namespace {

using Dir = std::tuple<int, int, int>;

//! the neighbour directions every exchange loop visited before they shared
//! one iterator: dz, dy, dx nested in that order, the centre left out, and
//! with skip_corner every direction with more than one nonzero component; dz
//! moves only on a layout with 27 buffers
std::vector<Dir> reference(LayoutImpl const& layout, SyncOptions const& opts) {
  int dz_min = 0, dz_max = 0;
  if (layout.num_exchange_buffers() >= 27) {
    dz_min = opts.dz_min();
    dz_max = opts.dz_max();
  }
  std::vector<Dir> out;
  for (int dz = dz_min; dz <= dz_max; ++dz)
    for (int dy = opts.dy_min(); dy <= opts.dy_max(); ++dy)
      for (int dx = opts.dx_min(); dx <= opts.dx_max(); ++dx) {
        if (dz == 0 && dy == 0 && dx == 0) continue;
        if (opts.skip_corner() &&
            std::abs(dz) + std::abs(dy) + std::abs(dx) > 1)
          continue;
        out.emplace_back(dy, dx, dz);
      }
  return out;
}

}  // namespace

TEST(LayoutDirections, every_layout_visits_the_reference_directions) {
  for (std::string type : {"slab", "cubed", "cubed-sphere"}) {
    auto lo = LayoutOptionsImpl::create();
    lo->type(type);
    lo->px(2).py(2).pz(type == "cubed" ? 2 : 1);
    auto layout = LayoutImpl::create(lo);
    for (int dim : {0, (int)SyncOptions::DIM1, (int)SyncOptions::DIM2,
                    (int)SyncOptions::DIM3})
      for (bool skip : {true, false}) {
        SyncOptions opts;
        opts.dim(dim).skip_corner(skip);
        std::vector<Dir> got;
        layout->for_each_direction(opts, [&](int dy, int dx, int dz) {
          got.emplace_back(dy, dx, dz);
        });
        EXPECT_EQ(got, reference(*layout, opts))
            << type << " dim " << dim << " skip_corner " << skip;
      }
  }
}
