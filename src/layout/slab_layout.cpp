// fmt
#include <fmt/format.h>

// snap
#include "connectivity.hpp"
#include "layout.hpp"

namespace snap {

void SlabLayoutImpl::_initialize() {
  // build the ranks
  TORCH_CHECK(options->pz() == 1,
              "SlabLayoutImpl: pz must be 1 for slab layout");

  int px = options->px();
  int py = options->py();

  _coords2.resize(px * py);
  build_zorder_coords2(px, py, _coords2.data());
  build_rank_of2(px, py, _coords2.data(), _rankof.data());

  // build backend
  _init_process_group();
}

void SlabLayoutImpl::pretty_print(std::ostream& os) const {
  options->report(os);
  os << " Rank | (rx,ry)\n";
  os << "----------------\n";
  for (int r = 0; r < options->nblocks(); ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2})\n", r, _coords2[r].x,
                      _coords2[r].y);
  }
}

std::tuple<int, int, int> SlabLayoutImpl::loc_of(int rank) const {
  if (rank < 0 || rank >= options->nblocks()) return {-1, -1, 0};
  return {_coords2[rank].x, _coords2[rank].y, 0};
}

int SlabLayoutImpl::neighbor_rank(std::tuple<int, int, int> iloc,
                                  std::tuple<int, int, int> offset) const {
  auto [rx, ry, _1] = iloc;
  auto [dy, dx, _2] = offset;

  int nx = wrap_or_reject(rx + dx, options->px(), options->periodic_x());
  if (nx < 0) return -1;

  int ny = wrap_or_reject(ry + dy, options->py(), options->periodic_y());
  if (ny < 0) return -1;

  return _rankof[linear_index2(options->px(), options->py(), ny, nx)];
}

}  // namespace snap
