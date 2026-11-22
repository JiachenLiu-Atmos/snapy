// fmt
#include <fmt/format.h>

// snap
#include "connectivity.hpp"
#include "layout.hpp"

namespace snap {

void CubedLayoutImpl::_initialize() {
  // build the ranks
  int px = options->px();
  int py = options->py();
  int pz = options->pz();

  _coords3.resize(px * py * pz);
  build_zorder_coords3(px, py, pz, _coords3.data());
  build_rank_of3(px, py, pz, _coords3.data(), _rankof.data());

  // build backend
  _init_process_group();
}

void CubedLayoutImpl::pretty_print(std::ostream &os) const {
  options->report(os);
  os << " Rank | (rx,ry,rz)\n";
  os << "-------------------\n";
  for (int r = 0; r < options->nblocks(); ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2},{:>2})\n", r, _coords3[r].x,
                      _coords3[r].y, _coords3[r].z);
  }
}

std::tuple<int, int, int> CubedLayoutImpl::loc_of(int rank) const {
  if (rank < 0 || rank >= options->nblocks()) return {-1, -1, -1};
  return {_coords3[rank].x, _coords3[rank].y, _coords3[rank].z};
}

int CubedLayoutImpl::neighbor_rank(std::tuple<int, int, int> iloc,
                                   std::tuple<int, int, int> offset) const {
  auto [rx, ry, rz] = iloc;
  auto [dy, dx, dz] =
      offset;  // dy=offset[0]=x3-dir, dx=offset[1]=x2-dir, dz=offset[2]=x1-dir

  int nx = rx + dx;
  int ny = ry + dy;
  int nz = rz + dz;

  if (options->periodic_x()) {
    if (nx < 0)
      nx += options->px();
    else if (nx >= options->px())
      nx -= options->px();
  } else {
    if (nx < 0 || nx >= options->px()) return -1;
  }

  if (options->periodic_y()) {
    if (ny < 0)
      ny += options->py();
    else if (ny >= options->py())
      ny -= options->py();
  } else {
    if (ny < 0 || ny >= options->py()) return -1;
  }

  if (options->periodic_z()) {
    if (nz < 0)
      nz += options->pz();
    else if (nz >= options->pz())
      nz -= options->pz();
  } else {
    if (nz < 0 || nz >= options->pz()) return -1;
  }

  return _rankof[linear_index3(options->px(), options->py(), nz, ny, nx)];
}

}  // namespace snap
