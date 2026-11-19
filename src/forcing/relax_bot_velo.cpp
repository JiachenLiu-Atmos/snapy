// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

#include <snap_options/relax_bot_velo_read.inc>

RelaxBotVeloOptions RelaxBotVeloOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  return read_forcing(forcing, "relax-bot-velo", read_relax_bot_velo_rows);
}

void RelaxBotVeloImpl::add(torch::Tensor du, StageState const& s) {
  auto const& w = s.w;
  auto bottom = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  auto rho = w[IDN].index(bottom);
  auto scale = s.dt / options->tau() * rho;

  // The force is built from CONTRAVARIANT primitive velocities, but `du` holds
  // COVARIANT momenta, so it must be lowered before it is added -- otherwise
  // the drag is not antiparallel to v and slowly ROTATES the horizontal wind
  // instead of braking it. Same conversion as the Coriolis force (#168).
  // `cosine_cell_kj` is zero on an orthogonal grid, so the lowering is the
  // identity there.
  auto force = torch::stack({scale * (options->bvx() - w[IVX].index(bottom)),
                             scale * (options->bvy() - w[IVY].index(bottom)),
                             scale * (options->bvz() - w[IVZ].index(bottom))});
  // expand_as is a view, so this costs nothing; it gives cth the slab's own
  // shape, which `bottom` can then index like any other field.
  auto cth =
      phydro->pmb->pcoord->cosine_cell_kj.expand_as(w[IDN]).index(bottom);
  coord_vec_lower_(force, cth);
  du[IVX].index(bottom) += force[VEL1];
  du[IVY].index(bottom) += force[VEL2];
  du[IVZ].index(bottom) += force[VEL3];
}

}  // namespace snap
