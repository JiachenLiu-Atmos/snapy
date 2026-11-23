// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

#include <snap_options/bot_sponge_lyr_read.inc>
#include <snap_options/top_sponge_lyr_read.inc>

TopSpongeLyrOptions TopSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  return read_forcing(forcing, "top-sponge-lyr", read_top_sponge_lyr_rows);
}

BotSpongeLyrOptions BotSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  return read_forcing(forcing, "bot-sponge-lyr", read_bot_sponge_lyr_rows);
}

// bot-sponge-lyr's rows are top-sponge-lyr's: tau, width
static TopSpongeLyrOptions as_top(BotSpongeLyrOptions const& op) {
  auto t = TopSpongeLyrOptionsImpl::create();
  t->tau(op->tau()).width(op->width());
  return t;
}

SpongeImpl::SpongeImpl(BotSpongeLyrOptions const& op, HydroImpl const* p)
    : SpongeImpl(as_top(op), p, false) {}

SpongeImpl::SpongeImpl(TopSpongeLyrOptions op, HydroImpl const* p, bool top)
    : SourceModule(std::move(op), p), top_(top) {
  decl_ = {top ? "top-sponge-lyr" : "bot-sponge-lyr",
           SourcePhase::pre_implicit,
           false,
           SourceKind::physical,
           kMomentum,
           top ? SourceRegion::top : SourceRegion::bottom};
}

void SpongeImpl::add(torch::Tensor du, StageState const& s) {
  auto pcoord = phydro->pmb->pcoord;
  auto const& x1f = pcoord->x1f;
  auto d = top_ ? x1f[pcoord->iu() + 1] - x1f.slice(0, 0, -1)
                : x1f.slice(0, 0, -1) - x1f[pcoord->il()];
  auto eta = (options->width() - d) / options->width();
  eta.clamp_(0., 1.0);
  auto scale = torch::sin(M_PI / 2. * eta).pow(2).unsqueeze(0).unsqueeze(0);

  // the force from contravariant velocities is lowered to du's covariant
  // momenta, or the drag rotates the wind (#168); the identity on an
  // orthogonal grid, with the operand order kept bit-identical
  auto force =
      s.w[IDN] * s.w.narrow(0, IVX, 3) / options->tau() * scale * s.dt.value();
  coord_vec_lower_(force, pcoord->cosine_cell_kj);
  du.narrow(0, IVX, 3) -= force;
}

}  // namespace snap
