#include <cmath>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

#include <snap_options/bot_heat_read.inc>
#include <snap_options/top_cool_read.inc>

TopCoolOptions TopCoolOptionsImpl::from_yaml(YAML::Node const& forcing) {
  return read_forcing(forcing, "top-cool", read_top_cool_rows);
}

BotHeatOptions BotHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  return read_forcing(forcing, "bot-heat", read_bot_heat_rows);
}

// bot-heat's rows are top-cool's: flux, depth, thickness
static TopCoolOptions as_layer(BotHeatOptions const& op) {
  auto l = TopCoolOptionsImpl::create();
  l->flux(op->flux()).depth(op->depth()).thickness(op->thickness());
  return l;
}

LayerHeatImpl::LayerHeatImpl(BotHeatOptions const& op, HydroImpl const* p)
    : LayerHeatImpl(as_layer(op), p, false) {}

LayerHeatImpl::LayerHeatImpl(TopCoolOptions op, HydroImpl const* p, bool top)
    : SourceModule(std::move(op), p), top_(top) {
  // a layer in x1 length runs on every block; one in cells at the wall only
  auto wall = top ? SourceRegion::top : SourceRegion::bottom;
  decl_ = {top ? "top-cool" : "bot-heat",
           SourcePhase::pre_implicit,
           false,
           SourceKind::physical,
           kEnergy,
           options->thickness() > 0. ? SourceRegion::interior : wall};
  reset();
}

void LayerHeatImpl::reset() {
  char const* tag = top_ ? "[TopCool]" : "[BotHeat]";
  TORCH_CHECK(phydro, tag, " Parent Hydro is null");
  if (!phydro->pmb || !phydro->pmb->pcoord) return;
  auto const& op = phydro->pmb->pcoord->options;
  TORCH_CHECK(options->thickness() > 0. || !op->stretched_x1(), tag,
              " an integer depth is not supported on a stretched x1 "
              "(geometry/x1f or cells/x1rat, #280): it counts cells; give "
              "thickness, the layer in x1 length units");
  TORCH_CHECK(options->thickness() <= op->global_x1max() - op->global_x1min(),
              tag, " thickness exceeds the x1 column");
}

void LayerHeatImpl::add(torch::Tensor du, StageState const& s) {
  auto pcoord = phydro->pmb->pcoord;
  // a layer in x1 length (#280): each cell takes its overlap with it
  if (options->thickness() > 0.) {
    int il = pcoord->il(), iu = pcoord->iu();
    auto const& g = *pcoord->options;
    double th = options->thickness();
    double lo = top_ ? g.global_x1max() - th : g.global_x1min();
    double hi = top_ ? g.global_x1max() : g.global_x1min() + th;
    auto xf = pcoord->x1f.slice(0, il, iu + 2);
    auto overlap =
        (xf.slice(0, 1).clamp_max(hi) - xf.slice(0, 0, -1).clamp_min(lo))
            .clamp_min(0.);
    du[IPR].slice(-1, il, iu + 1) += s.dt.value() * options->flux() /
                                     options->thickness() * overlap /
                                     pcoord->dx1f.slice(0, il, iu + 1);
    return;
  }
  int i = top_ ? pcoord->iu() : pcoord->il();
  auto dz = pcoord->dx1f[i];
  int i0 = top_ ? i + 1 - options->depth() : i;
  du[IPR].narrow(-1, i0, options->depth()) +=
      s.dt.value() * options->flux() / (dz * options->depth());
}

}  // namespace snap
