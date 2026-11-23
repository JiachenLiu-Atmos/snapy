// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

#include <snap_options/body_heat_read.inc>
#include <snap_options/plume_forcing_read.inc>

BodyHeatOptions BodyHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  return read_forcing(forcing, "body-heat", read_body_heat_rows);
}

PlumeForcingOptions PlumeForcingOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  return read_forcing(forcing, "plume-forcing", read_plume_forcing_rows);
}

BodyHeatImpl::BodyHeatImpl(BodyHeatOptions op, HydroImpl const* p)
    : SourceModule(std::move(op), p) {
  decl_ = {"body-heat", SourcePhase::pre_implicit, false, SourceKind::physical,
           kEnergy};
}

void BodyHeatImpl::add(torch::Tensor du, StageState const& s) {
  auto interior =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto pres = s.w[IPR].index(interior);
  auto rho = s.w[IDN].index(interior);
  auto cv = phydro->peos->specific_heat_cv(s.w, s.temp).index(interior);
  auto mask =
      torch::logical_and(pres >= options->pmin(), pres <= options->pmax());
  du[IPR].index(interior) += torch::where(
      mask, s.dt.value() * options->dTdt() * rho * cv, torch::zeros_like(pres));
}

void PlumeForcingImpl::add(torch::Tensor du, StageState const& s) {
  auto R = s.w[0];
  auto W = s.w[1];
  auto B = s.w[2];
  du[0] -= 2. * R * options->entrainment() * s.dt.value();
  du[1] += R * R * B * s.dt.value();
  du[2] -= options->N2() * R * R * W * s.dt.value();
}

}  // namespace snap
