// snap
#include <snap/snap.h>

#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "output_utils.hpp"

namespace snap {
namespace {
std::string append_suffix(std::string const& names, std::string const& suffix) {
  if (names.empty()) return names;

  std::string result;
  std::string::size_type start = 0;
  while (start <= names.size()) {
    auto end = names.find(';', start);
    auto name = names.substr(start, end - start);
    if (!result.empty()) result += ";";
    result += name + suffix;
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return result;
}
}  // namespace

void OutputType::loadHydroOutputData(MeshBlockImpl* pmb,
                                     Variables const& vars) {
  auto peos = pmb->phydro->peos;
  auto pcoord = pmb->pcoord;

  auto const& w = vars.at("hydro_w");
  // totals: anomaly storage adds the reference back for output
  auto u = pmb->phydro->total_u(vars.at("hydro_u"));
  int nhydro = peos->nvar();
  int ncomp = nhydro - ICY;

  if (OutputsPrimStat()) {
    auto mean = PrimStatMean(w);
    auto std = PrimStatStd(w);

    appendTensorSliceOutput("rho_mean", mean, IDN, 1);
    appendTensorSliceOutput("rho_std", std, IDN, 1);

    if (nhydro > 4) {
      appendTensorSliceOutput("press_mean", mean, IPR, 1);
      appendTensorSliceOutput("press_std", std, IPR, 1);
    }

    appendTensorSliceOutput("vel1_mean", mean, IVX, 1);
    appendTensorSliceOutput("vel2_mean", mean, IVY, 1);
    appendTensorSliceOutput("vel3_mean", mean, IVZ, 1);
    appendTensorSliceOutput("vel1_std", std, IVX, 1);
    appendTensorSliceOutput("vel2_std", std, IVY, 1);
    appendTensorSliceOutput("vel3_std", std, IVZ, 1);

    if (ncomp > 0) {
      auto hydro_names = get_hydro_names(pmb);
      if (!hydro_names.empty()) {
        appendTensorSliceOutput(append_suffix(hydro_names, "_mean"), mean, ICY,
                                ncomp);
        appendTensorSliceOutput(append_suffix(hydro_names, "_std"), std, ICY,
                                ncomp);
      }
    }
  }

  // (lab-frame) density
  if (shouldOutputConserved({"D"})) {
    appendTensorSliceOutput("dens", u, IDN, 1);
  }

  // (rest-frame) density
  if (shouldOutputPrimitive({"d"})) {
    appendTensorSliceOutput("rho", w, IDN, 1);
  }

  // total energy
  if (nhydro > 4) {
    if (shouldOutputConserved({"E"})) {
      appendTensorSliceOutput("Etot", u, IPR, 1);
    }

    // pressure
    if (shouldOutputPrimitive({"p"})) {
      appendTensorSliceOutput("press", w, IPR, 1);
    }
  }

  // momentum vector
  if (shouldOutputConserved({"m"})) {
    appendTensorSliceOutput("mom", u, IVX, 3);
  }

  // each component of momentum
  if (ContainVariable("m1")) {
    appendTensorSliceOutput("mom1", u, IVX, 1);
  }
  if (ContainVariable("m2")) {
    appendTensorSliceOutput("mom2", u, IVY, 1);
  }
  if (ContainVariable("m3")) {
    appendTensorSliceOutput("mom3", u, IVZ, 1);
  }

  // velocity vector
  if (shouldOutputPrimitive({"v"})) {
    appendTensorSliceOutput("vel", w, IVX, 3);
  }

  // each component of velocity
  if (ContainVariable("vx") || ContainVariable("v1")) {
    appendTensorSliceOutput("vel1", w, IVX, 1);
  }
  if (ContainVariable("vy") || ContainVariable("v2")) {
    appendTensorSliceOutput("vel2", w, IVY, 1);
  }
  if (ContainVariable("vz") || ContainVariable("v3")) {
    appendTensorSliceOutput("vel3", w, IVZ, 1);
  }

  // vapor + cloud
  if (ncomp > 0) {
    auto hydro_names = get_hydro_names(pmb);
    if (!hydro_names.empty()) {
      if (shouldOutputPrimitive()) {
        appendTensorSliceOutput(hydro_names, w, ICY, ncomp);
      }

      if (shouldOutputConserved()) {
        appendTensorSliceOutput(hydro_names, u, ICY, ncomp);
      }
    }
  }

  // lat/lon grid for cubed sphere
  if (shouldOutputPrimitive() || shouldOutputConserved()) {
    if (pcoord->options->type() == "gnomonic-equiangle") {
      int r = pmb->options->layout()->rank();
      auto [rx, ry, face_id] = pmb->get_layout()->loc_of(r);
      auto face = CS_FACE_NAMES[face_id];

      auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v}, "ij");
      auto alpha = mesh[1];
      auto beta = mesh[0];
      auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);

      // longitude
      appendTensorOutput("lon", lon);

      // latitude
      appendTensorOutput("lat", lat);
    }
  }
}
}  // namespace snap
