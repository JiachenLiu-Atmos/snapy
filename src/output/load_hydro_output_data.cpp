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

    appendTensorSliceOutput("SCALARS", "rho_mean", mean, 4, IDN, 1);
    appendTensorSliceOutput("SCALARS", "rho_std", std, 4, IDN, 1);

    if (nhydro > 4) {
      appendTensorSliceOutput("SCALARS", "press_mean", mean, 4, IPR, 1);
      appendTensorSliceOutput("SCALARS", "press_std", std, 4, IPR, 1);
    }

    appendTensorSliceOutput("SCALARS", "vel1_mean", mean, 4, IVX, 1);
    appendTensorSliceOutput("SCALARS", "vel2_mean", mean, 4, IVY, 1);
    appendTensorSliceOutput("SCALARS", "vel3_mean", mean, 4, IVZ, 1);
    appendTensorSliceOutput("SCALARS", "vel1_std", std, 4, IVX, 1);
    appendTensorSliceOutput("SCALARS", "vel2_std", std, 4, IVY, 1);
    appendTensorSliceOutput("SCALARS", "vel3_std", std, 4, IVZ, 1);

    if (ncomp > 0) {
      auto hydro_names = get_hydro_names(pmb);
      if (!hydro_names.empty()) {
        appendTensorSliceOutput("VECTORS", append_suffix(hydro_names, "_mean"),
                                mean, 4, ICY, ncomp);
        appendTensorSliceOutput("VECTORS", append_suffix(hydro_names, "_std"),
                                std, 4, ICY, ncomp);
      }
    }
  }

  // (lab-frame) density
  if (shouldOutputConserved({"D"})) {
    appendTensorSliceOutput("SCALARS", "dens", u, 4, IDN, 1);
  }

  // (rest-frame) density
  if (shouldOutputPrimitive({"d"})) {
    appendTensorSliceOutput("SCALARS", "rho", w, 4, IDN, 1);
  }

  // total energy
  if (nhydro > 4) {
    if (shouldOutputConserved({"E"})) {
      appendTensorSliceOutput("SCALARS", "Etot", u, 4, IPR, 1);
    }

    // pressure
    if (shouldOutputPrimitive({"p"})) {
      appendTensorSliceOutput("SCALARS", "press", w, 4, IPR, 1);
    }
  }

  // momentum vector
  if (shouldOutputConserved({"m"})) {
    appendTensorSliceOutput("VECTORS", "mom", u, 4, IVX, 3);
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(pmb->hydro_u, 4, IVX, 3);

      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "mom_xyz";
      pod->data.NewAthenaArray(3, pmb->hydro_u.GetDim3(),
                               pmb->hydro_u.GetDim2(), pmb->hydro_u.GetDim1());
      CalculateCartesianVector(src, pod->data, pmb->pcoord);
      AppendOutputDataNode(pod);
      num_vars_ += 3;
    }*/
  }

  // each component of momentum
  if (ContainVariable("m1")) {
    appendTensorSliceOutput("SCALARS", "mom1", u, 4, IVX, 1);
  }
  if (ContainVariable("m2")) {
    appendTensorSliceOutput("SCALARS", "mom2", u, 4, IVY, 1);
  }
  if (ContainVariable("m3")) {
    appendTensorSliceOutput("SCALARS", "mom3", u, 4, IVZ, 1);
  }

  // velocity vector
  if (shouldOutputPrimitive({"v"})) {
    appendTensorSliceOutput("VECTORS", "vel", w, 4, IVX, 3);
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(GET_SHARED("hydro/w"), 4, IVX, 3);

      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "vel_xyz";
      pod->data.NewAthenaArray(3, pmb->phydro_w.GetDim3(),
                               pmb->hydro_w.GetDim2(), pmb->hydro_w.GetDim1());
      CalculateCartesianVector(src, pod->data, pmb->pcoord);
      AppendOutputDataNode(pod);
      num_vars_ += 3;
    }*/
  }

  // each component of velocity
  if (ContainVariable("vx") || ContainVariable("v1")) {
    appendTensorSliceOutput("SCALARS", "vel1", w, 4, IVX, 1);
  }
  if (ContainVariable("vy") || ContainVariable("v2")) {
    appendTensorSliceOutput("SCALARS", "vel2", w, 4, IVY, 1);
  }
  if (ContainVariable("vz") || ContainVariable("v3")) {
    appendTensorSliceOutput("SCALARS", "vel3", w, 4, IVZ, 1);
  }

  // vapor + cloud
  if (ncomp > 0) {
    auto hydro_names = get_hydro_names(pmb);
    if (!hydro_names.empty()) {
      if (shouldOutputPrimitive()) {
        appendTensorSliceOutput("VECTORS", hydro_names, w, 4, ICY, ncomp);
      }

      if (shouldOutputConserved()) {
        appendTensorSliceOutput("VECTORS", hydro_names, u, 4, ICY, ncomp);
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
      appendTensorOutput("SCALARS", "lon", lon);

      // latitude
      appendTensorOutput("SCALARS", "lat", lat);
    }
  }
}
}  // namespace snap
