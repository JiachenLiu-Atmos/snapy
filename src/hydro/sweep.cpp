#include "sweep.hpp"

// C/C++
#include <algorithm>

// snap
#include <snap/eos/ideal_moist.hpp>
#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

void post_face_floors(ReconstructImpl const& recon, RegisterMonitor& monitor) {
  if (!recon.floored.defined()) return;
  monitor.count("face_floor.rho", recon.floored[0]);
  monitor.count("face_floor.p", recon.floored[1]);
  monitor.count("face_floor.species", recon.floored[2]);
}

torch::Tensor reconstruct_faces(HydroImpl& hydro, FaceDir const& dir,
                                torch::Tensor const& w,
                                IdealMoistPerturbations const* perturb,
                                torch::Tensor& background) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  int dim = dir.dim;
  auto& monitor = hydro.flux_register().monitor();
  auto recon = dim == DIM1 ? hydro.precon1 : hydro.precon23;
  if (!hydro.fp32()) {
    auto lr = recon->forward(w, dim);
    post_face_floors(*recon, monitor);
    return lr;
  }
  // The gauge: the midrange of pressure0 along the sweep. The mean is a
  // rounded sum over the block's own cells, so on a column that is uniform
  // along the sweep it is 1 ulp off by block extent and the FP32 face states
  // depend on the decomposition. The midrange of a uniform column is the
  // column's value exactly in every block, and elsewhere it bounds
  // |pressure0 - gauge| at half the range, as small as any constant can.
  auto [lo, hi] = torch::aminmax(perturb->pressure0, dim - 1, true);
  auto gauge = 0.5 * (lo + hi);
  auto input = w.to(torch::kFloat32);
  input[IPR].copy_((perturb->pressure0 - gauge) + perturb->pressure_delta);
  auto lr = recon->forward(input, dim, false);
  post_face_floors(*recon, monitor);
  // FaceFloor on the FP32 split path: the total face states this path
  // floors itself, on the faces the reconstruction computes
  auto count_below = [&](torch::Tensor const& x, double floor, int axis) {
    int ng = recon->pinterp1->stencils() / 2 + 1;
    int64_t n = x.size(axis);
    int64_t il = ng, iu = n - ng;
    return (x < floor).narrow(axis, il, std::min<int64_t>(iu + 2, n) - il);
  };
  // Horizontal Cartesian sweeps need FP64 only for pressure cancellation.
  // Covariance consumers still require total face states; retain their path.
  bool active_horizontal = (dim == DIM2 && !hydro.options->disable_flux_x2()) ||
                           (dim == DIM3 && !hydro.options->disable_flux_x3());
  if (active_horizontal && !hydro.flux_covariance()) {
    int axis = dim - 1;
    auto p0 = perturb->pressure0;
    int n = p0.size(axis);
    auto below =
        torch::cat({p0.narrow(axis, 0, 1), p0.narrow(axis, 0, n - 1)}, axis);
    background = 0.5 * (p0 + below);
    auto pressure = lr.select(1, IPR).to(torch::kFloat64);
    pressure += gauge;
    monitor.count(
        "face_floor.p",
        count_below(pressure, hydro.peos->options->pressure_floor(), dim));
    pressure.clamp_min_(hydro.peos->options->pressure_floor());
    pressure -= background;
    lr.select(1, IPR).copy_(pressure);
    monitor.count("face_floor.rho",
                  count_below(lr.select(1, IDN),
                              hydro.peos->options->density_floor(), dim));
    lr.select(1, IDN).clamp_min_(hydro.peos->options->density_floor());
    return lr;
  }
  lr = lr.to(torch::kFloat64);
  for (int side : {ILT, IRT}) {
    lr[side][IPR] += gauge;
    monitor.count("face_floor.p",
                  count_below(lr[side][IPR],
                              hydro.peos->options->pressure_floor(), dim - 1));
    monitor.count("face_floor.rho",
                  count_below(lr[side][IDN],
                              hydro.peos->options->density_floor(), dim - 1));
    lr[side][IPR].clamp_min_(hydro.peos->options->pressure_floor());
    lr[side][IDN].clamp_min_(hydro.peos->options->density_floor());
  }
  return lr;
}

void riemann_solve(HydroImpl& hydro, torch::Tensor lr, int dim,
                   torch::Tensor flux, torch::Tensor face_pressure,
                   torch::Tensor& background,
                   IdealMoistPerturbations const* perturb,
                   torch::Tensor central) {
  // the solve's events (C5): hllc's clamped contact pressure, roe's LLF
  auto post_events = [&] {
    if (!hydro.priemann->events.defined()) return;
    hydro.flux_register().monitor().count("riemann.pstar_clamp",
                                          hydro.priemann->events[0]);
    hydro.flux_register().monitor().count("riemann.llf",
                                          hydro.priemann->events[1]);
  };
  if (!perturb) {
    hydro.priemann->forward(lr[ILT], lr[IRT], dim, flux, face_pressure,
                            central);
    post_events();
    return;
  }
  if (!background.defined()) {
    // Horizontal (or gravity-free) sweeps: the shared face gauge need not
    // be hydrostatic. Its full gradient is retained in the momentum source.
    // A static gauge keeps physical pressure increments in the Riemann
    // flux instead of moving them into a difference of large backgrounds.
    int axis = dim - 1;
    auto p0 = perturb->pressure0;
    int n = p0.size(axis);
    auto below =
        torch::cat({p0.narrow(axis, 0, 1), p0.narrow(axis, 0, n - 1)}, axis);
    background = 0.5 * (p0 + below);
    lr = lr.clone();
    lr[ILT][IPR] -= background;
    lr[IRT][IPR] -= background;
  }
  if (hydro.fp32()) lr = lr.to(torch::kFloat32);
  hydro.priemann->forward_perturbations(lr[ILT], lr[IRT], dim, flux, background,
                                        face_pressure, central);
  post_events();
  // Public face pressure and all implicit/energy consumers keep TOTALS.
  if (face_pressure.defined() && face_pressure.numel())
    face_pressure += background;
}

void solve_faces(HydroImpl& hydro, FaceDir const& dir,
                 torch::Tensor const& faces, Variables const& other,
                 IdealMoistPerturbations const* perturb,
                 torch::Tensor& background, bool central,
                 FaceHook const& before_solve) {
  auto pmb = hydro.pmb;
  bool has_solid = other.count("solid");
  auto wlr =
      has_solid ? pmb->pib->forward(faces, dir.dim, other.at("solid")) : faces;
  if (dir.disabled) return;
  if (before_solve) before_solve(wlr, dir.dim);
  int d = 4 - dir.dim;
  auto& reg = hydro.flux_register();
  // riemann.central@d, kept for A until the next stage (R-2, C5)
  riemann_solve(
      hydro, wlr, dir.dim, reg.slot(d, "riemann"), torch::Tensor(), background,
      perturb, central ? reg.part_slot(d, "riemann.central") : torch::Tensor());
  reg.post_slot(d, "riemann");
}

}  // namespace snap
