#pragma once

// C/C++
#include <functional>
#include <map>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

class HydroImpl;
class ReconstructImpl;
class RegisterMonitor;
struct IdealMoistPerturbations;
using Variables = std::map<std::string, torch::Tensor>;

//! what differs between the face sweeps; everything else is shared
struct FaceDir {
  int dim;        // DIM1 = 3, DIM2 = 2, DIM3 = 1
  int seam_tag;   // x1_seam_rows tag: kTagFaceX2 / kTagFaceX3 (exchange.hpp)
  bool disabled;  // disable-flux-x<n>
};

//! FaceFloor (C.rst 3.7): the faces the reconstruction's floors changed
void post_face_floors(ReconstructImpl const& recon, RegisterMonitor& monitor);

//! the face states [wl, wr] of w along dir. perturb: the split path's
//! background and increments (null on the total path); the FP32 split path
//! reconstructs the gauge perturbation, floors the totals itself and, on an
//! active horizontal sweep, sets background, its static face gauge
torch::Tensor reconstruct_faces(HydroImpl& hydro, FaceDir const& dir,
                                torch::Tensor const& w,
                                IdealMoistPerturbations const* perturb,
                                torch::Tensor& background);

//! one Riemann solve of the face states lr into flux, with its events. On the
//! split path (perturb) the solver sees perturbations about background (the
//! static face gauge if undefined) and face_pressure keeps totals
void riemann_solve(HydroImpl& hydro, torch::Tensor lr, int dim,
                   torch::Tensor flux, torch::Tensor face_pressure,
                   torch::Tensor& background,
                   IdealMoistPerturbations const* perturb,
                   torch::Tensor central = torch::Tensor());

//! called with the post-solid, unrotated face states before the solve
using FaceHook = std::function<void(torch::Tensor const&, int)>;

//! the x2/x3 sweep after reconstruction: immersed solid -> before_solve ->
//! Riemann into slot(d, "riemann"), riemann.central@d when central, then
//! post_slot. Disabled: the solid step only
void solve_faces(HydroImpl& hydro, FaceDir const& dir,
                 torch::Tensor const& faces, Variables const& other,
                 IdealMoistPerturbations const* perturb,
                 torch::Tensor& background, bool central,
                 FaceHook const& before_solve = {});

}  // namespace snap
