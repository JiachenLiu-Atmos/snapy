#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! the Riemann solver kinds: four face functions (riemann_kernels.h) and two
//! torch formulas, upwind and plume-roe (riemann.cpp)
enum class RiemannKind : int {
  hllc,
  lmars,
  roe,
  shallow_roe,
  upwind,
  plume_roe
};

//! the scalars a face function reads: roe's ideal-moist switch, vapor count
//! and dry gamma, shallow-roe's direction
struct RiemannParams {
  bool ideal_moist = false;
  int nvapor = 0;
  double gammad = 0.;
  bool dir_yz = false;
  //! the iterator carries riemann.central (an output after the face pressure)
  bool central = false;
  //! roe: the iterator carries the LLF flag (an output after central)
  bool llf = false;
};

}  // namespace snap

namespace at::native {

// One stub for the face-function kinds (hllc, lmars, roe, shallow-roe). The
// iterator's operands are the flux, the face pressure (not shallow-roe), wl,
// wr, then e and gamma (hllc, lmars, roe), c (hllc, roe) and the face
// background p0 (hllc's perturbation entry). inv_mu_ratio_m1, cv_ratio_m1 and
// u0 are roe's ideal-moist tables, undefined otherwise; they are arguments so
// that OnHost copies them for MPS.
using riemann_fn = void (*)(at::TensorIterator &iter, int kind, int dim,
                            snap::RiemannParams const &params,
                            at::Tensor const &inv_mu_ratio_m1,
                            at::Tensor const &cv_ratio_m1,
                            at::Tensor const &u0);

DECLARE_DISPATCH(riemann_fn, call_riemann);

}  // namespace at::native
