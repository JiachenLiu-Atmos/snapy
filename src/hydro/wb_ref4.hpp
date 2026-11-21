#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/recon/x1_rows.hpp>

namespace snap {

//! #289, dynamics/conversion-order 4: fourth-order x1 reference FACE DENSITY
//! for the well-balanced reconstruction -- the face value of the kernel's own
//! cell density reference, which is what makes the face offset of
//! docs/derivations/wb-ref4.md section 2 vanish. The cell reference and the
//! face pressures (the hydrostatic scan) are never changed; a guarded face
//! keeps the kernel's value. The rows are x1_rows' face_density_rows.

//! the resolution flag (nc3, nc2, nc1): cells (dilated by two) where the
//! column has fewer than two cells per pressure scale height and the kernel's
//! face value stays. Needs no seam exchange, the scan pressures being local.
torch::Tensor wb_ref4_flag(FaceDensityRows const& st,
                           torch::Tensor const& psf_lo,
                           torch::Tensor const& psf_hi);

//! face part, after the x1 seam exchange: dsf on the owned faces is..iu+1 is
//! the fourth-order face value of dref itself. Needs is >= 1 (a ghost cell).
void wb_ref4_faces(FaceDensityRows const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag);

}  // namespace snap
