#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace snap {

//! the three stages of the VIC column solve
enum class VicStage : int { assemble, solve, redistribute };

//! what one vic_stage call does: the stage, the block size n (3: partial,
//! 5: full), the Woodbury solve (operands z0, z1), and the kernel scalars.
//! dir carries the gravity-work flags below for the assembly.
struct VicParams {
  VicStage stage = VicStage::assemble;
  int n = 3;
  bool woodbury = false;
  double dt = 0., grav = 0.;
  int dir = 0;
};

}  // namespace snap

namespace at::native {

using vic_stage_fn = void (*)(at::TensorIterator& iter,
                              snap::VicParams const& prm);
using donor_cap_fn = void (*)(at::Tensor& P, at::Tensor const& held, int is,
                              int ie);

//! the VIC column solve, one stage per call (vic_kernels.h)
DECLARE_DISPATCH(vic_stage_fn, vic_stage);
DECLARE_DISPATCH(donor_cap_fn, donor_cap);

}  // namespace at::native

namespace snap {

//! vic_assemble_*: added to dir, the energy row books the face gravity work of
//! the linearised x1 mass flux instead of the cell work
constexpr int kVicFaceWork = 16;
// Cell work plus gravity work of the implicit Roe mass diffusion.
constexpr int kVicDiffusiveCell = 32;
constexpr int kVicCartesianFaceWork = 64;

}  // namespace snap
