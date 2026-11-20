#pragma once

// C/C++
#include <cstdint>

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace snap {

//! the three stages of the VIC column solve
enum class VicStage : int { assemble, solve, redistribute };
//! Bits of each column's status (I.rst section 7; skeleton
//! vertical_implicit.hpp), written by the column's own thread, no atomics
enum VicStatusBit : int32_t {
  kVicInputNonfinite = 1,   //!< a column's du, w or gamma is not finite
  kVicSingular = 2,         //!< the forward sweep rejected the column
  kVicResultNonfinite = 4,  //!< a nonfinite solution or redistribution
  kVicFp64Retried = 8,      //!< the column predicate ran the FP64 stage (I-6)
  kVicFp64Failed = 16,      //!< the FP64 retry failed too (I-6)
  //! internal (I-6): a column the predicated FP64 stage does not solve; it
  //! returns at once and never carries another bit
  kVicSkipColumn = 1 << 20
};

//! the bits of a failed solve
constexpr int32_t kVicFailBits =
    kVicInputNonfinite | kVicSingular | kVicResultNonfinite;

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

}  // namespace snap
