#pragma once

//! x * x for the device kernels of riemann, recon and implicit. A macro, not
//! a function: the square stays an inline product in the caller's expression,
//! so a compiler that contracts within one expression (clang's default
//! -ffp-contract=on, e.g. AppleClang on arm64) fuses a * a + b * b into an FMA
//! exactly as it did when each header had its own macro. A function call ends
//! the expression there and changes those bits.
#define SNAP_SQR(x) ((x) * (x))
