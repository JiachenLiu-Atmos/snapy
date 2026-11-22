#pragma once

// base
#include <configure.h>

namespace snap {

//! x * x, for the device kernels of riemann, recon and implicit
template <typename T>
DISPATCH_MACRO constexpr T sqr(T x) {
  return x * x;
}

}  // namespace snap
