#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO T ideal_moist_feps(T const* y, int ny, int nvapor,
                                         T const* inv_mu_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < nvapor; ++n) out += y[n] * inv_mu_ratio_m1[n];
  for (int n = nvapor; n < ny; ++n) out -= y[n];
  return out;
}

template <typename T>
inline DISPATCH_MACRO T ideal_moist_fsig(T const* y, int ny,
                                         T const* cv_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < ny; ++n) out += y[n] * cv_ratio_m1[n];
  return out;
}

}  // namespace snap
