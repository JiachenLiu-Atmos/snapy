#pragma once

// base
#include <configure.h>

// C/C++
#include <cmath>

namespace snap {

template <typename T>
T DISPATCH_MACRO shallow_water_roe_sound_speed(T hl, T hr) {
  return sqrt(T(0.5) * (hl + hr));
}

}  // namespace snap
