#pragma once

#include <configure.h>

#include <cmath>
#include <limits>

namespace snap {
// Finite increments, not derivatives. Branches in the shared reference kernel
// inspect FP64 totals; arithmetic retains a separate small increment. The base
// follows the selected branch, so it is not a separately guarded evaluation.
struct SplitReferenceScalar {
  double base, delta;
  DISPATCH_MACRO SplitReferenceScalar(double b = 0., double d = 0.)
      : base(b), delta(d) {}
  DISPATCH_MACRO double total() const { return base + delta; }
  DISPATCH_MACRO explicit operator int() const {
    return static_cast<int>(total());
  }
};
using SRS = SplitReferenceScalar;
inline DISPATCH_MACRO SRS operator+(SRS a, SRS b) {
  return {a.base + b.base, a.delta + b.delta};
}
inline DISPATCH_MACRO SRS operator-(SRS a) { return {-a.base, -a.delta}; }
inline DISPATCH_MACRO SRS operator-(SRS a, SRS b) { return a + (-b); }
inline DISPATCH_MACRO SRS operator*(SRS a, SRS b) {
  return {a.base * b.base, a.base * b.delta + a.delta * b.total()};
}
inline DISPATCH_MACRO SRS operator/(SRS a, SRS b) {
  if (b.base == 0.) return {0., a.total() / b.total()};
  double q = a.base / b.base;
  return {q, (a.delta - q * b.delta) / b.total()};
}
inline DISPATCH_MACRO SRS& operator+=(SRS& a, SRS b) { return a = a + b; }
inline DISPATCH_MACRO SRS& operator*=(SRS& a, SRS b) { return a = a * b; }
inline DISPATCH_MACRO bool operator<(SRS a, SRS b) {
  return a.total() < b.total();
}
inline DISPATCH_MACRO bool operator>(SRS a, SRS b) { return b < a; }
inline DISPATCH_MACRO bool operator<=(SRS a, SRS b) {
  return a.total() <= b.total();
}
inline DISPATCH_MACRO bool operator>=(SRS a, SRS b) { return b <= a; }
inline DISPATCH_MACRO bool operator==(SRS a, SRS b) {
  return a.total() == b.total();
}
inline DISPATCH_MACRO SRS exp(SRS a) {
  double b = ::exp(a.base);
  return {b, b * ::expm1(a.delta)};
}
inline DISPATCH_MACRO SRS log(SRS a) {
  if (a.base <= 0.) return {0., ::log(a.total())};
  return {::log(a.base), ::log1p(a.delta / a.base)};
}
inline DISPATCH_MACRO SRS pow(SRS a, SRS b) {
  if (a.base <= 0.) return {0., ::pow(a.total(), b.total())};
  double p = ::pow(a.base, b.base);
  return {p, p * ::expm1(b.base * ::log1p(a.delta / a.base) +
                         b.delta * ::log(a.total()))};
}
inline DISPATCH_MACRO SRS fabs(SRS a) { return a.total() < 0. ? -a : a; }
inline DISPATCH_MACRO SRS floor(SRS a) { return {::floor(a.total()), 0.}; }
template <typename T>
inline DISPATCH_MACRO T reference_min_positive() {
  return std::numeric_limits<T>::min();
}
template <>
inline DISPATCH_MACRO SRS reference_min_positive<SRS>() {
  return {std::numeric_limits<double>::min(), 0.};
}
static_assert(sizeof(SRS) == 2 * sizeof(double),
              "reference pairs must be packed doubles");
}  // namespace snap
