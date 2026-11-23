#pragma once

// C/C++
#include <type_traits>

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace snap {

//! V3 X item D: a stage's dt. Eager stages carry the host double; a captured
//! stage carries the StageExecutor's persistent device scalar, which every
//! kernel of the graph reads, so one capture serves every dt (the executor
//! fills it before each replay). The products below are bitwise those of the
//! host double: x * dt rounds dt to x's dtype either way.
class StepDt {
 public:
  StepDt(double v = 0.) : _v(v) {}  // NOLINT: eager callers pass a double
  //! a device dt: t is a 0-dim FP64 tensor on the stage's device
  static StepDt device(torch::Tensor t);

  bool on_device() const { return _t.defined(); }
  //! the host value. A device dt has none: a host read inside a captured
  //! stage would freeze that dt into the graph, so it is refused
  double value() const;
  torch::Tensor const& tensor() const { return _t; }  // undefined if host

  torch::Tensor operator*(torch::Tensor const& x) const;
  //! x / dt. On CUDA torch divides a tensor by a host scalar as x * (1/dt),
  //! 1/dt formed in FP64 and rounded to x's dtype; a device dt on CUDA takes
  //! that path, so the quotient is bitwise the host dt's
  torch::Tensor divide(torch::Tensor const& x) const;
  StepDt operator-() const;
  StepDt scaled(double a) const;  // a * dt
  //! x *= dt in place
  torch::Tensor& mul_into(torch::Tensor& x) const;

 private:
  double _v;
  torch::Tensor _t;
};

//! x * dt; a template, so x * 2.0 stays torch's own (no StepDt conversion)
template <typename D, std::enable_if_t<std::is_same_v<D, StepDt>, int> = 0>
torch::Tensor operator*(torch::Tensor const& x, D const& dt) {
  return dt * x;
}

//! s += a * x, as s.add_(x, a): with a device a, one fused multiply-add per
//! element on the device, the form of add's alpha, so the bits are the same
void axpy_(torch::Tensor& s, torch::Tensor const& x, StepDt const& a);

}  // namespace snap

namespace at::native {

using axpy_device_fn = void (*)(at::Tensor& s, at::Tensor const& x,
                                at::Tensor const& a);

DECLARE_DISPATCH(axpy_device_fn, axpy_device);

}  // namespace at::native
