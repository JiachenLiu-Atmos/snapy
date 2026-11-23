// snap
#include "step_dt.hpp"

namespace snap {

StepDt StepDt::device(torch::Tensor t) {
  TORCH_CHECK(t.defined() && t.dim() == 0 && t.scalar_type() == torch::kFloat64,
              "StepDt: a device dt is a 0-dim FP64 tensor");
  StepDt dt;
  dt._t = t;
  return dt;
}

double StepDt::value() const {
  TORCH_CHECK(!_t.defined(),
              "StepDt: this stage reads dt on the host, so it cannot replay "
              "with another dt (a captured stage reads the device dt)");
  return _v;
}

torch::Tensor StepDt::operator*(torch::Tensor const& x) const {
  return _t.defined() ? _t * x : _v * x;
}

StepDt StepDt::operator-() const {
  if (!_t.defined()) return StepDt(-_v);
  return device(_t.neg());
}

StepDt StepDt::scaled(double a) const {
  if (!_t.defined()) return StepDt(a * _v);
  return device(_t * a);
}

torch::Tensor& StepDt::mul_into(torch::Tensor& x) const {
  if (_t.defined())
    x.mul_(_t);
  else
    x.mul_(_v);
  return x;
}

void axpy_(torch::Tensor& s, torch::Tensor const& x, StepDt const& a) {
  if (!a.on_device()) {
    s.add_(x, a.value());
    return;
  }
  at::native::axpy_device(s.device().type(), s, x, a.tensor());
}

//! a device dt off CUDA (no graph runs there): add's alpha from the host
void axpy_device_cpu(at::Tensor& s, at::Tensor const& x, at::Tensor const& a) {
  s.add_(x, a.item<double>());
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(axpy_device);

REGISTER_ALL_CPU_DISPATCH(axpy_device, &snap::axpy_device_cpu);
REGISTER_MPS_DISPATCH(axpy_device, &snap::axpy_device_cpu);

}  // namespace at::native
