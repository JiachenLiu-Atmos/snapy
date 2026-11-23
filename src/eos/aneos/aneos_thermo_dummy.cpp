// torch
#include <torch/torch.h>

// snap
#include "aneos_thermo.hpp"

namespace snap {
std::shared_ptr<ANEOSThermoImpl> ANEOSThermoImpl::create(
    std::string const& filename, torch::nn::Module* p,
    std::string const& name) {
  // Keep the pointer made here. register_module returns a
  // dynamic_pointer_cast back to ANEOSThermoImpl. When the table's methods
  // are defined in another image, overriding the weak stubs below (as
  // test_aneos_gamma does), that image emits its own vtable and type_info
  // for the class, and on macOS (arm64 CI) the cast returned null, leaving
  // pthermo an empty ModuleHolder.
  auto thermo = std::make_shared<ANEOSThermoImpl>(filename);
  p->register_module("thermo", thermo);
  return thermo;
}

__attribute__((weak)) ANEOSThermoImpl::ANEOSThermoImpl(
    const std::string& fname) {
  throw std::runtime_error("ANEOSThermoImpl constructor is not implemented");
}

__attribute__((weak)) void ANEOSThermoImpl::reset() {
  throw std::runtime_error("ANEOSThermoImpl::reset is not implemented");
}

__attribute__((weak)) void ANEOSThermoImpl::pretty_print(
    std::ostream& stream) const {
  throw std::runtime_error("ANEOSThermoImpl::pretty_print is not implemented");
}

__attribute__((weak)) std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ANEOSThermoImpl::compute(std::string ab,
                         std::vector<torch::Tensor> const& args) {
  throw std::runtime_error("ANEOSThermoImpl::compute is not implemented");
}

}  // namespace snap
