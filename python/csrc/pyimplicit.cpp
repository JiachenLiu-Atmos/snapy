// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/implicit/implicit_hydro.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_implicit(py::module &m) {
  auto pyImplicitOptions =
      py::class_<snap::ImplicitOptionsImpl, snap::ImplicitOptions>(
          m, "ImplicitOptions");

  pyImplicitOptions.def(py::init<>())
      .def_static("from_yaml",
                  py::overload_cast<std::string const &, bool>(
                      &snap::ImplicitOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const snap::ImplicitOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ImplicitOptions(\n{})", ss.str());
           })
      .def("type", &snap::ImplicitOptionsImpl::type)
      .ADD_OPTION(int, snap::ImplicitOptionsImpl, scheme)
      .ADD_OPTION(std::string, snap::ImplicitOptionsImpl, precision)
      .ADD_OPTION(bool, snap::ImplicitOptionsImpl, fallback_fp64)
      .ADD_OPTION(double, snap::ImplicitOptionsImpl, advection_cfl)
      .ADD_OPTION(double, snap::ImplicitOptionsImpl, shear_cfl);

  ADD_SNAP_MODULE(ImplicitHydro, ImplicitOptions)
      .def(py::init<snap::ImplicitOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr);
}
