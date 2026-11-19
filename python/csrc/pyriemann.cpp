// torch
#include <torch/extension.h>

// snap
#include <snap/riemann/riemann.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_riemann(py::module &m) {
  auto pyRiemannSolverOptions =
      py::class_<snap::RiemannSolverOptionsImpl, snap::RiemannSolverOptions>(
          m, "RiemannSolverOptions");

  pyRiemannSolverOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::RiemannSolverOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("RiemannSolverOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::RiemannSolverOptionsImpl, type);

  ADD_SNAP_MODULE(UpwindSolver, RiemannSolverOptions)
      .def(py::init<snap::RiemannSolverOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro"));

  ADD_SNAP_MODULE(RoeSolver, RiemannSolverOptions)
      .def(py::init<snap::RiemannSolverOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro"));

  ADD_SNAP_MODULE(LmarsSolver, RiemannSolverOptions)
      .def(py::init<snap::RiemannSolverOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro"));

  ADD_SNAP_MODULE(ShallowRoeSolver, RiemannSolverOptions)
      .def(py::init<snap::RiemannSolverOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro"));
}
