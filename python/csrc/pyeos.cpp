// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/eos/ideal_gas.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/moist_mixture.hpp>
#include <snap/eos/shallow_water.hpp>
#include <snap/hydro/hydro.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

namespace {
// Anomaly storage (fp32 roadmap Step 1): "U->W"/"W->U" mean totals, "U'->W"/
// "W->U'" the stored anomalies. A full-block tensor passed with an old key
// while the block stores anomalies is almost always vars["hydro_u"]: refuse
// it instead of converting anomalies as if they were totals.
torch::Tensor checked_compute(snap::EquationOfStateImpl &self, std::string ab,
                              std::vector<torch::Tensor> const &args) {
  if (self.phydro && self.phydro->has_reference() &&
      (ab == "U->W" || ab == "W->U") && !args.empty() &&
      self.phydro->is_full_block(args[0])) {
    throw py::value_error(
        "eos.compute('" + ab +
        "'): this block stores hydro_u rows IDN and IPR as anomalies "
        "(dynamics: anomaly). Use '" +
        (ab == "U->W" ? std::string("U'->W") : std::string("W->U'")) +
        "' for the stored representation. Totals of a stored state: "
        "block.total_u(vars); the stored form of totals: block.stored_u(u).");
  }
  return self.compute(ab, args);
}
}  // namespace

void bind_eos(py::module &m) {
  auto pyEquationOfStateOptions =
      py::class_<snap::EquationOfStateOptionsImpl,
                 snap::EquationOfStateOptions>(m, "EquationOfStateOptions");

  pyEquationOfStateOptions
      .def(py::init<>(&snap::EquationOfStateOptionsImpl::create))
      .def_static("from_yaml", &snap::EquationOfStateOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verobse") = false)
      .def("__repr__",
           [](const snap::EquationOfStateOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("EquationOfStateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::EquationOfStateOptionsImpl, type)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, density_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, pressure_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, temperature_floor)
      .ADD_OPTION(bool, snap::EquationOfStateOptionsImpl, limiter)
      .ADD_OPTION(bool, snap::EquationOfStateOptionsImpl, verbose)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, gammad)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, weight)
      .ADD_OPTION(std::string, snap::EquationOfStateOptionsImpl, eos_file)
      .ADD_OPTION(kintera::ThermoOptions, snap::EquationOfStateOptionsImpl,
                  thermo);

  auto pyEquationOfState =
      py::class_<snap::EquationOfStateImpl, snap::EquationOfState>(
          m, "EquationOfState");

  pyEquationOfState.def(py::init<>())
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def("__repr__",
           [](const snap::EquationOfStateImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("EquationOfState(\n{})", ss.str());
           })
      .def("nvar", &snap::EquationOfStateImpl::nvar)
      .def("species_weight", &snap::EquationOfStateImpl::species_weight,
           py::arg("n") = 0)
      .def("species_cv_ref", &snap::EquationOfStateImpl::species_cv_ref,
           py::arg("n") = 0)
      .def("specific_heat_cv", &snap::EquationOfStateImpl::specific_heat_cv,
           py::arg("prim"), py::arg("temp"))
      .def("compute", &checked_compute, py::arg("ab"),
           py::arg("args") = std::vector<torch::Tensor>{});

  auto pyIdealGas =
      py::class_<snap::IdealGasImpl, snap::EquationOfStateImpl,
                 torch::nn::Module, std::shared_ptr<snap::IdealGasImpl>>(
          m, "IdealGas");

  torch::python::add_module_bindings(pyIdealGas)
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def_readonly("options", &snap::IdealGasImpl::options)
      .def("__repr__",
           [](const snap::IdealGasImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("IdealGas(\n{})", ss.str());
           })
      .def("nvar", &snap::IdealGasImpl::nvar)
      .def("compute", &snap::IdealGasImpl::compute);

  auto pyIdealMoist =
      py::class_<snap::IdealMoistImpl, snap::EquationOfStateImpl,
                 torch::nn::Module, std::shared_ptr<snap::IdealMoistImpl>>(
          m, "IdealMoist");

  torch::python::add_module_bindings(pyIdealMoist)
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def_readonly("options", &snap::IdealMoistImpl::options)
      .def("__repr__",
           [](const snap::IdealMoistImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("IdealMoist(\n{})", ss.str());
           })
      .def("nvar", &snap::IdealMoistImpl::nvar)
      .def("compute", &checked_compute, py::arg("ab"),
           py::arg("args") = std::vector<torch::Tensor>{});

  auto pyMoistMixture =
      py::class_<snap::MoistMixtureImpl, snap::EquationOfStateImpl,
                 std::shared_ptr<snap::MoistMixtureImpl>>(m, "MoistMixture");

  torch::python::add_module_bindings(pyMoistMixture)
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def_readonly("options", &snap::MoistMixtureImpl::options)
      .def("__repr__",
           [](const snap::MoistMixtureImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("MoistMixture(\n{})", ss.str());
           })
      .def("nvar", &snap::MoistMixtureImpl::nvar)
      .def("compute", &snap::MoistMixtureImpl::compute);

  auto pyShallowWater =
      py::class_<snap::ShallowWaterImpl, snap::EquationOfStateImpl,
                 torch::nn::Module, std::shared_ptr<snap::ShallowWaterImpl>>(
          m, "ShallowWater");

  torch::python::add_module_bindings(pyShallowWater)
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def_readonly("options", &snap::ShallowWaterImpl::options)
      .def("__repr__",
           [](const snap::ShallowWaterImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("ShallowWater(\n{})", ss.str());
           })
      .def("nvar", &snap::ShallowWaterImpl::nvar)
      .def("compute", &snap::ShallowWaterImpl::compute);
}
