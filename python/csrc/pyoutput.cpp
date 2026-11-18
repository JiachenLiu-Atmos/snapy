// pybind11
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_output(py::module &m) {
  auto pyOutputOptions =
      py::class_<snap::OutputOptionsImpl, snap::OutputOptions>(m,
                                                               "OutputOptions");

  pyOutputOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::OutputOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("OutputOptions{}", ss.str());
           })
  // the rows of src/options/output.yaml
#include <snap_options/output_pybind.inc>
      ;  // NOLINT(whitespace/semicolon)

  auto pyOutputType =
      py::class_<snap::OutputType, std::shared_ptr<snap::OutputType>>(
          m, "OutputType");

  pyOutputType.def(py::init<>())
      .def(py::init<snap::OutputOptions>())
      .def("__repr__",
           [](const snap::OutputType &a) {
             return fmt::format("OutputType(file_number = {}; next_time = {})",
                                a.file_number, a.next_time);
           })
      .def_readwrite("file_number", &snap::OutputType::file_number)
      .def_readwrite("next_time", &snap::OutputType::next_time);

  auto pyNetcdfOutput =
      py::class_<snap::NetcdfOutput, snap::OutputType,
                 std::shared_ptr<snap::NetcdfOutput>>(m, "NetcdfOutput");

  pyNetcdfOutput.def(py::init<snap::OutputOptions>())
      .def("__repr__",
           [](const snap::NetcdfOutput &a) {
             return fmt::format(
                 "NetcdfOutput(file_number = {}; next_time = {})",
                 a.file_number, a.next_time);
           })
      .def(
          "write_output_file",
          [](snap::NetcdfOutput &self, py::object block_obj,
             py::dict const &vars, double time, int wtflag) {
            py::object cpp_module = block_obj.attr("cpp_module");
            auto pmb = cpp_module.cast<std::shared_ptr<snap::MeshBlockImpl>>();

            std::map<std::string, torch::Tensor> native;
            for (auto &kv : vars) {
              std::string key = py::cast<std::string>(kv.first);
              if (!kv.second.is_none()) {
                native[key] = py::cast<torch::Tensor>(kv.second);
              } else {
                native[key] = torch::Tensor();
              }
            }

            self.write_output_file(pmb.get(), native, time, wtflag);
          },
          py::arg("block"), py::arg("vars"), py::arg("time"),
          py::arg("wtflag") = 0);
}
