// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc.hpp>
#include <snap/bc/internal_boundary.hpp>
#include <snap/coord/coordinate.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_bc(py::module &m) {
  m.attr("kUnknown") = (int)snap::BoundaryFace::kUnknown;
  m.attr("kInnerX1") = (int)snap::BoundaryFace::kInnerX1;
  m.attr("kOuterX1") = (int)snap::BoundaryFace::kOuterX1;
  m.attr("kInnerX2") = (int)snap::BoundaryFace::kInnerX2;
  m.attr("kOuterX2") = (int)snap::BoundaryFace::kOuterX2;
  m.attr("kInnerX3") = (int)snap::BoundaryFace::kInnerX3;
  m.attr("kOuterX3") = (int)snap::BoundaryFace::kOuterX3;

  auto pyBoundaryFunctionOptions =
      py::class_<snap::BoundaryFuncOptions>(m, "BoundaryFuncOptions");

  pyBoundaryFunctionOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::BoundaryFuncOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("BoundaryFuncOptions(\n{})", ss.str());
           })
      .ADD_OPTION(int, snap::BoundaryFuncOptions, type)
      .ADD_OPTION(int, snap::BoundaryFuncOptions, nghost);

  auto pyInternalBoundaryOptions =
      py::class_<snap::InternalBoundaryOptionsImpl,
                 snap::InternalBoundaryOptions>(m, "InternalBoundaryOptions");

  pyInternalBoundaryOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::InternalBoundaryOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("InternalBoundaryOptions(\n{})", ss.str());
           })
  // the rows of src/options/internal_boundary.yaml
#include <snap_options/internal_boundary_pybind.inc>
      ;  // NOLINT(whitespace/semicolon)

  ADD_SNAP_MODULE(InternalBoundary, InternalBoundaryOptions)
      .def(py::init<snap::InternalBoundaryOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("block") = nullptr)
      .def("mark_prim_solid_", &snap::InternalBoundaryImpl::mark_prim_solid_)
      .def("fill_cons_solid_", &snap::InternalBoundaryImpl::fill_cons_solid_)
      .def(
          "rectify_solid",
          [](snap::InternalBoundaryImpl &self, torch::Tensor solid_in,
             std::vector<bcfunc_t> const &bfuncs) {
            int total_num_flips = 0;
            auto result = self.rectify_solid(solid_in, total_num_flips, bfuncs);
            return std::make_pair(result, total_num_flips);
          },
          py::arg("solid"), py::arg("bfuncs") = std::vector<bcfunc_t>{});
}
