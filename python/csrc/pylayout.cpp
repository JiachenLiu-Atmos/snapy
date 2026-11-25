// pybind11
#include <pybind11/functional.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/extension.h>

// snap
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/distributed.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_layout(py::module& m) {
  auto pyLayoutOptions =
      py::class_<snap::LayoutOptionsImpl, snap::LayoutOptions>(m,
                                                               "LayoutOptions");

  pyLayoutOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::LayoutOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("LayoutOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_x)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_y)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_z)
      .ADD_OPTION(std::string, snap::LayoutOptionsImpl, master_addr)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, master_port)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, root_rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, process_rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, process_world_size)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, world_size)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, local_rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, blocks_per_process)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, device_id)
  // the rows of src/options/layout.yaml
#include <snap_options/layout_pybind.inc>
      ;  // NOLINT(whitespace/semicolon)

  auto pySyncOptions = py::class_<snap::SyncOptions>(m, "SyncOptions");
  py::setattr(pySyncOptions, "DIM1", py::int_((int)snap::SyncOptions::DIM1));
  py::setattr(pySyncOptions, "DIM2", py::int_((int)snap::SyncOptions::DIM2));
  py::setattr(pySyncOptions, "DIM3", py::int_((int)snap::SyncOptions::DIM3));
  pySyncOptions.def(py::init<>())
      .def("dz_min", &snap::SyncOptions::dz_min)
      .def("dz_max", &snap::SyncOptions::dz_max)
      .def("dx_min", &snap::SyncOptions::dx_min)
      .def("dx_max", &snap::SyncOptions::dx_max)
      .def("dy_min", &snap::SyncOptions::dy_min)
      .def("dy_max", &snap::SyncOptions::dy_max)
      .ADD_OPTION(bool, snap::SyncOptions, intra_panel_only)
      .ADD_OPTION(bool, snap::SyncOptions, cross_panel_only)
      .ADD_OPTION(bool, snap::SyncOptions, skip_corner)
      .ADD_OPTION(bool, snap::SyncOptions, interpolate)
      .ADD_OPTION(int, snap::SyncOptions, type)
      .ADD_OPTION(int, snap::SyncOptions, dim)
      .ADD_OPTION(int, snap::SyncOptions, phyid);

  py::class_<snap::LayoutImpl, snap::Layout>(m, "Layout")
      .def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def_readonly("options", &snap::LayoutImpl::options)
      .def("__repr__",
           [](const snap::LayoutImpl& self) {
             std::stringstream ss;
             self.options->report(ss);
             return fmt::format("Layout(\n{})", ss.str());
           })
      .def("get_procs", &snap::LayoutImpl::get_procs)
      .def("rank_of", &snap::LayoutImpl::rank_of)
      .def("loc_of", &snap::LayoutImpl::loc_of)
      .def("neighbor_rank", &snap::LayoutImpl::neighbor_rank);

  auto pySlabLayout =
      py::class_<snap::SlabLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::SlabLayoutImpl>>(m, "SlabLayout");

  pySlabLayout.def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def_readonly("options", &snap::SlabLayoutImpl::options)
      .def("rank_of", &snap::SlabLayoutImpl::rank_of)
      .def("loc_of", &snap::SlabLayoutImpl::loc_of)
      .def("neighbor_rank", &snap::SlabLayoutImpl::neighbor_rank)
      .def("__repr__", [](const snap::SlabLayoutImpl& self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("SlabLayout(\n{})", ss.str());
      });

  auto pyCubedLayout =
      py::class_<snap::CubedLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::CubedLayoutImpl>>(m, "CubedLayout");

  pyCubedLayout.def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def_readonly("options", &snap::CubedLayoutImpl::options)
      .def("rank_of", &snap::CubedLayoutImpl::rank_of)
      .def("loc_of", &snap::CubedLayoutImpl::loc_of)
      .def("neighbor_rank", &snap::CubedLayoutImpl::neighbor_rank)
      .def("__repr__", [](const snap::CubedLayoutImpl& self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("CubedSphereLayout(\n{})", ss.str());
      });

  auto pyCubedSphereLayout =
      py::class_<snap::CubedSphereLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::CubedSphereLayoutImpl>>(
          m, "CubedSphereLayout");

  pyCubedSphereLayout.def(py::init<>())
      .def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def_readonly("options", &snap::CubedSphereLayoutImpl::options)
      .def("rank_of", &snap::CubedSphereLayoutImpl::rank_of)
      .def("loc_of", &snap::CubedSphereLayoutImpl::loc_of)
      .def("neighbor_rank", &snap::CubedSphereLayoutImpl::neighbor_rank)
      .def("__repr__", [](const snap::CubedSphereLayoutImpl& self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("CubedSphereLayout(\n{})", ss.str());
      });

  // distribution functions
  auto m_dist = m.def_submodule("distributed", "Distributed module");
  m_dist.def("get_rank", &snap::get_rank)
      .def("get_local_rank", &snap::get_local_rank)
      .def("get_layout", &snap::MeshBlockImpl::get_layout)
      .def("set_process_group", &snap::set_process_group)
      .def("is_process_group_initialized", &snap::is_process_group_initialized);
}
