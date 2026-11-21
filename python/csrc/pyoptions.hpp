#include <fmt/format.h>

#define ADD_OPTION(T, st_name, op_name)                             \
  def(#op_name, (T const& (st_name::*)() const) & st_name::op_name) \
      .def(#op_name, (st_name & (st_name::*)(const T&)) & st_name::op_name)

#define ADD_SNAP_MODULE(m_name, op_name)                       \
  torch::python::bind_module<snap::m_name##Impl>(m, #m_name)   \
      .def(py::init<>(), R"(Construct a new default module.)") \
      .def_readonly("options", &snap::m_name##Impl::options)   \
      .def("__repr__",                                         \
           [](const snap::m_name##Impl& a) {                   \
             std::stringstream ss;                             \
             a.options->report(ss);                            \
             return fmt::format(#m_name "(\n{})", ss.str());   \
           })                                                  \
      .def("module",                                           \
           [](snap::m_name##Impl& self, std::string name) {    \
             return self.named_modules()[name];                \
           })                                                  \
      .def("buffer",                                           \
           [](snap::m_name##Impl& self, std::string name) {    \
             return self.named_buffers()[name];                \
           })                                                  \
      .def("forward", &snap::m_name##Impl::forward,            \
           py::call_guard<py::gil_scoped_release>())

// A stepped module (forward(vars, dt, stage)) binds its own forward with the
// GIL released; torch::python::bind_module's automatic forward/__call__ holds
// the GIL and would shadow it (V3 S-K4: the chemistry step's autograd use
// inside forward() needs the GIL released)
#define ADD_SNAP_MODULE_STEPPED(m_name, op_name)                   \
  torch::python::bind_module<snap::m_name##Impl, true>(m, #m_name) \
      .def(py::init<>(), R"(Construct a new default module.)")     \
      .def_readonly("options", &snap::m_name##Impl::options)       \
      .def("__repr__",                                             \
           [](const snap::m_name##Impl& a) {                       \
             std::stringstream ss;                                 \
             a.options->report(ss);                                \
             return fmt::format(#m_name "(\n{})", ss.str());       \
           })                                                      \
      .def("module",                                               \
           [](snap::m_name##Impl& self, std::string name) {        \
             return self.named_modules()[name];                    \
           })                                                      \
      .def("buffer",                                               \
           [](snap::m_name##Impl& self, std::string name) {        \
             return self.named_buffers()[name];                    \
           })                                                      \
      .def("forward", &snap::m_name##Impl::forward,                \
           py::call_guard<py::gil_scoped_release>())
