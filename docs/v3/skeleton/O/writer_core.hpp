#pragma once
// V3 lane O skeleton: declarations only, not built. Pin 9e5de2d4.
// One writer core (O.rst 3.1): replaces OutputData, the two sink layout
// copies, vectorize and mppnccombine. Borrows vars; owns host copies only.
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace snap {
class MeshBlockImpl;
class Layout;
struct OutputOptionsImpl;  // generated from the options table
using Variables = std::map<std::string, torch::Tensor>;

//! staggering per axis (x3, x2, x1): 'C' centre, 'F' face, '-' reduced
struct GridKind {
  char x3 = 'C', x2 = 'C', x1 = 'C';
};

//! one output field; data is (ncomp, n3, n2, n1)
struct Field {
  std::vector<std::string> names;  // one per component
  GridKind grid;
  std::string units, long_name;  // from the field registry
  torch::Tensor data;            // device until Frame::to_host()
};

//! one block's contribution to one record
struct Frame {
  struct Axis {
    int64_t global = 1, start = 0, count = 1;  // domain_decomposition
    bool has_faces = false;
  };
  double time = 0.;
  int64_t cycle = 0;
  std::vector<Field> fields;
  torch::Tensor x1v, x1f, x2v, x2f, x3v, x3f;
  Axis ax1, ax2, ax3;
  int panel = 0;  // cubed-sphere panel, laid out 3 x 2

  bool slice(int axis, std::vector<double> const& coords);  // index_select
  void sum(int axis);                                         // torch sum
  void to_host();  // the only host read on the gridded path
};

//! one per format; all layout arithmetic stays in the core
class Sink {
 public:
  virtual ~Sink() = default;
  virtual bool collective() const = 0;  // pnetcdf: true
  virtual void open(std::string const& path, Frame const& shape) = 0;
  virtual void define(Field const& f, bool double_precision) = 0;
  virtual void put_coords(Frame const& frame) = 0;
  virtual void put(Field const& f, int comp, std::vector<int64_t> const& start,
                   std::vector<int64_t> const& count) = 0;
  virtual void close() = 0;
};
std::unique_ptr<Sink> make_netcdf_sink();
std::unique_ptr<Sink> make_pnetcdf_sink();  // PNETCDFOUTPUT builds only

//! replaces OutputType and the four format classes (output_formats.hpp)
class GriddedWriter {
 public:
  explicit GriddedWriter(std::shared_ptr<OutputOptionsImpl> options);
  int file_number = 0;
  double next_time = 0.;
  int64_t schedule_key() const;     // legacy (#313)
  int64_t schedule_key_v2() const;  // exact cadence bits (#313)
  //! dt: host double of the accepted step; dt <= 0 (first call) no-op
  void accumulate_stats(Variables const& vars, double dt);
  void write(MeshBlockImpl* pmb, Variables const& vars, double time,
             bool final_write);

 private:
  std::shared_ptr<OutputOptionsImpl> options_;
  std::unique_ptr<Sink> sink_;
  torch::Tensor prim_mean_, prim_m2_, scalar_mean_, scalar_m2_;
};

//! the `-r out parts...` form only (combine_netcdf.cpp:73): merge snapy's own
//! parts via domain_decomposition/NumFilesInSet, then remove the parts.
//! Defined in combine_parts.cpp (~200, addition row); gate G-2.
int combine_parts(std::string const& outfile,
                  std::vector<std::string> const& parts);

// kept from #350 (writer_core.hpp:19-46 at the pin)
std::tuple<int, int, int, int, int, int> output_range(MeshBlockImpl const* pmb,
                                                      bool include_ghost_zones);
void make_output_dir(MeshBlockImpl const* pmb);
void combine_on_root(Layout const& layout, std::string const& key,
                     std::function<void()> const& root_work);
}  // namespace snap
