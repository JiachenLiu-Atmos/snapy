#pragma once
// V3 lane O skeleton (declarations only; not built).
//
// One writer core for every gridded output format. It replaces, at pin
// 9e5de2d4:
//   - the OutputData doubly linked list and its AthenaArray copies
//     (src/output/output_type.hpp:93-105, :201-207; output_type.cpp:201-244),
//   - the hand-written slice/sum loops (output_type.cpp:415-557),
//   - the two copies of "expand names -> define variables -> pack buffers"
//     in netcdf.cpp:264-568 and pnetcdf.cpp:190-420,
//   - mppnccombine.cpp (1,014 lines of GFDL C with file-scope state,
//     mppnccombine.cpp:93) by a snapy-specific part combiner,
//   - utils/vectorize.* (strtok-based name splitting) and netcdf_utils.hpp.
//
// Ownership: the core owns no simulation state. A Frame borrows the block's
// variables for the duration of one write and holds host copies only.

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
class Layout;  // lane L/X: process group and block placement
using Variables = std::map<std::string, torch::Tensor>;

//! Where a field lives on the staggered grid, one letter per axis in the
//! order (x3, x2, x1): 'C' centre, 'F' face, '-' reduced away.
//! Replaces the string grid codes of MetadataTable (output_utils.cpp:28-180).
struct GridKind {
  char x3 = 'C', x2 = 'C', x1 = 'C';
  static GridKind parse(std::string const& code);  // "CCF", "--C", ...
  std::string str() const;
};

//! One named output field. `data` is (ncomp, n3, n2, n1) with the reduced
//! axes of `grid` having size 1. Components get names[i]; a single name with
//! ncomp > 1 expands to name1..nameN (today netcdf.cpp:292-318).
struct Field {
  std::vector<std::string> names;
  GridKind grid;
  std::string units, long_name;  // from the field registry (options table)
  torch::Tensor data;            // host, float64 after Frame::to_host()
};

//! Everything one block contributes to one output record.
//! Built by the loaders (hydro, scalar, diagnostics, user callback), then
//! transformed (slice, sum, ghost range), then handed to a Sink.
class Frame {
 public:
  double time = 0.;
  int64_t cycle = 0;
  std::vector<Field> fields;

  // coordinates of the retained cells/faces, host float64, after transform
  torch::Tensor x1v, x1f, x2v, x2f, x3v, x3f;

  //! position of this block's data in the combined file, per axis:
  //! (global extent, 0-based start); the same numbers netcdf.cpp writes as
  //! the "domain_decomposition" attribute (netcdf.cpp:163-237)
  struct AxisPlacement {
    int64_t global = 1, start = 0, count = 1;
    bool has_faces = false;
  };
  AxisPlacement ax1, ax2, ax3;
  int panel = 0;  //!< cubed-sphere panel (0 otherwise); layout 3 x 2

  //! index_select on every non-reduced field along `axis` (1..3) for the
  //! cells containing `coords`; false if this block holds none of them.
  //! Replaces SliceOutputData (output_type.cpp:415-493): one torch call per
  //! field, one host read of the face coordinates per axis (not per cell).
  bool slice(int axis, std::vector<double> const& coords);

  //! sum over `axis` (1..3); replaces SumOutputData (output_type.cpp:495-557)
  void sum(int axis);

  //! single device->host copy per field, cast to float64. This is the only
  //! host read of the gridded output path (see O.rst, question 7).
  void to_host();
};

//! The per-format backend. Each method maps to a handful of library calls;
//! all layout arithmetic is done once in the core.
class Sink {
 public:
  virtual ~Sink() = default;

  //! true: one file per process group written collectively (pnetcdf);
  //! false: one part per block, then combine_parts on the root (netcdf)
  virtual bool collective() const = 0;

  virtual void open(std::string const& path, Frame const& shape) = 0;
  virtual void define(Field const& f, bool double_precision) = 0;
  virtual void end_define() = 0;
  //! write one component; `start`/`count` already in file order
  //! (time, x1, x3, x2), as today (netcdf.cpp:277-282)
  virtual void put(Field const& f, int comp, std::vector<int64_t> const& start,
                   std::vector<int64_t> const& count) = 0;
  virtual void put_coords(Frame const& frame) = 0;
  virtual void close() = 0;
};

std::unique_ptr<Sink> make_netcdf_sink();   //!< NC_NETCDF4, NC_FLOAT|DOUBLE
std::unique_ptr<Sink> make_pnetcdf_sink();  //!< PNETCDFOUTPUT builds only;
                                            //!< float only, as today

//! The writer: schedule, statistics, one Sink. Replaces OutputType and the
//! NetcdfOutput / PNetcdfOutput / HDF5Output / FITSOutput classes
//! (output_formats.hpp). RestartOutput is separate (restart.hpp).
class GriddedWriter {
 public:
  //! `options` is the generated OutputOptions struct (options table)
  explicit GriddedWriter(std::shared_ptr<struct OutputOptionsImpl> options);

  // schedule; persisted by restart (restart.hpp, OutputSchedule)
  int file_number = 0;
  double next_time = 0.;
  int64_t schedule_key() const;     //!< legacy key, kept for old restarts
  int64_t schedule_key_v2() const;  //!< exact cadence bits (#313/#367)

  //! called after every accepted step, outside captured segments (lane X):
  //! running mean/std on device; no host read
  void accumulate_stats(Variables const& vars, torch::Tensor dt_device);

  //! build -> transform -> to_host -> sink -> combine (if requested)
  void write(MeshBlockImpl* pmb, Variables const& vars, double time,
             bool final_write);

 private:
  std::shared_ptr<struct OutputOptionsImpl> options_;
  std::unique_ptr<Sink> sink_;
  torch::Tensor prim_mean_, prim_m2_, scalar_mean_, scalar_m2_;
  double stat_elapsed_ = 0.;
};

//! Joins per-block netcdf parts into one file using only the attributes
//! snapy writes itself (domain_decomposition per axis, NumFilesInSet).
//! Replaces the call into mppnccombine (combine_netcdf.cpp:64-75) and the
//! 1,014-line mppnccombine.cpp. Also exposed as `snap-combine` in tools/ for
//! runs with `combine: false`. Returns 0 or a netcdf error code.
//! Hypothesis to gate: byte-identical variables to mppnccombine on the
//! example decks (O.rst, question 2).
int combine_parts(std::string const& outfile,
                  std::vector<std::string> const& parts);

// ---- kept from writer_core.hpp at the pin (PR #350), unchanged ----------
std::tuple<int, int, int, int, int, int> output_range(MeshBlockImpl const* pmb,
                                                      bool include_ghost_zones);
void make_output_dir(MeshBlockImpl const* pmb);
void combine_on_root(Layout const& layout, std::string const& key,
                     std::function<void()> const& root_work);
std::vector<std::string> glob_block_files(std::string const& pattern,
                                          char const* caller);

}  // namespace snap
