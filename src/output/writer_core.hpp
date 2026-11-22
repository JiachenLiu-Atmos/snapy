#pragma once

// C/C++
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// snap
#include <snap/layout/layout.hpp>

#include "output_type.hpp"

namespace snap {

class MeshBlockImpl;

//! one output variable group: its names (';'-separated, or one name with an
//! optional '?' for the component number) and its values. After
//! Frame::to_host() the values are CPU float64, shaped [nx4, nx3, nx2, nx1].
struct Field {
  std::string name;
  torch::Tensor data;
  std::string grid() const;  //!< grid kind of the first name ("" = cells)
  int nvar() const;          //!< components, counted by grid kind
  //! the names written, one per component
  std::vector<std::string> names() const;
};

//! the fields of one write, the output index range and the slice placement
struct Frame {
  std::vector<Field> fields;
  int is = 0, ie = 0, js = 0, je = 0, ks = 0, ke = 0;
  std::array<std::vector<int>, 3> slice;  //!< selected cells, x1..x3
  std::array<int, 3> slice_offset = {0, 0, 0};

  //! one copy per field to the host, cast to float64 once (#350)
  void to_host();
  //! slices, then sums; false when a slice misses this block
  bool transform(MeshBlockImpl const* pmb, OutputOptionsImpl const& op);
  //! how f is written: 'c' cells, '1'-'3' faces along x1-x3, 'z' a column
  //! (--C), 'f' a face column (--F), 'p' a plane (-CC), 's' a scalar (---)
  char kind(Field const& f) const;
  //! component n of f as written, contiguous, i outermost then k then j
  torch::Tensor pack(Field const& f, int n) const;
};

//! a file format: writes one Frame
class Sink {
 public:
  virtual ~Sink() = default;
  virtual void write(MeshBlockImpl* pmb, Frame const& fr,
                     OutputOptionsImpl const& op, int file_number,
                     double time) = 0;
  //! joins the block files of this output, if the format needs it
  virtual void combine(MeshBlockImpl* pmb, OutputOptionsImpl const& op,
                       int file_number) {}
};

//! netcdf (one file per block, combined) or pnetcdf (one shared file)
class GriddedWriter : public OutputType {
 public:
  //! the sink of options_->file_type() unless one is given
  explicit GriddedWriter(OutputOptions const& options_,
                         std::shared_ptr<Sink> sink = nullptr);
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
  void combine_blocks(MeshBlockImpl* pmb, bool) override;

 private:
  std::shared_ptr<Sink> sink_;
};

std::shared_ptr<Sink> make_netcdf_sink();
//! merges the part files into a new netcdf file `outfile`, then removes the
//! parts (the `-r <outfile> <parts...>` form of the mppnccombine it replaced,
//! O.rst 10.1). Returns 0 or a netcdf error code.
int combine_parts(std::string const& outfile,
                  std::vector<std::string> const& parts);
//! joins the block files of one netcdf output into <basename>.<fileid>.<n>.nc
void combine_netcdf_blocks(MeshBlockImpl* pmb, OutputOptionsImpl const& op,
                           int file_number);
std::shared_ptr<Sink> make_pnetcdf_sink(OutputOptionsImpl const& op);

//! a netcdf-legal name: [A-Za-z_][A-Za-z0-9_]*
std::string sanitize_netcdf_name(std::string name);

//! interior index range [is, ie] x [js, je] x [ks, ke] of a block's output,
//! widened by the ghost zones in every dimension with more than one cell when
//! include_ghost_zones is set (only checked for cell-centred grids)
std::tuple<int, int, int, int, int, int> output_range(MeshBlockImpl const* pmb,
                                                      bool include_ghost_zones);

//! create the block's output directory; throws with the filesystem error
void make_output_dir(MeshBlockImpl const* pmb);

//! where a block sits in the combined output grid
struct OutputBlockLocation {
  int level, lx1, lx2, lx3;  //!< refinement level, block indices
  int nb1, nb2, nb3;         //!< blocks per direction
  int face;                  //!< cubed-sphere panel (0 otherwise)
  int panel_col, panel_row;  //!< its place in the 3 x 2 panel layout
};

//! the location of `rank`; cubed-sphere panels are laid out 3 x 2 in (x2, x3)
OutputBlockLocation output_block_location(Layout const& layout, int rank);

//! the per-process combine protocol shared by the block writers: every block
//! of a process counts in under `key` and only the last one combines. All
//! ranks of the process group meet before root_work runs on the root rank and
//! again after it, also when it throws, so no rank returns before the combined
//! file exists.
void combine_on_root(Layout const& layout, std::string const& key,
                     std::function<void()> const& root_work);

//! the files matching `pattern` (glob, GLOB_TILDE); throws, naming `caller`,
//! if glob() fails
std::vector<std::string> glob_block_files(std::string const& pattern,
                                          char const* caller);

}  // namespace snap
