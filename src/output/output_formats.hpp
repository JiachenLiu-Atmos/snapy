#pragma once

// C/C++
#include <string>
#include <vector>

// snap
#include "output_type.hpp"

namespace snap {
/*! \brief derived OutputType class for history dumps
class HistoryOutput : public OutputType {
 public:
  explicit HistoryOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};

//! \brief derived OutputType class for formatted table (tabular) data
class FormattedTableOutput : public OutputType {
 public:
  explicit FormattedTableOutput(OutputParameters oparams)
      : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};

//! \brief derived OutputType class for vtk dumps
class VTKOutput : public OutputType {
 public:
  explicit VTKOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};*/

//! \brief derived OutputType class for restart dumps
class RestartOutput : public OutputType {
 public:
  explicit RestartOutput(OutputOptions const& options_);
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
  void combine_blocks(MeshBlockImpl* pmb, bool) override;
};

/*class DebugOutput : public OutputType {
 public:
  explicit DebugOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};*/

class NetcdfOutput : public OutputType {
 public:
  explicit NetcdfOutput(OutputOptions const& options_);
  ~NetcdfOutput() {}

  //!  \brief Cycles over all MeshBlocks and writes OutputData in NETCDF format,
  ///         one MeshBlock per file
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
  void combine_blocks(MeshBlockImpl* pmb, bool) override;
};

class PNetcdfOutput : public OutputType {
 public:
  explicit PNetcdfOutput(OutputOptions const& options_);
  ~PNetcdfOutput() {}
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
  void combine_blocks(MeshBlockImpl* pmb, bool) override {}
};
}  // namespace snap
