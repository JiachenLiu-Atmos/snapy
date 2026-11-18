#pragma once

// C/C++
#include <initializer_list>
#include <memory>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// base
#include <configure.h>

// kintera
#include <kintera/utils/format.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/interface/athena_arrays.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! \brief  container for parameters read from `<output>` block in the input
struct OutputOptionsImpl {
  static std::shared_ptr<OutputOptionsImpl> create() {
    return std::make_shared<OutputOptionsImpl>();
  }
  static std::shared_ptr<OutputOptionsImpl> from_yaml(YAML::Node const &node,
                                                      int fid = 0);
  std::string file_id() const { return "out" + std::to_string(fid()); }

  // the option rows and report(), generated from src/options/output.yaml by
  // tools/gen_options.py
#include <snap_options/output_members.inc>
};
using OutputOptions = std::shared_ptr<OutputOptionsImpl>;

//! \brief container for output data and metadata; node in nested doubly linked
//! list
struct OutputData {
  std::string type;  // one of (SCALARS,VECTORS) used for vtk outputs
  std::string name;
  std::string longname;
  std::string units;

  AthenaArray<double> data;  // array containing data

  // ptrs to previous and next nodes in doubly linked list:
  OutputData *pnext, *pprev;

  OutputData() : pnext(nullptr), pprev(nullptr) {}
};

//! OutputType is designed to be a node in a singly linked list created & stored
//! in the Output class
class OutputType {
 public:
  OutputOptions options;

  int file_number = 0;
  double next_time = 0.0;

  //! Legacy restart identity; retained so older readers can load new files.
  int64_t schedule_key() const;

  //! Restart identity with the exact cadence bits.
  int64_t schedule_key_v2() const;

  // constructors
  OutputType() : options(OutputOptionsImpl::create()) {}

  // mark single parameter constructors as "explicit" to prevent them from
  // acting as implicit conversion functions: for f(OutputType arg), prevent
  // f(anOutputParameters)
  explicit OutputType(OutputOptions const &options_);

  // rule of five:
  virtual ~OutputType() = default;
  // copy constructor and assignment operator (pnext_type, pfirst_data, etc. are
  // shallow copied)
  OutputType(const OutputType &copy_other) = default;
  OutputType &operator=(const OutputType &copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType &&) = default;
  OutputType &operator=(OutputType &&) = default;

  // data
  // OutputData array start/end index
  int out_is, out_ie, out_js, out_je, out_ks, out_ke;
  std::vector<int> islice, jslice, kslice;
  int slice_offset[3] = {0, 0, 0};

  // ptr to next node in singly linked list of OutputTypes
  OutputType *pnext_type;

  // functions
  //! \brief Create doubly linked list of OutputData's containing requested
  //! variables
  MeshBlockImpl *LoadOutputData(MeshBlockImpl *pmb, Variables const &vars);

  void AppendOutputDataNode(OutputData *pdata);
  void ReplaceOutputDataNode(OutputData *pold, OutputData *pnew);
  void ClearOutputData();

  bool TransformOutputData(MeshBlockImpl *pmb);

  //! \brief perform data slicing and update the data list
  bool SliceOutputData(MeshBlockImpl *pmb, int dim);

  //! \brief perform data summation and update the data list
  void SumOutputData(MeshBlockImpl *pmb, int dim);

  //! \brief Convert vectors in curvilinear coordinates into Cartesian
  void CalculateCartesianVector(torch::Tensor const &src, torch::Tensor dst,
                                Coordinate pco);
  bool ContainVariable(const std::string &var) const;
  bool ContainAnyVariable(std::initializer_list<std::string> vars) const;
  bool OutputsPrimStat() const;
  bool OutputsScalarStat() const;
  bool OutputsAnyStat() const;
  void AccumulateStats(Variables const &vars, double current_time);
  void ResetStats(double current_time);
  torch::Tensor PrimStatMean(torch::Tensor const &current) const;
  torch::Tensor PrimStatStd(torch::Tensor const &current) const;
  torch::Tensor ScalarStatMean(torch::Tensor const &current) const;
  torch::Tensor ScalarStatStd(torch::Tensor const &current) const;
  // following pure virtual function must be implemented in all derived classes
  virtual void write_output_file(MeshBlockImpl *pmb, Variables const &vars,
                                 double time, bool flag) {}
  virtual void combine_blocks(MeshBlockImpl *pmb, bool) {}

 protected:
  bool shouldOutputPrimitive(
      std::initializer_list<std::string> vars = {}) const;
  bool shouldOutputConserved(
      std::initializer_list<std::string> vars = {}) const;
  void appendTensorOutput(std::string type, std::string name,
                          torch::Tensor const &tensor);
  void appendTensorSliceOutput(std::string type, std::string name,
                               torch::Tensor const &tensor, int dim, int start,
                               int count);
  void loadHydroOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadDiagOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadScalarOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadUserOutputData(MeshBlockImpl *pmb, Variables const &vars);

  int num_vars_;  // number of variables in output
  // nested doubly linked list of OutputData nodes (of the same OutputType):

  // ptr to head OutputData node in doubly linked list
  OutputData *pfirst_data_;

  // ptr to tail OutputData node in doubly linked list
  OutputData *plast_data_;

  torch::Tensor prim_stat_mean_;
  torch::Tensor prim_stat_m2_;
  torch::Tensor scalar_stat_mean_;
  torch::Tensor scalar_stat_m2_;
  double stat_elapsed_ = 0.0;
  double stat_last_time_ = 0.0;
  bool stat_initialized_ = false;
};
}  // namespace snap

#undef ADD_ARG
