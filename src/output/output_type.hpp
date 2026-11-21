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

  //! the `domain` checks of src/options/output.yaml against the resolved
  //! global grid: per axis not both slice and sum, every slice in [min, max),
  //! strictly increasing; std::invalid_argument otherwise. MeshBlockImpl runs
  //! them for every output, read from a deck or built in code.
  void check_domain(CoordinateOptionsImpl const& co) const;

  // the option rows and report(), generated from src/options/output.yaml by
  // tools/gen_options.py
#include <snap_options/output_members.inc>
};
using OutputOptions = std::shared_ptr<OutputOptionsImpl>;

struct Field;
struct Frame;

//! one scheduled output: its options, schedule and running statistics
class OutputType {
 public:
  OutputOptions options;

  int file_number = 0;
  double next_time = 0.0;

  //! Legacy restart identity; retained so older readers can load new files.
  int64_t schedule_key() const;

  //! Restart identity with the exact cadence bits.
  int64_t schedule_key_v2() const;

  OutputType();
  explicit OutputType(OutputOptions const &options_);
  virtual ~OutputType();

  //! the fields the options request, on the block's device
  std::shared_ptr<Frame> LoadOutputData(MeshBlockImpl *pmb,
                                        Variables const &vars);

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
  virtual void write_output_file(MeshBlockImpl *pmb, Variables const &vars,
                                 double time, bool flag) {}
  virtual void combine_blocks(MeshBlockImpl *pmb, bool) {}

 protected:
  bool shouldOutputPrimitive(
      std::initializer_list<std::string> vars = {}) const;
  bool shouldOutputConserved(
      std::initializer_list<std::string> vars = {}) const;
  //! the whole tensor (1-3 dimensions), or `count` rows of it from `start`
  void appendTensorOutput(std::string name, torch::Tensor const &tensor);
  void appendTensorSliceOutput(std::string name, torch::Tensor const &tensor,
                               int start, int count);
  void loadHydroOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadDiagOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadScalarOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadUserOutputData(MeshBlockImpl *pmb, Variables const &vars);

  std::shared_ptr<Frame> frame_;  //!< the fields being loaded

  torch::Tensor prim_stat_mean_;
  torch::Tensor prim_stat_m2_;
  torch::Tensor scalar_stat_mean_;
  torch::Tensor scalar_stat_m2_;
  double stat_elapsed_ = 0.0;
  double stat_last_time_ = 0.0;
  bool stat_initialized_ = false;
};

//! restart dumps (merged with the reader in O7)
class RestartOutput : public OutputType {
 public:
  explicit RestartOutput(OutputOptions const &options_);
  void write_output_file(MeshBlockImpl *pmb, Variables const &vars, double time,
                         bool final_write) override;
  void combine_blocks(MeshBlockImpl *pmb, bool) override;
};
}  // namespace snap

#undef ADD_ARG
