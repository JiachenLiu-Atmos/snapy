// C/C++ headers
#include <cstring>
#include <sstream>
#include <stdexcept>

// snap
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "writer_core.hpp"

namespace snap {
int64_t OutputType::schedule_key() const {
  std::string id = options->file_type() + "|" + std::to_string(options->dt());
  for (auto const &v : options->variables()) id += "|" + v;
  uint64_t h = 14695981039346656037ull;
  for (unsigned char c : id) h = (h ^ c) * 1099511628211ull;
  return static_cast<int64_t>(h);
}

int64_t OutputType::schedule_key_v2() const {
  auto append = [](uint64_t &hash, unsigned char byte) {
    hash = (hash ^ byte) * 1099511628211ull;
  };
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : options->file_type()) append(hash, c);
  append(hash, 0);

  static_assert(sizeof(double) == sizeof(uint64_t));
  uint64_t cadence;
  double dt = options->dt();
  std::memcpy(&cadence, &dt, sizeof(cadence));
  for (int shift = 0; shift < 64; shift += 8) {
    append(hash, static_cast<unsigned char>(cadence >> shift));
  }
  for (auto const &variable : options->variables()) {
    append(hash, 0);
    for (unsigned char c : variable) append(hash, c);
  }
  return static_cast<int64_t>(hash);
}

#include <snap_options/output_read.inc>

OutputOptions OutputOptionsImpl::from_yaml(YAML::Node const &node, int fid) {
  auto options = OutputOptionsImpl::create();
  options->fid() = fid;
  // the rows of src/options/output.yaml
  read_output_rows(*options, node, "outputs/" + std::to_string(fid));
  return options;
}

OutputType::OutputType() : options(OutputOptionsImpl::create()) {}
OutputType::OutputType(OutputOptions const &options_) : options(options_) {}
OutputType::~OutputType() = default;

std::shared_ptr<Frame> OutputType::LoadOutputData(MeshBlockImpl *pmb,
                                                  Variables const &vars) {
  frame_ = std::make_shared<Frame>();
  loadHydroOutputData(pmb, vars);
  loadDiagOutputData(pmb, vars);
  loadScalarOutputData(pmb, vars);
  loadUserOutputData(pmb, vars);
  return frame_;
}

bool OutputType::ContainAnyVariable(
    std::initializer_list<std::string> vars) const {
  for (auto const &var : vars) {
    if (std::find(options->variables().begin(), options->variables().end(),
                  var) != options->variables().end()) {
      return true;
    }
  }
  return false;
}

bool OutputType::ContainVariable(const std::string &var) const {
  return std::find(options->variables().begin(), options->variables().end(),
                   var) != options->variables().end();
}

bool OutputType::OutputsPrimStat() const {
  return ContainVariable("prim_stat");
}

bool OutputType::OutputsScalarStat() const {
  return ContainVariable("scalar_stat");
}

bool OutputType::OutputsAnyStat() const {
  return OutputsPrimStat() || OutputsScalarStat();
}

bool OutputType::shouldOutputPrimitive(
    std::initializer_list<std::string> vars) const {
  return ContainVariable("prim") || ContainAnyVariable(vars);
}

bool OutputType::shouldOutputConserved(
    std::initializer_list<std::string> vars) const {
  return ContainVariable("cons") || ContainAnyVariable(vars);
}

namespace {
void update_weighted_moments(torch::Tensor const &value, double weight,
                             double previous_weight, torch::Tensor &mean,
                             torch::Tensor &m2) {
  if (!mean.defined()) {
    mean = torch::zeros_like(value);
    m2 = torch::zeros_like(value);
  }

  auto delta = value - mean;
  double total_weight = previous_weight + weight;
  mean.add_(delta * (weight / total_weight));
  m2.add_(weight * delta * (value - mean));
}
}  // namespace

void OutputType::AccumulateStats(Variables const &vars, double current_time) {
  if (!OutputsAnyStat()) return;
  if (!stat_initialized_) {
    stat_last_time_ = current_time;
    stat_initialized_ = true;
    return;
  }

  double dt = current_time - stat_last_time_;
  if (dt > 0.0) {
    if (OutputsPrimStat()) {
      auto it = vars.find("hydro_w");
      if (it != vars.end() && it->second.defined()) {
        update_weighted_moments(it->second, dt, stat_elapsed_, prim_stat_mean_,
                                prim_stat_m2_);
      }
    }

    if (OutputsScalarStat()) {
      auto it = vars.find("scalar_r");
      if (it != vars.end() && it->second.defined()) {
        update_weighted_moments(it->second, dt, stat_elapsed_,
                                scalar_stat_mean_, scalar_stat_m2_);
      }
    }

    stat_elapsed_ += dt;
  }
  stat_last_time_ = current_time;
}

void OutputType::ResetStats(double current_time) {
  if (!OutputsAnyStat()) return;
  if (prim_stat_mean_.defined()) {
    prim_stat_mean_.zero_();
  }
  if (prim_stat_m2_.defined()) {
    prim_stat_m2_.zero_();
  }
  if (scalar_stat_mean_.defined()) {
    scalar_stat_mean_.zero_();
  }
  if (scalar_stat_m2_.defined()) {
    scalar_stat_m2_.zero_();
  }
  stat_elapsed_ = 0.0;
  stat_last_time_ = current_time;
  stat_initialized_ = true;
}

namespace {
// the running mean, or the current value before any time has accumulated
torch::Tensor stat_mean(torch::Tensor const &mean, torch::Tensor const &current,
                        double elapsed) {
  if (elapsed <= 0.0 || !mean.defined()) return current;
  return mean;
}

// sqrt(M2 / elapsed) (Welford), zero before any time has accumulated
torch::Tensor stat_std(torch::Tensor const &m2, torch::Tensor const &current,
                       double elapsed) {
  if (elapsed <= 0.0 || !m2.defined()) return torch::zeros_like(current);
  auto variance = m2 / elapsed;
  return torch::sqrt(torch::clamp_min(variance, 0.0));
}
}  // namespace

torch::Tensor OutputType::PrimStatMean(torch::Tensor const &current) const {
  return stat_mean(prim_stat_mean_, current, stat_elapsed_);
}

torch::Tensor OutputType::PrimStatStd(torch::Tensor const &current) const {
  return stat_std(prim_stat_m2_, current, stat_elapsed_);
}

torch::Tensor OutputType::ScalarStatMean(torch::Tensor const &current) const {
  return stat_mean(scalar_stat_mean_, current, stat_elapsed_);
}

torch::Tensor OutputType::ScalarStatStd(torch::Tensor const &current) const {
  return stat_std(scalar_stat_m2_, current, stat_elapsed_);
}

void OutputType::appendTensorOutput(std::string name,
                                    torch::Tensor const &tensor) {
  TORCH_CHECK(tensor.dim() >= 1 && tensor.dim() <= 3, "output '", name,
              "': only 1D, 2D, or 3D is supported");
  frame_->fields.push_back({std::move(name), tensor});
}

void OutputType::appendTensorSliceOutput(std::string name,
                                         torch::Tensor const &tensor, int start,
                                         int count) {
  frame_->fields.push_back(
      {std::move(name), tensor.slice(0, start, start + count)});
}

}  // namespace snap
