// C/C++ headers
#include <cstring>
#include <sstream>
#include <stdexcept>

// snap
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/refine.hpp>

#include "output_formats.hpp"
#include "output_type.hpp"
#include "output_utils.hpp"

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

namespace {
std::string first_output_name(std::string const &name) {
  auto end = name.find_first_of(";,");
  return name.substr(0, end);
}

int data_dimension_for_axis(std::string const &grid, int physical_dim) {
  // Reduced outputs compact retained spatial axes into AthenaArray dimensions.
  if (grid.empty() || grid.size() != 3 || grid[0] != '-') {
    return physical_dim;
  }
  if (grid == "---") return 0;
  if (grid[1] == '-') return physical_dim == 1 ? 1 : 0;
  if (physical_dim == 1) return 0;
  return physical_dim - 1;
}
}  // namespace

OutputOptions OutputOptionsImpl::from_yaml(YAML::Node const &node, int fid) {
  auto options = OutputOptionsImpl::create();
  check_keys(node, "outputs/" + std::to_string(fid),
             {"dt", "output_sumx1", "output_sumx2", "output_sumx3",
              "include_ghost_zones", "cartesian_vector", "double_precision",
              "x1_slice", "x2_slice", "x3_slice", "type", "data_format",
              "variables", "combine", "verbose", "super-resolution"});

  options->fid() = fid;
  options->dt() = node["dt"].as<double>(0.);

  options->output_sumx1() = node["output_sumx1"].as<bool>(false);
  options->output_sumx2() = node["output_sumx2"].as<bool>(false);
  options->output_sumx3() = node["output_sumx3"].as<bool>(false);

  options->include_ghost_zones() = node["include_ghost_zones"].as<bool>(false);
  options->cartesian_vector() = node["cartesian_vector"].as<bool>(false);
  options->double_precision() = node["double_precision"].as<bool>(false);

  if (node["x1_slice"])
    options->x1_slice() = node["x1_slice"].as<std::vector<double>>();
  if (node["x2_slice"])
    options->x2_slice() = node["x2_slice"].as<std::vector<double>>();
  if (node["x3_slice"])
    options->x3_slice() = node["x3_slice"].as<std::vector<double>>();

  if (!options->x1_slice().empty() && options->output_sumx1()) {
    throw std::invalid_argument("Cannot request both slice and sum along x1");
  }
  if (!options->x2_slice().empty() && options->output_sumx2()) {
    throw std::invalid_argument("Cannot request both slice and sum along x2");
  }
  if (!options->x3_slice().empty() && options->output_sumx3()) {
    throw std::invalid_argument("Cannot request both slice and sum along x3");
  }

  if (node["type"]) {
    options->file_type() = node["type"].as<std::string>();
  } else {
    throw std::invalid_argument(
        "OutputOptions::from_yaml: output file type "
        "must be specified");
  }

  if (node["data_format"]) {
    options->data_format() = node["data_format"].as<std::string>();
  }

  if (node["variables"]) {
    options->variables() = node["variables"].as<std::vector<std::string>>();
  }

  if (node["combine"]) {
    options->combine() = node["combine"].as<bool>(true);
  }

  options->verbose() = node["verbose"].as<bool>(false);
  options->super_resolution() = node["super-resolution"].as<bool>(false);

  return options;
}

OutputType::OutputType(OutputOptions const &options_)
    : options(options_),
      pnext_type(),    // Terminate this node in singly linked list with nullptr
      num_vars_(),     // nested doubly linked list of OutputData:
      pfirst_data_(),  // Initialize head node to nullptr
      plast_data_() {  // Initialize tail node to nullptr
}

MeshBlockImpl *OutputType::LoadOutputData(MeshBlockImpl *pmb_in,
                                          Variables const &vars_in) {
  num_vars_ = 0;
  OutputData *pod;
  MeshBlockImpl *pmb;
  Variables vars;
  // set ptrs to data in OutputData linked list, then slice/sum as needed

  // create a refined output meshblock if super resolution is requested
  if (options->super_resolution()) {
    auto op = std::make_shared<MeshBlockOptionsImpl>(*(pmb_in->options));
    op->coord() = pmb_in->options->coord()->clone();
    // halving the spacing doubles ixN, so the refined slice still fits
    if (op->coord()->nx2() > 1) {
      op->coord()->nx2() *= 2;
      op->coord()->global_nx2() *= 2;
    }
    if (op->coord()->nx3() > 1) {
      op->coord()->nx3() *= 2;
      op->coord()->global_nx3() *= 2;
    }

    pmb = new MeshBlockImpl(op);
    // shall be deleted by caller of LoadOutputData

    auto peos = pmb->phydro->peos;
    auto pscalar = pmb->pscalar;

    int nghost = pmb->options->coord()->nghost();

    for (auto &[name, var] : vars_in) {
      auto interior_in = pmb_in->part(
          {0, 0, 0}, PartOptions().exterior(false).ndim(var.dim()));
      auto interior_out =
          pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(var.dim()));
      auto vec = var.sizes().vec();

      // dim 2
      if (vec.size() > 1 && vec[vec.size() - 2] > 1) {
        vec[vec.size() - 2] = 2 * (vec[vec.size() - 2] - nghost);
      }

      // dim 3
      if (vec.size() > 2 && vec[vec.size() - 3] > 1) {
        vec[vec.size() - 3] = 2 * (vec[vec.size() - 3] - nghost);
      }

      vars[name] = torch::zeros(vec, var.options());
      vars[name]
          .index(interior_out)
          .copy_(conservative_refine(var.index(interior_in)));
    }
    // vars["hydro_w"] = peos->compute("U->W", {vars["hydro_u"]});
    // scalar eos?
  } else {
    pmb = pmb_in;
    vars = vars_in;
  }

  loadHydroOutputData(pmb, vars);
  loadDiagOutputData(pmb, vars);
  loadScalarOutputData(pmb, vars);
  loadUserOutputData(pmb_in, vars);

  return pmb;
}

void OutputType::AppendOutputDataNode(OutputData *pnew_data) {
  if (pfirst_data_ == nullptr) {
    pfirst_data_ = pnew_data;
  } else {
    pnew_data->pprev = plast_data_;
    plast_data_->pnext = pnew_data;
  }
  // make the input node the new tail node of the doubly linked list
  plast_data_ = pnew_data;
}

void OutputType::ReplaceOutputDataNode(OutputData *pold, OutputData *pnew) {
  if (pold == pfirst_data_) {
    pfirst_data_ = pnew;
    if (pold->pnext != nullptr) {  // there is another node in the list
      pnew->pnext = pold->pnext;
      pnew->pnext->pprev = pnew;
    } else {  // there is only one node in the list
      plast_data_ = pnew;
    }
  } else if (pold == plast_data_) {
    plast_data_ = pnew;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
  } else {
    pnew->pnext = pold->pnext;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
    pnew->pnext->pprev = pnew;
  }
  delete pold;
}

void OutputType::ClearOutputData() {
  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    OutputData *pdata_old = pdata;
    pdata = pdata->pnext;
    delete pdata_old;
  }
  // reset pointers to head and tail nodes of doubly linked list:
  pfirst_data_ = nullptr;
  plast_data_ = nullptr;
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

void OutputType::appendTensorOutput(std::string type, std::string name,
                                    torch::Tensor const &tensor) {
  auto *pod = new OutputData;
  pod->type = std::move(type);
  pod->name = std::move(name);
  pod->data.CopyFromTensor(tensor);
  AppendOutputDataNode(pod);
  num_vars_++;
}

void OutputType::appendTensorSliceOutput(std::string type, std::string name,
                                         torch::Tensor const &tensor, int dim,
                                         int start, int count) {
  auto *pod = new OutputData;
  pod->type = std::move(type);
  pod->name = std::move(name);
  pod->data.InitFromTensor(tensor, dim, start, count);
  AppendOutputDataNode(pod);
  num_vars_ += count;
}

bool OutputType::TransformOutputData(MeshBlockImpl *pmb) {
  if (!options->x3_slice().empty() && !SliceOutputData(pmb, 3)) return false;
  if (!options->x2_slice().empty() && !SliceOutputData(pmb, 2)) return false;
  if (!options->x1_slice().empty() && !SliceOutputData(pmb, 1)) return false;

  if (options->output_sumx3()) SumOutputData(pmb, 3);
  if (options->output_sumx2()) SumOutputData(pmb, 2);
  if (options->output_sumx1()) SumOutputData(pmb, 1);
  return true;
}

bool OutputType::SliceOutputData(MeshBlockImpl *pmb, int dim) {
  auto pcoord = pmb->pcoord;
  auto coord_options = pcoord->options;

  auto const &slices = dim == 1   ? options->x1_slice()
                       : dim == 2 ? options->x2_slice()
                                  : options->x3_slice();
  auto &selected = dim == 1 ? islice : dim == 2 ? jslice : kslice;
  double lower = dim == 1   ? coord_options->x1min()
                 : dim == 2 ? coord_options->x2min()
                            : coord_options->x3min();
  double upper = dim == 1   ? coord_options->x1max()
                 : dim == 2 ? coord_options->x2max()
                            : coord_options->x3max();
  int begin = dim == 1 ? pcoord->il() : dim == 2 ? pcoord->jl() : pcoord->kl();
  int end = dim == 1 ? pcoord->iu() : dim == 2 ? pcoord->ju() : pcoord->ku();
  auto faces = dim == 1 ? pcoord->x1f : dim == 2 ? pcoord->x2f : pcoord->x3f;
  selected.clear();
  slice_offset[dim - 1] = 0;
  for (double slice : slices) {
    if (slice < lower) ++slice_offset[dim - 1];
    if (slice < lower || slice >= upper) continue;
    for (int i = begin; i <= end; ++i) {
      if (faces[i + 1].item<double>() > slice) {
        selected.push_back(i);
        break;
      }
    }
  }
  if (selected.empty()) return false;
  int count = selected.size();

  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto grid = MetadataTable::GetInstance()->GetGridType(
        first_output_name(pdata->name));
    int data_dim = data_dimension_for_axis(grid, dim);
    if (data_dim == 0) {
      pdata = pdata->pnext;
      continue;
    }

    auto *pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    pnew->longname = pdata->longname;
    pnew->units = pdata->units;

    int nx4 = pdata->data.GetDim4();
    int nx3 = pdata->data.GetDim3();
    int nx2 = pdata->data.GetDim2();
    int nx1 = pdata->data.GetDim1();
    pnew->data.NewAthenaArray(nx4, data_dim == 3 ? count : nx3,
                              data_dim == 2 ? count : nx2,
                              data_dim == 1 ? count : nx1);
    for (int n = 0; n < nx4; ++n)
      for (int k = 0; k < pnew->data.GetDim3(); ++k)
        for (int j = 0; j < pnew->data.GetDim2(); ++j)
          for (int i = 0; i < pnew->data.GetDim1(); ++i)
            pnew->data(n, k, j, i) =
                pdata->data(n, data_dim == 3 ? (nx3 > 1 ? selected[k] : 0) : k,
                            data_dim == 2 ? (nx2 > 1 ? selected[j] : 0) : j,
                            data_dim == 1 ? (nx1 > 1 ? selected[i] : 0) : i);
    ReplaceOutputDataNode(pdata, pnew);
    pdata = pnew->pnext;
  }

  if (dim == 3) {
    out_ks = 0;
    out_ke = count - 1;
  } else if (dim == 2) {
    out_js = 0;
    out_je = count - 1;
  } else {
    out_is = 0;
    out_ie = count - 1;
  }
  return true;
}

void OutputType::SumOutputData(MeshBlockImpl *, int dim) {
  int sum_begin = dim == 3 ? out_ks : (dim == 2 ? out_js : out_is);
  int sum_end = dim == 3 ? out_ke : (dim == 2 ? out_je : out_ie);

  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto grid = MetadataTable::GetInstance()->GetGridType(
        first_output_name(pdata->name));
    int data_dim = data_dimension_for_axis(grid, dim);
    if (data_dim == 0) {
      pdata = pdata->pnext;
      continue;
    }

    auto *pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    pnew->longname = pdata->longname;
    pnew->units = pdata->units;

    int nx4 = pdata->data.GetDim4();
    int nx3 = pdata->data.GetDim3();
    int nx2 = pdata->data.GetDim2();
    int nx1 = pdata->data.GetDim1();

    if (data_dim == 3) {
      pnew->data.NewAthenaArray(nx4, 1, nx2, nx1);
      for (int n = 0; n < nx4; ++n)
        for (int k = sum_begin; k <= sum_end; ++k)
          for (int j = 0; j < nx2; ++j)
            for (int i = 0; i < nx1; ++i)
              pnew->data(n, 0, j, i) += pdata->data(n, k, j, i);
    } else if (data_dim == 2) {
      pnew->data.NewAthenaArray(nx4, nx3, 1, nx1);
      for (int n = 0; n < nx4; ++n)
        for (int k = 0; k < nx3; ++k)
          for (int j = sum_begin; j <= sum_end; ++j)
            for (int i = 0; i < nx1; ++i)
              pnew->data(n, k, 0, i) += pdata->data(n, k, j, i);
    } else {
      pnew->data.NewAthenaArray(nx4, nx3, nx2, 1);
      for (int n = 0; n < nx4; ++n)
        for (int k = 0; k < nx3; ++k)
          for (int j = 0; j < nx2; ++j)
            for (int i = sum_begin; i <= sum_end; ++i)
              pnew->data(n, k, j, 0) += pdata->data(n, k, j, i);
    }

    ReplaceOutputDataNode(pdata, pnew);
    pdata = pnew->pnext;
  }

  if (dim == 3) {
    out_ks = 0;
    out_ke = 0;
  } else if (dim == 2) {
    out_js = 0;
    out_je = 0;
  } else {
    out_is = 0;
    out_ie = 0;
  }
}

}  // namespace snap
