// C/C++
#include <algorithm>
#include <cstdlib>
#include <limits>

// harp
#include <harp/integrator/integrator.hpp>

#include "flux_register.hpp"

namespace snap {

namespace {

int index_of(std::vector<std::string> const& names, std::string const& name,
             char const* what) {
  auto it = std::find(names.begin(), names.end(), name);
  TORCH_CHECK(it != names.end(), "[FluxRegister] ", what, " '", name,
              "' was not declared in the RegisterRequest");
  return static_cast<int>(it - names.begin());
}

}  // namespace

void RegisterMonitor::reset(RegisterRequest const& req) {
  _rows = req.row_names;
  _counter = req.counters;
  _minimum = req.minima;
  _retry = req.retry;
  counters =
      torch::zeros({static_cast<int64_t>(_counter.size())}, torch::kInt64);
  minima =
      torch::full({static_cast<int64_t>(_minimum.size())},
                  std::numeric_limits<double>::infinity(), torch::kFloat64);
  retry_flags =
      torch::zeros({static_cast<int64_t>(_retry.size())}, torch::kInt32);
}

void RegisterMonitor::begin_step() {
  if (counters.numel()) counters.zero_();
  if (minima.numel()) minima.fill_(std::numeric_limits<double>::infinity());
  if (retry_flags.numel()) retry_flags.zero_();
}

int RegisterMonitor::counter_index(std::string const& name) const {
  return index_of(_counter, name, "counter");
}

int RegisterMonitor::minimum_index(std::string const& name) const {
  return index_of(_minimum, name, "minimum");
}

int RegisterMonitor::retry_index(std::string const& name) const {
  return index_of(_retry, name, "retry cause");
}

void RegisterMonitor::count(std::string const& name, torch::Tensor const& n) {
  counters[counter_index(name)].add_(n.sum().to(counters.scalar_type()));
}

void RegisterMonitor::record_min(std::string const& name,
                                 torch::Tensor const& v) {
  auto slot = minima[minimum_index(name)];
  slot.copy_(torch::minimum(slot, v.min().to(minima.scalar_type())));
}

void RegisterMonitor::flag_retry(std::string const& cause,
                                 torch::Tensor const& any) {
  auto slot = retry_flags[retry_index(cause)];
  slot.copy_(torch::maximum(slot, any.any().to(retry_flags.scalar_type())));
}

void FluxRegister::reset(int nvar, std::array<int64_t, 3> const& cells,
                         std::array<bool, 3> const& active, torch::Dtype kflux,
                         RegisterRequest const& req) {
  TORCH_CHECK(
      req.row_names.empty() || static_cast<int>(req.row_names.size()) == nvar,
      "[FluxRegister] ", req.row_names.size(), " row names for ", nvar,
      " rows");
  auto shape = std::vector<int64_t>{nvar, cells[0], cells[1], cells[2]};
  for (int d = 0; d < 3; ++d) {
    _total[d] = active[d] ? torch::zeros(shape, kflux) : torch::Tensor();
    _step[d] = torch::Tensor();
    _want_step[d] = false;
  }
  for (int d : req.step_field) request_step_field(d);
  _parts.clear();
  for (auto const& key : req.keep) {
    auto at = key.rfind('@');
    int d = at == std::string::npos ? 0 : std::atoi(key.c_str() + at + 1);
    TORCH_CHECK(d >= 1 && d <= 3 && _total[d - 1].defined(), "[FluxRegister] ",
                "kept part '", key, "' needs an active direction as @dim");
    _parts[key] = torch::zeros(shape, kflux);
  }
  _slot = {};
  _monitor.reset(req);
  _stage = -1;
  _w1 = _w2 = 0.;
}

void FluxRegister::request_step_field(int dim) {
  TORCH_CHECK(dim >= 1 && dim <= 3 && _total[dim - 1].defined(),
              "[FluxRegister] step field ", dim,
              " requested for an inactive direction");
  _want_step[dim - 1] = true;
}

void FluxRegister::begin_step() {
  for (int d = 0; d < 3; ++d) {
    if (!_want_step[d]) continue;
    auto const& t = _total[d];
    auto wide = t.is_mps() ? torch::kFloat32 : torch::kFloat64;  // MPS: no f64
    if (!_step[d].defined() || _step[d].device() != t.device() ||
        _step[d].scalar_type() != wide)
      _step[d] = torch::zeros(t.sizes(), t.options().dtype(wide));
    else
      _step[d].zero_();
  }
  _monitor.begin_step();
}

torch::Tensor FluxRegister::slot(int dim, std::string const& name) {
  TORCH_CHECK(dim >= 1 && dim <= 3 && _total[dim - 1].defined(),
              "[FluxRegister] slot '", name, "' in an inactive direction ",
              dim);
  return _total[dim - 1];
}

void FluxRegister::post_slot(int dim, std::string const& name) {
  _slot[dim - 1] = name;
}

bool FluxRegister::keeps(int dim, std::string const& part) const {
  return _parts.count(part + "@" + std::to_string(dim)) > 0;
}

torch::Tensor FluxRegister::part_slot(int dim, std::string const& part) {
  auto key = part + "@" + std::to_string(dim);
  TORCH_CHECK(_parts.count(key), "[FluxRegister] part '", key,
              "' was not requested in RegisterRequest::keep");
  return _parts.at(key);
}

torch::Tensor FluxRegister::kept(int dim, std::string const& part) const {
  auto key = part + "@" + std::to_string(dim);
  TORCH_CHECK(_parts.count(key), "[FluxRegister] part '", key,
              "' was not requested in RegisterRequest::keep");
  return _parts.at(key);
}

void FluxRegister::add(int dim, std::string const& name,
                       std::vector<int> const& rows,
                       torch::Tensor const& increment) {
  auto t = slot(dim, name);
  TORCH_CHECK(increment.size(0) == static_cast<int64_t>(rows.size()),
              "[FluxRegister] add '", name, "': ", increment.size(0),
              " increment rows for ", rows.size(), " rows");
  for (size_t i = 0; i < rows.size(); ++i) t[rows[i]].add_(increment[i]);
  if (keeps(dim, name)) {
    auto part = part_slot(dim, name);
    for (size_t i = 0; i < rows.size(); ++i) part[rows[i]].copy_(increment[i]);
  }
}

void FluxRegister::scale(int dim, std::string const& name,
                         std::vector<int> const& rows,
                         torch::Tensor const& factor) {
  auto t = slot(dim, name);
  bool per_row = factor.dim() == t.dim();
  TORCH_CHECK(!per_row || factor.size(0) == static_cast<int64_t>(rows.size()),
              "[FluxRegister] scale '", name, "': ", factor.size(0),
              " factor rows for ", rows.size(), " rows");
  for (size_t i = 0; i < rows.size(); ++i)
    t[rows[i]].mul_(per_row ? factor[i] : factor);
}

void FluxRegister::begin_stage(int stage, harp::IntegratorWeight const& w) {
  _stage = stage;
  _w1 = w.wght1();
  _w2 = w.wght2();
}

void FluxRegister::end_stage(double dt) {
  for (int d = 0; d < 3; ++d) {
    if (!_step[d].defined()) continue;
    // in FP64 (FP32 on MPS), whatever the flux dtype: S <- w1 S + w2 dt T
    _step[d].mul_(_w1).add_(_total[d].to(_step[d].scalar_type()), _w2 * dt);
  }
}

void FluxRegister::rollback() { begin_step(); }

}  // namespace snap
