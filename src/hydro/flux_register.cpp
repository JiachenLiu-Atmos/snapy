// C/C++
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>

// harp
#include <harp/integrator/integrator.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/mesh/precision_policy.hpp>

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
  if (_step_amount.defined()) {
    _step_amount.zero_();
    _stage_amount.zero_();
    _stage_post.zero_();
  }
}

int RegisterMonitor::counter_index(std::string const& name) const {
  return index_of(_counter, name, "counter");
}

int RegisterMonitor::retry_index(std::string const& name) const {
  return index_of(_retry, name, "retry cause");
}

void RegisterMonitor::count(std::string const& name, torch::Tensor const& n) {
  counters[counter_index(name)].add_(n.sum().to(counters.scalar_type()));
}

void RegisterMonitor::record_min(std::string const& name,
                                 torch::Tensor const& v) {
  auto slot = minima[index_of(_minimum, name, "minimum")];
  slot.copy_(torch::minimum(slot, v.min().to(minima.scalar_type())));
}

void RegisterMonitor::flag_retry(std::string const& cause,
                                 torch::Tensor const& any) {
  auto slot = retry_flags[retry_index(cause)];
  slot.copy_(torch::maximum(slot, any.any().to(retry_flags.scalar_type())));
}

void RegisterMonitor::reset_amounts(int nrows) {
  _amount = {"reconcile.x1", "reconcile.panel", "boundary.prescribed"};
  _kind.assign(_amount.size(), AmountKind::flux_correction);
  auto shape =
      std::vector<int64_t>{static_cast<int64_t>(_amount.size()), nrows};
  _step_amount = torch::zeros(shape, torch::kFloat64);
  _stage_amount = torch::zeros(shape, torch::kFloat64);
  _stage_post = torch::zeros(shape, torch::kFloat64);
}

void RegisterMonitor::place_amounts(torch::Device device) {
  auto wide = device.is_mps() ? torch::kFloat32 : torch::kFloat64;  // no f64
  if (_step_amount.device() == device && _step_amount.scalar_type() == wide)
    return;
  _step_amount = _step_amount.to(device, wide);
  _stage_amount = _stage_amount.to(device, wide);
  _stage_post = _stage_post.to(device, wide);
}

int RegisterMonitor::amount_index(std::string const& name) const {
  return index_of(_amount, name, "amount");
}

AmountKind RegisterMonitor::kind(std::string const& name) const {
  return _kind[amount_index(name)];
}

torch::Tensor RegisterMonitor::amounts(std::string const& name) const {
  return _step_amount[amount_index(name)];
}

void RegisterMonitor::book(std::string const& name, torch::Tensor const& rate) {
  int i = amount_index(name);
  place_amounts(rate.device());
  _stage_amount[i].add_(rate.to(_stage_amount.scalar_type()));
}

void RegisterMonitor::post(std::string const& name, AmountKind kind,
                           torch::Tensor const& amount, double weight,
                           bool in_stage) {
  TORCH_CHECK(_step_amount.defined(), "[FluxRegister] amount '", name,
              "' before the register was reset");
  auto it = std::find(_amount.begin(), _amount.end(), name);
  int i = static_cast<int>(it - _amount.begin());
  if (it == _amount.end()) {  // created at its first post, with its kind
    _amount.push_back(name);
    _kind.push_back(kind);
    auto row = [](torch::Tensor const& t) {
      return torch::cat({t, torch::zeros_like(t.narrow(0, 0, 1))}, 0);
    };
    _step_amount = row(_step_amount);
    _stage_amount = row(_stage_amount);
    _stage_post = row(_stage_post);
  }
  TORCH_CHECK(_kind[i] == kind, "[FluxRegister] amount '", name,
              "' was first posted with another AmountKind");
  TORCH_CHECK(amount.dim() == 1 && amount.size(0) == _step_amount.size(1),
              "[FluxRegister] amount '", name, "' must be (",
              _step_amount.size(1), ",); got ", amount.sizes());
  place_amounts(amount.device());
  auto& to = in_stage ? _stage_post : _step_amount;
  to[i].add_(amount.to(to.scalar_type()), weight);
}

void RegisterMonitor::end_stage(double w1, double w2dt) {
  if (!_step_amount.defined()) return;
  _step_amount.mul_(w1).add_(_stage_amount, w2dt).add_(_stage_post);
  _stage_amount.zero_();
  _stage_post.zero_();
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
  _kept = {};
  for (auto const& key : req.keep) {
    auto at = key.rfind('@');
    int d = at == std::string::npos ? 0 : std::atoi(key.c_str() + at + 1);
    TORCH_CHECK(d >= 1 && d <= 3 && _total[d - 1].defined(), "[FluxRegister] ",
                "kept part '", key, "' needs an active direction as @dim");
    _parts[key] = torch::zeros(shape, kflux);
    _kept[d - 1].push_back(key.substr(0, at));
  }
  _pmask = {};
  _pvalue = {};
  _coord = nullptr;
  _slot = {};
  _monitor.reset(req);
  _monitor.reset_amounts(nvar);
  _stage = -1;
  _w1 = _w2 = 0.;
  _in_stage = false;
}

void FluxRegister::reset(CoordinateImpl const* coord,
                         PrecisionPolicy const& policy, int nvar,
                         RegisterRequest const& req) {
  TORCH_CHECK(coord, "[FluxRegister] reset needs a coordinate");
  auto const& co = coord->options;
  int64_t nc1 = co->nc1(), nc2 = co->nc2(), nc3 = co->nc3();
  reset(nvar, {nc3, nc2, nc1}, {nc1 > 1, nc2 > 1, nc3 > 1},
        policy.storage(Quantity::kFlux), req);
  _coord = coord;
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
  _rec = _direct = {};  // the step's split posts start again
  _in_stage = false;
  for (auto const& t : _total)
    if (t.defined() && _monitor._step_amount.defined()) {
      _monitor.place_amounts(t.device());
      break;
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
  return kept(dim, part);
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

void FluxRegister::overwrite_(int dim, std::string const& name,
                              torch::Tensor const& after) {
  TORCH_CHECK(_coord, "[FluxRegister] '", name,
              "' books A_f (after - before): reset the register with its "
              "coordinate");
  auto t = _total[dim - 1];
  auto wide = t.is_mps() ? torch::kFloat32 : torch::kFloat64;
  // A_f of face index i of T_d is face_area_d() at i (CoordinateImpl's
  // divergence pairs flux index i with area index i)
  auto area = dim == 1   ? _coord->face_area1().narrow(2, 0, t.size(3))
              : dim == 2 ? _coord->face_area2().narrow(1, 0, t.size(2))
                         : _coord->face_area3().narrow(0, 0, t.size(1));
  area = area.to(t.device(), wide);
  // only the faces that changed contribute: elsewhere after - before is 0
  auto rate = (area * (after.to(wide) - t.to(wide))).sum({1, 2, 3});
  t.copy_(after);
  _monitor.book(name, rate);
}

void FluxRegister::replace(int dim, torch::Tensor const& faces,
                           torch::Tensor const& peer, SeamRule rule) {
  auto t = slot(dim, "replace");
  auto face_shape = t.sizes().slice(1);
  TORCH_CHECK(
      faces.scalar_type() == torch::kBool && faces.sizes() == face_shape,
      "[FluxRegister] replace: faces must be a bool mask shaped like "
      "total(",
      dim, ") without the row axis, ", face_shape, "; got ", faces.sizes());
  auto const& kept = _kept[dim - 1];
  int64_t nvar = t.size(0);
  int64_t nrows = nvar * static_cast<int64_t>(1 + kept.size());
  TORCH_CHECK(peer.dim() == t.dim() && peer.size(0) == nrows &&
                  peer.sizes().slice(1) == face_shape,
              "[FluxRegister] replace: peer must be [", nrows, ", ", face_shape,
              "] (the total, then ", kept.size(), " kept parts); got ",
              peer.sizes());
  TORCH_CHECK(peer.scalar_type() == t.scalar_type() &&
                  peer.device() == t.device() && faces.device() == t.device(),
              "[FluxRegister] replace: peer and faces must be on the total's "
              "device, peer in its dtype");
  auto merged = [&](torch::Tensor const& own, torch::Tensor const& other) {
    // (own + peer) * 0.5 is the same on both sides: + commutes
    auto value = rule == SeamRule::mean ? (own + other) * 0.5 : other;
    return torch::where(faces, value, own);
  };
  for (size_t k = 0; k < kept.size(); ++k) {
    auto part = _parts.at(kept[k] + "@" + std::to_string(dim));
    auto row0 = nvar * static_cast<int64_t>(k + 1);
    part.copy_(merged(part, peer.narrow(0, row0, nvar)));
  }
  overwrite_(dim, dim == 1 ? "reconcile.x1" : "reconcile.panel",
             merged(t, peer.narrow(0, 0, nvar)));
}

void FluxRegister::set_prescribed(int dim, torch::Tensor const& mask,
                                  torch::Tensor const& value) {
  auto t = slot(dim, "boundary.prescribed");
  TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.sizes() == t.sizes() &&
                  value.sizes() == t.sizes(),
              "[FluxRegister] set_prescribed: mask (bool) and value must be "
              "shaped like total(",
              dim, "), ", t.sizes());
  _pmask[dim - 1] = mask;
  _pvalue[dim - 1] = value;
  _pwall_set[dim - 1] = false;
}

void FluxRegister::set_prescribed_walls(int dim,
                                        std::vector<int64_t> const& planes,
                                        std::vector<int> const& free_rows) {
  auto t = slot(dim, "boundary.prescribed");
  int d = dim - 1, ax = 4 - dim;  // the face axis of [nvar, nc3, nc2, nc1]
  for (auto p : planes)
    TORCH_CHECK(p >= 0 && p < t.size(ax),
                "[FluxRegister] set_prescribed_walls: plane ", p,
                " is outside total(", dim, ")'s ", t.size(ax), " faces");
  for (int r : free_rows)
    TORCH_CHECK(r >= 0 && r < t.size(0),
                "[FluxRegister] set_prescribed_walls: row ", r, " of ",
                t.size(0));
  _pmask[d] = _pvalue[d] = torch::Tensor();
  _pwall[d] = planes;
  _pwall_set[d] = true;
  auto& rows = _prow[d];
  if (!rows.defined() || _pfree[d] != free_rows || rows.size(0) != t.size(0) ||
      rows.device() != t.device()) {
    rows = torch::ones({t.size(0), 1, 1}, t.options().dtype(torch::kBool));
    for (int r : free_rows) rows[r].fill_(false);
    _pfree[d] = free_rows;
  }
}

void FluxRegister::apply_wall_planes_(int dim) {
  int d = dim - 1, ax = 4 - dim;
  if (_pwall[d].empty()) return;
  TORCH_CHECK(_coord,
              "[FluxRegister] 'boundary.prescribed' books A_f (after - "
              "before): reset the register with its coordinate");
  auto t = _total[d];
  auto& rows = _prow[d];
  if (rows.device() != t.device()) rows = rows.to(t.device());
  auto wide = t.is_mps() ? torch::kFloat32 : torch::kFloat64;
  auto area = dim == 1   ? _coord->face_area1().narrow(2, 0, t.size(3))
              : dim == 2 ? _coord->face_area2().narrow(1, 0, t.size(2))
                         : _coord->face_area3().narrow(0, 0, t.size(1));
  area = area.to(t.device(), wide);
  torch::Tensor rate;
  for (auto p : _pwall[d]) {
    // the full-mask path restricted to one plane: the same values, written
    // and booked on the wall faces only (elsewhere after - before is 0)
    for (auto const& name : _kept[d]) {
      auto part = _parts.at(name + "@" + std::to_string(dim)).select(ax, p);
      part.masked_fill_(rows, 0.);
    }
    auto tp = t.select(ax, p);
    auto after = tp.masked_fill(rows, 0.);
    auto r =
        (area.select(ax - 1, p) * (after.to(wide) - tp.to(wide))).sum({1, 2});
    rate = rate.defined() ? rate + r : r;
    tp.copy_(after);
  }
  _monitor.book("boundary.prescribed", rate);
}

void FluxRegister::keep(int dim, std::string const& part,
                        torch::Tensor const& value) {
  auto p = part_slot(dim, part);
  auto v = value.dim() == p.dim() ? value.squeeze(0) : value;
  TORCH_CHECK(v.sizes() == p.sizes().slice(1), "[FluxRegister] keep '", part,
              "': value must be shaped like total(", dim,
              ") without the row axis, ", p.sizes().slice(1), "; got ",
              value.sizes());
  p[IDN].copy_(v);
}

void FluxRegister::cell_amount(std::string const& name,
                               torch::Tensor const& vol_du, double stage_weight,
                               AmountKind kind) {
  auto wide = vol_du.is_mps() ? torch::kFloat32 : torch::kFloat64;
  auto a = vol_du.to(wide);
  if (a.dim() > 1) {
    std::vector<int64_t> dims(a.dim() - 1);
    std::iota(dims.begin(), dims.end(), 1);
    a = a.sum(dims);
  }
  _monitor.post(name, kind, a, stage_weight, _in_stage);
}

void FluxRegister::apply_prescribed(int dim) {
  auto t = slot(dim, "boundary.prescribed");
  if (_pwall_set[dim - 1]) return apply_wall_planes_(dim);
  auto& mask = _pmask[dim - 1];
  if (!mask.defined()) return;
  auto& value = _pvalue[dim - 1];
  if (mask.device() != t.device()) mask = mask.to(t.device());
  if (value.device() != t.device() || value.scalar_type() != t.scalar_type())
    value = value.to(t.device(), t.scalar_type());
  // R.rst 4.1 item 4: kept parts are prescribed too, with the total's mask
  // and value; only the total books
  for (auto const& name : _kept[dim - 1]) {
    auto part = _parts.at(name + "@" + std::to_string(dim));
    part.copy_(torch::where(mask, value, part));
  }
  overwrite_(dim, "boundary.prescribed", torch::where(mask, value, t));
}

void FluxRegister::begin_stage(int stage, harp::IntegratorWeight const& w) {
  _stage = stage;
  _w1 = w.wght1();
  _w2 = w.wght2();
  _in_stage = true;
}

void FluxRegister::end_stage(double dt) {
  for (int d = 0; d < 3; ++d) {
    if (!_step[d].defined()) continue;
    // in FP64 (FP32 on MPS), whatever the flux dtype: S <- w1 S + w2 dt T
    if (!_direct[d].defined()) {
      _step[d].mul_(_w1).add_(_total[d].to(_step[d].scalar_type()), _w2 * dt);
      continue;
    }
    // a split post this step: the recursion alone, then S = rec + direct
    _rec[d].mul_(_w1).add_(_total[d].to(_rec[d].scalar_type()), _w2 * dt);
    torch::add_out(_step[d], _rec[d], _direct[d]);
  }
  _monitor.end_stage(_w1, _w2 * dt);
  _in_stage = false;
}

void FluxRegister::rollback() { begin_step(); }

void FluxRegister::post_split(int dim, std::string const& name,
                              torch::Tensor const& flux, double weight,
                              double dt) {
  TORCH_CHECK(dim >= 1 && dim <= 3 && _step[dim - 1].defined(),
              "[FluxRegister] post_split '", name, "' needs the step field ",
              dim, " (request it, then begin_step)");
  TORCH_CHECK(!_in_stage, "[FluxRegister] post_split '", name,
              "' inside a stage; it belongs before stage 0 or after the "
              "last end_stage");
  int d = dim - 1;
  TORCH_CHECK(flux.sizes() == _total[d].sizes(), "[FluxRegister] post_split '",
              name, "' flux ", flux.sizes(), " is not shaped like total(", dim,
              ") ", _total[d].sizes());
  if (!_direct[d].defined()) {  // S_d so far is the recursion alone
    _rec[d] = _step[d].clone();
    _direct[d] = torch::zeros_like(_step[d]);
  }
  _direct[d].add_(flux.to(_direct[d].scalar_type()), weight * dt);
  torch::add_out(_step[d], _rec[d], _direct[d]);
}

}  // namespace snap
