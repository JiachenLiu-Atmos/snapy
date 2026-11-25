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

void RegisterMonitor::zero_attempt_() {
  if (counters.numel()) counters.zero_();
  if (minima.numel()) minima.fill_(std::numeric_limits<double>::infinity());
  if (retry_flags.numel()) retry_flags.zero_();
  if (_step_amount.defined()) {
    _step_amount.zero_();
    _rate.zero_();
    _post.zero_();
  }
  if (_pend_buf.defined()) _pend_buf.zero_();  // a rejected attempt's masks
  _pend_cursor = 0;
}

void RegisterMonitor::begin_step() {
  zero_attempt_();
  _stage = 0;
  _ended = 0;
}

void RegisterMonitor::rollback() { zero_attempt_(); }

int RegisterMonitor::counter_index(std::string const& name) const {
  return index_of(_counter, name, "counter");
}

int RegisterMonitor::retry_index(std::string const& name) const {
  return index_of(_retry, name, "retry cause");
}

void RegisterMonitor::count(std::string const& name, torch::Tensor const& n) {
  int i = counter_index(name);
  if (n.dim() > 0) {  // a mask or a vector: its slot of the stage's buffer
    auto flat = n.reshape({-1});
    pend_slot_(_pend_cursor, i, flat.numel(), flat);
    _pend_buf.narrow(0, 1 + _pend_off[_pend_cursor], flat.numel()).copy_(flat);
    ++_pend_cursor;
    return;
  }
  // a reduced count adds straight in, one launch: add_ reads an integer
  // count into the counters in its own kernel (int64, or float64 after a
  // Module::to(dtype)); only a float count into integer counters is cast
  auto v = n;
  if (at::isFloatingType(v.scalar_type()) &&
      !at::isFloatingType(counters.scalar_type()))
    v = v.to(counters.scalar_type());
  counters[i].add_(v);
}

void RegisterMonitor::pend_slot_(size_t k, int counter, int64_t numel,
                                 torch::Tensor const& like) {
  auto device = like.device();
  bool same = 2 * k + 1 < _pend_pattern.size() &&
              _pend_pattern[2 * k] == counter &&
              _pend_pattern[2 * k + 1] == numel && _pend_buf.defined() &&
              _pend_buf.device() == device;
  if (same) return;
  // a new or changed slot: the pattern from k on is this stage's, the buffer
  // and the segment ends are laid out again (the slots before k keep their
  // contents when the buffer stays on its device)
  _pend_pattern.resize(2 * k);
  _pend_pattern.push_back(counter);
  _pend_pattern.push_back(numel);
  _pend_off.clear();
  int64_t N = 0;
  for (size_t j = 0; 2 * j < _pend_pattern.size(); ++j) {
    _pend_off.push_back(N);
    N += _pend_pattern[2 * j + 1];
  }
  auto buf = torch::zeros({1 + N}, like.options().dtype(torch::kInt64));
  if (_pend_buf.defined() && _pend_buf.device() == device && k > 0) {
    int64_t keep = 1 + _pend_off[k];
    buf.narrow(0, 0, keep).copy_(_pend_buf.narrow(0, 0, keep));
  }
  _pend_buf = buf;
  int64_t nc = counters.size(0);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> seg(nc);
  for (size_t j = 0; 2 * j < _pend_pattern.size(); ++j) {
    // cs[0] is the zero element: a slot at offset o spans cs[o] .. cs[o + n]
    int64_t o = _pend_off[j], n = _pend_pattern[2 * j + 1];
    seg[_pend_pattern[2 * j]].emplace_back(o, o + n);
  }
  size_t m = 1;
  for (auto const& v : seg) m = std::max(m, v.size());
  _pend_dup = static_cast<int64_t>(m);
  std::vector<int64_t> ends(2 * nc * m, 0);  // [2, nc, m]: ends, starts
  for (int64_t c = 0; c < nc; ++c)
    for (size_t j = 0; j < seg[c].size(); ++j) {
      ends[c * m + j] = seg[c][j].second;
      ends[nc * m + c * m + j] = seg[c][j].first;
    }
  _pend_ends = torch::tensor(ends, torch::kInt64).to(device);
}

void RegisterMonitor::flush_counts() {
  if (!_pend_buf.defined()) return;
  // V3 S4: the segment sums without atomics. The slots lie behind one zero
  // element and are summed as integers by one cumulative sum over the
  // buffer, so a segment's count is cs[end] - cs[start] (exact: the
  // additions are integer); the ends of each counter's segments are gathered
  // densely (zero-padded to the largest number of segments per counter: the
  // pad points at the zero element) and summed along the segment axis; then
  // one add into the counters and one zero of the buffer, so a slot no
  // poster wrote this stage counts nothing. No index_add_: that scatter-adds
  // every mask element with FP64 atomics on CUDA.
  auto cs = _pend_buf.cumsum(0, torch::kInt64);
  auto g =
      cs.index_select(0, _pend_ends).view({2, counters.size(0), _pend_dup});
  counters.add_((g[0] - g[1]).sum(1));
  _pend_buf.zero_();
  _pend_cursor = 0;
}

void RegisterMonitor::record_min(std::string const& name,
                                 torch::Tensor const& v) {
  auto slot = minima[index_of(_minimum, name, "minimum")];
  slot.copy_(torch::minimum(slot, v.min().to(minima.scalar_type())));
}

void RegisterMonitor::flag_retry(std::string const& cause,
                                 torch::Tensor const& any) {
  auto slot = retry_flags[retry_index(cause)];
  // one reduction when the caller hands a mask, none for a flag; the OR is
  // in place (0/1 in the slot's dtype)
  slot.logical_or_(any.dim() == 0 ? any : any.any());
}

void RegisterMonitor::reset_amounts(int nrows) {
  _amount = {"reconcile.x1", "reconcile.panel", "boundary.prescribed"};
  _kind.assign(_amount.size(), AmountKind::flux_correction);
  int64_t na = static_cast<int64_t>(_amount.size());
  int64_t ns = std::max<int64_t>(1, _rate.defined() ? _rate.size(0) : 1);
  _step_amount = torch::zeros({na, nrows}, torch::kFloat64);
  _rate = torch::zeros({ns, na, nrows}, torch::kFloat64);
  _post = torch::zeros({ns, na, nrows}, torch::kFloat64);
  _w1.assign(ns, 0.);
  _w2dt.assign(ns, StepDt(0.));
  _stage = 0;
  _ended = 0;
}

void RegisterMonitor::grow_slots_(int nstage) {
  if (!_rate.defined() || nstage <= _rate.size(0)) return;
  auto grow = [&](torch::Tensor const& t) {
    auto z =
        torch::zeros({nstage - t.size(0), t.size(1), t.size(2)}, t.options());
    return torch::cat({t, z}, 0);
  };
  _rate = grow(_rate);
  _post = grow(_post);
  _w1.resize(nstage, 0.);
  _w2dt.resize(nstage, StepDt(0.));
}

void RegisterMonitor::set_stage_count(int nstage) {
  TORCH_CHECK(nstage >= 1, "[FluxRegister] set_stage_count: ", nstage);
  if (!_rate.defined()) {
    _rate = torch::zeros({nstage, 0, 0}, torch::kFloat64);  // sized at reset
    return;
  }
  if (nstage == _rate.size(0)) return;
  if (nstage > _rate.size(0)) return grow_slots_(nstage);
  _rate = _rate.narrow(0, 0, nstage).contiguous();
  _post = _post.narrow(0, 0, nstage).contiguous();
  _w1.resize(nstage);
  _w2dt.resize(nstage);
}

void RegisterMonitor::begin_stage(int stage) {
  TORCH_CHECK(stage >= 0, "[FluxRegister] begin_stage: ", stage);
  if (_rate.defined() && stage >= _rate.size(0)) grow_slots_(stage + 1);
  _stage = stage;
  _pend_cursor = 0;  // the stage's posts fill the count slots from the first
}

void RegisterMonitor::place_amounts(torch::Device device) {
  auto wide = PrecisionPolicy::wide(device);
  auto placed = [&](torch::Tensor const& t) {
    return t.device() == device && t.scalar_type() == wide;
  };
  if (placed(_step_amount) && placed(_rate) && placed(_post)) return;
  // the cast happens on the host, then the move: MPS cannot hold float64,
  // so a float64 tensor must never be moved there and cast afterwards
  if (!placed(_step_amount)) _step_amount = _step_amount.to(wide).to(device);
  if (!placed(_rate)) _rate = _rate.to(wide).to(device);
  if (!placed(_post)) _post = _post.to(wide).to(device);
}

std::vector<torch::Tensor> RegisterMonitor::amount_storage() const {
  if (!_step_amount.defined()) return {};
  return {_rate, _post, _step_amount};
}

void RegisterMonitor::place_amount_storage(torch::Device device) {
  if (_step_amount.defined()) place_amounts(device);
}

int RegisterMonitor::amount_index(std::string const& name) const {
  return index_of(_amount, name, "amount");
}

AmountKind RegisterMonitor::kind(std::string const& name) const {
  return _kind[amount_index(name)];
}

torch::Tensor RegisterMonitor::amounts(std::string const& name) const {
  return fold_()[amount_index(name)];
}

torch::Tensor RegisterMonitor::fold_() const {
  // A = sum_s c_s (w2_s dt a_s + p_s), c_s = prod_{t>s} w1_t over the stages
  // ended this step; the rates of a stage still open wait for its w2 dt, its
  // posts count at once (a post outside a stage is read straight away); one
  // device fold per read, nothing per stage
  int S = std::min<int>(_ended, _rate.size(0));  // stages ended
  int P = std::min<int>(std::max(S, _stage + 1), _post.size(0));  // with posts
  if (P == 0) return _step_amount;
  auto opts = _rate.options();
  std::vector<double> c(P, 1.);
  for (int s = S - 2; s >= 0; --s) c[s] = c[s + 1] * _w1[s + 1];
  auto cpost = torch::tensor(c, torch::TensorOptions().dtype(opts.dtype()))
                   .to(opts.device())
                   .view({P, 1, 1});
  auto a = (cpost * _post.narrow(0, 0, P)).sum(0);
  if (S > 0) {
    std::vector<torch::Tensor> w;
    for (int s = 0; s < S; ++s)
      w.push_back(_w2dt[s].on_device() ? _w2dt[s].tensor().to(opts)
                                       : torch::tensor(_w2dt[s].value(), opts));
    auto crate =
        (torch::stack(w) * cpost.view({P}).narrow(0, 0, S)).view({S, 1, 1});
    a.add_((crate * _rate.narrow(0, 0, S)).sum(0));
  }
  _step_amount.copy_(a);
  return _step_amount;
}

void RegisterMonitor::book(std::string const& name, torch::Tensor const& rate) {
  int i = amount_index(name);
  place_amounts(rate.device());
  _rate[_stage][i].add_(rate.to(_rate.scalar_type()));
}

void RegisterMonitor::book_index(int i, torch::Tensor const& rate) {
  place_amounts(rate.device());
  _rate[_stage][i].add_(rate.to(_rate.scalar_type()));
}

size_t WallBatch::slot_(torch::Tensor const& planes_flat,
                        torch::Tensor const& weights_flat,
                        RegisterMonitor* monitor, int amount_index,
                        int64_t nvar) {
  int64_t numel = planes_flat.numel();
  // the slots follow the stage's push order, so a register and direction
  // applied twice in a stage (the explicit total, then the implicit
  // correction's) books each apply's planes, as the queue did
  size_t k = _cursor++;
  if (k < _monitor.size() && _monitor[k] == monitor &&
      _index[k] == amount_index && _pattern[2 * k] == nvar &&
      _pattern[2 * k + 1] == numel && _weights[k].is_same(weights_flat))
    return k;
  // a new or changed slot (the first push of a stage's pattern, or a
  // register that rebuilt its weights, eager): the slots from k on are this
  // stage's, the buffer and the layout are built again over every slot
  _monitor.resize(k);
  _index.resize(k);
  _weights.resize(k);
  _pattern.resize(2 * k);
  _monitor.push_back(monitor);
  _index.push_back(amount_index);
  _weights.push_back(weights_flat);
  _pattern.push_back(nvar);
  _pattern.push_back(numel);
  layout_(planes_flat.options());
  return _monitor.size() - 1;
}

void WallBatch::layout_(torch::TensorOptions const& opts) {
  // every (register, direction) row laid into a dense [rows, M] matrix,
  // padded with the zero element behind the slots
  _off.clear();
  int64_t N = 0, rows = 0, M = 0;
  for (size_t k = 0; k < _monitor.size(); ++k) {
    int64_t nvar = _pattern[2 * k], n = _pattern[2 * k + 1];
    _off.push_back(N);
    N += n;
    rows += nvar;
    M = std::max(M, n / nvar);
  }
  auto buf = torch::zeros({N + 1}, opts);
  if (_buf.defined() && _buf.device() == opts.device() &&
      _buf.scalar_type() == opts.dtype()) {
    int64_t keep = std::min(N, _buf.numel() - 1);  // the slots already held
    buf.narrow(0, 0, keep).copy_(_buf.narrow(0, 0, keep));
  }
  _buf = buf;
  std::vector<int64_t> gidx(rows * M, N);
  std::vector<double> wpad(rows * M, 0.);
  int64_t off = 0;
  for (size_t k = 0; k < _monitor.size(); ++k) {
    int64_t nvar = _pattern[2 * k], n = _pattern[2 * k + 1], m = n / nvar;
    // to the host first, then to double: a device that holds no float64
    // (MPS) cannot cast before the copy
    auto w = _weights[k].to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto const* wp = w.data_ptr<double>();
    for (int64_t r = 0; r < nvar; ++r)
      for (int64_t c = 0; c < m; ++c) {
        gidx[(off + r) * M + c] = _off[k] + r * m + c;
        wpad[(off + r) * M + c] = wp[r * m + c];
      }
    off += nvar;
  }
  _gidx = torch::tensor(gidx, torch::kInt64).to(opts.device());
  _w = torch::tensor(wpad, torch::TensorOptions().dtype(opts.dtype()))
           .to(opts.device())
           .view({rows, M});
  _rows = rows;
  _M = M;
}

void WallBatch::push(torch::Tensor const& planes_flat,
                     torch::Tensor const& weights_flat,
                     RegisterMonitor* monitor, int amount_index, int64_t nvar) {
  size_t k = slot_(planes_flat, weights_flat, monitor, amount_index, nvar);
  if (_buf.device() != planes_flat.device() ||
      _buf.scalar_type() != planes_flat.scalar_type())
    layout_(planes_flat.options());  // the planes moved (eager)
  _buf.narrow(0, _off[k], planes_flat.numel()).copy_(planes_flat);
}

void WallBatch::clear() {
  if (_buf.defined()) _buf.zero_();
  _cursor = 0;
}

void WallBatch::flush() {
  if (!_buf.defined()) return;
  auto dense = _buf.index_select(0, _gidx.view({-1})).view({_rows, _M});
  auto rate = (dense * _w).sum(1);  // one dense row reduction, deterministic
  int64_t off = 0;
  for (size_t k = 0; k < _monitor.size(); ++k) {
    int64_t nvar = _pattern[2 * k];
    _monitor[k]->book_index(_index[k], rate.narrow(0, off, nvar));
    off += nvar;
  }
  _buf.zero_();
  _cursor = 0;
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
    auto row = [](torch::Tensor const& t, int dim) {
      return torch::cat({t, torch::zeros_like(t.narrow(dim, 0, 1))}, dim);
    };
    _step_amount = row(_step_amount, 0);
    _rate = row(_rate, 1);
    _post = row(_post, 1);
  }
  TORCH_CHECK(_kind[i] == kind, "[FluxRegister] amount '", name,
              "' was first posted with another AmountKind");
  TORCH_CHECK(amount.dim() == 1 && amount.size(0) == _step_amount.size(1),
              "[FluxRegister] amount '", name, "' must be (",
              _step_amount.size(1), ",); got ", amount.sizes());
  (void)in_stage;  // both land in this stage's slot (scaled by later w1)
  place_amounts(amount.device());
  _post[_stage][i].add_(amount.to(_post.scalar_type()), weight);
}

void RegisterMonitor::end_stage(double w1, StepDt w2dt) {
  flush_counts();
  if (!_step_amount.defined()) return;
  if (_stage >= _rate.size(0)) grow_slots_(_stage + 1);
  _w1[_stage] = w1;
  _w2dt[_stage] = w2dt;
  _ended = std::max(_ended, _stage + 1);
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
  _pwall = {};  // E's wall planes go with the mask: S supplies again
  _pfree = {};
  _prow = {};
  _pwall_set = {false, false, false};
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

void FluxRegister::begin_attempt_() {
  for (int d = 0; d < 3; ++d) {
    if (!_want_step[d]) continue;
    auto const& t = _total[d];
    auto wide = PrecisionPolicy::wide(t.device());
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
}

void FluxRegister::begin_step() {
  begin_attempt_();
  _monitor.begin_step();
}

// V3 S4: the monitor's rollback, not its begin_step, so its host stage
// bookkeeping survives for a graph-replayed redo (RegisterMonitor::rollback)
void FluxRegister::rollback() {
  begin_attempt_();
  _monitor.rollback();
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
  _written[dim - 1] = true;
  TORCH_CHECK(increment.size(0) == static_cast<int64_t>(rows.size()),
              "[FluxRegister] add '", name, "': ", increment.size(0),
              " increment rows for ", rows.size(), " rows");
  // V3 S4: a contiguous row range is one plain add_ on a narrow; other
  // row lists add row by row. (An earlier V3 S4 version used index_add_, whose
  // CUDA kernel adds atomically: for FP64 rows that is a compare-and-swap loop
  // per element, +0.3 ms a step on c1 and +2.9 ms on S3 for three launches.)
  bool contiguous = true;
  for (size_t i = 1; i < rows.size(); ++i)
    if (rows[i] != rows[i - 1] + 1) contiguous = false;
  if (contiguous && !rows.empty()) {
    auto n = static_cast<int64_t>(rows.size());
    t.narrow(0, rows[0], n).add_(increment);
    if (keeps(dim, name))
      part_slot(dim, name).narrow(0, rows[0], n).copy_(increment);
    return;
  }
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
  _written[dim - 1] = true;
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
  _written[dim - 1] = true;  // a reconcile; apply_prescribed clears it after
  auto wide = PrecisionPolicy::wide(t.device());
  // A_f of face index i of T_d is face_area_d() at i (CoordinateImpl's
  // divergence pairs flux index i with area index i)
  auto area = dim == 1   ? _coord->face_area1().narrow(2, 0, t.size(3))
              : dim == 2 ? _coord->face_area2().narrow(1, 0, t.size(2))
                         : _coord->face_area3().narrow(0, 0, t.size(1));
  area = area.to(wide).to(t.device());  // cast on the host, then move
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
  // supply_prescribed sets the same planes every stage: the gathered-plane
  // tensors stay unless the planes or the free rows change (V3 S4)
  if (_pwall[d] != planes || _pfree[d] != free_rows)
    _pidx[d] = _parea[d] = _pneg[d] = _pwt[d] = _pwtflat[d] = torch::Tensor();
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
  auto wide = PrecisionPolicy::wide(t.device());
  auto wopts = t.options().dtype(wide);
  if (!_pidx[d].defined() || _pidx[d].device() != t.device() ||
      _parea[d].scalar_type() != wide) {
    // once per set_prescribed_walls (and per device): the planes, their face
    // areas and the signed row weights; a stage launches none of this
    _pidx[d] = torch::tensor(_pwall[d], t.options().dtype(torch::kInt64));
    auto area = dim == 1   ? _coord->face_area1().narrow(2, 0, t.size(3))
                : dim == 2 ? _coord->face_area2().narrow(1, 0, t.size(2))
                           : _coord->face_area3().narrow(0, 0, t.size(1));
    _parea[d] = area.to(wide).to(t.device()).index_select(ax - 1, _pidx[d]);
    // the scalar operands in the wide dtype on the device: where(mask,
    // Scalar, Scalar) would build them as float64 scalar tensors there,
    // which a device without float64 (MPS) refuses
    _pneg[d] = torch::where(rows, torch::scalar_tensor(-1., wopts),
                            torch::scalar_tensor(0., wopts))
                   .unsqueeze(-1);
    _pwt[d] = _parea[d] * _pneg[d];  // -area on the prescribed rows, else 0
    // V3 S4: the prescribed rows of the wall planes as one mask over the
    // total, and the weights laid out like the gathered planes, flattened
    auto shape = t.sizes().vec();
    shape[ax] = static_cast<int64_t>(_pwall[d].size());
    _pwtflat[d] = _pwt[d].expand(shape).contiguous().reshape({-1});
  }
  // V3 S4: the full-mask path on every wall plane at once: after - before is
  // -T on the prescribed rows and 0 on the free rows, so the booked rate is
  // area * T * (-1 on the prescribed rows), reduced over the plane's faces
  // (as before, one plane's faces first, then the planes); the write back
  // zeroes the prescribed rows of the gathered planes in one scatter
  // V3 S4: gather the wall planes (before), zero their prescribed rows in
  // one masked_fill_ on the total (and on each kept part), and book the rate
  // in the stage's batch (one product and one segment sum for every
  // register and direction at the stage's end); without a batch, book now
  auto planes = t.index_select(ax, _pidx[d]).to(wide);
  // the zeroing touches the wall planes only (one masked_fill_ per plane):
  // a mask over the whole total is one launch on CUDA but a full pass over
  // the flux field on CPU, which cost the CPU cells more than the launches
  // it saved (an earlier V3 S4 timing)
  for (auto p : _pwall[d]) {
    t.select(ax, p).masked_fill_(rows, 0.);
    for (auto const& name : _kept[d])
      _parts.at(name + "@" + std::to_string(dim))
          .select(ax, p)
          .masked_fill_(rows, 0.);
  }
  if (_wall_batch) {
    _wall_batch->push(planes.reshape({-1}), _pwtflat[d], &_monitor,
                      _monitor.amount_index("boundary.prescribed"),
                      planes.size(0));
  } else {
    _monitor.book("boundary.prescribed", (planes * _pwt[d]).sum({1, 2, 3}));
  }
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
  auto wide = PrecisionPolicy::wide(vol_du.device());
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
  if (_pwall_set[dim - 1]) {
    apply_wall_planes_(dim);
  } else {
    auto& mask = _pmask[dim - 1];
    if (mask.defined()) {
      auto& value = _pvalue[dim - 1];
      if (mask.device() != t.device()) mask = mask.to(t.device());
      if (value.device() != t.device() ||
          value.scalar_type() != t.scalar_type())
        value = value.to(t.device(), t.scalar_type());
      // R.rst 4.1 item 4: kept parts are prescribed too, with the total's
      // mask and value; only the total books
      for (auto const& name : _kept[dim - 1]) {
        auto part = _parts.at(name + "@" + std::to_string(dim));
        part.copy_(torch::where(mask, value, part));
      }
      overwrite_(dim, "boundary.prescribed", torch::where(mask, value, t));
    }
  }
  // prescribed now (or nothing to prescribe); the next add, scale or
  // replace that does not keep the prescribed faces marks it again
  _written[dim - 1] = false;
}

void FluxRegister::begin_stage(int stage, harp::IntegratorWeight const& w) {
  _stage = stage;
  _w1 = w.wght1();
  _w2 = w.wght2();
  _in_stage = true;
  _monitor.begin_stage(stage);
}

void FluxRegister::end_stage(StepDt dt) {
  auto w2dt = dt.scaled(_w2);  // V3 D: on the device in a captured stage
  for (int d = 0; d < 3; ++d) {
    if (!_step[d].defined()) continue;
    // in FP64 (FP32 on MPS), whatever the flux dtype: S <- w1 S + w2 dt T
    if (!_direct[d].defined()) {
      axpy_(_step[d].mul_(_w1), _total[d].to(_step[d].scalar_type()), w2dt);
      continue;
    }
    // a split post this step: the recursion alone, then S = rec + direct
    axpy_(_rec[d].mul_(_w1), _total[d].to(_rec[d].scalar_type()), w2dt);
    torch::add_out(_step[d], _rec[d], _direct[d]);
  }
  _monitor.end_stage(_w1, w2dt);
  _in_stage = false;
}

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
