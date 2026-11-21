// The budget output and its time series (V3 lane O step O8, budget part).

// C/C++
#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

// base
#include <configure.h>

// snap
#include <snap/hydro/flux_register.hpp>
#include <snap/mesh/meshblock.hpp>

#include "budget_series.hpp"
#include "writer_core.hpp"

#ifdef NETCDFOUTPUT
#include <netcdf.h>
#endif

namespace snap {

namespace {

#ifdef NETCDFOUTPUT
void check(int status, std::string const& path, char const* what) {
  if (status != NC_NOERR)
    throw std::runtime_error("[BudgetSeries] " + path + ": " + what + ": " +
                             nc_strerror(status));
}

constexpr size_t kNameLength = 48;

void put_names(int nc, int var, std::vector<std::string> const& names,
               std::string const& path) {
  for (size_t n = 0; n < names.size(); ++n) {
    std::string s = names[n].substr(0, kNameLength);
    s.resize(kNameLength, '\0');
    size_t start[2] = {n, 0}, count[2] = {1, kNameLength};
    check(nc_put_vara_text(nc, var, start, count, s.data()), path, "names");
  }
}

//! a new file with the record's names (budget_series.cdl, its O8 part)
void create(std::string const& path, BudgetRecord const& r) {
  int nc;
  check(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &nc), path, "create");
  int time, row, nchar, counter = -1, minimum = -1, var;
  check(nc_def_dim(nc, "time", NC_UNLIMITED, &time), path, "time");
  check(nc_def_dim(nc, "row", r.rows.size(), &row), path, "row");
  check(nc_def_dim(nc, "nchar", kNameLength, &nchar), path, "nchar");
  if (!r.counters.empty())
    check(nc_def_dim(nc, "counter", r.counters.size(), &counter), path,
          "counter");
  if (!r.minima.empty())
    check(nc_def_dim(nc, "minimum", r.minima.size(), &minimum), path,
          "minimum");
  for (auto name : {"time", "dt", "interval"})
    check(nc_def_var(nc, name, NC_DOUBLE, 1, &time, &var), path, name);
  check(nc_def_var(nc, "cycle", NC_INT64, 1, &time, &var), path, "cycle");
  int names[2] = {row, nchar};
  check(nc_def_var(nc, "row_name", NC_CHAR, 2, names, &var), path, "row_name");
  int rowdims[2] = {time, row};
  for (auto name : {"total", "change", "boundary"})
    check(nc_def_var(nc, name, NC_DOUBLE, 2, rowdims, &var), path, name);
  if (counter >= 0) {
    int cn[2] = {counter, nchar}, cd[2] = {time, counter};
    check(nc_def_var(nc, "counter_name", NC_CHAR, 2, cn, &var), path, "cname");
    check(nc_def_var(nc, "counter", NC_INT64, 2, cd, &var), path, "counter");
  }
  if (minimum >= 0) {
    int mn[2] = {minimum, nchar}, md[2] = {time, minimum};
    check(nc_def_var(nc, "minimum_name", NC_CHAR, 2, mn, &var), path, "mname");
    check(nc_def_var(nc, "minimum", NC_DOUBLE, 2, md, &var), path, "minimum");
  }
  int version = 1;
  check(nc_put_att_int(nc, NC_GLOBAL, "snapy_budget_version", NC_INT, 1,
                       &version),
        path, "version");
  check(nc_enddef(nc), path, "enddef");
  check(nc_inq_varid(nc, "row_name", &var), path, "row_name");
  put_names(nc, var, r.rows, path);
  if (counter >= 0) {
    check(nc_inq_varid(nc, "counter_name", &var), path, "counter_name");
    put_names(nc, var, r.counters, path);
  }
  if (minimum >= 0) {
    check(nc_inq_varid(nc, "minimum_name", &var), path, "minimum_name");
    put_names(nc, var, r.minima, path);
  }
  check(nc_close(nc), path, "close");
}
#endif

//! the files this process has started, so a later record appends
std::mutex series_mutex;
std::set<std::string> started_files;
//! the local blocks that have recorded this interval, by series file
std::map<std::string, std::vector<BudgetOutput*>> waiting;

int local_blocks(LayoutOptionsImpl const& lo) {
  int bpp = std::max(1, lo.blocks_per_process());
  return std::max(1, std::min(bpp, lo.world_size() - lo.process_rank() * bpp));
}

}  // namespace

void BudgetSeries::append(std::string const& path, BudgetRecord const& r,
                          bool resume) {
#ifdef NETCDFOUTPUT
  {
    std::lock_guard<std::mutex> lock(series_mutex);
    if (!started_files.count(path)) {
      if (!(resume && std::filesystem::exists(path))) create(path, r);
      started_files.insert(path);
    }
  }
  int nc, var, dim;
  check(nc_open(path.c_str(), NC_WRITE, &nc), path, "open");
  size_t n;
  check(nc_inq_dimid(nc, "time", &dim), path, "time");
  check(nc_inq_dimlen(nc, dim, &n), path, "time length");
  size_t one = 1;
  for (auto [name, value] : {std::pair<char const*, double>{"time", r.time},
                             {"dt", r.dt},
                             {"interval", r.interval}}) {
    check(nc_inq_varid(nc, name, &var), path, name);
    check(nc_put_vara_double(nc, var, &n, &one, &value), path, name);
  }
  long long cycle = r.cycle;
  check(nc_inq_varid(nc, "cycle", &var), path, "cycle");
  check(nc_put_vara_longlong(nc, var, &n, &one, &cycle), path, "cycle");
  auto put_row = [&](char const* name, torch::Tensor const& t, bool count) {
    if (t.numel() == 0) return;
    check(nc_inq_varid(nc, name, &var), path, name);
    size_t start[2] = {n, 0}, size[2] = {1, static_cast<size_t>(t.numel())};
    if (count) {
      auto c = t.to(torch::kInt64).contiguous();
      std::vector<long long> v(c.data_ptr<int64_t>(),
                               c.data_ptr<int64_t>() + c.numel());
      check(nc_put_vara_longlong(nc, var, start, size, v.data()), path, name);
    } else {
      auto c = t.to(torch::kFloat64).contiguous();
      check(nc_put_vara_double(nc, var, start, size, c.data_ptr<double>()),
            path, name);
    }
  };
  put_row("total", r.total, false);
  put_row("change", r.change, false);
  put_row("boundary", r.boundary, false);
  put_row("counter", r.counter, true);
  put_row("minimum", r.minimum, false);
  check(nc_close(nc), path, "close");
#else
  throw std::runtime_error(
      "Output type 'budget' writes NetCDF and requires NETCDF=ON.");
#endif
}

BudgetOutput::BudgetOutput(OutputOptions const& options_)
    : OutputType(options_) {}

std::string BudgetOutput::path(MeshBlockImpl const* pmb) {
  return pmb->options->output_dir() + "/" + pmb->options->basename() +
         ".budget.nc";
}

void BudgetOutput::bind(MeshBlockImpl* pmb) {
  // the step fields S_d of the active directions carry the boundary transfer
  auto& freg = pmb->phydro->flux_register();
  for (int dim = 1; dim <= 3; ++dim)
    if (freg.total(dim).defined()) freg.request_step_field(dim);
  _rows = BudgetConsumer::row_names(*pmb);
}

void BudgetOutput::observe(MeshBlockImpl const* pmb, Variables const& vars,
                           double dt) {
  auto const& freg = pmb->phydro->flux_register();
  _consumer.observe(*pmb, vars, freg, freg.monitor(), dt);
}

void BudgetOutput::write_output_file(MeshBlockImpl* pmb, Variables const& vars,
                                     double time, bool final_write) {
  _consumer.flush(*pmb, vars, time);

  // the last local block to record sums them all
  auto file = path(pmb);
  std::vector<BudgetOutput*> group;
  {
    std::lock_guard<std::mutex> lock(series_mutex);
    auto& g = waiting[file];
    g.push_back(this);
    if (static_cast<int>(g.size()) < local_blocks(*pmb->options->layout()))
      return;
    group.swap(g);
    waiting.erase(file);
  }
  std::vector<BudgetRecord const*> parts;
  bool resume = false;
  for (auto out : group) {
    parts.push_back(&out->_consumer.local());
    resume = resume || out->_resumed;
  }
  auto record = reduce_budget(parts, *pmb->get_layout());
  for (auto out : group) out->_record = record;

  auto const& lo = *pmb->options->layout();
  if (lo.process_rank() == lo.process_root_rank()) {
    make_output_dir(pmb);
    BudgetSeries::append(file, record, resume);
  }
}

void BudgetOutput::save_carry(MeshBlockImpl const* pmb, Variables const& vars,
                              bool fires_now, Variables& out) const {
  write_budget_carry(_consumer.carry(*pmb, vars, fires_now), out);
}

void BudgetOutput::load_carry(Variables const& data) {
  auto carry = read_budget_carry(data, _rows);
  if (!carry.defined()) return;  // a new interval at the first observe
  _consumer.resume(carry);
  _resumed = true;
}

}  // namespace snap
