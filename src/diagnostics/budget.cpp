// The budget consumer and the cycle line (V3 lane O step O8, its budget
// part; O.rst section 3.2 and the O8 row). The fixer, source and residual
// columns follow lane R's amounts (R-7) as O8-cols.

#include "budget.hpp"

// C/C++
#include <iomanip>
#include <limits>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/layout/layout.hpp>
#include <snap/layout/process_group.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

namespace snap {

namespace {

using torch::indexing::Slice;

auto const kF64 = torch::kFloat64;
double const kInf = std::numeric_limits<double>::infinity();

//! names of the register's rows: its own, or the conserved layout's
std::vector<std::string> names_of(RegisterMonitor const& mon, int nvar) {
  if (static_cast<int>(mon.row_names().size()) == nvar) return mon.row_names();
  std::vector<std::string> rows(nvar);
  for (int n = 0; n < nvar; ++n) {
    if (n == IDN)
      rows[n] = "mass";
    else if (n == IVX || n == IVY || n == IVZ)
      rows[n] = "momentum" + std::to_string(n - IVX + 1);
    else if (n == IPR)
      rows[n] = "energy";
    else
      rows[n] = "species/" + std::to_string(n - ICY + 1);
  }
  return rows;
}

//! a block face is the fluid domain's boundary when a boundary function
//! acts on it and does not wrap: the wall, outflow and custom faces (the
//! base nulls the functions of interior, seam and periodic-layout faces)
bool boundary_face(MeshBlockImpl const& block, int dim, bool upper) {
  int side = upper ? 1 : -1;
  int dy = dim == 3 ? side : 0, dx = dim == 2 ? side : 0,
      dz = dim == 1 ? side : 0;
  auto const& op = block.options;
  if (!op->is_physical_boundary(dy, dx, dz)) return false;
  int face = 2 * (dim - 1) + (upper ? 1 : 0);
  auto const& names = op->bcnames();
  return face >= static_cast<int>(names.size()) ||
         names[face].rfind("periodic", 0) != 0;
}

}  // namespace

torch::Tensor BudgetConsumer::totals(MeshBlockImpl const& block,
                                     torch::Tensor const& u) {
  auto interior = block.part({0, 0, 0}, PartOptions().exterior(false));
  auto vol = block.pcoord->cell_volume();
  return (u * vol).index(interior).sum({1, 2, 3});
}

std::vector<std::string> BudgetConsumer::row_names(MeshBlockImpl const& block) {
  return names_of(block.phydro->flux_register().monitor(),
                  block.phydro->peos->nvar());
}

torch::Tensor BudgetConsumer::boundary(MeshBlockImpl const& block,
                                       Variables const& vars,
                                       FluxRegister const& reg) {
  auto c = block.pcoord;
  std::array<int64_t, 3> lo = {c->kl(), c->jl(), c->il()};
  std::array<int64_t, 3> hi = {c->ku(), c->ju(), c->iu()};
  if (!_weights_ready) {
    torch::Tensor solid;
    if (vars.count("solid") && vars.at("solid").defined())
      solid = vars.at("solid").ne(0);
    std::array<torch::Tensor, 3> area = {c->face_area1(), c->face_area2(),
                                         c->face_area3()};
    for (int dim = 1; dim <= 3; ++dim) {
      _weight[dim - 1] = torch::Tensor();
      if (!reg.has_step_field(dim)) continue;
      int ax = 3 - dim;  // axis of a [k, j, i] tensor
      std::array<int64_t, 3> b = lo, n;
      for (int a = 0; a < 3; ++a) n[a] = hi[a] - lo[a] + 1;
      n[ax] += 1;  // the faces of the interior cells, both ends
      auto window = [&](torch::Tensor const& t, int shift) {
        return t.narrow(0, b[0] - (ax == 0) * shift, n[0])
            .narrow(1, b[1] - (ax == 1) * shift, n[1])
            .narrow(2, b[2] - (ax == 2) * shift, n[2]);
      };
      auto sign = torch::zeros(n, area[dim - 1].options().dtype(kF64));
      if (solid.defined()) {
        // a face with one solid neighbour, outward from the fluid cell, which
        // must be this block's: the first face's left and the last face's
        // right cell are ghosts
        auto left = window(solid, 1), right = window(solid, 0);
        sign = (left.logical_not() & right).to(kF64) -
               (left & right.logical_not()).to(kF64);
        sign.select(ax, 0).clamp_(c10::nullopt, 0.);
        sign.select(ax, n[ax] - 1).clamp_(0., c10::nullopt);
      }
      for (bool upper : {false, true}) {
        if (!boundary_face(block, dim, upper)) continue;
        int p = upper ? n[ax] - 1 : 0;
        auto plane = sign.select(ax, p);
        plane.fill_(upper ? 1. : -1.);
        if (solid.defined()) {
          int cell = upper ? hi[ax] : lo[ax];
          auto inner =
              solid.narrow(0, ax == 0 ? cell : b[0], ax == 0 ? 1 : n[0])
                  .narrow(1, ax == 1 ? cell : b[1], ax == 1 ? 1 : n[1])
                  .narrow(2, ax == 2 ? cell : b[2], ax == 2 ? 1 : n[2])
                  .squeeze(ax);
          plane.mul_(inner.logical_not().to(kF64));
        }
      }
      _weight[dim - 1] = sign * window(area[dim - 1], 0).to(kF64);
      for (int a = 0; a < 3; ++a) {
        _window[dim - 1][a] = b[a];
        _window[dim - 1][3 + a] = n[a];
      }
    }
    _weights_ready = true;
  }

  torch::Tensor out;
  for (int dim = 1; dim <= 3; ++dim) {
    auto const& w = _weight[dim - 1];
    if (!w.defined() || !reg.has_step_field(dim)) continue;
    auto const& win = _window[dim - 1];
    auto s = reg.step_field(dim)
                 .narrow(1, win[0], win[3])
                 .narrow(2, win[1], win[4])
                 .narrow(3, win[2], win[5]);
    auto t = (s.to(kF64) * w.to(s.device())).sum({1, 2, 3});
    out = out.defined() ? out + t : t;
  }
  if (!out.defined()) {
    int nvar = block.phydro->peos->nvar();
    out = torch::zeros({nvar},
                       torch::dtype(kF64).device(vars.at("hydro_u").device()));
  }
  return out;
}

void BudgetConsumer::start(MeshBlockImpl const& block, Variables const& vars) {
  auto const& mon = block.phydro->flux_register().monitor();
  auto u = block.phydro->total_u(vars.at("hydro_u"));
  _total_start = totals(block, u).to(kF64);
  _pending = torch::zeros_like(_total_start);
  auto dev = _total_start.device();
  int nc = mon.counter_names().size(), nm = mon.minimum_names().size();
  _counter = torch::zeros({nc}, torch::dtype(torch::kInt64).device(dev));
  _minimum = torch::full({nm}, kInf, torch::dtype(kF64).device(dev));
  _interval = 0.;
  _started = true;
}

void BudgetConsumer::observe(MeshBlockImpl const& block, Variables const& vars,
                             FluxRegister const& reg,
                             RegisterMonitor const& mon, double dt) {
  _dt = dt;
  if (!_started) {  // the interval starts after this step (Xi note 10)
    start(block, vars);
    return;
  }
  _pending += boundary(block, vars, reg).to(_pending.device());
  if (mon.counters.defined() && mon.counters.numel() == _counter.numel())
    _counter += mon.counters.to(_counter.device(), torch::kInt64);
  if (mon.minima.defined() && mon.minima.numel() == _minimum.numel())
    _minimum = torch::minimum(_minimum, mon.minima.to(_minimum.device(), kF64));
  _interval += dt;
}

void BudgetConsumer::flush(MeshBlockImpl const& block, Variables const& vars,
                           double time) {
  if (!_started) start(block, vars);  // a record before any step
  auto const& mon = block.phydro->flux_register().monitor();
  auto u = block.phydro->total_u(vars.at("hydro_u"));
  auto total = totals(block, u).to(kF64);

  _local.rows = names_of(mon, total.size(0));
  _local.counters = mon.counter_names();
  _local.minima = mon.minimum_names();
  // copies: on the CPU, cpu() is the tensor itself, reset below
  _local.total = total.cpu().clone();
  _local.change = (total - _total_start).cpu();
  _local.boundary = _pending.cpu().clone();
  _local.counter = _counter.cpu().clone();
  _local.minimum = _minimum.cpu().clone();
  _local.time = time;
  _local.interval = _interval;
  _local.dt = _dt;
  _local.cycle = block.cycle;

  _total_start = total;
  _pending.zero_();
  _counter.zero_();
  _minimum.fill_(kInf);
  _interval = 0.;
}

BudgetCarry BudgetConsumer::carry(MeshBlockImpl const& block,
                                  Variables const& vars, bool fires_now) const {
  BudgetCarry c;
  if (!_started) return c;  // nothing to resume: a new interval on load
  auto const& mon = block.phydro->flux_register().monitor();
  c.rows = names_of(mon, _total_start.size(0));
  c.interval = _interval;
  c.total_prev = _total_start.cpu().clone();
  c.pending = _pending.cpu().clone();
  c.counter = _counter.cpu().clone();
  c.minimum = _minimum.cpu().clone();
  if (fires_now) {  // the record at this time closes the interval
    auto u = block.phydro->total_u(vars.at("hydro_u"));
    c.total_prev = totals(block, u).to(kF64).cpu();
    c.pending = torch::zeros_like(c.pending);
    c.counter = torch::zeros_like(c.counter);
    c.minimum = torch::full_like(c.minimum, kInf);
    c.interval = 0.;
  }
  return c;
}

void BudgetConsumer::resume(BudgetCarry const& carry, torch::Device device) {
  if (!carry.defined()) return;
  // the file's tensors are on the host; the sums live with the block's state
  _total_start = carry.total_prev.to(device, kF64, false, true);
  _pending = carry.pending.to(device, kF64, false, true);
  _counter = carry.counter.to(device, torch::kInt64, false, true);
  _minimum = carry.minimum.to(device, kF64, false, true);
  _interval = carry.interval;
  _started = true;
}

BudgetRecord reduce_budget(std::vector<BudgetRecord const*> const& parts,
                           LayoutImpl const& layout) {
  TORCH_CHECK(!parts.empty(), "[reduce_budget] no block");
  BudgetRecord r = *parts.front();
  r.total = r.total.clone();
  r.change = r.change.clone();
  r.boundary = r.boundary.clone();
  r.counter = r.counter.clone();
  r.minimum = r.minimum.clone();
  for (size_t b = 1; b < parts.size(); ++b) {
    r.total += parts[b]->total;
    r.change += parts[b]->change;
    r.boundary += parts[b]->boundary;
    r.counter += parts[b]->counter;
    r.minimum = torch::minimum(r.minimum, parts[b]->minimum);
  }
  if (layout.has_process_group()) {
    int nr = r.total.numel(), nc = r.counter.numel();
    std::vector<torch::Tensor> sums = {
        torch::cat({r.total, r.change, r.boundary, r.counter.to(kF64)})};
    layout.comm->allreduce(sums, c10d::ReduceOp::SUM);
    r.total = sums[0].narrow(0, 0, nr).clone();
    r.change = sums[0].narrow(0, nr, nr).clone();
    r.boundary = sums[0].narrow(0, 2 * nr, nr).clone();
    r.counter = sums[0].narrow(0, 3 * nr, nc).to(torch::kInt64);
    if (r.minimum.numel() > 0) {
      std::vector<torch::Tensor> mins = {r.minimum};
      layout.comm->allreduce(mins, c10d::ReduceOp::MIN);
      r.minimum = mins[0];
    }
  }
  return r;
}

void print_cycle_line(
    std::vector<std::pair<MeshBlockImpl const*, Variables const*>> const&
        blocks,
    double time, double dt, int precision, char const* energy_label) {
  auto root = blocks.front().first;
  auto pintg = root->pintg;
  if (pintg->options->ncycle_out() == 0 ||
      root->cycle % pintg->options->ncycle_out() != 0)
    return;

  SINFO() << "cycle=" << root->cycle << " redo=" << pintg->current_redo
          << std::scientific << std::setprecision(precision) << " time=" << time
          << " dt=" << dt;
  if (!blocks.front().second->count("hydro_u")) {
    SINFO() << std::endl;
    return;
  }

  torch::Tensor conserved, ke_sum, pe_sum, meters, theta_min, vic_max;
  auto add = [](torch::Tensor& total, torch::Tensor value) {
    if (total.defined())
      total += value;
    else
      total = value.clone();
  };
  for (auto const& [block, vars] : blocks) {
    auto hydro = block->phydro;
    auto coord = block->pcoord;
    auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
    auto vol = coord->cell_volume();
    // totals: anomaly storage adds the reference back (fp64 in Step 1)
    auto u = hydro->total_u(vars->at("hydro_u"));
    add(conserved, BudgetConsumer::totals(*block, u));

    // Read the conserved state; hydro_w can still describe an earlier stage.
    if (u.size(0) >= IVX + 3) {
      auto rho = u[IDN].unsqueeze(0).clone();
      for (int n = ICY; n < u.size(0); ++n) rho += u[n].unsqueeze(0);
      auto mom = u.narrow(0, IVX, 3).clone();
      coord_vec_raise_(mom, coord->cosine_cell_kj);
      auto ke = 0.5 * (u.narrow(0, IVX, 3) * mom).sum(0, true) / rho;
      add(ke_sum, (ke * vol).index(interior).sum({1, 2, 3}));
      if (hydro->options->grav() && hydro->options->grav()->grav1() != 0.) {
        auto grav1 = hydro->options->grav()->grav1();
        auto pe = rho * (-grav1 * coord->x1v);
        // gravity-work: face books the work of the corrected potential
        // energy P, so log P: E + P is what that work conserves
        if (hydro->radial_exact_work()) {
          int is = coord->il(), ie = coord->iu() + 1;
          // across an x1 seam the ghost rho (exchanged after every stage) is
          // the neighbour's: P of a split column is one block's (#303); a
          // periodic x1 column keeps the one-sided slope (hydro_forward.cpp)
          auto [below, above] = hydro->x1_neighbors();
          pe.slice(-1, is, ie) -= corrected_pe_work_seam(
              rho, coord->x1f, coord->x1v, is, ie, grav1,
              x1_measure(coord->options->type()), below >= 0, above >= 0);
        }
        add(pe_sum, (pe * vol).index(interior).sum({1, 2, 3}));
      }
    }

    add(meters, torch::stack(
                    {hydro->lim_cut()[0].cpu().to(torch::kFloat64),
                     hydro->lim_flux()[0].cpu().to(torch::kFloat64),
                     hydro->positivity_severe()[0].cpu().to(torch::kFloat64)}));
    auto tmin = hydro->positivity_min().cpu().to(torch::kFloat64);
    theta_min =
        theta_min.defined() ? torch::minimum(theta_min, tmin) : tmin.clone();
    if (hydro->picorr) {
      auto vc = hydro->picorr->clamp_residual().cpu().to(torch::kFloat64);
      vic_max = vic_max.defined() ? torch::maximum(vic_max, vc) : vc.clone();
    }
  }

  auto layout = root->get_layout();
  auto reduce = [&](torch::Tensor& value, c10d::ReduceOp op) {
    if (!value.defined() || !layout->has_process_group()) return;
    std::vector<at::Tensor> values = {value};
    layout->comm->reduce(values, op,
                         root->options->layout()->process_root_rank());
    value = values[0];
  };
  reduce(conserved, c10d::ReduceOp::SUM);
  reduce(ke_sum, c10d::ReduceOp::SUM);
  reduce(pe_sum, c10d::ReduceOp::SUM);
  reduce(meters, c10d::ReduceOp::SUM);
  reduce(theta_min, c10d::ReduceOp::MIN);
  reduce(vic_max, c10d::ReduceOp::MAX);

  auto print = [&](char const* name, double value) {
    SINFO() << std::scientific << std::setprecision(precision) << name << value;
  };
  auto mass = conserved[IDN];
  print(" mass0=", mass.item<double>());
  if (conserved.size(0) > ICY) {
    for (int n = ICY; n < conserved.size(0); ++n) mass += conserved[n];
    print(" masst=", mass.item<double>());
  }
  if (ke_sum.defined()) print(" ke=", ke_sum[0].item<double>());
  if (root->phydro->peos->nvar() > IPR)
    print(energy_label, conserved[IPR].item<double>());
  if (pe_sum.defined()) print(" pe=", pe_sum[0].item<double>());

  SINFO() << " run-to-date:";
  double total = meters[1].item<double>();
  if (total > 0.) print(" limcut=", meters[0].item<double>() / total);
  print(" thetamin=", theta_min[0].item<double>());
  SINFO() << " thetasevere="
          << static_cast<long long>(meters[2].item<double>());
  if (vic_max.defined()) print(" vicclamp=", vic_max[0].item<double>());
  SINFO() << std::endl;
}

}  // namespace snap
