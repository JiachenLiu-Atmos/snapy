// MeshBlock, part BlockRedo (owner X; flags from EF, K, R): retry flags, the
// decision, restore. (V3 X5: src/mesh/meshblock.cpp in six parts)

// C/C++
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/output/restart.hpp>
#include <snap/output/restart_metadata.hpp>
#include <snap/output/writer_core.hpp>
#include <snap/utils/log.hpp>

#include "cycle_diagnostics.hpp"
#include "meshblock.hpp"
#include "meshblock_internal.hpp"
#include "stage_executor.hpp"

namespace snap {

int64_t MeshBlockImpl::saturation_failures() {
  auto pthermo = thermo();
  return pthermo ? pthermo->take_saturation_adjustment_failures() : 0;
}

int MeshBlockImpl::apply_redo(Variables &vars, int causes) {
  // V3 I-5: the VIC's failed columns (cause bit 5), logged on a redo only
  if ((causes & 32) && phydro->picorr)
    phydro->picorr->report_failure(std::cerr);
  if (causes) {
    SINFO(MeshBlock)
        << "Density/pressure at or within 0.1% of the floor, the VIC dry-gas "
           "clamp emptied a cell, the limiter patched one or found a NaN, or "
           "the saturation adjustment left a cell unadjusted, or the VIC "
           "solve rejected a singular/near-singular or nonfinite column. "
           "Redoing the step with smaller dt (causes:"
        << (causes & 1 ? " floor" : "") << (causes & 2 ? " clamp" : "")
        << (causes & 4 ? " limiter" : "") << (causes & 8 ? " nan" : "")
        << (causes & 16 ? " saturation" : "")
        << (causes & 32 ? " vic-solve" : "") << ")." << std::endl;
    pintg->current_redo += 1;
    if (pintg->current_redo > pintg->options->max_redo() && !(causes & 32)) {
      SINFO(MeshBlock)
          << "Maximum number of redo attempts exceeded. Terminating."
          << std::endl;
      return -1;  // terminate
    }

    // reset variables
    phydro->flux_register().rollback();  // the census keeps accepted steps
    _state.adopt(vars);
    _state.restore();
    for (auto &s : phydro->sources.splits()) s->restore_split(vars["hydro_u"]);
    phydro->peos->forward(vars["hydro_u"], vars["hydro_w"]);
    if (vars.count("scalar_s")) {
      set_scalar_primitive(vars, vars["scalar_s"], vars["hydro_u"], *phydro);
    }

    // The caller advances the cycle only after an accepted step.
    // decided: floor_hit's and the restore's cons2prim must not mark the redo
    phydro->peos->reset_limiter_marks(vars["hydro_u"]);
    if (pintg->current_redo > pintg->options->max_redo()) {
      SINFO(MeshBlock)
          << "Maximum number of redo attempts exceeded. Terminating after "
             "restoring the rejected VIC step input."
          << std::endl;
      return -1;
    }
    return 1;  // redo
  }

  // good to go
  pintg->current_redo = 0;
  return 0;
}

torch::Tensor MeshBlockImpl::local_redo_flag_tensor(Variables const &vars) {
  // floor_hit's test, kept on the device
  auto hydro_u = vars.at("hydro_u");
  auto interior = part({0, 0, 0}, PartOptions().exterior(false));
  TORCH_CHECK(vars.count("hydro_w"),
              "MeshBlock::check_redo needs hydro_w to restore the primitives");
  // Retain the checked primitives for output and the next timestep.
  // A rejected step restores both fields in apply_redo.
  auto const &hydro_w = vars.at("hydro_w");
  {
    DeferVaporCheck defer(phydro->peos.get());
    phydro->peos->forward(hydro_u.clone(), hydro_w);
  }
  auto w = hydro_w.index(interior);
  auto const &eos = phydro->peos->options;
  // negated so a NaN, which fails every comparison, counts as a hit
  auto floor = (w[IDN].min() > 1.001 * eos->density_floor()).logical_not();
  if (w.size(0) > IPR) {  // shallow water carries no pressure row
    floor = floor.logical_or(
        (w[IPR].min() > 1.001 * eos->pressure_floor()).logical_not());
  }
  auto opts = hydro_u.options();
  auto zero = torch::zeros({}, opts);

  auto clamp = phydro->picorr
                   ? (phydro->picorr->dry_clamp_step()[0] > 0.).to(opts)
                   : zero;

  // floor's cons2prim may mark a limiter repair: stacked after it, in order
  auto const &m = phydro->peos->limiter_marks();
  auto marks = m.defined() ? m.to(opts) : torch::zeros({2}, opts);

  // drains kintera's count exactly once, as saturation_failures() does
  auto sat = zero;
  auto pthermo = thermo();
  if (pthermo && pthermo->nfail_.defined()) {
    sat = (pthermo->nfail_ > 0).to(opts).reshape({});
    pthermo->nfail_ = torch::Tensor();
  }

  // the VIC failure latch, read in the one copy below
  auto solve =
      phydro->picorr ? phydro->picorr->solve_failed_tensor(opts) : zero;
  // 7th: the step's deferred vapor-repair failures, a limiter cause
  auto const &nfv = phydro->peos->vapor_failures();
  auto vapor = nfv.defined() ? (nfv > 0).to(opts) : zero;
  auto moist = dynamic_cast<IdealMoistImpl *>(phydro->peos.get());
  auto heat = moist && moist->heat_capacity_failure().defined()
                  ? moist->heat_capacity_failure().to(opts)
                  : zero;
  auto out = torch::stack(
      {floor.to(opts), clamp, marks[0], marks[1], sat, solve, vapor, heat});
  if (nfv.defined()) nfv.zero_();  // read once: after the stack, in order
  return out;
}

std::array<bool, 7> MeshBlockImpl::read_redo_flags(torch::Tensor const &flags) {
  auto c = flags.cpu().to(torch::kFloat64);  // the one device-to-host read
  auto a = c.accessor<double, 1>();
  std::array<bool, 7> out{};
  for (size_t i = 0; i < 6; ++i) out[i] = a[i] > 0.;
  // a vapor column the limiter could not repair: redo at a smaller dt (#317)
  if (c.size(0) > 6 && a[6] > 0.) {
    std::cerr << "[MeshBlock] rank=" << get_rank()
              << " vapor repair failed; check_redo will redo the step."
              << std::endl;
    out[2] = true;
  }
  if (c.size(0) > 7 && a[7] > 0.) out[2] = true;
  if (c.size(0) > 8 && a[8] > 0.) out[6] = true;  // diffusion.radial (V3)
  return out;
}

std::array<bool, 7> MeshBlockImpl::local_redo_flags(Variables const &vars) {
  return read_redo_flags(local_redo_flag_tensor(vars));
}

int MeshBlockImpl::decide_retry(std::vector<MeshBlockImpl *> const &blocks,
                                std::vector<Variables const *> const &vars) {
  // A hit in any local block rolls back every block on every process: one
  // device-to-host read for all local blocks on a common device
  std::vector<torch::Tensor> local;
  for (size_t i = 0; i < blocks.size(); ++i)
    local.push_back(blocks[i]->local_redo_flag_tensor(*vars[i]));
  bool same_device = true;
  for (auto const &t : local)
    same_device = same_device && t.device() == local.front().device();
  std::array<bool, 7> flags{};
  if (same_device) {
    flags = read_redo_flags(torch::stack(local).amax(0));
  } else {
    for (auto const &t : local) {
      auto f = read_redo_flags(t);
      for (size_t j = 0; j < flags.size(); ++j) flags[j] = flags[j] || f[j];
    }
  }
  auto flag = torch::zeros({7}, torch::dtype(torch::kFloat64));
  auto f = flag.accessor<double, 1>();
  for (size_t i = 0; i < flags.size(); ++i) f[i] = flags[i] ? 1. : 0.;
  std::vector<at::Tensor> reduced = {flag};
  auto layout = blocks.front()->get_layout();
  if (layout->has_process_group())
    layout->comm->allreduce(reduced, c10d::ReduceOp::MAX);
  auto global = reduced[0].accessor<double, 1>();
  int causes = 0;
  for (size_t i = 0; i < flags.size(); ++i) causes |= (global[i] > 0.) << i;
  return causes;
}

int MeshBlockImpl::check_redo(Variables &vars) {
  return apply_redo(vars, decide_retry({this}, {&vars}));
}

}  // namespace snap
