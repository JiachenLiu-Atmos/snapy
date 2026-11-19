// MeshBlock, part BlockOutput (owner O): outputs, cycle diagnostics, the end of
// a run. (V3 X5: src/mesh/meshblock.cpp in six parts)

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
#include <snap/input/read_restart_file.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/restart_metadata.hpp>
#include <snap/utils/log.hpp>

#include "cycle_diagnostics.hpp"
#include "meshblock.hpp"
#include "meshblock_internal.hpp"
#include "stage_executor.hpp"

namespace snap {

void MeshBlockImpl::make_outputs(Variables const &vars, double current_time,
                                 bool final_write) {
  for (auto &output_type : output_types) {
    output_type->AccumulateStats(vars, current_time);
    if (final_write) {
      output_type->write_output_file(this, vars, current_time, final_write);
    } else if (current_time >= output_type->next_time) {
      output_type->write_output_file(this, vars, current_time, final_write);
      output_type->ResetStats(current_time);
      output_type->next_time += output_type->options->dt();
      output_type->file_number += 1;
    }
  }
}

void print_cycle_diagnostics(
    std::vector<std::pair<MeshBlockImpl const *, Variables const *>> const
        &blocks,
    double time, double dt, int precision, char const *energy_label) {
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
  auto add = [](torch::Tensor &total, torch::Tensor value) {
    if (total.defined())
      total += value;
    else
      total = value.clone();
  };
  for (auto const &[block, vars] : blocks) {
    auto hydro = block->phydro;
    auto coord = block->pcoord;
    auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
    auto vol = coord->cell_volume();
    // totals: anomaly storage adds the reference back (fp64 in Step 1)
    auto u = hydro->total_u(vars->at("hydro_u"));
    add(conserved, (u * vol).index(interior).sum({1, 2, 3}));

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
  auto reduce = [&](torch::Tensor &value, c10d::ReduceOp op) {
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

  auto print = [&](char const *name, double value) {
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

void MeshBlockImpl::print_cycle_info(Variables const &vars, double time,
                                     double dt) const {
  print_cycle_diagnostics({{this, &vars}}, time, dt,
                          std::numeric_limits<double>::max_digits10 - 4,
                          " ie=");
}

int MeshBlockImpl::finalize(Variables const &vars, double time) {
  make_outputs(vars, time, /*final_write=*/true);
  return finish_run({this}, time);
}

int MeshBlockImpl::finish_run(std::vector<MeshBlockImpl *> const &blocks,
                              double time) {
  auto root = blocks.front();
  int status = 0;
  auto sig = SignalHandler::GetInstance();
  if (sig->GetSignalFlag(SIGTERM) != 0) {
    SINFO() << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGINT) != 0) {
    SINFO() << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGALRM) != 0) {
    SINFO() << std::endl << "Terminating on wall-time limit" << std::endl;
  } else if (root->pintg->options->nlim() >= 0 &&
             root->cycle >= root->pintg->options->nlim()) {
    SINFO() << std::endl << "Terminating on cycle limit" << std::endl;
  } else if (time >= root->pintg->options->tlim()) {
    SINFO() << std::endl << "Terminating on time limit" << std::endl;
  } else {
    SINFO() << std::endl << "Terminating abnormally" << std::endl;
    status = 1;
  }

  SINFO() << "time=" << time << " cycle=" << root->cycle << std::endl;
  SINFO() << "tlim=" << root->pintg->options->tlim()
          << " nlim=" << root->pintg->options->nlim() << std::endl;

  for (auto block : blocks) {
    block->send_bufs.clear();
    block->send_bufs.shrink_to_fit();
    block->recv_bufs.clear();
    block->recv_bufs.shrink_to_fit();
  }

  auto layout = root->get_layout();
  if (layout->has_process_group()) {
    layout->comm->barrier();
    if (layout->comm->owns_process_group()) {
      layout->comm->shutdown();
    }
  }
  return status;
}

}  // namespace snap
