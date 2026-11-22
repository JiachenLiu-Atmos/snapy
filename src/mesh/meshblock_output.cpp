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
#include <snap/diagnostics/budget.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/output/restart.hpp>
#include <snap/output/restart_metadata.hpp>
#include <snap/output/writer_core.hpp>
#include <snap/utils/log.hpp>

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

void MeshBlockImpl::print_cycle_info(Variables const &vars, double time,
                                     double dt) const {
  print_cycle_line({{this, &vars}}, time, dt,
                   std::numeric_limits<double>::max_digits10 - 4, " ie=");
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
