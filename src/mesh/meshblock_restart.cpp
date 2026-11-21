// MeshBlock, part BlockRestart (owner O): initialization from a restart file.
// (V3 X5: src/mesh/meshblock.cpp in six parts)

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

double MeshBlockImpl::_init_from_restart(Variables &vars, std::string fname) {
  std::filesystem::path restart_path(fname);
  if (!restart_path.is_absolute() && !std::filesystem::exists(restart_path)) {
    restart_path = std::filesystem::path(options->output_dir()) / fname;
  }

  auto data = load_restart(restart_path.string(), options->layout()->rank());

  // check required variables
  TORCH_CHECK(data.count("hydro_u"),
              "Restart file is missing required variable: hydro_u");
  TORCH_CHECK(data.count("last_time"),
              "Restart file is missing required variable: last_time");
  TORCH_CHECK(data.count("last_cycle"),
              "Restart file is missing required variable: last_cycle");
  TORCH_CHECK(data.count("file_number"),
              "Restart file is missing required variable: file_number");
  TORCH_CHECK(data.count("next_time"),
              "Restart file is missing required variable: next_time");

  check_restart_metadata(data, {pcoord->x1f, pcoord->x2f, pcoord->x3f},
                         phydro->options->perturbation_flux());

  // Resume from the saved cycle so nlim-based runs do not take an extra step.
  cycle = data.at("last_cycle").item<int64_t>();

  auto current_time = data.at("last_time").item<double>();

  // schedules restored by key; an edited block keeps its position, as before
  int nsaved = data.at("file_number").size(0);
  std::vector<bool> claimed(nsaved, false);
  std::vector<int> match(output_types.size(), -1);
  // New files append N exact keys after the N legacy keys. Old readers consume
  // only the prefix; an old writer replaces it with an N-entry legacy tensor.
  auto saved_keys =
      data.count("output_key") ? data.at("output_key") : torch::Tensor();
  bool precise_keys = saved_keys.defined() && saved_keys.dim() == 1 &&
                      saved_keys.size(0) == 2 * nsaved;
  int key_offset = precise_keys ? nsaved : 0;
  auto current_key = [precise_keys](std::shared_ptr<OutputType> const &output) {
    return precise_keys ? output->schedule_key_v2() : output->schedule_key();
  };
  if (saved_keys.defined()) {
    // a block still matching its own saved slot keeps it; nothing can steal it
    for (int n = 0; n < nsaved && n < (int)output_types.size(); ++n) {
      if (saved_keys[key_offset + n].item<int64_t>() ==
          current_key(output_types[n])) {
        match[n] = n;
        claimed[n] = true;
      }
    }
    for (int n = 0; n < output_types.size(); ++n) {
      if (match[n] >= 0) continue;
      auto key = current_key(output_types[n]);
      for (int k = 0; k < nsaved; ++k) {
        if (!claimed[k] && saved_keys[key_offset + k].item<int64_t>() == key) {
          match[n] = k;
          claimed[k] = true;
          break;
        }
      }
    }
  }
  for (int n = 0; n < output_types.size(); ++n) {
    if (match[n] < 0 && n < nsaved && !claimed[n]) {
      match[n] = n;
      claimed[n] = true;
    }
  }
  for (int n = 0; n < output_types.size(); ++n) {
    // the file name is out<n>, so the counter follows the POSITION, not the key
    if (n < nsaved) {
      output_types[n]->file_number = data.at("file_number")[n].item<int64_t>();
    }
    int m = match[n];
    if (m >= 0) {
      double saved = data.at("next_time")[m].item<double>();
      double dt = output_types[n]->options->dt();
      // A disabled output stores a next_time far past the resume. Keeping it
      // after dt shrinks means the output never fires. A time still within
      // one dt is a live cadence and stays. dt <= 0 keeps the saved time.
      if (dt > 0.0 && saved > current_time + dt) {
        output_types[n]->next_time = current_time + dt;
      } else {
        output_types[n]->next_time = saved;
      }
    } else {  // new output joins the restored grid
      auto dt = output_types[n]->options->dt();
      output_types[n]->next_time =
          dt > 0.0 ? current_time - std::fmod(current_time, dt) : current_time;
    }
  }

  for (auto &output_type : output_types) output_type->load_carry(data);

  bool rebuild_scalar_r =
      pscalar->nvar() > 0 && data.count("scalar_s") && data.count("hydro_u");

  // move to device
  for (auto &[name, tensor] : data) {
    if (rebuild_scalar_r && name == "scalar_r") {
      continue;
    }
    // held by hydro.ref / used below, or a budget carry: not kept in vars
    if (name.rfind(kBudgetCarryPrefix, 0) == 0 || name == "hydro_ref" ||
        name == "hydro_ref_species" || name == "hydro_u_stored" ||
        name == kHydroRestartMetadata ||
        std::find(kHydroReferenceGrid.begin(), kHydroReferenceGrid.end(),
                  name) != kHydroReferenceGrid.end())
      continue;
    vars[name] = tensor.to(torch::Device(options->device_str()));
  }

  // anomaly storage: the file holds totals. Reuse its reference when it has
  // one; otherwise rebuild it from the restored state at rest (single-block
  // runs only: a per-block rebuild would not match across block seams)
  if (phydro->anomaly()) {
    auto &hydro_u = vars.at("hydro_u");
    bool stored = false;
    if (data.count("hydro_ref")) {
      auto r = data.at("hydro_ref").to(hydro_u.device());
      phydro->pref->set("u0", r);
      phydro->u_ref_like(hydro_u);  // shape check
      if (data.count("hydro_ref_species")) {
        auto species = data.at("hydro_ref_species").to(hydro_u.device());
        TORCH_CHECK(
            species.dim() == 4 && species.size(0) == hydro_u.size(0) - ICY &&
                (species.size(1) == 1 || species.size(1) == hydro_u.size(1)) &&
                (species.size(2) == 1 || species.size(2) == hydro_u.size(2)) &&
                species.size(3) == hydro_u.size(3),
            "restart: hydro_ref_species does not match the block state");
        phydro->pref->set("species0", species);
      } else {
        // Composition at restart time is not the initial background. Never
        // silently substitute it (or keep a previous initialization's cache).
        phydro->pref->set("species0", torch::empty({0}, r.options()));
      }
      // resume the anomaly bit for bit when the file carries it
      if (data.count("hydro_u_stored")) {
        // Totals in the file are FP64; retain the stored anomaly dtype.
        hydro_u = data.at("hydro_u_stored").to(hydro_u.device());
        stored = true;
      }
    } else {
      auto const &lo = _playout->options;
      TORCH_CHECK(
          !_playout->has_process_group() && lo->px() * lo->py() * lo->pz() == 1,
          "restart: anomaly mode needs 'hydro_ref' in a multi-block "
          "restart file (this one was written without it). Resume "
          "with SNAP_ANOMALY=0, or from a file written with anomaly "
          "storage on.");
      Variables tmp = vars;
      tmp["hydro_w"] = phydro->peos->compute("U->W", {hydro_u.clone()});
      build_hydro_reference(tmp);
    }
    if (!stored) phydro->sub_reference_(hydro_u);
    // the solid fill values are kept in the stored form too; the file has
    // them in totals
    if (vars.count("fill_solid_hydro_u") &&
        vars.at("fill_solid_hydro_u").defined() &&
        !vars.at("fill_solid_hydro_u").is_same(hydro_u))
      phydro->sub_reference_(vars.at("fill_solid_hydro_u"));
  }

  if (phydro->options->perturbation_flux()) {
    vars["hydro_u"] = policy.store(Quantity::kHydroState, vars.at("hydro_u"));
    vars["hydro_w"] = vars.at("hydro_w").to(torch::kFloat64);
    if (vars.count("scalar_s"))
      vars["scalar_s"] =
          policy.store(Quantity::kScalarState, vars.at("scalar_s"));
  }
  if (has_radiating_boundary()) {
    TORCH_CHECK(
        data.count("boundary_reference_w") &&
            (pscalar->nvar() == 0 || data.count("boundary_reference_r")),
        "Characteristic outflow restart is missing boundary reference "
        "tensors. Initialize afresh or select extrapolation.");
    for (auto key : {"boundary_reference_w", "boundary_reference_r"}) {
      if (vars.count(key))
        vars[key] = vars.at(key).to(vars.at("hydro_w").options());
    }
    // Validate without changing the serialized numerical state.
    apply_boundaries(
        vars, vars.at("hydro_u").clone(),
        vars.count("scalar_s") ? vars.at("scalar_s").clone() : torch::Tensor());
  }

  // remove timing data
  vars.erase("last_time");
  vars.erase("last_cycle");
  vars.erase("file_number");
  vars.erase("next_time");
  vars.erase("output_key");

  if (rebuild_scalar_r) {
    set_scalar_primitive(vars, vars.at("scalar_s"), vars.at("hydro_u"),
                         *phydro);
  }

  // a restart is a new state: new storage, adopted at the next step
  _state.reallocate();
  _executor->invalidate();
  return current_time;
}

}  // namespace snap
