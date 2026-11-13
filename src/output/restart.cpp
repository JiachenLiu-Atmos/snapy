// C/C++
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"
#include "output_utils.hpp"
#include "writer_core.hpp"

namespace snap {

RestartOutput::RestartOutput(OutputOptions const& options_)
    : OutputType(options_) {
  // restart files are always combined
  options->combine(true);
}

void RestartOutput::write_output_file(MeshBlockImpl* pmb, Variables const& vars,
                                      double current_time, bool final_write) {
  // make a cpu copy of variables
  Variables out_vars;
  bool skip_scalar_r = vars.count("scalar_s") && vars.count("hydro_u");
  for (auto const& [name, var] : vars) {
    if (skip_scalar_r && name == "scalar_r") {
      continue;
    }
    if (var.defined()) {
      out_vars[name] = var.to(torch::kCPU);
    }
  }
  // anomaly storage (fp32 roadmap Step 1): the file always holds totals, and
  // the fp64 reference rides along so a resume reuses the same bits
  // (hydro_u_stored keeps the anomaly bits: (u' + ref) - ref is not u')
  if (pmb->phydro->has_reference() && vars.count("hydro_u")) {
    auto const& u = vars.at("hydro_u");
    out_vars["hydro_u"] = pmb->phydro->total_u(u).to(torch::kCPU);
    out_vars["hydro_u_stored"] = u.to(torch::kCPU);
    out_vars["hydro_ref"] = pmb->phydro->u_ref().to(torch::kCPU);
    if (pmb->phydro->species_ref().defined()) {
      out_vars["hydro_ref_species"] =
          pmb->phydro->species_ref().to(torch::kCPU);
    }
    if (vars.count("fill_solid_hydro_u") &&
        vars.at("fill_solid_hydro_u").defined()) {
      out_vars["fill_solid_hydro_u"] =
          pmb->phydro->total_u(vars.at("fill_solid_hydro_u")).to(torch::kCPU);
    }
  }

  // store last time and cycle
  out_vars["last_time"] = torch::tensor({current_time}, torch::kFloat64);
  out_vars["last_cycle"] = torch::tensor({(int64_t)pmb->cycle}, torch::kInt64);

  // save file number and next time for each output type
  std::vector<int> output_file_numbers;
  std::vector<double> output_next_times;
  std::vector<int64_t> output_keys;

  // make_outputs advances an output's schedule only after writing it. Save the
  // advanced schedule for this restart and the outputs after it that fire in
  // the same pass, so a resume does not write their t0 files again.
  bool firing = false;
  for (auto out : pmb->output_types) {
    if (out.get() == this) firing = !final_write;
    bool advance = firing && current_time >= out->next_time;
    output_file_numbers.push_back(out->file_number + (advance ? 1 : 0));
    output_next_times.push_back(out->next_time +
                                (advance ? out->options->dt() : 0.0));
    output_keys.push_back(out->schedule_key());
  }
  // Old readers use the first N entries. Precise keys follow them in the same
  // tensor, so an old writer naturally replaces the whole value with N legacy
  // entries instead of carrying unknown metadata into a later new restart.
  for (auto out : pmb->output_types) {
    output_keys.push_back(out->schedule_key_v2());
  }

  out_vars["file_number"] = torch::tensor(output_file_numbers, torch::kInt64);
  out_vars["next_time"] = torch::tensor(output_next_times, torch::kFloat64);
  out_vars["output_key"] = torch::tensor(output_keys, torch::kInt64);

  // create filename: <basename>.<blockid>.<file_number>.part
  std::string fname;
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options->layout()->rank());

  fname.assign(pmb->options->output_dir());
  fname.append("/");
  fname.append(pmb->options->basename());
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  if (final_write) {
    fname.append("final");
  } else {
    fname.append(number);
  }
  fname.append(".part");

  // save to disk
  make_output_dir(pmb);
  kintera::save_tensors(out_vars, fname);

  if (options->combine()) {
    combine_blocks(pmb, final_write);
  }
}

}  // namespace snap
