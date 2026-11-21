// Restart: the writer, the bundle of block parts and the reader, with the
// #367 metadata checks (O.rst 3.4). Every defined variable is written; a
// bundle is one magic line, an index of "<part name>\t<bytes>" lines, an
// empty line, then the parts in index order.

// C/C++
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "restart.hpp"
#include "writer_core.hpp"

namespace fs = std::filesystem;

namespace snap {
namespace {
constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";
}  // namespace

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
    // a carry read back from an earlier restart is not state: the budget
    // output writes its own below, a deck without one writes none
    if (name.rfind(kBudgetCarryPrefix, 0) == 0) continue;
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
    TORCH_CHECK(u.scalar_type() == torch::kFloat32 ||
                    u.scalar_type() == torch::kFloat64,
                "restart: hydro state must be float32 or float64");
    out_vars[kHydroRestartMetadata] =
        torch::tensor({int64_t(kHydroRestartVersion),
                       int64_t(u.scalar_type() == torch::kFloat32 ? 32 : 64)},
                      torch::kInt64);
    std::array<torch::Tensor, 3> grid = {pmb->pcoord->x1f, pmb->pcoord->x2f,
                                         pmb->pcoord->x3f};
    for (int d = 0; d < 3; ++d)
      out_vars[kHydroReferenceGrid[d]] = grid[d].to(torch::kCPU);
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
    out->save_carry(pmb, vars, advance, out_vars);
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

static int make_restart_bundle(std::string const& bundle_name,
                               std::vector<std::string> const& file_list) {
  namespace fs = std::filesystem;

  std::vector<std::uintmax_t> sizes;
  sizes.reserve(file_list.size());
  for (auto const& file : file_list) {
    sizes.push_back(fs::file_size(file));
  }

  std::ofstream out(bundle_name, std::ios::binary | std::ios::trunc);
  if (!out) return -1;

  out << kRestartBundleMagic << "\n";
  out << file_list.size() << "\n";
  for (size_t i = 0; i < file_list.size(); ++i) {
    out << fs::path(file_list[i]).filename().string() << "\t" << sizes[i]
        << "\n";
  }
  out << "\n";
  if (!out) return -1;

  std::vector<char> buffer(1 << 20);
  for (auto const& file : file_list) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return -1;

    while (in) {
      in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      auto count = in.gcount();
      if (count > 0) {
        out.write(buffer.data(), count);
      }
    }
    if (!out) return -1;
  }

  out.flush();
  return out ? 0 : -1;
}

void RestartOutput::combine_blocks(MeshBlockImpl* pmb, bool final_write) {
  auto layout = pmb->get_layout();
  char number[64];
  snprintf(number, sizeof(number), "%05d", file_number);

  std::string key;
  key.assign(pmb->options->output_dir());
  key.append("|");
  key.append(pmb->options->basename());
  key.append("|restart|");
  key.append(final_write ? "final" : number);

  combine_on_root(layout, key, [&] {
    std::string infile;
    infile.assign(pmb->options->output_dir());
    infile.append("/");
    infile.append(pmb->options->basename());
    infile.append(".block*.");
    if (final_write) {
      infile.append("final");
    } else {
      infile.append(number);
    }
    infile.append(".part");
    auto file_list = glob_block_files(infile, "RestartOutput::combine_blocks");

    std::string outfile;
    outfile.assign(pmb->options->output_dir());
    outfile.append("/");
    outfile.append(pmb->options->basename());
    outfile.append(".");
    if (final_write) {
      outfile.append("final");
    } else {
      outfile.append(number);
    }
    outfile.append(".restart");

    int err = 0;
    remove(outfile.c_str());
    if (file_list.size() == 1) {
      err = std::rename(file_list.front().c_str(), outfile.c_str());
    } else {
      err = make_restart_bundle(outfile, file_list);
    }

    if (err) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [RestartOutput::combine_blocks]"
          << std::endl
          << "restart bundling failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

    // remove input part files
    if (file_list.size() != 1) {
      for (auto const& f : file_list) {
        remove(f.c_str());
      }
    }
  });
}

namespace {
[[noreturn]] void fail(std::string const& path, std::string const& what) {
  throw std::runtime_error(path + ": " + what);
}

// the rank in a part name <basename>.block<rank>.<number>.part, or -1
int part_rank(std::string const& name) {
  auto dot2 = name.rfind('.', name.size() - std::strlen(".part") - 1);
  auto dot1 = dot2 == std::string::npos ? dot2 : name.rfind('.', dot2 - 1);
  if (dot1 == std::string::npos || name.compare(dot1 + 1, 5, "block") != 0 ||
      name.size() < 5 || name.compare(name.size() - 5, 5, ".part") != 0)
    return -1;
  return std::stoi(name.substr(dot1 + 6, dot2 - dot1 - 6));
}
}  // namespace

Variables load_restart(std::string const& path, int block_rank) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "failed to open restart file");
  std::string line;
  std::getline(in, line);
  if (line != kRestartBundleMagic) return kintera::load_tensors(path);

  // the index: a count, then "<part name>\t<bytes>" lines, then an empty line
  std::getline(in, line);
  if (line.empty()) fail(path, "restart bundle missing entry count");
  size_t count = std::stoull(line);
  std::vector<std::pair<std::string, std::uintmax_t>> entries;
  for (size_t i = 0; i < count; ++i) {
    std::getline(in, line);
    auto tab = line.find('\t');
    if (!in || tab == std::string::npos)
      fail(path, "restart bundle index line missing tab");
    entries.emplace_back(line.substr(0, tab),
                         std::stoull(line.substr(tab + 1)));
  }
  std::getline(in, line);
  if (!in || !line.empty())
    fail(path, "restart bundle header missing terminator");

  std::streamoff offset = in.tellg();
  for (auto const& [name, size] : entries) {
    if (part_rank(name) != block_rank) {
      offset += static_cast<std::streamoff>(size);
      continue;
    }
    std::vector<char> bytes(size);
    in.seekg(offset);
    in.read(bytes.data(), static_cast<std::streamsize>(size));
    if (static_cast<std::uintmax_t>(in.gcount()) != size)
      fail(path, "truncated restart bundle payload");
    // kintera reads a file: the part goes through a temporary copy
    auto tmp = fs::temp_directory_path() /
               ("snapy_restart_" + std::to_string(std::random_device{}()) +
                "_" + std::to_string(block_rank) + ".part");
    {
      std::ofstream out(tmp, std::ios::binary);
      out.write(bytes.data(), static_cast<std::streamsize>(size));
      if (!out) fail(path, "failed to create temp file for restart bundle");
    }
    auto vars = kintera::load_tensors(tmp.string());
    std::error_code ec;
    fs::remove(tmp, ec);
    return vars;
  }
  fail(path, "no part for block " + std::to_string(block_rank) +
                 " in restart bundle");
}

void check_restart_metadata(Variables const& data,
                            std::array<torch::Tensor, 3> const& grid,
                            bool perturbation_flux) {
  if (data.count(kHydroRestartMetadata)) {
    auto const& meta = data.at(kHydroRestartMetadata);
    TORCH_CHECK(meta.scalar_type() == torch::kInt64 && meta.dim() == 1 &&
                    meta.numel() == 2,
                "restart: invalid hydro restart metadata");
    TORCH_CHECK(meta[0].item<int64_t>() == kHydroRestartVersion,
                "restart: unsupported hydro restart metadata version");
    TORCH_CHECK(data.count("hydro_ref") && data.count("hydro_u_stored"),
                "restart: metadata requires hydro_ref and hydro_u_stored");
    auto const& stored = data.at("hydro_u_stored");
    int bits = stored.scalar_type() == torch::kFloat32   ? 32
               : stored.scalar_type() == torch::kFloat64 ? 64
                                                         : 0;
    TORCH_CHECK(bits && meta[1].item<int64_t>() == bits &&
                    stored.sizes() == data.at("hydro_u").sizes(),
                "restart: stored hydro dtype/shape disagrees with metadata");
    // Regridding needs an explicit remap of totals and a new background.
    // Never attach a serialized reference to a different same-shaped grid.
    for (int d = 0; d < 3; ++d) {
      auto key = kHydroReferenceGrid[d];
      TORCH_CHECK(data.count(key) &&
                      torch::equal(data.at(key), grid[d].to(torch::kCPU)),
                  "restart: reference grid mismatch at ", key,
                  "; remap totals and initialize a new background");
    }
  }
  TORCH_CHECK(!perturbation_flux ||
                  (data.count("hydro_ref") && data.count("hydro_ref_species")),
              "restart: perturbation-flux requires the original hydro_ref "
              "and hydro_ref_species");
}

namespace {
// the row names as one int8 tensor, newline-terminated
torch::Tensor pack_names(std::vector<std::string> const& names) {
  std::string all;
  for (auto const& n : names) all += n + "\n";
  auto t = torch::empty({static_cast<int64_t>(all.size())}, torch::kInt8);
  std::memcpy(t.data_ptr<int8_t>(), all.data(), all.size());
  return t;
}

std::vector<std::string> unpack_names(torch::Tensor const& t) {
  auto c = t.to(torch::kCPU, torch::kInt8).contiguous();
  std::string all(reinterpret_cast<char const*>(c.data_ptr<int8_t>()),
                  c.numel());
  std::vector<std::string> names;
  std::istringstream in(all);
  for (std::string n; std::getline(in, n);) names.push_back(n);
  return names;
}

std::string carry_key(char const* field) {
  return std::string(kBudgetCarryPrefix) + field;
}
}  // namespace

void write_budget_carry(BudgetCarry const& carry, Variables& out) {
  if (!carry.defined()) return;
  out[carry_key("rows")] = pack_names(carry.rows);
  out[carry_key("total_prev")] = carry.total_prev.to(torch::kCPU);
  out[carry_key("pending")] = carry.pending.to(torch::kCPU);
  out[carry_key("counter")] = carry.counter.to(torch::kCPU);
  out[carry_key("minimum")] = carry.minimum.to(torch::kCPU);
  out[carry_key("interval")] = torch::tensor({carry.interval}, torch::kFloat64);
}

BudgetCarry read_budget_carry(Variables const& data,
                              std::vector<std::string> const& rows) {
  BudgetCarry carry;
  for (auto field :
       {"rows", "total_prev", "pending", "counter", "minimum", "interval"})
    if (!data.count(carry_key(field))) return carry;
  auto saved = unpack_names(data.at(carry_key("rows")));
  if (saved != rows) return carry;  // other rows: a new interval
  carry.rows = saved;
  carry.total_prev = data.at(carry_key("total_prev")).to(torch::kFloat64);
  carry.pending = data.at(carry_key("pending")).to(torch::kFloat64);
  carry.counter = data.at(carry_key("counter")).to(torch::kInt64);
  carry.minimum = data.at(carry_key("minimum")).to(torch::kFloat64);
  carry.interval = data.at(carry_key("interval")).item<double>();
  return carry;
}

}  // namespace snap
