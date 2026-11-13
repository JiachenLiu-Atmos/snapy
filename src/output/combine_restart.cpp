// C/C++ headers
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"
#include "writer_core.hpp"

namespace snap {
namespace {
constexpr char kRestartBundleMagic[] = "SNAPY_RESTART_BUNDLE_V1";
}  // namespace

int make_restart_bundle(std::string const& bundle_name,
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

}  // namespace snap
