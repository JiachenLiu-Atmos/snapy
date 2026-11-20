// C/C++ headers
#include <cstdio>
#include <sstream>  // stringstream
#include <stdexcept>

// base
#include <configure.h>

// snap
#include <snap/mesh/meshblock.hpp>

#include "writer_core.hpp"

namespace snap {

void combine_netcdf_blocks(MeshBlockImpl* pmb, OutputOptionsImpl const& op,
                           int file_number) {
// Only proceed if NETCDF output enabled
#ifdef NETCDFOUTPUT
  auto layout = pmb->get_layout();
  char number[64];
  snprintf(number, sizeof(number), "%05d", file_number);

  std::string key;
  key.assign(pmb->options->output_dir());
  key.append("|");
  key.append(pmb->options->basename());
  key.append("|");
  key.append(op.file_id());
  key.append("|");
  key.append(number);

  combine_on_root(layout, key, [&] {
    std::string infile;
    infile.assign(pmb->options->output_dir());
    infile.append("/");
    infile.append(pmb->options->basename());
    infile.append(".block*.");
    infile.append(op.file_id());
    infile.append(".");
    infile.append(number);
    infile.append(".nc");
    auto files = glob_block_files(infile, "combine_netcdf_blocks");

    std::string outfile;
    outfile.assign(pmb->options->output_dir());
    outfile.append("/");
    outfile.append(pmb->options->basename());
    outfile.append(".");
    outfile.append(op.file_id());
    outfile.append(".");
    outfile.append(number);
    outfile.append(".nc");

    int err = 0;
    remove(outfile.c_str());
    if (files.size() == 1) {
      err = std::rename(files.front().c_str(), outfile.c_str());
    } else {
      err = combine_parts(outfile, files);
    }
    if (err) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [combine_netcdf_blocks]" << std::endl
          << "NetCDF block combine failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  });
#endif  // NETCDFOUTPUT
}
}  // namespace snap
