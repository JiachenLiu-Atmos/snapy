// C/C++ headers
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>  // stringstream
#include <stdexcept>

// base
#include <configure.h>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"
#include "writer_core.hpp"

int mppnccombine(int argc, char* argv[]);

namespace snap {

void NetcdfOutput::combine_blocks(MeshBlockImpl* pmb, bool) {
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
  key.append(options->file_id());
  key.append("|");
  key.append(number);

  combine_on_root(layout, key, [&] {
    std::string infile;
    infile.assign(pmb->options->output_dir());
    infile.append("/");
    infile.append(pmb->options->basename());
    infile.append(".block*.");
    infile.append(options->file_id());
    infile.append(".");
    infile.append(number);
    infile.append(".nc");
    auto files = glob_block_files(infile, "NetcdfOutput::combine_blocks");

    std::string outfile;
    outfile.assign(pmb->options->output_dir());
    outfile.append("/");
    outfile.append(pmb->options->basename());
    outfile.append(".");
    outfile.append(options->file_id());
    outfile.append(".");
    outfile.append(number);
    outfile.append(".nc");

    int err = 0;
    remove(outfile.c_str());
    if (files.size() == 1) {
      err = std::rename(files.front().c_str(), outfile.c_str());
    } else {
      int argc = 3 + files.size();
      char** argv = new char*[argc];
      for (int i = 0; i < argc; ++i) argv[i] = new char[2048];
      snprintf(argv[0], 2048, "%s", "CombineBlocks");
      snprintf(argv[1], 2048, "%s", "-r");
      snprintf(argv[2], 2048, "%s", outfile.c_str());
      for (int i = 3; i < argc; ++i)
        snprintf(argv[i], 2048, "%s", files[i - 3].c_str());

      err = mppnccombine(argc, argv);
      for (int i = 0; i < argc; ++i) delete[] argv[i];
      delete[] argv;
    }
    if (err) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [NetcdfOutput::combine_blocks]"
          << std::endl
          << "NetCDF block combine failed with error " << err << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  });
#endif  // NETCDFOUTPUT
}
}  // namespace snap
