#include <configure.h>

#ifdef PNETCDFOUTPUT

#include <pnetcdf_comm.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/vectorize.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "netcdf_utils.hpp"
#include "output_formats.hpp"
#include "output_utils.hpp"
#include "writer_core.hpp"

namespace snap {
namespace {

#define SNAP_PINC_COMM_WORLD PINC_COMM_WORLD
#define SNAP_PINC_INFO_NULL PINC_INFO_NULL

#define SNAP_PINC_CHECK(call)                                            \
  do {                                                                   \
    int status__ = (call);                                               \
    if (status__ != NC_NOERR) {                                          \
      throw std::runtime_error(std::string(#call) +                      \
                               " failed: " + ncmpix_strerror(status__)); \
    }                                                                    \
  } while (false)

void put_vara_float(int ncid, int varid, const PINC_Offset* start,
                    const PINC_Offset* count, const float* data) {
  int req = -1;
  int status = NC_NOERR;
  SNAP_PINC_CHECK(
      ncmpix_iput_vara_float(ncid, varid, start, count, data, &req));
  SNAP_PINC_CHECK(ncmpix_wait_all(ncid, 1, &req, &status));
  SNAP_PINC_CHECK(status);
}

}  // namespace

PNetcdfOutput::PNetcdfOutput(OutputOptions const& options_)
    : OutputType(options_) {
  TORCH_CHECK(options->x1_slice().empty() && options->x2_slice().empty() &&
                  options->x3_slice().empty(),
              "Slice lists require type: netcdf with block combining.");
  // This writer is float-only (every nc*_def_var below is NC_FLOAT). Reject
  // at construction, which MeshBlock does at setup (meshblock.cpp, beside the
  // validate_slice calls), so a misconfigured run fails before its first step
  // rather than at its first output time.
  TORCH_CHECK(!options->double_precision(),
              "PNetcdfOutput: double_precision is not implemented for the "
              "parallel-netcdf writer; it would silently write NC_FLOAT. Use "
              "type: netcdf for double-precision output.");
}

void PNetcdfOutput::write_output_file(MeshBlockImpl* pmb_in,
                                      Variables const& vars,
                                      double current_time, bool final_write) {
  if (final_write) return;

  // Belt and braces: the real rejection is in the constructor above, which
  // fires at setup. This one catches a double_precision() mutated afterwards.
  TORCH_CHECK(!options->double_precision(),
              "PNetcdfOutput: double_precision is not implemented for the "
              "parallel-netcdf writer; it would silently write NC_FLOAT. Use "
              "type: netcdf for double-precision output.");

  auto pmb = LoadOutputData(pmb_in, vars);
  auto layout = pmb->get_layout();
  auto layout_options = layout->options;
  if (layout_options->blocks_per_process() != 1) {
    if (pmb != pmb_in) delete pmb;
    throw std::runtime_error(
        "PNetcdfOutput currently requires blocks_per_process=1");
  }

  auto pmeta = MetadataTable::GetInstance();

  int rank = layout_options->rank();
  std::tie(out_is, out_ie, out_js, out_je, out_ks, out_ke) =
      output_range(pmb, options->include_ghost_zones());

  if (!TransformOutputData(pmb)) {
    ClearOutputData();
    if (pmb != pmb_in) delete pmb;
    return;
  }

  try {
    make_output_dir(pmb);
  } catch (...) {
    if (pmb != pmb_in) delete pmb;
    throw;
  }

  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  std::string fname = pmb->options->output_dir() + "/" +
                      pmb->options->basename() + "." + options->file_id() +
                      "." + number + ".nc";

  int ifile;
  SNAP_PINC_CHECK(ncmpix_create(SNAP_PINC_COMM_WORLD, fname.c_str(), NC_CLOBBER,
                                SNAP_PINC_INFO_NULL, &ifile));

  int ncells1 = out_ie - out_is + 1;
  int ncells2 = out_je - out_js + 1;
  int ncells3 = out_ke - out_ks + 1;
  int nfaces1 = ncells1 > 1 ? ncells1 + 1 : ncells1;
  int nfaces2 = ncells2 > 1 ? ncells2 + 1 : ncells2;
  int nfaces3 = ncells3 > 1 ? ncells3 + 1 : ncells3;

  auto [level, lx1, lx2, lx3, nb1, nb2, nb3, face] =
      output_block_location(layout, rank);

  PINC_Offset nx1 = static_cast<PINC_Offset>(ncells1) * nb1;
  PINC_Offset nx2 = static_cast<PINC_Offset>(ncells2) * nb2;
  PINC_Offset nx3 = static_cast<PINC_Offset>(ncells3) * nb3;
  PINC_Offset nx1f = ncells1 > 1 ? nx1 + 1 : nx1;
  PINC_Offset nx2f = ncells2 > 1 ? nx2 + 1 : nx2;
  PINC_Offset nx3f = ncells3 > 1 ? nx3 + 1 : nx3;

  int idt, idx1, idx2, idx3, idx1f, idx2f, idx3f;
  SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "time", NC_UNLIMITED, &idt));
  SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x1", nx1, &idx1));
  if (ncells1 > 1) SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x1f", nx1f, &idx1f));
  SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x2", nx2, &idx2));
  if (ncells2 > 1) SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x2f", nx2f, &idx2f));
  SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x3", nx3, &idx3));
  if (ncells3 > 1) SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "x3f", nx3f, &idx3f));

  int ivt, ivx1, ivx2, ivx3, ivx1f, ivx2f, ivx3f;
  SNAP_PINC_CHECK(ncmpix_def_var(ifile, "time", NC_FLOAT, 1, &idt, &ivt));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivt, "axis", 1, "T"));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivt, "units", 1, "s"));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivt, "long_name", 4, "time"));

  SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x1", NC_FLOAT, 1, &idx1, &ivx1));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx1, "axis", 1, "Z"));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx1, "units", 1, "m"));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx1, "long_name", 27,
                                      "Z-coordinate at cell center"));
  if (ncells1 > 1) {
    SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x1f", NC_FLOAT, 1, &idx1f, &ivx1f));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx1f, "units", 1, "m"));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx1f, "long_name", 25,
                                        "Z-coordinate at cell face"));
  }

  std::string const hunits =
      horizontal_axis_units(pmb->options->coord()->type());
  SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x2", NC_FLOAT, 1, &idx2, &ivx2));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx2, "axis", 1, "X"));
  SNAP_PINC_CHECK(
      ncmpix_put_att_text(ifile, ivx2, "units", hunits.size(), hunits.c_str()));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx2, "long_name", 27,
                                      "X-coordinate at cell center"));
  if (ncells2 > 1) {
    SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x2f", NC_FLOAT, 1, &idx2f, &ivx2f));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx2f, "units", hunits.size(),
                                        hunits.c_str()));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx2f, "long_name", 25,
                                        "X-coordinate at cell face"));
  }

  SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x3", NC_FLOAT, 1, &idx3, &ivx3));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx3, "axis", 1, "Y"));
  SNAP_PINC_CHECK(
      ncmpix_put_att_text(ifile, ivx3, "units", hunits.size(), hunits.c_str()));
  SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx3, "long_name", 27,
                                      "Y-coordinate at cell center"));
  if (ncells3 > 1) {
    SNAP_PINC_CHECK(ncmpix_def_var(ifile, "x3f", NC_FLOAT, 1, &idx3f, &ivx3f));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx3f, "units", hunits.size(),
                                        hunits.c_str()));
    SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, ivx3f, "long_name", 25,
                                        "Y-coordinate at cell face"));
  }

  int nbtotal = nb1 * nb2 * nb3;
  SNAP_PINC_CHECK(ncmpix_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT,
                                     1, &nbtotal));

  OutputData* pdata = pfirst_data_;
  int total_vars = 0;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    total_vars += get_num_variables(grid, pdata->data);
    pdata = pdata->pnext;
  }

  int iaxis[4] = {idt, idx1, idx3, idx2};
  int iaxis1[4] = {idt, idx1f, idx3, idx2};
  int iaxis2[4] = {idt, idx1, idx3, idx2f};
  int iaxis3[4] = {idt, idx1, idx3f, idx2};
  int iaxis_23[3] = {idt, idx3, idx2};
  std::vector<int> var_ids(total_vars);
  int* ivar = var_ids.data();

  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    std::vector<std::string> varnames;
    if (names.size() >= static_cast<size_t>(nvar)) {
      for (int n = 0; n < nvar; ++n) varnames.push_back(names[n]);
    } else {
      for (int n = 0; n < nvar; ++n) {
        size_t pos = pdata->name.find('?');
        if (nvar == 1) {
          varnames.push_back(pos < pdata->name.length()
                                 ? pdata->name.substr(0, pos) +
                                       pdata->name.substr(pos + 1)
                                 : pdata->name);
        } else {
          char c[16];
          snprintf(c, sizeof(c), "%d", n + 1);
          varnames.push_back(pos < pdata->name.length()
                                 ? pdata->name.substr(0, pos) + c +
                                       pdata->name.substr(pos + 1)
                                 : pdata->name + c);
        }
      }
    }

    for (int n = 0; n < nvar; ++n) {
      auto const& raw_name = varnames[n];
      auto name = sanitize_netcdf_name(raw_name);
      if (grid == "CCF" && ncells1 > 1)
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis1, ivar));
      else if (grid == "CFC" && ncells2 > 1)
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis2, ivar));
      else if (grid == "FCC" && ncells3 > 1)
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis3, ivar));
      else if (grid == "--C")
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis, ivar));
      else if (grid == "-CC")
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 3, iaxis_23, ivar));
      else if (grid == "--F")
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis1, ivar));
      else if (grid == "---")
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 1, iaxis, ivar));
      else
        SNAP_PINC_CHECK(
            ncmpix_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis, ivar));

      auto attr = pmeta->GetUnits(raw_name);
      if (attr != "") {
        SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, *ivar, "units",
                                            attr.length(), attr.c_str()));
      }
      attr = pmeta->GetLongName(raw_name);
      if (attr != "") {
        SNAP_PINC_CHECK(ncmpix_put_att_text(ifile, *ivar, "long_name",
                                            attr.length(), attr.c_str()));
      }
      ivar++;
    }
    pdata = pdata->pnext;
  }

  SNAP_PINC_CHECK(ncmpix_enddef(ifile));

  std::vector<float> data(nfaces1 * nfaces3 * nfaces2);
  PINC_Offset start[4] = {0, static_cast<PINC_Offset>(ncells1) * lx1,
                          static_cast<PINC_Offset>(ncells3) * lx3,
                          static_cast<PINC_Offset>(ncells2) * lx2};
  PINC_Offset count[4] = {1, ncells1, ncells3, ncells2};
  PINC_Offset count1[4] = {1, nfaces1, ncells3, ncells2};
  PINC_Offset count2[4] = {1, ncells1, nfaces3, ncells2};
  PINC_Offset count3[4] = {1, ncells1, ncells3, nfaces2};
  PINC_Offset start_23[3] = {0, start[2], start[3]};
  PINC_Offset count_23[3] = {1, ncells3, ncells2};

  float timef = current_time;
  PINC_Offset stime = 0, ctime = 1;
  SNAP_PINC_CHECK(
      ncmpix_put_vara_float_all(ifile, ivt, &stime, &ctime, &timef));

  int coord_is = out_is;
  int coord_ie = out_ie;
  int coord_js = out_js;
  int coord_je = out_je;
  int coord_ks = out_ks;
  int coord_ke = out_ke;

  PINC_Offset start_x1[1] = {start[1]};
  PINC_Offset count_x1[1] = {ncells1};
  PINC_Offset count_x1f[1] = {nfaces1};
  for (int i = coord_is; i <= coord_ie; ++i)
    data[i - coord_is] = pmb->pcoord->x1v[i].item<float>();
  put_vara_float(ifile, ivx1, start_x1, count_x1, data.data());
  if (ncells1 > 1) {
    for (int i = coord_is; i <= coord_ie + 1; ++i)
      data[i - coord_is] = pmb->pcoord->x1f[i].item<float>();
    put_vara_float(ifile, ivx1f, start_x1, count_x1f, data.data());
  }

  PINC_Offset start_x2[1] = {start[3]};
  PINC_Offset count_x2[1] = {ncells2};
  PINC_Offset count_x2f[1] = {nfaces2};
  for (int j = coord_js; j <= coord_je; ++j) {
    data[j - coord_js] =
        pmb->pcoord->x2v[j].item<float>() + (face % 3) * M_PI / 2.;
  }
  put_vara_float(ifile, ivx2, start_x2, count_x2, data.data());
  if (ncells2 > 1) {
    for (int j = coord_js; j <= coord_je + 1; ++j) {
      data[j - coord_js] =
          pmb->pcoord->x2f[j].item<float>() + (face % 3) * M_PI / 2.;
    }
    put_vara_float(ifile, ivx2f, start_x2, count_x2f, data.data());
  }

  PINC_Offset start_x3[1] = {start[2]};
  PINC_Offset count_x3[1] = {ncells3};
  PINC_Offset count_x3f[1] = {nfaces3};
  for (int k = coord_ks; k <= coord_ke; ++k) {
    data[k - coord_ks] =
        pmb->pcoord->x3v[k].item<float>() + (face / 3) * M_PI / 2.;
  }
  put_vara_float(ifile, ivx3, start_x3, count_x3, data.data());
  if (ncells3 > 1) {
    for (int k = coord_ks; k <= coord_ke + 1; ++k) {
      data[k - coord_ks] =
          pmb->pcoord->x3f[k].item<float>() + (face / 3) * M_PI / 2.;
    }
    put_vara_float(ifile, ivx3f, start_x3, count_x3f, data.data());
  }

  ivar = var_ids.data();
  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ",");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    if (grid == "CCF" && ncells1 > 1) {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie + 1; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je; ++j)
              *it++ = pdata->data(n, k, j, i);
        put_vara_float(ifile, *ivar++, start, count1, data.data());
      }
    } else if (grid == "CFC" && ncells2 > 1) {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je + 1; ++j)
              *it++ = pdata->data(n, k, j, i);
        put_vara_float(ifile, *ivar++, start, count2, data.data());
      }
    } else if (grid == "FCC" && ncells3 > 1) {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke + 1; ++k)
            for (int j = out_js; j <= out_je; ++j)
              *it++ = pdata->data(n, k, j, i);
        put_vara_float(ifile, *ivar++, start, count3, data.data());
      }
    } else if (grid == "--C") {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie; ++i) *it++ = pdata->data(n, i);
        put_vara_float(ifile, *ivar++, start, count, data.data());
      }
    } else if (grid == "-CC") {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int k = out_ks; k <= out_ke; ++k)
          for (int j = out_js; j <= out_je; ++j) *it++ = pdata->data(n, k, j);
        put_vara_float(ifile, *ivar++, start_23, count_23, data.data());
      }
    } else if (grid == "--F") {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie + 1; ++i) *it++ = pdata->data(n, i);
        put_vara_float(ifile, *ivar++, start, count1, data.data());
      }
    } else if (grid == "---") {
      for (int n = 0; n < nvar; n++) {
        data[0] = pdata->data(n);
        put_vara_float(ifile, *ivar++, start, count, data.data());
      }
    } else {
      for (int n = 0; n < nvar; n++) {
        float* it = data.data();
        for (int i = out_is; i <= out_ie; ++i)
          for (int k = out_ks; k <= out_ke; ++k)
            for (int j = out_js; j <= out_je; ++j)
              *it++ = pdata->data(n, k, j, i);
        put_vara_float(ifile, *ivar++, start, count, data.data());
      }
    }
    pdata = pdata->pnext;
  }

  SNAP_PINC_CHECK(ncmpix_close(ifile));
  ClearOutputData();
  if (pmb != pmb_in) delete pmb;
}

#undef SNAP_PINC_CHECK
#undef SNAP_PINC_INFO_NULL
#undef SNAP_PINC_COMM_WORLD

}  // namespace snap

#endif  // PNETCDFOUTPUT
