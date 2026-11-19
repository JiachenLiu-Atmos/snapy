#include <configure.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_utils.hpp"
#include "writer_core.hpp"

#ifdef PNETCDFOUTPUT
#include <pnetcdf_comm.h>
#endif

namespace snap {
#ifdef PNETCDFOUTPUT
namespace {

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

void put_text(int f, int v, char const* key, std::string const& text) {
  SNAP_PINC_CHECK(ncmpix_put_att_text(f, v, key, text.size(), text.c_str()));
}

// one shared file for all blocks; float only, no slices
class PNetcdfSink : public Sink {
 public:
  void write(MeshBlockImpl* pmb, Frame const& fr, OutputOptionsImpl const& op,
             int file_number, double time) override;
};

void PNetcdfSink::write(MeshBlockImpl* pmb, Frame const& fr,
                        OutputOptionsImpl const& op, int file_number,
                        double time) {
  // Belt and braces: the real rejection is in make_pnetcdf_sink, at setup.
  // This one catches a double_precision() mutated afterwards.
  TORCH_CHECK(!op.double_precision(),
              "PNetcdfOutput: double_precision is not implemented for the "
              "parallel-netcdf writer; it would silently write NC_FLOAT. Use "
              "type: netcdf for double-precision output.");
  auto layout = pmb->get_layout();
  if (layout->options->blocks_per_process() != 1)
    throw std::runtime_error(
        "PNetcdfOutput currently requires blocks_per_process=1");
  make_output_dir(pmb);

  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  std::string fname = pmb->options->output_dir() + "/" +
                      pmb->options->basename() + "." + op.file_id() + "." +
                      number + ".nc";
  int ifile;
  SNAP_PINC_CHECK(ncmpix_create(PINC_COMM_WORLD, fname.c_str(), NC_CLOBBER,
                                PINC_INFO_NULL, &ifile));

  int nc[3] = {fr.ie - fr.is + 1, fr.je - fr.js + 1, fr.ke - fr.ks + 1};
  int nf[3];
  for (int d = 0; d < 3; ++d) nf[d] = nc[d] > 1 ? nc[d] + 1 : nc[d];
  auto [level, lx1, lx2, lx3, nb1, nb2, nb3, face] =
      output_block_location(layout, layout->options->rank());
  int nb[3] = {nb1, nb2, nb3};

  int idt, idx[3], idxf[3];
  SNAP_PINC_CHECK(ncmpix_def_dim(ifile, "time", NC_UNLIMITED, &idt));
  for (int d = 0; d < 3; ++d) {
    std::string x = "x" + std::to_string(d + 1);
    PINC_Offset n = static_cast<PINC_Offset>(nc[d]) * nb[d];
    SNAP_PINC_CHECK(ncmpix_def_dim(ifile, x.c_str(), n, &idx[d]));
    if (nc[d] > 1)
      SNAP_PINC_CHECK(
          ncmpix_def_dim(ifile, (x + "f").c_str(), n + 1, &idxf[d]));
  }

  int ivt, ivx[3], ivxf[3];
  SNAP_PINC_CHECK(ncmpix_def_var(ifile, "time", NC_FLOAT, 1, &idt, &ivt));
  put_text(ifile, ivt, "axis", "T");
  put_text(ifile, ivt, "units", "s");
  put_text(ifile, ivt, "long_name", "time");
  std::string const hunits =
      horizontal_axis_units(pmb->options->coord()->type());
  char const* axis[3] = {"Z", "X", "Y"};
  for (int d = 0; d < 3; ++d) {
    std::string x = "x" + std::to_string(d + 1), units = d ? hunits : "m";
    SNAP_PINC_CHECK(
        ncmpix_def_var(ifile, x.c_str(), NC_FLOAT, 1, &idx[d], &ivx[d]));
    put_text(ifile, ivx[d], "axis", axis[d]);
    put_text(ifile, ivx[d], "units", units);
    put_text(ifile, ivx[d], "long_name",
             std::string(axis[d]) + "-coordinate at cell center");
    if (nc[d] > 1) {
      SNAP_PINC_CHECK(ncmpix_def_var(ifile, (x + "f").c_str(), NC_FLOAT, 1,
                                     &idxf[d], &ivxf[d]));
      put_text(ifile, ivxf[d], "units", units);
      put_text(ifile, ivxf[d], "long_name",
               std::string(axis[d]) + "-coordinate at cell face");
    }
  }
  int nbtotal = nb1 * nb2 * nb3;
  SNAP_PINC_CHECK(ncmpix_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT,
                                     1, &nbtotal));

  // file axes are (time, x1, x3, x2); a face grid swaps in its face axis
  auto pmeta = MetadataTable::GetInstance();
  std::vector<int> var_ids;
  for (auto const& f : fr.fields) {
    char k = fr.kind(f);
    int ax[4] = {idt, k == '1' ? idxf[0] : idx[0], k == '3' ? idxf[2] : idx[2],
                 k == '2' ? idxf[1] : idx[1]};
    int ax_p[3] = {idt, idx[2], idx[1]}, ax_f[2] = {idt, idxf[0]};
    int ndim = k == 'z' ? 2 : k == 'p' ? 3 : k == 'f' ? 2 : k == 's' ? 1 : 4;
    int* dims = k == 'p' ? ax_p : k == 'f' ? ax_f : ax;
    for (auto const& raw_name : f.names()) {
      int iv;
      SNAP_PINC_CHECK(ncmpix_def_var(ifile,
                                     sanitize_netcdf_name(raw_name).c_str(),
                                     NC_FLOAT, ndim, dims, &iv));
      auto attr = pmeta->GetUnits(raw_name);
      if (attr != "") put_text(ifile, iv, "units", attr);
      attr = pmeta->GetLongName(raw_name);
      if (attr != "") put_text(ifile, iv, "long_name", attr);
      var_ids.push_back(iv);
    }
  }
  SNAP_PINC_CHECK(ncmpix_enddef(ifile));

  float timef = time;
  PINC_Offset stime = 0, ctime = 1;
  SNAP_PINC_CHECK(
      ncmpix_put_vara_float_all(ifile, ivt, &stime, &ctime, &timef));

  // this block's place: x1, x3, x2 in the file's axis order
  PINC_Offset start[4] = {0, static_cast<PINC_Offset>(nc[0]) * lx1,
                          static_cast<PINC_Offset>(nc[2]) * lx3,
                          static_cast<PINC_Offset>(nc[1]) * lx2};
  PINC_Offset off[3] = {start[1], start[3], start[2]};
  int lo[3] = {fr.is, fr.js, fr.ks};
  double shift[3] = {0., (face % 3) * M_PI / 2., (face / 3) * M_PI / 2.};
  auto pc = pmb->pcoord;
  torch::Tensor cv[3] = {pc->x1v, pc->x2v, pc->x3v};
  torch::Tensor cf[3] = {pc->x1f, pc->x2f, pc->x3f};
  for (int d = 0; d < 3; ++d) {
    for (int face_axis = 0; face_axis < 1 + (nc[d] > 1); ++face_axis) {
      // the float value, then the panel shift (in double), as before
      auto x = (face_axis ? cf[d] : cv[d]).to(torch::kCPU, torch::kFloat32);
      auto const* xp = x.contiguous().data_ptr<float>();
      std::vector<float> v;
      for (int i = lo[d]; i < lo[d] + (face_axis ? nf[d] : nc[d]); ++i)
        v.push_back(d ? xp[i] + shift[d] : xp[i]);
      PINC_Offset count = v.size();
      put_vara_float(ifile, face_axis ? ivxf[d] : ivx[d], &off[d], &count,
                     v.data());
    }
  }

  PINC_Offset count[4], start_23[3] = {0, start[2], start[3]};
  auto iv = var_ids.begin();
  for (auto const& f : fr.fields) {
    char k = fr.kind(f);
    count[0] = 1;
    count[1] = k == '1' || k == 'f' ? nf[0] : nc[0];
    count[2] = k == 'p' ? nc[1] : k == '3' ? nf[2] : nc[2];
    count[3] = k == '2' ? nf[1] : nc[1];
    if (k == 'p') count[1] = nc[2];
    for (int n = 0; n < f.nvar(); ++n) {
      auto t = fr.pack(f, n).to(torch::kFloat32).contiguous();
      put_vara_float(ifile, *iv++, k == 'p' ? start_23 : start, count,
                     t.data_ptr<float>());
    }
  }
  SNAP_PINC_CHECK(ncmpix_close(ifile));
}

#undef SNAP_PINC_CHECK
}  // namespace

std::shared_ptr<Sink> make_pnetcdf_sink(OutputOptionsImpl const& op) {
  TORCH_CHECK(
      op.x1_slice().empty() && op.x2_slice().empty() && op.x3_slice().empty(),
      "Slice lists require type: netcdf with block combining.");
  // float only: reject at setup (MeshBlock builds its writers then), not at
  // the first output time
  TORCH_CHECK(!op.double_precision(),
              "PNetcdfOutput: double_precision is not implemented for the "
              "parallel-netcdf writer; it would silently write NC_FLOAT. Use "
              "type: netcdf for double-precision output.");
  return std::make_shared<PNetcdfSink>();
}
#else
std::shared_ptr<Sink> make_pnetcdf_sink(OutputOptionsImpl const&) {
  throw std::runtime_error("Output type 'pnetcdf' requires PNETCDF=ON.");
}
#endif  // PNETCDFOUTPUT
}  // namespace snap
