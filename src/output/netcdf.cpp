// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_utils.hpp"
#include "writer_core.hpp"

#ifdef NETCDFOUTPUT
#include <netcdf.h>
#endif

namespace snap {
#ifdef NETCDFOUTPUT
namespace {

#define SNAP_NETCDF_CHECK(call)                                      \
  do {                                                               \
    int status__ = (call);                                           \
    if (status__ != NC_NOERR) {                                      \
      throw std::runtime_error(std::string(#call) +                  \
                               " failed: " + nc_strerror(status__)); \
    }                                                                \
  } while (false)

void put_text(int f, int v, char const* key, std::string const& text) {
  nc_put_att_text(f, v, key, text.size(), text.c_str());
}

// NC_FLOAT keeps ~7 significant digits: a relative change below ~1e-7 in a
// field, or a time step below ~1e-7 * t, does not show in the file.
// `double_precision: true` on the output block writes NC_DOUBLE. The default
// casts to float here, so an overflow stays +inf: NetCDF is not asked to
// narrow, because that stores the float fill value.
struct Values {
  bool dbl;
  void put(int f, int v, torch::Tensor t, size_t const* start = nullptr,
           size_t const* count = nullptr) const {
    if (dbl)
      t = t.contiguous();
    else
      t = t.to(torch::kFloat32).contiguous();
    if (!start && dbl) nc_put_var_double(f, v, t.data_ptr<double>());
    if (!start && !dbl) nc_put_var_float(f, v, t.data_ptr<float>());
    if (start && dbl)
      nc_put_vara_double(f, v, start, count, t.data_ptr<double>());
    if (start && !dbl)
      nc_put_vara_float(f, v, start, count, t.data_ptr<float>());
  }
};

class NetcdfSink : public Sink {
 public:
  void write(MeshBlockImpl* pmb, Frame const& fr, OutputOptionsImpl const& op,
             int file_number, double time) override;
  void combine(MeshBlockImpl* pmb, OutputOptionsImpl const& op,
               int file_number) override;
};

void NetcdfSink::write(MeshBlockImpl* pmb, Frame const& fr,
                       OutputOptionsImpl const& op, int file_number,
                       double time) {
  int rank = pmb->options->layout()->rank();
  // <basename>.block<rank>.<fileid>.<XXXXX>.nc
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  std::string fname =
      pmb->options->output_dir() + "/" + pmb->options->basename() + ".block" +
      std::to_string(rank) + "." + op.file_id() + "." + number + ".nc";
  int ifile;
  make_output_dir(pmb);
  SNAP_NETCDF_CHECK(nc_create(fname.c_str(), NC_NETCDF4, &ifile));

  bool sl[3] = {!op.x1_slice().empty(), !op.x2_slice().empty(),
                !op.x3_slice().empty()};
  int nc[3] = {fr.ie - fr.is + 1, fr.je - fr.js + 1, fr.ke - fr.ks + 1};
  int nf[3];
  for (int d = 0; d < 3; ++d) nf[d] = nc[d] + (nc[d] > 1 && !sl[d]);
  // dimensions: time, then x<d> and its faces when it has cells or a slice
  int idt, idx[3], idxf[3];
  nc_def_dim(ifile, "time", NC_UNLIMITED, &idt);
  for (int d = 0; d < 3; ++d) {
    std::string x = "x" + std::to_string(d + 1);
    nc_def_dim(ifile, x.c_str(), nc[d], &idx[d]);
    if (nc[d] > 1 || sl[d])
      nc_def_dim(ifile, (x + "f").c_str(), nf[d], &idxf[d]);
  }

  auto layout = pmb->get_layout();
  auto [level, lx1, lx2, lx3, nb1, nb2, nb3, face, pcol, prow] =
      output_block_location(layout, rank);
  int loc[3] = {lx1, lx2, lx3}, nb[3] = {nb1, nb2, nb3};
  for (int d = 0; d < 3; ++d)
    if (sl[d]) loc[d] = 0, nb[d] = 1;
  bool cs = layout->options->is_cubed_sphere();

  Values val{op.double_precision()};
  const nc_type nctype = val.dbl ? NC_DOUBLE : NC_FLOAT;
  int ivt, ivx[3], ivxf[3];
  nc_def_var(ifile, "time", nctype, 1, &idt, &ivt);
  put_text(ifile, ivt, "axis", "T");
  put_text(ifile, ivt, "units", "s");
  put_text(ifile, ivt, "long_name", "time");

  std::string const hunits =
      horizontal_axis_units(pmb->options->coord()->type());
  char const* axis[3] = {"Z", "X", "Y"};
  for (int d = 0; d < 3; ++d) {
    std::string x = "x" + std::to_string(d + 1), units = d ? hunits : "m";
    nc_def_var(ifile, x.c_str(), nctype, 1, &idx[d], &ivx[d]);
    put_text(ifile, ivx[d], "axis", axis[d]);
    put_text(ifile, ivx[d], "units", units);
    put_text(ifile, ivx[d], "long_name",
             std::string(axis[d]) + "-coordinate at cell center");
    auto const& want = d == 0   ? op.x1_slice()
                       : d == 1 ? op.x2_slice()
                                : op.x3_slice();
    int pos[4] = {1, sl[d] ? int(want.size()) : nc[d] * nb[d],
                  (sl[d] ? fr.slice_offset[d] : nc[d] * loc[d]) + 1, 0};
    pos[3] = pos[2] + nc[d] - 1;
    if (cs && d > 0 && sl[d]) {  // the panels are laid out 3 x 2
      int offset = (d == 1 ? pcol : prow) * want.size();
      pos[1] *= d == 1 ? 3 : 2;
      pos[2] += offset;
      pos[3] += offset;
    }
    nc_put_att_int(ifile, ivx[d], "domain_decomposition", NC_INT, 4, pos);
    if (nc[d] > 1 || sl[d]) {
      nc_def_var(ifile, (x + "f").c_str(), nctype, 1, &idxf[d], &ivxf[d]);
      put_text(ifile, ivxf[d], "units", units);
      put_text(ifile, ivxf[d], "long_name",
               std::string(axis[d]) + "-coordinate at cell face");
      if (!sl[d]) pos[0]--, pos[2]--;
      nc_put_att_int(ifile, ivxf[d], "domain_decomposition", NC_INT, 4, pos);
    }
  }

  // Count participating partitions, even when each owns a different number
  // of requested planes. This metadata also supports combining files later.
  auto count_blocks = [](std::vector<double> const& slices, double lower,
                         double upper, int nblocks) {
    if (slices.empty()) return nblocks;
    int count = 0;
    for (int b = 0; b < nblocks; ++b) {
      double lo = lower + b * (upper - lower) / nblocks;
      double hi = lo + (upper - lower) / nblocks;
      count += std::any_of(slices.begin(), slices.end(),
                           [=](double x) { return x >= lo && x < hi; });
    }
    return count;
  };
  auto co = pmb->pcoord->options;
  int nbtotal = count_blocks(op.x1_slice(), co->global_x1min(),
                             co->global_x1max(), layout->options->nb(1)) *
                count_blocks(op.x2_slice(), co->global_x2min(),
                             co->global_x2max(), layout->options->nb(2)) *
                count_blocks(op.x3_slice(), co->global_x3min(),
                             co->global_x3max(), layout->options->nb(3));
  if (cs) nbtotal *= 6;
  nc_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT, 1, &nbtotal);

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
      SNAP_NETCDF_CHECK(nc_def_var(ifile,
                                   sanitize_netcdf_name(raw_name).c_str(),
                                   nctype, ndim, dims, &iv));
      auto attr = pmeta->GetUnits(raw_name);
      if (attr != "") put_text(ifile, iv, "units", attr);
      attr = pmeta->GetLongName(raw_name);
      if (attr != "") put_text(ifile, iv, "long_name", attr);
      var_ids.push_back(iv);
    }
  }
  SNAP_NETCDF_CHECK(nc_enddef(ifile));

  size_t start[4] = {0, 0, 0, 0}, one = 1;
  val.put(ifile, ivt, torch::tensor({time}, torch::kFloat64), start, &one);

  // coordinates: the cells (or the selected ones) and their faces; x2 and x3
  // of a cubed-sphere panel shift by its place in the 3 x 2 layout
  int lo[3] = {fr.is, fr.js, fr.ks}, hi[3] = {fr.ie, fr.je, fr.ke};
  double shift[3] = {0., pcol * M_PI / 2., prow * M_PI / 2.};
  auto pc = pmb->pcoord;
  torch::Tensor cv[3] = {pc->x1v, pc->x2v, pc->x3v};
  torch::Tensor cf[3] = {pc->x1f, pc->x2f, pc->x3f};
  for (int d = 0; d < 3; ++d) {
    for (int face_axis = 0; face_axis < 2; ++face_axis) {
      if (face_axis && !(nc[d] > 1 || sl[d])) continue;
      auto x = (face_axis ? cf[d] : cv[d])
                   .to(torch::kCPU, torch::kFloat64)
                   .contiguous();
      auto const* xp = x.data_ptr<double>();
      std::vector<double> v;
      for (int i = lo[d]; i <= hi[d] + (face_axis && !sl[d]); ++i)
        v.push_back(d ? xp[sl[d] ? fr.slice[d][i] : i] + shift[d]
                      : xp[sl[d] ? fr.slice[d][i] : i]);
      val.put(ifile, face_axis ? ivxf[d] : ivx[d],
              torch::tensor(v, torch::kFloat64));
    }
  }

  size_t count[4];
  auto iv = var_ids.begin();
  for (auto const& f : fr.fields) {
    char k = fr.kind(f);
    int n0 = k == '1' ? nf[0] : nc[0], n2 = k == '3' ? nf[2] : nc[2],
        n1 = k == '2' ? nf[1] : nc[1];
    if (k == 'p')
      count[0] = 1, count[1] = nc[2], count[2] = nc[1];
    else
      count[0] = 1, count[1] = k == 'f' ? nf[0] : n0, count[2] = n2,
      count[3] = n1;
    for (int n = 0; n < f.nvar(); ++n)
      val.put(ifile, *iv++, fr.pack(f, n), start, count);
  }
  SNAP_NETCDF_CHECK(nc_close(ifile));
}

void NetcdfSink::combine(MeshBlockImpl* pmb, OutputOptionsImpl const& op,
                         int file_number) {
  combine_netcdf_blocks(pmb, op, file_number);
}

#undef SNAP_NETCDF_CHECK
}  // namespace

std::shared_ptr<Sink> make_netcdf_sink() {
  return std::make_shared<NetcdfSink>();
}
#else
namespace {
// NETCDF=OFF: a netcdf output writes nothing, as NetcdfOutput did
class NoNetcdfSink : public Sink {
 public:
  void write(MeshBlockImpl*, Frame const&, OutputOptionsImpl const&, int,
             double) override {}
};
}  // namespace

std::shared_ptr<Sink> make_netcdf_sink() {
  return std::make_shared<NoNetcdfSink>();
}
#endif  // NETCDFOUTPUT
}  // namespace snap
