// combine_parts: joins the per-block netcdf files snapy writes into one file,
// the `-r <outfile> <parts...>` form of mppnccombine and nothing else
// (O.rst 10.1). Not called by runs yet: G-2 compares the two first (O6a).

// C/C++
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>

// base
#include <configure.h>

#include "writer_core.hpp"

#ifdef NETCDFOUTPUT
#include <netcdf.h>
#endif

namespace snap {

#ifdef NETCDFOUTPUT
namespace {

// the output variable of one input variable: whether it is split across the
// parts, and the whole record assembled from them (zero where none writes)
struct Var {
  std::vector<int> dims;
  size_t esize = 0;
  bool split = false;
  std::vector<char> whole;
};

struct Open {
  int id = -1;
  ~Open() {
    if (id >= 0) nc_close(id);
  }
};

}  // namespace

#define SNAP_TRY(call)              \
  do {                              \
    if ((err = (call))) return err; \
  } while (false)

int combine_parts(std::string const& outfile,
                  std::vector<std::string> const& parts) {
  struct stat st;
  if (parts.empty()) return NC_EINVAL;
  if (stat(outfile.c_str(), &st) == 0) return NC_EEXIST;
  int err;
  Open out;
  SNAP_TRY(nc_create(outfile.c_str(), NC_NETCDF4, &out.id));
  SNAP_TRY(nc_set_fill(out.id, NC_NOFILL, nullptr));

  std::vector<Var> vars;
  std::vector<size_t> full;  // dimension lengths of the joined file
  int unlim = -1, pending = 0;
  char name[NC_MAX_NAME + 1];
  for (size_t f = 0; f < parts.size(); ++f) {
    Open in;
    SNAP_TRY(nc_open(parts[f].c_str(), NC_NOWRITE, &in.id));
    int nfiles = 0, ndims, nvars, ngatts;
    nc_get_att_int(in.id, NC_GLOBAL, "NumFilesInSet", &nfiles);
    SNAP_TRY(nc_inq(in.id, &ndims, &nvars, &ngatts, &unlim));
    // a dimension with a coordinate variable that carries
    // domain_decomposition = {first, last, start, end} (1-based) is split
    std::vector<size_t> len(ndims), start(ndims, 0);
    std::vector<bool> split(ndims, false);
    for (int d = 0; d < ndims; ++d) SNAP_TRY(nc_inq_dimlen(in.id, d, &len[d]));
    if (f == 0) full = len;
    for (int v = 0, d, dd[4]; v < nvars; ++v) {
      SNAP_TRY(nc_inq_varname(in.id, v, name));
      if (nc_inq_dimid(in.id, name, &d) != NC_NOERR ||
          nc_get_att_int(in.id, v, "domain_decomposition", dd) != NC_NOERR)
        continue;
      if (f == 0) full[d] = dd[1] - dd[0] + 1;
      start[d] = dd[2] - dd[0];
      split[d] = true;
    }
    if (f == 0) {  // the joined file's structure is the first part's
      for (int d = 0; d < ndims; ++d) {
        int id;
        SNAP_TRY(nc_inq_dimname(in.id, d, name));
        SNAP_TRY(
            nc_def_dim(out.id, name, d == unlim ? NC_UNLIMITED : full[d], &id));
      }
      vars.resize(nvars);
      for (int v = 0; v < nvars; ++v) {
        nc_type type;
        int nd, natts, id, dimids[NC_MAX_VAR_DIMS];
        SNAP_TRY(nc_inq_var(in.id, v, name, &type, &nd, dimids, &natts));
        SNAP_TRY(nc_inq_type(in.id, type, nullptr, &vars[v].esize));
        vars[v].dims.assign(dimids, dimids + nd);
        SNAP_TRY(nc_def_var(out.id, name, type, nd, dimids, &id));
        for (int a = 0; a < natts; ++a) {
          SNAP_TRY(nc_inq_attname(in.id, v, a, name));
          if (std::string(name) != "domain_decomposition")
            SNAP_TRY(nc_copy_att(in.id, v, name, out.id, id));
        }
      }
      for (int a = 0; a < ngatts; ++a) {
        SNAP_TRY(nc_inq_attname(in.id, NC_GLOBAL, a, name));
        std::string key = name;
        if (key == "filename")
          SNAP_TRY(nc_put_att_text(out.id, NC_GLOBAL, name, outfile.size(),
                                   outfile.c_str()));
        else if (key != "NumFilesInSet")
          SNAP_TRY(nc_copy_att(in.id, NC_GLOBAL, name, out.id, NC_GLOBAL));
      }
      SNAP_TRY(nc_enddef(out.id));
    }
    if (unlim < 0 || len[unlim] != 1) return NC_EINVAL;  // one time record

    for (int v = 0; v < nvars; ++v) {
      auto& var = vars[v];
      size_t nd = var.dims.size(), n = 1, nfull = 1;
      std::vector<size_t> zero(nd, 0), count(nd), whole(nd);
      if (f == 0)
        for (int d : var.dims) var.split = var.split || split[d];
      for (size_t i = 0; i < nd; ++i) {
        count[i] = len[var.dims[i]];
        whole[i] = var.dims[i] == unlim ? 1 : full[var.dims[i]];
        n *= count[i];
        nfull *= whole[i];
      }
      if (!var.split && f > 0) continue;  // the first part's copy
      std::vector<char> buf(n * var.esize);
      SNAP_TRY(nc_get_vara(in.id, v, zero.data(), count.data(), buf.data()));
      if (!var.split) {
        SNAP_TRY(nc_put_vara(out.id, v, zero.data(), count.data(), buf.data()));
        continue;
      }
      if (f == 0) var.whole.assign(nfull * var.esize, 0);
      // place the part, row-major, at its offset in the whole record
      for (size_t k = 0; k < n; ++k) {
        size_t rem = k, at = 0, stride = 1;
        for (size_t i = nd; i-- > 0;) {
          size_t idx = rem % count[i];
          rem /= count[i];
          if (var.dims[i] != unlim) at += (idx + start[var.dims[i]]) * stride;
          stride *= whole[i];
        }
        std::copy_n(&buf[k * var.esize], var.esize, &var.whole[at * var.esize]);
      }
    }
    // write the split variables once a set of NumFilesInSet parts is in
    if (++pending == nfiles || f + 1 == parts.size()) {
      pending = 0;
      for (size_t v = 0; v < vars.size(); ++v) {
        if (!vars[v].split) continue;
        size_t nd = vars[v].dims.size();
        std::vector<size_t> zero(nd, 0), whole(nd);
        for (size_t i = 0; i < nd; ++i)
          whole[i] = vars[v].dims[i] == unlim ? 1 : full[vars[v].dims[i]];
        SNAP_TRY(nc_put_vara(out.id, v, zero.data(), whole.data(),
                             vars[v].whole.data()));
      }
    }
  }
  SNAP_TRY(nc_close(out.id));
  out.id = -1;
  for (auto const& p : parts) unlink(p.c_str());  // -r: the parts go
  return NC_NOERR;
}

#undef SNAP_TRY
#else
int combine_parts(std::string const&, std::vector<std::string> const&) {
  return -1;
}
#endif  // NETCDFOUTPUT

}  // namespace snap
