// MeshBlock, part BlockExchange (owner S): exchange phases and boundary
// functions. (V3 X5: src/mesh/meshblock.cpp in six parts)

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

#include "exchange_cut.hpp"
#include "meshblock.hpp"
#include "meshblock_internal.hpp"
#include "stage_executor.hpp"

namespace snap {

void MeshBlockImpl::exchange(Variables &vars, SyncOptions const &opts) const {
  // A subdivided cubed-sphere panel packs its cross-panel strip EXTENDED into
  // its own tangential halo (cs_interp_margin), and that halo is only written
  // by deserialize(). So the intra-panel round must COMPLETE before the
  // cross-panel round packs, which one round cannot do. Corners are synthesized
  // from the edge strips, so they come after both.
  auto const &lo = *options->layout();
  if (opts.interpolate() && !opts.intra_panel_only() &&
      !opts.cross_panel_only() && lo.is_cubed_sphere() &&
      (lo.px() > 1 || lo.py() > 1)) {
    SyncOptions intra = opts, cross = opts;
    _exchange_once(vars, intra.intra_panel_only(true).interpolate(false));
    _exchange_once(vars, cross.cross_panel_only(true));
    if (opts.skip_corner()) _playout->fill_corners(this, vars);
    return;
  }

  _exchange_once(vars, opts);
}

void MeshBlockImpl::_exchange_once(Variables &vars,
                                   SyncOptions const &opts) const {
  std::vector<CommWorkPtr> works;
  begin_exchange(vars, opts);
  launch_exchange(opts, works);
  finalize_exchange(vars, opts, works);
}

torch::Tensor MeshBlockImpl::board_buffer(std::tuple<int, int, int> key,
                                          torch::Tensor const &like) const {
  auto &buf = board_buffers[key];
  if (!buffer_matches(buf, like)) buf = torch::empty_like(like);
  return buf;
}

void MeshBlockImpl::begin_exchange(Variables &vars,
                                   SyncOptions const &opts) const {
  // the pack: device copies into the cached send buffers (captured in a
  // segment, B3; the warm-up allocates them)
  _playout->serialize(this, vars, opts);
}

void MeshBlockImpl::launch_exchange(SyncOptions const &opts,
                                    std::vector<CommWorkPtr> &works) const {
  // a lone block (one per process, no process group) exchanges with no
  // other block: its launch is at most its own periodic copies, device work
  // that stays inside the graph, as before B3
  bool lone = _playout->options->blocks_per_process() <= 1 &&
              !_playout->has_process_group();
  if (!in_segment_capture() || lone) {
    _playout->launch_exchange(this, opts, works);
    return;
  }
  // B3: a cut. The host step is the rendezvous with this process's blocks
  // (and their seam copies) or the communicator, waited on, so the unpack
  // (finalize_exchange: deserialize into the ghosts) is device work in the
  // segment's next graph. It runs again at every replay, when the pack ran
  // inside a graph and set no host state: it restores the send and receive
  // buffers this pack chose (serialize points them at its cache entry)
  auto sends = send_bufs;
  auto recvs = recv_bufs;
  exchange_cut([this, opts, sends, recvs]() {
    send_bufs = sends;
    recv_bufs = recvs;
    std::vector<CommWorkPtr> w;
    _playout->launch_exchange(this, opts, w);
    for (auto &x : w)
      if (x) x->wait();
  });
}

void MeshBlockImpl::finalize_exchange(Variables &vars, SyncOptions const &opts,
                                      std::vector<CommWorkPtr> &works) const {
  _playout->finalize(this, vars, opts, works);
}

void MeshBlockImpl::exchange_state(Variables &vars, char const *hydro, int type,
                                   char const *scalar) {
  SyncOptions opts;
  opts.interpolate(true).type(type);
  Variables sync_vars;
  sync_vars[hydro] = vars.at(hydro);
  exchange(sync_vars, opts);
  if (pscalar->nvar() > 0) {
    opts.type(kScalar);
    sync_vars.clear();
    sync_vars[scalar] = vars.count(scalar) ? vars.at(scalar) : torch::Tensor();
    exchange(sync_vars, opts);
  }
}

void MeshBlockImpl::exchange_ghost_zones(Variables &vars) {
  auto hydro_u = vars.at("hydro_u");
  auto scalar_s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();
  exchange_state(vars, "hydro_u", kConserved, "scalar_s");

  // Tangential exchange fills face ghosts but not corners (skip_corner).
  // Re-apply each installed wall function on the slabs exchange owns in the
  // other directions, so the corner matches the updated column. A physical
  // tangential face already wrote that corner. Outflow is not reapplied: its
  // background must match the full block and it needs primitives (#264).
  // The x2/x3 walls are included (C8.gauge): fill_corners synthesizes a seam
  // x wall corner from the edge strips, where an undivided block has the
  // wall's own ghost of the column, so without this the corner, and the
  // cells whose stencils reach it, depend on the decomposition.
  {
    int ng = pcoord->options->nghost();
    auto const &fns = options->bfuncs();
    auto const &names = options->bcnames();
    BoundaryFuncOptions op;
    op.nghost(ng);
    op.eos = phydro->peos.get();
    op.coord = pcoord.get();
    auto physical = [&](int face) {
      if (face < 0 || face >= static_cast<int>(fns.size()) || !fns[face])
        return false;
      // Cubed-sphere horizontal custom callbacks are built-in no-ops:
      // exchange owns these faces and their radial wall intersections.
      // Check callback identity, not its name, to retain genuine custom BCs.
      if (options->layout()->is_cubed_sphere() && face >= 2 &&
          is_custom_noop(fns[face]))
        return false;
      return !(static_cast<size_t>(face) < names.size() &&
               names[face].compare(0, 8, "periodic") == 0);
    };
    auto refresh = [&](torch::Tensor field, int type) {
      if (!field.defined() || field.numel() == 0) return;
      op.type(type);
      int nface =
          static_cast<int>(fns.size()) < 6 ? static_cast<int>(fns.size()) : 6;
      for (int f = 0; f < nface; ++f) {
        if (!fns[f] || is_outflow(fns[f])) continue;
        // an x2/x3 face that exchange owns (periodic, cubed-sphere no-op)
        if (f >= 2 && !physical(f)) continue;
        int spatial = 3 - f / 2;
        if (field.size(spatial) == 1) continue;
        // a reflecting wall is row-local (each row normal to it
        // mirrors its own interior), so when every tangential face is
        // exchange-owned one full-extent call writes the slabs' corner values
        // and rewrites the other rows with the same bits
        {
          bool any_orth = false, all_owned = true;
          for (int orth = 1; orth <= 3; ++orth) {
            if (orth == spatial || field.size(orth) <= 1) continue;
            any_orth = true;
            int inner = (3 - orth) * 2;
            if (physical(inner) || physical(inner + 1)) all_owned = false;
          }
          if (any_orth && all_owned && static_cast<size_t>(f) < names.size() &&
              names[f].compare(0, 10, "reflecting") == 0) {
            fns[f](field, spatial, op);
            continue;
          }
        }
        for (int orth = 1; orth <= 3; ++orth) {
          if (orth == spatial || field.size(orth) <= 1) continue;
          int n = static_cast<int>(field.size(orth));
          int g = ng < n / 2 ? ng : n / 2;
          if (g <= 0) continue;
          int inner = (3 - orth) * 2;
          if (!physical(inner)) fns[f](field.narrow(orth, 0, g), spatial, op);
          if (!physical(inner + 1))
            fns[f](field.narrow(orth, n - g, g), spatial, op);
        }
      }
    };
    refresh(hydro_u, kConserved);
    refresh(scalar_s, kScalar);
  }

  if (pscalar->nvar() > 0) {
    set_scalar_primitive(vars, scalar_s, hydro_u, *phydro);
  }
}

bool MeshBlockImpl::has_radiating_boundary() const {
  if (phydro->peos->options->type() == "shallow-water") return false;
  for (int f = 0; f < options->bfuncs().size(); ++f) {
    int nc = f / 2 == 0   ? pcoord->options->nc1()
             : f / 2 == 1 ? pcoord->options->nc2()
                          : pcoord->options->nc3();
    if (nc > 1 && is_outflow(options->bfuncs()[f])) return true;
  }
  return false;
}

void MeshBlockImpl::apply_boundaries(Variables &vars, torch::Tensor hydro,
                                     torch::Tensor tracers, bool primitive) {
  BoundaryFuncOptions op;
  op.nghost(pcoord->options->nghost());
  op.eos = phydro->peos.get();
  op.coord = pcoord.get();
  bool radiating = has_radiating_boundary();
  bool has_tracers = tracers.defined() && tracers.size(0) > 0;
  if (radiating) {
    TORCH_CHECK(vars.count("boundary_reference_w"),
                "outflow: missing initial boundary reference");
    op.reference = vars.at("boundary_reference_w");
    if (has_tracers) {
      TORCH_CHECK(vars.count("boundary_reference_r"),
                  "outflow: missing initial tracer reference");
      op.tracer_reference = vars.at("boundary_reference_r");
    }
  }

  torch::Tensor w, r;
  std::vector<int> pending_faces;
  // Consecutive radiating faces share one primitive conversion. Commit their
  // ghosts before another callback so it sees the selected representation and
  // the results of preceding faces, including corners.
  auto flush = [&]() {
    if (pending_faces.empty()) return;
    auto u = phydro->peos->compute("W->U", {w});
    // Tracer densities ride the conserved (dry) density, as in
    // set_scalar_primitive, not the total density w[IDN].
    auto s = has_tracers ? r * u[IDN] : torch::Tensor();
    // anomaly storage: the ghosts are filled in totals, stored as anomalies
    phydro->sub_reference_(u);
    for (int f : pending_faces) {
      int dim = 3 - f / 2, ng = op.nghost();
      int start = f % 2 ? hydro.size(dim) - ng : 0;
      hydro.narrow(dim, start, ng).copy_(u.narrow(dim, start, ng));
      if (has_tracers)
        tracers.narrow(dim, start, ng).copy_(s.narrow(dim, start, ng));
    }
    pending_faces.clear();
  };

  for (int f = 0; f < options->bfuncs().size(); ++f) {
    auto const &fn = options->bfuncs()[f];
    int dim = 3 - f / 2;
    if (!fn || hydro.size(dim) == 1) continue;
    if (radiating && is_outflow(fn)) {
      if (primitive) {
        w = hydro;
        r = tracers;
      } else if (pending_faces.empty()) {
        w = phydro->peos->compute(phydro->has_reference() ? "U'->W" : "U->W",
                                  {hydro.clone()});
        r = has_tracers ? tracers / phydro->dry_density(hydro)
                        : torch::Tensor();
      }
      op.type(kPrimitive);
      op.tracers = has_tracers ? r : torch::Tensor();
      fn(w, dim, op);
      if (!primitive) pending_faces.push_back(f);
    } else {
      flush();
      op.tracers = torch::Tensor();
      op.type(primitive ? kPrimitive : kConserved);
      fn(hydro, dim, op);
      if (has_tracers) {
        op.type(kScalar);
        fn(tracers, dim, op);
      }
    }
  }
  flush();
}

}  // namespace snap
