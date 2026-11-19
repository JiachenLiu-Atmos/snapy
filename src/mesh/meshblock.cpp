// MeshBlock, part BlockInit (owner X): options, modules, the state store,
// initial state. (V3 X5: src/mesh/meshblock.cpp in six parts)

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
#include <snap/input/read_restart_file.hpp>
#include <snap/mesh/signal_handler.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/restart_metadata.hpp>
#include <snap/utils/log.hpp>

#include "cycle_diagnostics.hpp"
#include "meshblock.hpp"
#include "meshblock_internal.hpp"
#include "stage_executor.hpp"

namespace snap {

void MeshBlockImpl::to(torch::Device device, torch::Dtype dtype,
                       bool non_blocking) {
  if (policy.fp32())
    torch::nn::Module::to(device, non_blocking);
  else
    torch::nn::Module::to(device, dtype, non_blocking);
  _state.reallocate();  // like the moved u0 and s0
  if (_executor) _executor->invalidate();
}
void MeshBlockImpl::to(torch::Dtype dtype, bool non_blocking) {
  if (policy.fp32()) return;
  torch::nn::Module::to(dtype, non_blocking);
  _state.reallocate();
  if (_executor) _executor->invalidate();
}
void MeshBlockImpl::to(torch::Device device, bool non_blocking) {
  torch::nn::Module::to(device, non_blocking);
  _state.reallocate();
  if (_executor) _executor->invalidate();
}

void set_scalar_primitive(Variables &vars, torch::Tensor const &scalar_s,
                          torch::Tensor const &hydro_u,
                          HydroImpl const &hydro) {
  // the tracers ride the TOTAL dry density (anomaly storage: + ref)
  auto scalar_r = (scalar_s / hydro.dry_density(hydro_u).unsqueeze(0))
                      .to(scalar_s.options());
  auto it = vars.find("scalar_r");
  if (it != vars.end()) {
    it->second.set_data(scalar_r);
  } else {
    vars["scalar_r"] = scalar_r;
  }
}

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const &options_)
    : options(options_) {
  // Output-slice validation below reads the global grid directly, long before
  // pcoord exists.
  options->coord()->resolve_global_grid();

  // two boundary functions per active dimension x1, x2, x3
  auto const &c = *options->coord();
  std::array<int, 3> nc = {c.nc1(), c.nc2(), c.nc3()};
  for (int d = 0; d < 3; ++d)
    if (nc[d] > 1 && options->bfuncs().size() < 2 * (d + 1))
      throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least " +
                               std::to_string(2 * (d + 1)));

  reset();
}

MeshBlockImpl::~MeshBlockImpl() {
  // destroy signal handler
  SignalHandler::Destroy();
}

void MeshBlockImpl::set_user_stage_forcings(
    std::vector<std::string> const &filenames) {
  std::vector<std::shared_ptr<torch::jit::Module>> loaded;
  loaded.reserve(filenames.size());
  auto device = torch::Device(options->device_str());
  for (auto const &filename : filenames) {
    auto module =
        std::make_shared<torch::jit::Module>(torch::jit::load(filename));
    module->to(device);
    module->eval();
    loaded.push_back(std::move(module));
  }
  user_stage_forcings = std::move(loaded);
}

void MeshBlockImpl::reset() {
  //// ---- (1) set up distributed environment ---- ////
  // blocks are built one after another (MeshImpl::reset): no lock
  if (_playout == nullptr)
    _playout = LayoutImpl::create(options->layout(), this);
  send_bufs.resize(_playout->num_exchange_buffers());
  recv_bufs.resize(_playout->num_exchange_buffers());
  exchange_buffer_cache.clear();

  int px = options->layout()->px();
  int py = options->layout()->py();
  int pz = options->layout()->pz();

  int nranks = px * py * pz;
  if (options->layout()->type() == "cubed-sphere") {
    nranks *= 6;
  }
  int rank = options->layout()->rank();

  TORCH_CHECK(options->layout()->world_size() == nranks,
              "MeshBlockImpl: world_size (", options->layout()->world_size(),
              ") does not match layout partitioning (", nranks, ").");

  //// ---- (2) reset internal block boundaries ---- ////
  if (options->layout()->type() != "cubed-sphere") {  // slab or cubed layout
    auto [lx2, lx3, lx1] = _playout->loc_of(rank);

    // a face inside the layout (or periodic) has no external boundary:
    // x1 runs over pz blocks, x2 over px, x3 over py
    auto const &lo = *options->layout();
    std::array<std::array<int, 3>, 3> faces = {{{lx1, pz, lo.periodic_z()},
                                                {lx2, px, lo.periodic_x()},
                                                {lx3, py, lo.periodic_y()}}};
    for (int d = 0; d < 3; ++d) {
      auto [l, n, periodic] = faces[d];
      if (l != 0 || periodic) options->bfuncs()[2 * d] = nullptr;
      if (l != n - 1 || periodic) options->bfuncs()[2 * d + 1] = nullptr;
    }
  }

  //// --------- (3) set up output --------- ////
  for (auto const &out_op : options->outputs()) {
    auto validate_slice = [](std::vector<double> const &slices, double lower,
                             double upper, bool sum, char const *axis) {
      if (!slices.empty() && sum) {
        throw std::invalid_argument(std::string("Cannot request both slice and "
                                                "sum along ") +
                                    axis);
      }
      for (size_t i = 0; i < slices.size(); ++i) {
        double slice = slices[i];
        if (!(slice >= lower && slice < upper)) {
          throw std::invalid_argument(
              std::string("Slice out of mesh bounds: ") + axis);
        }
        if (i && slice <= slices[i - 1]) {
          throw std::invalid_argument(
              std::string("Slices must be strictly increasing: ") + axis);
        }
      }
    };
    validate_slice(out_op->x1_slice(), options->coord()->global_x1min(),
                   options->coord()->global_x1max(), out_op->output_sumx1(),
                   "x1");
    validate_slice(out_op->x2_slice(), options->coord()->global_x2min(),
                   options->coord()->global_x2max(), out_op->output_sumx2(),
                   "x2");
    validate_slice(out_op->x3_slice(), options->coord()->global_x3min(),
                   options->coord()->global_x3max(), out_op->output_sumx3(),
                   "x3");

    if (out_op->file_type() == "restart") {
      output_types.push_back(std::make_shared<RestartOutput>(out_op));
    } else if (out_op->file_type() == "netcdf") {
      output_types.push_back(std::make_shared<NetcdfOutput>(out_op));
    } else if (out_op->file_type() == "pnetcdf") {
#ifdef PNETCDFOUTPUT
      output_types.push_back(std::make_shared<PNetcdfOutput>(out_op));
#else
      throw std::runtime_error("Output type 'pnetcdf' requires PNETCDF=ON.");
#endif
    } else {
      throw std::runtime_error("Output type '" + out_op->file_type() +
                               "' is not implemented.");
    }
  }

  //// -------- (4) set up integrator -------- ////
  pintg = harp::IntegratorImpl::create(options->intg(), this);

  //// ----- (5) set up coordinate model ------ ////
  pcoord = CoordinateImpl::create(options->coord(), this);

  //// -------- (6) set up hydro model -------- ////
  pefs = std::make_shared<EfStatus>();
  phydro = HydroImpl::create(options->hydro(), this);
  policy = PrecisionPolicy(phydro->options->precision());

  //// -------- (7) set up scalar model ------- ////
  pscalar = ScalarImpl::create(options->scalar(), this);

  // an s-point stencil reads s / 2 + 1 ghosts on each axis it runs on (S184)
  auto co = options->coord();
  for (auto [r, nc] : {std::pair{phydro->precon1, co->nc1()},
                       {phydro->precon23, co->nc2() * co->nc3()},
                       {pscalar->precon, co->nc1() * co->nc2() * co->nc3()}})
    TORCH_CHECK(
        !r || nc == 1 || r->pinterp1->stencils() / 2 + 1 <= co->nghost(),
        "[MeshBlock] ", r->pinterp1->options->type(),
        " reconstruction needs nghost >= ", r->pinterp1->stencils() / 2 + 1,
        ", got ", co->nghost());

  //// ------ (8) set up internal boundary ---- ////
  pib = InternalBoundaryImpl::create(options->ib(), this);

  // dimensions
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();

  //// ------- (9) set up the state store and its RK backups ------ ////
  TORCH_CHECK(phydro->peos->nvar() > 0, "Hydro model must have nvar > 0.");
  _state.reset(policy, phydro->peos->nvar(), pscalar->nvar(), nc3, nc2, nc1);
  register_buffer("u0", _state.u0());
  register_buffer("s0", _state.s0());
  register_flux_storage();

  //// ------- (10) stage executor (eager unless cuda-graph) ------- ////
  TORCH_CHECK(
      options->execution() == "eager" || options->execution() == "cuda-graph",
      "integration/execution must be eager or cuda-graph, got '",
      options->execution(), "'");
  TORCH_CHECK(!(options->execution() == "eager" && options->execution_strict()),
              "integration/execution-strict needs execution: cuda-graph "
              "(eager is never refused)");
  _executor = std::make_shared<StageExecutor>(this);
}

std::vector<torch::indexing::TensorIndex> MeshBlockImpl::part(
    std::tuple<int, int, int> offset, PartOptions const &opts) const {
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
  int nghost_coord = options->coord()->nghost();

  int is_ghost = opts.exterior() ? 1 : 0;

  auto [o3, o2, o1] = offset;
  // per dimension x1, x2, x3: the offset (-1 inner, 0 interior, 1 outer),
  // the cell count with ghosts and the interior extension
  std::array<int, 3> off = {o1, o2, o3}, nc = {nc1, nc2, nc3},
                     ext = {opts.extend_x1(), opts.extend_x2(),
                            opts.extend_x3()};
  std::array<torch::indexing::Slice, 3> sl;
  for (int d = 0; d < 3; ++d) {
    int nx = nc[d] > 1 ? nc[d] - 2 * nghost_coord : 1;
    int nghost = nx == 1 ? 0 : nghost_coord;
    int start, len;
    if (off[d] == -1) {
      start = nghost * (1 - is_ghost);
      len = std::min(nghost, opts.depth());
    } else if (off[d] == 0) {
      start = nghost - ext[d];
      len = nx + 2 * ext[d];
    } else {  // 1
      start = nx + nghost * is_ghost;
      len = std::min(nghost, opts.depth());
    }
    sl[d] = torch::indexing::Slice(start, start + len);
  }
  auto slice1 = sl[0], slice2 = sl[1], slice3 = sl[2];
  auto slice4 = torch::indexing::Slice();

  if (opts.ndim() == 1) {
    return {slice1};
  } else if (opts.ndim() == 2) {
    return {slice2, slice1};
  } else if (opts.ndim() == 3) {
    return {slice3, slice2, slice1};
  } else if (opts.ndim() == 4) {
    return {slice4, slice3, slice2, slice1};
  } else {
    throw std::runtime_error("part: ndim must be 1-4.");
  }
}

double MeshBlockImpl::initialize(Variables &vars, char const *restart_file) {
  if (_playout->has_process_group()) {
    _playout->comm->barrier();
  }

  //// ------------ (1) Set up a signal handler ------------ ////
  SignalHandler::GetInstance();

  if (restart_file != nullptr) {
    return _init_from_restart(vars, std::string(restart_file));
  }

  initialize_under_mesh(vars);
  return 0.;  // default start time is 0.0
}

void MeshBlockImpl::initialize_local(Variables &vars) {
  torch::Tensor hydro_w, scalar_r, solid;

  //// ------------ (2) Check hydro primitive ------------ ////
  int64_t nc3 = options->coord()->nc3();
  int64_t nc2 = options->coord()->nc2();
  int64_t nc1 = options->coord()->nc1();

  TORCH_CHECK(vars.count("hydro_w"),
              "initialize: hydro_w is required for hydro model.");
  if (policy.fp32())
    vars["hydro_w"] = policy.cast(Quantity::kPrimitive, vars.at("hydro_w"));
  hydro_w = vars.at("hydro_w");

  TORCH_CHECK(hydro_w.sizes() ==
                  std::vector<int64_t>({phydro->peos->nvar(), nc3, nc2, nc1}),
              "initialize: hydro_w has incorrect shape.", " Expected [",
              phydro->peos->nvar(), ", ", nc3, ", ", nc2, ", ", nc1,
              "] but got ", hydro_w.sizes());

  if (pscalar->nvar() > 0) {
    TORCH_CHECK(vars.count("scalar_r"), "initialize: scalar_r is required");
    scalar_r = vars.at("scalar_r");
    TORCH_CHECK(scalar_r.sizes() ==
                    std::vector<int64_t>({pscalar->nvar(), nc3, nc2, nc1}),
                "initialize: scalar_r has incorrect shape");
  }
  // Capture before any physical face fill, independently of mutable inputs.
  if (has_radiating_boundary()) {
    vars["boundary_reference_w"] = hydro_w.clone();
    if (scalar_r.defined()) vars["boundary_reference_r"] = scalar_r.clone();
  }
  apply_boundaries(vars, hydro_w, scalar_r, true);
}

void MeshBlockImpl::initialize_under_mesh(Variables &vars) {
  initialize_local(vars);
  exchange_state(vars, "hydro_w", kPrimitive, "scalar_r");
  finalize_initialization(vars);
}

void MeshBlockImpl::build_hydro_reference(Variables &vars) {
  // the initial state with the velocities zeroed, in totals; its ghosts
  // come from the same boundary functions as the state, so for the linear
  // fills (reflect, periodic, exchange copies) BC(ref + u') = BC(ref) + BC(u')
  TORCH_CHECK(!phydro->has_reference(),
              "build_hydro_reference: the reference already exists");
  auto w_ref = vars.at("hydro_w").clone();
  w_ref.narrow(0, IVX, 3).zero_();
  auto u_ref = phydro->peos->compute("W->U", {w_ref});
  Variables tmp = vars;  // apply_boundaries reads only the references
  apply_boundaries(tmp, u_ref, torch::Tensor());
  phydro->set_reference(u_ref);
  if (options->verbose()) {
    SINFO(MeshBlock) << "anomaly reference built, shape "
                     << phydro->u_ref().sizes() << std::endl;
  }
}

void MeshBlockImpl::finalize_initialization(Variables &vars) {
  auto hydro_w = vars.at("hydro_w");
  auto scalar_r =
      vars.count("scalar_r") ? vars.at("scalar_r") : torch::Tensor();
  auto solid = vars.count("solid") ? vars.at("solid") : torch::Tensor();
  int64_t nc3 = options->coord()->nc3();
  int64_t nc2 = options->coord()->nc2();
  int64_t nc1 = options->coord()->nc1();

  //// ------ (7) Computer hydro and scalar conserved -------- ////

  vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});
  // anomaly storage (fp32 roadmap Step 1, D1): the static reference is the
  // initial state at rest, built here from totals
  if (phydro->anomaly()) build_hydro_reference(vars);
  if (pscalar->nvar() > 0) {
    // Seed with the conserved (dry) density, as set_scalar_primitive divides.
    vars["scalar_s"] = vars.at("hydro_u")[IDN] * scalar_r;
  }
  // from here on hydro_u holds rows {IDN, IPR} as anomalies
  if (phydro->has_reference()) phydro->sub_reference_(vars.at("hydro_u"));
  if (policy.fp32()) {
    vars["hydro_u"] = policy.store(Quantity::kHydroState, vars.at("hydro_u"));
    if (vars.count("scalar_s"))
      vars["scalar_s"] =
          policy.store(Quantity::kScalarState, vars.at("scalar_s"));
  }

  //// ------------- (8) Fill solid boundaries -------------- ////
  if (vars.count("solid")) {
    solid = vars.at("solid");
    TORCH_CHECK(solid.sizes() == std::vector<int64_t>({nc3, nc2, nc1}),
                "initialize: solid has incorrect shape.", " Expected [", nc3,
                ", ", nc2, ", ", nc1, "] but got ", solid.sizes());
    vars["fill_solid_hydro_w"] =
        torch::where(solid.unsqueeze(0).expand_as(hydro_w), hydro_w, 0.);
    vars["fill_solid_hydro_w"].narrow(0, IVX, 3).zero_();
    pib->mark_prim_solid_(hydro_w, solid);

    vars["fill_solid_hydro_u"] =
        torch::where(solid.unsqueeze(0).expand_as(vars.at("hydro_u")),
                     vars.at("hydro_u"), 0.);
    vars["fill_solid_hydro_u"].narrow(0, IVX, 3).zero_();
  } else {
    vars["fill_solid_hydro_w"] = hydro_w;
    vars["fill_solid_hydro_u"] = vars.at("hydro_u");
  }

  apply_boundaries(
      vars, vars.at("hydro_u"),
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor());
}

std::map<std::string, int64_t> MeshBlockImpl::execution_report() const {
  return _executor->report();
}

void MeshBlockImpl::bump_config_generation() {
  _executor->bump_config_generation();
}

void MeshBlockImpl::register_flux_storage() {
  auto &freg = phydro->flux_register();
  for (int d = 1; d <= 3; ++d) {
    auto name = std::to_string(d);
    if (freg.total(d).defined())
      _state.register_storage("freg.total" + name, freg.total(d));
    if (freg.has_step_field(d))
      _state.register_storage("freg.step" + name, freg.step_field(d));
  }
  auto &mon = freg.monitor();
  for (auto [name, t] :
       {std::pair<char const *, torch::Tensor>{"freg.counters", mon.counters},
        {"freg.minima", mon.minima},
        {"freg.retry_flags", mon.retry_flags}})
    if (t.defined()) _state.register_storage(name, t);
}

std::shared_ptr<kintera::ThermoYImpl> MeshBlockImpl::thermo() const {
  // by name: the EOS types keep their ThermoY private
  auto modules = named_modules();
  auto m = modules.find("hydro.eos.thermo");
  return m ? std::dynamic_pointer_cast<kintera::ThermoYImpl>(*m) : nullptr;
}

}  // namespace snap
