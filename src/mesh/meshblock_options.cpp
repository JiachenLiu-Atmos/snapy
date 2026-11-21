// C/C++
#include <array>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/input/check_keys.hpp>
#include <snap/output/output_type.hpp>
#include <snap/utils/log.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptionsImpl::from_yaml(std::string input_file,
                                                 bool verbose) {
  auto op = MeshBlockOptionsImpl::create();

  // -------------- layout -------------- //
  op->layout() = LayoutOptionsImpl::from_yaml(input_file);

  // ------------- basename ------------- //
  // Extract filename from path using utility function, then remove extension
  std::string filename = get_filename(input_file);
  // Remove file extension
  size_t last_dot = filename.find_last_of('.');
  op->basename() =
      (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);

  // -------------- hydro --------------- //
  op->hydro() = HydroOptionsImpl::from_yaml(input_file, verbose);
  if (verbose) op->hydro()->report(SINFO(MeshBlockOptions));

  // ------------- scalar --------------- //
  op->scalar() = ScalarOptionsImpl::from_yaml(input_file, verbose);
  if (op->scalar()->recon() == nullptr) {
    op->scalar()->recon() = op->hydro()->recon23();
  }
  if (verbose) op->scalar()->report(SINFO(MeshBlockOptions));

  // ------------- integrator ------------ //
  op->intg() = harp::IntegratorOptionsImpl::from_yaml(input_file);
  if (verbose) op->intg()->report(SINFO(MeshBlockOptions));

  auto config = YAML::LoadFile(input_file);
  op->verbose() = config["verbose"].as<bool>(verbose);

  // ------------- execution ------------- //
  // integration.execution / execution-strict (V3 X3; provisional names)
  if (config["integration"]) {
    auto intg = config["integration"];
    op->execution() = intg["execution"].as<std::string>("eager");
    op->execution_strict() = intg["execution-strict"].as<bool>(false);
  }
  TORCH_CHECK(op->execution() == "eager" || op->execution() == "cuda-graph",
              "integration/execution must be eager or cuda-graph, got '",
              op->execution(), "'");
  TORCH_CHECK(!(op->execution() == "eager" && op->execution_strict()),
              "integration/execution-strict needs execution: cuda-graph "
              "(eager is never refused)");

  // --------------- outputs ------------- //
  int fid = 0;
  if (config["outputs"]) {
    for (auto const &out_cfg : config["outputs"]) {
      op->outputs().push_back(OutputOptionsImpl::from_yaml(out_cfg, fid++));
    }

    if (verbose) {
      for (auto const &out_op : op->outputs()) {
        out_op->report(SINFO(MeshBlockOptions));
      }
    }
  }

  // ------------- coordinate ------------- //
  op->coord() = CoordinateOptionsImpl::from_yaml(input_file);
  // output slices are checked once, by MeshBlock (validate_slice)
  if (verbose) op->coord()->report(SINFO(MeshBlockOptions));

  // --------- internal boundary ---------- //
  op->ib() = InternalBoundaryOptionsImpl::from_yaml(input_file);
  if (op->ib() && verbose) op->ib()->report(SINFO(MeshBlockOptions));

  // --------- external boundary ---------- //
  if (!config["boundary-condition"]) return op;
  check_keys(config["boundary-condition"], "boundary-condition",
             {"external", "internal"});
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];
  check_keys(
      external_bc, "boundary-condition/external",
      {"x1-inner", "x1-outer", "x2-inner", "x2-outer", "x3-inner", "x3-outer"});

  // two faces per active dimension; an inactive dimension below an active one
  // keeps two null slots. periodic marks the layout: x1 on "cubed" only
  std::array<int, 3> nc = {op->coord()->nc1(), op->coord()->nc2(),
                           op->coord()->nc3()};
  for (int d = 0; d < 3; ++d) {
    if (nc[d] <= 1) {
      bool above =
          (d == 0 && (nc[1] > 1 || nc[2] > 1)) || (d == 1 && nc[2] > 1);
      for (int k = 0; above && k < 2; ++k) {
        op->bfuncs().push_back(nullptr);
        op->bcnames().push_back("");
      }
      continue;
    }
    for (std::string side : {"inner", "outer"}) {
      auto key = "x" + std::to_string(d + 1) + "-" + side;
      auto name = external_bc[key].as<std::string>("reflecting");
      if (name == "periodic") {
        if (d == 0 && op->layout()->type() == "cubed")
          op->layout()->periodic_z(true);
        if (d == 1) op->layout()->periodic_x(true);
        if (d == 2) op->layout()->periodic_y(true);
      }
      name += "_" + side;
      TORCH_CHECK(get_bc_func().find(name) != get_bc_func().end(),
                  "Boundary function '", name, "' is not defined.");
      op->bfuncs().push_back(get_bc_func()[name]);
      op->bcnames().push_back(name);
      if (verbose) {
        SINFO(MeshBlockOptions) << key << " BC: " << name << std::endl;
      }
    }
  }

  if (verbose) op->layout()->report(SINFO(MeshBlockOptions));

  return op;
}

namespace {
// First match wins, in x3/x2/x1 order. is_physical_boundary always did;
// is_wall_boundary used an assignment chain, i.e. the opposite precedence, for
// a multi-axis direction -- which only diffusion calls, and only with
// (0,0,+-1).
int face_of(int dy, int dx, int dz) {
  if (dy == -1) return BoundaryFace::kInnerX3;
  if (dy == 1) return BoundaryFace::kOuterX3;
  if (dx == -1) return BoundaryFace::kInnerX2;
  if (dx == 1) return BoundaryFace::kOuterX2;
  if (dz == -1) return BoundaryFace::kInnerX1;
  if (dz == 1) return BoundaryFace::kOuterX1;
  return BoundaryFace::kUnknown;
}
}  // namespace

bool MeshBlockOptionsImpl::is_physical_boundary(int dy, int dx, int dz) const {
  // bfuncs() can be shorter than the face index: set_bfuncs (and options.bfuncs
  // from python) takes a vector of any length, after construction.
  auto i = static_cast<size_t>(face_of(dy, dx, dz));
  return i < bfuncs().size() && bfuncs()[i] != nullptr;
}

MeshBlockOptionsImpl &MeshBlockOptionsImpl::set_bfuncs(
    std::vector<bcfunc_t> const &funcs) {
  bfuncs() = funcs;
  bcnames().assign(funcs.size(), "");
  return *this;
}

bool MeshBlockOptionsImpl::is_wall_boundary(int dy, int dx, int dz) const {
  auto i = static_cast<size_t>(face_of(dy, dx, dz));
  // A boundary function with no recorded name cannot be classified. That is
  // the reachable silent-disable path -- set_bfuncs pads bcnames with empty
  // strings, so a length mismatch is not what a caller actually hits.
  if (i < bfuncs().size() && bfuncs()[i] != nullptr &&
      (i >= bcnames().size() || bcnames()[i].empty())) {
    TORCH_WARN_ONCE(
        "[MeshBlockOptions] a boundary function is installed with no recorded "
        "name, so it cannot be classified as a wall and diffusion falls back "
        "to "
        "averaging the ghost there. Pass `name` to set_bfunc to declare what "
        "was installed.");
  }
  return face_kind(dy, dx, dz) == FaceKind::wall;
}

FaceKind MeshBlockOptionsImpl::face_kind(int dy, int dx, int dz) const {
  auto i = static_cast<size_t>(face_of(dy, dx, dz));
  if (i >= bfuncs().size() || bfuncs()[i] == nullptr) return FaceKind::interior;
  if (i >= bcnames().size() || bcnames()[i].empty()) return FaceKind::custom;

  // `wall` is a whitelist, deliberately. `reflecting` mirrors the interior,
  // so its ghost is a physical state sitting at the wrong place;
  // `fixed_temperature` engineers its ghost, so it is no state at all. The
  // operator may extrapolate past both. Everything else keeps
  // the two-cell average -- periodic's ghost is a true neighbour, outflow's
  // zero-gradient ghost IS that condition's statement about the face, custom is
  // written by user code this cannot interpret, and `solid` writes a bare 1
  // into every variable, which makes the GRADIENT nonsense too; fixing the
  // coefficient alone would not rescue that face.
  auto const &name = bcnames()[i];
  if (name.compare(0, 8, "periodic") == 0) return FaceKind::periodic;
  if (name == "reflecting_inner" || name == "reflecting_outer" ||
      name == "fixed_temperature_inner" || name == "fixed_temperature_outer")
    return FaceKind::wall;
  if (is_outflow(bfuncs()[i])) return FaceKind::outflow;
  return FaceKind::custom;
}

std::string MeshBlockOptionsImpl::device_str() const {
  if (!layout()) return "";

  if (layout()->device() == "cuda") {
    int device_index = layout()->device_id();
    if (device_index < 0) {
      device_index = layout()->local_rank();
    }
    return "cuda:" + std::to_string(device_index);
  } else if (layout()->device() == "cpu") {
    return "cpu";
  } else {
    return "";
  }
}

}  // namespace snap
