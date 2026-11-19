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
  if (verbose) {
    SINFO(MeshBlockOptions) << "basename = " << op->basename() << std::endl;
  }

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
  for (auto const &out_op : op->outputs()) {
    auto check_slice = [](std::vector<double> const &slices, double lower,
                          double upper, char const *axis) {
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
    check_slice(out_op->x1_slice(), op->coord()->global_x1min(),
                op->coord()->global_x1max(), "x1");
    check_slice(out_op->x2_slice(), op->coord()->global_x2min(),
                op->coord()->global_x2max(), "x2");
    check_slice(out_op->x3_slice(), op->coord()->global_x3min(),
                op->coord()->global_x3max(), "x3");
  }
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

  if (op->coord()->nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    if (ix1 == "periodic" && op->layout()->type() == "cubed") {
      op->layout()->periodic_z(true);
    }

    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix1]);
    op->bcnames().push_back(ix1);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x1-inner BC: " << ix1 << std::endl;
    }

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    if (ox1 == "periodic" && op->layout()->type() == "cubed") {
      op->layout()->periodic_z(true);
    }

    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox1]);
    op->bcnames().push_back(ox1);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x1-outer BC: " << ox1 << std::endl;
    }
  } else if (op->coord()->nc2() > 1 || op->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
    op->bcnames().push_back("");
    op->bcnames().push_back("");
  }

  if (op->coord()->nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    if (ix2 == "periodic") op->layout()->periodic_x(true);

    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix2]);
    op->bcnames().push_back(ix2);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x2-inner BC: " << ix2 << std::endl;
    }

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    if (ox2 == "periodic") op->layout()->periodic_x(true);

    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox2]);
    op->bcnames().push_back(ox2);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x2-outer BC: " << ox2 << std::endl;
    }
  } else if (op->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
    op->bcnames().push_back("");
    op->bcnames().push_back("");
  }

  if (op->coord()->nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    if (ix3 == "periodic") op->layout()->periodic_y(true);

    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix3]);
    op->bcnames().push_back(ix3);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x3-inner BC: " << ix3 << std::endl;
    }

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    if (ox3 == "periodic") op->layout()->periodic_y(true);

    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox3]);
    op->bcnames().push_back(ox3);

    if (verbose) {
      SINFO(MeshBlockOptions) << "x3-outer BC: " << ox3 << std::endl;
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
