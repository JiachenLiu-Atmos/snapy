// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/input/check_keys.hpp>

#include "scalar.hpp"

namespace snap {

ScalarOptions ScalarOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = ScalarOptionsImpl::create();
  op->verbose() = verbose;

  auto config = YAML::LoadFile(filename);
  auto node = config["scalar"];
  if (!node) {
    return op;
  }

  check_keys(node, "scalar",
             {"verbose", "nvar", "upper-bound", "names", "reconstruct",
              "riemann-solver"});
  op->verbose() = node["verbose"].as<bool>(verbose);
  op->nvar() = node["nvar"].as<int>(op->nvar());
  op->upper_bound() = node["upper-bound"].as<double>(op->upper_bound());
  if (node["names"]) {
    op->names() = node["names"].as<std::vector<std::string>>();
    op->nvar() = op->names().size();
  }

  if (node["reconstruct"]) {
    // no scale key: the scalar always reconstructs scale-invariant (Scalar)
    check_keys(node["reconstruct"], "scalar/reconstruct", {"shock", "type"});
    op->recon() = ReconstructOptionsImpl::from_node(node["reconstruct"],
                                                    "scalar/reconstruct");
  } else {
    op->recon() = nullptr;
  }

  if (node["riemann-solver"]) {
    op->riemann() = RiemannSolverOptionsImpl::from_yaml(
        node["riemann-solver"], "scalar/riemann-solver");
  } else {
    op->riemann() = RiemannSolverOptionsImpl::create();
    op->riemann()->type() = "upwind";
  }

  if (op->names().empty()) {
    op->names().reserve(op->nvar());
    for (int n = 0; n < op->nvar(); ++n) {
      op->names().push_back("scalar" + std::to_string(n));
    }
  }

  return op;
}

}  // namespace snap
