// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/scalar/scalar.hpp>

#include "output_type.hpp"

namespace snap {
void OutputType::loadScalarOutputData(MeshBlockImpl* pmb,
                                      Variables const& vars) {
  if (!vars.count("scalar_r") && !vars.count("scalar_s")) {
    return;
  }

  auto const& r =
      vars.count("scalar_r") ? vars.at("scalar_r") : torch::Tensor();
  auto const& s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();
  bool output_all_scalar = ContainVariable("scalar");
  bool output_all_scalar_cons =
      output_all_scalar || ContainVariable("scalar_cons");
  bool output_all_scalar_prim =
      output_all_scalar || ContainVariable("scalar_prim");

  std::string root_name_cons = "s";
  std::string root_name_prim = "r";

  torch::Tensor r_mean, r_std;
  if (r.defined() && OutputsScalarStat()) {
    r_mean = ScalarStatMean(r);
    r_std = ScalarStatStd(r);
  }

  for (int n = 0; n < pmb->pscalar->nvar(); n++) {
    std::string scalar_name_cons, scalar_name_prim;
    auto const& names = pmb->pscalar->options->names();
    if (n < names.size() && !names[n].empty()) {
      scalar_name_cons = root_name_cons + "_" + names[n];
      scalar_name_prim = root_name_prim + "_" + names[n];
    } else {
      scalar_name_cons = root_name_cons + std::to_string(n);
      scalar_name_prim = root_name_prim + std::to_string(n);
    }

    if (s.defined() &&
        (output_all_scalar_cons || ContainVariable(scalar_name_cons))) {
      appendTensorSliceOutput(scalar_name_cons, s, n, 1);
    }

    if (r.defined() &&
        (output_all_scalar_prim || ContainVariable(scalar_name_prim))) {
      appendTensorSliceOutput(scalar_name_prim, r, n, 1);
    }

    if (r_mean.defined()) {
      appendTensorSliceOutput(scalar_name_prim + "_mean", r_mean, n, 1);
      appendTensorSliceOutput(scalar_name_prim + "_std", r_std, n, 1);
    }
  }
}
}  // namespace snap
