// kintere
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "eos_dispatch.hpp"
#include "ideal_gas.hpp"

namespace snap {

void IdealGasImpl::reset() {}

double IdealGasImpl::species_weight(int n) const { return options->weight(); }

double IdealGasImpl::species_cv_ref(int n) const {
  return gas_constant_dry() / (options->gammad() - 1.);
}

torch::Tensor IdealGasImpl::specific_heat_cv(torch::Tensor prim,
                                             torch::Tensor temp) {
  return torch::full_like(temp, species_cv_ref());
}

torch::Tensor IdealGasImpl::compute(std::string ab,
                                    std::vector<torch::Tensor> const &args) {
  torch::Tensor out;
  if (convert_(ab, args, &out)) {
    return out;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    return w[IPR] / (w[IDN] * gas_constant_dry());
  } else if (ab == "UT->I") {
    auto w = args[0];
    auto temp = args[1];
    return _temp2intEng(w, temp);
  } else if (ab == "W->A") {
    auto w = args[0];
    return options->gammad() * torch::ones_like(w[IDN]);
  } else if (ab == "WA->L") {
    return ideal_sound_speed_(args[0], args[1]);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void IdealGasImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  apply_primitive_limiter_(prim);

  // den -> den
  cons[IDN] = prim[IDN];

  // vel -> mom
  _mom_from_vel_(prim, cons);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons);
}

void IdealGasImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  apply_conserved_limiter_(cons);
  auto pcoord = phydro->pmb->pcoord;

  auto gammad = options->gammad();

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(false)
          .declare_static_shape(prim.sizes(), /*squash_dims=*/0)
          .add_output(prim)
          .add_input(cons)
          .add_owned_input(pcoord->cosine_cell_kj.unsqueeze(0).expand_as(prim))
          .build();

  at::native::ideal_gas_cons2prim(cons.device().type(), iter, gammad);

  apply_primitive_limiter_(prim);
}

torch::Tensor IdealGasImpl::_prim2intEng(torch::Tensor prim) {
  return prim[IPR] / (options->gammad() - 1);
}

torch::Tensor IdealGasImpl::_temp2intEng(torch::Tensor cons,
                                         torch::Tensor temp) {
  return cons[IDN] * species_cv_ref() * temp;
}

}  // namespace snap
