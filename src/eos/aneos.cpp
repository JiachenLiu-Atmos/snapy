// snap
#include "aneos.hpp"

#include <snap/snap.h>

namespace snap {

void ANEOSImpl::reset() {
  pthermo = ANEOSThermoImpl::create(options->eos_file(), this);
}

torch::Tensor ANEOSImpl::compute(std::string ab,
                                 std::vector<torch::Tensor> const &args) {
  torch::Tensor out;
  if (convert_(ab, args, &out)) {
    return out;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    return _prim2temp(w);
  } else if (ab == "UT->I") {
    auto u = args[0];
    auto temp = args[1];
    return _temp2intEng(u, temp);
  } else if (ab == "W->L") {
    auto w = args[0];
    auto dens = w[IDN];
    auto pres = w[IPR];
    auto [T, U, L] = pthermo->compute("DP->TUL", {dens, pres});
    return L;
  } else if (ab == "WL->A") {
    auto w = args[0];
    auto cs = args[1];
    auto pres = w[IPR];
    auto dens = w[IDN];
    return cs * cs * dens / pres;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void ANEOSImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  apply_primitive_limiter_(prim);

  // den -> den
  cons[IDN] = prim[IDN];

  // vel -> mom
  auto out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons);
}

void ANEOSImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  apply_conserved_limiter_(cons);

  // den -> den
  prim[IDN] = cons[IDN];

  // mom -> vel
  auto out = prim.narrow(0, IVX, 3);
  torch::div_out(out, cons.narrow(0, IVX, 3), prim[IDN]);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
  auto ie = cons[IPR] - ke;

  auto [P, T, L] = pthermo->compute("DU->PTL", {cons[IDN], ie});
  prim[IPR] = P;

  //! cache temperature
  pthermo->cache["temp"] = T;

  apply_primitive_limiter_(prim);
}

torch::Tensor ANEOSImpl::_prim2intEng(torch::Tensor prim) {
  auto [T, U, L] = pthermo->compute("DP->TUL", {prim[IDN], prim[IPR]});
  pthermo->cache["temp"] = T;
  return U;
}

torch::Tensor ANEOSImpl::_prim2temp(torch::Tensor prim) {
  auto [T, U, L] = pthermo->compute("DP->TUL", {prim[IDN], prim[IPR]});
  pthermo->cache["temp"] = T;
  return T;
}

torch::Tensor ANEOSImpl::_temp2intEng(torch::Tensor cons, torch::Tensor temp) {
  auto [P, U, L] = pthermo->compute("DT->PUL", {cons[IDN], temp});
  return U;
}

}  // namespace snap
