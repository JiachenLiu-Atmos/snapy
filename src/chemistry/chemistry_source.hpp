#pragma once

// C/C++
#include <map>
#include <memory>
#include <string>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/thermo/thermo.hpp>

namespace snap {

class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! V3 lane K, S-K4 (K.rst rev 3.1 section 3.6, row S-K4): the chemistry step
//! after the last stage as one snapy-owned op over kintera's Kinetics,
//! ThermoX and evolve_implicit. One adapter per block, built on the calling
//! thread from a by-value copy of the reaction options (member-wise: the
//! thermo vectors the first Kinetics build rewrites are the copy's, the
//! reaction sub-options stay shared) and the block's thermo options;
//! afterwards read-only. forward() is the pin step of
//! examples/run_hydro.cpp:171-198: FP64 rates, Jacobian and one implicit
//! step; only the species rows of hydro_u change (by del_rho), the energy
//! row does not, so the heat of reaction appears through the composition.
//! An empty stoichiometry is a no-op (run_hydro.cpp:177).
class ChemistrySource {
 public:
  ChemistrySource(kintera::KineticsOptionsImpl reactions,
                  kintera::ThermoOptions thermo);

  //! the species increment of one step of length dt, applied to
  //! vars["hydro_u"] in place after vars["hydro_w"] is refreshed from the
  //! final dynamics state (the EOS cons2prim, which may apply EF's limiter:
  //! that write is EF's, not this op's)
  void forward(MeshBlockImpl& block, Variables& vars, double dt);

  bool empty() const { return kinet->stoich.numel() == 0; }

  kintera::Kinetics kinet = nullptr;
  kintera::ThermoX thermo_x = nullptr;

 private:
  kintera::KineticsOptions _reactions;  // the by-value copy, read-only
};

}  // namespace snap
