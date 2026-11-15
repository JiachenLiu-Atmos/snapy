// kintera
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../eos/ideal_moist.hpp"
#include "sed_hydro_dispatch.hpp"
#include "sedimentation.hpp"

namespace snap {
namespace {

torch::Tensor sedimentation_flux_tensor(SedHydroImpl& sed, torch::Tensor wr,
                                        torch::Tensor flux, int il, int iu) {
  auto pcoord = sed.phydro->pmb->pcoord;
  auto peos = sed.phydro->peos;
  auto vel = wr.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, pcoord->cosine_cell_kj);

  auto temp = peos->compute("W->T", {wr});
  auto vsed = sed.psedvel->forward(wr[IDN], wr[IPR], temp);

  // ICY is the number of hydro variables
  auto en = peos->compute("W->E", {wr}).index_select(0, sed.hydro_ids - ICY);

  auto rhos = wr[IDN] * wr.index_select(0, sed.hydro_ids);
  auto rhos_vsed = rhos * vsed;

  // each particle is carried from its upwind cell; walls are sealed
  sed.vsed.set_data(sedimentation_upwind(vsed, vsed, il, iu));
  flux.index_add_(0, sed.hydro_ids,
                  sedimentation_upwind(rhos_vsed, vsed, il, iu));
  flux.narrow(0, IVX, 3) +=
      sedimentation_upwind(vel.unsqueeze(1) * rhos_vsed, vsed, il, iu).sum(1);
  flux[IPR] += sedimentation_upwind(vsed * en, vsed, il, iu).sum(0);

  return flux;
}

}  // namespace

SedHydroImpl::SedHydroImpl(SedHydroOptions const& options_,
                           torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void SedHydroImpl::reset() {
  TORCH_CHECK(phydro, "[SedHydro] Parent Hydro is null");
  TORCH_CHECK(
      phydro->options->grav(),
      "[SedHydro] sedimentation reads grav1; set forcing/const-gravity");

  psedvel = SedVelImpl::create(options->sedvel(), this);

  // register buffer
  vsed = register_buffer("vsed", torch::empty({0}, torch::kFloat64));
  hydro_ids = register_buffer(
      "hydro_ids", torch::tensor(options->hydro_ids(), torch::kLong));
}

torch::Tensor SedHydroImpl::forward(torch::Tensor wr,
                                    torch::optional<torch::Tensor> out) {
  auto flux = out.value_or(torch::zeros_like(wr));

  // null-op
  if (phydro->options->grav()->grav1() == 0. ||
      options->sedvel()->species().size() == 0) {
    return flux;
  }

  auto pmb = phydro->pmb;
  int il =
      pmb->options->is_physical_boundary(0, 0, -1) ? pmb->pcoord->il() : -1;
  int iu = pmb->options->is_physical_boundary(0, 0, 1)
               ? pmb->pcoord->iu()
               : static_cast<int>(wr.size(3));

  auto ideal_moist = dynamic_cast<IdealMoistImpl const*>(phydro->peos.get());
  if (ideal_moist == nullptr || !wr.is_contiguous() || !flux.is_contiguous()) {
    return sedimentation_flux_tensor(*this, wr, flux, il, iu);
  }

  int ny = ideal_moist->pthermo->options->vapor_ids().size() +
           ideal_moist->pthermo->options->cloud_ids().size() - 1;
  int nvapor = ideal_moist->pthermo->options->vapor_ids().size() - 1;
  vsed.set_data(torch::empty(
      {hydro_ids.size(0), wr.size(1), wr.size(2), wr.size(3)}, wr.options()));

  // the EOS's own dry gas, not kintera's process-global tables
  double mud = ideal_moist->options->weight();
  double gas_constant_dry = kintera::constants::Rgas / mud;
  double cv_dry = ideal_moist->pthermo->options->cref_R()[0] * gas_constant_dry;
  auto cosine_cell_kj = pmb->pcoord->cosine_cell_kj.to(wr.options());
  while (cosine_cell_kj.dim() > 2) {
    TORCH_CHECK(cosine_cell_kj.size(-1) == 1,
                "SedHydro kernel expects singleton trailing coordinate "
                "dimensions, got cosine_cell_kj shape ",
                cosine_cell_kj.sizes());
    cosine_cell_kj = cosine_cell_kj.select(-1, 0);
  }
  if (cosine_cell_kj.dim() == 0) {
    cosine_cell_kj = cosine_cell_kj.view({1, 1});
  } else if (cosine_cell_kj.dim() == 1) {
    cosine_cell_kj = cosine_cell_kj.view({1, cosine_cell_kj.size(0)});
  }
  cosine_cell_kj = cosine_cell_kj.expand({wr.size(1), wr.size(2)});

  sedimentation_flux_dispatch(
      wr, flux, vsed, cosine_cell_kj.contiguous(),
      psedvel->radius.to(wr.options()).contiguous(),
      psedvel->density.to(wr.options()).contiguous(),
      psedvel->const_vsed.to(wr.options()).contiguous(),
      hydro_ids.to(wr.device(), torch::kLong).contiguous(),
      ideal_moist->inv_mu_ratio_m1.to(wr.options()).contiguous(),
      ideal_moist->cv_ratio_m1.to(wr.options()).contiguous(),
      ideal_moist->u0.to(wr.options()).contiguous(), il, iu, ny, nvapor,
      phydro->options->grav()->grav1(), gas_constant_dry, cv_dry,
      psedvel->options->a_diameter(), psedvel->options->a_epsilon_LJ(),
      psedvel->options->a_mass(), psedvel->options->upper_limit());

  return flux;
}

std::shared_ptr<SedHydroImpl> SedHydroImpl::create(SedHydroOptions const& opts,
                                                   torch::nn::Module* p,
                                                   std::string const& name) {
  TORCH_CHECK(opts != nullptr, "SedHydro options is nullptr");
  TORCH_CHECK(p != nullptr, "Parent module is nullptr");
  return p->register_module(name, SedHydro(opts, p));
}

}  // namespace snap
