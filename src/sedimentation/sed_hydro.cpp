// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/flux_register.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "sed_hydro_dispatch.hpp"
#include "sedimentation.hpp"

namespace snap {

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

  auto const& sv = options->sedvel();
  TORCH_CHECK(sv->radius().size() == sv->density().size(),
              "[SedHydro] radius and density must have the same size");
  radius =
      register_buffer("radius", torch::tensor(sv->radius(), torch::kFloat64));
  density =
      register_buffer("density", torch::tensor(sv->density(), torch::kFloat64));
  const_vsed = register_buffer(
      "const_vsed", torch::tensor(sv->const_vsed(), torch::kFloat64));

  // register buffer
  vsed = register_buffer("vsed", torch::empty({0}, torch::kFloat64));
  settling = register_buffer("settling", torch::empty({0}, torch::kFloat64));
  hydro_ids = register_buffer(
      "hydro_ids", torch::tensor(options->hydro_ids(), torch::kLong));
}

torch::Tensor SedHydroImpl::forward(torch::Tensor wr,
                                    torch::optional<torch::Tensor> out) {
  auto flux = out.value_or(torch::zeros_like(wr));
  if (!settles()) return flux;

  auto pmb = phydro->pmb;
  SedParams p;
  p.il = pmb->options->is_physical_boundary(0, 0, -1) ? pmb->pcoord->il() : -1;
  p.iu = pmb->options->is_physical_boundary(0, 0, 1)
             ? pmb->pcoord->iu()
             : static_cast<int>(wr.size(3));
  p.grav = phydro->options->grav()->grav1();
  auto const& so = options->sedvel();
  p.gas_diameter = so->a_diameter();
  p.gas_epsilon_lj = so->a_epsilon_LJ();
  p.gas_mass = so->a_mass();
  p.upper_limit = so->upper_limit();

  // the one kernel (D8) for every EOS: temperature and species energy from it
  wr = wr.contiguous();
  auto temp = phydro->peos->compute("W->T", {wr}).to(wr.options()).contiguous();
  auto en = phydro->peos->compute("W->E", {wr}).to(wr.options()).contiguous();
  auto& a = kernel_args(wr);
  vsed.set_data(torch::empty(
      {hydro_ids.size(0), wr.size(1), wr.size(2), wr.size(3)}, wr.options()));
  TORCH_CHECK(flux.is_contiguous(), "[SedHydro] the flux must be contiguous");
  sedimentation_flux(wr, temp, en, flux, vsed, a.cosine_cell_kj, a.radius,
                     a.density, a.const_vsed, a.hydro_ids, p);
  return flux;
}

SedKernelArgs& SedHydroImpl::kernel_args(torch::Tensor const& like) {
  auto& a = args_;
  if (a.radius.defined() && a.radius.device() == like.device() &&
      a.radius.scalar_type() == like.scalar_type())
    return a;
  auto o = like.options();
  a.radius = radius.to(o).contiguous();
  a.density = density.to(o).contiguous();
  a.const_vsed = const_vsed.to(o).contiguous();
  a.hydro_ids = hydro_ids.to(like.device(), torch::kLong).contiguous();
  auto c = phydro->pmb->pcoord->cosine_cell_kj.to(o);
  while (c.dim() > 2) {
    TORCH_CHECK(c.size(-1) == 1,
                "SedHydro kernel expects singleton trailing coordinate "
                "dimensions, got cosine_cell_kj shape ",
                c.sizes());
    c = c.select(-1, 0);
  }
  if (c.dim() < 2) c = c.view({1, c.dim() ? c.size(0) : 1});
  a.cosine_cell_kj = c.expand({like.size(1), like.size(2)}).contiguous();
  return a;
}

bool SedHydroImpl::settles() const {
  return phydro->options->grav()->grav1() != 0. &&
         options->sedvel()->species().size() > 0;
}

torch::Tensor const& SedHydroImpl::post(FluxRegister& reg, torch::Tensor wr) {
  auto t = reg.total(1);
  if (settling.sizes() != t.sizes() || settling.dtype() != t.dtype() ||
      settling.device() != t.device())
    settling.set_data(torch::empty(t.sizes(), t.options()));
  // -0. so that an entry the flux leaves alone adds nothing: x + (-0.) == x
  settling.fill_(-0.);
  forward(wr, settling);
  // one add per run of consecutive rows, so no row is copied
  std::vector<int> rows = options->hydro_ids();
  rows.insert(rows.end(), {IVX, IVY, IVZ, IPR});
  std::sort(rows.begin(), rows.end());
  for (size_t a = 0, b = 1; a < rows.size(); a = b++) {
    while (b < rows.size() && rows[b] == rows[b - 1] + 1) ++b;
    reg.add(1, "sedimentation", {rows.begin() + a, rows.begin() + b},
            settling.narrow(0, rows[a], b - a));
  }
  return settling;
}

std::shared_ptr<SedHydroImpl> SedHydroImpl::create(SedHydroOptions const& opts,
                                                   torch::nn::Module* p,
                                                   std::string const& name) {
  TORCH_CHECK(opts != nullptr, "SedHydro options is nullptr");
  TORCH_CHECK(p != nullptr, "Parent module is nullptr");
  return p->register_module(name, SedHydro(opts, p));
}

}  // namespace snap
