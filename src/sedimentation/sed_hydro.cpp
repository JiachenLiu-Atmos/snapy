// snap
#include <snap/snap.h>

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
  auto f64 = [](auto const& v) { return torch::tensor(v, torch::kFloat64); };
  radius = register_buffer("radius", f64(sv->radius()));
  density = register_buffer("density", f64(sv->density()));
  const_vsed = register_buffer("const_vsed", f64(sv->const_vsed()));
  vsed = register_buffer("vsed", torch::empty({0}, torch::kFloat64));
  settling = register_buffer("settling", torch::empty({0}, torch::kFloat64));
  hydro_ids = register_buffer(
      "hydro_ids", torch::tensor(options->hydro_ids(), torch::kLong));
  args_ = SedKernelArgs();  // rebuilt from the new buffers on first use
}

torch::Tensor SedHydroImpl::forward(torch::Tensor wr,
                                    torch::optional<torch::Tensor> out) {
  // contiguous, whatever wr's strides (zeros_like keeps a dense layout)
  auto flux = out.value_or(torch::zeros(wr.sizes(), wr.options()));
  if (!settles()) return flux;

  // the one kernel (D8) for every EOS: temperature and species energy from it
  auto pmb = phydro->pmb;
  auto const& so = options->sedvel();
  SedParams p{
      pmb->options->is_physical_boundary(0, 0, -1) ? pmb->pcoord->il() : -1,
      pmb->options->is_physical_boundary(0, 0, 1) ? pmb->pcoord->iu()
                                                  : int(wr.size(3)),
      phydro->options->grav()->grav1(),
      so->a_diameter(),
      so->a_epsilon_LJ(),
      so->a_mass(),
      so->upper_limit()};
  wr = wr.contiguous();
  auto temp = phydro->peos->compute("W->T", {wr}).to(wr.options()).contiguous();
  auto en = phydro->peos->compute("W->E", {wr}).to(wr.options()).contiguous();
  auto& a = kernel_args(wr);
  // fixed storage, the kernel writes it in place; sized (eager) once per shape
  std::vector<int64_t> vs{hydro_ids.size(0), wr.size(1), wr.size(2),
                          wr.size(3)};
  if (vsed.sizes() != vs || vsed.dtype() != wr.dtype() ||
      vsed.device() != wr.device())
    vsed.set_data(torch::empty(vs, wr.options()));
  vsed.zero_();  // the kernel adds into it
  TORCH_CHECK(flux.is_contiguous(), "[SedHydro] the flux must be contiguous");
  sedimentation_flux(wr, temp, en, flux, vsed, a.cosine_cell_kj, a.radius,
                     a.density, a.const_vsed, a.hydro_ids, p);
  return flux;
}

SedKernelArgs& SedHydroImpl::kernel_args(torch::Tensor const& like) {
  auto& a = args_;
  // a cached tensor can be the module buffer itself (.to on its own dtype and
  // device) and move with the block, while a view of it does not: check all
  auto here = [&](torch::Tensor const& t, torch::Dtype dtype) {
    return t.defined() && t.device() == like.device() &&
           t.scalar_type() == dtype;
  };
  auto dt = like.scalar_type();
  if (here(a.radius, dt) && here(a.density, dt) && here(a.const_vsed, dt) &&
      here(a.cosine_cell_kj, dt) && here(a.hydro_ids, torch::kLong))
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
