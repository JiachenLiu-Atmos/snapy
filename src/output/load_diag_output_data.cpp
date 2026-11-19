// C/C++
#include <type_traits>
#include <vector>

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "output_utils.hpp"

namespace snap {

torch::Tensor interp_to_face(torch::Tensor value, int dim) {
  auto face = torch::empty_like(value);
  int n = value.size(dim);
  if (n <= 1) {
    face.copy_(value);
    return face;
  }

  auto full = torch::indexing::Slice();
  std::vector<torch::indexing::TensorIndex> dst(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> lo(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> hi(value.dim(), full);

  dst[dim] = 0;
  lo[dim] = 0;
  face.index_put_(dst, value.index(lo));

  dst[dim] = torch::indexing::Slice(1, n);
  lo[dim] = torch::indexing::Slice(0, n - 1);
  hi[dim] = torch::indexing::Slice(1, n);
  face.index_put_(dst, 0.5 * (value.index(lo) + value.index(hi)));

  return face;
}

void OutputType::loadDiagOutputData(MeshBlockImpl* pmb, Variables const& vars) {
  auto peos = pmb->phydro->peos;
  auto pcoord = pmb->pcoord;

  auto modules = pmb->named_modules(std::string(), /*include_self=*/false);
  if (ContainVariable("thermo") && pmb->phydro->options->eos()->thermo() &&
      modules.contains("hydro.eos.thermo")) {
    auto const& w = vars.at("hydro_w");

    auto m = modules["hydro.eos.thermo"];
    auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);
    kintera::ThermoX thermo_x(thermo_y->options);
    thermo_x->to(w.device());

    auto const species_count = thermo_y->options->species().size();
    int ny = species_count > 1 ? static_cast<int>(species_count - 1) : 0;
    auto temp = peos->compute("W->T", {w});
    auto dens = w[IDN];
    auto pres = w[IPR];
    torch::Tensor xfrac;
    if (ny > 0) {
      auto yfrac = w.narrow(0, ICY, ny);
      xfrac = thermo_y->compute("Y->X", {yfrac});
    } else {
      // Dry single-species cases have no tracer state to convert.
      xfrac = torch::ones_like(temp).unsqueeze(-1);
    }

    // mole concentration [mol/m^3]
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

    // volumetric entropy [J/(m^3 K)]
    auto entropy_vol = thermo_x->compute("TPV->S", {temp, pres, conc});

    // volumetric heat capacity [J/(m^3 K)]
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});

    // molar entropy [J/(mol K)]
    auto entropy_mole = entropy_vol / conc.sum(-1);

    // molar heat capacity [J/(mol K)]
    auto cp_mole = cp_vol / conc.sum(-1);

    // mean molecular weight [kg/mol]
    auto mu = (thermo_x->mu * xfrac).sum(-1);

    // specific entropy [J/(kg K)]
    auto entropy = entropy_mole / mu;

    // potential temperature [K]
    auto theta = (entropy_vol / cp_vol).exp();

    // virtual potential temperature [K]
    auto Rd = kintera::constants::Rgas / thermo_x->mu[0];
    auto feps = pres / (dens * Rd * temp);
    auto theta_v = theta * feps;

    // relative humidity
    auto nucleation = thermo_x->options->nucleation();
    using Reactions =
        std::remove_reference_t<decltype(nucleation->reactions())>;
    Reactions reactions = nucleation ? nucleation->reactions() : Reactions{};
    torch::Tensor rh;
    if (!reactions.empty()) {
      rh = kintera::relative_humidity(temp, conc, thermo_x->stoich, nucleation);
    }

    appendTensorOutput("temp", temp);
    appendTensorOutput("theta", theta);
    appendTensorOutput("theta_v", theta_v);
    appendTensorOutput("entropy", entropy);

    // relative humidity
    for (int i = 0; i < reactions.size(); ++i) {
      auto name = fmt::format("rh_{}", reactions[i].products().begin()->first);
      // replace special characters '(' ')' and ',' with '_'
      for (char& c : name) {
        if (c == '(' || c == ')' || c == ',') {
          c = '_';
        }
      }
      appendTensorOutput(name, rh.select(-1, i));
    }
  }

  if (ContainVariable("diagnostics") || ContainVariable("div") ||
      ContainVariable("div_h") || ContainVariable("curl")) {
    auto const& w = vars.at("hydro_w");

    if (ContainVariable("diagnostics") || ContainVariable("div") ||
        ContainVariable("div_h")) {
      torch::Tensor v1f;
      torch::Tensor v2f;
      torch::Tensor v3f;
      if (pcoord->options->nx1() > 1) {
        auto w1f = interp_to_face(w, 3);
        pcoord->prim2local1_(w1f);
        v1f = w1f[IVX].unsqueeze(0);
      }
      if (pcoord->options->nx2() > 1) {
        auto w2f = interp_to_face(w, 2);
        pcoord->prim2local2_(w2f);
        v2f = w2f[IVY].unsqueeze(0);
      }
      if (pcoord->options->nx3() > 1) {
        auto w3f = interp_to_face(w, 1);
        pcoord->prim2local3_(w3f);
        v3f = w3f[IVZ].unsqueeze(0);
      }

      if (ContainVariable("diagnostics") || ContainVariable("div")) {
        appendTensorOutput("div", pcoord->divergence(v1f, v2f, v3f).squeeze(0));
      }

      if (ContainVariable("diagnostics") || ContainVariable("div_h")) {
        torch::Tensor div_h;
        if (v2f.defined() || v3f.defined()) {
          div_h = pcoord->divergence(torch::Tensor(), v2f, v3f).squeeze(0);
        } else {
          div_h = torch::zeros_like(w[IDN]);
        }
        appendTensorOutput("div_h", div_h);
      }
    }

    if (ContainVariable("diagnostics") || ContainVariable("curl")) {
      auto w_local = w.clone();
      pcoord->prim2local1_(w_local);
      auto curl = pcoord->curl(w_local.narrow(0, IVX, 3));
      if (pcoord->options->nx3() > 1) {
        appendTensorSliceOutput("curl", curl, 0, 3);
      } else {
        appendTensorOutput("curl", curl[VEL3]);
      }
    }
  }

  // implicit correction
  if (ContainVariable("implicit") && pmb->phydro->picorr) {
    auto du = pmb->phydro->picorr->correction();

    appendTensorSliceOutput("ic_dry", du, IDN, 1);
    appendTensorSliceOutput("ic_mom", du, IVX, 3);
    appendTensorSliceOutput("ic_etot", du, IPR, 1);
    // vapor + cloud
    auto ny = peos->nvar() - ICY;
    if (ny > 0)
      appendTensorSliceOutput(get_hydro_names(pmb, "ic_"), du, ICY, ny);
  }

  // vapor and cloud paths
  auto vol = pcoord->cell_volume();
  if (ContainVariable("path")) {
    auto const& u = vars.at("hydro_u");
    auto area = pcoord->face_area1();
    auto ny = peos->nvar() - ICY;
    int il = pcoord->il();
    int iu = pcoord->iu();

    if (ny > 0) {
      auto u_sum = (u * vol).narrow(0, ICY, ny).slice(-1, il, iu + 1).sum(-1) /
                   area.select(-1, il);
      appendTensorOutput(get_hydro_names(pmb, "path_"), u_sum);
    }
  }

  // zonal mean profiles
  if (ContainVariable("avg")) {
    auto layout = pmb->get_layout();
    c10d::ReduceOptions opsum;
    opsum.reduceOp = c10d::ReduceOp::SUM;
    opsum.rootRank = layout->options->process_root_rank();

    auto hydro_w_tol = vars.at("hydro_w") * vol;
    std::vector<at::Tensor> sum1 = {hydro_w_tol.sum({1, 2})};
    if (layout->has_process_group()) {
      layout->comm->reduce(sum1, opsum.reduceOp, opsum.rootRank);
    }

    std::vector<at::Tensor> sum2 = {vol.unsqueeze(0).sum({1, 2})};
    if (layout->has_process_group()) {
      layout->comm->reduce(sum2, opsum.reduceOp, opsum.rootRank);
    }
    auto avg_w = sum1[0] / sum2[0];

    appendTensorSliceOutput("avg_rho", avg_w, IDN, 1);
    appendTensorSliceOutput("avg_vel", avg_w, IVX, 3);
    appendTensorSliceOutput("avg_press", avg_w, IPR, 1);
    auto ny = peos->nvar() - ICY;
    if (ny > 0)
      appendTensorOutput(get_hydro_names(pmb, "avg_"),
                         avg_w.narrow(0, ICY, ny));
  }
}
}  // namespace snap
