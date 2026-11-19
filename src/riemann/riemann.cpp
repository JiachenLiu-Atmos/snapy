// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../eos/ideal_moist.hpp"
#include "riemann.hpp"

namespace snap {
void to_face_frame_(CoordinateImpl const& coord, int dim,
                    torch::Tensor const& wl, torch::Tensor const& wr) {
  switch (dim) {
    case 1:
      coord.prim2local3_(wl);
      coord.prim2local3_(wr);
      break;
    case 2:
      coord.prim2local2_(wl);
      coord.prim2local2_(wr);
      break;
    case 3:
      coord.prim2local1_(wl);
      coord.prim2local1_(wr);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }
}

void to_global_frame_(CoordinateImpl const& coord, int dim,
                      torch::Tensor const& flx) {
  switch (dim) {
    case 1:
      coord.flux2global3_(flx);
      break;
    case 2:
      coord.flux2global2_(flx);
      break;
    case 3:
      coord.flux2global1_(flx);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }
}

RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
    std::string const& filename, std::string const& section) {
  auto op = RiemannSolverOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  if (!config[section]) return op;
  if (!config[section]["riemann-solver"]) return op;
  check_keys(config["dynamics"]["riemann-solver"], "dynamics/riemann-solver",
             {"type", "dir"});
  return from_yaml(config["dynamics"]["riemann-solver"]);
}

RiemannSolverOptions RiemannSolverOptionsImpl::from_yaml(
    YAML::Node const& node) {
  auto op = RiemannSolverOptionsImpl::create();

  // without the key: the type of create() (hllc), as for a deck with no
  // riemann-solver block
  op->type() = node["type"].as<std::string>(op->type());
  op->dir() = node["dir"].as<std::string>(op->dir());

  return op;
}

RiemannKind parse_riemann_kind(std::string const& type) {
  if (type == "hllc") return RiemannKind::hllc;
  if (type == "lmars") return RiemannKind::lmars;
  if (type == "roe") return RiemannKind::roe;
  if (type == "shallow-roe") return RiemannKind::shallow_roe;
  if (type == "upwind") return RiemannKind::upwind;
  if (type == "plume-roe") return RiemannKind::plume_roe;
  TORCH_CHECK(false, "RiemannSolver: unknown type " + type);
}

RiemannSolverImpl::RiemannSolverImpl(const RiemannSolverOptions& options_,
                                     torch::nn::Module* p)
    : RiemannSolverImpl(options_, parse_riemann_kind(options_->type()), p) {}

RiemannSolverImpl::RiemannSolverImpl(const RiemannSolverOptions& options_,
                                     RiemannKind kind_, torch::nn::Module* p)
    : options(options_), kind(kind_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void RiemannSolverImpl::reset() {
  if (kind == RiemannKind::upwind) return;
  TORCH_CHECK(phydro, "[RiemannSolver] ", options->type(),
              ": parent Hydro is null");
  if (kind == RiemannKind::hllc || kind == RiemannKind::lmars) {
    auto pcoord = phydro->pmb->pcoord;

    // register buffers
    auto nc1 = pcoord->options->nc1();
    auto nc2 = pcoord->options->nc2();
    auto nc3 = pcoord->options->nc3();

    elr = register_buffer("elr",
                          torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
    clr = register_buffer("clr",
                          torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
    glr = register_buffer("glr",
                          torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
  } else if (kind == RiemannKind::shallow_roe) {
    auto pcoord = phydro->pmb->pcoord;
    if (pcoord->options->type() == "gnomonic_equiangle") {
      TORCH_CHECK(options->dir() == "yz",
                  "ShallowRoeSolver with GnomonicEquiangle coordinate "
                  "only supports options->dir() = 'yz' but got "
                  "options->dir() = ",
                  options->dir());
    }
  }
}

//! calculate flux for plume equations
/*!
 * w = [R, W, B, V]
 *
 * F1 = (1 - 1/e) * R^2 * W
 * F2 = 1/2 * R^2 * W * W - 1/2 * R^3 * V - 1/8 * R^2 * V^2
 * F3 = 1/2 * R^2 * B * W
 * F4 = 1/4 * R^3 * V * W + 1/2 * R^4 * W
 */
static void plume_flux(torch::Tensor& flx, torch::Tensor const& prim) {
  auto R = prim[0];
  auto W = prim[1];
  auto B = prim[2];
  auto V = prim[3];

  flx[0] = (1.0 - 1.0 / std::exp(1.0)) * R * R * W;
  flx[1] = 0.5 * R * R * W * V - 0.5 * R * R * R * V - 0.125 * R * R * V * V;
  flx[2] = 0.5 * R * R * B * W;
  flx[3] = 0.25 * R * R * R * V * W + 0.5 * R * R * R * R * W;
}

static torch::Tensor lax_friedrichs_flux(torch::Tensor const& priml,
                                         torch::Tensor const& primr,
                                         torch::Tensor const& csl,
                                         torch::Tensor const& csr) {
  auto flxl = torch::zeros_like(priml);
  auto flxr = torch::zeros_like(primr);

  plume_flux(flxl, priml);
  plume_flux(flxr, primr);

  auto alpha = torch::max(csl, csr);

  return 0.5 * (flxl + flxr) - 0.5 * alpha * (primr - priml);
}

torch::Tensor RiemannSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                         int dim, torch::Tensor vel_or_flux,
                                         torch::Tensor face_pressure) {
  if (kind == RiemannKind::upwind) {
    TORCH_CHECK(
        !face_pressure.defined(),
        "Face-pressure output is not implemented for Riemann solver type ",
        options->type());
    auto ui = (vel_or_flux > 0).to(torch::kInt);
    return vel_or_flux * (ui * wl + (1 - ui) * wr);
  }

  if (kind == RiemannKind::plume_roe) {
    TORCH_CHECK(!face_pressure.defined(),
                "Face-pressure output is not implemented for PlumeRoeSolver");
    TORCH_CHECK(dim == 1,
                "PlumeRoeSolver only supports dim = 1, but got dim = ", dim);
    auto peos = phydro->peos;

    // Approximate sound speeds
    auto csl = peos->compute("W->L", {wl});
    auto csr = peos->compute("W->L", {wr});

    // Lax-Friedrichs flux
    return lax_friedrichs_flux(wl, wr, csl, csr);
  }

  return solve(wl, wr, dim, vel_or_flux, face_pressure, torch::Tensor());
}

torch::Tensor RiemannSolverImpl::forward_perturbations(
    torch::Tensor wl, torch::Tensor wr, int dim, torch::Tensor flx,
    torch::Tensor reference_pressure, torch::Tensor face_pressure_delta) {
  TORCH_CHECK(kind == RiemannKind::hllc,
              "Riemann solver perturbations need hllc, not ", options->type());
  TORCH_CHECK(reference_pressure.defined(),
              "HLLC perturbations require a shared face pressure background");
  return solve(wl, wr, dim, flx, face_pressure_delta, reference_pressure);
}

torch::Tensor RiemannSolverImpl::solve(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor flx,
                                       torch::Tensor face_pressure,
                                       torch::Tensor reference_pressure) {
  auto pcoord = phydro->pmb->pcoord;
  auto peos = phydro->peos;
  bool aneos = peos->options->type() == "aneos";
  RiemannParams params;
  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;

  // FaceThermo: one EOS face pass, (2, ...) e, gamma and c, in each kind's
  // order of the EOS calls
  torch::Tensor e, gamma, c;
  if (kind == RiemannKind::hllc) {
    for (auto* buffer : {&elr, &clr, &glr}) {
      if (buffer->scalar_type() != wl.scalar_type() ||
          buffer->device() != wl.device())
        buffer->set_data(torch::empty(buffer->sizes(), wl.options()));
    }
    // the perturbation entry: the EOS sees the totals
    auto wl_total = wl, wr_total = wr;
    if (reference_pressure.defined()) {
      reference_pressure =
          reference_pressure.to(wl.options()).expand_as(wl[IPR]);
      wl_total = wl.clone();
      wr_total = wr.clone();
      wl_total[IPR] += reference_pressure;
      wr_total[IPR] += reference_pressure;
    }
    for (auto [side, w] :
         {std::pair{ILT, wl_total}, std::pair{IRT, wr_total}}) {
      elr[side] = peos->compute("W->I", {w});
      if (aneos) {
        clr[side] = peos->compute("W->L", {w});
        glr[side] = peos->compute("WL->A", {w, clr[side]});
      } else {
        glr[side] = peos->compute("W->A", {w});
        clr[side] = peos->compute("WA->L", {w, glr[side]});
      }
    }
    e = elr, gamma = glr, c = clr;
  } else if (kind == RiemannKind::lmars) {
    for (auto [side, w] : {std::pair{ILT, wl}, std::pair{IRT, wr}}) {
      elr[side] = peos->compute("W->I", {w}) / w[IDN];
      if (aneos) {
        clr[side] = peos->compute("W->L", {w});
        glr[side] = peos->compute("WL->A", {w, clr[side]});
      } else {
        glr[side] = peos->compute("W->A", {w});
      }
    }
    e = elr, gamma = glr;
  } else if (kind == RiemannKind::roe) {
    params.ideal_moist = peos->options->type() == "ideal-moist";
    if (params.ideal_moist) {
      auto moist = dynamic_cast<IdealMoistImpl*>(peos.get());
      TORCH_CHECK(moist != nullptr, "[RoeSolver] ideal-moist EOS cast failed");
      params.nvapor = moist->pthermo->options->vapor_ids().size() - 1;
      inv_mu_ratio_m1 = moist->inv_mu_ratio_m1.to(wl);
      cv_ratio_m1 = moist->cv_ratio_m1.to(wl);
      u0 = moist->u0.to(wl);
    }
    params.gammad = peos->options->gammad();

    auto eil = peos->compute("W->I", {wl});
    auto eir = peos->compute("W->I", {wr});
    auto gammal = peos->compute("W->A", {wl});
    auto gammar = peos->compute("W->A", {wr});
    auto cl = peos->compute("WA->L", {wl, gammal});
    auto cr = peos->compute("WA->L", {wr, gammar});
    e = torch::stack({eil, eir});
    gamma = torch::stack({gammal, gammar});
    c = torch::stack({cl, cr});
  } else {  // shallow-roe
    TORCH_CHECK(!face_pressure.defined(),
                "Face-pressure output is not implemented for ShallowRoeSolver");
    TORCH_CHECK(options->dir() == "xy" || options->dir() == "yz",
                "ShallowRoeSolver takes options->dir() = 'xy' or 'yz'"
                " but got options->dir() = ",
                options->dir());
    params.dir_yz = options->dir() == "yz";
  }

  // solve in the face-local orthonormal frame
  to_face_frame_(*pcoord, dim, wl, wr);

  at::TensorIteratorConfig config;
  config.resize_outputs(false)
      .check_all_same_dtype(kind != RiemannKind::hllc)
      .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
      .add_output(flx);
  torch::Tensor face_pressure_out;
  if (kind != RiemannKind::shallow_roe) {
    // hllc: a face pressure of another dtype is written through a copy
    bool direct = face_pressure.defined() &&
                  (kind != RiemannKind::hllc ||
                   face_pressure.scalar_type() == wl.scalar_type());
    face_pressure_out = direct ? face_pressure : torch::empty_like(wl[IDN]);
    config.add_owned_output(face_pressure_out.unsqueeze(0));
  }
  config.add_input(wl).add_input(wr);
  for (auto const& t : {e, gamma, c})
    if (t.defined()) config.add_input(t);
  if (reference_pressure.defined())
    config.add_owned_input(reference_pressure.unsqueeze(0));
  auto iter = config.build();

  at::native::call_riemann(flx.device().type(), iter, static_cast<int>(kind),
                           dim, params, inv_mu_ratio_m1, cv_ratio_m1, u0);
  if (face_pressure.defined() && !face_pressure.is_same(face_pressure_out))
    face_pressure.copy_(face_pressure_out);

  to_global_frame_(*pcoord, dim, flx);

  return flx;
}

RiemannSolver RiemannSolverImpl::create(RiemannSolverOptions const& opts,
                                        torch::nn::Module* p,
                                        std::string const& name) {
  TORCH_CHECK(p, "[RiemannSolver] parent module is nullptr");
  TORCH_CHECK(opts, "[RiemannSolver] options pointer is nullptr");
  parse_riemann_kind(opts->type());
  return p->register_module(name, std::make_shared<RiemannSolverImpl>(opts, p));
}

}  // namespace snap
