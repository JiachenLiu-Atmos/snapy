// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "internal_boundary.hpp"

namespace snap {

#include <snap_options/internal_boundary_read.inc>

InternalBoundaryOptions InternalBoundaryOptionsImpl::from_yaml(
    std::string const &filename) {
  YAML::Node config = YAML::LoadFile(filename);
  auto op = InternalBoundaryOptionsImpl::create();

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["internal"]) return op;

  return from_yaml(config["boundary-condition"]["internal"]);
}

InternalBoundaryOptions InternalBoundaryOptionsImpl::from_yaml(
    const YAML::Node &node) {
  auto op = InternalBoundaryOptionsImpl::create();
  read_internal_boundary_rows(*op, node, "boundary-condition/internal");
  return op;
}

InternalBoundaryImpl::InternalBoundaryImpl(
    InternalBoundaryOptions const &options_, torch::nn::Module *p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const *>(p);
  reset();
}

void InternalBoundaryImpl::reset() {}

void InternalBoundaryImpl::mark_prim_solid_(torch::Tensor w,
                                            torch::Tensor solid) const {
  if (!solid.defined()) return;

  w[IDN].masked_fill_(solid, options->solid_density());
  w[IPR].masked_fill_(solid, options->solid_pressure());
  w.narrow(0, IVX, 3).masked_fill_(solid, 0.);

  int ny = w.size(0) - ICY;
  if (ny > 0) {
    w.narrow(0, ICY, ny)
        .masked_fill_(solid.unsqueeze(0).expand_as(w.narrow(0, ICY, ny)), 0.);
  }
}

void InternalBoundaryImpl::fill_cons_solid_(torch::Tensor u,
                                            torch::Tensor solid,
                                            torch::Tensor fill) const {
  if (!solid.defined()) return;

  u.set_(torch::where(solid.unsqueeze(0).expand_as(u), fill, u));
}

torch::Tensor InternalBoundaryImpl::forward(torch::Tensor wlr, int dim,
                                            torch::Tensor solid) const {
  if (!solid.defined()) return wlr;

  auto solidl = solid;
  auto solidr = solid.roll(1, dim - 1);
  solidr.select(dim - 1, 0) = solidl.select(dim - 1, 0);

  for (size_t n = 0; n < wlr.size(1); ++n) {
    wlr[IRT][n] = torch::where(solidl, wlr[ILT][n], wlr[IRT][n]);
    wlr[ILT][n] = torch::where(solidr, wlr[IRT][n], wlr[ILT][n]);
  }

  if (dim == 3) {
    wlr[IRT][IVX] = torch::where(solidl, -wlr[ILT][IVX], wlr[IRT][IVX]);
    wlr[ILT][IVX] = torch::where(solidr, -wlr[IRT][IVX], wlr[ILT][IVX]);
  } else if (dim == 2) {
    wlr[IRT][IVY] = torch::where(solidl, -wlr[ILT][IVY], wlr[IRT][IVY]);
    wlr[ILT][IVY] = torch::where(solidr, -wlr[IRT][IVY], wlr[ILT][IVY]);
  } else if (dim == 1) {
    wlr[IRT][IVZ] = torch::where(solidl, -wlr[ILT][IVZ], wlr[IRT][IVZ]);
    wlr[ILT][IVZ] = torch::where(solidr, -wlr[IRT][IVZ], wlr[ILT][IVZ]);
  }

  return wlr;
}
std::shared_ptr<InternalBoundaryImpl> InternalBoundaryImpl::create(
    InternalBoundaryOptions const &opts, torch::nn::Module *p,
    std::string const &name) {
  TORCH_CHECK(p != nullptr, "[InternalBoundary] Parent module is null");
  TORCH_CHECK(opts != nullptr, "[InternalBoundary] Options pointer is null");

  return p->register_module(name, InternalBoundary(opts, p));
}

}  // namespace snap
