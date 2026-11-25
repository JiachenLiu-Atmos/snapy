// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>

#include "recon_dispatch.hpp"
#include "reconstruct.hpp"
#include "x1_rows.hpp"

namespace snap {

Scheme parse_scheme(std::string const& type) {
  if (type == "dc") return Scheme::dc;
  if (type == "plm") return Scheme::plm;
  if (type == "ppm") return Scheme::ppm;
  if (type == "cp3") return Scheme::cp3;
  if (type == "weno3") return Scheme::weno3;
  if (type == "cp5") return Scheme::cp5;
  if (type == "weno5") return Scheme::weno5;
  TORCH_CHECK(false, "Interp: unknown type " + type);
}

Scheme velocity_pressure_scheme(Scheme s) {
  return s == Scheme::weno3   ? Scheme::cp3
         : s == Scheme::weno5 ? Scheme::cp5
                              : s;
}

Interp InterpImpl::create(InterpOptions const& opts, Scheme s,
                          torch::nn::Module* p, std::string const& name) {
  TORCH_CHECK(opts, "Interp options is null");
  TORCH_CHECK(p, "Parent module is null");
  return p->register_module(name, std::make_shared<InterpImpl>(opts, s));
}

InterpImpl::InterpImpl(InterpOptions const& options_)
    : InterpImpl(options_, parse_scheme(options_->type())) {}

InterpImpl::InterpImpl(InterpOptions const& options_, Scheme s)
    : options(options_), scheme(s) {
  reset();
}

void InterpImpl::reset() {
  switch (scheme) {
    case Scheme::cp3:
      cm = register_buffer(
          "cm", torch::tensor({1. / 3., 5. / 6., -1. / 6.}, torch::kFloat64));
      cp = register_buffer("cp", cm.flip({0}));
      break;
    case Scheme::weno3:
      cm = register_buffer("cm", torch::tensor({{1. / 2., 1. / 2., 0.},
                                                {0., 3. / 2., -1. / 2.},
                                                {1., -1., 0.},
                                                {0., 1., -1.}},
                                               torch::kFloat64));
      cp = register_buffer("cp", cm.flip({1}));
      break;
    case Scheme::cp5:
      cm = register_buffer("cm", torch::tensor({-1. / 20., 9. / 20., 47. / 60.,
                                                -13. / 60., 1. / 30.},
                                               torch::kFloat64));
      cp = register_buffer("cp", cm.flip({0}));
      x1_reset_tables(*this, cm_x1, cp_x1, kPoly5NuCoeff);
      break;
    case Scheme::weno5: {
      // nine stencil rows, then the linear weights d (V3 C9: the uniform
      // table holds d like a per-face row, so FP32 reads them in FP32)
      auto rows = torch::tensor({{-1. / 6., 5. / 6., 1. / 3., 0., 0.},
                                 {0., 1. / 3., 5. / 6., -1. / 6., 0.},
                                 {0., 0., 11. / 6., -7. / 6., 1. / 3.},
                                 {1., -2., 1., 0., 0.},
                                 {1., -4., 3., 0., 0.},
                                 {0., 1., -2., 1., 0.},
                                 {0., -1., 0., 1., 0.},
                                 {0., 0., 1., -2., 1.},
                                 {0., 0., 3., -4., 1.}},
                                torch::kFloat64);
      auto d = torch::tensor({.3, .6, .1}, torch::kFloat64);
      cm = register_buffer("cm", torch::cat({rows.flatten(), d}));
      cp = register_buffer("cp", torch::cat({rows.flip({1}).flatten(), d}));
      x1_reset_tables(*this, cm_x1, cp_x1, kWeno5NuCoeff);
      break;
    }
    default:
      break;
  }
}

int InterpImpl::stencils() const {
  using S = Scheme;  // indexed in Scheme's order
  constexpr int n[] = {
      SchemeTraits<S::dc>::stencil,    SchemeTraits<S::plm>::stencil,
      SchemeTraits<S::ppm>::stencil,   SchemeTraits<S::cp3>::stencil,
      SchemeTraits<S::weno3>::stencil, SchemeTraits<S::cp5>::stencil,
      SchemeTraits<S::weno5>::stencil};
  return n[static_cast<int>(scheme)];
}

void InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  if (scheme == Scheme::weno5)
    return x1_set_coeffs(*this, x1f, &weno5_x1_coeffs);
  if (scheme == Scheme::cp5) return x1_set_coeffs(*this, x1f, &cp5_x1_coeffs);
  TORCH_CHECK(false, "[", options->type(),
              "] has no stretched-x1 reconstruction: route A builds "
              "per-face coefficients for weno5 and cp5 only, and running ",
              options->type(),
              " here would silently apply uniform-grid arithmetic to a "
              "non-uniform x1.  A deck that declares a stretched x1 "
              "(cells/x1rat or geometry/x1f) must reconstruct x1 with "
              "weno5 or cp5 (note Sec. 4.7)");
}

std::pair<torch::Tensor, torch::Tensor> InterpImpl::forward(
    torch::Tensor w, int dim, torch::optional<torch::Tensor> wl,
    torch::optional<torch::Tensor> wr) {
  auto vec = w.sizes().vec();
  vec[dim] -= stencils() - 1;  // reduce size by stencils - 1

  if (scheme == Scheme::ppm) {
    throw std::runtime_error("PPM interpolation is not implemented yet.");
  }

  auto wlv = wl.value_or(torch::empty(vec, w.options()));
  auto wrv = wr.value_or(torch::empty(vec, w.options()));

  if (scheme == Scheme::plm) {
    auto size = w.size(dim);
    auto dw = w.narrow(dim, 1, size - 1) - w.narrow(dim, 0, size - 1);
    auto dwl = dw.narrow(dim, 0, size - 2);
    auto dwr = dw.narrow(dim, 1, size - 2);
    auto dw2 = dwl * dwr;
    // dw2 > 0 keeps dwl + dwr away from zero; an epsilon cannot (0/0 at 1e-38)
    auto dwm =
        torch::where(dw2 > 0, 2. * dw2 / (dwl + dwr), torch::zeros_like(dw2));

    wlv.copy_(w.narrow(dim, 1, size - 2) - 0.5 * dwm);
    wrv.copy_(w.narrow(dim, 1, size - 2) + 0.5 * dwm);
    return {wlv, wrv};
  }
  left(w, dim, wlv);
  right(w, dim, wrv);
  return {wlv, wrv};
}

void InterpImpl::left(torch::Tensor w, int dim, torch::Tensor const& out) {
  if (scheme == Scheme::dc)
    out.copy_(w.slice(dim, 0, w.size(dim)));
  else if (scheme == Scheme::plm || scheme == Scheme::ppm)
    forward(w, dim, out, torch::nullopt);
  else
    apply(w, dim, out, cm, cm_cast, cm_x1, cm_x1_cast);
}

void InterpImpl::right(torch::Tensor w, int dim, torch::Tensor const& out) {
  if (scheme == Scheme::dc)
    out.copy_(w.slice(dim, 0, w.size(dim) - 1));
  else if (scheme == Scheme::plm || scheme == Scheme::ppm)
    forward(w, dim, torch::nullopt, out);
  else
    apply(w, dim, out, cp, cp_cast, cp_x1, cp_x1_cast);
}

void InterpImpl::apply(torch::Tensor const& w, int dim,
                       torch::Tensor const& out, torch::Tensor const& table,
                       TensorCastCache& cache, torch::Tensor const& rows,
                       TensorCastCache& rows_cache) {
  int s = static_cast<int>(scheme);
  bool scale = options->scale();
  if (has_x1_coeffs()) {
    x1_forward_nu(w, dim, out, rows, rows_cache,
                  scheme == Scheme::weno5 ? "weno5" : "cp5",
                  [&](at::TensorIterator& iter, torch::Tensor const& c) {
                    at::native::call_recon(out.device().type(), iter, c, dim, s,
                                           scale, c.size(1));
                  });
    return;
  }
  auto iter = x1_uniform_iterator(w, dim, out);
  at::native::call_recon(out.device().type(), iter, cache.get(table, w), dim, s,
                         scale, /*row_stride=*/0);
}

ReconstructOptions ReconstructOptionsImpl::from_yaml(
    std::string const& filename, std::string const& section) {
  auto op = ReconstructOptionsImpl::create();
  auto config = YAML::LoadFile(filename);
  if (!config["dynamics"] || !config["dynamics"]["reconstruct"]) return op;
  return from_yaml(config["dynamics"]["reconstruct"], section);
}

ReconstructOptions ReconstructOptionsImpl::from_yaml(
    const YAML::Node& node, std::string const& section) {
  check_keys(node, "dynamics/reconstruct", {"vertical", "horizontal"});
  if (!node[section]) return ReconstructOptionsImpl::create();
  return from_node(node[section], "dynamics/reconstruct/" + section);
}

ReconstructOptions ReconstructOptionsImpl::from_node(
    const YAML::Node& node, std::string const& path,
    std::vector<std::string> const& keys) {
  auto op = ReconstructOptionsImpl::create();
  check_keys(node, path, keys);
  op->shock() = node["shock"].as<bool>(op->shock());
  op->interp()->type() = node["type"].as<std::string>(op->interp()->type());
  op->interp()->scale() = node["scale"].as<bool>(op->interp()->scale());
  return op;
}

// faces il..iu+1 of cells il..iu; the ghost faces copy the end values
// (index diagram: lane-c-notes.md)
void _apply_inplace(int dim, int il, int iu, const torch::Tensor& w,
                    Interp& pinterp, torch::Tensor wlr) {
  if (il > iu) return;

  auto outl = wlr[IRT].slice(dim, il - 1, iu + 1);
  auto outr = wlr[ILT].slice(dim, il, iu + 2);

  pinterp->forward(w, dim, outl, outr);
  for (int s : {(int)IRT, (int)ILT}) {  // populate dummy regions
    wlr[s].slice(dim, 0, il) = wlr[s].select(dim, il).unsqueeze(dim);
    wlr[s].slice(dim, iu + 1) = wlr[s].select(dim, iu).unsqueeze(dim);
  }
}

ReconstructImpl::ReconstructImpl(const ReconstructOptions& options_,
                                 torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ReconstructImpl::reset() {
  // the group table: interp1 reconstructs density and the species,
  // interp2 velocity and pressure (weno5 with cp5, weno3 with cp3)
  auto s = parse_scheme(options->interp()->type());
  pinterp1 = InterpImpl::create(options->interp(), s, this, "interp1");
  pinterp2 = InterpImpl::create(options->interp(), velocity_pressure_scheme(s),
                                this, "interp2");
}

void ReconstructImpl::set_x1_faces(torch::Tensor const& x1f) {
  pinterp1->set_x1_coeffs(x1f);
  pinterp2->set_x1_coeffs(x1f);
}

torch::Tensor ReconstructImpl::forward(torch::Tensor w, int dim, bool floor) {
  auto vec = w.sizes().vec();
  vec.insert(vec.begin(), 2);
  auto result = torch::zeros(vec, w.options());
  int nghost = pinterp1->stencils() / 2 + 1;
  int il = nghost;
  int iu = w.size(dim) - nghost;
  int nvar = w.size(0);
  TORCH_CHECK(il <= iu, "il > iu");

  if (phydro == nullptr) {
    _apply_inplace(dim, il, iu, w, pinterp1, result);
    result.clamp_min_(0.);
    return result;
  }

  // FaceFloor's counts: the faces a floor changes, on the computed faces;
  // a floor that runs leaves its mask, which the sweep posts and the monitor
  // reduces with the stage's other masks (no zero-filled slots)
  floored.clear();
  auto faces = [&](torch::Tensor const& x) {
    // face iu+1 exists only when a ghost cell follows (as the slices of
    // _apply_inplace, which stop at the end of the axis)
    return x.narrow(dim, il, std::min<int64_t>(iu + 2, x.size(dim)) - il);
  };

  // the species faces are floored at 0 with the eos limiter. clamp_min keeps
  // -0., so non-negative faces are unchanged bit for bit
  auto floor_species = [&](bool on) {
    if (!on || nvar <= ICY) return;
    auto y = result.narrow(1, ICY, nvar - ICY);
    floored.emplace_back(2, faces((y < 0.).any(1)));
    y.clamp_min_(0.);
  };

  // density and pressure are floored with the eos limiter, on both paths
  auto eos = phydro->options->eos();
  bool floors = eos->limiter() && floor;
  auto floor_density = [&] {
    if (!floors) return;
    auto d = result.select(1, IDN);
    floored.emplace_back(0, faces(d < eos->density_floor()));
    d.clamp_min_(eos->density_floor());
  };
  auto floor_pressure = [&] {
    if (!floors || result.size(1) <= IPR) return;
    auto pr = result.select(1, IPR);
    floored.emplace_back(1, faces(pr < eos->pressure_floor()));
    pr.clamp_min_(eos->pressure_floor());
  };

  if (options->shock()) {
    _apply_inplace(dim, il, iu, w, pinterp1, result);
    // the floors of the non-shock path below (V3 C10); for the species, a
    // WENO undershoot would otherwise carry vapor up the gradient
    floor_density();
    floor_pressure();
    floor_species(eos->limiter());
    return result;
  }

  // density
  _apply_inplace(dim, il, iu, w.narrow(0, IDN, 1), pinterp1,
                 result.narrow(1, IDN, 1));
  floor_density();

  // velocity/pressure
  int len = std::min((int)IPR, nvar - 1);
  _apply_inplace(dim, il, iu, w.narrow(0, IVX, len), pinterp2,
                 result.narrow(1, IVX, len));
  floor_pressure();

  int ny = nvar - ICY;
  if (ny <= 0) return result;

  // others
  _apply_inplace(dim, il, iu, w.narrow(0, ICY, ny), pinterp1,
                 result.narrow(1, ICY, ny));
  floor_species(eos->limiter());

  return result;
}

std::shared_ptr<ReconstructImpl> ReconstructImpl::create(
    ReconstructOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p, "[Reconstruct] Parent module is null");
  TORCH_CHECK(opts, "[Reconstruct] Options pointer is null");
  return p->register_module(name, Reconstruct(opts, p));
}

}  // namespace snap
