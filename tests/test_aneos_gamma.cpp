// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// POSIX
#include <unistd.h>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/eos/aneos/aneos_thermo.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

// A stand-in for the ANEOS table (the library ships weak stubs that throw):
// an ideal gas whose adiabatic index depends on the density, so the index of
// a state tells which state it was computed from. Every "DP->TUL" call (the
// sound speed of "W->L") records the density it was given.
namespace {
constexpr double kR = 287.;
std::vector<torch::Tensor> g_sound_speed_dens;
torch::Tensor gamma_of(torch::Tensor const& rho) {
  return 1.2 + 0.2 * torch::tanh(rho);
}
}  // namespace

namespace snap {
ANEOSThermoImpl::ANEOSThermoImpl(const std::string& fname) : filename(fname) {}
void ANEOSThermoImpl::reset() {}
void ANEOSThermoImpl::pretty_print(std::ostream& os) const {
  os << "test ANEOS stand-in";
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ANEOSThermoImpl::compute(std::string ab,
                         std::vector<torch::Tensor> const& args) {
  auto rho = args[0];
  auto g = gamma_of(rho);
  if (ab == "DP->TUL") {
    g_sound_speed_dens.push_back(rho.clone());
    auto p = args[1];
    return {p / (rho * kR), p / (g - 1.), torch::sqrt(g * p / rho)};
  } else if (ab == "DU->PTL") {
    auto p = (g - 1.) * args[1];
    return {p, p / (rho * kR), torch::sqrt(g * p / rho)};
  } else if (ab == "DT->PUL") {
    auto p = rho * kR * args[1];
    return {p, p / (g - 1.), torch::sqrt(g * p / rho)};
  }
  TORCH_CHECK(false, "test ANEOS stand-in: unknown ", ab);
}
}  // namespace snap

namespace {
// names HydroImpl's protected implicit step from a derived scope
struct ImplicitStep : HydroImpl {
  static auto ptr() { return &ImplicitStep::_apply_implicit_correction; }
};

MeshBlock make_block() {
  char fname[] = "/tmp/test-aneos-gamma-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 8000., x2min: 0., x2max: 1000., x3min: 0., x3max: 1000.}
  cells: {nx1: 16, nx2: 1, nx3: 1, nghost: 3}
dynamics:
  equation-of-state: {type: aneos, eos-file: stand-in}
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver: {type: lmars}
integration: {type: rk3, cfl: 0.5, implicit-scheme: 1}
forcing:
  const-gravity: {grav1: -10.}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic
)";
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  return MeshBlock(options);
}
}  // namespace

// The implicit solve's adiabatic index is that of the state it solves: under
// ANEOS with a solid mask, wi (the solid cells filled), not w (audit 1 row E1)
TEST(AneosGamma, implicit_index_takes_the_solved_state) {
  auto block = make_block();
  auto hydro = block->phydro;
  ASSERT_EQ(hydro->options->eos()->type(), "aneos");
  int nvar = hydro->peos->nvar();
  int nc1 = block->pcoord->options->nc1(), nc2 = block->pcoord->options->nc2(),
      nc3 = block->pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  auto w = torch::zeros({nvar, nc3, nc2, nc1}, opts);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  auto fill = w.clone();
  fill[IDN].fill_(2.);
  auto solid = torch::zeros({nc3, nc2, nc1}, opts.dtype(torch::kBool));
  int il = block->pcoord->il();
  solid.narrow(-1, il, 4).fill_(true);  // the lowest four interior cells
  Variables other{{"solid", solid}, {"fill_solid_hydro_w", fill}};
  auto du = torch::zeros_like(w);

  g_sound_speed_dens.clear();
  (hydro.get()->*ImplicitStep::ptr())(du, w, 1., other, 0.);

  int checked = 0;
  for (auto const& rho : g_sound_speed_dens) {
    if (rho.sizes() != solid.sizes()) continue;
    ++checked;
    EXPECT_TRUE(torch::all(rho.masked_select(solid) == 2.).item<bool>())
        << "the index was computed from w's solid cells (density "
        << rho.masked_select(solid)[0].item<double>()
        << "), not from the filled state the solve uses (density 2)";
  }
  EXPECT_GT(checked, 0) << "no sound speed was taken on the solved state";
}
