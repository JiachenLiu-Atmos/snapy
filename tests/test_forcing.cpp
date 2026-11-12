// C/C++
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

#include "cuda_test_gate.hpp"

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

class DryDensitySourceImpl : public torch::nn::Cloneable<DryDensitySourceImpl> {
 public:
  double increment = 0.;

  DryDensitySourceImpl() = default;
  explicit DryDensitySourceImpl(double increment_) : increment(increment_) {}
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor /*w*/,
                        torch::Tensor /*temp*/, double /*dt*/) {
    du[IDN].select(-1, du.size(-1) / 2).add_(increment);
    return du;
  }
};
TORCH_MODULE(DryDensitySource);

std::shared_ptr<MeshBlockImpl> make_block(
    std::string const& filename = "test_diffusion_moist.yaml") {
  return std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml(filename));
}

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN] = 1.;
  w[IVX] = 1.;
  w[IVY] = 2.;
  w[IVZ] = 3.;
  w[IPR] = 1.e5;
  w[ICY] = 0.1;
  w[ICY + 1] = 0.2;
  return w;
}

std::vector<torch::indexing::TensorIndex> bottom3(
    std::shared_ptr<MeshBlockImpl> const& block) {
  return block->part({0, 0, -1},
                     PartOptions().exterior(false).depth(1).ndim(3));
}

std::vector<torch::indexing::TensorIndex> bottom4(
    std::shared_ptr<MeshBlockImpl> const& block) {
  return block->part({0, 0, -1}, PartOptions().exterior(false).depth(1));
}

void expect_only_bottom(torch::Tensor const& du,
                        std::shared_ptr<MeshBlockImpl> const& block) {
  auto expected = torch::zeros_like(du);
  expected.index_put_(bottom4(block), du.index(bottom4(block)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

std::shared_ptr<MeshBlockImpl> make_cubed_sphere_block(
    int face, std::string const& eos_type,
    std::string const& filename = "test_exchange.yaml") {
  auto options = MeshBlockOptionsImpl::from_yaml(filename);
  options->layout()->rank(face);
  options->layout()->world_size(6);
  options->layout()->blocks_per_process(6);
  options->hydro()->eos()->type() = eos_type;
  return std::make_shared<MeshBlockImpl>(options);
}

torch::Tensor expected_cubed_sphere_coriolis(
    std::shared_ptr<MeshBlockImpl> const& block, torch::Tensor global_velocity,
    torch::Tensor density, std::array<double, 3> const& omega,
    bool traditional) {
  auto coord = block->pcoord;
  auto mesh = torch::meshgrid({coord->x3v, coord->x2v, coord->x1v}, "ij");
  auto face = std::get<2>(
      block->get_layout()->loc_of(block->options->layout()->rank()));
  auto momentum = density.unsqueeze(0) * global_velocity;
  cs_cart_to_contra_(momentum, mesh[1], mesh[0], face);
  cs_contra_to_sph_(momentum, mesh[1], mesh[0], face);

  auto effective_omega = torch::empty_like(global_velocity);
  effective_omega[VEL1].fill_(omega[0]);
  effective_omega[VEL2].fill_(omega[1]);
  effective_omega[VEL3].fill_(omega[2]);
  cs_cart_to_contra_(effective_omega, mesh[1], mesh[0], face);
  cs_contra_to_sph_(effective_omega, mesh[1], mesh[0], face);
  if (traditional) {
    effective_omega[VEL2].zero_();
    effective_omega[VEL3].zero_();
  }

  auto expected = torch::empty_like(global_velocity);
  expected[VEL1] = 2. * (effective_omega[VEL3] * momentum[VEL2] -
                         effective_omega[VEL2] * momentum[VEL3]);
  expected[VEL2] = 2. * (effective_omega[VEL1] * momentum[VEL3] -
                         effective_omega[VEL3] * momentum[VEL1]);
  expected[VEL3] = 2. * (effective_omega[VEL2] * momentum[VEL1] -
                         effective_omega[VEL1] * momentum[VEL2]);
  cs_sph_to_contra_(expected, mesh[1], mesh[0], face);
  coord_vec_lower_(expected, coord->cosine_cell_kj);
  if (traditional) expected[VEL1].zero_();
  return expected;
}

void check_cubed_sphere_coriolis(std::string const& eos_type,
                                 bool configure_traditional,
                                 bool expected_traditional) {
  std::array<double, 3> omega = {0.7, -0.4, 0.2};
  for (int face = 0; face < 6; ++face) {
    auto block = make_cubed_sphere_block(face, eos_type);
    auto coord = block->pcoord;
    auto shape = std::vector<int64_t>{
        3, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()};
    auto global_velocity = torch::empty(shape, torch::kFloat64);
    global_velocity[VEL1].fill_(1.3);
    global_velocity[VEL2].fill_(-2.1);
    global_velocity[VEL3].fill_(0.8);

    auto panel_velocity = global_velocity.clone();
    auto mesh = torch::meshgrid({coord->x3v, coord->x2v, coord->x1v}, "ij");
    cs_cart_to_contra_(panel_velocity, mesh[1], mesh[0], face);

    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.7);
    w.narrow(0, IVX, 3).copy_(panel_velocity);
    if (w.size(0) > IPR) w[IPR].fill_(1.e5);
    auto du = torch::zeros_like(w);
    auto op = CoriolisOptionsImpl::from_yaml(YAML::Load(
        "coriolis: {type: xyz, omega1: 0.7, omega2: -0.4, omega3: 0.2}"));
    op->traditional() = configure_traditional;

    CoriolisXYZ(op, block->phydro.get())->forward(du, w, torch::Tensor(), 1.);

    auto expected = expected_cubed_sphere_coriolis(
        block, global_velocity, w[IDN], omega, expected_traditional);
    EXPECT_TRUE(torch::allclose(du.narrow(0, IVX, 3), expected, 1.e-12, 1.e-12))
        << "face " << face;
    if (expected_traditional) {
      EXPECT_TRUE(torch::equal(du[IVX], torch::zeros_like(du[IVX])))
          << "face " << face;
    }
  }
}

}  // namespace

TEST(forcing_options, reject_invalid_values) {
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 0., species: [vapor], xfrac: [0.1]}")));
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor, vapor], xfrac: "
                 "[0.1, 0.2]}")));
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor], xfrac: [1.1]}")));
  EXPECT_ANY_THROW(RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 0., btemp: 300.}")));
  EXPECT_ANY_THROW(RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 1.}")));
  EXPECT_ANY_THROW(RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 1., btemp: -1.}")));
  EXPECT_ANY_THROW(RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 0.}")));
  EXPECT_ANY_THROW(BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {pmin: 2., pmax: 1.}")));
  EXPECT_ANY_THROW(TopSpongeLyrOptionsImpl::from_yaml(
      YAML::Load("top-sponge-lyr: {width: 1.e3}")));
  EXPECT_ANY_THROW(TopSpongeLyrOptionsImpl::from_yaml(
      YAML::Load("top-sponge-lyr: {tau: 1.e4}")));
  EXPECT_ANY_THROW(BotSpongeLyrOptionsImpl::from_yaml(
      YAML::Load("bot-sponge-lyr: {width: 1.e3}")));
  EXPECT_ANY_THROW(BotSpongeLyrOptionsImpl::from_yaml(
      YAML::Load("bot-sponge-lyr: {tau: 1.e4}")));
}

TEST(forcing_options, reject_unknown_composition_species) {
  auto block = make_block();
  auto op = RelaxBotCompOptionsImpl::from_yaml(YAML::Load(
      "relax-bot-comp: {tau: 1., species: [unknown], xfrac: [0.1]}"));
  EXPECT_ANY_THROW(RelaxBotComp(op, block->phydro.get()));
}

TEST(forcing_options, parse_coriolis_traditional) {
  auto default_op =
      CoriolisOptionsImpl::from_yaml(YAML::Load("coriolis: {type: xyz}"));
  auto traditional_op = CoriolisOptionsImpl::from_yaml(
      YAML::Load("coriolis: {type: xyz, traditional: true}"));

  EXPECT_FALSE(default_op->traditional());
  EXPECT_TRUE(traditional_op->traditional());
}

TEST(forcing, meshblock_registers_parent_dependent_modules) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  options->hydro()->bodyHeat() = BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {dTdt: 1., pmin: 0., pmax: 2.e5}"));
  options->hydro()->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor], xfrac: [0.1]}"));
  options->hydro()->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 1., btemp: 300.}"));
  options->hydro()->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 1.}"));

  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST(forcing, cubed_sphere_coriolis_uses_cartesian_rotation_vector) {
  check_cubed_sphere_coriolis("ideal-gas", false, false);
}

TEST(forcing, cubed_sphere_coriolis_supports_traditional_approximation) {
  check_cubed_sphere_coriolis("ideal-gas", true, true);
}

TEST(forcing, cubed_sphere_shallow_water_uses_traditional_coriolis) {
  check_cubed_sphere_coriolis("shallow-water", false, true);
}

// Coriolis and the u w / r metric take rho w from the x1 mass transfer: with w
// alternating in sign every x1 face (a wall face too) moves no mass, so w must
// exert no torque on the horizontal momenta, whatever the cell-centre w.
TEST(forcing, coriolis_and_metric_take_w_from_the_x1_mass_flux) {
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_forcing_cubed_sphere.yaml");
  options->layout()->type() = "slab";  // one panel, no exchange
  options->hydro()->coriolis() = CoriolisOptionsImpl::from_yaml(YAML::Load(
      "coriolis: {type: xyz, omega1: 0.7e-4, omega2: -0.4e-4, omega3: 1.e-4}"));
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN] = 1.;
  w[IVY] = 30.;
  w[IVZ] = -20.;
  w[IPR] = 1.e5;
  auto run = [&](double w1) {
    w[IVX] = w1 * (1. - 2. * (torch::arange(w.size(-1)) % 2));
    Variables vars;
    vars["hydro_w"] = torch::empty_like(w);
    auto u = block->phydro->peos->compute("W->U", {w});
    return block->phydro->forward(10., u, vars)
        .narrow(0, IVY, 2)
        .slice(-1, coord->il(), coord->iu() + 1);
  };
  double torque = (run(5.) - run(0.)).abs().max().item<double>();
  EXPECT_LT(torque, 1.e-12) << "torque from a w that moves no mass";
  // nx1 = 1, no Coriolis (no x1 mass flux): -rho w v_{2,3} / r at the cell
  options->coord()->nx1() = options->coord()->global_nx1() = 1;
  options->hydro()->coriolis() = nullptr;
  block = std::make_shared<MeshBlockImpl>(options), coord = block->pcoord;
  w = w.narrow(-1, 0, 1).clone();
  auto c = coord->cosine_cell_kj, r = coord->x1v.view({1, 1, -1});
  auto exact = torch::stack({30. - 20. * c, -20. + 30. * c}) * (-10. * 5. / r);
  auto err =
      (run(5.) - run(0.) - exact).slice(-2, coord->jl(), coord->ju() + 1);
  EXPECT_LT(
      err.slice(-3, coord->kl(), coord->ku() + 1).abs().max().item<double>(),
      1.e-15);
}

TEST(forcing, relax_bottom_temperature) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 350.}"));

  RelaxBotTemp(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto bot = bottom3(block);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp);
  expected[IPR].index_put_(
      bot, 0.25 * w[IDN].index(bot) * cv.index(bot) * (350. - temp.index(bot)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, relax_bottom_temperature_at_face_rejects_non_bool) {
  for (char const* bad : {"1", "maybe", "yes", "on", "~", "[]"}) {
    auto yaml =
        std::string("relax-bot-temp: {tau: 2., btemp: 350., at-face: ") + bad +
        "}";
    EXPECT_THROW(RelaxBotTempOptionsImpl::from_yaml(YAML::Load(yaml)),
                 c10::Error)
        << bad;
  }
  auto off = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 350., at-face: false}"));
  EXPECT_FALSE(off->at_face());
  auto bare = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 350.}"));
  EXPECT_FALSE(bare->at_face());
  auto on = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 350., at-face: true}"));
  EXPECT_TRUE(on->at_face());
}

// A bottom inversion (T0 < T1) is a legal state and the extrapolation is
// well-defined there: at-face must relax it by value, not abort. Placed before
// the stratified case so that it is the first at-face call in the process.
TEST(forcing, relax_bottom_temperature_at_face_under_an_inversion) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto x = block->pcoord->x1v.view({1, 1, -1});
  w[IPR] = 1.e5 + 1.e3 * x + 1.e2 * x * x;
  auto temp = block->phydro->peos->compute("W->T", {w});
  int ng = block->pcoord->options->nghost();
  auto T0 = temp.narrow(-1, ng, 1), T1 = temp.narrow(-1, ng + 1, 1);
  ASSERT_LT((T0 - T1).max().item<double>(), 0.) << "need an inversion";
  auto bot = bottom3(block);
  auto rho = w[IDN].index(bot);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp).index(bot);

  auto on = torch::zeros_like(w);
  RelaxBotTemp(RelaxBotTempOptionsImpl::from_yaml(YAML::Load(
                   "relax-bot-temp: {tau: 2., btemp: 350., at-face: true}")),
               block->phydro.get())
      ->forward(on, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  expected[IPR].index_put_(
      bot, 1. / 1.5 * 0.5 / 2. * rho * cv * (350. - (1.5 * T0 - 0.5 * T1)));
  EXPECT_TRUE(torch::allclose(on, expected, 1.e-13, 0.))
      << "at-face tendency " << on[IPR].index(bot) << " expected "
      << expected[IPR].index(bot);
}

// A linear temperature on a stretched grid must extrapolate to its exact
// value at the physical face.
TEST(forcing, relax_bottom_temperature_at_face_uses_coordinate_spacing) {
  auto block = make_block();
  auto coord = block->pcoord;
  int ng = coord->options->nghost();

  // The first two active centres are 0.5 and 2.5 above a face at zero.
  // A linear field T=100+10*x1 therefore has T_face=btemp exactly.
  coord->x1f[ng] = 0.;
  coord->x1v[ng] = 0.5;
  coord->x1v[ng + 1] = 2.5;

  auto w = make_primitive(block);
  auto temp = (100. + 10. * coord->x1v).view({1, 1, -1}).expand_as(w[IDN]);
  auto du = torch::zeros_like(w);
  auto op = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 100., at-face: true}"));

  RelaxBotTemp(op, block->phydro.get())->forward(du, w, temp, 0.5);

  EXPECT_TRUE(torch::equal(du, torch::zeros_like(du)))
      << "linear temperature at the requested face produced forcing "
      << du[IPR].index(bottom3(block));
}

// On a uniform grid, at-face retains the original 1.5/-0.5 weights; the
// default leaves the old cell-centre tendency bit-identical.
TEST(forcing, relax_bottom_temperature_at_face) {
  auto block = make_block();
  auto w = make_primitive(block);
  // quadratic, so no linear-exact face estimate but 1.5*T0 - 0.5*T1 matches
  auto x = block->pcoord->x1v.view({1, 1, -1});
  w[IPR] = 1.e5 - 1.e3 * x - 1.e2 * x * x;
  auto temp = block->phydro->peos->compute("W->T", {w});
  int ng = block->pcoord->options->nghost();
  auto T0 = temp.narrow(-1, ng, 1), T1 = temp.narrow(-1, ng + 1, 1);
  ASSERT_GT((T0 - T1).min().item<double>(), 0.) << "need a stratified column";
  for (auto v : {w, temp}) {  // the ghosts must not be read
    v.narrow(-1, 0, ng).fill_(NAN);
    v.narrow(-1, v.size(-1) - ng, ng).fill_(NAN);
  }
  auto bot = bottom3(block);
  auto rho = w[IDN].index(bot);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp).index(bot);

  auto off = torch::zeros_like(w), on = torch::zeros_like(w);
  RelaxBotTemp(RelaxBotTempOptionsImpl::from_yaml(
                   YAML::Load("relax-bot-temp: {tau: 2., btemp: 350.}")),
               block->phydro.get())
      ->forward(off, w, temp, 0.5);
  RelaxBotTemp(RelaxBotTempOptionsImpl::from_yaml(YAML::Load(
                   "relax-bot-temp: {tau: 2., btemp: 350., at-face: true}")),
               block->phydro.get())
      ->forward(on, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  expected[IPR].index_put_(bot, 0.5 / 2. * rho * cv * (350. - T0));
  EXPECT_TRUE(torch::equal(off, expected));
  expected[IPR].index_put_(
      bot, 1. / 1.5 * 0.5 / 2. * rho * cv * (350. - (1.5 * T0 - 0.5 * T1)));
  EXPECT_TRUE(torch::allclose(on, expected, 1.e-13, 0.))
      << "at-face tendency " << on[IPR].index(bot) << " expected "
      << expected[IPR].index(bot);
}

// The sponge/drag modules build their force from CONTRAVARIANT
// primitive velocities but add it to `du`, which holds COVARIANT momenta. The
// force must therefore be antiparallel to the LOWERED momentum, not to the raw
// velocity. The two coincide wherever cos_theta == 0 -- a panel centre -- which
// is exactly where one would look, and exactly why this went unnoticed.
//
// Design notes, each paid for:
//
//  * The fixture is `test_forcing_cubed_sphere.yaml`, NOT `test_exchange.yaml`.
//    The latter has nx1 == 1, which suppresses the x1 boundary functions
//    entirely, so `is_physical_boundary(0, 0, +-1)` is false and every module
//    below early-returns with `du` identically zero. The first draft of this
//    test used it and passed its covariance check on `0 == 0`.
//
//  * The assertions are over the WHOLE FIELD, not at a single argmax cell. An
//    argmax over `cosine_cell_kj.expand_as(...)` selects x1 index 0 because the
//    tensor is stride-0 along x1 -- a cell where `relax-bot-velo` applies no
//    force at all. A field-wide statement cannot be defeated that way, and
//    `EXPECT_GT(dumax, 0.)` makes vacuity impossible rather than merely
//    detectable.
//
//  * |v2| != |v3| is LOAD-BEARING. The pre-fix cross product is
//    cos_theta * (v3^2 - v2^2); with |v2| == |v3| it vanishes identically and
//    the test would pass against the broken code. Do not "tidy" the fills.
namespace {

// Assert that `du`'s horizontal part is antiparallel to the covariant momentum
// of `w`, and that the pre-fix (contravariant) answer would have differed.
void expect_drag_is_covariant(std::shared_ptr<MeshBlockImpl> const& block,
                              torch::Tensor const& w, torch::Tensor const& du,
                              std::string const& what) {
  auto coord = block->pcoord;
  auto mom = torch::zeros(
      {3, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::kFloat64);
  mom.copy_(w.narrow(0, IVX, 3));
  coord_vec_lower_(mom, coord->cosine_cell_kj);

  double dumax = du.narrow(0, IVX, 3).abs().max().item<double>();
  EXPECT_GT(dumax, 0.) << what << ": no force applied -- the test is vacuous";

  auto d2 = du[IVY], d3 = du[IVZ];
  auto cross = d2 * mom[VEL3] - d3 * mom[VEL2];
  // Scale on the LARGER of the two products: scaling on one alone collapses
  // the tolerance to zero if that product happens to vanish.
  auto scale = torch::maximum((d2 * mom[VEL3]).abs(), (d3 * mom[VEL2]).abs())
                   .max()
                   .item<double>();
  EXPECT_LE(cross.abs().max().item<double>(), 1.e-12 * scale)
      << what << ": drag is not antiparallel to the covariant momentum";

  // and the contravariant direction is genuinely different somewhere, so a
  // regression cannot pass this test by coincidence
  auto alt = d2 * w[IVZ] - d3 * w[IVY];
  EXPECT_GT(alt.abs().max().item<double>(), 1.e-10)
      << what
      << ": contravariant and covariant directions coincide -- this "
         "fixture cannot discriminate the fix from the bug";
}

}  // namespace

TEST(forcing, cubed_sphere_sponge_drag_is_covariant) {
  for (int face = 0; face < 6; ++face) {
    auto block = make_cubed_sphere_block(face, "ideal-gas",
                                         "test_forcing_cubed_sphere.yaml");
    auto coord = block->pcoord;
    ASSERT_TRUE(block->options->is_physical_boundary(0, 0, 1))
        << "face " << face << ": fixture has no outer-x1 boundary";
    ASSERT_TRUE(block->options->is_physical_boundary(0, 0, -1))
        << "face " << face << ": fixture has no inner-x1 boundary";
    ASSERT_GT(coord->cosine_cell_kj.abs().max().item<double>(), 0.)
        << "face " << face << ": grid is orthogonal, nothing to test";

    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.7);
    w[IVX].fill_(0.5);
    w[IVY].fill_(-2.1);  // |v2| != |v3| is load-bearing -- see the note above
    w[IVZ].fill_(0.8);
    if (w.size(0) > IPR) w[IPR].fill_(1.e5);
    auto temp = block->phydro->peos->compute("W->T", {w});

    std::string at = "face " + std::to_string(face);
    {
      auto du = torch::zeros_like(w);
      auto op = TopSpongeLyrOptionsImpl::from_yaml(
          YAML::Load("top-sponge-lyr: {tau: 100.0, width: 1.0e30}"));
      TopSpongeLyr(op, block->phydro.get())->forward(du, w, temp, 1.);
      expect_drag_is_covariant(block, w, du, at + " top-sponge-lyr");
    }
    {
      auto du = torch::zeros_like(w);
      auto op = BotSpongeLyrOptionsImpl::from_yaml(
          YAML::Load("bot-sponge-lyr: {tau: 100.0, width: 1.0e30}"));
      BotSpongeLyr(op, block->phydro.get())->forward(du, w, temp, 1.);
      expect_drag_is_covariant(block, w, du, at + " bot-sponge-lyr");
    }
    {
      // zero reference wind, so the relaxation is a pure drag and the same
      // antiparallel statement applies
      auto du = torch::zeros_like(w);
      auto op = RelaxBotVeloOptionsImpl::from_yaml(
          YAML::Load("relax-bot-velo: {tau: 100., bvx: 0., bvy: 0., bvz: 0.}"));
      RelaxBotVelo(op, block->phydro.get())->forward(du, w, temp, 1.);
      expect_drag_is_covariant(block, w, du, at + " relax-bot-velo");
    }
  }
}

TEST(forcing, relax_bottom_velocity) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 2., bvx: 5., bvy: 6., bvz: 7.}"));

  RelaxBotVelo(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto bot = bottom3(block);
  expected[IVX].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  expected[IVY].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  expected[IVZ].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, relax_bottom_composition_preserves_state) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto u = block->phydro->peos->compute("W->U", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 2., species: [vapor], xfrac: [0.2]}"));
  RelaxBotComp forcing(op, block->phydro.get());

  forcing->forward(du, w, temp, 2.);

  auto updated = block->phydro->peos->compute("U->W", {u + du});
  auto bot = bottom4(block);
  auto bot3 = bottom3(block);
  auto updated_bot = updated.index(bot);
  auto xfrac =
      forcing->pthermo_y->compute("Y->X", {updated_bot.narrow(0, ICY, 2)});
  EXPECT_TRUE(torch::allclose(xfrac.select(-1, 1),
                              torch::full_like(xfrac.select(-1, 1), 0.2)));
  EXPECT_TRUE(torch::allclose(updated_bot[IDN], w.index(bot)[IDN]));
  EXPECT_TRUE(torch::allclose(updated_bot.narrow(0, IVX, 3),
                              w.index(bot).narrow(0, IVX, 3)));
  EXPECT_TRUE(torch::allclose(
      block->phydro->peos->compute("W->T", {updated}).index(bot3),
      temp.index(bot3), 1.e-8, 1.e-8));
  expect_only_bottom(du, block);
}

TEST(forcing, implicit_correction_reports_total_energy_delta) {
  auto make_hydro_block = [](bool implicit) {
    auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
    options->hydro()->diffusion() = nullptr;

    auto gravity = ConstGravityOptionsImpl::create();
    gravity->grav1(-1.);
    gravity->gravity_work("face");  // the form these tests pin
    options->hydro()->grav() = gravity;

    if (implicit) {
      auto icorr = ImplicitOptionsImpl::create();
      icorr->scheme(1);
      options->hydro()->icorr() = icorr;
    } else {
      options->hydro()->icorr() = nullptr;
    }
    return std::make_shared<MeshBlockImpl>(options);
  };

  auto explicit_block = make_hydro_block(false);
  auto implicit_block = make_hydro_block(true);
  auto w = make_primitive(explicit_block);

  auto run = [&](std::shared_ptr<MeshBlockImpl> const& block) {
    auto u = block->phydro->peos->compute("W->U", {w});
    Variables vars;
    vars["hydro_w"] = torch::empty_like(w);
    return block->phydro->forward(0.1, u, vars);
  };

  auto explicit_du = run(explicit_block);
  auto implicit_du = run(implicit_block);
  auto expected = implicit_du - explicit_du;
  auto correction = implicit_block->phydro->picorr->correction();

  EXPECT_TRUE(torch::allclose(correction, expected, 1.e-10, 1.e-10));
  EXPECT_GT(implicit_block->phydro->peos->internal_energy_offset(correction)
                .abs()
                .max()
                .item<double>(),
            0.);
}

// The corrected potential-energy work gravity-work: face books for a density
// change drho of the interior x1 cells [is, ie) of a cartesian deck,
// grav1 <(x1 - x1v)^2> d(drho)/dx1 (gravity_work_radial.hpp). E + P, not
// E + PE_d, is what face work conserves, so the energy checks below add it.
static torch::Tensor booked_pe_work(Coordinate const& coord,
                                    torch::Tensor const& drho, double grav1) {
  int is = coord->il(), ie = coord->iu() + 1;
  return corrected_pe_work(drho.slice(-1, is, ie), coord->x1f, coord->x1v, is,
                           ie, grav1, X1Measure::plain);
}

// One hydro forward with the VIC on: {column energy residual
// sum V*(dE + phi*dmass) + dt*boundary flux, energy tendency, dry clamp fired}.
std::tuple<double, torch::Tensor, bool> implicit_gravity_energy(
    double pres, double top_rho, double dt, double x1shift = 0.) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_gravity_energy.yaml");
  auto oc = options->coord();
  oc->global_x1min(oc->global_x1min() + x1shift);
  oc->global_x1max(oc->global_x1max() + x1shift);
  oc->x1min(oc->x1min() + x1shift);
  oc->x1max(oc->x1max() + x1shift);

  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;
  auto icorr = ImplicitOptionsImpl::create();
  icorr->scheme(1);
  options->hydro()->icorr() = icorr;

  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN] = 1. + 0.05 * (coord->x1v - x1shift);
  if (top_rho > 0.) w[IDN].select(-1, coord->iu()).fill_(top_rho);
  w[IVX].zero_();
  w[IPR].fill_(pres);

  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  auto du = block->phydro->forward(dt, vars.at("hydro_u"), vars);

  int is = coord->il();
  int ie = coord->iu() + 1;
  int ny = du.size(0) - ICY;
  auto mass_du = du[IDN].clone();
  auto mass_flux = block->phydro->flux1()[IDN].clone();
  if (ny > 0) {
    mass_du += du.narrow(0, ICY, ny).sum(0);
    mass_flux += block->phydro->flux1().narrow(0, ICY, ny).sum(0);
  }
  auto phi_cell = -gravity->grav1() * coord->x1v;
  auto phi_face = -gravity->grav1() * coord->x1f;
  auto mechanical_du = du[IPR] + phi_cell * mass_du;
  mechanical_du.slice(-1, is, ie) -=
      booked_pe_work(coord, mass_du, gravity->grav1());
  auto mechanical_flux = block->phydro->flux1()[IPR] +
                         phi_face.narrow(0, 0, mass_flux.size(-1)) * mass_flux;
  auto integrated_du =
      (mechanical_du.slice(-1, is, ie) * coord->cell_volume().slice(-1, is, ie))
          .sum();
  auto boundary_flux =
      (coord->face_area1().select(-1, ie) * mechanical_flux.select(-1, ie) -
       coord->face_area1().select(-1, is) * mechanical_flux.select(-1, is))
          .sum();
  return {(integrated_du + dt * boundary_flux).item<double>(),
          du[IPR].slice(-1, is, ie).clone(),
          block->phydro->picorr->dry_clamp_step().item<double>() > 0.};
}

TEST(forcing, implicit_gravity_work_uses_redistributed_mass) {
  auto [residual, energy, clamped] = implicit_gravity_energy(1.e5, 0., 0.1);
  EXPECT_FALSE(clamped);
  EXPECT_NEAR(residual, 0., 1.e-9);
}

// A clamp breaks drho*V == M(i) - M(i+1); the work must not pick up the
// absolute potential, so a constant shift of x1 leaves every cell unchanged.
TEST(forcing, implicit_gravity_work_ignores_the_potential_origin_under_clamp) {
  auto [r0, e0, clamped0] = implicit_gravity_energy(10., 1.e-3, 1.);
  auto [r1, e1, clamped1] = implicit_gravity_energy(10., 1.e-3, 1., 1.e3);
  ASSERT_TRUE(clamped0 && clamped1);
  EXPECT_LT((e1 - e0).abs().max().item<double>(),
            1.e-9 * e0.abs().max().item<double>());
}

// Below the temperature floor the RK average must report a limiter patch; at
// 350 K it must not.
TEST(forcing, limiter_patch_is_reported_below_the_temperature_floor) {
  for (double pres : {1.e5, 1.e3}) {
    auto options = MeshBlockOptionsImpl::from_yaml("test_gravity_energy.yaml");
    options->hydro()->eos()->limiter(true);
    auto block = std::make_shared<MeshBlockImpl>(options);
    auto coord = block->pcoord;
    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(pres);

    Variables vars;
    vars["hydro_w"] = w;
    block->initialize(vars);
    vars["hydro_u"][IPR].fill_(pres /
                               0.4);  // at rest, gamma = 1.4: 3.5 K / 350 K
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, 1.e-3, stage);
    }
    EXPECT_EQ(block->limiter_patch_hit(), pres < 1.e4) << "pres=" << pres;
  }
}

namespace {
struct LimiterStep {
  int redo;         // check_redo after the planted step
  std::string log;  // what that check_redo printed
  int retry;        // check_redo after one more, unplanted step
};

// One step of a limiter-on block at rest (350 K) with `value` written into
// row `row` of one interior cell of hydro_u just before stage `at` (row < 0:
// nothing planted), or, with `prim`, into a W->U call's primitive input.
LimiterStep limiter_step(std::string const& yaml, int row, double value, int at,
                         bool prim = false, torch::Device device = torch::kCPU,
                         torch::Dtype dtype = torch::kFloat64) {
  auto options = MeshBlockOptionsImpl::from_yaml(yaml);
  options->hydro()->eos()->limiter(true);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::dtype(dtype).device(device));
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  if (w.size(0) > ICY) w[ICY].fill_(0.01);  // unsaturated vapor at 350 K

  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  EXPECT_GT(block->pintg->stages.size(), at);
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto step = [&](bool plant) {
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      if (plant && row >= 0 && stage == at) {
        auto u = prim ? vars["hydro_w"].clone() : vars["hydro_u"];
        u.index(interior)[row].select(-1, 1).fill_(value);
        if (prim) block->phydro->peos->compute("W->U", {u});
      }
      block->forward(vars, 1.e-3, stage);
    }
    testing::internal::CaptureStdout();
    int redo = block->check_redo(vars);
    return std::make_pair(redo, testing::internal::GetCapturedStdout());
  };
  auto [redo, log] = step(true);
  return {redo, log, step(false).first};
}

double const kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

// A NaN the limiter zeroes in a momentum row leaves density and energy intact.
// Planted before stage 0 it models a state inherited from between steps: the
// restore repairs it in place, so the retry is clean.
TEST(forcing, limiter_nan_in_a_velocity_row_redoes_the_step) {
  auto r = limiter_step("test_gravity_energy.yaml", IVX, kNaN, 0);
  EXPECT_EQ(r.redo, 1) << r.log;
  EXPECT_NE(r.log.find("(causes: nan)"), std::string::npos) << r.log;
  EXPECT_EQ(r.retry, 0);
}

// Planted at stage 1, the NaN is seen only by the stage-entry limiter.
TEST(forcing, limiter_nan_in_a_vapor_row_redoes_the_step) {
  auto r = limiter_step("test_diffusion_moist.yaml", ICY, kNaN, 1);
  EXPECT_EQ(r.redo, 1) << r.log;
  EXPECT_NE(r.log.find("(causes: nan)"), std::string::npos) << r.log;
}

// 3.5 K at stage 1 entry is floored there; the RK average with the 350 K
// stage-0 state is far above the floor, so only the stage-entry call sees it.
TEST(forcing, limiter_patch_at_stage_entry_redoes_the_step) {
  auto r = limiter_step("test_gravity_energy.yaml", IPR, 1.e3 / 0.4, 1);
  EXPECT_EQ(r.redo, 1) << r.log;
  EXPECT_NE(r.log.find("(causes: limiter)"), std::string::npos) << r.log;
}

// A NaN primitive handed to W->U is zeroed by the primitive limiter alone.
TEST(forcing, limiter_nan_in_a_primitive_redoes_the_step) {
  auto r = limiter_step("test_gravity_energy.yaml", IVX, kNaN, 1, true);
  EXPECT_EQ(r.redo, 1) << r.log;
  EXPECT_NE(r.log.find("(causes: nan)"), std::string::npos) << r.log;
}

// moist first: the species table is process-global and the first yaml sets it
TEST(forcing, limiter_clean_step_is_not_redone) {
  for (auto yaml : {"test_diffusion_moist.yaml", "test_gravity_energy.yaml"}) {
    auto r = limiter_step(yaml, -1, 0., 0);
    EXPECT_EQ(r.redo, 0) << yaml << "\n" << r.log;
    EXPECT_EQ(r.log.find("Redoing"), std::string::npos) << yaml << "\n"
                                                        << r.log;
    EXPECT_EQ(r.retry, 0) << yaml;
  }
}

// The marks are device tensors: the same cases on a GPU.
TEST(forcing, limiter_marks_on_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA device";
  torch::Device cuda(torch::kCUDA);
  auto moist = "test_diffusion_moist.yaml";
  auto dry = "test_gravity_energy.yaml";
  EXPECT_EQ(limiter_step(moist, -1, 0., 0, false, cuda).redo, 0);
  EXPECT_EQ(limiter_step(dry, -1, 0., 0, false, cuda).redo, 0);
  for (auto [yaml, row, value, at, prim, cause] :
       {std::tuple{moist, int(ICY), kNaN, 1, false, "(causes: nan)"},
        std::tuple{dry, int(IVX), kNaN, 0, false, "(causes: nan)"},
        std::tuple{dry, int(IPR), 1.e3 / 0.4, 1, false, "(causes: limiter)"},
        std::tuple{dry, int(IVX), kNaN, 1, true, "(causes: nan)"},
        std::tuple{moist, int(ICY), -1.e-4, 1, false, "(causes: limiter)"},
        std::tuple{moist, int(ICY) + 1, -1.e-4, 1, false, "(causes: limiter)"},
        std::tuple{moist, int(ICY), -1.e-4, 1, true, "(causes: limiter)"}}) {
    auto r = limiter_step(yaml, row, value, at, prim, cuda);
    EXPECT_EQ(r.redo, 1) << yaml << " row " << row << "\n" << r.log;
    EXPECT_NE(r.log.find(cause), std::string::npos) << r.log;
    EXPECT_EQ(r.retry, 0) << yaml << " row " << row;
  }
}

// A species-only repair changes the state with density and energy intact:
// the limiter must still mark the step, like a floor patch. -1e-4 at stage 1
// entry: a negative vapor is refilled from its column (fix_vapor), a negative
// cloud borrows from its parent vapor, and a negative primitive vapor handed
// to W->U is clamped by the primitive limiter.
TEST(forcing, limiter_species_repair_redoes_the_step) {
  auto moist = "test_diffusion_moist.yaml";
  for (auto [row, prim] :
       {std::pair{int(ICY), false}, std::pair{int(ICY) + 1, false},
        std::pair{int(ICY), true}}) {
    auto r = limiter_step(moist, row, -1.e-4, 1, prim);
    EXPECT_EQ(r.redo, 1) << "row " << row << " prim " << prim << "\n" << r.log;
    EXPECT_NE(r.log.find("(causes: limiter)"), std::string::npos) << r.log;
    EXPECT_EQ(r.retry, 0) << "row " << row << " prim " << prim;
  }
}

// A repair of round-off size is applied but not marked (#256): run_hydro's
// kinetics leaves ~-1e-304 cloud in cloud-free cells, and no smaller dt
// removes it. The mark measures the deficit the repair covers, against
// kSpeciesRepairMarkUlp ulp (9.1e-13) of the cell's gas density: -1e-300
// cloud in a cloud-free unit-density cell, in u or in a W->U primitive, is
// not redone; -1e-9 is 1.1e3 times above the bound and is. A -1e-300 vapor
// next to 0.01 vapor is not redone either: fix_vapor borrows only the 1e-300
// (it used to re-mix the column and move 5e-3).
TEST(forcing, limiter_roundoff_species_repair_is_not_redone) {
  auto moist = "test_diffusion_moist.yaml";
  for (bool prim : {false, true}) {
    auto r = limiter_step(moist, ICY + 1, -1.e-300, 1, prim);
    EXPECT_EQ(r.redo, 0) << "prim " << prim << "\n" << r.log;
    EXPECT_EQ(r.log.find("Redoing"), std::string::npos) << r.log;
    EXPECT_EQ(r.retry, 0) << "prim " << prim;

    auto g = limiter_step(moist, ICY + 1, -1.e-9, 1, prim);
    EXPECT_EQ(g.redo, 1) << "prim " << prim << "\n" << g.log;
    EXPECT_NE(g.log.find("(causes: limiter)"), std::string::npos) << g.log;
    EXPECT_EQ(g.retry, 0) << "prim " << prim;
  }

  auto v = limiter_step(moist, ICY, -1.e-300, 1);
  EXPECT_EQ(v.redo, 0) << v.log;
  EXPECT_EQ(v.log.find("Redoing"), std::string::npos) << v.log;
}

// The same bound in float32 is kPositivityRoundoffUlpFloat float32 ulp
// (7.6e-6), not 4096 of them (4.9e-4): at 4096, the -1e-4 cloud repair that
// is redone in float64 was accepted as round-off. -1e-5 is 1.3 times above
// the bound and is redone; -8 ulp, eight times the rounding of a species
// update, and -1e-37, the float32 analogue of the kinetics residue, are not.
TEST(forcing, limiter_float32_roundoff_is_ulp_of_float32) {
  auto moist = "test_diffusion_moist.yaml";
  double ulp = std::numeric_limits<float>::epsilon();
  for (bool prim : {false, true}) {
    for (double value : {-1.e-4, -1.e-5}) {
      auto g = limiter_step(moist, ICY + 1, value, 1, prim, torch::kCPU,
                            torch::kFloat32);
      EXPECT_EQ(g.redo, 1) << "value " << value << " prim " << prim << "\n"
                           << g.log;
      EXPECT_NE(g.log.find("(causes: limiter)"), std::string::npos) << g.log;
      EXPECT_EQ(g.retry, 0) << "value " << value << " prim " << prim;
    }
    for (double value : {-8. * ulp, -1.e-37}) {
      auto r = limiter_step(moist, ICY + 1, value, 1, prim, torch::kCPU,
                            torch::kFloat32);
      EXPECT_EQ(r.redo, 0) << "value " << value << " prim " << prim << "\n"
                           << r.log;
      EXPECT_EQ(r.retry, 0) << "value " << value << " prim " << prim;
    }
  }
}

TEST(forcing, vertical_gravity_work_uses_continuity_mass_flux) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  options->hydro()->diffusion() = nullptr;
  options->hydro()->icorr() = nullptr;

  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;

  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = make_primitive(block);
  w[IDN] = 1. + 0.05 * coord->x1v;
  w[IVX] = 0.3 + 0.07 * coord->x1v.square();

  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  double dt = 0.1;
  auto du = block->phydro->forward(dt, u, vars);

  int is = coord->il();
  int ie = coord->iu() + 1;
  int ny = du.size(0) - ICY;
  auto mass_du = du[IDN].clone();
  auto mass_flux = block->phydro->flux1()[IDN].clone();
  if (ny > 0) {
    mass_du += du.narrow(0, ICY, ny).sum(0);
    mass_flux += block->phydro->flux1().narrow(0, ICY, ny).sum(0);
  }

  auto phi_cell = -gravity->grav1() * coord->x1v;
  auto phi_face = -gravity->grav1() * coord->x1f;
  auto total_energy_du = du[IPR] + phi_cell * mass_du;
  total_energy_du.slice(-1, is, ie) -=
      booked_pe_work(coord, mass_du, gravity->grav1());
  auto total_energy_flux =
      block->phydro->flux1()[IPR] +
      phi_face.narrow(0, 0, mass_flux.size(-1)) * mass_flux;
  auto volume = coord->cell_volume();
  auto area = coord->face_area1();

  auto integrated_du =
      (total_energy_du.slice(-1, is, ie) * volume.slice(-1, is, ie)).sum();
  auto boundary_flux = (area.select(-1, ie) * total_energy_flux.select(-1, ie) -
                        area.select(-1, is) * total_energy_flux.select(-1, is))
                           .sum();
  EXPECT_NEAR(integrated_du.item<double>(),
              (-dt * boundary_flux).item<double>(), 1.e-9);
}

TEST(forcing, vertical_gravity_work_includes_sedimentation_mass_flux) {
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_gravity_sedimentation.yaml");
  options->hydro()->icorr() = nullptr;

  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;

  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = make_primitive(block);
  w[IDN].fill_(1.);
  w.narrow(0, IVX, 3).zero_();
  w[IPR].fill_(1.e5);
  w[ICY].fill_(0.05);
  w[ICY + 1].fill_(0.02);

  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  double dt = 0.1;
  auto du = block->phydro->forward(dt, u, vars);

  ASSERT_TRUE(block->phydro->psed);
  int is = coord->il();
  int ie = coord->iu() + 1;
  auto sedimentation_velocity = block->phydro->psed->vsed[0];
  EXPECT_TRUE(torch::allclose(
      sedimentation_velocity.slice(-1, is + 1, ie),
      torch::full_like(sedimentation_velocity.slice(-1, is + 1, ie), -2.)));

  int ny = du.size(0) - ICY;
  auto mass_du = du[IDN].clone();
  auto mass_flux = block->phydro->flux1()[IDN].clone();
  if (ny > 0) {
    mass_du += du.narrow(0, ICY, ny).sum(0);
    mass_flux += block->phydro->flux1().narrow(0, ICY, ny).sum(0);
  }

  auto phi_cell = -gravity->grav1() * coord->x1v;
  auto phi_face = -gravity->grav1() * coord->x1f;
  auto mechanical_du = du[IPR] + phi_cell * mass_du;
  mechanical_du.slice(-1, is, ie) -=
      booked_pe_work(coord, mass_du, gravity->grav1());
  auto mechanical_flux = block->phydro->flux1()[IPR] +
                         phi_face.narrow(0, 0, mass_flux.size(-1)) * mass_flux;
  auto integrated_du =
      (mechanical_du.slice(-1, is, ie) * coord->cell_volume().slice(-1, is, ie))
          .sum();
  auto boundary_flux =
      (coord->face_area1().select(-1, ie) * mechanical_flux.select(-1, ie) -
       coord->face_area1().select(-1, is) * mechanical_flux.select(-1, is))
          .sum();

  EXPECT_NEAR(integrated_du.item<double>(),
              (-dt * boundary_flux).item<double>(), 1.e-9);
}

TEST(forcing, vertical_gravity_work_excludes_horizontal_mass_divergence) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_forcing_3d.yaml");
  options->hydro()->diffusion() = nullptr;
  options->hydro()->icorr() = nullptr;

  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;

  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = make_primitive(block);
  auto x1 = coord->x1v.view({1, 1, -1});
  auto x2 = coord->x2v.view({1, -1, 1});
  w[IDN] = 1. + 0.04 * x1 + 0.08 * x2;
  w[IVX] = 0.2 + 0.03 * x1;
  w[IVY] = -0.4 + 0.15 * x2;
  w[IPR] = 1.e5 + 20. * x1 + 40. * x2;

  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  double dt = 0.1;
  auto du = block->phydro->forward(dt, u, vars);

  int ny = du.size(0) - ICY;
  auto total_mass_du = du[IDN].clone();
  auto mass_flux1 = block->phydro->flux1()[IDN].clone();
  auto mass_flux2 = block->phydro->flux2()[IDN].clone();
  auto mass_flux3 = block->phydro->flux3()[IDN].clone();
  if (ny > 0) {
    total_mass_du += du.narrow(0, ICY, ny).sum(0);
    mass_flux1 += block->phydro->flux1().narrow(0, ICY, ny).sum(0);
    mass_flux2 += block->phydro->flux2().narrow(0, ICY, ny).sum(0);
    mass_flux3 += block->phydro->flux3().narrow(0, ICY, ny).sum(0);
  }

  auto phi_cell = -gravity->grav1() * coord->x1v;
  auto phi_face = -gravity->grav1() * coord->x1f;
  auto flux1 = block->phydro->flux1()[IPR] +
               phi_face.narrow(0, 0, mass_flux1.size(-1)) * mass_flux1;
  auto flux2 = block->phydro->flux2()[IPR] + phi_cell * mass_flux2;
  auto flux3 = block->phydro->flux3()[IPR] + phi_cell * mass_flux3;
  auto expected =
      -dt * coord->divergence(flux1.unsqueeze(0), flux2.unsqueeze(0),
                              flux3.unsqueeze(0))[0];
  auto actual = du[IPR] + phi_cell * total_mass_du;
  // the corrected-PE work follows the x1 mass flux only
  {
    int is = coord->il(), ie = coord->iu() + 1;
    auto area1 = coord->face_area1();
    auto drho1 = torch::zeros_like(total_mass_du);
    drho1.slice(-1, is, ie)
        .copy_(-dt *
               (area1.slice(-1, is + 1, ie + 1) *
                    mass_flux1.slice(-1, is + 1, ie + 1) -
                area1.slice(-1, is, ie) * mass_flux1.slice(-1, is, ie)) /
               coord->cell_volume().slice(-1, is, ie));
    actual.slice(-1, is, ie) -= booked_pe_work(coord, drho1, gravity->grav1());
  }
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));

  EXPECT_TRUE(torch::allclose(actual.index(interior), expected.index(interior),
                              1.e-10, 1.e-8));
}

static void implicit_gravity_work_holds_under_rk3_stage_weighting(
    torch::Device device) {
  // publish rk_stage as advance_local does; #202's tests call forward directly
  std::vector<double> stage_momentum;
  for (int stage = 0; stage < 3; ++stage) {
    auto options = MeshBlockOptionsImpl::from_yaml("test_gravity_energy.yaml");

    auto gravity = ConstGravityOptionsImpl::create();
    gravity->grav1(-1.);
    gravity->gravity_work("face");  // the form these tests pin
    options->hydro()->grav() = gravity;
    auto icorr = ImplicitOptionsImpl::create();
    icorr->scheme(1);
    options->hydro()->icorr() = icorr;

    auto block = std::make_shared<MeshBlockImpl>(options);
    block->to(device, torch::kFloat64);
    auto coord = block->pcoord;

    ASSERT_EQ(block->pintg->stages.size(), 3u)
        << "stage weighting is guarded on a 3-stage integrator";

    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::dtype(torch::kFloat64).device(device));
    w[IDN] = 1. + 0.05 * coord->x1v;
    w[IVX].zero_();
    w[IPR].fill_(1.e5);

    Variables vars;
    vars["hydro_w"] = w;
    block->initialize(vars);

    // What advance_local publishes before calling phydro->forward.
    block->phydro->rk_stage = stage;

    double dt = 0.1;
    auto du = block->phydro->forward(dt, vars.at("hydro_u"), vars);
    ASSERT_EQ(du.device(), device);
    stage_momentum.push_back(du[IVX].abs().sum().item<double>());

    int is = coord->il();
    int ie = coord->iu() + 1;
    int ny = du.size(0) - ICY;
    auto mass_du = du[IDN].clone();
    auto mass_flux = block->phydro->flux1()[IDN].clone();
    if (ny > 0) {
      mass_du += du.narrow(0, ICY, ny).sum(0);
      mass_flux += block->phydro->flux1().narrow(0, ICY, ny).sum(0);
    }
    auto phi_cell = -gravity->grav1() * coord->x1v;
    auto phi_face = -gravity->grav1() * coord->x1f;
    auto mechanical_du = du[IPR] + phi_cell * mass_du;
    mechanical_du.slice(-1, is, ie) -=
        booked_pe_work(coord, mass_du, gravity->grav1());
    auto mechanical_flux =
        block->phydro->flux1()[IPR] +
        phi_face.narrow(0, 0, mass_flux.size(-1)) * mass_flux;
    auto integrated_du = (mechanical_du.slice(-1, is, ie) *
                          coord->cell_volume().slice(-1, is, ie))
                             .sum();
    auto boundary_flux =
        (coord->face_area1().select(-1, ie) * mechanical_flux.select(-1, ie) -
         coord->face_area1().select(-1, is) * mechanical_flux.select(-1, is))
            .sum();

    EXPECT_NEAR(integrated_du.item<double>(),
                (-dt * boundary_flux).item<double>(), 1.e-9)
        << "total-energy conservation broken at rk3 stage " << stage;
  }

  // stage 1 weights dt by 1/4; these coincide if the weighting stops arriving.
  EXPECT_GT(std::abs(stage_momentum[0] - stage_momentum[1]), 1.e-6)
      << "the stage weighting is not reaching the implicit correction";
}

TEST(forcing, implicit_gravity_work_holds_under_rk3_stage_weighting) {
  implicit_gravity_work_holds_under_rk3_stage_weighting(torch::kCPU);
}

TEST(forcing, implicit_gravity_work_holds_under_rk3_stage_weighting_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  implicit_gravity_work_holds_under_rk3_stage_weighting(
      torch::Device(torch::kCUDA, 0));
}

TEST(forcing, rk3_stage_is_published_by_block_step) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_gravity_energy.yaml");
  options->intg()->type("rk3");
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;
  auto icorr = ImplicitOptionsImpl::create();
  icorr->scheme(1);
  options->hydro()->icorr() = icorr;

  auto block = std::make_shared<MeshBlockImpl>(options);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN] = 1. + 0.05 * coord->x1v;
  w[IPR].fill_(1.e5);
  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  ASSERT_EQ(block->pintg->stages.size(), 3u);
  for (int stage = 0; stage < 3; ++stage) {
    block->phydro->rk_stage = -1;
    block->advance_local(vars, 0.1, stage);
    EXPECT_EQ(block->phydro->rk_stage, stage);
    EXPECT_TRUE(torch::isfinite(vars.at("hydro_u")).all().item<bool>());
  }
}

TEST(forcing, relax_bottom_composition_handles_multidimensional_ghost_zones) {
  auto block = make_block("test_forcing_3d.yaml");
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 2., species: [vapor], xfrac: [0.2]}"));

  RelaxBotComp(op, block->phydro.get())->forward(du, w, temp, 0.5);

  expect_only_bottom(du, block);
}

TEST(forcing, body_heat_uses_pressure_mask_and_mixture_cv) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int il = block->pcoord->il();
  w[IPR].select(-1, il + 1).fill_(2.e5);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {dTdt: 2., pmin: 1.5e5, pmax: 2.5e5}"));

  BodyHeat(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto pres = w[IPR].index(interior);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp).index(interior);
  expected[IPR].index_put_(
      interior,
      torch::where(torch::logical_and(pres >= 1.5e5, pres <= 2.5e5),
                   w[IDN].index(interior) * cv, torch::zeros_like(pres)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, boundary_fluxes_scale_with_timestep) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du1 = torch::zeros_like(w);
  auto du2 = torch::zeros_like(w);
  auto top_op = TopCoolOptionsImpl::from_yaml(
      YAML::Load("top-cool: {flux: -100., depth: 2}"));
  auto bot_op = BotHeatOptionsImpl::from_yaml(
      YAML::Load("bot-heat: {flux: 100., depth: 2}"));
  TopCool top(top_op, block->phydro.get());
  BotHeat bot(bot_op, block->phydro.get());

  top->forward(du1, w, temp, 0.25);
  bot->forward(du1, w, temp, 0.25);
  top->forward(du2, w, temp, 0.5);
  bot->forward(du2, w, temp, 0.5);

  EXPECT_TRUE(torch::allclose(du2, 2. * du1));
}

namespace {
// the same data behind a non-contiguous view: SedHydro routes it to the
// tensor fallback instead of the fused kernel
torch::Tensor strided_copy(torch::Tensor const& w) {
  auto padded = torch::zeros({w.size(0), w.size(1), w.size(2), w.size(3) + 1},
                             w.options());
  return padded.narrow(-1, 0, w.size(3)).copy_(w);
}

std::shared_ptr<MeshBlockImpl> make_sedimenting_block(std::string const& yaml,
                                                      double grav1) {
  auto options = MeshBlockOptionsImpl::from_yaml(yaml);
  options->hydro()->icorr() = nullptr;
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(grav1);
  options->hydro()->grav() = gravity;
  return std::make_shared<MeshBlockImpl>(options);
}

TEST(forcing, native_dry_source_uses_each_rk_stage_weight) {
  struct Case {
    char const* type;
    std::vector<double> wght2;
  };
  std::vector<Case> cases = {
      {"rk1", {1.}},
      {"rk2", {1., 0.5}},
      {"rk3", {1., 0.25, 2. / 3.}},
      {"rk3s4", {0.5, 0.5, 1. / 6., 0.5}},
  };

  auto run = [](char const* type, int target_stage, double increment) {
    auto options =
        MeshBlockOptionsImpl::from_yaml("test_scalar_source_bound.yaml");
    options->intg()->type() = type;
    auto block = std::make_shared<MeshBlockImpl>(options);
    auto source = DryDensitySource(0.);
    block->phydro->forcings.push_back(torch::nn::AnyModule(source));

    auto coord = block->pcoord;
    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    auto r = torch::full({1, coord->options->nc3(), coord->options->nc2(),
                          coord->options->nc1()},
                         0.9, w.options());
    Variables vars{{"hydro_w", w}, {"scalar_r", r}};
    block->initialize(vars);

    for (int stage = 0; stage <= target_stage; ++stage) {
      source->increment = stage == target_stage ? increment : 0.;
      block->forward(vars, 6.e-4, stage);
    }
    return vars.at("hydro_u")[IDN]
        .select(-1, vars.at("hydro_u").size(-1) / 2)
        .item<double>();
  };

  for (auto const& c : cases) {
    for (int stage = 0; stage < c.wght2.size(); ++stage) {
      double base = run(c.type, stage, 0.);
      for (double increment : {-0.2, 0.2}) {
        double actual = run(c.type, stage, increment) - base;
        EXPECT_NEAR(actual, c.wght2[stage] * increment, 1.e-12)
            << "integrator=" << c.type << " stage=" << stage
            << " increment=" << increment;
      }
    }
  }
}

TEST(forcing, native_dry_source_preserves_scalar_bounds_at_every_rk_order) {
  constexpr double dt = 6.e-4;
  for (auto type : {"rk1", "rk2", "rk3", "rk3s4"}) {
    for (double increment : {-0.5, 0.5}) {
      auto options =
          MeshBlockOptionsImpl::from_yaml("test_scalar_source_bound.yaml");
      options->intg()->type() = type;
      auto block = std::make_shared<MeshBlockImpl>(options);
      block->phydro->forcings.push_back(
          torch::nn::AnyModule(DryDensitySource(increment)));

      auto coord = block->pcoord;
      auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                             coord->options->nc2(), coord->options->nc1()},
                            torch::kFloat64);
      w[IDN].fill_(1.);
      w[IVX].fill_(1000.);
      w[IPR].fill_(1.e5);
      auto r = torch::full({1, coord->options->nc3(), coord->options->nc2(),
                            coord->options->nc1()},
                           0.9, w.options());
      int il = coord->il(), iu = coord->iu();
      r.select(-1, il + 7).fill_(1.);

      Variables vars{{"hydro_w", w}, {"scalar_r", r}};
      block->initialize(vars);
      ASSERT_GE(block->max_time_step(vars), dt);
      for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
        block->forward(vars, dt, stage);
      }

      auto interior = vars.at("scalar_r").slice(-1, il, iu + 1);
      EXPECT_LE(interior.max().item<double>(), 1. + 1.e-12)
          << "integrator=" << type << " increment=" << increment;
      EXPECT_GE(interior.min().item<double>(), -1.e-12)
          << "integrator=" << type << " increment=" << increment;
    }
  }
}

}  // namespace

// three copies of one Stokes formula: the fused kernel must agree with the
// tensor fallback (which carries the reference mean free path)
TEST(forcing, fused_sedimentation_matches_tensor_path) {
  auto block = make_sedimenting_block("test_stokes_sedimentation.yaml", -10.);
  auto psed = block->phydro->psed;
  ASSERT_TRUE(psed);
  auto w = make_primitive(block);
  w.narrow(0, IVX, 3).zero_();
  w[ICY].fill_(0.05);
  w[ICY + 1].fill_(0.02);

  psed->forward(w);
  auto fused = psed->vsed.clone();
  auto strided = strided_copy(w);
  ASSERT_FALSE(strided.is_contiguous());
  psed->forward(strided);
  auto tensor = psed->vsed.clone();

  int il = block->pcoord->il(), iu = block->pcoord->iu();
  EXPECT_LT(fused[0].slice(-1, il + 1, iu + 1).max().item<double>(), 0.);
  EXPECT_TRUE(torch::allclose(fused, tensor, 1.e-10, 0.));
}

// a block edge that is not a physical wall (an x1 seam) must not seal the
// settling flux; the physical top still does
TEST(forcing, sedimentation_is_sealed_only_at_physical_walls) {
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_gravity_sedimentation.yaml");
  options->hydro()->icorr() = nullptr;
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  gravity->gravity_work("face");  // the form these tests pin
  options->hydro()->grav() = gravity;
  options->bfuncs()[BoundaryFace::kInnerX1] = nullptr;
  auto block = std::make_shared<MeshBlockImpl>(options);
  auto w = make_primitive(block);
  w.narrow(0, IVX, 3).zero_();
  w[ICY].fill_(0.05);
  w[ICY + 1].fill_(0.02);
  int il = block->pcoord->il(), iu = block->pcoord->iu();

  for (auto wr : {w, strided_copy(w)}) {
    block->phydro->psed->forward(wr);
    auto vsed = block->phydro->psed->vsed[0];
    EXPECT_DOUBLE_EQ(vsed.select(-1, il).item<double>(), -2.);
    EXPECT_DOUBLE_EQ(vsed.select(-1, iu + 1).item<double>(), 0.);
  }
}
