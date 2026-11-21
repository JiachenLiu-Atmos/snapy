// Lane A phase 1 (A.rst 3.6, 5.1): the SourceTerm registry keeps the
// declaration order, splits it into runs only where `capturable` changes, and
// the budgets-on path (one FP64 scratch per source, added to du once) gives
// the same du as the budgets-off path.

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

struct Fake : SourceTerm {
  Fake(char const* name, bool capturable) {
    decl_.name = name;
    decl_.capturable = capturable;
  }
  void add(torch::Tensor du, StageState const&) override {}
};

std::shared_ptr<MeshBlockImpl> forced_block() {
  auto op = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  auto h = op->hydro();
  auto f = YAML::Load(
      "coriolis: {omega1: 3.e-4, omega2: 1.e-4, omega3: -2.e-4}\n"
      "body-heat: {dTdt: 1., pmin: 0., pmax: 2.e5}\n"
      "top-cool: {flux: -50., depth: 2}\n"
      "bot-heat: {flux: 70., depth: 1}\n"
      "relax-bot-comp: {tau: 2., species: [vapor], xfrac: [0.1]}\n"
      "relax-bot-temp: {tau: 2., btemp: 350., at-face: true}\n"
      "relax-bot-velo: {tau: 2., bvx: 1., bvy: -2.}\n"
      "top-sponge-lyr: {tau: 3., width: 1.e9}\n"
      "bot-sponge-lyr: {tau: 4., width: 1.e9}\n");
  h->coriolis() = CoriolisOptionsImpl::from_yaml(f);
  h->bodyHeat() = BodyHeatOptionsImpl::from_yaml(f);
  h->topCool() = TopCoolOptionsImpl::from_yaml(f);
  h->botHeat() = BotHeatOptionsImpl::from_yaml(f);
  h->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(f);
  h->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(f);
  h->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(f);
  h->topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(f);
  h->botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(f);
  return std::make_shared<MeshBlockImpl>(op);
}

StageState stage_of(std::shared_ptr<MeshBlockImpl> const& block) {
  auto c = block->pcoord;
  auto shape =
      std::vector<int64_t>{block->phydro->peos->nvar(), c->options->nc3(),
                           c->options->nc2(), c->options->nc1()};
  auto x = torch::linspace(0., 1., shape[3], torch::kFloat64);
  auto w = torch::zeros(shape, torch::kFloat64);
  w[IDN] = 1. + 0.1 * x;
  w[IVX] = 1. - x;
  w[IVY] = 2. * x;
  w[IVZ] = 3.;
  w[IPR] = 1.e5 * (1. - 0.2 * x);
  w[ICY] = 0.01 + 0.02 * x;
  return {w, block->phydro->peos->compute("W->T", {w}), 0.37};
}

}  // namespace

TEST(SourceRegistry, runs_keep_the_order_and_split_on_capturable) {
  SourceRegistry reg;
  for (auto [name, cap] : {std::pair{"a", true},
                           {"b", true},
                           {"c", false},
                           {"d", true},
                           {"e", true}})
    reg.add(std::make_shared<Fake>(name, cap));
  reg.add(std::make_shared<Fake>("f", true), SourcePhase::post_implicit);

  auto pre = reg.runs(SourcePhase::pre_implicit);
  ASSERT_EQ(pre.size(), 3u);
  EXPECT_TRUE(pre[0].capturable && pre[2].capturable);
  EXPECT_FALSE(pre[1].capturable);
  EXPECT_EQ(pre[0].begin, 0u);
  EXPECT_EQ(pre[0].end, 2u);
  EXPECT_EQ(pre[1].end, 3u);
  EXPECT_EQ(pre[2].end, 5u);
  auto post = reg.runs(SourcePhase::post_implicit);
  ASSERT_EQ(post.size(), 1u);
  EXPECT_EQ(post[0].end, 1u);
}

TEST(SourceRegistry, a_deck_declares_its_sources_in_deck_order) {
  auto block = forced_block();
  auto& reg = block->phydro->sources;
  std::vector<std::string> names;
  auto sink = [&](std::string const& n, torch::Tensor) { names.push_back(n); };
  auto s = stage_of(block);
  auto du = torch::zeros_like(s.w);
  for (auto const& r : reg.runs(SourcePhase::pre_implicit))
    reg.run(r, du, s, sink);
  EXPECT_EQ(names, (std::vector<std::string>{
                       "coriolis", "diffusion", "body-heat", "top-cool",
                       "bot-heat", "relax-bot-comp", "relax-bot-temp",
                       "relax-bot-velo", "top-sponge-lyr", "bot-sponge-lyr"}));
  // diffusion (host reads of its profiles) and relax-bot-comp (kintera's
  // VT->U, A.rst 8) are not capturable: five runs, in deck order
  auto runs = reg.runs(SourcePhase::pre_implicit);
  ASSERT_EQ(runs.size(), 5u);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(runs[i].capturable, i % 2 == 0) << i;
  EXPECT_EQ(runs[1].begin, 1u);
  EXPECT_EQ(runs[3].begin, 5u);
  EXPECT_EQ(runs[3].end, 6u);
  // Coriolis runs again after the implicit solve (#355)
  names.clear();
  for (auto const& r : reg.runs(SourcePhase::post_implicit))
    reg.run(r, du, s, sink);
  EXPECT_EQ(names, std::vector<std::string>{"coriolis"});
  EXPECT_TRUE(reg.splits().empty());
}

TEST(SourceRegistry, budgets_on_give_the_budgets_off_du) {
  auto block = forced_block();
  auto& reg = block->phydro->sources;
  auto s = stage_of(block);
  auto du0 = 1.e-3 * torch::randn_like(s.w);
  auto off = du0.clone(), on = du0.clone();
  auto total = torch::zeros_like(du0);
  int calls = 0;
  for (auto const& r : reg.runs(SourcePhase::pre_implicit)) {
    reg.run(r, off, s);
    reg.run(r, on, s, [&](std::string const&, torch::Tensor x) {
      total += x;
      ++calls;
    });
  }
  EXPECT_EQ(calls, 10);
  EXPECT_TRUE(torch::equal(off, on));
  EXPECT_FALSE(torch::equal(off, du0));
  EXPECT_TRUE(torch::allclose(on - du0, total, 1.e-12, 1.e-12));
}

TEST(SourceRegistry, wall_sources_skip_a_block_without_that_wall) {
  auto block = forced_block();
  auto& reg = block->phydro->sources;
  auto s = stage_of(block);
  std::vector<std::string> names;
  auto sink = [&](std::string const& n, torch::Tensor) { names.push_back(n); };
  reg.on_block = [](SourceRegion r) { return r != SourceRegion::top; };
  auto du = torch::zeros_like(s.w);
  for (auto const& r : reg.runs(SourcePhase::pre_implicit))
    reg.run(r, du, s, sink);
  EXPECT_EQ(names, (std::vector<std::string>{
                       "coriolis", "diffusion", "body-heat", "bot-heat",
                       "relax-bot-comp", "relax-bot-temp", "relax-bot-velo",
                       "bot-sponge-lyr"}));
}

TEST(SourceRegistry, the_options_table_keeps_the_defaults_and_checks) {
  auto none = YAML::Load("{}");
  EXPECT_EQ(CoriolisOptionsImpl::from_yaml(none), nullptr);
  auto c = CoriolisOptionsImpl::from_yaml(YAML::Load("coriolis: {}"));
  EXPECT_EQ(c->type(), "xyz");
  EXPECT_FALSE(c->traditional());
  EXPECT_ANY_THROW(
      CoriolisOptionsImpl::from_yaml(YAML::Load("coriolis: {type: 321}")));
  EXPECT_ANY_THROW(
      CoriolisOptionsImpl::from_yaml(YAML::Load("coriolis: {omega4: 1.}")));
  auto t = TopCoolOptionsImpl::from_yaml(YAML::Load("top-cool: {flux: -1.}"));
  EXPECT_EQ(t->depth(), 1);
  EXPECT_EQ(t->thickness(), 0.);
  EXPECT_ANY_THROW(TopCoolOptionsImpl::from_yaml(
      YAML::Load("top-cool: {flux: -1., depth: 2, thickness: 3.}")));
  EXPECT_ANY_THROW(
      BotHeatOptionsImpl::from_yaml(YAML::Load("bot-heat: {flux: -1.}")));
  auto p = PlumeForcingOptionsImpl::from_yaml(YAML::Load("plume-forcing: {}"));
  EXPECT_EQ(p->entrainment(), 0.1);
  auto d = DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {}"));
  EXPECT_EQ(d->wall_velocity(), "stress-free");
  EXPECT_ANY_THROW(
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {K: 1.}")));
  EXPECT_ANY_THROW(
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {dynamic: yes}")));
}
