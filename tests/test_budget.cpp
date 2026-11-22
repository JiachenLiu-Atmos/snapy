// V3 lane O step O8, its budget part (O.rst section 3.2 and the O8 row;
// gate G-4's budget cases, O.rst 1100-1111): the budget consumer closes
// every row on the boundary transfer of a source-free run; a restart of a
// deck with a budget output resumes its interval from BudgetCarry; a file
// without a carry starts a new interval at the first observe, and a deck
// without a budget output writes none. The _cuda entry (SNAP_BUDGET_CUDA)
// runs every case with the blocks and hydro_w on the GPU, restarts included.

// C/C++
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// external
#include <gtest/gtest.h>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>
#include <snap/output/budget_series.hpp>
#include <snap/output/output_type.hpp>
#include <snap/output/restart.hpp>

#include "cuda_test_gate.hpp"

using namespace snap;
namespace fs = std::filesystem;

namespace {

// the _cuda entry: every block (and so every restart arm) on the GPU
bool const kCuda = getenv("SNAP_BUDGET_CUDA") != nullptr;

fs::path fresh_dir(std::string const& tag) {
  // the files one by one: remove_all resolves into libtorch's copy here
  auto dir = fs::temp_directory_path() /
             ("snapy_budget_" + tag + (kCuda ? "_cuda" : ""));
  if (fs::exists(dir))
    for (auto const& e : fs::directory_iterator(dir)) fs::remove(e.path());
  fs::create_directories(dir);
  return dir;
}

Mesh make_mesh(fs::path const& dir, bool budget, bool restart) {
  auto bo = MeshBlockOptionsImpl::from_yaml("test_budget.yaml");
  bo->layout()->device(kCuda ? "cuda" : "cpu");
  std::vector<OutputOptions> outs;
  if (budget) {
    auto b = OutputOptionsImpl::create();
    b->file_type("budget");
    b->dt(0.3);
    outs.push_back(b);
  }
  if (restart) {
    auto r = OutputOptionsImpl::create();
    r->file_type("restart");
    r->dt(0.5);
    outs.push_back(r);
  }
  bo->outputs(outs);
  bo->output_dir(dir.string()).basename("budget");
  auto mo = MeshOptionsImpl::create();
  mo->block(bo);
  mo->blocks_per_process(2);
  auto mesh = Mesh(mo);
  mesh->to(torch::Device(bo->device_str()));  // as a driver does
  return mesh;
}

std::shared_ptr<BudgetOutput> budget_of(Mesh const& mesh, size_t b = 0) {
  for (auto const& out : mesh->blocks[b]->output_types)
    if (auto p = std::dynamic_pointer_cast<BudgetOutput>(out)) return p;
  return nullptr;
}

struct Run {
  std::vector<BudgetRecord> records;  // every record, in order
  std::vector<double> dts;            // every accepted step
  double time = 0.;
};

// step to cycle `last`, from a restart file when given; at_start: the
// outputs are made once before the first step, as the example drivers do
Run run(Mesh mesh, int last, std::string const& restart = "",
        bool at_start = true) {
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto const& blk = mesh->blocks[i];
    auto co = blk->pcoord->options;
    int nvar = blk->phydro->peos->nvar();
    // hydro_w on the block's device, as a driver hands it in
    auto x1 = blk->pcoord->x1v.to(torch::kFloat64);
    auto w =
        torch::zeros({nvar, co->nc3(), co->nc2(), co->nc1()}, x1.options());
    w[IDN] = (1. + 0.3 * torch::sin(2. * M_PI * x1 / 1600.)).view({1, 1, -1});
    w[IVX] = 50.;
    w[IVY] = 5.;
    w[IPR] = 1.e5;
    vars[i]["hydro_w"] = w;
  }
  Run r;
  r.time = restart.empty() ? mesh->initialize(vars)
                           : mesh->initialize(vars, restart.c_str());
  auto out = budget_of(mesh);
  double seen = -1.;
  auto note = [&] {
    if (out && out->record().total.defined() && out->record().time != seen) {
      r.records.push_back(out->record());
      seen = out->record().time;
    }
  };
  if (at_start) {
    mesh->make_outputs(vars, r.time);
    note();
  }
  int cycle = mesh->blocks.front()->cycle;
  while (cycle < last) {
    double dt = mesh->max_time_step(vars);
    for (int s = 0; s < mesh->blocks.front()->pintg->stages.size(); ++s)
      mesh->forward(vars, dt, s);
    int redo = mesh->check_redo(vars);
    if (redo > 0) continue;
    EXPECT_EQ(redo, 0);
    if (redo < 0) break;
    mesh->set_cycle(++cycle);
    r.time += dt;
    r.dts.push_back(dt);
    mesh->make_outputs(vars, r.time);
    note();
  }
  return r;
}

void expect_equal(BudgetRecord const& a, BudgetRecord const& b) {
  EXPECT_EQ(a.cycle, b.cycle);
  EXPECT_EQ(a.time, b.time);
  EXPECT_EQ(a.interval, b.interval);
  EXPECT_TRUE(torch::equal(a.total, b.total));
  EXPECT_TRUE(torch::equal(a.change, b.change));
  EXPECT_TRUE(torch::equal(a.boundary, b.boundary));
  EXPECT_TRUE(torch::equal(a.counter, b.counter));
  EXPECT_TRUE(torch::equal(a.minimum, b.minimum));
}

fs::path restart_file(fs::path const& dir, int number) {
  char name[32];
  snprintf(name, sizeof(name), "budget.%05d.restart", number);
  return dir / name;
}

}  // namespace

TEST(Budget, every_row_closes_on_the_boundary_transfer) {
  if (kCuda && !snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA";
  auto dir = fresh_dir("closure");
  auto r = run(make_mesh(dir, true, false), 12);
  ASSERT_GE(r.records.size(), 3u);
  auto const& names = r.records.front().rows;
  ASSERT_EQ(names.front(), "mass");
  for (size_t k = 1; k < r.records.size(); ++k) {
    auto const& rec = r.records[k];
    EXPECT_GT(rec.interval, 0.);
    // the outflow carries mass out: the check is not empty
    EXPECT_GT(std::abs(rec.boundary[IDN].item<double>()), 0.);
    for (int64_t n = 0; n < rec.total.numel(); ++n) {
      double change = rec.change[n].item<double>();
      double out = rec.boundary[n].item<double>();
      double scale = std::max(
          {std::abs(rec.total[n].item<double>()), std::abs(out), 1.e-300});
      EXPECT_LT(std::abs(change + out), 1.e-12 * scale)
          << names[n] << " record " << k;
    }
  }
  EXPECT_TRUE(fs::exists(dir / "budget.budget.nc"));
}

TEST(Budget, a_restart_resumes_the_interval_from_its_carry) {
  if (kCuda && !snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA";
  auto dir = fresh_dir("carry_a");
  auto full = run(make_mesh(dir, true, true), 12);
  auto file = restart_file(dir, 1);
  ASSERT_TRUE(fs::exists(file));
  auto data = load_restart(file.string(), 0);
  ASSERT_TRUE(data.count(std::string(kBudgetCarryPrefix) + "pending"));
  // the restart falls inside a budget interval, which the carry closes
  ASSERT_GT(
      data.at(std::string(kBudgetCarryPrefix) + "interval").item<double>(), 0.);

  auto dir2 = fresh_dir("carry_b");
  auto resumed = run(make_mesh(dir2, true, true), 12, file.string());
  ASSERT_FALSE(resumed.records.empty());
  // every record after the restart is the uninterrupted run's
  for (auto const& rec : resumed.records) {
    bool found = false;
    for (auto const& ref : full.records)
      if (ref.cycle == rec.cycle) {
        expect_equal(rec, ref);
        found = true;
      }
    EXPECT_TRUE(found) << "cycle " << rec.cycle;
  }
}

TEST(Budget, a_file_without_a_carry_starts_at_the_first_observe) {
  if (kCuda && !snapy_cuda_test_enabled()) GTEST_SKIP() << "no CUDA";
  // a deck without a budget output writes exactly the old stored set
  auto dir = fresh_dir("nocarry_a");
  run(make_mesh(dir, false, true), 12);
  auto file = restart_file(dir, 1);
  ASSERT_TRUE(fs::exists(file));
  for (auto const& [key, value] : load_restart(file.string(), 0))
    EXPECT_NE(key.rfind(kBudgetCarryPrefix, 0), 0u) << key;

  // the budget output is new to the resumed deck, so it records at the
  // first make_outputs; with none before the first step, the interval starts
  // at the first observe, after that step: its record (one step later) has
  // an empty interval and no change
  auto dir2 = fresh_dir("nocarry_b");
  auto resumed = run(make_mesh(dir2, true, true), 12, file.string(), false);
  ASSERT_GE(resumed.records.size(), 2u);
  int start = 12 - static_cast<int>(resumed.dts.size());
  auto const& first = resumed.records.front();
  EXPECT_EQ(first.cycle, start + 1);
  EXPECT_EQ(first.interval, 0.);
  EXPECT_EQ(first.change.abs().max().item<double>(), 0.);
  EXPECT_EQ(first.boundary.abs().max().item<double>(), 0.);
  // the next interval is a whole one, and it closes
  auto const& next = resumed.records[1];
  double expected = 0.;
  for (int c = first.cycle; c < next.cycle; ++c)
    expected += resumed.dts[c - start];
  EXPECT_EQ(next.interval, expected);
  for (int64_t n = 0; n < next.total.numel(); ++n) {
    double change = next.change[n].item<double>();
    double out = next.boundary[n].item<double>();
    double scale = std::max(
        {std::abs(next.total[n].item<double>()), std::abs(out), 1.e-300});
    EXPECT_LT(std::abs(change + out), 1.e-12 * scale) << next.rows[n];
  }
}
