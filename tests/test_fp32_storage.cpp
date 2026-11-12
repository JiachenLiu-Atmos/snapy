// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/utils/fp32_storage.hpp>

using namespace snap;

// The emulation hook (fp32 roadmap, Step 0): mode and row parsing, and the
// rounding itself. The environment is not read here: the parsers are pure.

TEST(fp32_storage, parse_mode) {
  EXPECT_EQ(parse_fp32_storage_mode(""), Fp32StorageConfig::kOff);
  EXPECT_EQ(parse_fp32_storage_mode("off"), Fp32StorageConfig::kOff);
  EXPECT_EQ(parse_fp32_storage_mode("0"), Fp32StorageConfig::kOff);
  EXPECT_EQ(parse_fp32_storage_mode(" Step "), Fp32StorageConfig::kStep);
  EXPECT_EQ(parse_fp32_storage_mode("STAGE"), Fp32StorageConfig::kStage);
  EXPECT_THROW(parse_fp32_storage_mode("always"), c10::Error);
}

TEST(fp32_storage, parse_and_resolve_rows) {
  EXPECT_TRUE(parse_fp32_storage_rows("all").empty());
  EXPECT_TRUE(parse_fp32_storage_rows("").empty());

  int64_t nvar = ICY + 3;
  auto rows =
      resolve_fp32_storage_rows(parse_fp32_storage_rows("IPR, idn"), nvar);
  EXPECT_EQ(rows, (std::vector<int64_t>{IDN, IPR}));

  rows = resolve_fp32_storage_rows(parse_fp32_storage_rows("ICY"), nvar);
  EXPECT_EQ(rows, (std::vector<int64_t>{ICY, ICY + 1, ICY + 2}));

  rows =
      resolve_fp32_storage_rows(parse_fp32_storage_rows("ICY+1,IPR,IPR"), nvar);
  EXPECT_EQ(rows, (std::vector<int64_t>{IPR, ICY + 1}));

  rows = resolve_fp32_storage_rows(parse_fp32_storage_rows("1,2"), nvar);
  EXPECT_EQ(rows, (std::vector<int64_t>{1, 2}));

  EXPECT_THROW(resolve_fp32_storage_rows({"icy+3"}, nvar), c10::Error);
  EXPECT_THROW(resolve_fp32_storage_rows({"theta"}, nvar), c10::Error);
}

TEST(fp32_storage, due) {
  Fp32StorageConfig cfg;
  cfg.mode = Fp32StorageConfig::kOff;
  EXPECT_FALSE(fp32_storage_due(cfg, 2, 3));
  cfg.mode = Fp32StorageConfig::kStep;
  EXPECT_FALSE(fp32_storage_due(cfg, 0, 3));
  EXPECT_FALSE(fp32_storage_due(cfg, 1, 3));
  EXPECT_TRUE(fp32_storage_due(cfg, 2, 3));
  cfg.mode = Fp32StorageConfig::kStage;
  EXPECT_TRUE(fp32_storage_due(cfg, 0, 3));
}

TEST(fp32_storage, rounds_selected_rows) {
  auto opts = torch::dtype(torch::kFloat64);
  auto u = torch::full({ICY + 1, 1, 2, 3}, 1.0 + 1.0e-12, opts);
  auto all = u.clone();
  round_through_fp32_(all, {});
  EXPECT_TRUE(torch::equal(all, all.to(torch::kFloat32).to(torch::kFloat64)));
  EXPECT_TRUE(torch::equal(all, torch::ones_like(all)));  // 1 + 1e-12 -> 1

  auto some = u.clone();
  round_through_fp32_(some, {IPR});
  EXPECT_TRUE(torch::equal(some[IPR], torch::ones_like(some[IPR])));
  EXPECT_TRUE(torch::equal(some[IDN], u[IDN]));  // untouched, bit for bit

  // a float32 state is already representable: left alone
  auto f = torch::randn({ICY, 1, 1, 4}, torch::dtype(torch::kFloat32));
  auto g = f.clone();
  round_through_fp32_(g, {});
  EXPECT_TRUE(torch::equal(f, g));
}
