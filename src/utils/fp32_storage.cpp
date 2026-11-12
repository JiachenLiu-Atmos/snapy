// C/C++
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

// snap
#include <snap/snap.h>

#include "fp32_storage.hpp"

namespace snap {

namespace {
std::string lower_trim(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
                         [](unsigned char c) { return std::isspace(c); }),
          s.end());
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string env_or(char const* name, char const* def) {
  char const* v = std::getenv(name);
  return v ? std::string(v) : std::string(def);
}
}  // namespace

Fp32StorageConfig::Mode parse_fp32_storage_mode(std::string const& s) {
  auto v = lower_trim(s);
  if (v.empty() || v == "0" || v == "off" || v == "false" || v == "no")
    return Fp32StorageConfig::kOff;
  if (v == "step") return Fp32StorageConfig::kStep;
  if (v == "stage") return Fp32StorageConfig::kStage;
  TORCH_CHECK(false, "SNAP_FP32_STORAGE must be off, step or stage, got '", s,
              "'");
  return Fp32StorageConfig::kOff;
}

std::vector<std::string> parse_fp32_storage_rows(std::string const& s) {
  std::vector<std::string> out;
  auto v = lower_trim(s);
  if (v.empty() || v == "all") return out;
  std::stringstream ss(v);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(tok);
  }
  return out;
}

std::vector<int64_t> resolve_fp32_storage_rows(
    std::vector<std::string> const& rows, int64_t nvar) {
  std::vector<int64_t> out;
  auto add = [&](int64_t r) {
    TORCH_CHECK(r >= 0 && r < nvar, "SNAP_FP32_STORAGE_ROWS: row ", r,
                " is outside a state of ", nvar, " rows");
    if (std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
  };
  for (auto const& t : rows) {
    if (t == "idn") {
      add(IDN);
    } else if (t == "ivx") {
      add(IVX);
    } else if (t == "ivy") {
      add(IVY);
    } else if (t == "ivz") {
      add(IVZ);
    } else if (t == "ipr") {
      add(IPR);
    } else if (t == "icy") {
      for (int64_t r = ICY; r < nvar; ++r) add(r);
    } else if (t.rfind("icy+", 0) == 0) {
      add(ICY + std::stoll(t.substr(4)));
    } else {
      char* end = nullptr;
      long long r = std::strtoll(t.c_str(), &end, 10);
      TORCH_CHECK(end && *end == '\0', "SNAP_FP32_STORAGE_ROWS: unknown row '",
                  t, "'");
      add(r);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

Fp32StorageConfig const& fp32_storage_config() {
  // read once: every block and rank must round the same rows
  static const Fp32StorageConfig cfg = [] {
    Fp32StorageConfig c;
    c.mode = parse_fp32_storage_mode(env_or("SNAP_FP32_STORAGE", "off"));
    c.rows = parse_fp32_storage_rows(env_or("SNAP_FP32_STORAGE_ROWS", "all"));
    return c;
  }();
  return cfg;
}

bool fp32_storage_due(Fp32StorageConfig const& cfg, int stage, int nstages) {
  if (cfg.mode == Fp32StorageConfig::kStage) return true;
  if (cfg.mode == Fp32StorageConfig::kStep) return stage == nstages - 1;
  return false;
}

void round_through_fp32_(torch::Tensor u, std::vector<int64_t> const& rows) {
  if (!u.defined() || u.numel() == 0) return;
  if (u.scalar_type() == torch::kFloat32) return;
  TORCH_CHECK(u.is_floating_point(), "round_through_fp32_: not a float tensor");
  if (rows.empty()) {
    u.copy_(u.to(torch::kFloat32));
    return;
  }
  for (auto r : rows) {
    auto row = u[r];
    row.copy_(row.to(torch::kFloat32));
  }
}

}  // namespace snap
