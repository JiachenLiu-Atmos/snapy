#pragma once

// C/C++
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// torch
#include <torch/nn/module.h>

// snap
#include <snap/utils/step_dt.hpp>

namespace snap {

class FluxRegister;

//! before the implicit solve, or after it with the stage's x1 mass transfer
//! (#355); the two RegisterPhase values sources use (A.rst 3.6)
enum class SourcePhase : uint8_t { pre_implicit, post_implicit };
enum class SourceKind : uint8_t { physical, numerical, fixer };
enum SourceRows : uint32_t {
  kMass = 1,
  kSpecies = 2,
  kMomentum = 4,
  kEnergy = 8
};
//! a bottom/top source runs only on a block with that physical x1 boundary
enum class SourceRegion : uint8_t { interior, bottom, top };

//! what a source declares at construction (A.rst 3.6)
struct SourceDeclaration {
  std::string name;  //!< deck key under forcing:, e.g. "top-sponge-lyr"
  SourcePhase phase = SourcePhase::pre_implicit;
  bool split = false;  //!< operator-split around the step (radial conduction)
  SourceKind kind = SourceKind::physical;
  uint32_t rows = 0;  //!< SourceRows bit set
  SourceRegion region = SourceRegion::interior;
  bool capturable = true;  //!< false: host reads, an eager segment (N14)
};

//! read-only stage view; wc: the Coriolis input (rho, m1, rho v2, rho v3)
//! under the x1 mass-flux gate, m1 = 0 before the solve (#355)
struct StageState {
  torch::Tensor w, temp;
  StepDt dt;  // V3 D: a captured stage's is on the device
  torch::Tensor wc;
};

//! one source term; each module is also a torch module of its Hydro
class SourceTerm {
 public:
  virtual ~SourceTerm() = default;
  SourceDeclaration const& declaration() const { return decl_; }
  //! add the increment over s.dt to du, writing each element at most once
  virtual void add(torch::Tensor du, StageState const& s) = 0;
  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt) {
    add(du, {w, temp, dt});
    return du;
  }
  //! divergence phase, its first entry (ruling A-ph3.diffusion_phase):
  //! after limit and before du = -dt div(total), post face fluxes into the
  //! register; s.temp may be undefined
  virtual void post_fluxes(FluxRegister& reg, StageState const& s) {}
  virtual double max_time_step(StageState const&) const {
    return std::numeric_limits<double>::infinity();
  }
  //! split sources: before / after the stages, and on a redo
  virtual void begin_split(torch::Tensor u, double dt) {}
  virtual void finish_split(torch::Tensor u, double dt) {}
  //! the same, run by MeshBlock: a split source posts its step transfer with
  //! reg.post_split (R.rst 3.4)
  virtual void begin_split(torch::Tensor u, double dt, FluxRegister& reg) {
    begin_split(u, dt);
  }
  virtual void finish_split(torch::Tensor u, double dt, FluxRegister& reg) {
    finish_split(u, dt);
  }
  virtual void restore_split(torch::Tensor u) const {}

 protected:
  SourceDeclaration decl_;
};

//! the sources of one Hydro, run by phase in declaration order
class SourceRegistry {
 public:
  using List = std::vector<std::shared_ptr<SourceTerm>>;
  //! budgets on: each source's increment, by name, per stage
  using AmountSink = std::function<void(std::string const&, torch::Tensor)>;
  struct Run {
    SourcePhase phase;
    bool capturable;
    size_t begin, end;
  };
  //! set by Hydro: whether this block has a region's physical x1 boundary
  std::function<bool(SourceRegion)> on_block;

  void add(std::shared_ptr<SourceTerm> s, SourcePhase p) { at(p).push_back(s); }
  void add(std::shared_ptr<SourceTerm> s) { add(s, s->declaration().phase); }
  bool empty() const { return at(kPre).empty() && at(kPost).empty(); }
  //! maximal runs of equal `capturable`; never reorders (A.rst 3.6)
  std::vector<Run> runs(SourcePhase p) const {
    std::vector<Run> out;
    for (size_t i = 0; i < at(p).size(); ++i) {
      bool c = at(p)[i]->declaration().capturable;
      if (out.empty() || out.back().capturable != c) out.push_back({p, c, i});
      out.back().end = i + 1;
    }
    return out;
  }
  //! with a sink, each source writes into one FP64 scratch, zeroed per
  //! source, which is then added to du once (N17)
  void run(Run const& r, torch::Tensor du, StageState const& s,
           AmountSink const& sink = nullptr) {
    for (size_t i = r.begin; i < r.end; ++i) {
      auto const& src = at(r.phase)[i];
      if (on_block && !on_block(src->declaration().region)) continue;
      if (!sink) {
        src->add(du, s);
        continue;
      }
      if (!scratch_.defined() || !scratch_.sizes().equals(du.sizes()) ||
          scratch_.device() != du.device())
        scratch_ = torch::empty_like(du, du.options().dtype(torch::kFloat64));
      src->add(scratch_.zero_(), s);
      sink(src->declaration().name, scratch_);
      du += scratch_;
    }
  }
  //! the divergence-phase posts of the pre_implicit sources, in order
  void post_fluxes(FluxRegister& reg, StageState const& s) {
    for (auto const& src : at(kPre))
      if (!on_block || on_block(src->declaration().region))
        src->post_fluxes(reg, s);
  }
  double max_time_step(StageState const& s) const {
    double dt = std::numeric_limits<double>::infinity();
    for (auto const& src : at(kPre)) dt = std::min(dt, src->max_time_step(s));
    return dt;
  }
  //! the split sources (pre_implicit), in order
  List splits() const {
    List out;
    std::copy_if(at(kPre).begin(), at(kPre).end(), std::back_inserter(out),
                 [](auto const& s) { return s->declaration().split; });
    return out;
  }

 private:
  static constexpr auto kPre = SourcePhase::pre_implicit;
  static constexpr auto kPost = SourcePhase::post_implicit;
  List& at(SourcePhase p) { return lists_[static_cast<int>(p)]; }
  List const& at(SourcePhase p) const { return lists_[static_cast<int>(p)]; }
  List lists_[2];
  torch::Tensor scratch_;  //!< one FP64 field, budgets on only (N17)
};

}  // namespace snap
