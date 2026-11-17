// V3 lane A skeleton: one source-term interface (A.rst 8). Declarations
// only; types marked "assumed" belong to other lanes.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snap::v3 {

class FluxRegister;      // assumed, register owner (kickoff 2/3; Q K-2)
class RotationGeometry;  // lane S, see rotation_geometry.hpp
class CoordinateImpl;    // lane S

//! when in the stage a source adds its increment (#355 needs two phases)
//! pre_implicit (default) | post_implicit (m1 rows) | split (around stage)
enum class SourcePhase : uint8_t { pre_implicit, post_implicit, split };

//! physical: sponge, drag, heating, relaxation, viscosity (kept);
//! numerical: a discretisation torque; fixer: global correction (closure)
enum class SourceKind : uint8_t { physical, numerical, fixer };

//! conserved rows a source may touch; a write outside them is a debug error
enum SourceRows : uint32_t { kMass = 1, kSpecies = 2, kMomentum = 4, kEnergy = 8 };

//! where a source acts; the base does the physical-boundary test
struct SourceRegion {
  enum class Where : uint8_t { interior, bottom, top } where = Where::interior;
  int depth = 0;           //!< cells, used when thickness == 0
  double thickness = 0.0;  //!< x1 length (#280); overlap-weighted cells
};

//! what a source declares at construction; read by the monitor and the graph
struct SourceDeclaration {
  std::string name;  //!< deck key, e.g. "top-sponge-lyr"
  SourcePhase phase = SourcePhase::pre_implicit;
  SourceKind kind = SourceKind::physical;
  uint32_t rows = 0;  //!< SourceRows bit set
  SourceRegion region;
  bool moves_ke_to_ie = false;  //!< momentum changed, energy row not (closure)
};

//! read-only stage view; replaces (du, w, temp, dt) and phydro back pointers
struct StageState {
  torch::Tensor w;     //!< primitive state, this block
  torch::Tensor temp;  //!< W->T, computed once per stage by Hydro
  double dt = 0.0;
  int stage = 0;
  CoordinateImpl const* coord = nullptr;    //!< non-owning
  RotationGeometry const* rot = nullptr;    //!< non-owning; null: no axis
  FluxRegister const* fluxes = nullptr;     //!< non-owning; this stage's
  torch::Tensor m1;  //!< post_implicit only: x1-transfer momentum (A.rst 6.1)
};

//! one source term. Owned by the SourceRegistry of a Hydro; no back pointer.
class SourceTerm {
 public:
  virtual ~SourceTerm() = default;
  virtual SourceDeclaration const& declaration() const = 0;  //!< fixed
  //! add the increment over dt to du (conserved rows, covariant momentum);
  //! the registry passes a scratch tensor instead when budgets are on
  virtual void add(torch::Tensor du, StageState const& s) = 0;
  //! largest stable dt this source allows; +inf if none (diffusion)
  virtual double max_time_step(StageState const& s) const;
  //! split sources only: before / after the stage, and on a redo
  virtual void begin_split(torch::Tensor u, double dt);
  virtual void finish_split(torch::Tensor u, double dt);
  virtual void restore_split(torch::Tensor u) const;
};

//! per-source, per-stage budget the monitor derives from the increment dS:
//! sum V dS over rows; aam = sum V r K . dS[IVY, IVZ] (contravariant K)
struct SourceBudget {
  std::string name;
  SourceKind kind;
  torch::Tensor totals;  //!< FP64 device [mass, m1, m2, m3, energy, aam]
};

//! owns the sources of one Hydro and runs them by phase
class SourceRegistry {
 public:
  void add(std::unique_ptr<SourceTerm> s);
  //! run `phase` in declaration order; budgets on: one SourceBudget each
  void run(SourcePhase phase, torch::Tensor du, StageState const& s,
           std::vector<SourceBudget>* out = nullptr);
  double max_time_step(StageState const& s) const;
  std::vector<SourceDeclaration> declarations() const;
 private:
  std::vector<std::unique_ptr<SourceTerm>> sources_;
  torch::Tensor scratch_;  //!< reused dS buffer when budgets are on
};

}  // namespace snap::v3
