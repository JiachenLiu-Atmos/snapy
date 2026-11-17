// V3 lane A skeleton: source-term interface (A.rst 3.6, 5.1, 7.1). Declarations
// only; "assumed" types and names marked † are provisional (A.rst 1.5).
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snap::v3 {

class FluxRegister;      // assumed †, lane R (requirements 1-10)
class RotationGeometry;  // lane S (S-1), see rotation_geometry.hpp
class CoordinateImpl;    // lane S

//! pre_implicit (default) | post_implicit (m1 rows, #355) | split (around stage)
enum class SourcePhase : uint8_t { pre_implicit, post_implicit, split };

//! physical: sponge, drag, heating, relaxation, viscosity (kept);
//! numerical: a discretisation torque; fixer: global correction (closure)
enum class SourceKind : uint8_t { physical, numerical, fixer };

//! conserved rows a source may touch; a write outside them is a debug error
enum SourceRows : uint32_t { kMass = 1, kSpecies = 2, kMomentum = 4, kEnergy = 8 };

//! where a source acts; the base does the physical-boundary test
struct SourceRegion {
  enum class Where : uint8_t { interior, bottom, top } where = Where::interior;
  int depth = 0; double thickness = 0.0;  //!< cells, or x1 length (#280)
};

//! what a source declares at construction; read by the monitor and the graph
struct SourceDeclaration {
  std::string name;  //!< deck key, e.g. "top-sponge-lyr"
  SourcePhase phase = SourcePhase::pre_implicit;
  SourceKind kind = SourceKind::physical;
  uint32_t rows = 0;  //!< SourceRows bit set
  SourceRegion region;
  bool capturable = true;  //!< false: X runs it in an eager segment
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
  torch::Tensor m1;  //!< post_implicit only: x1-transfer momentum (A.rst 3.4)
};

//! one source term. Owned by the SourceRegistry of a Hydro; no back pointer.
class SourceTerm {
 public:
  virtual ~SourceTerm() = default;
  virtual SourceDeclaration const& declaration() const = 0;  //!< fixed
  //! add the increment over dt to du, writing each element at most once; on
  //! monitor steps du is the registry's FP64 scratch, added to du once (N17)
  virtual void add(torch::Tensor du, StageState const& s) = 0;
  //! largest stable dt this source allows; +inf if none (diffusion)
  virtual double max_time_step(StageState const& s) const;
  //! split sources only: before / after the stage, and on a redo
  virtual void begin_split(torch::Tensor u, double dt);
  virtual void finish_split(torch::Tensor u, double dt);
  virtual void restore_split(torch::Tensor u) const;
};

//! per-source, per-stage budget the monitor derives from the increment dS:
//! host view of budget_ after replay (N6); aam = sum V r K . dS[IVY, IVZ]
struct SourceBudget {
  std::string name;
  SourceKind kind;
  torch::Tensor totals;  //!< FP64 device [mass, m1, m2, m3, energy, aam]
};

//! owns the sources of one Hydro and runs them by phase
class SourceRegistry {
 public:
  void add(std::unique_ptr<SourceTerm> s);
  //! ordered runs of equal `capturable` in a phase; X maps each to a segment
  struct Run { SourcePhase phase; bool capturable; size_t begin, end; };
  std::vector<Run> runs(SourcePhase phase) const;
  //! budgets on (monitor cadence): each source's reductions go to budget_
  void run(Run const& r, torch::Tensor du, StageState const& s, bool budgets);
  double max_time_step(StageState const& s) const;
  std::vector<SourceDeclaration> declarations() const;
 private:
  std::vector<std::unique_ptr<SourceTerm>> sources_;
  torch::Tensor scratch_;  //!< one FP64 field, allocated at reset (N17)
  torch::Tensor budget_;   //!< [nsrc, 6] FP64 device, read at step end (N6)
};

}  // namespace snap::v3
