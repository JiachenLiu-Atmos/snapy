// V3 lane A skeleton: source-term interface (A.rst 3.6, 5.1, 7.1). Declarations
// only. Register types are lane R's header, happysky19/snapy@543ff84f
// docs/v3/skeleton/R/flux_register.hpp (cited "R:line").
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flux_register.hpp"  // lane R: FluxRegister, RegisterPhase, Fixer

namespace snap {

class RotationGeometry;  // lane S (S-1), see rotation_geometry.hpp

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
  std::string name;  //!< deck key, e.g. "top-sponge-lyr"; the cell_amount name
  //! R:43-51; sources use pre_implicit or post_implicit (m1 rows, #355)
  RegisterPhase phase = RegisterPhase::pre_implicit;
  bool split = false;  //!< operator-split around the step (radial conduction)
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
  double dt = 0.0;     //!< host double, read at step start (N14; R:113-115)
  int stage = 0;
  CoordinateImpl const* coord = nullptr;    //!< non-owning
  RotationGeometry const* rot = nullptr;    //!< non-owning; null: no axis
  FluxRegister const* reg = nullptr;        //!< non-owning; reads total, kept
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
  virtual void begin_split(torch::Tensor u, double dt, FluxRegister& reg);
  virtual void finish_split(torch::Tensor u, double dt, FluxRegister& reg);
  virtual void restore_split(torch::Tensor u) const;
};

//! owns the sources of one Hydro and runs them by phase
class SourceRegistry {
 public:
  void add(std::unique_ptr<SourceTerm> s);
  //! ordered runs of equal `capturable` in a phase; X maps each to a segment
  struct Run { RegisterPhase phase; bool capturable; size_t begin, end; };
  std::vector<Run> runs(RegisterPhase phase) const;
  //! budgets on (monitor cadence): each source's V * scratch goes to
  //! reg.cell_amount(name, vol_du) (R:83); R tracks AAM by name (R-3)
  void run(Run const& r, torch::Tensor du, StageState const& s,
           FluxRegister& reg, bool budgets);
  double max_time_step(StageState const& s) const;
  std::vector<SourceDeclaration> declarations() const;
 private:
  std::vector<std::unique_ptr<SourceTerm>> sources_;
  torch::Tensor scratch_;  //!< one FP64 field, allocated at reset (N17)
};

}  // namespace snap
