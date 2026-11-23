#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/eos/ef_status.hpp>
#include <snap/utils/step_dt.hpp>

namespace snap {

//! Round-off, in ulp of the working precision, for the species mass the
//! positivity machinery withholds or repairs. flux_positivity_theta already
//! leaves this fraction of a cell's own species mass behind on purpose; a
//! withheld or repaired mass below this fraction of the cell's TOTAL gas mass
//! (rho_total * volume, all species) is reported as round-off, not as a
//! positivity event: it neither counts as severe nor marks a limiter patch
//! for MeshBlock::check_redo. The clip or repair itself is always applied.
constexpr double kPositivityRoundoffUlp = 4096.;

//! The same bound in float32, in float32 ulp. 4096 of them is 4.9e-4 of the
//! gas mass, the size of a real repair; a species update rounds by at most
//! 1 ulp of the gas mass. The margin of flux_positivity_theta stays 4096 ulp.
constexpr double kPositivityRoundoffUlpFloat = 64.;

//! A species deficit (a negative conserved density before the EOS species
//! repair), in ulp of the cell's TOTAL gas density, above which the repair
//! marks a limiter patch for MeshBlock::check_redo; a smaller one is repaired
//! without a redo. It defaults to the round-off bound above; its final value
//! is to be set from the redo logs of the TJ16 species deck.
constexpr double kSpeciesRepairMarkUlp = kPositivityRoundoffUlp;

//! The same mark in float32, in float32 ulp
constexpr double kSpeciesRepairMarkUlpFloat = kPositivityRoundoffUlpFloat;

//! machine epsilon of `dtype` (float32 for kFloat, float64 otherwise)
double machine_epsilon(c10::ScalarType dtype);

//! kPositivityRoundoffUlp (kPositivityRoundoffUlpFloat in float32) times the
//! machine epsilon of `dtype`
double positivity_roundoff(c10::ScalarType dtype);

//! kSpeciesRepairMarkUlp (kSpeciesRepairMarkUlpFloat in float32) times the
//! machine epsilon of `dtype`
double species_repair_mark(c10::ScalarType dtype);

//! \brief Per-cell positivity limiter factors for donor-form tracer fluxes:
//! theta_i = min(1, u_i V_i / (dt out_i)), out_i = sum_faces max(+/- A F, 0)
//! over the faces the divergence uses, the largest uniform scaling of cell i's
//! outgoing fluxes that cannot drain it below zero in one Euler step of dt.
//! Every shipped integrator is a convex combination of such steps, so the
//! stages stay non-negative; each face takes its donor's theta, so
//! conservation is exact. Ghosts get 1: the caller fills them like conserved
//! ghosts (flux_positivity_sync_theta_) before any face reads them.
//! \param u      conserved tracer densities, (nchan, nc3, nc2, nc1)
//! \param flux1/2/3  channel fluxes through each cell's lower face; undefined
//!               tensors are skipped
//! \param drain  if given, receives dt * out_i (species mass); the limiter
//!               withholds (1 - theta_i) * drain_i of it
torch::Tensor flux_positivity_theta(torch::Tensor const& u,
                                    torch::Tensor const& flux1,
                                    torch::Tensor const& flux2,
                                    torch::Tensor const& flux3,
                                    Coordinate const& pcoord, StepDt dt,
                                    torch::Tensor* drain = nullptr);

//! \brief phi = clamp(min of the neighbours' q along every resolved dimension
//! / q, 0, 1), 0 where q <= 0: the share of a relaxed drain the neighbours
//! repay. \param q  ratio to the dry density, (nchan, nc3, nc2, nc1)
torch::Tensor flux_positivity_share(torch::Tensor const& q);

//! \brief The withhold factor (P2): the donor's theta on the faces lo..hi+1
//! along tensor axis `dim` (3: x1), 1 elsewhere; the donor is the lower cell
//! where the flux is positive.
torch::Tensor flux_positivity_factor(torch::Tensor const& theta,
                                     torch::Tensor const& flux, int dim,
                                     Coordinate const& pcoord);

class FluxRegister;
//! \brief P3, before the withhold: add(d, "positivity.carry") of -dm hspec to
//! {IPR} and -dm vel to {IVX, IVY, IVZ}, dm = (1 - theta_donor) F_species read
//! from the pre-scale reg.total(d). With fsed1 (x1 settling part, EF's P1
//! buffer) the advected part F - fsed1 posts first, then fsed1; each carries
//! hspec and vel of its own donor. vface2/3: per x2 (x3) face, the lower [0]
//! and upper [1] donor velocity in the face's flux basis (#370); undefined:
//! vel.
void flux_positivity_carry_(torch::Tensor const& theta,
                            torch::Tensor const& hspec,
                            torch::Tensor const& vel, FluxRegister& reg,
                            Coordinate const& pcoord,
                            torch::Tensor const& fsed1 = torch::Tensor(),
                            torch::Tensor const& vface2 = torch::Tensor(),
                            torch::Tensor const& vface3 = torch::Tensor());

//! \brief Where 0 < td < 1 (td: theta of the mass flux) theta moves toward
//! min(theta / td, 1) by the share phi; elsewhere it is unchanged.
torch::Tensor flux_positivity_relax(torch::Tensor const& theta,
                                    torch::Tensor const& td,
                                    torch::Tensor const& phi);

class MeshBlockImpl;
//! \brief Fill theta's ghosts in place like a conserved variable's: a raw
//! copy at internal seams (never interpolated), then the boundary functions.
void flux_positivity_sync_theta_(MeshBlockImpl const* pmb,
                                 torch::Tensor const& theta,
                                 std::string const& key);

//! What one limiter call needs (views only)
struct PositivityInputs {
  torch::Tensor u;        //!< channel densities (nchan, nc3, nc2, nc1)
  torch::Tensor rho_dry;  //!< total dry density (1, nc3, nc2, nc1)
  torch::Tensor flux[3];  //!< R1: the channel rows of reg.total(d)
  torch::Tensor mass[3];  //!< the hydro total(d)'s dry-mass row (1, ...)
  torch::Tensor fsed1;    //!< R2: the settling part (x1); undefined: scalars
  torch::Tensor hspec;    //!< energy per unit species mass; undefined: no carry
  torch::Tensor vel;      //!< donor momentum per unit mass, with hspec
  torch::Tensor vface[2];  //!< x2/x3 seam donor velocity per face (#370)
  int first = 0;           //!< the register row of channel 0
  StepDt dt;
};

//! \brief One driver for the hydro species (step 4.C) and the scalars (D7).
class PositivityLimiter {
 public:
  //! hits: kPosHits (hydro: hits, severe, minimum, x1 cut) or kScalarPosHits
  PositivityLimiter(MeshBlockImpl const* pmb, EfStatus* status,
                    EfStatus::Slot hits);
  //! theta relaxed where the dry mass drains too (phi 0 where settling drains)
  torch::Tensor theta(PositivityInputs const& in,
                      torch::Tensor* drain = nullptr) const;
  //! the census of the interior cells, before the ghost fill
  void census_(PositivityInputs const& in, torch::Tensor const& theta,
               torch::Tensor const& drain) const;
  //! the ghost fill of theta (key_theta) and hspec (key_hspec)
  void sync_(PositivityInputs const& in, torch::Tensor const& theta,
             std::string const& key) const;
  //! the carry (when hspec is defined), then scale(d, withhold, rows, factor)
  void post_(FluxRegister& reg, PositivityInputs const& in,
             torch::Tensor const& theta, char const* withhold) const;
  //! theta, census_, sync_, post_
  void apply(FluxRegister& reg, PositivityInputs const& in,
             std::string const& key, char const* withhold) const;

 private:
  MeshBlockImpl const* pmb_;
  EfStatus* status_;
  EfStatus::Slot hits_;
};

}  // namespace snap
