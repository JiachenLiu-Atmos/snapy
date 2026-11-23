#pragma once

// C/C++
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/utils/step_dt.hpp>

namespace harp {
struct IntegratorWeight;
}  // namespace harp

namespace snap {

class CoordinateImpl;
class PrecisionPolicy;

//! V3 lane R, PR R-1: the flux register and its monitor, with no reader
//! (docs/v3/design/R.rst sections 3 and 4; skeleton R/flux_register.hpp).
//! A post is a flux density per unit face area and time at the step dt,
//! before stage weights. R-1 holds the stage totals T_d (hydro's _flux1/2/3
//! are these tensors), the requested step fields S_d and the monitor; R-2
//! the slots (slot, post_slot, part_slot, kept); R-3 add and scale; R-4
//! replace, set_prescribed/apply_prescribed and the monitor's amounts for
//! them; R-5 keep and cell_amount (lane I's posts arrive with I-4); the
//! rest (derived rows, phases, Fixer) with R-6 and R-7.

//! S rows 9 and 16r: both sides of an x1 seam call mean; on a panel seam
//! only the non-owner calls owner_copy
enum class SeamRule : uint8_t { mean, owner_copy };

//! O's R-O3: how the budget treats an amount
enum class AmountKind : uint8_t {
  flux_correction,  //! a face change, already inside boundary(d): report only
  cell_source,      //! volume source (A, K, vic.cell): the budget adds it
  floor,            //! cell floor or repair (EF); C face floors: count() only
  fixer,            //! global Fixer (A's closure): the budget adds it
  diagnostic,       //! changes no row (chem.elements, vic.momentum_t.top)
};

struct RegisterRequest {
  std::vector<std::string> row_names;  //! nvar conserved rows, in row order
  std::vector<std::string> counters;   //! int64 count() names
  std::vector<std::string> minima;     //! record_min() names
  std::vector<std::string> retry;      //! flag_retry() causes
  std::vector<int> step_field;         //! dims 1-3 whose S_d is read
  //! kept parts "part@dim" (e.g. "riemann.central@2"), allocated at reset()
  //! like T_d and valid from their post to the next begin_stage
  std::vector<std::string> keep;
};

//! one per register instance; O reads after accept, never posts
class RegisterMonitor {
 public:
  void reset(RegisterRequest const& req);

  //! zero the counters, minima, retry flags and amounts (X: step start)
  void begin_step();
  //! drop a rejected attempt: the census counts accepted steps only
  void rollback() { begin_step(); }

  void count(std::string const& name, torch::Tensor const& n);  // += n
  void record_min(std::string const& name, torch::Tensor const& v);
  void flag_retry(std::string const& cause, torch::Tensor const& any);  // OR

  //! R-4: the amounts, FP64 (FP32 on MPS), one (nrows,) per name, in
  //! amount_names() order: "reconcile.x1", "reconcile.panel" and
  //! "boundary.prescribed", all AmountKind::flux_correction, then R-5's
  //! cell_amount names in the order of their first post
  void reset_amounts(int nrows);
  std::vector<std::string> const& amount_names() const { return _amount; }
  AmountKind kind(std::string const& name) const;
  int amount_index(std::string const& name) const;
  //! the step amount, sum over stages of w2_s prod_{t>s} w1_t dt a_s
  torch::Tensor amounts(std::string const& name) const;
  //! a_s += rate, (nrows,): an area-integrated face change of this stage
  void book(std::string const& name, torch::Tensor const& rate);
  //! R-5: amount (nrows,) times weight, into this stage's posts when
  //! in_stage (folded at end_stage, R.rst 3.4), else straight into the step
  //! amount (after_combine: only later stages' w1 scale it). The name is
  //! created at its first post with that kind; another kind is an error.
  void post(std::string const& name, AmountKind kind,
            torch::Tensor const& amount, double weight, bool in_stage);
  //! A <- w1 A + w2 dt a_s + p_s, then a_s, p_s <- 0 (FluxRegister::end_stage)
  void end_stage(double w1, StepDt w2dt);
  //! the amount tensors (step amounts, stage rates, stage posts), placed on
  //! `device` first; empty before reset_amounts. Every stage writes them, so
  //! the stage executor's save list holds them (X: warm-up and capture)
  std::vector<torch::Tensor> amount_storage(torch::Device device);

  std::vector<std::string> const& row_names() const { return _rows; }
  std::vector<std::string> const& counter_names() const { return _counter; }
  std::vector<std::string> const& minimum_names() const { return _minimum; }
  int counter_index(std::string const& name) const;
  int retry_index(std::string const& name) const;

  //! the tensors, on the register's device; int64, FP64 (+inf when unset),
  //! int32 0/1 per cause. HydroImpl registers them as buffers.
  torch::Tensor counters, minima, retry_flags;

 private:
  std::vector<std::string> _rows, _counter, _minimum, _retry, _amount;
  std::vector<AmountKind> _kind;
  //! [namounts, nrows]: the step amounts, this stage's booked rates a_s and
  //! weighted posts p_s, placed on the register's device by
  //! FluxRegister::begin_step or the first book()/post()
  torch::Tensor _step_amount, _stage_amount, _stage_post;
  void place_amounts(torch::Device device);
  friend class FluxRegister;
};

class FluxRegister {
 public:
  //! allocate T_d for the active directions (dims 1, 2, 3 are x1, x2, x3),
  //! shape {nvar, nc3, nc2, nc1}, dtype kFlux, zero; mark the step fields of
  //! req.step_field
  void reset(int nvar, std::array<int64_t, 3> const& cells,
             std::array<bool, 3> const& active, torch::Dtype kflux,
             RegisterRequest const& req);
  //! R-4: the same, with cells and active directions from coord's options
  //! and kFlux from policy; coord gives the face areas A_f that replace and
  //! apply_prescribed book with, and must outlive the register
  void reset(CoordinateImpl const* coord, PrecisionPolicy const& policy,
             int nvar, RegisterRequest const& req);

  //! a reader asks for S_d; it exists from the next begin_step()
  void request_step_field(int dim);
  //! X, step start: zero the step fields and the monitor. S_d lives on
  //! T_d's device in FP64 (FP32 on MPS), allocated here only when T_d moved.
  void begin_step();
  //! X, before stage s: its integrator weights
  void begin_stage(int stage, harp::IntegratorWeight const& w);
  double weight2() const { return _w2; }
  int stage() const { return _stage; }

  //! the stage total T_d, written in place by its producers (kFlux)
  torch::Tensor total(int dim) const { return _total[dim - 1]; }

  //! the first post of a stage (C's "riemann"): T_d itself, written in place
  //! by the producer, which then calls post_slot with the same name
  torch::Tensor slot(int dim, std::string const& name);
  void post_slot(int dim, std::string const& name);
  //! the name of dim's last posted slot ("" before the first post)
  std::string const& slot_name(int dim) const { return _slot[dim - 1]; }
  //! a requested part, written in place by its producer (C's kernel)
  torch::Tensor part_slot(int dim, std::string const& part);
  bool keeps(int dim, std::string const& part) const;
  torch::Tensor kept(int dim, std::string const& part) const;
  //! R-3: total[rows] += increment, in call order and never fused, so
  //! successive adds round as the pin's in-place ones; increment is
  //! [rows.size(), faces...]. A requested part of the same name gets a copy.
  void add(int dim, std::string const& name, std::vector<int> const& rows,
           torch::Tensor const& increment);
  //! total[rows] *= factor in place; factor [faces...] (every row) or
  //! [rows.size(), faces...]
  void scale(int dim, std::string const& name, std::vector<int> const& rows,
             torch::Tensor const& factor);
  //! R-4, a shared face (S): faces is a bool mask shaped like total(dim)
  //! without the row axis; peer is dense, shaped like total(dim) but with
  //! nvar * (1 + kept_parts(dim).size()) rows: the total's rows, then each
  //! kept part's, in kept_parts(dim) order, already mapped by S into this
  //! face's dim, edge order, outward sign and basis. On the masked faces
  //! mean writes (own + peer) * 0.5 (the same on both sides) and owner_copy
  //! writes peer. Books sum A_f (after - before) per row of the total as
  //! "reconcile.x1" (dim 1) or "reconcile.panel" (dims 2, 3).
  void replace(int dim, torch::Tensor const& faces, torch::Tensor const& peer,
               SeamRule rule);
  //! the kept parts of dim, by part name, in RegisterRequest::keep order
  std::vector<std::string> const& kept_parts(int dim) const {
    return _kept[dim - 1];
  }
  //! S: mask bool [nvar, faces...] and value [nvar, faces...], held as given
  void set_prescribed(int dim, torch::Tensor const& mask,
                      torch::Tensor const& value);
  //! V3 E: S's prescription without a solid. The face planes `planes`
  //! (indices along dim's face axis: the wall block faces) of every row but
  //! free_rows are set to 0, and apply_prescribed works on those planes
  //! alone. It replaces a set_prescribed mask, and set_prescribed replaces
  //! it. No planes: nothing in dim is prescribed
  void set_prescribed_walls(int dim, std::vector<int64_t> const& planes,
                            std::vector<int> const& free_rows);
  //! total[mask] = value and, for each kept part of dim, part[mask] = value;
  //! books sum A_f mask (value - total before) per row of the total as
  //! "boundary.prescribed". No-op when S set no mask for dim.
  void apply_prescribed(int dim);
  //! R-5: a computed one-quantity part ("vic.mass_total", "mass.dry.limited"):
  //! value [faces...] goes to row IDN of the requested nvar-row part; the
  //! other rows stay zero
  void keep(int dim, std::string const& part, torch::Tensor const& value);
  //! R-5: vol_du [nrows, cells...] (summed) or (nrows,), already times the
  //! cell volume; stage_weight is w2_s for this stage's du, 1.0 for a change
  //! already in u (R.rst 4.2 I)
  void cell_amount(std::string const& name, torch::Tensor const& vol_du,
                   double stage_weight, AmountKind kind);
  //! the parts, by "part@dim", for HydroImpl to register as buffers
  std::map<std::string, torch::Tensor>& parts() { return _parts; }
  //! S_d = direct + sum over stages of w2_s prod_{t>s} w1_t dt T_s, per
  //! unit area; direct holds this step's post_split entries
  torch::Tensor step_field(int dim) const { return _step[dim - 1]; }
  bool has_step_field(int dim) const { return _step[dim - 1].defined(); }

  //! S_d <- w1_s S_d + w2_s dt T_d for every requested direction; the
  //! monitor's amounts likewise
  void end_stage(StepDt dt);
  //! X's decide_retry: zero the step fields and the monitor
  void rollback();
  //! R.rst 3.4, outside the stages (the radial split halves: after
  //! begin_step and before stage 0, and after the last end_stage): S_d gets
  //! weight dt flux in its direct part, which no later w1 scales. flux is
  //! shaped like total(dim); total(dim) is untouched. dim must have a step
  //! field; begin_step clears the direct part.
  void post_split(int dim, std::string const& name, torch::Tensor const& flux,
                  double weight, double dt);

  RegisterMonitor& monitor() { return _monitor; }
  RegisterMonitor const& monitor() const { return _monitor; }

  //! the tensors HydroImpl registers as buffers, so module moves reach them
  std::array<torch::Tensor, 3>& totals() { return _total; }

 private:
  std::array<torch::Tensor, 3> _total, _step;
  //! from a direction's first post_split of the step: S_d's stage recursion
  //! and its direct part, S_d = _rec + _direct; undefined otherwise, when
  //! S_d is the recursion itself
  std::array<torch::Tensor, 3> _rec, _direct;
  std::array<std::string, 3> _slot;
  std::map<std::string, torch::Tensor> _parts;
  std::array<std::vector<std::string>, 3> _kept;
  std::array<torch::Tensor, 3> _pmask, _pvalue;
  //! V3 E: set_prescribed_walls' planes and free rows per direction, and its
  //! row mask ([nvar, 1, 1] bool on the total's device, kept between calls)
  std::array<std::vector<int64_t>, 3> _pwall;
  std::array<std::vector<int>, 3> _pfree;
  std::array<torch::Tensor, 3> _prow;
  std::array<bool, 3> _pwall_set = {false, false, false};
  //! apply_prescribed on the wall planes: as overwrite_, plane by plane
  void apply_wall_planes_(int dim);
  CoordinateImpl const* _coord = nullptr;
  //! total(dim) <- after; books sum A_f (after - before) per row as name
  void overwrite_(int dim, std::string const& name, torch::Tensor const& after);
  std::array<bool, 3> _want_step = {false, false, false};
  RegisterMonitor _monitor;
  int _stage = -1;
  double _w1 = 0., _w2 = 0.;
  bool _in_stage = false;  //! between begin_stage and end_stage
};

}  // namespace snap
