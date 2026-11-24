#pragma once

// C/C++
#include <array>
#include <cstdint>
#include <map>
#include <memory>
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
  //! drop a rejected attempt: the census counts accepted steps only. The
  //! device slots are zeroed as at begin_step; the host stage bookkeeping
  //! (_stage, _ended, _w1, _w2dt) stays: on the cuda-graph path the redo
  //! replays the captured stages, whose host begin_stage/end_stage do not
  //! run again, and fold_ would otherwise read S = 0 (no rates, no later
  //! stages' posts) for the redo and every replayed step after it; eager
  //! rewrites them at the redo's own begin_stage/end_stage (V3 S4)
  void rollback();

  //! += n. A reduced count (0-d) adds at once; a mask or a vector is copied
  //! into its slot of the stage's fixed count buffer and reduced with the
  //! others by flush_counts() (V3 S4: one segment sum per stage, not one
  //! per mask)
  void count(std::string const& name, torch::Tensor const& n);
  //! the stage's count buffer into the counters: one integer cumsum, one
  //! gather of the segment ends, one add, then the buffer is zeroed (no
  //! atomics; V3 S4); called by end_stage, inside the captured stage, and
  //! always the same launches once the buffer exists, so a capture of the
  //! combine replays the reduction over what the replayed advance wrote
  void flush_counts();
  //! the count buffer, for the stage executor's save list: a warm-up of the
  //! combine reduces and zeroes it, and the capture must read what the
  //! advance wrote (empty before the first mask of a stage)
  torch::Tensor count_storage() const { return _pend_buf; }
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
  //! book by amount index (WallBatch::flush)
  void book_index(int i, torch::Tensor const& rate);
  //! R-5: amount (nrows,) times weight into this stage's posts (R.rst 3.4);
  //! in_stage or after the combine, the post is scaled by the later stages'
  //! w1 alone, so both land in the stage's post slot. The name is created
  //! at its first post with that kind; another kind is an error.
  void post(std::string const& name, AmountKind kind,
            torch::Tensor const& amount, double weight, bool in_stage);
  //! V3 S4 (ruling 2026-11-24): the stages' rates a_s and posts p_s stay in
  //! per-stage slots and are folded once, on the device, when the amounts
  //! are read: A = sum_s c_s (w2_s dt a_s + p_s), c_s = prod_{t>s} w1_t.
  //! end_stage records the stage's w1 and w2 dt; it launches nothing.
  void end_stage(double w1, StepDt w2dt);
  //! the slots are sized for the integrator's stages (the executor's save
  //! list is taken after this; a later stage index grows them, eager only)
  void set_stage_count(int nstage);
  //! the slot the next book() / post() lands in (FluxRegister::begin_stage)
  void begin_stage(int stage);
  //! the amount tensors (stage rates [nstage, namounts, nrows], stage posts
  //! [nstage, namounts, nrows], the folded step amounts [namounts, nrows]),
  //! wherever they are; empty before reset_amounts. Every stage writes the
  //! slots, so the stage executor's save list holds them (X: warm-up and
  //! capture)
  std::vector<torch::Tensor> amount_storage() const;
  //! move them to `device` now, as begin_step would (a copy, which syncs:
  //! the executor calls it in its sync-exempt prepare step, never in a step)
  void place_amount_storage(torch::Device device);

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
  //! [nstage, namounts, nrows]: the stages' booked rates a_s and weighted
  //! posts p_s, placed on the register's device by FluxRegister::begin_step
  //! or the first book()/post(); _step_amount [namounts, nrows] holds the
  //! fold of the stages ended so far, written by amounts()
  torch::Tensor _rate, _post;
  mutable torch::Tensor _step_amount;
  //! the stage's count masks in fixed device storage: one int64 buffer
  //! behind a leading zero element, one slot per post in the stage's order
  //! (the pattern: counter index and numel per slot, grown at a new post,
  //! eager). A host queue of the masks would be drained by the combine's
  //! first warm-up and never refilled by a replayed advance, so the captured
  //! combine would count nothing (V3 S4). The segment ends [2, ncounters,
  //! dup] into the cumulative sum are cached per pattern
  torch::Tensor _pend_buf;
  std::vector<int64_t> _pend_pattern;
  std::vector<int64_t> _pend_off;
  size_t _pend_cursor = 0;
  torch::Tensor _pend_ends;
  int64_t _pend_dup = 1;
  //! slot k of the stage holds (counter, numel): grow or rebuild the buffer
  //! and the segment ends when it is new or differs (eager, never captured)
  void pend_slot_(size_t k, int counter, int64_t numel,
                  torch::Tensor const& like);
  int _stage = 0, _ended = 0;  // the current slot; stages ended this step
  std::vector<double> _w1;     // per stage, recorded at end_stage
  std::vector<StepDt> _w2dt;   // per stage, recorded at end_stage
  void place_amounts(torch::Device device);
  void grow_slots_(int nstage);
  void zero_attempt_();  // the device slots and pending masks, not the stages
  torch::Tensor fold_() const;
  friend class FluxRegister;
};

//! V3 S4: the prescribed-wall rates of one stage, hydro and scalar
//! registers, x1 to x3, booked in one batch at the stage's end
//! (MeshBlock::stage_combine, before end_stage): the gathered wall planes
//! sit in one fixed device buffer, one slot per register and direction;
//! the flush is one product with the cached -area weights, one segment sum
//! over the rows, one add per register, then the buffer is zeroed. Fixed
//! storage, not a host queue of the planes: a user forcing splits the stage
//! into a captured advance and a captured combine, the combine's first
//! warm-up would drain a queue, its capture would then record no flush and
//! its replays never run the host push again, so the wall rates would be
//! lost under graph while the state stays equal. A register without a
//! batch books its rate at the call (tests).
class WallBatch {
 public:
  //! planes_flat: the gathered wall planes of one register and direction,
  //! [nvar, faces] flattened, copied into their slot; the slot is registered
  //! at its first push (eager, before any capture) and the batch sums each
  //! of the nvar rows at the flush
  void push(torch::Tensor const& planes_flat, torch::Tensor const& weights_flat,
            RegisterMonitor* monitor, int amount_index, int64_t nvar);
  //! book every slot's rate and zero the buffer: always the same launches
  //! once a slot exists, so eager and a captured combine do the same
  void flush();
  //! zero the buffer (a rejected attempt's planes)
  void clear();
  //! the stage's pushes start at the first slot (MeshBlock::stage_advance:
  //! every pass of the advance, the capture's warm-ups included, runs it)
  void begin_stage() { _cursor = 0; }
  bool empty() const { return _pattern.empty(); }
  //! the buffer, for the stage executor's save list: a warm-up of the
  //! combine books and zeroes it, and the capture must read what the
  //! advance wrote (empty before the first push)
  torch::Tensor storage() const { return _buf; }

 private:
  //! per slot, in the stage's push order: the monitor and amount index it
  //! books, the weights it was registered with, rows (nvar), numel and the
  //! offset into the buffer; _cursor is the next slot of this stage
  std::vector<RegisterMonitor*> _monitor;
  std::vector<int> _index;
  std::vector<torch::Tensor> _weights;
  std::vector<int64_t> _pattern;  // per slot: rows, numel
  std::vector<int64_t> _off;
  size_t _cursor = 0;
  //! the slots, then one zero element the dense rows are padded from
  torch::Tensor _buf;
  //! per layout: the padded weights [rows, M] and the gather index [rows, M]
  //! into the buffer; a dense row sum is deterministic, which a scatter-add
  //! (index_add_) on CUDA is not, and graph must equal eager bitwise
  torch::Tensor _w, _gidx;
  int64_t _rows = 0, _M = 0;
  //! the next slot of the stage, (monitor, amount_index, nvar, numel),
  //! registered when new (eager, never captured)
  size_t slot_(torch::Tensor const& planes_flat,
               torch::Tensor const& weights_flat, RegisterMonitor* monitor,
               int amount_index, int64_t nvar);
  void layout_(torch::TensorOptions const& opts);
};

class FluxRegister {
 public:
  //! the stage's wall batch, shared by the block's registers (MeshBlock)
  void set_wall_batch(std::shared_ptr<WallBatch> batch) {
    _wall_batch = std::move(batch);
  }
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
  //! add, scale or replace wrote total(dim) since the last
  //! apply_prescribed(dim) without keeping the prescribed faces: the
  //! post-implicit apply re-prescribes only then, or after the implicit
  //! correction, which writes total(1) directly (V3 S4)
  bool written_since_prescribed(int dim) const { return _written[dim - 1]; }
  //! the name of dim's last posted slot ("" before the first post)
  std::string const& slot_name(int dim) const { return _slot[dim - 1]; }
  //! a requested part, written in place by its producer (C's kernel)
  torch::Tensor part_slot(int dim, std::string const& part);
  bool keeps(int dim, std::string const& part) const;
  torch::Tensor kept(int dim, std::string const& part) const;
  //! R-3: total[rows] += increment, in call order and never fused, so
  //! successive adds round as the pin's in-place ones; increment is
  //! [rows.size(), faces...]. A requested part of the same name gets a copy.
  //! keeps_prescribed: the caller leaves the prescribed faces at their
  //! prescribed value (S hands them as 0 on the prescribed rows and the
  //! positivity carry adds -0 there), so the write does not mark the
  //! direction for the post-implicit apply (V3 S4)
  void add(int dim, std::string const& name, std::vector<int> const& rows,
           torch::Tensor const& increment, bool keeps_prescribed = false);
  //! total[rows] *= factor in place; factor [faces...] (every row) or
  //! [rows.size(), faces...]; keeps_prescribed as for add (0 * factor = 0)
  void scale(int dim, std::string const& name, std::vector<int> const& rows,
             torch::Tensor const& factor, bool keeps_prescribed = false);
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
  //! V3 S4: the wall planes as an index tensor, the face areas of those
  //! planes and -1 on the prescribed rows (0 on the free rows), wide dtype on
  //! the total's device; built at the first apply_wall_planes_ after
  //! set_prescribed_walls, so a stage launches no setup
  std::array<torch::Tensor, 3> _pidx, _parea, _pneg, _pwt;
  //! V3 S4: the -area weights laid out like the gathered planes,
  //! flattened, for the batch
  std::array<torch::Tensor, 3> _pwtflat;
  std::shared_ptr<WallBatch> _wall_batch;
  std::array<bool, 3> _pwall_set = {false, false, false};
  std::array<bool, 3> _written = {false, false, false};
  //! apply_prescribed on the wall planes: as overwrite_, plane by plane
  void apply_wall_planes_(int dim);
  CoordinateImpl const* _coord = nullptr;
  //! total(dim) <- after; books sum A_f (after - before) per row as name
  void overwrite_(int dim, std::string const& name, torch::Tensor const& after);
  std::array<bool, 3> _want_step = {false, false, false};
  //! the step fields, split posts and stage flag of a fresh attempt: the
  //! shared part of begin_step and rollback
  void begin_attempt_();
  RegisterMonitor _monitor;
  int _stage = -1;
  double _w1 = 0., _w2 = 0.;
  bool _in_stage = false;  //! between begin_stage and end_stage
};

}  // namespace snap
