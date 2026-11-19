#pragma once

// C/C++
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

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
//! them; the other posts (keep, cell_amount) arrive with R-5 to R-7.

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
  //! "boundary.prescribed", all AmountKind::flux_correction
  void reset_amounts(int nrows);
  std::vector<std::string> const& amount_names() const { return _amount; }
  AmountKind kind(std::string const& name) const;
  int amount_index(std::string const& name) const;
  //! the step amount, sum over stages of w2_s prod_{t>s} w1_t dt a_s
  torch::Tensor amounts(std::string const& name) const;
  //! a_s += rate, (nrows,): an area-integrated face change of this stage
  void book(std::string const& name, torch::Tensor const& rate);
  //! A <- w1 A + w2 dt a_s, then a_s <- 0 (FluxRegister::end_stage)
  void end_stage(double w1, double w2dt);

  std::vector<std::string> const& row_names() const { return _rows; }
  std::vector<std::string> const& counter_names() const { return _counter; }
  std::vector<std::string> const& minimum_names() const { return _minimum; }
  std::vector<std::string> const& retry_names() const { return _retry; }
  int counter_index(std::string const& name) const;
  int minimum_index(std::string const& name) const;
  int retry_index(std::string const& name) const;

  //! the tensors, on the register's device; int64, FP64 (+inf when unset),
  //! int32 0/1 per cause. HydroImpl registers them as buffers.
  torch::Tensor counters, minima, retry_flags;

 private:
  std::vector<std::string> _rows, _counter, _minimum, _retry, _amount;
  std::vector<AmountKind> _kind;
  //! [namounts, nrows]: the step amounts and this stage's a_s, placed on
  //! the register's device by FluxRegister::begin_step or the first book()
  torch::Tensor _step_amount, _stage_amount;
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
  double weight1() const { return _w1; }
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
  //! total[mask] = value and, for each kept part of dim, part[mask] = value;
  //! books sum A_f mask (value - total before) per row of the total as
  //! "boundary.prescribed". No-op when S set no mask for dim.
  void apply_prescribed(int dim);
  //! the parts, by "part@dim", for HydroImpl to register as buffers
  std::map<std::string, torch::Tensor>& parts() { return _parts; }
  //! S_d = sum over stages of w2_s prod_{t>s} w1_t dt T_s, per unit area
  torch::Tensor step_field(int dim) const { return _step[dim - 1]; }
  bool has_step_field(int dim) const { return _step[dim - 1].defined(); }

  //! S_d <- w1_s S_d + w2_s dt T_d for every requested direction; the
  //! monitor's amounts likewise
  void end_stage(double dt);
  //! X's decide_retry: zero the step fields and the monitor
  void rollback();

  RegisterMonitor& monitor() { return _monitor; }
  RegisterMonitor const& monitor() const { return _monitor; }

  //! the tensors HydroImpl registers as buffers, so module moves reach them
  std::array<torch::Tensor, 3>& totals() { return _total; }

 private:
  std::array<torch::Tensor, 3> _total, _step;
  std::array<std::string, 3> _slot;
  std::map<std::string, torch::Tensor> _parts;
  std::array<std::vector<std::string>, 3> _kept;
  std::array<torch::Tensor, 3> _pmask, _pvalue;
  CoordinateImpl const* _coord = nullptr;
  //! total(dim) <- after; books sum A_f (after - before) per row as name
  void overwrite_(int dim, std::string const& name, torch::Tensor const& after);
  std::array<bool, 3> _want_step = {false, false, false};
  RegisterMonitor _monitor;
  int _stage = -1;
  double _w1 = 0., _w2 = 0.;
};

}  // namespace snap
