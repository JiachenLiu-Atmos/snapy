#pragma once

// C/C++
#include <array>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

namespace harp {
struct IntegratorWeight;
}  // namespace harp

namespace snap {

//! V3 lane R, PR R-1: the flux register and its monitor, with no reader
//! (docs/v3/design/R.rst sections 3 and 4; skeleton R/flux_register.hpp).
//! A post is a flux density per unit face area and time at the step dt,
//! before stage weights. R-1 holds the stage totals T_d (hydro's _flux1/2/3
//! are these tensors), the requested step fields S_d and the monitor; the
//! posts (slot, add, scale, replace, apply_prescribed, keep, cell_amount)
//! arrive with R-2 to R-7.

struct RegisterRequest {
  std::vector<std::string> row_names;  //! nvar conserved rows, in row order
  std::vector<std::string> counters;   //! int64 count() names
  std::vector<std::string> minima;     //! record_min() names
  std::vector<std::string> retry;      //! flag_retry() causes
  std::vector<int> step_field;         //! dims 1-3 whose S_d is read
};

//! one per register instance; O reads after accept, never posts
class RegisterMonitor {
 public:
  void reset(RegisterRequest const& req);

  //! zero the counters, minima and retry flags (X: step start)
  void begin_step();
  //! drop a rejected attempt: the census counts accepted steps only
  void rollback() { begin_step(); }

  void count(std::string const& name, torch::Tensor const& n);  // += n
  void record_min(std::string const& name, torch::Tensor const& v);
  void flag_retry(std::string const& cause, torch::Tensor const& any);  // OR

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
  std::vector<std::string> _rows, _counter, _minimum, _retry;
};

class FluxRegister {
 public:
  //! allocate T_d for the active directions (dims 1, 2, 3 are x1, x2, x3),
  //! shape {nvar, nc3, nc2, nc1}, dtype kFlux, zero; mark the step fields of
  //! req.step_field
  void reset(int nvar, std::array<int64_t, 3> const& cells,
             std::array<bool, 3> const& active, torch::Dtype kflux,
             RegisterRequest const& req);

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
  //! S_d = sum over stages of w2_s prod_{t>s} w1_t dt T_s, per unit area
  torch::Tensor step_field(int dim) const { return _step[dim - 1]; }
  bool has_step_field(int dim) const { return _step[dim - 1].defined(); }

  //! S_d <- w1_s S_d + w2_s dt T_d for every requested direction
  void end_stage(double dt);
  //! X's decide_retry: zero the step fields and the monitor
  void rollback();

  RegisterMonitor& monitor() { return _monitor; }
  RegisterMonitor const& monitor() const { return _monitor; }

  //! the tensors HydroImpl registers as buffers, so module moves reach them
  std::array<torch::Tensor, 3>& totals() { return _total; }

 private:
  std::array<torch::Tensor, 3> _total, _step;
  std::array<bool, 3> _want_step = {false, false, false};
  RegisterMonitor _monitor;
  int _stage = -1;
  double _w1 = 0., _w2 = 0.;
};

}  // namespace snap
