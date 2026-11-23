#pragma once

// C/C++
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/utils/step_dt.hpp>

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

//! V3 lane A (A2), the segment boundary agreed in the graph thread: the kinds
//! of a stage's segments. host_exchange and global are always eager, each in
//! a segment of its own; eager is an op that cannot be captured.
enum class SegmentKind : uint8_t {
  compute,
  pack,
  host_exchange,
  unpack,
  global,
  eager
};
char const* to_string(SegmentKind kind);
//! host_exchange and global: a segment that runs on the host by design (a
//! segment boundary), never a refusal: strict does not refuse it and it is
//! not a reason for eager
bool runs_on_host(SegmentKind kind);

//! what an op's run sees besides the variables
struct StageContext {
  int block = 0;
  int stage = 0;
  //! the step's dt: the block's device dt in a captured segment (V3 D), the
  //! host dt in an eager one or one with an op that sets reads_host_dt
  StepDt dt{0.};
};

//! one device op of a stage, declared by its owner
struct Op {
  std::string name;
  //! false: the op runs eagerly, and eager_reason says why (required)
  bool capturable = true;
  std::string eager_reason;
  //! an op that reads the host dt (StepDt::value) even when its segment is
  //! captured. A segment holding one captures with the host dt and keeps dt
  //! in its graph key, so a new dt recaptures; all others read the device dt
  bool reads_host_dt = false;
  //! runs outside capture, before the save list and the warm-up, with the
  //! state's device, dtype and shape (may be empty)
  std::function<void(torch::Device, torch::Dtype, at::IntArrayRef)> prepare;
  std::function<void(Variables&, StageContext const&)> run;
};

//! a run of ops of one kind and one capturability
struct Segment {
  SegmentKind kind = SegmentKind::compute;
  bool capturable = true;
  bool reads_host_dt = false;      //!< some op reads the host dt
  std::vector<int> ops;            //!< indices into StageSchedule::ops()
  std::vector<std::string> names;  //!< the ops' names, in order
  std::vector<std::string> eager_reasons;  //!< the eager ops' reasons
};

//! the ordered ops of one stage and the segments they form
class StageSchedule {
 public:
  //! appends op (checked: an eager op names its reason, a run is set)
  void add(Op op, SegmentKind kind = SegmentKind::compute);

  std::vector<Op> const& ops() const { return _ops; }
  std::vector<SegmentKind> const& kinds() const { return _kinds; }
  //! consecutive ops of the same kind and capturability share a segment;
  //! host_exchange and global ops are eager and each start their own
  std::vector<Segment> const& segments() const { return _segments; }

 private:
  void regroup();

  std::vector<Op> _ops;
  std::vector<SegmentKind> _kinds;
  std::vector<Segment> _segments;
};

}  // namespace snap
