#pragma once

// C/C++
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>

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

//! what an op's run sees besides the variables
struct StageContext {
  int block = 0;
  int stage = 0;
  double dt = 0.;  //!< host dt; a captured op that reads it sets reads_host_dt
};

//! one device op of a stage, declared by its owner
struct Op {
  std::string name;
  //! false: the op runs eagerly, and eager_reason says why (required)
  bool capturable = true;
  std::string eager_reason;
  //! until dt is a graph input (C4/D): an op that reads the host dt. A
  //! segment holding one keeps dt in its graph key, so a new dt recaptures
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
