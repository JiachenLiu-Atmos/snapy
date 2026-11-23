// snap
#include "segment.hpp"

namespace snap {

char const* to_string(SegmentKind kind) {
  switch (kind) {
    case SegmentKind::compute:
      return "compute";
    case SegmentKind::pack:
      return "pack";
    case SegmentKind::host_exchange:
      return "host-exchange";
    case SegmentKind::unpack:
      return "unpack";
    case SegmentKind::global:
      return "global";
    case SegmentKind::eager:
      return "eager";
  }
  return "?";
}

bool runs_on_host(SegmentKind kind) {
  return kind == SegmentKind::host_exchange || kind == SegmentKind::global;
}

namespace {
// host_exchange, global and eager ops never run inside a graph
bool always_eager(SegmentKind kind) {
  return kind == SegmentKind::host_exchange || kind == SegmentKind::global ||
         kind == SegmentKind::eager;
}
// host_exchange and global ops are a segment each
bool own_segment(SegmentKind kind) {
  return kind == SegmentKind::host_exchange || kind == SegmentKind::global;
}
}  // namespace

void StageSchedule::add(Op op, SegmentKind kind) {
  TORCH_CHECK(!op.name.empty(), "StageSchedule: an op needs a name");
  TORCH_CHECK(static_cast<bool>(op.run), "StageSchedule: op '", op.name,
              "' has no run");
  if (always_eager(kind) && op.capturable) {
    op.capturable = false;
    if (op.eager_reason.empty())
      op.eager_reason = std::string("a ") + to_string(kind) + " op";
  }
  TORCH_CHECK(op.capturable || !op.eager_reason.empty(), "StageSchedule: op '",
              op.name, "' is eager without a reason");
  _ops.push_back(std::move(op));
  _kinds.push_back(kind);
  regroup();
}

void StageSchedule::regroup() {
  _segments.clear();
  for (int i = 0; i < static_cast<int>(_ops.size()); ++i) {
    auto const& op = _ops[i];
    auto kind = _kinds[i];
    bool start = _segments.empty() || own_segment(kind) ||
                 own_segment(_segments.back().kind) ||
                 _segments.back().kind != kind ||
                 _segments.back().capturable != op.capturable;
    if (start) {
      Segment s;
      s.kind = kind;
      s.capturable = op.capturable;
      _segments.push_back(s);
    }
    auto& s = _segments.back();
    s.ops.push_back(i);
    s.names.push_back(op.name);
    s.reads_host_dt = s.reads_host_dt || op.reads_host_dt;
    if (!op.capturable) s.eager_reasons.push_back(op.eager_reason);
  }
}

}  // namespace snap
