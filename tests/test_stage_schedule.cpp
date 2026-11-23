// external
#include <gtest/gtest.h>

// snap
#include <snap/mesh/segment.hpp>

using namespace snap;

// V3 A2: the segment types of the agreed graph interface. A stage's ops form
// segments: consecutive ops of one kind and one capturability share a
// segment; host-exchange and global ops are eager and alone; a segment
// holding an op that reads the host dt says so (its graph key keeps dt).
namespace {

Op op(std::string name, bool capturable = true, bool host_dt = false) {
  Op o;
  o.name = std::move(name);
  o.capturable = capturable;
  if (!capturable) o.eager_reason = o.name + " reads the host";
  o.reads_host_dt = host_dt;
  o.run = [](Variables&, StageContext const&) {};
  return o;
}

std::vector<std::vector<std::string>> names(StageSchedule const& s) {
  std::vector<std::vector<std::string>> out;
  for (auto const& seg : s.segments()) out.push_back(seg.names);
  return out;
}

}  // namespace

TEST(StageSchedule, one_capturable_run_is_one_segment) {
  StageSchedule s;
  s.add(op("exchange"));
  s.add(op("advance_local", true, true));
  ASSERT_EQ(s.segments().size(), 1u);
  auto const& seg = s.segments()[0];
  EXPECT_EQ(seg.kind, SegmentKind::compute);
  EXPECT_TRUE(seg.capturable);
  EXPECT_TRUE(seg.reads_host_dt) << "advance_local reads the host dt";
  EXPECT_EQ(seg.ops, (std::vector<int>{0, 1}));
  EXPECT_TRUE(seg.eager_reasons.empty());
}

TEST(StageSchedule, capturability_and_kind_changes_cut_segments) {
  StageSchedule s;
  s.add(op("a"));
  s.add(op("b"));
  s.add(op("c", false));
  s.add(op("d", false));
  s.add(op("e"));
  s.add(op("pack"), SegmentKind::pack);
  s.add(op("f"));
  EXPECT_EQ(names(s), (std::vector<std::vector<std::string>>{
                          {"a", "b"}, {"c", "d"}, {"e"}, {"pack"}, {"f"}}));
  EXPECT_FALSE(s.segments()[1].capturable);
  EXPECT_EQ(s.segments()[1].eager_reasons,
            (std::vector<std::string>{"c reads the host", "d reads the host"}));
  EXPECT_FALSE(s.segments()[0].reads_host_dt);
}

TEST(StageSchedule, host_exchange_and_global_are_eager_and_alone) {
  StageSchedule s;
  s.add(op("pack"), SegmentKind::pack);
  s.add(op("ucx"), SegmentKind::host_exchange);
  s.add(op("ucx2"), SegmentKind::host_exchange);
  s.add(op("unpack"), SegmentKind::unpack);
  s.add(op("sums"), SegmentKind::global);
  s.add(op("allreduce"), SegmentKind::global);
  EXPECT_EQ(
      names(s),
      (std::vector<std::vector<std::string>>{
          {"pack"}, {"ucx"}, {"ucx2"}, {"unpack"}, {"sums"}, {"allreduce"}}));
  for (int i : {1, 2, 4, 5}) {
    EXPECT_FALSE(s.segments()[i].capturable) << i;
    EXPECT_EQ(s.segments()[i].eager_reasons.size(), 1u) << i;
  }
  EXPECT_TRUE(s.segments()[0].capturable);
  EXPECT_TRUE(s.segments()[3].capturable);
}

TEST(StageSchedule, an_op_without_run_or_reason_is_refused) {
  StageSchedule s;
  Op none;
  none.name = "x";
  EXPECT_THROW(s.add(none), c10::Error);
  auto eager = op("y");
  eager.capturable = false;  // no reason given
  EXPECT_THROW(s.add(eager), c10::Error);
  Op unnamed = op("z");
  unnamed.name.clear();
  EXPECT_THROW(s.add(unnamed), c10::Error);
  EXPECT_TRUE(s.segments().empty());
}

TEST(StageSchedule, host_dt_marks_only_its_segment) {
  StageSchedule s;
  s.add(op("a"));
  s.add(op("b", false));
  s.add(op("c", true, true));
  ASSERT_EQ(s.segments().size(), 3u);
  EXPECT_FALSE(s.segments()[0].reads_host_dt);
  EXPECT_FALSE(s.segments()[1].reads_host_dt);
  EXPECT_TRUE(s.segments()[2].reads_host_dt);
}
