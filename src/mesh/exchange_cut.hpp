#pragma once

// C/C++
#include <functional>

namespace snap {

//! V3 B3: a cut at an exchange, the one helper the in-stage cross-block
//! exchanges use (a Mesh's block seams, the cubed-sphere panel seams; P2's
//! x1 relays later). An exchange is pack | host | unpack: the pack (device
//! copies into fixed send buffers) and the unpack (fixed receive buffers into
//! the ghosts) are device work; the host part is the same-process rendezvous
//! with its seam copies, or the communicator's sends and receives.
//!
//! On a thread capturing a segment (StageExecutor::capture), exchange_cut
//! ends the segment's current graph, runs `host` and keeps it, then captures
//! what follows into a new graph of the same segment: a replay runs the
//! graphs in order with the kept host steps between them. Anywhere else
//! (eager, warm-up) it runs `host`. `host` must read and write only storage
//! that outlives the capture (an exchange's cached buffers), since it runs
//! again at every replay.
//!
//! The blocks of one process meet at the rendezvous, so a multi-block Mesh
//! captures a segment with cuts on all its blocks together (each on its
//! worker thread, thread-local capture mode) and replays them together.
void exchange_cut(std::function<void()> host);

//! true on a thread capturing a segment, outside the host step of a cut: a
//! cross-block exchange there must go through exchange_cut (a board or
//! communicator call that does not throws, naming itself)
bool in_segment_capture();

//! throws when called inside a segment capture: `what` is a cross-block
//! exchange that has no cut yet
void refuse_in_segment_capture(char const* what);

}  // namespace snap
