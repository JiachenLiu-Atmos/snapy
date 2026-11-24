#!/usr/bin/env python3
"""V3: the block's EfStatus meters under cuda-graph equal eager, bitwise.

The capture of a graph segment runs the segment eagerly three times (the
warm-ups) before it records it, and restores the state the StageExecutor saved
(StageExecutor::save_list) after them. A meter the segment adds to that is not
on the save list keeps the warm-ups' additions: every capture counts the
segment four times.

The moist column of test_graph_chemistry_slot.py (H2O, H2O(l)) with
flux-covariance on, so the tracer flux-positivity limiter
(flux_positivity.hpp) runs inside the captured hydro forward (the EOS limiter,
the other switch, keeps a case eager) and adds to the block's EfStatus meters
at every stage: hydro.positivity_hits, positivity_severe, positivity_min,
lim_cut and lim_flux (and hydro.conversion_trips if the deck has one), and the
scalar's positivity_hits where the scalar has one. A cuda-graph block (not
strict) and an eager block step the same dt schedule (1.0, then 0.5, a rebind
that recaptures, an in-place edit) in FP64 and FP32. After every step, from
the first (the capture) on, every meter, check_redo (the limiter marks and the
vapor-failure count), hydro_u, hydro_w, scalar_s and scalar_r must match
bitwise, and so must the budget's cycle line (print_cycle_info: limcut,
thetamin, thetasevere). The case does something: the eager lim_flux is
positive after step 1, the eager positivity_hits and lim_cut (and in FP64
positivity_severe) are positive at the end, and the graph block captured and
replayed with no eager segment. On CPU the cuda-graph setting runs eager and
must equal eager.

RED at 939f7928: after the first step (three captures) the graph block's
positivity_hits, positivity_severe, positivity_min and lim_flux differ from
eager's; the warm-ups of each capture are counted, and they ran on states past
the stage's.

  python test_graph_positivity_meters_match_eager.py
"""
import ctypes
import os
import sys
import tempfile

import torch

from snapy import kIPR

from cuda_build import build_has_cuda, skip
from test_graph_chemistry_slot import (assert_equal, build, config,
                                       forward_both, moist_state)

METERS = ("positivity_hits", "positivity_severe", "positivity_min",
          "lim_cut", "lim_flux", "conversion_trips")


def make(precision, device, execution="eager"):
    cfg = config(precision, execution)
    cfg["dynamics"]["flux-covariance"] = True  # runs the positivity limiter
    cfg["integration"]["ncycle_out"] = 1  # the budget's cycle line, every cycle
    block = build(cfg, device)
    return block, moist_state(block)


def meters(block):
    out = {}
    for module in ("hydro", "scalar"):
        bufs = dict(block.module(module).named_buffers())
        for name in METERS:
            if name in bufs:
                out[module + "." + name] = bufs[name].clone()
    return out


LIBC = ctypes.CDLL(None)


def cycle_line(block, state, dt):
    """print_cycle_info's line, read from the C++ stdout (fd 1)."""
    sys.stdout.flush()
    LIBC.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+") as f:
        os.dup2(f.fileno(), 1)
        try:
            block.print_cycle_info(state, 0.0, dt)
            LIBC.fflush(None)  # std::cout shares the C stdout buffer
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        return f.read()


def assert_meters(pair, dt, where):
    (block, state), (ref, expected) = pair
    mg, me = meters(block), meters(ref)
    assert mg.keys() == me.keys(), (where, mg.keys(), me.keys())
    bad = ["%s graph %s eager %s" % (k, mg[k].tolist(), me[k].tolist())
           for k in mg if not torch.equal(mg[k], me[k])]
    assert not bad, (where, bad)
    lg, le = cycle_line(block, state, dt), cycle_line(ref, expected, dt)
    assert lg == le, (where, "budget line", lg, le)
    assert "limcut=" in le, (where, "the budget line has no limcut", le)
    return me


def run(precision, device, nsteps):
    pair = (make(precision, device, "cuda-graph"), make(precision, device))
    (block, state), (ref, expected) = pair
    for step in range(nsteps):
        dt = 0.5 if nsteps // 3 <= step < 2 * nsteps // 3 else 1.0
        if step == nsteps - 3:  # a rebind between steps recaptures
            for fields in (state, expected):
                for k in ("hydro_u", "scalar_s"):
                    if k in fields:
                        fields[k] = fields[k].clone()
        if step == nsteps - 2:  # an in-place edit reaches the captured inputs
            for fields in (state, expected):
                fields["hydro_u"][kIPR, 0, 4, 10] += 1.0e-4
        forward_both(pair, dt)
        ra, rb = block.check_redo(state), ref.check_redo(expected)
        assert ra == rb == 0, (precision, step, "check_redo", ra, rb)
        assert_equal(state, expected, (precision, step))
        me = assert_meters(pair, dt, (precision, device.type, step))
        if step == 0:
            assert me["hydro.lim_flux"].item() > 0, (
                precision, "the limiter was offered no flux; the case is vacuous", me)
            print(precision, device.type, "eager meters after step 1:",
                  {k: v.tolist() for k, v in me.items()})
    # positivity_severe is above FP32 round-off only in FP64 here
    nonzero = ["hydro.positivity_hits", "hydro.lim_cut"]
    if precision == "float64":
        nonzero.append("hydro.positivity_severe")
    for k in nonzero:
        assert me[k].item() > 0, (precision, "vacuous: eager %s is zero" % k, me)
    print(precision, device.type, "eager meters at the end:",
          {k: v.tolist() for k, v in me.items()})
    return block.execution_report()


def main():
    torch.set_num_threads(1)
    for precision in ("float64", "float32"):
        report = run(precision, torch.device("cpu"), 3)
        assert report["captures"] == 0 and report["replays"] == 0, report
        print(precision, "CPU: cuda-graph runs eager and its meters equal eager:", report)
    if not build_has_cuda():
        skip("CUDA unavailable (graph part)")
    failed = []
    for precision in ("float64", "float32"):
        try:
            report = run(precision, torch.device("cuda"), 9)
            assert report["captures"] > 0 and report["replays"] > 0, report
            assert report["eager"] == 0, report
            print(precision, "graph meters and budget line equal eager:", report)
        except (AssertionError, RuntimeError) as e:
            failed.append("%s: %s" % (precision, str(e).strip()))
    for f in failed:
        print("FAIL", f)
    print("PASSED" if not failed else "FAILED (%d)" % len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
