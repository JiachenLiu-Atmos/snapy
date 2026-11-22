#!/usr/bin/env python3
"""V3 lane X, C / G3 (X.rst section 6, row kImplicit; section 10, row
"vertical implicit correction"): the vertical implicit correction runs inside
the hydro forward's compute segment and captures with it. The FP64 retry of
the failed columns is a predicated second launch, always run, with no host
read (X.rst:296, 1030-1032, 1043-1045 at 882a3ab3), so it is part of the same
captured segment.

Two decks, each strict cuda-graph against eager, bitwise, every forward() of
the graph block under torch.cuda.set_sync_debug_mode("error"):

  moist   the 2D moist column of test_graph_chemistry_slot.py with
          implicit-scheme 9, FP64 and FP32 (FP32: the FP32 solve with
          implicit-fallback-fp64, so the predicated FP64 launch runs every
          stage), a dt change, a rebind, an in-place edit and a forced retry
  retry   one column, FP32, zero gravity at a very large dt: the FP32 solve
          fails in the near-null acoustic mode and the FP64 launch recovers
          it (the case of test_fp32_implicit.py), so the retried columns are
          nonempty; the solution, the column status and the FP64-column
          meters must match eager

After every step hydro_u, hydro_w, scalar_s and scalar_r, check_redo, the
icorr's fp64_fallbacks and the hydro monitor buffers must match bitwise. On
CPU the cuda-graph setting runs eager and must equal eager.

RED at dc0e2571: capture_reasons() lists "implicit-scheme is not 0", so a
strict cuda-graph block with the implicit correction is refused at its first
launch.

  python test_graph_implicit.py
"""
import sys

import torch

from snapy import kIDN, kIPR, kIV1
from test_graph_chemistry_slot import (assert_equal, build, config,
                                       forward_both, make_case, run_schedule)

SCHEME = 9


def meters(block):
    out = {"fp64_fallbacks": block.module("hydro.icorr").buffer("fp64_fallbacks")}
    for name, t in block.named_buffers():
        if name.startswith("hydro.monitor_"):
            out[name] = t
    return {k: v.clone() for k, v in out.items()}


def assert_meters(a, b, where):
    ma, mb = meters(a), meters(b)
    assert ma.keys() == mb.keys(), (where, ma.keys(), mb.keys())
    for k in ma:
        assert torch.equal(ma[k], mb[k]), (where, k, ma[k], mb[k])


def check_cpu(precision):
    cpu = torch.device("cpu")
    pair = (make_case(precision, cpu, "cuda-graph", implicit=SCHEME),
            make_case(precision, cpu, implicit=SCHEME))
    run_schedule(pair, precision, nsteps=3)
    assert_meters(pair[0][0], pair[1][0], (precision, "cpu"))
    report = pair[0][0].execution_report()
    assert report["captures"] == 0 and report["replays"] == 0, report
    print(precision, "CPU: cuda-graph runs eager and equals eager:", report)


def check_moist(precision):
    cuda = torch.device("cuda")
    pair = (make_case(precision, cuda, "cuda-graph", True, implicit=SCHEME),
            make_case(precision, cuda, implicit=SCHEME))
    (block, state), (ref, expected) = pair
    nstages = len(block.intg.stages)
    nsteps = 12
    keys = run_schedule(pair, precision, nsteps, sync_error=True,
                        extra=lambda step: assert_meters(block, ref, (precision, step)))

    # a forced retry restores the step; the next dt recaptures
    before = {k: state[k].clone() for k in ("hydro_u", "scalar_s") if k in state}
    forward_both(pair, 1.0, sync_error=True)
    for fields in (state, expected):
        fields["hydro_u"][kIPR, 0, 4, 10] = float("nan")
    assert block.check_redo(state) == ref.check_redo(expected) != 0
    for k, v in before.items():
        assert torch.equal(state[k], v), (precision, "restore", k)
        assert torch.equal(expected[k], v), (precision, "restore eager", k)
    for dt in (0.5, 0.5):
        forward_both(pair, dt, sync_error=True)
        assert block.check_redo(state) == ref.check_redo(expected) == 0
        assert_equal(state, expected, (precision, "after retry"))
        assert_meters(block, ref, (precision, "after retry"))
    keys += 1
    # a dt change keys a capture only while the graph segment reads the host
    # dt; with the device dt (C4) only the first launch and the rebind capture
    host_dt = any(r["reads_host_dt"] for r in block.execution_capability(state)["segments"]
                  if r["capturable"])
    if not host_dt:
        keys = 2

    report = block.execution_report()
    assert report["eager"] == 0, report
    assert report["captures"] == nstages * keys, (report, keys)
    assert report["replays"] == nstages * (nsteps + 3), report
    print(precision, "implicit graph equals eager over %d steps:" % (nsteps + 3), report)


def retry_case(device, execution="eager", strict=False):
    cfg = config("float32", execution, strict, implicit=SCHEME, grav1=0.0,
                 one_column=True)
    cfg["integration"]["implicit-precision"] = "float32"
    cfg["integration"]["implicit-fallback-fp64"] = True
    block = build(cfg, device)
    w = block.module("hydro").buffer("D").clone().zero_()
    z = block.module("coord").buffer("x1v")
    w[kIDN].fill_(1.0)
    w[kIPR].fill_(1.0e5)
    w[kIV1] = 1.0e-10 * torch.sin(torch.pi * z / 1.0e4)
    state, _ = block.initialize({"hydro_w": w})
    return block, state


def check_retry():
    cuda = torch.device("cuda")
    pair = (retry_case(cuda, "cuda-graph", True), retry_case(cuda))
    (block, state), (ref, expected) = pair
    for step in range(2):
        forward_both(pair, 1.0e9, sync_error=True)
        assert block.check_redo(state) == ref.check_redo(expected) == 0, step
        assert_equal(state, expected, ("retry", step))
        assert_meters(block, ref, ("retry", step))
    n = ref.module("hydro.icorr").buffer("fp64_fallbacks").item()
    assert n > 0, "no column was retried in FP64; the case is vacuous"
    report = block.execution_report()
    assert report["eager"] == 0 and report["replays"] > 0, report
    print("FP32 with FP64 retry: %d retried columns, graph equals eager:" % n, report)


def main():
    torch.set_num_threads(1)
    for precision in ("float64", "float32"):
        check_cpu(precision)
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (graph part)")
        return 0
    failed = []
    for name, check in (("moist float64", lambda: check_moist("float64")),
                        ("moist float32", lambda: check_moist("float32")),
                        ("retry float32", check_retry)):
        try:
            check()
        except (AssertionError, RuntimeError) as e:
            failed.append("%s: %s" % (name, str(e).strip()))
    for f in failed:
        print("FAIL", f)
    print("PASSED" if not failed else "FAILED (%d)" % len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
