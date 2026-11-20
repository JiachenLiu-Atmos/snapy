#!/usr/bin/env python3
"""V3 lane K, S-K1 (K.rst 3.2): kintera's fixed failure counter in the step's
reset list and in the retry stack, with no host read inside the step.

Scenario: tests/test_check_redo_saturation.py's supersaturated NH4SH cell with
max-iter 1, so the last stage's saturation adjustment leaves cells unadjusted.

  1. Retry cause through consume_failures: after the step row 0 of the
     counter is > 0; check_redo redoes the step with the saturation cause and
     leaves row 0 consumed in place (0), rows 1-2 untouched. RED before S-K1:
     check_redo read and rebound ThermoY.nfail_ and left row 0 > 0.
  2. The two solver counters: the register's monitor carries
     thermo.temp_solve and thermo.pres_solve (two more counters than the
     parent's thirteen), equal to rows 1-2 of the ThermoY counter after the
     step; counts planted in those rows before the last stage (this deck's
     solves converge, so the rows would otherwise be zero) reach the
     monitor. RED before S-K1: the monitor has thirteen counters.
  3. The reset list: a failure counted between steps is cleared at stage 0;
     on CUDA that stage 0 (of a block that has already stepped once) runs
     under torch.cuda.set_sync_debug_mode("error"), so the reset must not
     read the host. RED before S-K1: stage 0 drained the count through
     take_saturation_adjustment_failures() (a host read).

  python test_sk1_failure_counter.py [--device cuda]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_check_redo_saturation as base  # noqa: E402

PARENT_COUNTERS = 13  # ef.* (7) + face_floor/riemann (5) + I-6's implicit.fp64_columns at the parent
TEMP_SOLVE, PRES_SOLVE = PARENT_COUNTERS, PARENT_COUNTERS + 1


def monitor_counters(block):
    return dict(block.named_buffers())["hydro.monitor_counters"]


def arm_consume(device, yaml_file):
    failures = []
    block = base.make_block(device, yaml_file, 1)
    block_vars = base.initialize(block)
    thermo = base.modules(block)[1]
    base.step(block, block_vars)
    f = thermo.failures().cpu()
    print("after the step: counter", f.tolist())
    if f[0].item() < 1:
        failures.append("row 0 of the counter is %d after a step that fails the adjustment"
                        % f[0].item())
    rows12 = f[1:].clone()
    err, causes = base.check_redo_causes(block, block_vars)
    g = thermo.failures().cpu()
    print("after check_redo %d causes %s: counter %s" % (err, causes, g.tolist()))
    if err != 1 or causes != ["saturation"]:
        failures.append("check_redo %d causes %s, expected a redo for saturation alone"
                        % (err, causes))
    if g[0].item() != 0:
        failures.append("row 0 not consumed in place by check_redo: %d" % g[0].item())
    if not torch.equal(g[1:], rows12):
        failures.append("rows 1-2 changed by check_redo: %s -> %s" % (rows12.tolist(), g[1:].tolist()))
    return failures


def arm_monitor(device, yaml_file):
    failures = []
    block = base.make_block(device, yaml_file, 1)
    block_vars = base.initialize(block)
    thermo = base.modules(block)[1]
    c = monitor_counters(block)
    if c.numel() != PARENT_COUNTERS + 2:
        failures.append("monitor has %d counters, expected %d (thermo.temp_solve, thermo.pres_solve added)"
                        % (c.numel(), PARENT_COUNTERS + 2))
        return failures
    # the solves converge on this deck, so plant counts in rows 1-2 after
    # stage 0's reset and before the last stage posts them to the monitor
    dt = block.max_time_step(block_vars)
    nstage = len(block.intg.stages)
    for stage in range(nstage - 1):
        block.forward(block_vars, dt, stage)
    counter = thermo.failures()
    counter[1] += 5
    counter[2] += 7
    block.forward(block_vars, dt, nstage - 1)
    f = thermo.failures().cpu()
    c = monitor_counters(block).cpu()
    print("monitor counters", c.tolist(), "thermo rows 1-2", f[1:].tolist())
    if f[1].item() != 5 or f[2].item() != 7:
        failures.append("rows 1-2 planted as [5, 7] read %s after the last stage" % f[1:].tolist())
    if c[TEMP_SOLVE].item() != f[1].item() or c[PRES_SOLVE].item() != f[2].item():
        failures.append("monitor thermo.temp_solve/pres_solve %s != counter rows 1-2 %s"
                        % ([c[TEMP_SOLVE].item(), c[PRES_SOLVE].item()], f[1:].tolist()))
    return failures


def arm_reset(device, yaml_file):
    failures = []
    block = base.make_block(device, yaml_file, 1)
    block_vars = base.initialize(block, cell=False)
    thermo = base.modules(block)[1]
    base.step(block, block_vars)  # a block's first eager stage syncs once; this one is past it
    # a failed adjustment between steps, as test_check_redo_saturation's between_steps_arm
    _, temp, rho, y = base.fields(block)
    sl = block.part((0, 0, 0), False)[1:]
    temp, rho, y = temp[sl], rho[sl].contiguous(), y[(slice(None),) + sl].contiguous()
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    diag = torch.zeros(rho.shape + (1,), dtype=rho.dtype, device=rho.device)
    thermo.forward(rho, ie, y, False, diag)
    nbad = int((diag < 0).sum())
    # rows 1-2 (temperature and pressure solves) are written by the solves;
    # plant counts there too: reset_failures() clears all three rows at stage
    # 0 (K.rst 3.2), where the old take_* read cleared row 0 only
    counter = thermo.failures()
    counter[1] += 5
    counter[2] += 7
    before = thermo.failures().cpu().tolist()
    print("between steps: %d cell(s) failed, counter %s" % (nbad, before))
    if nbad < 1 or before[0] < 1:
        failures.append("the between-steps call counted no failure (%d cells, counter %s); the arm is vacuous"
                        % (nbad, before))
        return failures
    dt = block.max_time_step(block_vars)
    if device == "cuda":  # the reset list must not read the host
        torch.cuda.set_sync_debug_mode("error")
    try:
        block.forward(block_vars, dt, 0)  # stage 0: the reset list
    finally:
        if device == "cuda":
            torch.cuda.set_sync_debug_mode("default")
    after = thermo.failures().cpu().tolist()
    print("after stage 0: counter", after)
    if after != [0, 0, 0]:
        failures.append("stage 0 left a stale count %s (all three rows must be zero)" % after)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; skipping")
        return 0
    yaml_file = os.path.join(HERE, "test_check_redo_saturation.yaml")
    failures = []
    for arm in (arm_consume, arm_monitor, arm_reset):
        print("==", arm.__name__)
        failures += arm(args.device, yaml_file)
    for f in failures:
        print("FAIL:", f)
    print("PASSED" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
