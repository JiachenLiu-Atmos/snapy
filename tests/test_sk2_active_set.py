#!/usr/bin/env python3
"""V3 lane K, S-K2 (K.rst 3.4, gate K-G5): the ThermoY active set is
snapshotted at stage 0 of every attempt and restored on a retry, so a retry
repeats the first attempt from the same warm start.

Scenario: tests/test_check_redo_saturation.py's NH4SH deck with the default
max-iter, where the last stage's saturation adjustment converges.

  Run A: one step; the active set at the start of the second step is read
  (snapshot_active_set into the test's own storage); the second step runs,
  then the active set is changed by a ThermoY call on a different state (no
  NH3 anywhere) with the warm start on; then a retry is forced (a NaN written
  into hydro_u, as test_stage_executor_capture does). After the rollback the
  module's active set must equal the one at the step's start again, and the
  step redone at the retry's dt must equal a clean run at that dt bitwise.
  Run B: a fresh block takes one step at that same dt from the same state.
  RED before S-K2: the rollback did not touch kintera, so after it the module
  still holds the changed active set (K.rst 3.4). The bitwise comparison is
  reported either way (on this deck the solver reaches the same fixed point
  from both warm starts, so it alone does not distinguish the two).

  python test_sk2_active_set.py [--device cuda]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_check_redo_saturation as base  # noqa: E402

KEYS = ("hydro_u", "hydro_w")


def perturb_active_set(block):
    """A converged adjustment on the interior without NH3, warm start on: the
    buffers now hold a different active set (the shape advance_local passes)."""
    thermo = base.modules(block)[1]
    _, temp, rho, y = base.fields(block, cell=False)
    sl = block.part((0, 0, 0), False)[1:]
    temp, rho, y = temp[sl], rho[sl].contiguous(), y[(slice(None),) + sl].contiguous()
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    diag = torch.zeros(rho.shape + (1,), dtype=rho.dtype, device=rho.device)
    thermo.forward(rho, ie, y, True, diag)
    return int((diag < 0).sum())


def active_set(thermo):
    """A copy of the module's (reaction_set, nactive) in the test's storage."""
    rs, na = thermo.snapshot_active_set(thermo.make_active_set_storage())
    return rs.clone(), na.clone()


def same_set(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def run_a(device, yaml_file):
    block = base.make_block(device, yaml_file, None)
    block_vars = base.initialize(block)
    u0 = block_vars["hydro_u"].clone()
    dt = block.max_time_step(block_vars)
    base.step(block, block_vars)
    err, causes = base.check_redo_causes(block, block_vars)
    if err != 0:
        return None, "the first attempt was not accepted (check_redo %d, causes %s)" % (err, causes)
    # the first attempt is accepted; now a second step whose attempt fails
    dt = block.max_time_step(block_vars)
    u1 = block_vars["hydro_u"].clone()
    w1 = block_vars["hydro_w"].clone()
    thermo = base.modules(block)[1]
    as0 = active_set(thermo)  # the set stage 0 of the second step snapshots
    base.step(block, block_vars)
    nbad = perturb_active_set(block)
    as1 = active_set(thermo)
    if same_set(as1, as0):
        return None, "the perturbing call left the active set unchanged; the arm is vacuous"
    k, j, i = (u1.size(1) // 2, u1.size(2) // 2, u1.size(3) // 2)
    block_vars["hydro_u"][0, k, j, i] = float("nan")
    err, causes = base.check_redo_causes(block, block_vars)
    if err != 1:
        return None, "the forced retry did not happen (check_redo %d, causes %s)" % (err, causes)
    sl4 = block.part((0, 0, 0), False)  # the backup is taken after the ghost exchange
    if not torch.equal(block_vars["hydro_u"][sl4], u1[sl4]):
        return None, "rollback did not restore the interior of hydro_u"
    as2 = active_set(thermo)
    print("active set after the rollback: %s" % (
        "the step-start set" if same_set(as2, as0) else
        "the changed set" if same_set(as2, as1) else "neither"))
    restored = same_set(as2, as0)
    dt2 = block.max_time_step(block_vars)
    base.step(block, block_vars)
    err, causes = base.check_redo_causes(block, block_vars)
    if err != 0:
        return None, "the retried step was not accepted (check_redo %d, causes %s)" % (err, causes)
    out = {k: block_vars[k].clone() for k in KEYS}
    return (u1, w1, dt, dt2, nbad, out, restored), None


def run_b(device, yaml_file, u1, w1, dt2):
    """A fresh block, the same state as A's second step start, one step at dt2."""
    block = base.make_block(device, yaml_file, None)
    block_vars = base.initialize(block)
    # bring it to A's state after the accepted first step: the same dt, same path
    dt = block.max_time_step(block_vars)
    base.step(block, block_vars)
    assert block.check_redo(block_vars) == 0
    if not torch.equal(block_vars["hydro_u"], u1):
        return None, "run B's first step differs from run A's (determinism)"
    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt2, stage)
    err, causes = base.check_redo_causes(block, block_vars)
    if err != 0:
        return None, "run B's step at the retry dt was not accepted (check_redo %d, causes %s)" % (err, causes)
    return {k: block_vars[k].clone() for k in KEYS}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; skipping")
        return 0
    yaml_file = os.path.join(HERE, "test_check_redo_saturation.yaml")
    failures = []
    a, msg = run_a(args.device, yaml_file)
    if a is None:
        failures.append(msg)
    else:
        u1, w1, dt, dt2, nbad, out_a, restored = a
        print("run A: dt %.6g, retry dt %.6g, perturbing call left %d cell(s) unadjusted" % (dt, dt2, nbad))
        if not restored:
            failures.append("the rollback did not restore the active set of the step's start")
        out_b, msg = run_b(args.device, yaml_file, u1, w1, dt2)
        if out_b is None:
            failures.append(msg)
        else:
            for k in KEYS:
                d = (out_a[k].to(torch.float64) - out_b[k].to(torch.float64)).abs()
                same = torch.equal(out_a[k], out_b[k])
                print("%s: %s (max |A - B| = %.3e, %d of %d differ)"
                      % (k, "bitwise" if same else "DIFFERS", d.max().item(),
                         int((d > 0).sum()), d.numel()))
                if not same:
                    failures.append("%s of the retried step differs from the clean run at the same dt" % k)
    for f in failures:
        print("FAIL:", f)
    print("PASSED" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
