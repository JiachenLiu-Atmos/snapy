#!/usr/bin/env python3
"""V3 lane X, C (X.rst sections 3 and 7, row "kintera moist" of section 10):
the chemistry slot is an eager FP64 segment beside the captured segments of
the same stage, so a block with kintera moist thermodynamics (the saturation
adjustment, captured since S-K3) and a registered chemistry step runs as
cuda-graph and equals eager, bitwise.

A 2D moist column (tests/test_anomaly_storage.yaml: H2O / H2O(l), Cartesian,
const gravity) with a rain species H2O(l,p) and two kinetic reactions,
H2O(l) => H2O(l,p) (coagulation, a constant rate) and H2O(l,p) => H2O
(evaporation), registered with MeshBlock.set_chemistry. The vapor is uniform, so the cold upper column
condenses in the first step and the slot moves cloud into rain. The graph
block (cuda-graph, not strict) and an eager block step the same dt schedule,
with a dt change, a rebind and an in-place edit, in FP64 and FP32; after
every step hydro_u, hydro_w, scalar_s and scalar_r, the ThermoY failure
counter and check_redo must match bitwise, and the graph block must have
captured and replayed its compute segments (no stage falls back to eager as
a whole). strict mode still refuses a case with an eager segment
(X.rst section 10, ``execution-strict``: "reject a case with any eager
segment"). On CPU the cuda-graph setting runs
eager and must equal eager.

RED at dc0e2571: capture_reasons() lists "a chemistry step is registered",
so every stage of the graph block runs eager (captures == 0).

  python test_graph_chemistry_slot.py
"""
import copy
import os
import sys
import tempfile

import torch
import yaml

import kintera as kt
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1, kIV2

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")
QVAPOR = 0.005  # kg/kg, uniform: saturated above ~2 km in this column
RAIN = "H2O(l,p)"


def config(precision, execution="eager", strict=False, chemistry=False,
           implicit=0, grav1=None, one_column=False):
    """The test_anomaly_storage deck, made capturable apart from the case
    under test: limiter off, hllc, anomaly storage with the split flux."""
    with open(os.path.join(HERE, "test_anomaly_storage.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True,
                            "precision": precision,
                            "riemann-solver": {"type": "hllc"}})
    cfg["dynamics"]["equation-of-state"]["limiter"] = False
    if chemistry:
        rain = copy.deepcopy(next(s for s in cfg["species"] if s["name"] == "H2O(l)"))
        rain["name"] = RAIN
        cfg["species"].append(rain)
        cfg["reactions"] += [
            {"equation": "H2O(l) => " + RAIN, "type": "coagulation",
             "rate-constant": {"A": 0.01, "b": 0, "Ea_R": 0.}},
            {"equation": RAIN + " => H2O", "type": "evaporation",
             "rate-constant": {"formula": "h2o_bryan", "diff_c": 0.2e-4, "diff_T": 1.75,
                               "diff_P": -1., "vm": 18.e-6, "diameter": 1.0e-4}}]
    if one_column:
        cfg["geometry"]["cells"].update({"nx2": 1, "nx3": 1})
    if grav1 is not None:
        cfg["forcing"]["const-gravity"]["grav1"] = grav1
    cfg["integration"].update({"implicit-scheme": implicit, "ncycle_out": 0,
                               "execution": execution,
                               "execution-strict": strict})
    cfg["outputs"] = []
    return cfg


def build(cfg, device, chemistry=False):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(path))
        if chemistry:
            block.set_chemistry(kt.KineticsOptions.from_yaml(path))
    finally:
        os.unlink(path)
    block.to(device)
    return block


def moist_state(block):
    """A stratified column, uniform vapor, a horizontal perturbation."""
    thermo = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")
    u = block.module("hydro").buffer("D").double().clone().zero_()
    z = block.module("coord").buffer("x1v").view(1, 1, -1).expand_as(u[kIDN])
    x2 = block.module("coord").buffer("x2v").view(1, -1, 1).expand_as(u[kIDN])
    g, t0, gam, p0, rd = 10.44, 300.0, 6.5e-3, 1.0e5, 287.0
    temp = (t0 - gam * z).clamp_min(200.0)
    pres = p0 * (temp / t0) ** (g / (rd * gam))
    rho = pres / (rd * temp)
    y = torch.zeros_like(u[kICY:])
    y[0] = QVAPOR
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    u[kIDN] = rho * (1.0 - y.sum(0))
    u[kICY:] = rho * y
    u[kIPR] = ie
    w = eos.compute("U->W", [u.to(block.module("hydro").buffer("D").dtype)])
    w[kIV2] = 0.5 * torch.sin(2.0 * torch.pi * x2 / 4.0e3)
    w[kIV1] = 0.2 * torch.cos(2.0 * torch.pi * x2 / 4.0e3) * torch.sin(torch.pi * z / 1.0e4)
    ratio = torch.zeros_like(w[:1]) + (z / 1.0e4).clamp(0.0, 1.0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return state


def make_case(precision, device, execution="eager", strict=False, **kw):
    chemistry = kw.get("chemistry", False)
    block = build(config(precision, execution, strict, **kw), device, chemistry)
    return block, moist_state(block)


def forward_both(pair, dt, sync_error=False):
    """One step of both blocks; with sync_error the graph block's stages run
    under torch.cuda.set_sync_debug_mode("error") (no host sync in a segment)."""
    (block, state), (ref, expected) = pair
    for stage in range(len(block.intg.stages)):
        if sync_error:
            torch.cuda.set_sync_debug_mode("error")
        try:
            block.forward(state, dt, stage)
        finally:
            if sync_error:
                torch.cuda.set_sync_debug_mode("default")
        ref.forward(expected, dt, stage)


def assert_equal(state, expected, where):
    assert set(state) == set(expected), (where, set(state) ^ set(expected))
    for k in KEYS:
        if k not in state:
            continue
        a, b = state[k], expected[k]
        if not torch.equal(a, b):
            d = (a.double() - b.double()).abs().nan_to_num(float("inf"))
            raise AssertionError("%s: %s differs in %d of %d cells, max |d| %.3e"
                                 % (where, k, int((d > 0).sum()), d.numel(), d.max().item()))


def failures(block):
    return block.module("hydro.eos.thermo").failures().clone()


def run_schedule(pair, precision, nsteps=12, sync_error=False, extra=None):
    """dt 1.0, then 0.5 (a recapture), a rebind and an in-place edit; bitwise
    after every step. Returns the number of keys the graph block saw."""
    (block, state), (ref, expected) = pair
    keys, last_dt = 0, None
    for step in range(nsteps):
        dt = 0.5 if nsteps // 3 <= step < 2 * nsteps // 3 else 1.0
        if step == nsteps - 3:  # a rebind between steps recaptures
            for fields in (state, expected):
                for k in ("hydro_u", "scalar_s"):
                    if k in fields:
                        fields[k] = fields[k].clone()
            keys += 1
        elif dt != last_dt:
            keys += 1
        if step == nsteps - 2:  # an in-place edit reaches the captured inputs
            for fields in (state, expected):
                fields["hydro_u"][kIPR, 0, 4, 10] += 1.0e-4
        last_dt = dt
        forward_both(pair, dt, sync_error)
        ra, rb = block.check_redo(state), ref.check_redo(expected)
        assert ra == rb == 0, (precision, step, "check_redo", ra, rb)
        assert_equal(state, expected, (precision, step))
        assert torch.equal(failures(block), failures(ref)), (precision, step, "thermo failures")
        if extra is not None:
            extra(step)
    return keys


def check_cpu(precision):
    cpu = torch.device("cpu")
    pair = (make_case(precision, cpu, "cuda-graph", chemistry=True),
            make_case(precision, cpu, chemistry=True))
    run_schedule(pair, precision, nsteps=3)
    report = pair[0][0].execution_report()
    assert report["captures"] == 0 and report["replays"] == 0, report
    print(precision, "CPU: cuda-graph runs eager and equals eager:", report)


def check_graph(precision):
    cuda = torch.device("cuda")
    pair = (make_case(precision, cuda, "cuda-graph", chemistry=True),
            make_case(precision, cuda, chemistry=True))
    (block, state), (ref, expected) = pair

    def nonvacuous(step):
        if step != 0:
            return
        sl = tuple(ref.part((0, 0, 0), False))
        # hydro_u's species rows: H2O, H2O(l), H2O(l,p) (dry is not stored)
        cloud = int((expected["hydro_u"][sl][kICY + 1] > 0).sum())
        rain = int((expected["hydro_u"][sl][kICY + 2] > 0).sum())
        assert cloud > 0, "no cell condensed in the first step; the case is vacuous"
        assert rain > 0, "the chemistry slot made no rain in the first step; the case is vacuous"
        print(precision, "after step 1: %d cells with cloud, %d with rain" % (cloud, rain))

    run_schedule(pair, precision, extra=nonvacuous)
    report = block.execution_report()
    # the compute segments are captured and replayed; the chemistry slot is
    # the one eager segment, so a stage never runs eager as a whole
    assert report["captures"] > 0 and report["replays"] > 0, (
        "the graph block captured nothing: a registered chemistry step still "
        "makes every stage eager", report, block.execution_capability(state)["eager_reasons"])
    print(precision, "graph with an eager chemistry segment equals eager:", report)

    # strict refuses any eager segment, before the first launch
    strict, sv = make_case(precision, cuda, "cuda-graph", True, chemistry=True)
    before = sv["hydro_u"].clone()
    try:
        strict.forward(sv, 1.0, 0)
    except RuntimeError as e:
        assert "execution-strict" in str(e), str(e)
    else:
        raise AssertionError("strict cuda-graph with an eager chemistry segment was not refused")
    assert torch.equal(sv["hydro_u"], before), "the refused block changed its state"


def main():
    torch.set_num_threads(1)
    for precision in ("float64", "float32"):
        check_cpu(precision)
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (graph part)")
        return 0
    failed = []
    for precision in ("float64", "float32"):
        try:
            check_graph(precision)
        except (AssertionError, RuntimeError) as e:
            failed.append("%s: %s" % (precision, str(e).strip()))
    for f in failed:
        print("FAIL", f)
    print("PASSED" if not failed else "FAILED (%d)" % len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
