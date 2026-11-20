#!/usr/bin/env python3
"""V3 lane K, S-K3 (K.rst 3.5, 7; row S-K3 of section 10): the saturation
adjustment runs inside the captured segment of the last stage. A moist
Straka case with H2O condensation (the deck of test_anomaly_storage.yaml's
species and nucleation reaction on examples/straka.yaml, a uniform vapor
mass fraction that condenses in the cold bubble and aloft) is run as
cuda-graph, strict, against eager, bitwise, in FP64 and FP32, with
test_stage_executor_capture's schedule: every forward() of the graph block
under torch.cuda.set_sync_debug_mode("error"), dt changes, a rebind, an
in-place edit and a forced retry. The ThermoY failure counter is compared
after each step too (counts survive replays, K-G4 style). RED before S-K3:
a strict cuda-graph case with reactions is refused at its first launch
("reactions are set", "more than one species").

  python test_sk3_capture_moist.py
"""
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR  # noqa: E402
from test_stage_executor_capture import KEYS, assert_equal, expect_error, forward_both  # noqa: E402

QVAPOR = 0.005  # kg/kg, saturated below ~287 K at 1 bar: the bubble and aloft


def make_case(precision, device, execution="eager", strict=False):
    os.environ["DEVICE"] = device.type
    os.environ["DEVICE_ID"] = "0"
    root = os.path.dirname(HERE)
    cfg = yaml.safe_load(open(os.path.join(root, "examples/straka.yaml")).read())
    moist = yaml.safe_load(open(os.path.join(HERE, "test_anomaly_storage.yaml")).read())
    cfg["reference-state"] = moist["reference-state"]
    cfg["species"] = moist["species"]
    cfg["reactions"] = moist["reactions"]
    cfg["distribute"]["nb2"] = 1
    cfg["outputs"] = []
    cfg["dynamics"].update(
        {
            "anomaly": True,
            "perturbation-flux": True,
            "precision": precision,
            "riemann-solver": {"type": "hllc"},
        }
    )
    cfg["dynamics"]["equation-of-state"] = {
        "type": "ideal-moist",
        "limiter": False,
        "density-floor": 1.0e-10,
        "pressure-floor": 1.0e-10,
    }
    cfg["integration"]["ncycle_out"] = 0
    cfg["integration"]["execution"] = execution
    cfg["integration"]["execution-strict"] = strict
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)
    block.to(device)
    thermo = block.module("hydro.eos.thermo")
    rd = 8.31446261815324 * thermo.buffer("inv_mu")[0].item()
    cp = 3.5 * rd
    z = block.module("coord").buffer("x1v").view(1, 1, -1)
    x = block.module("coord").buffer("x2v").view(1, -1, 1)
    w = block.module("hydro").buffer("D").double().clone().zero_()
    temp = 300.0 - 9.8 * z / cp
    w[kIPR] = 1.0e5 * (temp / 300.0) ** (cp / rd)
    radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
    temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
    w[kIDN] = w[kIPR] / (rd * temp)
    w[kICY] = QVAPOR  # vapor; the cloud row stays zero: the adjustment makes it
    ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return block, state


def counter(block):
    return block.module("hydro.eos.thermo").failures().clone()


def check_capture(precision):
    cuda = torch.device("cuda")
    block, state = make_case(precision, cuda, "cuda-graph", True)
    ref, expected = make_case(precision, cuda)
    pair = ((block, state), (ref, expected))
    nstages = len(block.intg.stages)
    sl = tuple(block.part((0, 0, 0), False))

    recaptures, last_dt = 0, None
    for step in range(40):
        dt = 0.05 if 10 <= step < 20 else 0.1
        if step == 25:  # a rebind between steps: X2 adopts, the graph recaptures
            for fields in (state, expected):
                fields["hydro_u"] = fields["hydro_u"].clone()
                fields["scalar_s"] = fields["scalar_s"].clone()
            recaptures += 1
        elif dt != last_dt:
            recaptures += 1
        if step == 30:  # an in-place edit reaches the captured inputs
            for fields in (state, expected):
                fields["hydro_u"][0, 0, 20, 20] += 1.0e-4
        last_dt = dt
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0
        assert_equal(state, expected, (precision, step))
        assert torch.equal(counter(block), counter(ref)), (precision, step, "counter")
        if step == 0:  # the adjustment did something: cloud made in the first step
            cloud = state["hydro_u"][sl][kICY + 1]
            ncloud = int((cloud > 0).sum())
            assert ncloud > 0, "no cell condensed in the first step; the case is vacuous"
            print(precision, "cells with cloud after step 1:", ncloud, "of", cloud.numel())

    # a forced retry restores the step; the smaller dt recaptures
    before = {k: state[k].clone() for k in ("hydro_u", "scalar_s")}
    forward_both(pair, 0.1)
    for fields in (state, expected):
        fields["hydro_u"][0, 0, 20, 20] = float("nan")
    assert block.check_redo(state) == ref.check_redo(expected) == 1
    for k, v in before.items():
        assert torch.equal(state[k], v), (precision, "restore", k)
        assert torch.equal(expected[k], v), (precision, "restore eager", k)
    for dt in (0.05, 0.05):
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0
        assert_equal(state, expected, (precision, "after retry"))
        assert torch.equal(counter(block), counter(ref)), (precision, "after retry", "counter")
    recaptures += 1

    report = block.execution_report()
    assert report["eager"] == 0, report
    assert report["captures"] == nstages * recaptures, (report, recaptures)
    assert report["replays"] == nstages * 43, report
    print(precision, "graph equals eager over 43 steps with condensation:", report)


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return 0
    for precision in ("float64", "float32"):
        check_capture(precision)
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
