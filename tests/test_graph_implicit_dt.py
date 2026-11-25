#!/usr/bin/env python3
"""V3 C4: the vertical implicit solve reads the block's device dt, so a case
with an implicit scheme captures, and one capture per stage replays at every
dt. The moist Straka case of test_sk3_capture_moist (H2O condensation) with
implicit-scheme 1 (partial matrix) and 9 (full matrix), face gravity work
(the Woodbury corners and the radial-exact rows, which divide by dt), is run
as cuda-graph, strict, against eager, bitwise, in FP64 and FP32 (FP32: the
FP32 solve with its predicated FP64 stage). dt changes every step; every
forward() of the graph block runs under torch.cuda.set_sync_debug_mode
("error"). The implicit op keeps no dt in its graph key: its segment reads
no host dt (capability), a dt change neither recaptures nor runs eager, and
a forced retry restores the step and replays the same graphs.

The six-panel cubed sphere of test_cubed_sphere_capture (B3) with an
implicit scheme, FP64, strict, is run the same way, with the x1 flux on and
off: with it off, the x1 mass-flux w term reads the solve's posts in the x1
total, which divide by the device dt too.

RED before C4: a strict cuda-graph case with an implicit scheme is refused
at its first launch ("implicit-scheme is not 0"); the sphere with the x1
flux off reads the host dt (StepDt::value) inside the capture.

  python test_graph_implicit_dt.py
"""
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR  # noqa: E402
from test_stage_executor_capture import KEYS, assert_equal, forward_both  # noqa: E402
from cuda_build import build_has_cuda, skip  # noqa: E402

QVAPOR = 0.005  # kg/kg, as test_sk3_capture_moist: condenses in the bubble
DTS = (0.1, 0.05, 0.075, 0.1, 0.0625, 0.05, 0.08, 0.1)


def make_case(precision, device, scheme, execution="eager", strict=False):
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
    cfg["integration"]["implicit-scheme"] = scheme
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
    w[kICY] = QVAPOR
    ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return block, state


def check(precision, scheme):
    cuda = torch.device("cuda")
    block, state = make_case(precision, cuda, scheme, "cuda-graph", True)
    ref, expected = make_case(precision, cuda, scheme)
    assert block.module("hydro.icorr") is not None
    pair = ((block, state), (ref, expected))
    nstages = len(block.intg.stages)
    where = (precision, scheme)

    rows = block.execution_capability(state)["segments"]
    compute = [r for r in rows if not r["host"]]
    assert compute and all(r["capturable"] for r in compute), (where, rows)
    assert not any(r["reads_host_dt"] for r in rows), (where, rows)

    for step, dt in enumerate(DTS):
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0, (where, step)
        assert_equal(state, expected, (where, "dt", step, dt))
        corr = ref.module("hydro.icorr").buffer("corr")
        if step == 0:  # the solve did something: the case is not vacuous
            assert corr.numel() > 0 and corr.abs().max().item() > 0, where
    fb = [b.module("hydro.icorr").buffer("fp64_fallbacks").item() for b in (block, ref)]
    assert fb[0] == fb[1], (where, fb)

    report = block.execution_report()
    assert report["eager"] == 0, (where, report)
    assert report["captures"] == nstages, (where, report)
    assert report["graphs"] == nstages, (where, report)
    assert report["replays"] == nstages * len(DTS), (where, report)

    # a forced retry restores the step; other dt values replay the same graphs
    before = {k: state[k].clone() for k in ("hydro_u", "scalar_s")}
    forward_both(pair, 0.1)
    for fields in (state, expected):
        fields["hydro_u"][0, 0, 20, 20] = float("nan")
    assert block.check_redo(state) == ref.check_redo(expected) == 1, where
    for k, v in before.items():
        assert torch.equal(state[k], v), (where, "restore", k)
        assert torch.equal(expected[k], v), (where, "restore eager", k)
    for dt in (0.05, 0.07):
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0, where
        assert_equal(state, expected, (where, "after retry", dt))
    report = block.execution_report()
    assert report["captures"] == nstages, (where, report)
    assert report["replays"] == nstages * (len(DTS) + 3), (where, report)
    assert report["eager"] == 0, (where, report)
    print(precision, f"implicit-scheme {scheme}: one capture per stage, "
          f"{len(set(DTS)) + 1} dt values bitwise:", report, flush=True)


def check_sphere(scheme, flux_off, nsteps=6):
    import test_cubed_sphere_capture as cs

    base = cs.deck

    def deck(execution, strict, scalar, co4):
        d = base(execution, strict, scalar, co4)
        d["integration"]["implicit-scheme"] = scheme
        if flux_off:
            d["dynamics"]["disable-flux-x1"] = True
        return d

    cs.deck = deck
    try:
        graph, gb, gv = cs.make("cuda-graph", True, False, False)
        eager, _, ev = cs.make("eager", False, False, False)
    finally:
        cs.deck = base
    where = ("sphere", scheme, "x1 flux off" if flux_off else "x1 flux on")
    for i, b in enumerate(gb):
        cap = b.execution_capability(gv[i])
        assert not cap["eager_reasons"], (where, i, cap["eager_reasons"])
        assert not any(r["reads_host_dt"] for r in cap["segments"]), (where, i)
    dt0 = eager.max_time_step(ev)
    for n in range(nsteps):
        dt = 0.5 * dt0 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(3):  # as test_cubed_sphere_capture: no sync-debug mode
            graph.forward(gv, dt, s)
            eager.forward(ev, dt, s)
        for i in range(len(gb)):
            for k in KEYS:
                if k in gv[i]:
                    assert torch.equal(gv[i][k], ev[i][k]), (where, "step", n, i, k)
    r = graph.execution_report()
    nb = len(gb)
    assert r["captures"] == 3 * nb and r["graphs"] == 3 * nb, (where, r)
    assert r["replays"] == 3 * nb * nsteps and r["eager"] == 0, (where, r)
    print(*where, f": one capture per (block, stage), {nsteps} dt values bitwise:",
          r, flush=True)


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    for precision in ("float64", "float32"):
        for scheme in (1, 9):
            check(precision, scheme)
    for scheme in (1, 9):
        for flux_off in (False, True):
            check_sphere(scheme, flux_off)
    print(*KEYS, "equal; PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
