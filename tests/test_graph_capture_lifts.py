"""V3 capture-rule lifts: the shipped dry Straka deck captures.

examples/straka.yaml as shipped (ideal-gas EOS, lmars, weno5, rk3, const-gravity,
one scalar tracer) was refused as a CUDA graph by two rules that only named the
X3 prototype's case (ideal-moist, hllc), and a deck with no forcing block by the
const-gravity condition. Nothing in the ideal-gas EOS or the lmars solver reads
the host, so these cases capture. Strict cuda-graph Mesh against an eager Mesh,
bitwise after every step, FP64 (the deck's ideal-gas/lmars physics has no FP32
path: float32 needs perturbation-flux, which needs ideal-moist and hllc):

  straka_1block     the shipped deck in one block
  straka_2block     the shipped deck as shipped (nb2 = 2, one process)
  straka_noforcing  the shipped deck without its forcing block, one block

Each case first asserts that no capture reason is left, so a refused case
fails at once; a watchdog ends a hung run. Without CUDA the test skips.

  python test_graph_capture_lifts.py
"""
import faulthandler
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NSTEPS = 10
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")
WATCHDOG_S = 480
CASES = {
    "straka_1block": {"nb2": 1, "forcing": True},
    "straka_2block": {"nb2": 2, "forcing": True},
    "straka_noforcing": {"nb2": 1, "forcing": False},
}


def deck(case, execution, strict):
    root = os.path.dirname(HERE)
    cfg = yaml.safe_load(open(os.path.join(root, "examples/straka.yaml")))
    nb2 = CASES[case]["nb2"]
    cfg["distribute"].update({"nb2": nb2, "blocks_per_process": nb2})
    if not CASES[case]["forcing"]:
        cfg.pop("forcing")
    cfg["outputs"] = []
    cfg["dynamics"]["precision"] = "float64"
    cfg["integration"]["ncycle_out"] = 0
    cfg["integration"]["execution"] = execution
    cfg["integration"]["execution-strict"] = strict
    return cfg


def make(case, execution, strict):
    from snapy import Mesh, MeshOptions, kIDN, kIPR

    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    cfg = deck(case, execution, strict)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(CASES[case]["nb2"])
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    for b in blocks:
        b.to(torch.device("cuda", 0))
    eos = cfg["dynamics"]["equation-of-state"]
    rd = 8.31446261815324 / eos["weight"]
    cp = rd * eos["gammad"] / (eos["gammad"] - 1.0)
    mvars = []
    for b in blocks:
        z = b.module("coord").buffer("x1v").view(1, 1, -1).double()
        x = b.module("coord").buffer("x2v").view(1, -1, 1).double()
        w = b.module("hydro").buffer("D").double().clone().zero_()
        temp = 300.0 - 9.8 * z / cp
        w[kIPR] = 1.0e5 * (temp / 300.0) ** (cp / rd)
        radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
        temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
        w[kIDN] = w[kIPR] / (rd * temp)
        ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
        mvars.append({"hydro_w": w, "scalar_r": ratio})
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def check(case):
    graph, gb, gv = make(case, "cuda-graph", True)
    for i, b in enumerate(gb):
        why = b.execution_capability(gv[i])["eager_reasons"]
        assert not why, (case, "block", i, "still refused:", why)
    eager, _, ev = make(case, "eager", False)
    dt0 = eager.max_time_step(ev)
    worst = 0.0
    for n in range(NSTEPS):
        dt = 0.5 * dt0 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(3):
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.forward(gv, dt, s)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            eager.forward(ev, dt, s)
        for i in range(len(gb)):
            for k in KEYS:
                d = (gv[i][k].double() - ev[i][k].double()).abs().max().item()
                worst = max(worst, d)
                assert torch.equal(gv[i][k], ev[i][k]), (case, "step", n, "block", i, k, d)
    for i in range(len(gb)):
        assert bool(torch.isfinite(ev[i]["hydro_u"]).all()), (case, i)
    r = graph.execution_report()
    nb = len(gb)
    assert r["eager"] == 0, (case, r)
    assert r["captures"] == 3 * nb and r["replays"] == 3 * nb * NSTEPS, (case, r)
    print(case, ": graph equals eager over", NSTEPS, "steps, max |g - e| =", worst,
          ", report:", {k: r[k] for k in ("captures", "replays", "cuts", "eager")},
          flush=True)


def main():
    if not torch.cuda.is_available() or os.environ.get("SNAPY_BUILD_CUDA", "1") == "0":
        print("SKIP: CUDA unavailable")
        return 0
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True)  # no hang: dump and exit 1
    for case in CASES:
        check(case)
    faulthandler.cancel_dump_traceback_later()
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
