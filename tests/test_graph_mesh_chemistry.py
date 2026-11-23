#!/usr/bin/env python3
"""V3 lane C, C5: a Mesh with a registered chemistry step runs its blocks'
stages as CUDA graphs (B2) and its chemistry in the global phase, eagerly,
after the stages and before the final exchange (Mesh::forward, S-K4), and
equals an eager Mesh bitwise.

The moist column of test_graph_chemistry_slot.py (H2O, H2O(l), H2O(l,p);
the saturation adjustment, coagulation and evaporation) as a two-block slab
Mesh (nb2 = 2), flux-covariance off (with species and more than one block it
syncs across blocks inside the stage, a case-wide refusal of its own). Two
Meshes from one deck, one cuda-graph (not strict) and one eager, both with
Mesh.set_chemistry, step the same dt schedule (1.0, then 0.5: a recapture)
in FP64 and FP32. After every step each block's hydro_u, hydro_w and
scalar_r and check_redo must match bitwise, the graph Mesh must have
captured and replayed its blocks' stages, and the case does something: the
eager Mesh has cloud and rain after step 1. On CPU the cuda-graph setting
runs eager and equals eager.

RED at c371bc53 (C2): a Mesh block with a chemistry step keeps a case-wide
reason ("a chemistry step is registered on a Mesh block ... (C5)"), so every
block-stage of the graph Mesh runs eager (captures == 0).

  python test_graph_mesh_chemistry.py
"""
import os
import sys
import tempfile

import torch
import yaml

import kintera as kt
from snapy import Mesh, MeshOptions, kICY, kIDN, kIPR, kIV1, kIV2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_graph_chemistry_slot import QVAPOR, config  # noqa: E402

KEYS = ("hydro_u", "hydro_w", "scalar_r")


def state(block):
    """moist_state of test_graph_chemistry_slot.py for one block of a Mesh."""
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
    return {"hydro_w": w, "scalar_r": ratio}


def make_mesh(precision, device, execution="eager"):
    cfg = config(precision, execution, False, chemistry=True)
    assert cfg["integration"]["type"] == "rk3", cfg["integration"]
    cfg["distribute"]["nb2"] = 2
    cfg["dynamics"]["flux-covariance"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        options = MeshOptions.from_yaml(path)
        options.blocks_per_process(2)
        mesh = Mesh(options)
        mesh.set_chemistry(kt.KineticsOptions.from_yaml(path))
    finally:
        os.unlink(path)
    blocks = list(mesh.blocks)
    assert len(blocks) == 2, "the slab Mesh should hold two blocks, got %d" % len(blocks)
    for b in blocks:
        b.to(device)
    mv, _ = mesh.initialize([state(b) for b in blocks])
    return mesh, mv


NSTAGES = 3  # the deck's rk3, the only integrator a graph takes


def step(mesh, mv, dt):
    for stage in range(NSTAGES):
        mesh.forward(mv, dt, stage)


def assert_equal(gv, ev, where):
    for ib, (a, b) in enumerate(zip(gv, ev)):
        for k in KEYS:
            if k not in a:
                continue
            if not torch.equal(a[k], b[k]):
                d = (a[k].double() - b[k].double()).abs().nan_to_num(float("inf"))
                raise AssertionError("%s: block %d %s differs in %d of %d cells, max |d| %.3e"
                                     % (where, ib, k, int((d > 0).sum()), d.numel(), d.max().item()))


def run(precision, device, nsteps):
    graph, gv = make_mesh(precision, device, "cuda-graph")
    eager, ev = make_mesh(precision, device)
    for s in range(nsteps):
        dt = 0.5 if nsteps // 2 <= s else 1.0
        step(graph, gv, dt)
        step(eager, ev, dt)
        rg, re = graph.check_redo(gv), eager.check_redo(ev)
        assert rg == re == 0, (precision, s, "check_redo", rg, re)
        assert_equal(gv, ev, (precision, s))
        if s == 0:
            cloud = sum(int((v["hydro_u"][kICY + 1] > 0).sum()) for v in ev)
            rain = sum(int((v["hydro_u"][kICY + 2] > 0).sum()) for v in ev)
            assert cloud > 0 and rain > 0, (precision, "vacuous: no cloud or rain", cloud, rain)
            print(precision, "after step 1: %d cells with cloud, %d with rain" % (cloud, rain))
    return graph.execution_report()


def main():
    torch.set_num_threads(1)
    failed = []
    for precision in ("float64", "float32"):
        report = run(precision, torch.device("cpu"), 2)
        assert report["captures"] == 0 and report["replays"] == 0, report
        print(precision, "CPU: cuda-graph Mesh runs eager and equals eager:", report)
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (graph part)")
        return 0
    for precision in ("float64", "float32"):
        try:
            report = run(precision, torch.device("cuda"), 8)
            assert report["captures"] > 0 and report["replays"] > 0, (
                "the graph Mesh captured nothing: a Mesh chemistry step still "
                "makes every block-stage eager", report)
            print(precision, "graph Mesh with chemistry equals eager:", report)
        except (AssertionError, RuntimeError) as e:
            failed.append("%s: %s" % (precision, str(e).strip()))
    for f in failed:
        print("FAIL", f)
    print("PASSED" if not failed else "FAILED (%d)" % len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
