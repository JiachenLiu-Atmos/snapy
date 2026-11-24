"""A stretched cubed sphere runs its stages as CUDA graphs, bitwise to an
eager Mesh.

The deck is test_stage2_cubed_stretched.yaml: a C24 sphere (six panels, one
process) whose x1 faces are an explicit stretched list, so the well-balanced
x1 reference evaluates its quadrature rows (six_face_rows) every stage. The
rows are a function of the grid alone; an evaluation inside a captured
segment must not copy them from the host (a pageable host-to-device copy is
not permitted while a stream captures).

Strict cuda-graph against eager, FP64, NSTEPS steps from the hydrostatic rest
of test_stage2_cubed_stretched.py, bitwise after every step. One capture per
(block, stage, segment) and replays after.

  python test_graph_stretched_x1.py
"""
import os
import tempfile
from pathlib import Path

import torch
import yaml

from cuda_build import build_has_cuda, skip

NSTEPS = 3
P0, H0, G = 1.e6, 6.e4, 22.
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")
DECK = Path(__file__).with_name("test_stage2_cubed_stretched.yaml")


def make(execution, strict):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIPR

    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    cfg = yaml.safe_load(DECK.read_text())
    cfg["integration"]["execution"] = execution
    cfg["integration"]["execution-strict"] = strict
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        mesh = Mesh(MeshOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    x1f = blocks[0].module("coord").buffer("x1f").double()[3:34]
    p = P0 * torch.exp(-((x1f[1:] + x1f[:-1]) / 2 - x1f[0]) / H0)
    col = torch.stack([p / (G * H0)] + [0 * p] * 3 + [p]).view(5, 1, 1, 30)
    wb, _, _ = snapy.balance_column(col, (x1f[1:] - x1f[:-1]).contiguous(), G,
                                    rtol=1.e-13, stretched=True)
    wb = torch.cat([wb[..., :1]] * 3 + [wb] + [wb[..., -1:]] * 3, -1)
    mvars = []
    for b in blocks:
        b.to(torch.device("cuda", 0))
        bufs = dict(b.named_buffers())
        w = bufs["hydro.D"].clone().zero_()
        w[kIDN], w[kIPR] = wb[kIDN, 0, 0].to(w), wb[kIPR, 0, 0].to(w)
        mvars.append({"hydro_w": w,
                      "scalar_r": bufs["scalar.D"].clone().fill_(.01)})
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    graph, gb, gv = make("cuda-graph", True)
    eager, _, ev = make("eager", False)
    for i, b in enumerate(gb):
        why = b.execution_capability(gv[i])["eager_reasons"]
        assert not why, (i, why)
    dt = eager.max_time_step(ev)
    nstage = len(gb[0].module("intg").stages)
    for n in range(NSTEPS):
        for s in range(nstage):
            graph.forward(gv, dt, s)
            eager.forward(ev, dt, s)
        for i in range(len(gb)):
            for k in KEYS:
                if k not in gv[i]:
                    continue
                d = (gv[i][k].double() - ev[i][k].double()).abs().max().item()
                assert torch.equal(gv[i][k], ev[i][k]), ("step", n, "block", i,
                                                         k, d)
    r = graph.execution_report()
    nb = len(gb)
    assert r["captures"] >= nstage * nb and r["eager"] == 0, r
    assert r["replays"] >= nstage * nb * (NSTEPS - 1), r
    print("stretched cubed sphere: graph equals eager over", NSTEPS, "steps:",
          r)
    print("PASSED")


if __name__ == "__main__":
    main()
