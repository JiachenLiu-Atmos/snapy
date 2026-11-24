"""integration.execution defaults to cuda-graph on a CUDA device.

The shipped examples/straka.yaml in one block, with no execution key:

  CPU   it runs eager (execution_report: no capture, every stage eager), and
        equals bitwise the same deck with an explicit `execution: eager`
  CUDA  it captures (captures and replays, no eager stage) and equals bitwise
        the same deck with an explicit `execution: eager`, which stays eager

and on CUDA a six-panel cubed-sphere Mesh with no execution key: one block per
panel captures and equals explicit eager bitwise; panels split into two by two
blocks are refused (a panel split into several blocks cannot be captured yet),
run eager and equal explicit eager bitwise.

FP64, 10 steps with dt changing, bitwise after every step. The CPU part runs
on every build; the CUDA part needs a CUDA build of snapy and a device
(cuda_build.py) and is skipped otherwise.

  python test_default_execution.py
"""
import os
import sys
import tempfile

import torch
import yaml

from cuda_build import build_has_cuda, skip

HERE = os.path.dirname(os.path.abspath(__file__))
NSTEPS = 10
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def make(device, execution=None):
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    root = os.path.dirname(HERE)
    cfg = yaml.safe_load(open(os.path.join(root, "examples/straka.yaml")))
    cfg["distribute"]["nb2"] = 1
    cfg["outputs"] = []
    cfg["dynamics"]["precision"] = "float64"
    cfg["integration"]["ncycle_out"] = 0
    cfg["integration"].pop("execution", None)
    if execution is not None:
        cfg["integration"]["execution"] = execution
    os.environ["DEVICE"] = device.type
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)
    block.to(device)
    eos = cfg["dynamics"]["equation-of-state"]
    rd = 8.31446261815324 / eos["weight"]
    cp = rd * eos["gammad"] / (eos["gammad"] - 1.0)
    z = block.module("coord").buffer("x1v").view(1, 1, -1).double()
    x = block.module("coord").buffer("x2v").view(1, -1, 1).double()
    w = block.module("hydro").buffer("D").double().clone().zero_()
    temp = 300.0 - 9.8 * z / cp
    w[kIPR] = 1.0e5 * (temp / 300.0) ** (cp / rd)
    radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
    temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
    w[kIDN] = w[kIPR] / (rd * temp)
    ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return block, state


def run_pair(device):
    """the default deck against the explicit-eager deck on `device`"""
    block, state = make(device)
    ref, expected = make(device, "eager")
    nstage = len(block.intg.stages)
    dt0 = 0.5 * ref.max_time_step(expected)
    for n in range(NSTEPS):
        dt = dt0 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(nstage):
            block.forward(state, dt, s)
            ref.forward(expected, dt, s)
        for k in KEYS:
            d = (state[k].double() - expected[k].double()).abs().max().item()
            assert torch.equal(state[k], expected[k]), (device.type, "step", n, k, d)
    return block.execution_report(), ref.execution_report(), nstage


def cubed(nb, execution=None):
    """a six-panel cubed-sphere Mesh on CUDA, nb x nb blocks per panel"""
    from snapy import Mesh, MeshOptions, kIDN, kIPR

    intg = {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0}
    if execution is not None:
        intg["execution"] = execution
    deck = {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry", "composition": {"O": 0.42, "N": 1.56, "Ar": 0.01},
                     "cv_R": 2.5}],
        "dynamics": {"equation-of-state": {"type": "ideal-gas"}},
        "distribute": {"layout": "cubed-sphere", "nb2": nb, "nb3": nb,
                       "blocks_per_process": 6 * nb * nb, "verbose": False},
        "geometry": {"type": "gnomonic-equiangle",
                     "cells": {"nx1": 1, "nx2": 8, "nx3": 8, "nghost": 2},
                     "bounds": {"x2min_pi": -0.25, "x2max_pi": 0.25,
                                "x3min_pi": -0.25, "x3max_pi": 0.25}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "custom", "x2-outer": "custom",
            "x3-inner": "custom", "x3-outer": "custom"}},
        "integration": intg,
    }
    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck, f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(6 * nb * nb)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    mvars = []
    for b in blocks:
        b.to(torch.device("cuda", 0))
        coord = b.module("coord")
        w = b.module("hydro").buffer("D").double().clone().zero_()
        w[kIDN] = (1.0 + 0.2 * coord.buffer("x2v").double().view(1, -1, 1)
                   + 0.05 * coord.buffer("x3v").double().view(-1, 1, 1))
        w[kIPR] = 1.0e5
        mvars.append({"hydro_w": w})
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def run_cubed(nb):
    mesh, blocks, mvars = cubed(nb)
    ref, _, evars = cubed(nb, "eager")
    reasons = blocks[0].execution_capability(mvars[0])["eager_reasons"]
    for n in range(NSTEPS):
        dt = 1.0e-4 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(3):
            mesh.forward(mvars, dt, s)
            ref.forward(evars, dt, s)
        for i in range(len(blocks)):
            d = (mvars[i]["hydro_u"] - evars[i]["hydro_u"]).abs().max().item()
            assert torch.equal(mvars[i]["hydro_u"], evars[i]["hydro_u"]), (
                "cubed sphere nb", nb, "step", n, "block", i, d)
    return mesh.execution_report(), ref.execution_report(), reasons


def main():
    torch.set_num_threads(1)
    default, eager, nstage = run_pair(torch.device("cpu"))
    print("cpu default:", default, "| explicit eager:", eager, flush=True)
    assert default["captures"] == 0 and default["eager"] == nstage * NSTEPS, default
    assert eager["captures"] == 0 and eager["eager"] == nstage * NSTEPS, eager
    print("cpu: the default runs eager and equals explicit eager bitwise", flush=True)

    if not build_has_cuda():
        skip("CUDA unavailable (CUDA part)")
    default, eager, nstage = run_pair(torch.device("cuda", 0))
    print("cuda default:", default, "| explicit eager:", eager, flush=True)
    assert eager["captures"] == 0 and eager["replays"] == 0, eager
    assert eager["eager"] == nstage * NSTEPS, eager
    assert default["captures"] > 0, ("the default did not capture on CUDA", default)
    assert default["replays"] == nstage * NSTEPS and default["eager"] == 0, default
    print("cuda: the default captures and equals explicit eager bitwise", flush=True)

    default, eager, reasons = run_cubed(1)
    print("cuda cubed sphere 1 block/panel default:", default, flush=True)
    assert default["captures"] > 0 and default["eager"] == 0, default
    assert eager["captures"] == 0, eager
    print("cuda: a cubed-sphere Mesh captures by default and equals explicit eager bitwise",
          flush=True)
    default, eager, reasons = run_cubed(2)
    print("cuda cubed sphere 2x2 blocks/panel default:", default, reasons, flush=True)
    assert default["captures"] == 0 and default["eager"] > 0, default
    assert any("panel split into several blocks" in r for r in reasons), reasons
    print("cuda: split panels run eager by default, with the reason, and equal explicit "
          "eager bitwise", flush=True)
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
