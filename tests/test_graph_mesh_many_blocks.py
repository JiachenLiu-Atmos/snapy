"""A Mesh of many blocks captures and equals eager bitwise.

A Cartesian slab Mesh of 8, 24 and 40 blocks along x2 in one process (dry
ideal gas, weno5, lmars, periodic x2), strict cuda-graph against an eager
Mesh, FP64, 3 steps, bitwise after every step. Its blocks capture together
on their worker threads, many segments each: every capture needs a stream
that no other block is using, whether to capture or as its worker stream.

  python test_graph_mesh_many_blocks.py
"""
import os
import sys
import tempfile

import torch
import yaml

from cuda_build import build_has_cuda, skip

NSTEPS = 3


def make(nblocks, execution):
    from snapy import Mesh, MeshOptions, kIDN, kIPR

    deck = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": 1.0e3, "x2min": 0.0,
                                "x2max": 1.0e3 * nblocks, "x3min": 0.0, "x3max": 1.0},
                     "cells": {"nx1": 8, "nx2": 8 * nblocks, "nx3": 1, "nghost": 3}},
        "distribute": {"layout": "slab", "nb2": nblocks, "nb3": 1,
                       "blocks_per_process": nblocks, "verbose": False},
        "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": 1.4,
                                           "weight": 29.0e-3, "limiter": False},
                     "reconstruct": {"vertical": {"type": "weno5"},
                                     "horizontal": {"type": "weno5"}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0,
                        "execution": execution,
                        "execution-strict": execution == "cuda-graph"},
    }
    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck, f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(nblocks)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    mvars = []
    for b in mesh.blocks:
        b.to(torch.device("cuda", 0))
        x2 = b.module("coord").buffer("x2v").double().view(1, -1, 1)
        w = b.module("hydro").buffer("D").double().clone().zero_()
        w[kIDN] = 1.0 + 0.1 * torch.sin(x2 / 1.0e3)
        w[kIPR] = 1.0e5
        mvars.append({"hydro_w": w})
    mvars, _ = mesh.initialize(mvars)
    return mesh, mvars


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    for nblocks in (8, 24, 40):
        graph, gv = make(nblocks, "cuda-graph")
        eager, ev = make(nblocks, "eager")
        for n in range(NSTEPS):
            for s in range(3):
                graph.forward(gv, 1.0e-3, s)
                eager.forward(ev, 1.0e-3, s)
            for i in range(nblocks):
                assert torch.equal(gv[i]["hydro_u"], ev[i]["hydro_u"]), (
                    nblocks, "blocks: step", n, "block", i)
        r = graph.execution_report()
        print(nblocks, "blocks:", r, flush=True)
        assert r["captures"] > 0 and r["eager"] == 0, r
        print(nblocks, "blocks capture and equal eager bitwise", flush=True)
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
