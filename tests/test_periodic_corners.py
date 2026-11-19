#!/usr/bin/env python3
"""Periodic corners (V3 S step (2), identity P; S.rst rev 2.4 section 5).

On a domain periodic in x2 and x3, every x2-x3 corner ghost of every block
must equal the periodic image of the interior cell it stands for, after
initialize and after one stage. Before CornerRule::diagonal the corners were
0.5 * (edge + edge) on every layout: 36 of 36 corner cells wrong on one slab
block, 144 of 144 on 2 x 2 blocks. Configs: slab 1 block, slab 2 x 2 blocks,
cubed 2 x 2 blocks with pz = 2, all in one process.

  python test_periodic_corners.py [cpu|cuda]
"""
import math, os, sys, tempfile
import torch, yaml
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3

N1, N2, N3, NG = 8, 12, 12, 3

def deck(layout, nb1, nb2, nb3):
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": 1., "x2min": 0., "x2max": 1., "x3min": 0., "x3max": 1.},
                     "cells": {"nx1": N1, "nx2": N2, "nx3": N3, "nghost": NG}},
        "distribute": {"layout": layout, "nb1": nb1, "nb2": nb2, "nb3": nb3,
                       "blocks_per_process": nb1 * nb2 * nb3},
        "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 29.e-3,
                                           "density-floor": 1.e-10, "pressure-floor": 1.e-10, "limiter": False},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False, "shock": False}},
                     "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
    }

def field(x1, x2, x3):
    # smooth, periodic in x2 and x3, distinct in every cell
    return (1. + 0.1 * torch.sin(2 * math.pi * x2) + 0.07 * torch.cos(2 * math.pi * x3)
            + 0.05 * torch.sin(2 * math.pi * (x2 + 2 * x3)) + 0.03 * x1)

def run(layout, nb1, nb2, nb3, device):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(deck(layout, nb1, nb2, nb3), f); f.close()
    mesh = Mesh(MeshOptions.from_yaml(f.name)); os.unlink(f.name)
    blocks = list(mesh.blocks)
    dev = torch.device(device)
    mv = []
    for b in blocks:
        b.to(dev, torch.float64)
        buf = dict(b.named_buffers())
        x1, x2, x3 = buf["coord.x1v"].cpu(), buf["coord.x2v"].cpu(), buf["coord.x3v"].cpu()
        X3, X2, X1 = torch.meshgrid(x3, x2, x1, indexing="ij")
        w = buf["hydro.D"].detach().clone().zero_().cpu()
        w[kIDN] = field(X1, X2, X3); w[kIPR] = 1.e5 * field(X1, X3, X2)
        w[kIV1] = 0.; w[kIV2] = 10. * field(X1, X2, X3) - 10.; w[kIV3] = 5. * field(X1, X3, X2) - 5.
        mv.append({"hydro_w": w.to(dev)})
    mv, _ = mesh.initialize(mv)
    res = {"init": check(blocks, mv)}
    intg = blocks[0].module("intg")
    mesh.set_cycle(1)
    dt = mesh.max_time_step(mv)
    mesh.forward(mv, dt, 0)
    res["stage1"] = check(blocks, mv)
    return res

def check(blocks, mv):
    """corner ghosts vs the periodic image, by global cell index"""
    # global interior map (x3, x2, x1-interior) per variable
    glob = {}
    loc = []
    for b, v in zip(blocks, mv):
        buf = dict(b.named_buffers())
        x2, x3 = buf["coord.x2v"].cpu(), buf["coord.x3v"].cpu()
        i2 = torch.floor(x2 * N2).long(); i3 = torch.floor(x3 * N3).long()
        # the global x1 index of the block's first interior cell (pz > 1)
        k0 = int(torch.floor(buf["coord.x1v"].cpu()[NG] * N1))
        u = v["hydro_u"].detach().cpu()
        loc.append((u, i2, i3, k0))
        n3, n2, n1 = u.shape[1] - 2 * NG, u.shape[2] - 2 * NG, u.shape[3] - 2 * NG
        for jj in range(NG, NG + n3):
            for ii in range(NG, NG + n2):
                for kk in range(n1):
                    glob[(int(i3[jj]), int(i2[ii]), k0 + kk)] = u[:, jj, ii, NG + kk]
    tot = bad = 0; worst = 0.
    for u, i2, i3, k0 in loc:
        n3, n2, n1 = u.shape[1] - 2 * NG, u.shape[2] - 2 * NG, u.shape[3] - 2 * NG
        for jj in list(range(NG)) + list(range(NG + n3, n3 + 2 * NG)):
            for ii in list(range(NG)) + list(range(NG + n2, n2 + 2 * NG)):
                tot += 1
                d = 0.
                for kk in range(n1):
                    ref = glob[(int(i3[jj]) % N3, int(i2[ii]) % N2, k0 + kk)]
                    d = max(d, float((u[:, jj, ii, NG + kk] - ref).abs().max()) / float(ref.abs().max()))
                bad += int(d > 0); worst = max(worst, d)
    return tot, bad, worst

if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_num_threads(1)
    fails = 0
    for name, cfg in (("slab 1 block", ("slab", 1, 1, 1)), ("slab 2x2", ("slab", 1, 2, 2)),
                      ("cubed 2x2 pz2", ("cubed", 2, 2, 2))):
        r = run(*cfg, dev)
        for when, (tot, bad, worst) in r.items():
            print(f"P {name:14s} {when:6s} corner cells {tot:4d} wrong {bad:4d} max rel {worst:.3e}")
            fails += bad
    sys.exit(1 if fails else 0)
