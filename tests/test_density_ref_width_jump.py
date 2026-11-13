#!/usr/bin/env python3
"""The x1 density reference at a cell-width jump: state-error ladder.

Column on [0, 1], g = 1, T = 1 - z/2, p0 = T^2, rho0 = T (hydrostatic), gamma 1.4, with an
isentropic pulse 1e-3 exp(-((z - 0.4)/0.1)^2). The grid has N/2 cells of 1.2/N below z = 0.6 and
N/2 cells of 0.8/N above (width ratio 1.5). Reflecting walls, weno5 + lmars + rk3, constant gravity
(so the x1 hydrostatic split is on), conversion-order 4 and 2, float64. Every run uses the same dt
(set by the reference grid) to t = 0.1.

The error in rho in the two cells next to the jump, against the exact restriction of a nested
N = NREF run, must converge on N = 64 ... 512 at order 4 or better for conversion-order 4
and 1.8 or better for conversion-order 2 (measured: 5.0-5.1 and 3.6, 2.4, 2.1). A density reference
whose rho - dref has a kink at the width change keeps that error flat (order ~0) instead.
"""
import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import yaml
from snapy import Mesh, MeshOptions, kIDN, kIPR

GAMMA, GRAV, NG = 1.4, 1.0, 3
AMP, C0, WIDTH, TEND = 1.0e-3, 0.4, 0.1, 0.1
GLX, GLW = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(8))


def faces(n):
    h = n // 2
    x = np.concatenate([np.linspace(0.0, 0.6, h + 1), np.linspace(0.6, 1.0, n - h + 1)[1:]])
    # the build refuses a rounded width ratio above 1.5: move the jump face down by ulps
    while (x[h] - x[h - 1]) / (x[h + 1] - x[h]) > 1.5:
        x[h] = np.nextafter(x[h], 0.0)
    return x.tolist()


def mesh(n, key):
    weno = {"type": "weno5", "scale": True, "shock": False}
    periodic = {"x2-inner": "periodic", "x2-outer": "periodic", "x3-inner": "periodic", "x3-outer": "periodic"}
    cfg = {"geometry": {"type": "cartesian",
                        "bounds": {"x1min": 0.0, "x1max": 1.0, "x2min": 0.0, "x2max": 1.0,
                                   "x3min": 0.0, "x3max": 1.0},
                        "cells": {"nx1": n, "nx2": 1, "nx3": 1, "nghost": NG}, "x1f": faces(n)},
           "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                              "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                              "temperature-floor": 1.e-12, "limiter": False},
                        "reconstruct": {"vertical": weno, "horizontal": weno},
                        "riemann-solver": {"type": "lmars"}, "conversion-order": key},
           "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                               **periodic}},
           "integration": {"type": "rk3", "cfl": 0.8, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
           "forcing": {"const-gravity": {"grav1": -GRAV}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
    try:
        return Mesh(MeshOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)


def cell_means(xf):
    """exact cell means of (rho, v, p) by 8-point Gauss-Legendre (v = 0)"""
    r = (xf[:-1, None] + xf[1:, None] + (xf[1:, None] - xf[:-1, None]) * GLX) / 2
    d = AMP * torch.exp(-((r - C0) / WIDTH) ** 2)
    t = 1.0 - r / 2
    rho = (t * (1 + d) * GLW).sum(-1) / 2
    p = (t ** 2 * (1 + d) ** GAMMA * GLW).sum(-1) / 2
    return rho, p


def run(n, key, dt, nstep, device):
    m = mesh(n, key)
    b = list(m.blocks)[0]
    b.to(device, torch.float64)
    w = b.buffer("hydro.D").clone().zero_()
    rho, p = cell_means(b.buffer("coord.x1f").double().cpu())
    w[kIDN], w[kIPR] = rho.to(w), p.to(w)
    state = m.initialize([{"hydro_w": w}])[0]
    for k in range(nstep):
        m.set_cycle(k + 1)
        for st in range(len(b.module("intg").stages)):
            m.forward(state, dt, st)
    return state[0]["hydro_u"][kIDN, 0, 0, NG:-NG].double().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nz", default="64,128,256,512")
    ap.add_argument("--nref", type=int, default=2048)
    # conversion order : smallest pairwise order of the jump-cell error
    ap.add_argument("--keys", default="4:4.0,2:1.8")
    a = ap.parse_args()
    nzs = [int(x) for x in a.nz.split(",")]
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("no CUDA device: skipped")
        sys.exit(125)
    nstep = math.ceil(TEND / (0.8 * (0.8 / a.nref) / 1.2))
    dt = TEND / nstep
    failed = []
    for item in a.keys.split(","):
        key, min_order = int(item.split(":")[0]), float(item.split(":")[1])
        t0 = time.time()
        ref = run(a.nref, key, dt, nstep, device)
        err = []
        for n in nzs:
            rho = run(n, key, dt, nstep, device)
            e = (rho - ref.view(n, -1).mean(-1)).abs()
            err.append(float(e[n // 2 - 1:n // 2 + 1].max()))
        orders = [math.log2(err[i] / err[i + 1]) for i in range(len(err) - 1)]
        print(f"key {key}, dt {dt:.6e} x {nstep} steps, N {nzs} vs {a.nref}: jump-cell rho error "
              + " ".join(f"{x:.3e}" for x in err) + ", orders " + " ".join(f"{x:.2f}" for x in orders)
              + f", required >= {min_order} ({time.time() - t0:.1f} s)", flush=True)
        if min(orders) < min_order:
            failed.append(f"key {key}: jump-cell order {min(orders):.2f} < {min_order}")
    assert not failed, "; ".join(failed)
    print("PASSED")


if __name__ == "__main__":
    main()
