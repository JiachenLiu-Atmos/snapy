#!/usr/bin/env python3
"""#289, SNAP_WB_REF4: the order of the one-step spurious stratification of a neutral column.

The oracle of the fourth-order well-balanced reference (docs/derivations/wb-ref4.md, section 6), Cartesian:

  neutral polytrope, R_d = 1, gamma 5/3, m = 1/(gamma-1), g = m+1:
      T0 = 1 + Lz - z, rho0 = T0^m, p0 = T0^(m+1),  Lz = exp(n/(m+1)) - 1 for n pressure e-folds;
  overturning mode seeded as MOMENTUM (divergence-free):
      rho0 v1 = A k sin(qz) cos(kx),  rho0 v2 = -A q cos(qz) sin(kx),  q = pi/Lz, k = 2 pi/Lx, Lx = 2 Lz;
  IC = exact cell averages (4x4 Gauss-Legendre) of rho, m1, m2, E;
  one RK3 step at dt = 0.3 dz / sqrt(gamma T_bottom), the flux covariance on, face gravity work;
  ds = [(s1 - s0)_mode - (s1 - s0)_rest] / dt per cell, s = ln(p rho^-gamma), from the end-of-step state;
  N2_eff = (g/gamma) sum (-ds) w / sum w^2,  w = v1 at t = 0.

For this column the exact cell-average tendencies of rho and E vanish at linear order, so every N2_eff is
scheme error. The kernel's density reference leaves an O(dz^2) face-density offset: N2_eff nz^2 tends to a
constant (observed order ~2). With SNAP_WB_REF4 it is O(dz^4), and what is left is a wall-band term of
order ~3. Asserted, at 1 and 3 e-folds, nz 32/64/128:
  switch on:  observed order of |N2_eff| >= ORDER_ON at the LAST doubling, where the asymptotic term
              dominates. Not over every doubling: the switched error can cross zero between the two
              coarsest grids (3 e-folds, face reference: +8.6e-7, -1.5e-6, -1.7e-7, -1.4e-8 at
              nz 32/64/128/256), and an order read across that crossing is meaningless, while the
              doublings after it read 3.12 then 3.63 and keep rising towards 4;
  switch on:  and |N2_eff| on the coarsest grid at least COARSE_GAIN below the switch-off arm's, so a
              reference that only converges, from a coarse level no better than the kernel's, fails;
  switch off: below ORDER_OFF at the last doubling, so an ignored or misspelled switch fails.
The switch is read once per process, so each arm runs in its own process.

  python test_wb_ref4_order.py [--device cpu|cuda]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

NZS = (32, 64, 128)
EFOLDS = (1.0, 3.0)
ORDER_ON = 2.75
ORDER_OFF = 2.5
COARSE_GAIN = 10.0
AMP = 1.0e-4
GAMMA = 5.0 / 3.0
M = 1.0 / (GAMMA - 1.0)
G = M + 1.0
NG, IV1, IV2 = 3, 1, 2
GLX, GLW = np.polynomial.legendre.leggauss(4)


def config(nz, lz):
    return {"geometry": {"type": "cartesian",
                         "bounds": {"x1min": 0.0, "x1max": lz, "x2min": 0.0, "x2max": 2.0 * lz,
                                    "x3min": 0.0, "x3max": 1.0},
                         "cells": {"nx1": nz, "nx2": 2 * nz, "nx3": 1, "nghost": NG}},
            "dynamics": {"equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                               "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                               "temperature-floor": 1.e-12, "limiter": True},
                         "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                                         "horizontal": {"type": "weno5", "scale": True, "shock": False}},
                         "riemann-solver": {"type": "lmars"}},
            "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                                "x2-inner": "periodic", "x2-outer": "periodic",
                                                "x3-inner": "periodic", "x3-outer": "periodic"}},
            "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
            "forcing": {"const-gravity": {"grav1": -G, "gravity-work": "face"}}}


def cell_average_prims(zf, xf, lz, amp):
    """exact (4x4 Gauss-Legendre) cell averages of rho, m1, m2, E -> primitives [rho, v1, v2, p]"""
    q, k = math.pi / lz, math.pi / lz  # Lx = 2 Lz
    zc, hz = 0.5 * (zf[1:] + zf[:-1]), 0.5 * (zf[1:] - zf[:-1])
    xc, hx = 0.5 * (xf[1:] + xf[:-1]), 0.5 * (xf[1:] - xf[:-1])
    acc = np.zeros((4, len(xc), len(zc)))
    for a, wa in zip(GLX, GLW):
        for b, wb in zip(GLX, GLW):
            z = (zc + a * hz)[None, :]
            x = (xc + b * hx)[:, None]
            t0 = 1.0 + lz - z
            rho, p = t0 ** M, t0 ** (M + 1.0)
            m1 = amp * k * np.sin(q * z) * np.cos(k * x)
            m2 = -amp * q * np.cos(q * z) * np.sin(k * x)
            e = p / (GAMMA - 1.0) + (m1 * m1 + m2 * m2) / (2.0 * rho)
            acc += 0.25 * wa * wb * np.stack(np.broadcast_arrays(rho, m1, m2, e))
    rho, m1, m2, e = acc
    return rho, m1 / rho, m2 / rho, (GAMMA - 1.0) * (e - (m1 * m1 + m2 * m2) / (2.0 * rho))


def build(nz, lz, amp, device):
    import yaml
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(nz, lz), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device), torch.float64)
    zf = block.buffer("coord.x1f").cpu().numpy()
    xf = block.buffer("coord.x2f").cpu().numpy()
    w = torch.zeros((5, 1, len(xf) - 1, len(zf) - 1), dtype=torch.float64)
    rho, v1, v2, p = cell_average_prims(zf, xf, lz, amp)
    for c, v in ((kIDN, rho), (IV1, v1), (IV2, v2), (kIPR, p)):
        w[c, 0] = torch.from_numpy(v)
    bv, _ = block.initialize({"hydro_w": w.to(device)})
    return block, bv, w


def n2_eff(nz, efolds, device):
    from snapy import kIDN, kIPR
    lz = math.exp(efolds / (M + 1.0)) - 1.0
    dt = 0.3 * (lz / nz) / math.sqrt(GAMMA * (1.0 + lz))
    inner = slice(NG, -NG)
    ds = []
    for amp in (0.0, AMP):
        block, bv, w0 = build(nz, lz, amp, device)
        eos = block.module("hydro.eos")
        s = []
        for _ in range(2):
            w = eos.compute("U->W", [bv["hydro_u"]]).cpu()[:, 0, inner, inner]
            s.append(torch.log(w[kIPR]) - GAMMA * torch.log(w[kIDN]))
            if len(s) == 1:
                block.inc_cycle()
                for st in range(len(block.intg.stages)):
                    block.forward(bv, dt, st)
                assert block.check_redo(bv) == 0
        ds.append(s[1] - s[0])
    vz = w0[IV1, 0, inner, inner]
    return float(G / GAMMA * (-(ds[1] - ds[0]) / dt * vz).sum() / (vz * vz).sum())


def child(device):
    return {str(n): [n2_eff(nz, n, device) for nz in NZS] for n in EFOLDS}


def run(switch, device, tmpdir):
    env = dict(os.environ)
    env.pop("SNAP_WB_REF4", None)
    env["SNAP_FLUX_COVARIANCE"] = "1"
    if switch is not None:
        env["SNAP_WB_REF4"] = switch
    out = os.path.join(tmpdir, f"order_{switch}.json")
    subprocess.run([sys.executable, __file__, "--child", "--out", out, "--device", device],
                   env=env, check=True)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        return 125
    if a.child:
        json.dump(child(a.device), open(a.out, "w"))
        return 0
    failures = []
    arms = {}
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for switch in (None, "1"):
            arms[switch] = res = run(switch, a.device, tmp)
            for n in EFOLDS:
                v = res[str(n)]
                order = [math.log2(abs(v[i]) / abs(v[i + 1])) for i in range(len(v) - 1)]
                print(f"SNAP_WB_REF4 {switch or 'unset'}, {n:g} e-folds: N2_eff nz^2 "
                      + " / ".join(f"{x * nz * nz:+.4f}" for x, nz in zip(v, NZS))
                      + "  (nz " + "/".join(map(str, NZS)) + "), order "
                      + ", ".join(f"{o:.2f}" for o in order), flush=True)
                if switch and order[-1] < ORDER_ON:
                    failures.append(f"{n:g} e-folds, switch on: order {order[-1]:.2f} < {ORDER_ON} at the "
                                    f"last doubling (nz {NZS[-2]} -> {NZS[-1]})")
                if not switch and not order[-1] < ORDER_OFF:
                    failures.append(f"{n:g} e-folds, switch off: order {order[-1]:.2f} >= {ORDER_OFF}, "
                                    "the oracle does not tell the references apart")
    for n in EFOLDS:
        off, on = (abs(arms[s][str(n)][0]) for s in (None, "1"))
        print(f"SNAP_WB_REF4 1 vs unset, {n:g} e-folds, nz {NZS[0]}: |N2_eff| {on:.3e} vs {off:.3e}, "
              f"gain {off / on:.1f}x (needs {COARSE_GAIN:g}x)", flush=True)
        if on * COARSE_GAIN > off:
            failures.append(f"{n:g} e-folds, switch on: coarse |N2_eff| {on:.3e} is not {COARSE_GAIN:g}x "
                            f"below the switch-off {off:.3e}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### wb_ref4 order test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
