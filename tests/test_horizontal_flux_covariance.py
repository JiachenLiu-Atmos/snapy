#!/usr/bin/env python3
"""The x1 covariance of the x2 face energy flux (key dynamics/flux-covariance).

The x2/x3 face energy flux is formed from x1-averaged face states. On an isentropic
background (T0 = 1 - z/Cp, R = g = 1, gamma = 1.4) a convective roll
psi ~ sin(pi z) sin(k x) must leave the entropy unchanged, yet the missing covariance
gamma/(gamma-1) dz^2/12 p d(ln T)/dz du/dz acts as a spurious stratification
eps_spur ~ -(pi^2/12) beta dz^2 = -0.235 / nz^2. One RK3 step of a seeded box minus an
unseeded one gives the entropy tendency S, projected on the seeded w at the roll's k:
  eps_eff = Re sum rho0 (T0 S)^ conj(w^) / (Cp sum rho0 |w^|^2).
The physical value is 0 here. Face gravity work keeps the cell form's own O(dz^2) error
(-0.05 / nz^2 at nz 16 with the term on) out of the number. Checked (Cartesian, nz = 16, 32):
  1. term off: eps_eff nz^2 in [-0.32, -0.20] (the defect is there);
  2. term on:  |eps_eff nz^2| < 0.04;
  3. key unset and flux-covariance: true give the same step bit for bit (the default is on), and
     flux-covariance: false differs from it;
  4. E+P = sum (E + rho g z + g dz^2/12 drho/dz) dV of the seeded box, term on, closes over
     NSTEP steps: with face gravity work E+P is the conserved invariant (hydro.hpp), not E+PE.
The key is read per block, so the three arms (ARMS) run in one process.

  python test_horizontal_flux_covariance.py [--device cpu]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

GAMMA, CP, CV, NG = 1.4, 3.5, 2.5, 3
IV1, IV2 = 1, 2
LX = 2.0 * math.sqrt(2.0)
BASE = (-0.32, -0.20)
FIXED = 0.04
NSTEP = 50
EP_TOL = 1.e-13
ARMS = {"unset": None, "zero": False, "on": True}


def config(nz, cov=None):
    """the deck; cov sets dynamics/flux-covariance (None: the key is absent)"""
    cfg = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": 1.0, "x2min": 0.0, "x2max": LX,
                                "x3min": 0.0, "x3max": 1.0},
                     "cells": {"nx1": nz, "nx2": 2 * nz, "nx3": 1, "nghost": NG}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-12, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -1.0, "gravity-work": "face"}},
    }
    if cov is not None:
        cfg["dynamics"]["flux-covariance"] = cov
    return cfg


def build(nz, seed, device, cov=None):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(nz, cov), f)
        tmp = f.name
    try:
        options = MeshBlockOptions.from_yaml(tmp)
    finally:
        os.unlink(tmp)
    block = MeshBlock(options)
    block.to(torch.device("cpu"))
    coord, eos = block.module("coord"), block.module("hydro.eos")
    x1v, x2v, x3v = coord.buffer("x1v"), coord.buffer("x2v"), coord.buffer("x3v")
    _, X2, X1 = torch.meshgrid(x3v, x2v, x1v, indexing="ij")
    z = X1 - float(coord.buffer("x1f")[NG])
    x = X2
    T0 = 1.0 - z / CP
    w = torch.zeros((eos.nvar(),) + tuple(X1.shape), dtype=X1.dtype)
    w[kIDN], w[kIPR] = torch.pow(T0, CP) / T0, torch.pow(T0, CP)
    I = slice(NG, -NG)
    dx1f = coord.buffer("dx1f")[I].contiguous()
    wb, _, _ = snapy.balance_column(w[:, :, I, I].contiguous(), dx1f, 1.0, True, 3e-14, 400)
    w[:, :, I, I] = wb
    if seed:
        k = 2.0 * math.pi / LX
        zi, xi, rho = z[0, I, I], x[0, I, I], w[kIDN, 0, I, I]
        uu = -math.pi * torch.cos(math.pi * zi) * torch.sin(k * xi) / rho
        ww = k * torch.sin(math.pi * zi) * torch.cos(k * xi) / rho
        amp = 1.e-5 / float(ww.abs().max())
        w[IV1, 0, I, I], w[IV2, 0, I, I] = amp * ww, amp * uu
    block.to(torch.device(device))
    bv, _ = block.initialize({"hydro_w": w.to(device)})
    return block, bv, eos, coord, x[0, I, NG]


def step(block, bv, dt):
    block.inc_cycle()
    for st in range(len(block.intg.stages)):
        block.forward(bv, dt, st)
    assert block.check_redo(bv) == 0


def eps_eff_nz2(nz, device, cov=None):
    """eps_eff nz^2 of one step, and the seeded box's conserved state after it."""
    from snapy import kIDN, kIPR

    b0, bv0, eos, _, x = build(nz, False, device, cov)
    b1, bv1, _, _, _ = build(nz, True, device, cov)
    dt = float(b1.max_time_step(bv1))
    I = slice(NG, -NG)

    def prim(bv):
        return eos.compute("U->W", [bv["hydro_u"]]).cpu()

    def s_of(W):
        return CV * torch.log(W[kIPR, 0, I, I]) - CP * torch.log(W[kIDN, 0, I, I])

    W0, W1 = prim(bv0), prim(bv1)
    step(b0, bv0, dt)
    step(b1, bv1, dt)
    S = ((s_of(prim(bv1)) - s_of(W1)) - (s_of(prim(bv0)) - s_of(W0))) / dt
    k = 2.0 * math.pi / LX
    emk = torch.exp(-1j * k * x.cpu().to(torch.complex128))[:, None]
    nx = x.shape[0]
    rho0 = W0[kIDN, 0, NG, I]
    T0 = W0[kIPR, 0, NG, I] / rho0
    what = (2.0 / nx) * (W1[IV1, 0, I, I] * emk).sum(0)
    Shat = (2.0 / nx) * (S * emk).sum(0)
    num = (rho0 * (T0 * Shat * what.conj()).real).sum()
    den = (CP * rho0 * what.abs() ** 2).sum()
    return float(num / den) * nz ** 2, bv1["hydro_u"].clone()


def ep_drift(nz, device, cov=None):
    """|d(E+P)| / |E+P| of the seeded Cartesian box, term on, over NSTEP steps. P is the
    corrected PE, rho g z less g1 <(z - z_c)^2> drho/dz = dz^2/12 drho/dz with g = 1; the
    slope is the 3-point one of centroid_slope, one-sided at the walls (edge_order 2)."""
    from snapy import kIDN, kIPR

    block, bv, _, coord, _ = build(nz, True, device, cov)
    I = slice(NG, -NG)
    phi = coord.buffer("x1v")[I].to(bv["hydro_u"])
    dz = float(coord.buffer("dx1f")[NG])

    def ep():
        U = bv["hydro_u"][:, :, I, I]
        slope = torch.gradient(U[kIDN], spacing=dz, dim=-1, edge_order=2)[0]
        return float((U[kIPR] + U[kIDN] * phi + dz * dz / 12.0 * slope).sum())

    e0 = ep()
    for _ in range(NSTEP):
        step(block, bv, float(block.max_time_step(bv)))
    return abs(ep() - e0) / abs(e0)


def arm(cov, device):
    """One arm: eps_eff nz^2 at nz 16 and 32, the stepped states, E+P drift."""
    res, u = {}, {}
    for nz in (16, 32):
        res[str(nz)], u[nz] = eps_eff_nz2(nz, device, cov)
    res["drift"] = ep_drift(16, device, cov)
    res["u"] = {nz: v.cpu() for nz, v in u.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    failures = []
    res = {name: arm(cov, a.device) for name, cov in ARMS.items()}
    for nz in (16, 32):
        base, fixed = res["zero"][str(nz)], res["on"][str(nz)]
        print(f"nz {nz:3d}  eps_eff nz^2: off {base:+.5f}  on {fixed:+.5f}", flush=True)
        if not BASE[0] <= base <= BASE[1]:
            failures.append(f"nz {nz}: off {base:+.5f} not in {BASE}")
        if not abs(fixed) < FIXED:
            failures.append(f"nz {nz}: on {fixed:+.5f}, |.| >= {FIXED}")
        if not torch.equal(res["unset"]["u"][nz], res["on"]["u"][nz]):
            failures.append(f"nz {nz}: key unset and true differ (the default is on)")
        if torch.equal(res["on"]["u"][nz], res["zero"]["u"][nz]):
            failures.append(f"nz {nz}: the key changed nothing")
    drift = res["on"]["drift"]
    print(f"nz  16  E+P drift over {NSTEP} steps, term on: {drift:.2e}", flush=True)
    if not drift <= EP_TOL:
        failures.append(f"E+P drift {drift:.2e} > {EP_TOL}")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
