#!/usr/bin/env python3
"""The x1 face density reference does not read SNAP_WB_REF4.

A stratified Cartesian column (32 x1 cells, reflecting walls, gravity-work: face, rk3, explicit) with a
seeded vertical wind is stepped three times at a fixed dt in a child process, once with SNAP_WB_REF4
unset and once with SNAP_WB_REF4=1. hydro_u must be bitwise equal between the two, for
  plm,   dynamics/conversion-order 2 (no x1 linear rows: the kernel's face density everywhere);
  plm,   dynamics/conversion-order 4 (the fourth-order face density of wb_ref4.cpp on every face, in both runs);
  weno5, dynamics/conversion-order 4 (the x1 linear rows; the fourth-order face density at a guarded face).
The fourth-order face density runs with conversion-order 4 only. RED while SNAP_WB_REF4 also switched it on
at conversion-order 2 (on 53d99726 the plm conversion-order 2 case differs by max |du| 1.6e-6).

  python test_x1_face_density_one_path.py [--device cpu|cuda]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch
import yaml

NZ, DZ, GRAV, RD, T0, PS = 32, 300.0, 9.8, 287.0, 250.0, 1.0e5
NSTEP, DT = 3, 0.5
CASES = (("plm", 2), ("plm", 4), ("weno5", 4))


def config(recon, order):
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": 4.0 * DZ,
                                "x3min": 0.0, "x3max": DZ},
                     "cells": {"nx1": NZ, "nx2": 4, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 8.31446261815324 / RD},
            "reconstruct": {"vertical": {"type": recon, "scale": False, "shock": False},
                            "horizontal": {"type": recon, "scale": False, "shock": False}},
            "riemann-solver": {"type": "lmars"},
            "conversion-order": order},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face", "gravity-work-fixer": False}},
    }


def child(recon, order, device, out):
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(recon, order), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    ng = (w.shape[-1] - NZ) // 2
    z = (torch.arange(-ng, NZ + ng, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    w[kIDN] = (p / (RD * T0)).to(w)
    w[kIPR] = p.to(w)
    x2 = torch.arange(w.shape[-2], dtype=torch.float64).unsqueeze(-1)
    w[kIV1] = (torch.sin(torch.pi * z / (NZ * DZ)) * torch.cos(torch.pi * x2 / 2)).to(w)
    v, _ = block.initialize({"hydro_w": w})
    for _ in range(NSTEP):
        for stage in range(len(block.module("intg").stages)):
            block.forward(v, DT, stage)
    torch.save(v["hydro_u"].cpu(), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", nargs=3, default=None, metavar=("RECON", "ORDER", "OUT"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        return 125
    torch.set_default_dtype(torch.float64)
    if a.child:
        child(a.child[0], int(a.child[1]), a.device, a.child[2])
        return 0
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for recon, order in CASES:
            u = {}
            for switch in (None, "1"):
                env = dict(os.environ)
                env.pop("SNAP_WB_REF4", None)
                if switch:
                    env["SNAP_WB_REF4"] = switch
                out = os.path.join(tmp, f"{recon}_{order}_{switch}.pt")
                subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                                "--child", recon, str(order), out], env=env, check=True)
                u[switch] = torch.load(out)
            diff = (u["1"] - u[None]).abs().max().item()
            same = torch.equal(u["1"], u[None])
            print(f"{recon}, conversion-order {order}: SNAP_WB_REF4=1 vs unset, max |du| {diff:.3e}, "
                  f"bitwise {'equal' if same else 'DIFFERENT'}", flush=True)
            if not same:
                failures.append(f"{recon}, conversion-order {order}: max |du| {diff:.3e}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### x1 face density one-path test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
