#!/usr/bin/env python3
"""SNAP_X1_CENTROID_EXACT on blocks with few x1 cells (#298).

The x1 stencil caches (x1_plain_mean_stencils, x1_pressure_source_stencils)
are not usable when a block has too few owned x1 cells; they then leave their weights undefined
and the kernel's values stand. The cache-validity check must not read the device of those
undefined weights: before the fix the second RK stage threw "tensor does not have a device".

Each case runs two RK3 steps of a resting isothermal column in one block of nx1 cells (reflecting
x1 walls, weno5, explicit, gravity-work: face) in its own process, since the switches are read
once per process. A case passes when every stage runs and the state stays finite, or when the
block is refused at setup with a clean error; an exception during a step fails it.
  SNAP_X1_CENTROID_EXACT=1, spherical-polar, nx1 = 2..6;
  the switch-off controls: Cartesian at nx1 = 2..5, spherical-polar at nx1 = 2 and 3.

  python test_stencil_cache_small_block.py [--device cpu|cuda]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import torch
import yaml

GRAV, RD, T0, PS, DZ, R0 = 9.8, 287.0, 300.0, 1.0e5, 500.0, 6.4e6
NSTEP = 2
CASES = ([("SNAP_X1_CENTROID_EXACT", "1", "spherical-polar", n) for n in (2, 3, 4, 5, 6)]
         + [("SNAP_X1_CENTROID_EXACT", "0", "cartesian", n) for n in (2, 3, 4, 5)]
         + [("SNAP_X1_CENTROID_EXACT", "0", "spherical-polar", n) for n in (2, 3)])
SWITCHES = ("SNAP_X1_CENTROID_EXACT",)


def config(nx1, geometry):
    x1min = R0 if geometry == "spherical-polar" else 0.0
    bounds = {"x1min": x1min, "x1max": x1min + nx1 * DZ, "x2min": 0.0, "x2max": 1.0e4,
              "x3min": 0.0, "x3max": 1.0e4}
    if geometry == "spherical-polar":
        bounds.update(x2min=math.pi / 2 - 0.01, x2max=math.pi / 2 + 0.01, x3min=0.0, x3max=0.01)
    return {
        "geometry": {"type": geometry, "bounds": bounds,
                     "cells": {"nx1": nx1, "nx2": 1, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 8.31446261815324 / RD},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face", "gravity-work-fixer": False}},
    }


def child(nx1, geometry, device):
    """prints SETUP-ERROR, STEP-ERROR or OK and exits 0 (the parent judges)"""
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(nx1, geometry), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
        block.to(torch.device(device))
    except Exception as e:  # noqa: BLE001 -- a clean refusal at setup is allowed
        print("SETUP-ERROR " + str(e).splitlines()[0], flush=True)
        return
    finally:
        os.unlink(tmp)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    ng = (w.shape[-1] - nx1) // 2
    z = (torch.arange(-ng, nx1 + ng, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    w[kIDN] = (p / (RD * T0)).to(w)
    w[kIPR] = p.to(w)
    v, _ = block.initialize({"hydro_w": w})
    nstage = len(block.module("intg").stages)
    for step in range(NSTEP):
        for stage in range(nstage):
            try:
                block.forward(v, 1.0, stage)
            except Exception as e:  # noqa: BLE001 -- reported to the parent
                print(f"STEP-ERROR step {step} stage {stage}: " + str(e).splitlines()[0], flush=True)
                return
    finite = bool(torch.isfinite(v["hydro_u"]).all())
    print(f"OK {NSTEP} steps of {nstage} stages, finite {finite}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", nargs=2, default=None, metavar=("NX1", "GEOMETRY"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    if a.child:
        child(int(a.child[0]), a.child[1], a.device)
        return
    failures = []
    for switch, value, geometry, nx1 in CASES:
        env = dict(os.environ)
        for s in SWITCHES:
            env.pop(s, None)
        env[switch] = value
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                              "--child", str(nx1), geometry], env=env, capture_output=True, text=True)
        verdict = [line for line in out.stdout.splitlines()
                   if line.startswith(("OK", "SETUP-ERROR", "STEP-ERROR"))]
        line = verdict[-1] if verdict else f"no verdict (exit {out.returncode})"
        print(f"{switch}={value} {geometry:15s} nx1 {nx1}: {line}", flush=True)
        if out.returncode != 0 or not verdict:
            print(out.stderr[-2000:], flush=True)
            failures.append(f"{switch}={value} nx1 {nx1}: child exit {out.returncode}")
        elif line.startswith("STEP-ERROR") or line.endswith("finite False"):
            failures.append(f"{switch}={value} nx1 {nx1}: {line}")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
