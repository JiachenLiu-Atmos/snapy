#!/usr/bin/env python3
"""Time-stepped Cartesian and spherical x1 face-work columns (#288).

The discretely balanced 11.3H column must remain at rest through the tested
vertical acoustic Courants up to 250. Results include the first failed rung;
passing a finite ladder does not establish a universal stability threshold.
Cartesian retains eight periodic copies; spherical x1 uses one angular cell
to test a strictly radial column. Angular-mode stability is a separate gate.
--geometry gnomonic-equiangle runs the same ladder on one cubed-sphere face block of
4 x 4 columns at the spherical radius, where the corrected-PE work carries the
centroid term of the derivation's sec 12 (#300).

The face ladder runs in a child process, with the corrected-PE work of
docs/derivations/curved-gravity-work-weight.md sec 7 that gravity-work: face books
inside the implicit operator (#296), including the two wall rows of its slope.
Every rung must stay below W_TOL and end the run below W_SETTLED.
A child that exits non-zero or prints an incomplete ladder fails the parent (#299).
"""
import argparse
import math
import json
import os
import subprocess
import sys
import tempfile

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
COURANTS = (6.6, 65.6, 100.0, 197.0, 250.0)
RADIUS = 7.e7
W_TOL = 1.0e-7  # m/s
W_SETTLED = 1.0e-10  # m/s, max w at the last step: the column has settled


def config(geometry="cartesian", cell_work=False):
    rgas = 8.31446261815324
    cfg = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0, "x2max": 31415926.53589793,
                                "x3min": 0.0, "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 3.4682e-18, "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 9, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": "face",
                                          "gravity-work-fixer": False}},
    }

    if cell_work:
        # Cell work, with the fixer key omitted: the global fixer is on by default with cell.
        cfg["forcing"]["const-gravity"]["gravity-work"] = "cell"
        cfg["forcing"]["const-gravity"].pop("gravity-work-fixer")

    if geometry == "spherical-polar":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"]["nx2"] = 1
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min": math.pi / 2 - 0.2, "x2max": math.pi / 2 + 0.2,
            "x3min": 0.0, "x3max": 0.05}
    elif geometry == "gnomonic-equiangle":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"].update(nx2=4, nx3=4)
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min_pi": -0.25, "x2max_pi": 0.25, "x3min_pi": -0.25, "x3max_pi": 0.25}
        for side in ("x2-inner", "x2-outer", "x3-inner", "x3-outer"):
            cfg["boundary-condition"]["external"][side] = "reflecting"
    return cfg


def run(dt, nstep, device, geometry="cartesian", cell_work=False):

    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(geometry, cell_work), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))

    ng = 3
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * DZ
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (RD * T0)
    col[kIPR, 0, 0] = p
    wb, err, _ = snapy.balance_column(col, torch.full((NZ,), DZ, dtype=torch.float64), GRAV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., ng:ng + NZ] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    block_vars, _ = block.initialize({"hydro_w": w})

    cells = config(geometry)["geometry"]["cells"]
    nx2, nx3 = cells["nx2"], cells["nx3"]
    jl = ng if nx2 > 1 else 0
    kl = ng if nx3 > 1 else 0
    interior = (slice(None), slice(kl, kl + nx3), slice(jl, jl + nx2), slice(ng, ng + NZ))
    nstage = len(block.module("intg").stages)
    wmax, history = 0.0, []
    for n in range(1, nstep + 1):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        redo = block.check_redo(block_vars)
        if redo:
            return False, n, wmax, history, err
        u = block_vars["hydro_u"][interior]
        if not torch.isfinite(u).all():
            return False, n, wmax, history, err
        wn = (u[kIV1] / u[kIDN]).abs().max().item()
        wmax = max(wmax, wn)
        if wmax >= W_TOL:
            return False, n, wmax, history, err
        if n in (1, 10, 20, 30, nstep):
            history.append((n, wn))
    return True, nstep, wmax, history, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=40)
    ap.add_argument("--geometry", choices=("cartesian", "spherical-polar", "gnomonic-equiangle", "both"),
                    default="both")
    ap.add_argument("--courants", type=float, nargs="+", default=COURANTS)
    ap.add_argument("--ladder", action="store_true",
                    help="child: the face ladder only")
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    cs = math.sqrt(GAMMA * RD * T0)
    failures = []
    geometries = ("cartesian", "spherical-polar") if args.geometry == "both" else (args.geometry,)
    if not args.ladder:
        ladder = {}
        for value in ("1",):  # one face ladder, the corrected-PE work always on
            env = dict(os.environ)
            cmd = [sys.executable, os.path.abspath(__file__), "--ladder", "--device", args.device,
                   "--nstep", str(args.nstep), "--geometry", args.geometry,
                   "--courants"] + [str(c) for c in args.courants]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            rungs, summaries = set(), set()
            for line in out.stdout.splitlines():
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                row["radial_exact"] = True
                print(json.dumps(row), flush=True)
                if "first_failing_courant" in row:
                    summaries.add(row["geometry"])
                if "passed" not in row:
                    continue
                rungs.add((row["geometry"], row["courant"]))
                arm = "radial-exact"
                ladder[arm, row["geometry"], row["courant"]] = row
                if not row["passed"]:
                    failures.append((arm, row["geometry"], row["courant"], row["steps"], row["max_w"]))
                elif row["history"][-1][1] > W_SETTLED:
                    failures.append((arm, row["geometry"], row["courant"], "not settled",
                                     row["history"][-1]))
            # a child exits 1 for a failing rung (already listed above) and for an
            # exception alike, so any non-zero exit fails, and the ladder must be complete
            missing = [(g, c) for g in geometries for c in args.courants if (g, c) not in rungs]
            missing += [(g, "summary") for g in geometries if g not in summaries]
            if out.returncode != 0 or missing:
                print(out.stderr[-2000:], flush=True)
                failures.append(("child", value, out.returncode, "missing rows", missing))
    for geometry in (geometries if args.ladder else ()):
        first_failure = None
        for courant in args.courants:
            dt = courant * DZ / cs
            finite, n, wmax, history, err = run(dt, args.nstep, args.device, geometry)
            passed = finite and wmax < W_TOL
            print(json.dumps({"geometry": geometry, "courant": courant, "dt": dt,
                              "steps": n, "passed": passed, "max_w": wmax,
                              "balance_error": err, "history": history}), flush=True)
            if not passed:
                if first_failure is None:
                    first_failure = courant
                failures.append((geometry, courant, n, wmax))
        print(json.dumps({"geometry": geometry, "first_failing_courant": first_failure,
                          "tested_courants": args.courants}), flush=True)
    # Preserve the Cartesian cell work + global fixer coverage alongside the
    # explicitly face-only, fixer-off ladder.
    for dt in (() if args.ladder else (997.0, 100.0)):
        finite, n, wmax, history, err = run(
            dt, args.nstep, args.device, cell_work=True)
        passed = finite and wmax < W_TOL
        print(json.dumps({"arm": "cell-global-fixer", "dt": dt,
                          "steps": n, "passed": passed, "max_w": wmax,
                          "balance_error": err, "history": history}), flush=True)
        if not passed:
            failures.append(("cell-global-fixer", dt, n, wmax))
    for failure in failures:
        print("FAIL", failure, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
