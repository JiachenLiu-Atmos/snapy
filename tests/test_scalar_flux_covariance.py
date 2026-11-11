#!/usr/bin/env python3
"""A passive scalar and the same tracer carried as a hydro species agree to round-off, with and
without the x2/x3 face-flux covariance (dynamics/flux-covariance, #289).

One run carries the tracer twice: as a hydro vapour of density rho_n, kept far below saturation
so it never condenses (its cloud stays 0), and as a passive scalar initialised to the same density (r = q / (1 - q), s = rho_dry r = rho q). Both
reconstruct with donor cell, so the scalar flux F_dry r_up and the LMARS species flux rho_up q_up
are the same flux to round-off. With the covariance on, the species row carries
sigma1^2 rho D1[u_n] D1[q] plus the centroid part -delta d1(rho u_n q), while the dry mass flux
the scalar rides on carries the dry row; the scalar must carry the species row, not r times the
dry one. Checked after NSTEP explicit rk3 steps:
  max |s - rho_n| / max |rho_n| <= TOL, key on and off,
on a sheared Cartesian x1-x2 column and on one cubed-sphere face block (where the centroid part is
nonzero). Each arm runs in a child process (kintera's species table is per process).

  python test_scalar_flux_covariance.py [--device cpu|cuda] [--save DIR]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import torch
import yaml

NSTEP, TOL = 10, 1.e-12
CASES = ("cartesian", "gnomonic-equiangle")


def config(geometry):
    if geometry == "cartesian":
        bounds = {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 1.e4, "x3min": 0., "x3max": 1.e3}
        cells = {"nx1": 32, "nx2": 16, "nx3": 1, "nghost": 3}
        lateral = ("periodic", "periodic", "periodic", "periodic")
    else:  # one face block at a radius where the centroid offset is not small
        bounds = {"x1min": 3.e4, "x1max": 4.e4, "x2min_pi": -0.25, "x2max_pi": 0.25,
                  "x3min_pi": -0.25, "x3max_pi": 0.25}
        cells = {"nx1": 32, "nx2": 8, "nx3": 8, "nghost": 3}
        lateral = ("reflecting",) * 4
    return {
        "reference-state": {"Tref": 0., "Pref": 1.e5},
        "species": [
            {"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
            {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
            {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.},
        ],
        # a species is a hydro species only through a reaction; this one never fires (q << q_sat)
        "reactions": [{"equation": "vapor <=> cloud", "type": "nucleation",
                       "rate-constant": {"formula": "h2o_ideal"}}],
        "geometry": {"type": geometry, "bounds": bounds, "cells": cells},
        "dynamics": {
            "equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                  "pressure-floor": 1.e-10, "limiter": False},
            "reconstruct": {"vertical": {"type": "dc"}, "horizontal": {"type": "dc"}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": lateral[0], "x2-outer": lateral[1],
            "x3-inner": lateral[2], "x3-outer": lateral[3]}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -10.}},
        "scalar": {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "dc"}},
    }


def run(geometry, device, save, switch=None):
    """switch "0"/"1" sets dynamics/flux-covariance; None leaves the key out"""
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1
    cfg = config(geometry)
    if switch is not None:
        cfg["dynamics"]["flux-covariance"] = switch == "1"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device), torch.float64)
    coord = block.module("coord")
    x1v = coord.buffer("x1v").cpu()
    x2v = coord.buffer("x2v").cpu()
    b = cfg["geometry"]["bounds"]
    z = x1v - float(b["x1min"])
    lz = float(b["x1max"]) - float(b["x1min"])
    xi = (x2v - x2v[0]) / (x2v[-1] - x2v[0])
    w = dict(block.named_buffers())["hydro.D"].clone().zero_().cpu()
    T, R = 320., 3900.
    p = 1.e5 * torch.exp(-z * 10. / (R * T))
    w[kIPR] = p
    w[kIDN] = p / (R * T)
    # the tracer varies along x1 and x2; the lateral wind is sheared in x1, so
    # D1[u_n] D1[q] is nonzero on the x2 faces
    q = 0.002 * torch.exp(-z / (0.5 * lz))[None, :] * (1. + 0.5 * torch.sin(2. * math.pi * xi))[:, None]
    w[kICY] = q
    w[kIV1 + 1] = (8. * (0.5 + z / lz))[None, :].expand_as(q)
    if geometry != "cartesian":  # and on the x3 faces too
        w[kIV1 + 2] = (5. * (0.25 + z / lz))[None, :].expand_as(q)
    r = (q / (1. - q)).expand_as(w[kIDN]).unsqueeze(0).clone()
    v, _ = block.initialize({"hydro_w": w.to(device), "scalar_r": r.to(device)})
    ng = cfg["geometry"]["cells"]["nghost"]
    nx2, nx3 = cfg["geometry"]["cells"]["nx2"], cfg["geometry"]["cells"]["nx3"]
    sl = (slice(ng, ng + nx3) if nx3 > 1 else slice(None), slice(ng, ng + nx2), slice(ng, -ng))
    out = {"geometry": geometry, "device": device,
           "switch": switch or "unset"}
    stages = len(block.module("intg").stages)
    for n in range(1, NSTEP + 1):
        dt = block.max_time_step(v)
        for st in range(stages):
            block.forward(v, dt, st)
        s, rn = v["scalar_s"][0][sl].cpu(), v["hydro_u"][kICY][sl].cpu()
        out["cloud"] = float(v["hydro_u"][kICY + 1][sl].abs().max())
        diff = float((s - rn).abs().max() / rn.abs().max())
        if n == 1:
            out["step1"] = diff
    out["final"] = diff
    out["finite"] = bool(torch.isfinite(v["scalar_s"]).all() and torch.isfinite(v["hydro_u"]).all())
    if save:
        torch.save({"scalar_s": v["scalar_s"].cpu(), "hydro_u": v["hydro_u"].cpu()},
                   os.path.join(save, f"{geometry}_{out['switch']}.pt"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", default=None, choices=CASES)
    ap.add_argument("--save", default=None, help="also save the final states to this directory")
    ap.add_argument("--flux-covariance", choices=("0", "1"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    if a.child:
        print(json.dumps(run(a.child, a.device, a.save, a.flux_covariance)), flush=True)
        return
    failures = []
    for geometry in CASES:
        for switch in ("1", "0"):
            cmd = [sys.executable, os.path.abspath(__file__), "--device", a.device, "--child", geometry,
                   "--flux-covariance", switch]
            if a.save:
                cmd += ["--save", a.save]
            out = subprocess.run(cmd, capture_output=True, text=True)
            rows = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if out.returncode or not rows:
                print(out.stderr[-2000:], flush=True)
                failures.append(f"{geometry} switch {switch}: child exit {out.returncode}")
                continue
            res = json.loads(rows[-1])
            print(f"{geometry:18s} flux-covariance={res['switch']:5s}: max|s - rho_n|/max|rho_n| "
                  f"step 1 {res['step1']:.3e}, step {NSTEP} {res['final']:.3e}", flush=True)
            if not (res["finite"] and res["final"] <= TOL and res["step1"] <= TOL and res["cloud"] == 0.):
                failures.append(f"{geometry} switch {res['switch']}: scalar and species differ "
                                f"by {res['final']:.3e}")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
