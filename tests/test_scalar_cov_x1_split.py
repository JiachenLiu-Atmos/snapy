#!/usr/bin/env python3
"""A passive scalar under the face-flux covariance (dynamics/flux-covariance, #289) on a column
split in x1: four blocks in one process match one block to round-off.

The tracer is carried twice, as a hydro vapour of density rho_n that never condenses and as a
passive scalar initialised to the same density (see test_scalar_flux_covariance.py). Donor cell,
LMARS, NSTEP explicit rk3 steps on a Cartesian x1-x2 column with reflecting x1 walls and
periodic x2, sheared lateral wind. The column runs as one block and as four x1 blocks in one
process (the test_pref_local_seam layout). Next to an in-process x1 seam the hydro rows take
their x1 differences from the seam-row copies; the scalar row must read the same ones. Two
tracer profiles: a steep exponential and a step on the middle seam. Checked after NSTEP steps:
  (a) max |s - rho_n| / max |rho_n|, on both meshes,
  (b) max |s_4 - s_1| / max |s_1|, the scalar, four blocks against one,
  (c) |M(t) - M(0)| / M(0), the scalar mass, on both meshes,
all <= TOL. The hydro side, four blocks against one (vapour, density, pressure), is printed.

  python test_scalar_cov_x1_split.py [--device cpu|cuda]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

NSTEP, NB1, TOL = 10, 4, 1.e-13
PROFILES = ("steep", "seam-step")


def config(nb1):
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
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 1.e4,
                                "x3min": 0., "x3max": 1.e3},
                     "cells": {"nx1": 32, "nx2": 16, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                  "pressure-floor": 1.e-10, "limiter": False},
            "reconstruct": {"vertical": {"type": "dc"}, "horizontal": {"type": "dc"}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "distribute": {"layout": "cubed" if nb1 > 1 else "slab", "nb1": nb1, "nb2": 1, "nb3": 1,
                       "blocks_per_process": nb1},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -10.}},
        "scalar": {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "dc"}},
    }


def run(nb1, profile, device):
    from snapy import Mesh, MeshOptions, kICY, kIDN, kIPR, kIV1
    cfg = config(nb1)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(nb1)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    assert len(blocks) == nb1, f"{len(blocks)} blocks, expected {nb1}"
    for block in blocks:
        block.to(torch.device(device), torch.float64)

    b = cfg["geometry"]["bounds"]
    lz = b["x1max"] - b["x1min"]
    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        z = coord.buffer("x1v").cpu() - b["x1min"]
        x2v = coord.buffer("x2v").cpu()
        xi = (x2v - b["x2min"]) / (b["x2max"] - b["x2min"])
        w = dict(block.named_buffers())["hydro.D"].clone().zero_().cpu()
        T, R = 320., 3900.
        p = 1.e5 * torch.exp(-z * 10. / (R * T))
        w[kIPR] = p
        w[kIDN] = p / (R * T)
        if profile == "steep":
            qz = 0.002 * torch.exp(-z / (0.08 * lz))
        else:  # a step on the middle seam, x1 = lz / 2
            qz = torch.where(z < 0.5 * lz, 0.002, 0.0002).to(z.dtype)
        q = qz[None, :] * (1. + 0.5 * torch.sin(2. * math.pi * xi))[:, None]
        w[kICY] = q
        w[kIV1 + 1] = (8. * (0.5 + z / lz))[None, :].expand_as(q)
        r = (q / (1. - q)).expand_as(w[kIDN]).unsqueeze(0).clone()
        mesh_vars.append({"hydro_w": w.to(device), "scalar_r": r.to(device)})
    mesh_vars, _ = mesh.initialize(mesh_vars)

    # interior slices, blocks ordered along x1
    order = sorted(range(nb1), key=lambda i: float(blocks[i].module("coord").buffer("x1f")[0]))
    parts = [blocks[i].part((0, 0, 0), False)[1:] for i in order]
    vols = [blocks[i].module("coord").cell_volume()[sl].cpu() for i, sl in zip(order, parts)]

    def column(name, var):
        return torch.cat([mesh_vars[i][name][var][sl].cpu() for i, sl in zip(order, parts)], -1)

    def mass():
        return float(sum((mesh_vars[i]["scalar_s"][0][sl].cpu() * v).sum()
                         for i, sl, v in zip(order, parts, vols)))

    m0 = mass()
    intg = blocks[0].module("intg")
    for cycle in range(1, NSTEP + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
    s, rn = column("scalar_s", 0), column("hydro_u", kICY)
    return {
        "s": s, "vapor": rn, "rho": column("hydro_w", kIDN), "p": column("hydro_w", kIPR),
        "a": float((s - rn).abs().max() / rn.abs().max()),
        "c": abs(mass() - m0) / m0,
        "cloud": float(column("hydro_u", kICY + 1).abs().max()),
        "finite": bool(all(torch.isfinite(v["scalar_s"]).all() and torch.isfinite(v["hydro_u"]).all()
                           for v in mesh_vars)),
    }


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    failures = []
    for profile in PROFILES:
        one, four = run(1, profile, a.device), run(NB1, profile, a.device)
        checks = {
            "(a) max|s - rho_n|/max|rho_n|, 1 block": one["a"],
            f"(a) max|s - rho_n|/max|rho_n|, {NB1} blocks": four["a"],
            f"(b) scalar, {NB1} blocks vs 1": rel(four["s"], one["s"]),
            "(c) scalar mass drift, 1 block": one["c"],
            f"(c) scalar mass drift, {NB1} blocks": four["c"],
        }
        print(f"{profile} (flux-covariance on), {NSTEP} rk3 steps:", flush=True)
        for name, val in checks.items():
            print(f"  {name:42s} {val:.3e}", flush=True)
            if not val <= TOL:
                failures.append(f"{profile}: {name} = {val:.3e} > {TOL:.0e}")
        print(f"  hydro, {NB1} blocks vs 1: vapour {rel(four['vapor'], one['vapor']):.3e}, "
              f"density {rel(four['rho'], one['rho']):.3e}, "
              f"pressure {rel(four['p'], one['p']):.3e}", flush=True)
        for res, nb in ((one, 1), (four, NB1)):
            if not res["finite"] or res["cloud"] != 0.:
                failures.append(f"{profile}, {nb} blocks: finite {res['finite']}, "
                                f"cloud {res['cloud']:.3e}")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
