"""V3 B3: a six-panel cubed sphere in one process runs its stages as CUDA
graphs, bitwise to an eager Mesh.

A cubed-sphere stage exchanges across panels inside the hydro step: the face
states and the face-flux velocities across the panel seams, the seam rows of
the flux register (reconcile_panel_seams), and, with a scalar and
flux-covariance, the positivity limiters' theta sync; conversion-order 4 adds
the primitive ghosts. Each is a cut (pack and unpack in the segment's graphs,
the same-process rendezvous on the host between them), and the six blocks
capture together on their worker threads.

Deck: gnomonic-equiangle, 6 panels of 8 x 16 x 16 cells, ideal-moist (dry),
hllc, weno5, rk3, constant radial gravity, solid-body zonal wind and a warm
bubble. Strict cuda-graph against eager, FP64, 10 steps with dt changing
every step, bitwise after every step, four variants: dry; a tracer with
flux-covariance; conversion-order 4; both. One capture per (block, stage,
segment) and replays after. FP32 is not run: on this base float32 requires
perturbation-flux, which requires anomaly storage, which is Cartesian only
(G5: the cubed-sphere gate is FP64).

  python test_cubed_sphere_capture.py
"""
import os
import tempfile

import torch
import yaml

NX, NX1, NSTEPS = 16, 8, 10
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def deck(execution, strict, scalar, co4):
    d = {
        "reference-state": {"Tref": 0.0, "Pref": 1.0e5},
        "species": [{"name": "dry",
                     "composition": {"O": 0.4200039819342909, "N": 1.560014790041652,
                                     "Ar": 0.01},
                     "cv_R": 2.5}],
        "geometry": {"type": "gnomonic-equiangle",
                     "bounds": {"x1min": 6.371e6, "x1max": 6.381e6,
                                "x2min_pi": -0.25, "x2max_pi": 0.25,
                                "x3min_pi": -0.25, "x3max_pi": 0.25},
                     "cells": {"nx1": NX1, "nx2": NX, "nx3": NX, "nghost": 3}},
        "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1,
                       "blocks_per_process": 6, "verbose": False},
        "dynamics": {
            "precision": "float64",
            "flux-covariance": scalar,
            "equation-of-state": {"type": "ideal-moist", "limiter": False,
                                  "density-floor": 1.0e-10, "pressure-floor": 1.0e-10},
            "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                            "horizontal": {"type": "weno5", "scale": False, "shock": False}},
            "riemann-solver": {"type": "hllc"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "custom", "x2-outer": "custom",
            "x3-inner": "custom", "x3-outer": "custom"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0, "nlim": 1,
                        "ncycle_out": 0, "execution": execution,
                        "execution-strict": strict},
        "forcing": {"const-gravity": {"grav1": -9.8}},
    }
    if co4:
        d["dynamics"]["conversion-order"] = 4
    if scalar:
        d["scalar"] = {"nvar": 1, "names": ["tracer"]}
    return d


def make(execution, strict, scalar, co4):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3

    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(execution, strict, scalar, co4), f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(6)
        opts.set_local_horizontal_cells(NX, NX)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    for b in blocks:
        b.to(torch.device("cuda", 0))
    mvars = []
    rd, t0 = 287.0, 280.0
    for ib, b in enumerate(blocks):
        coord = b.module("coord")
        beta, alpha = torch.meshgrid(coord.buffer("x3v").double(),
                                     coord.buffer("x2v").double(), indexing="ij")
        face = int(b.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)
        z = coord.buffer("x1v").double().view(1, 1, -1) - 6.371e6
        bufs = dict(b.named_buffers())
        w = bufs["hydro.D"].double().clone().zero_()
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=alpha.device)
        vel[2] = 20.0 * torch.cos(lat)
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face)
        bubble = (2.0 * torch.exp(-(((lon - 0.3) / 0.2) ** 2 + (lat / 0.2) ** 2)).unsqueeze(-1)
                  * torch.exp(-((z - 5.0e3) / 2.0e3) ** 2))
        p = 1.0e5 * torch.exp(-9.8 * z / (rd * t0))
        w[kIPR] = p + 0.0 * bubble
        w[kIDN] = p / (rd * (t0 + bubble))
        w[kIV1] = vel[0].unsqueeze(-1) + 0.0 * z
        w[kIV2] = vel[1].unsqueeze(-1) + 0.0 * z
        w[kIV3] = vel[2].unsqueeze(-1) + 0.0 * z
        v = {"hydro_w": w}
        if scalar:
            s = bufs["scalar.D"].double().clone().zero_()
            s[0] = (0.5 + 0.5 * torch.tanh((lon - 0.2) / 0.1)).unsqueeze(-1) + 0.0 * z
            v["scalar_r"] = s
        mvars.append(v)
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def check(scalar, co4):
    name = "dry" + (" + tracer, flux-covariance" if scalar else "") + (
        " + conversion-order 4" if co4 else "")
    graph, gb, gv = make("cuda-graph", True, scalar, co4)
    eager, _, ev = make("eager", False, scalar, co4)
    for i, b in enumerate(gb):
        why = b.execution_capability(gv[i])["eager_reasons"]
        assert not why, (name, i, why)
    dt0 = eager.max_time_step(ev)
    for n in range(NSTEPS):
        dt = 0.5 * dt0 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(3):
            graph.forward(gv, dt, s)
            eager.forward(ev, dt, s)
        for i in range(len(gb)):
            for k in KEYS:
                if k not in gv[i]:
                    continue
                d = (gv[i][k].double() - ev[i][k].double()).abs().max().item()
                assert torch.equal(gv[i][k], ev[i][k]), (name, "step", n, "block", i, k, d)
    r = graph.execution_report()
    nb = len(gb)
    assert r["captures"] == 3 * nb and r["graphs"] == 3 * nb, (name, r)
    assert r["replays"] == 3 * nb * NSTEPS and r["eager"] == 0, (name, r)
    assert r["cuts"] > 0, (name, r)
    print(name, ": graph equals eager over", NSTEPS, "steps:", r)


def main():
    if not torch.cuda.is_available() or os.environ.get("SNAPY_BUILD_CUDA", "1") == "0":
        print("SKIP: CUDA unavailable")
        return
    for scalar, co4 in ((False, False), (True, False), (False, True), (True, True)):
        check(scalar, co4)
    print("PASSED")


if __name__ == "__main__":
    main()
