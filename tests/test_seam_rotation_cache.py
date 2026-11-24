"""The panel-seam rotation operands are built once per grid, and rebuilt when
the grid tensors are reallocated.

reconcile_panel_seams (seam_register.cpp) rotates every panel seam's face
fluxes between the covariant face basis and the Cartesian frame each stage.
The metric of that rotation (tan of the face points, delta, C, D, the face
cosine) depends on the grid alone, so it is built once per block side and
reused. Six cubed-sphere panels in one process, eager, CUDA:

  A  reuse: after a warm step, one more step launches no tan kernel (the
     metric is no longer rebuilt every stage). Kernels are counted with the
     CUDA activity of torch.profiler, which sees the Mesh's block threads.
  B  a grid change after setup: the grid of a set-up Mesh changes only by a
     move to another device or dtype (its other grid caches, e.g. the
     exchange's, assume that too). A Mesh stepped on CPU, moved to CUDA and
     stepped again rebuilds the operands (tan kernels in that step) and
     equals bitwise a fresh Mesh given the same state and moved before its
     first step: operands left on the old grid would not.

Without a CUDA build and device the test skips.

  python test_seam_rotation_cache.py
"""
import os
import sys
import tempfile

import torch
import yaml

NX, NX1 = 8, 4
KEYS = ("hydro_u",)


def deck():
    return {
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
                        "ncycle_out": 0},
        "forcing": {"const-gravity": {"grav1": -9.8}},
    }


def make(dev):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3

    os.environ["DEVICE"] = dev.type
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(), f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(6)
        opts.set_local_horizontal_cells(NX, NX)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    for b in blocks:
        b.to(dev)
    mvars = []
    rd, t0 = 287.0, 280.0
    for ib, b in enumerate(blocks):
        coord = b.module("coord")
        beta, alpha = torch.meshgrid(coord.buffer("x3v").double(),
                                     coord.buffer("x2v").double(), indexing="ij")
        face = int(b.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)
        z = coord.buffer("x1v").double().view(1, 1, -1) - 6.371e6
        w = dict(b.named_buffers())["hydro.D"].double().clone().zero_()
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=dev)
        vel[2] = 20.0 * torch.cos(lat)
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face)
        bubble = (2.0 * torch.exp(-(((lon - 0.3) / 0.2) ** 2 + (lat / 0.2) ** 2)).unsqueeze(-1)
                  * torch.exp(-((z - 5.0e3) / 2.0e3) ** 2))
        p = 1.0e5 * torch.exp(-9.8 * z / (rd * t0))
        w[kIPR] = p
        w[kIDN] = p / (rd * (t0 + bubble))
        w[kIV1] = vel[0].unsqueeze(-1) + 0.0 * z
        w[kIV2] = vel[1].unsqueeze(-1) + 0.0 * z
        w[kIV3] = vel[2].unsqueeze(-1) + 0.0 * z
        mvars.append({"hydro_w": w})
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def step(mesh, mvars, dt):
    for s in range(3):
        mesh.forward(mvars, dt, s)


def tan_kernels(mesh, mvars, dt, dev):
    """tan kernels launched in one step (CUDA); None on CPU"""
    if dev.type != "cuda":
        step(mesh, mvars, dt)
        return None
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(mesh, mvars, dt)
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages()
               if e.device_type.name == "CUDA" and "tan" in e.key.lower()
               and "atan" not in e.key.lower())


def move(blocks, mvars, dev):
    for b in blocks:
        b.to(dev)
    return [{k: v[k].to(dev) for k in v} for v in mvars]


def main():
    torch.set_num_threads(1)
    # a CUDA build of snapy (tests/CMakeLists.txt sets SNAPY_BUILD_CUDA=0 for a
    # CPU build) and a device
    if os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return 0
    cpu, cuda = torch.device("cpu"), torch.device("cuda", 0)

    # A: no rebuild of the seam metric in a steady step
    mesh, blocks, mvars = make(cuda)
    dt = 0.5 * mesh.max_time_step(mvars)
    step(mesh, mvars, dt)  # warm: builds whatever is built once
    n = tan_kernels(mesh, mvars, dt, cuda)
    print("A: tan kernels in one steady step:", n, flush=True)
    assert n == 0, f"the panel-seam rotation metric is rebuilt every stage ({n} tan kernels/step)"

    # B: a set-up Mesh moved to another device rebuilds its operands
    mesh, blocks, mvars = make(cpu)
    dt = 0.5 * mesh.max_time_step(mvars)
    step(mesh, mvars, dt)  # builds the operands on CPU
    state = [{k: v[k].clone() for k in KEYS} for v in mvars]
    mvars = move(blocks, mvars, cuda)
    n = tan_kernels(mesh, mvars, dt, cuda)
    print("B: tan kernels in the first step after the move to CUDA:", n, flush=True)
    assert n > 0, "the move to CUDA did not rebuild the seam rotation operands"
    fresh, fblocks, fvars = make(cpu)
    for i, v in enumerate(fvars):
        for k in KEYS:
            v[k].copy_(state[i][k])
    fvars = move(fblocks, fvars, cuda)
    step(fresh, fvars, dt)
    for i in range(len(blocks)):
        for k in KEYS:
            d = (mvars[i][k] - fvars[i][k]).abs().max().item()
            assert torch.equal(mvars[i][k], fvars[i][k]), (
                "stale seam rotation operands after the move", i, k, d)
    print("B: after the move the stepped Mesh equals a fresh one bitwise", flush=True)
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
