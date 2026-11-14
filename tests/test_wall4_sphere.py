#!/usr/bin/env python3
"""Key-4 reflecting x1 walls on the cubed sphere (C12 x 30, balanced columns, 20 cos(lat) m/s, VIC 9):
every step accepted and |v1| < 1 m/s (one-sided v ghosts: step rejected at cycle ~11, |v1| ~ 60 m/s)."""
import os, sys, tempfile
import torch, yaml, snapy
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3

A, G, RD, T0, NZ, NG = 6371229.0, 9.80616, 287.0, 250.0, 30, 3
cfg = {"reference-state": {"Tref": 0.0, "Pref": 1.0e5},
       "species": [{"name": "dry", "composition": {"N": 1.5617, "O": 0.4190, "Ar": 0.009713}, "cv_R": 2.5}],
       "geometry": {"type": "gnomonic-equiangle", "cells": {"nx1": NZ, "nx2": 12, "nx3": 12, "nghost": NG},
                    "bounds": {"x1min": A, "x1max": A + 3.0e4, "x2min_pi": -0.25, "x2max_pi": 0.25,
                               "x3min_pi": -0.25, "x3max_pi": 0.25}},
       "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6},
       "dynamics": {"equation-of-state": {"type": "ideal-gas", "limiter": False}, "conversion-order": 4,
                    "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": True},
                                    "horizontal": {"type": "weno5", "scale": True, "shock": True}},
                    "riemann-solver": {"type": "lmars"}},
       "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting", "x2-inner": "custom",
                                           "x2-outer": "custom", "x3-inner": "custom", "x3-outer": "custom"}},
       "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": 9, "nlim": 20, "tlim": 1.0e9},
       "forcing": {"const-gravity": {"grav1": -G}, "coriolis": {"type": "xyz", "omega1": 7.29212e-5, "traditional": True}}}
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
    yaml.safe_dump(cfg, f)
mesh = Mesh(MeshOptions.from_yaml(f.name))
os.unlink(f.name)
state = []
for blk in mesh.blocks:
    co = blk.module("coord")
    x1f = co.buffer("x1f").double()
    beta, alpha = torch.meshgrid(co.buffer("x3v").double(), co.buffer("x2v").double(), indexing="ij")
    face = int(blk.get_layout().loc_of(blk.get_layout().options.rank())[2])
    lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face), alpha, beta)[1]
    p = 1.0e5 * torch.exp(-G * (0.5 * (x1f[NG:NG + NZ] + x1f[NG + 1:NG + NZ + 1]) - A) / (RD * T0))
    wcol = torch.zeros((5,) + alpha.shape + (NZ,), dtype=torch.float64)
    wcol[kIDN], wcol[kIPR] = p / (RD * T0), p
    wb = snapy.balance_column(wcol, x1f[NG + 1:NG + NZ + 1] - x1f[NG:NG + NZ], G, wall_clamp=True, rtol=1.e-13)[0]
    w = dict(blk.named_buffers())["hydro.D"].clone().zero_()
    for s in (kIDN, kIPR):
        w[s, ..., NG:NG + NZ] = wb[s]
        w[s, ..., :NG], w[s, ..., NG + NZ:] = wb[s, ..., :1], wb[s, ..., -1:]
    vel = torch.zeros((3,) + alpha.shape, dtype=torch.float64)
    vel[2] = 20.0 * torch.cos(lat)
    snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face)
    w[kIV2], w[kIV3] = vel[1].unsqueeze(-1), vel[2].unsqueeze(-1)
    state.append({"hydro_w": w})
bv, t = mesh.initialize(state)
for cycle in range(1, 21):
    mesh.set_cycle(cycle)
    dt = mesh.max_time_step(bv)
    for stage in range(3):
        mesh.forward(bv, dt, stage)
    assert mesh.check_redo(bv) == 0, "step rejected at cycle %d" % cycle
vmax = max(float(s["hydro_w"][kIV1].abs().max()) for s in bv)
print("max|v1| %.3e" % vmax)
sys.exit(0 if vmax < 1.0 else 1)
