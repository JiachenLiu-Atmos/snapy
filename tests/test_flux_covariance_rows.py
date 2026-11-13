#!/usr/bin/env python3
"""#289 all-rows face-flux covariance (switch SNAP_FLUX_COVARIANCE): rest, uniform tracer, offsets.

The x2/x3 face fluxes carry, per row, a covariance part (density-weighted pairs only: tracer
s2 rho D1[u_n] D1[q], dry mass minus their sum, energy s2 rho D1[h] D1[u_n]) and the centroid part
-delta d_1(F); the normal-momentum pressure p* = p - delta d_1 p is mirrored in the geometric
source. Checked, each arm in its own process because the switch is read once per process:
  1. rest: a discretely balanced isothermal shell, spherical-polar (one block) and gnomonic
     (six panels), stays at rest with the term on as well as off (max |v| / c_s);
  2. uniform tracer: on the six-panel moist shell under a solid-body wind, a vapor mass fraction
     that starts uniform stays uniform to round-off with the term on;
  3. offset invariance: shifting the vapor energy offset u0 leaves the primitive state of a sheared,
     stratified moist box unchanged to round-off with the term on (the energy row's shift is then
     exactly the offset times the tracer rows');
  4. dry Cartesian limit: with the term on, the one-step eps_eff nz^2 of
     test_horizontal_flux_covariance stays within 2e-3 of its value recorded with the x1 wall
     continuation of the default reference and the corrected-PE face work at nz 16, 32, 64.

  python test_flux_covariance_rows.py [--device cpu]
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

NG = 3
REST_TOL = 1e-9       # max |v| / c_s after NREST steps
UNIFORM_TOL = 1e-13   # max |q - q0|
OFFSET_TOL = 1e-10    # max relative primitive difference between the two offsets
NREST, NFLOW = 20, 20
# eps_eff nz^2 of the dry Cartesian one-step test, term on, with the x1 wall continuation
# and the corrected-PE face work (gravity-work: face; no dx^2/12 curvature flux)
CART_REF = {16: 0.003339, 32: 0.005166, 64: 0.005633}
CART_TOL = 2e-3
P0, RHO0, G = 1.0e5, 0.1, 10.0

DRY = {"name": "dry", "composition": {"H": 1.64813, "He": 0.17591}, "cv_R": 2.5}
VAPOR = {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 3.5}
CLOUD = {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.0}
# the condensable pair makes vapor and cloud hydro species rows; the columns here stay far below
# saturation (T ~ 280 K, vapor mass fraction <= 0.02), so the cloud row stays zero
REACTIONS = [{"equation": "vapor <=> cloud", "type": "nucleation",
              "rate-constant": {"formula": "h2o_ideal"}}]


def card(geom, moist=False, u0_vapor=None):
    if geom == "spherical-polar":
        geometry = {"type": "spherical-polar",
                    "bounds": {"x1min": 6.0e6, "x1max": 6.4e6, "x2min": 1.2, "x2max": 1.9,
                               "x3min": 0.0, "x3max": 0.7},
                    "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": NG}}
        bc = {"x1-inner": "reflecting", "x1-outer": "reflecting", "x2-inner": "reflecting",
              "x2-outer": "reflecting", "x3-inner": "periodic", "x3-outer": "periodic"}
        dist = None
    elif geom == "gnomonic-equiangle":
        geometry = {"type": "gnomonic-equiangle",
                    "bounds": {"x1min": 6.0e6, "x1max": 6.4e6, "x2min_pi": -0.25, "x2max_pi": 0.25,
                               "x3min_pi": -0.25, "x3max_pi": 0.25},
                    "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": NG}}
        bc = {"x1-inner": "reflecting", "x1-outer": "reflecting", "x2-inner": "custom",
              "x2-outer": "custom", "x3-inner": "custom", "x3-outer": "custom"}
        dist = {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6,
                "verbose": False}
    else:  # cartesian box, x1 vertical
        geometry = {"type": "cartesian",
                    "bounds": {"x1min": 0.0, "x1max": 4.0e5, "x2min": 0.0, "x2max": 8.0e5,
                               "x3min": 0.0, "x3max": 1.0},
                    "cells": {"nx1": 16, "nx2": 32, "nx3": 1, "nghost": NG}}
        bc = {"x1-inner": "reflecting", "x1-outer": "reflecting", "x2-inner": "periodic",
              "x2-outer": "periodic", "x3-inner": "periodic", "x3-outer": "periodic"}
        dist = None
    species = [dict(DRY)]
    if moist:
        v = dict(VAPOR)
        if u0_vapor is not None:
            v["u0_R"] = u0_vapor
        species += [v, dict(CLOUD)]
    cfg = {"reference-state": {"Tref": 300.0, "Pref": 1.0e5}, "species": species,
           "geometry": geometry,
           "dynamics": {"equation-of-state": {"type": "ideal-moist" if moist else "ideal-gas",
                                              "density-floor": 1.e-20, "pressure-floor": 1.e-20,
                                              "limiter": False},
                        "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                                        "horizontal": {"type": "weno5", "scale": True,
                                                       "shock": False}},
                        "riemann-solver": {"type": "lmars"}},
           "boundary-condition": {"external": bc},
           "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0, "nlim": -1,
                           "tlim": 1.e12},
           "forcing": {"const-gravity": {"grav1": -G}}}
    if moist:
        cfg["reactions"] = REACTIONS
    if dist:
        cfg["distribute"] = dist
    return cfg


def blocks_of(cfg, device):
    import snapy
    from snapy import Mesh, MeshOptions, MeshBlock, MeshBlockOptions
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        if "distribute" in cfg:
            options = MeshOptions.from_yaml(tmp)
            options.blocks_per_process(6)
            nx = cfg["geometry"]["cells"]["nx2"]
            options.set_local_horizontal_cells(nx, nx)
            mesh = Mesh(options)
            blocks = list(mesh.blocks)
        else:
            mesh = None
            blocks = [MeshBlock(MeshBlockOptions.from_yaml(tmp))]
    finally:
        os.unlink(tmp)
    for b in blocks:
        b.to(torch.device(device), torch.float64)
    return mesh, blocks


def balanced_w(block, nx1, q=None, wind=None):
    """isothermal hydrostatic column (balance_column), optional uniform species q and wind"""
    import snapy
    from snapy import kIDN, kIPR
    r = block.buffer("coord.x1v")[NG:NG + nx1].cpu()
    dr = block.buffer("coord.dx1f")[NG:NG + nx1].cpu()
    p = P0 * torch.exp(-(r - r[0]) * RHO0 * G / P0)
    col = torch.zeros(kIPR + 1, 1, 1, nx1, dtype=torch.float64)
    col[kIDN, 0, 0] = p * RHO0 / P0
    col[kIPR, 0, 0] = p
    wb, _, _ = snapy.balance_column(col, dr.contiguous(), G)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    for c in (kIDN, kIPR):
        w[c][..., NG:NG + nx1] = wb[c, 0, 0].to(w)
        w[c][..., :NG] = w[c][..., NG:NG + 1]
        w[c][..., NG + nx1:] = w[c][..., NG + nx1 - 1:NG + nx1]
    if q is not None:
        w[kIPR + 1] = q  # vapor, the first species row (ICY); cloud stays 0
    if wind is not None:
        wind(block, w)
    return w


def advance(mesh, blocks, mv, n):
    intg = blocks[0].module("intg")
    for cycle in range(1, n + 1):
        if mesh is not None:
            mesh.set_cycle(cycle)
            dt = mesh.max_time_step(mv)
            for st in range(len(intg.stages)):
                mesh.forward(mv, dt, st)
            assert mesh.check_redo(mv) == 0
        else:
            b = blocks[0]
            b.inc_cycle()
            dt = float(b.max_time_step(mv[0]))
            for st in range(len(intg.stages)):
                b.forward(mv[0], dt, st)
            assert b.check_redo(mv[0]) == 0


def end_prims(blocks, mv):
    out = []
    for ib, b in enumerate(blocks):
        sl = (slice(None),) + tuple(b.part((0, 0, 0), False)[1:])
        w = b.module("hydro.eos").compute("U->W", [mv[ib]["hydro_u"]])  # end of step
        out.append(w[sl].cpu())
    return out


def arm_rest(geom, device):
    from snapy import kIDN, kIPR
    cfg = card(geom)
    mesh, blocks = blocks_of(cfg, device)
    nx1 = cfg["geometry"]["cells"]["nx1"]
    mv = [{"hydro_w": balanced_w(b, nx1)} for b in blocks]
    if mesh is not None:
        mv, _ = mesh.initialize(mv)
    else:
        v, _ = blocks[0].initialize(mv[0])
        mv = [v]
    advance(mesh, blocks, mv, NREST)
    cs = math.sqrt(1.4 * P0 / RHO0)
    return max(float(w[1:4].abs().max()) for w in end_prims(blocks, mv)) / cs


def arm_uniform(device):
    import snapy
    from snapy import kIDN
    q0 = 0.01
    cfg = card("gnomonic-equiangle", moist=True, u0_vapor=-1000.0)
    mesh, blocks = blocks_of(cfg, device)
    nx1 = cfg["geometry"]["cells"]["nx1"]
    mv = []
    for ib, b in enumerate(blocks):
        w = balanced_w(b, nx1, q=q0)
        beta, alpha = torch.meshgrid(b.buffer("coord.x3v").cpu(), b.buffer("coord.x2v").cpu(),
                                     indexing="ij")
        face_id = int(b.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64)
        vel[2] = 30.0 * torch.cos(lat)
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
        for k in range(3):
            w[1 + k] = vel[k].unsqueeze(-1).to(w)
        mv.append({"hydro_w": w})
    mv, _ = mesh.initialize(mv)
    advance(mesh, blocks, mv, NFLOW)
    from snapy import kIPR
    return max(float((w[kIPR + 1] - q0).abs().max()) for w in end_prims(blocks, mv))


def arm_offset(u0, device):
    from snapy import kIDN
    cfg = card("cartesian", moist=True, u0_vapor=u0)
    mesh, blocks = blocks_of(cfg, device)
    b = blocks[0]
    nx1 = cfg["geometry"]["cells"]["nx1"]
    z = b.buffer("coord.x1v").cpu()
    x = b.buffer("coord.x2v").cpu()
    zz = z[None, None, :]
    xx = x[None, :, None]
    q = (0.02 * torch.exp(-zz / 1.5e5)).expand(1, x.numel(), z.numel())

    def shear(block, w):
        w[2] = (20.0 * torch.tanh((zz - 2.0e5) / 5.0e4) + 0.0 * xx).expand_as(w[2]).to(w)
        w[1] = (0.5 * torch.sin(2 * math.pi * xx / 8.0e5) * torch.sin(math.pi * zz / 4.0e5)
                ).expand_as(w[1]).to(w)
    w = balanced_w(b, nx1, q=q.to(torch.float64), wind=shear)
    v, _ = b.initialize({"hydro_w": w.to(device)})
    mv = [v]
    advance(None, blocks, mv, NFLOW)
    return end_prims(blocks, mv)[0]


def arm_cartesian(device):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_horizontal_flux_covariance as hfc
    return {nz: hfc.eps_eff_nz2(nz, device)[0] for nz in CART_REF}


def child(task, out, device):
    if task.startswith("rest:"):
        res = arm_rest(task[5:], device)
        json.dump(res, open(out, "w"))
    elif task == "uniform":
        json.dump(arm_uniform(device), open(out, "w"))
    elif task == "cartesian":
        json.dump(arm_cartesian(device), open(out, "w"))
    elif task.startswith("offset:"):
        torch.save(arm_offset(float(task[7:]), device), out)


def run(task, switch, device, tmpdir):
    env = dict(os.environ)
    env.pop("SNAP_FLUX_COVARIANCE", None)
    if switch is not None:
        env["SNAP_FLUX_COVARIANCE"] = switch
    out = os.path.join(tmpdir, task.replace(":", "_").replace("-", "_") + f"_{switch}.out")
    subprocess.run([sys.executable, __file__, "--child", task, "--out", out, "--device", device],
                   env=env, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.out, a.device)
        return 0
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for geom in ("spherical-polar", "gnomonic-equiangle"):
            off = json.load(open(run(f"rest:{geom}", "0", a.device, tmp)))
            on = json.load(open(run(f"rest:{geom}", "1", a.device, tmp)))
            print(f"rest {geom:20s} max|v|/c_s after {NREST} steps: off {off:.3e}  on {on:.3e}",
                  flush=True)
            if not on < REST_TOL:
                failures.append(f"rest {geom}: on {on:.3e} >= {REST_TOL}")
        dq = json.load(open(run("uniform", "1", a.device, tmp)))
        print(f"uniform vapor, six panels, {NFLOW} steps, term on: max|q - q0| = {dq:.3e}",
              flush=True)
        if not dq < UNIFORM_TOL:
            failures.append(f"uniform tracer drifted: {dq:.3e}")
        for sw in ("0", "1"):
            wa = torch.load(run("offset:0", sw, a.device, tmp))
            wb = torch.load(run("offset:2000", sw, a.device, tmp))
            scale = wa.abs().amax(dim=tuple(range(1, wa.dim())), keepdim=True).clamp_min(1e-300)
            rel = float(((wa - wb).abs() / scale).max())
            print(f"offset u0_R 0 vs 2000, switch {sw}: max rel primitive diff {rel:.3e}",
                  flush=True)
            if sw == "1" and not rel < OFFSET_TOL:
                failures.append(f"offset invariance broken with the term on: {rel:.3e}")
        eps = json.load(open(run("cartesian", "1", a.device, tmp)))
        for nz, ref in CART_REF.items():
            got = eps[str(nz)]
            print(f"dry cartesian nz {nz:3d} eps_eff nz^2 on: {got:+.6f}  (recorded "
                  f"{ref:+.6f}, diff {got - ref:+.1e})", flush=True)
            if not abs(got - ref) < CART_TOL:
                failures.append(f"dry cartesian nz {nz}: {got:+.6f} vs {ref:+.6f}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### flux covariance rows test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
