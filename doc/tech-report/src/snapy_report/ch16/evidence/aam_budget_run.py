"""Measured global budgets of a six-panel cubed-sphere run at the code pin.

This is a RUN record, not one of the chapter's checks: it needs a built snapy
in the active Python and is not part of `src/tests`. The committed check
`src/snapy_report/ch16/checks/invariants_budget_check.py` runs the same three
budgets on a port of the operators and needs nothing but numpy.

Six gnomonic-equiangular panels in one process on a shell, closed radial walls,
a discretely balanced isothermal rest state plus a solid-body zonal wind. Each
cycle it measures, over the interiors of all six panels,

  M   = sum rho V                               [kg]
  E   = sum (total energy + rho g (r - r_in)) V [J]
  L_z = sum rho u_east r cos(lat) V             [kg m^2 s^-1]

The energy total carries the gravitational potential, because constant radial
gravity moves energy between the stored total-energy row and the potential of
the column; the row alone is not a closed budget.

and reports the drift per step of each as a least-squares slope over the
recorded cycles, relative to the initial value. L_z is the relative axial
angular momentum of snapy issue #314, formed from the same integrand that
issue uses, but from the live state rather than from output frames.

The closed system has no external torque: the walls are reflecting, there is no
drag, no sponge and no topography, and constant radial gravity exerts none. So
the exact drift of all three is zero and every measured slope is numerical.

  python3 src/snapy_report/ch16/evidence/aam_budget_run.py \
      [--device cpu|cuda] [--nx 8] [--cycles 200] [--omega 2.0e-5] [--json PATH]

Exits 0 when the run completes; it asserts nothing, it reports.
"""
import argparse
import datetime
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

PIN = "ccf40f9565642c32be69f4b9ef0cfa6daff3f201"   # doc/tech-report/code_pin.yml

DECK = {
    "reference-state": {"Tref": 300.0, "Pref": 1.0e5},
    "species": [{"name": "dry",
                 "composition": {"H": 1.64813, "He": 0.17591},
                 "cv_R": 2.5}],
    "geometry": {
        "type": "gnomonic-equiangle",
        "bounds": {"x1min": 6.0e6, "x1max": 6.4e6,
                   "x2min_pi": -0.25, "x2max_pi": 0.25,
                   "x3min_pi": -0.25, "x3max_pi": 0.25},
        "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": 3},
    },
    "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1,
                   "blocks_per_process": 6, "verbose": False},
    "dynamics": {
        "equation-of-state": {"type": "ideal-gas", "density-floor": 1.0e-20,
                              "pressure-floor": 1.0e-20, "limiter": False},
        "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                        "horizontal": {"type": "weno5", "scale": True, "shock": False}},
        "riemann-solver": {"type": "lmars"},
    },
    "boundary-condition": {"external": {
        "x1-inner": "reflecting", "x1-outer": "reflecting",
        "x2-inner": "custom", "x2-outer": "custom",
        "x3-inner": "custom", "x3-outer": "custom"}},
    "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0,
                    "nlim": 1000000, "tlim": 1.0e12, "ncycle_out": 0},
    "forcing": {"const-gravity": {"grav1": -10.0}},
}

P0, RHO0 = 1.0e5, 0.1          # isothermal reference, scale height p0/(rho0 g)


def load_cuda_sidecars():
    """Make the CUDA kernels of a CUDA-enabled build reachable.

    `setup.py` passes `snap_cuda_release` to the extension, but the linker
    drops the DT_NEEDED entry under --as-needed: the extension references no
    symbol from the sidecar, because its kernels register themselves through
    static initialisers. Without an explicit dlopen, a device tensor reaches
    torch's dispatcher with no registered kernel and the run dies with
    "DispatchStub: missing kernel for cuda". Loading it RTLD_GLOBAL, after
    torch and after the CPU library it needs, makes the registrations visible.
    """
    import ctypes
    import glob
    import site
    loaded = []
    for pattern in ("kintera/lib/libkintera_release.so",
                    "kintera/lib/libkintera_cuda_release.so",
                    "snapy/lib/libsnap_release.so",
                    "snapy/lib/libsnap_cuda_release.so"):
        for base in site.getsitepackages():
            for path in sorted(glob.glob(os.path.join(base, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                loaded.append(os.path.basename(path))
    return loaded


def build(cfg, device):
    from snapy import Mesh, MeshOptions
    nx = cfg["geometry"]["cells"]["nx2"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(nx, nx)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for block in blocks:
        block.to(torch.device(device))
    return mesh, blocks


def interior(block, cfg):
    ng = cfg["geometry"]["cells"]["nghost"]
    nx1 = cfg["geometry"]["cells"]["nx1"]
    k, j, _ = block.part((0, 0, 0), False)[1:]
    return k, j, slice(ng, ng + nx1)


def initial_state(cfg, mesh, blocks, omega):
    """A discretely balanced isothermal shell plus solid-body rotation u_east =
    omega r cos(lat), set through snapy's own spherical-to-contravariant
    transform so that the panel components are the ones the solver carries."""
    import snapy
    from snapy import kIDN, kIPR, kIV1, kIV3
    from snapy.coord import cs_sph_to_contra_

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    ng = cfg["geometry"]["cells"]["nghost"]
    nx1 = cfg["geometry"]["cells"]["nx1"]
    mesh_vars = []
    for face_id, block in enumerate(blocks):
        coord = block.module("coord")
        r = coord.buffer("x1v")[ng:ng + nx1].cpu()
        dr = coord.buffer("dx1f")[ng:ng + nx1].cpu()
        p = P0 * torch.exp(-(r - r[0]) * RHO0 * g / P0)
        col = torch.zeros(kIPR + 1, 1, 1, nx1, dtype=torch.float64)
        col[kIDN, 0, 0] = p * RHO0 / P0
        col[kIPR, 0, 0] = p
        wb, _, _ = snapy.balance_column(col, dr.contiguous(), g)

        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        for c in (kIDN, kIPR):
            w[c][..., ng:ng + nx1] = wb[c, 0, 0].to(w)
            w[c][..., :ng] = w[c][..., ng:ng + 1]
            w[c][..., ng + nx1:] = w[c][..., ng + nx1 - 1:ng + nx1]

        # solid body about the z axis, written as a spherical (r, theta, phi)
        # velocity and rotated into the panel's contravariant components
        alpha = coord.buffer("x2v").unsqueeze(0).unsqueeze(-1).to(w)
        beta = coord.buffer("x3v").unsqueeze(-1).unsqueeze(-1).to(w)
        lat = coord.buffer("latitude").to(w)
        radius = coord.buffer("x1v").unsqueeze(0).unsqueeze(0).to(w)
        vel = torch.zeros_like(w[kIV1:kIV3 + 1])
        vel[2] = omega * radius * torch.cos(lat)        # eastward, index 2 = phi
        a = alpha.expand_as(vel[0]).contiguous()
        b = beta.expand_as(vel[0]).contiguous()
        cs_sph_to_contra_(vel, a, b, face_id)
        w[kIV1:kIV3 + 1] = vel
        mesh_vars.append({"hydro_w": w})
    mesh_vars, _ = mesh.initialize(mesh_vars)
    return mesh_vars


def budgets(cfg, blocks, mesh_vars):
    """M, E and L_z over the interiors of all six panels, from the end-of-step
    conserved state. The momentum rows are covariant, so they are raised with
    the coordinate's own cosine before the spherical rotation."""
    from snapy import kIDN, kIPR, kIV1, kIV3
    from snapy.coord import coord_vec_raise_, cs_contra_to_sph_
    ng = cfg["geometry"]["cells"]["nghost"]
    g = -cfg["forcing"]["const-gravity"]["grav1"]
    r_in = cfg["geometry"]["bounds"]["x1min"]
    mass = energy = lz = 0.0
    for face_id, block in enumerate(blocks):
        coord = block.module("coord")
        sl = interior(block, cfg)
        u = mesh_vars[face_id]["hydro_u"]
        vol = coord.cell_volume()[sl]
        rho = u[kIDN][sl]
        mass += float((rho * vol).sum())
        phi = g * (coord.buffer("x1v").unsqueeze(0).unsqueeze(0)
                   .expand_as(u[kIDN])[sl] - r_in)
        energy += float(((u[kIPR][sl] + rho * phi) * vol).sum())

        vel = (u[kIV1:kIV3 + 1] / u[kIDN]).clone()       # covariant / rho
        coord_vec_raise_(vel, coord.buffer("cosine_cell_kj"))
        alpha = coord.buffer("x2v").unsqueeze(0).unsqueeze(-1).expand_as(vel[0])
        beta = coord.buffer("x3v").unsqueeze(-1).unsqueeze(-1).expand_as(vel[0])
        cs_contra_to_sph_(vel, alpha.contiguous(), beta.contiguous(), face_id)
        u_east = vel[2][sl]
        lat = coord.buffer("latitude").expand_as(vel[0])[sl]
        radius = coord.buffer("x1v").unsqueeze(0).unsqueeze(0).expand_as(vel[0])[sl]
        lz += float((rho * u_east * radius * torch.cos(lat) * vol).sum())
    return mass, energy, lz


def slope_per_step(series):
    """Least-squares slope per cycle of a recorded series."""
    n = len(series)
    if n < 2:
        return 0.0
    x = [float(i) for i in range(n)]
    mx = sum(x) / n
    my = sum(series) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, series))
    den = sum((xi - mx) ** 2 for xi in x)
    return num / den if den else 0.0


def run(args):
    cfg = json.loads(json.dumps(DECK))
    for key in ("nx1", "nx2", "nx3"):
        cfg["geometry"]["cells"][key] = args.nx
    mesh, blocks = build(cfg, args.device)
    mesh_vars = initial_state(cfg, mesh, blocks, args.omega)
    intg = blocks[0].module("intg")

    m, e, l = budgets(cfg, blocks, mesh_vars)
    hist = {"M": [m], "E": [e], "Lz": [l], "dt": []}
    t0 = time.time()
    for cycle in range(1, args.cycles + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        if mesh.check_redo(mesh_vars) != 0:
            print(f"step rejected at cycle {cycle}; stopping")
            break
        m, e, l = budgets(cfg, blocks, mesh_vars)
        hist["M"].append(m)
        hist["E"].append(e)
        hist["Lz"].append(l)
        hist["dt"].append(float(dt))
    wall = time.time() - t0

    out = {
        "pin": PIN,
        "snapy": __import__("snapy").__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "nx": args.nx,
        "panels": 6,
        "cycles_run": len(hist["dt"]),
        "omega": args.omega,
        "wall_s": wall,
        "date": datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for key in ("M", "E", "Lz"):
        s = hist[key]
        out[key] = {
            "initial": s[0],
            "final": s[-1],
            "abs_drift_per_step": slope_per_step(s),
            "rel_drift_per_step": slope_per_step(s) / abs(s[0]) if s[0] else float("nan"),
            "rel_total": (s[-1] - s[0]) / abs(s[0]) if s[0] else float("nan"),
        }
    out["dt_mean"] = sum(hist["dt"]) / len(hist["dt"]) if hist["dt"] else 0.0

    print(f"# snapy {out['snapy']} at pin {PIN[:12]}, torch {out['torch']}, "
          f"python {out['python']}")
    print(f"# device {out['device']}, 6 panels, nx = {args.nx}, "
          f"{out['cycles_run']} cycles, mean dt {out['dt_mean']:.4f} s, "
          f"wall {wall:.1f} s")
    print(f"{'budget':<4} {'initial':>14} {'rel drift / step':>18} "
          f"{'rel drift, total':>18}")
    for key, unit in (("M", "kg"), ("E", "J"), ("Lz", "kg m^2/s")):
        d = out[key]
        print(f"{key:<4} {d['initial']:>14.7e} {d['rel_drift_per_step']:>18.3e} "
              f"{d['rel_total']:>18.3e}   [{unit}]")
    print(f"Lz absolute drift per step {out['Lz']['abs_drift_per_step']:.6e} "
          f"kg m^2 s^-1 per cycle, i.e. "
          f"{out['Lz']['abs_drift_per_step'] / out['dt_mean']:.6e} per second")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": out, "series": hist}, indent=1) + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--omega", type=float, default=2.0e-5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("SKIP: cuda requested but not available")
            return 125
        print("# CUDA sidecars loaded: " + ", ".join(load_cuda_sidecars()))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
