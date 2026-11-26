"""The ported cubed-sphere geometry of `invariants_budget_check.py` against a
build at the code pin.

This is a RUN record, not a check: it needs a built snapy in the active Python.
It compares, cell by cell on all six panels, every quantity the budget check
ports from `GnomonicEquiangleImpl` and `cubed_sphere_utils`, so that the
chapter may read the check's numbers as the solver's and not only as algebra.

  python3 src/snapy_report/ch16/evidence/port_validation.py [--nx 8] [--nr 8]

Prints one line per quantity with the largest relative difference, and exits
with the number of quantities above the tolerance.
"""
import argparse
import math
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from snapy_report.ch16.checks.invariants_budget_check import Panel, RI, RO  # noqa: E402

#  The angular widths dx2f_ang_kj and dx3f_ang_kj are arccos of a quantity that
#  approaches 1 as the grid refines: 1 - u is about (d/2)^2 for an angular cell
#  width d, so arccos amplifies a last-bit difference in u by 1/sqrt(1 - u^2)
#  ~ 1/d, and the RELATIVE error of the width itself goes as eps_m / d^2. Two
#  conforming implementations of the same formula therefore differ by that much,
#  and x_ov_rD, which is a difference of two such widths, by more. The tolerance
#  below is that bound with a factor of 64; a disagreement above it is a port
#  error, one below it is the conditioning of the formula.
#  x_ov_rD and y_ov_rC are differences OF two such widths, so they carry it
#  twice; they get the same bound with a factor of 4.
DIFFERENCE_OF_WIDTHS = ("x_ov_rD_kji", "y_ov_rC_kji")


def tolerance(n_ang, name=""):
    d = math.pi / (2.0 * n_ang)
    factor = 256.0 if name in DIFFERENCE_OF_WIDTHS else 64.0
    return max(1.0e-13, factor * np.finfo(np.float64).eps / (d * d))

DECK = {
    "reference-state": {"Tref": 300.0, "Pref": 1.0e5},
    "species": [{"name": "dry", "composition": {"H": 1.64813, "He": 0.17591},
                 "cv_R": 2.5}],
    "geometry": {"type": "gnomonic-equiangle",
                 "bounds": {"x1min": RI, "x1max": RO,
                            "x2min_pi": -0.25, "x2max_pi": 0.25,
                            "x3min_pi": -0.25, "x3max_pi": 0.25},
                 "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": 3}},
    "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1,
                   "blocks_per_process": 6, "verbose": False},
    "dynamics": {"equation-of-state": {"type": "ideal-gas", "limiter": False},
                 "reconstruct": {"vertical": {"type": "weno5", "scale": True,
                                              "shock": False},
                                 "horizontal": {"type": "weno5", "scale": True,
                                                "shock": False}},
                 "riemann-solver": {"type": "lmars"}},
    "boundary-condition": {"external": {
        "x1-inner": "reflecting", "x1-outer": "reflecting",
        "x2-inner": "custom", "x2-outer": "custom",
        "x3-inner": "custom", "x3-outer": "custom"}},
    "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": 0,
                    "nlim": 1, "tlim": 1.0e12, "ncycle_out": 0},
    "forcing": {"const-gravity": {"grav1": -10.0}},
}


def rel(a, b):
    """Largest difference, scaled by the largest magnitude of the quantity
    itself. A pointwise ratio would read a cancelling difference (x_ov_rD at
    the panel centre is a difference of two nearly equal face quantities) as a
    large error when the two sides differ only in the order of the same
    floating-point operations."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()))
    return float(np.abs(a - b).max() / (scale if scale else 1.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--nr", type=int, default=8)
    args = ap.parse_args(argv)

    from snapy import Mesh, MeshOptions
    from snapy.coord import cs_contra_to_cart_

    cfg = yaml.safe_load(yaml.safe_dump(DECK))
    cfg["geometry"]["cells"].update(nx1=args.nr, nx2=args.nx, nx3=args.nx)
    ng = cfg["geometry"]["cells"]["nghost"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(args.nx, args.nx)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)

    i = slice(ng, ng + args.nr)
    i1 = slice(ng, ng + args.nr + 1)
    k = j = slice(ng, ng + args.nx)
    kj1 = slice(ng, ng + args.nx + 1)
    worst = {}
    rng = np.random.default_rng(99)
    for face_id, block in enumerate(mesh.blocks):
        coord = block.module("coord")
        port = Panel(args.nx, args.nr, face_id)
        buf = {n: t for n, t in coord.named_buffers()}

        def cmp(name, ported, native):
            worst[name] = max(worst.get(name, 0.0), rel(ported, native))

        cmp("solid_angle_kj", port.solid_angle,
            buf["solid_angle_kj"][k, j].numpy())
        cmp("cell_volume", port.volume, coord.cell_volume()[k, j, i].numpy())
        cmp("face_area1", port.area1, coord.face_area1()[k, j, i1].numpy())
        cmp("face_area2", port.area2, coord.face_area2()[k, kj1, i].numpy())
        cmp("face_area3", port.area3, coord.face_area3()[kj1, j, i].numpy())
        cmp("cosine_cell_kj", port.cosine_cell, buf["cosine_cell_kj"][k, j].numpy())
        cmp("sine_cell_kj", port.sine_cell, buf["sine_cell_kj"][k, j].numpy())
        cmp("sine_face2_kj", port.sine_face2, buf["sine_face2_kj"][k, kj1].numpy())
        cmp("sine_face3_kj", port.sine_face3, buf["sine_face3_kj"][kj1, j].numpy())
        cmp("dx2f_ang_kj", port.dx2f_ang, buf["dx2f_ang_kj"][k, j].numpy())
        cmp("dx3f_ang_kj", port.dx3f_ang, buf["dx3f_ang_kj"][k, j].numpy())
        cmp("x_ov_rD_kji", port.x_ov_rD, buf["x_ov_rD_kji"][k, j, i].numpy())
        cmp("y_ov_rC_kji", port.y_ov_rC, buf["y_ov_rC_kji"][k, j, i].numpy())

        # the velocity transform, on a random contravariant field
        shape = port.volume.shape
        vel = rng.normal(0.0, 10.0, (3,) + shape)
        native = torch.from_numpy(vel.copy())
        alpha = torch.from_numpy(np.broadcast_to(
            np.tan(port.x2v)[None, :, None] * 0.0
            + port.x2v[None, :, None], shape).copy())
        beta = torch.from_numpy(np.broadcast_to(
            port.x3v[:, None, None] * np.ones(shape), shape).copy())
        cs_contra_to_cart_(native, alpha.contiguous(), beta.contiguous(), face_id)
        cmp("cs_contra_to_cart_", port.contra_to_cart(vel), native.numpy())

        # the cell centres, against the panel's own longitude and latitude
        lon = buf["longitude"][k, j].numpy()
        lat = buf["latitude"][k, j].numpy()
        unit = port.position / port.radius[None]
        cmp("cell centre (lat)", np.arcsin(unit[2]), np.broadcast_to(lat, shape))
        # longitude is compared as an angle: the two differ by 2 pi across the
        # branch cut of atan2, which is not a difference in position
        dlon = np.arctan2(unit[1], unit[0]) - np.broadcast_to(lon, shape)
        wrapped = np.arctan2(np.sin(dlon), np.cos(dlon))
        worst["cell centre (lon)"] = max(worst.get("cell centre (lon)", 0.0),
                                         float(np.abs(wrapped).max()))

    import snapy
    print(f"# snapy {snapy.__version__}, torch {torch.__version__}, "
          f"{args.nx} by {args.nx} by {args.nr} cells on each of 6 panels")
    print(f"# angular cell width d = {math.pi / (2.0 * args.nx):.4f} rad; "
          f"arccos conditioning bound 64 eps_m / d^2 = "
          f"{tolerance(args.nx):.2e} (x4 for a difference of two widths)")
    bad = 0
    exact = 0
    for name, value in worst.items():
        tol = tolerance(args.nx, name)
        ok = value < tol
        bad += 0 if ok else 1
        exact += 1 if value == 0.0 else 0
        print(f"{'OK  ' if ok else 'FAIL'} {name:<22} {value:.3e} "
              f"(bound {tol:.2e})")
    print(f"{len(worst) - bad}/{len(worst)} ported quantities agree with the "
          f"build at the pin within the conditioning bound; {exact} agree "
          f"bit for bit")
    return bad


if __name__ == "__main__":
    sys.exit(main())
