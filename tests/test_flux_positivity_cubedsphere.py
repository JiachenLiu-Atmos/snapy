#!/usr/bin/env python3
"""Tracer flux-positivity limiter across a cubed-sphere PANEL SEAM.

Six panels in one process (Mesh API), solid-body zonal wind, a tracer hat whose downwind edge
sits on the +X/+Y seam and its complement 1 - hat. The limiter's donor factor theta is 0 along
the seam inside the hat's latitude band and 1 outside, so the seam ghost fill is exercised with
a jump: a ghost theta that is not the neighbour's edge-cell value leaks mass there (measured
1e-3 per 300 cycles with an interpolated fill). Both arms must conserve every tracer to
round-off; the limited arm must fire (hits > 0) and keep the hat in [0, 1].

  python test_flux_positivity_cubedsphere.py [--device cuda] [--yaml PATH]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

DRIFT_TOL = 1e-13


def run_arm(yaml_file: str, limiter: bool, device: str, flux_covariance=None):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIV1, kIV2, kIV3, kIPR

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"]["equation-of-state"]["limiter"] = limiter
    if flux_covariance is not None:
        cfg["dynamics"]["flux-covariance"] = flux_covariance == "1"
    if not limiter:
        cfg["scalar"].pop("upper-bound", None)
    nx = cfg["geometry"]["cells"]["nx2"]
    nlim = cfg["integration"]["nlim"]

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
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

    lon0, half, band = math.radians(15.0), math.radians(30.0), math.radians(30.0)
    mesh_vars = []
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
        face_id = int(block.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
        bufs = dict(block.named_buffers())
        w = bufs["hydro.D"].clone().zero_()
        r = bufs["scalar.D"].clone().zero_()
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=alpha.device)
        vel[2] = 10.0 * torch.cos(lat)  # (v_r, v_theta, v_phi) -> contravariant, in place
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
        w[kIDN] = 1.0
        w[kIPR] = 1.0e5
        w[kIV1] = vel[0].unsqueeze(-1)
        w[kIV2] = vel[1].unsqueeze(-1)
        w[kIV3] = vel[2].unsqueeze(-1)
        dlon = torch.remainder(lon - lon0 + math.pi, 2 * math.pi) - math.pi
        hat = ((dlon.abs() < half) & (lat.abs() < band)).to(torch.float64).unsqueeze(-1)
        r[0] = hat
        r[1] = 1.0 - hat
        mesh_vars.append({"hydro_w": w, "scalar_r": r})
    assert str(mesh_vars[0]["hydro_w"].device).startswith(device)
    mesh_vars, t = mesh.initialize(mesh_vars)

    def totals():
        tot = torch.zeros(2, dtype=torch.float64)
        rmax, rmin = -1e300, 1e300
        for ib, block in enumerate(blocks):
            sl = block.part((0, 0, 0), False)[1:]
            vol = block.module("coord").cell_volume()[sl]
            s = mesh_vars[ib]["scalar_s"][(slice(None),) + tuple(sl)]
            tot += (s * vol).sum(dim=(1, 2, 3)).cpu()
            rr = s[0] / mesh_vars[ib]["hydro_w"][kIDN][sl]
            rmax, rmin = max(rmax, float(rr.max())), min(rmin, float(rr.min()))
        return tot, rmax, rmin

    tot0, _, _ = totals()
    intg = blocks[0].module("intg")
    for cycle in range(1, nlim + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
    tot1, rmax, rmin = totals()
    hits = sum(int(dict(b.named_buffers())["scalar.positivity_hits"].item()) for b in blocks)
    drift = ((tot1 - tot0) / tot0).abs().max().item()
    return {"drift": drift, "max": rmax, "min": rmin, "hits": hits}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_flux_positivity_cubedsphere.yaml"),
    )
    ap.add_argument("--flux-covariance", choices=("0", "1"))
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125

    base = run_arm(args.yaml, False, args.device, args.flux_covariance)
    lim = run_arm(args.yaml, True, args.device, args.flux_covariance)
    for name, arm in (("base", base), ("limited", lim)):
        print("%-8s drift=%.3e max=%.12f min=%+.3e hits=%d"
              % (name, arm["drift"], arm["max"], arm["min"], arm["hits"]))

    failures = []
    if not base["drift"] < DRIFT_TOL:
        failures.append("base arm does not conserve across the seam: %g" % base["drift"])
    if not lim["drift"] < DRIFT_TOL:
        failures.append(
            "limited arm leaks across the seam: %g -- theta ghost not the donor's?" % lim["drift"])
    if not lim["hits"] > 0:
        failures.append("limiter never fired: the test no longer bites")
    if not (lim["min"] >= -1e-15 and lim["max"] <= 1.0 + 1e-5):
        failures.append("limited arm left [0, 1]: min=%g max=%g" % (lim["min"], lim["max"]))
    for msg in failures:
        print("FAIL:", msg)
    if failures:
        sys.exit(1)
    print("### flux positivity cubed-sphere seam test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
