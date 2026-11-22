#!/usr/bin/env python3
"""The netcdf lon/lat of a gnomonic block are its coordinate's (audit 1 row O1).

A regional gnomonic-equiangle deck on the "cubed" layout with nb1 = 2: two blocks stacked
in x1, one process. The coordinate registers each block's lon/lat on face 0, since only a
"cubed-sphere" rank has a panel, and lon/lat do not depend on x1. So each block's file
(out0, combine: false) and the combined file (out1) must hold the coordinate's lon/lat in
the file's precision. The output used to take loc_of's third entry as the face, which on
this layout is the x1 block index, so block 1 wrote panel 1's lon/lat.
"""
import os
import sys
import tempfile

import numpy as np
import yaml

SKIP_CODE = 125
try:
  import netCDF4
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_netcdf_cubed_lonlat: netCDF4 import failed: {exc}")
  sys.exit(SKIP_CODE)

DECK = {
    "reference-state": {"Tref": 300., "Pref": 1.e5},
    "species": [{"name": "dry", "composition": {"N": 2}, "cv_R": 2.5}],
    "geometry": {
        "type": "gnomonic-equiangle",
        "bounds": {"x1min": 6.4e6, "x1max": 6.41e6, "x2min_pi": -0.1, "x2max_pi": 0.1,
                   "x3min_pi": -0.05, "x3max_pi": 0.15},
        "cells": {"nx1": 8, "nx2": 6, "nx3": 6, "nghost": 3}},
    "distribute": {"layout": "cubed", "nb1": 2, "nb2": 1, "nb3": 1,
                   "blocks_per_process": 2, "backend": "gloo"},
    "dynamics": {
        "equation-of-state": {"type": "ideal-gas"},
        "reconstruct": {"vertical": {"type": "weno5", "shock": False},
                        "horizontal": {"type": "weno5", "shock": False}},
        "riemann-solver": {"type": "lmars"}},
    "boundary-condition": {"external": {
        "x1-inner": "reflecting", "x1-outer": "reflecting",
        "x2-inner": "reflecting", "x2-outer": "reflecting",
        "x3-inner": "reflecting", "x3-outer": "reflecting"}},
    "integration": {"type": "rk3", "cfl": 0.4, "nlim": 1, "tlim": 1.e12, "ncycle_out": 0},
    "output_dir": ".",
    # out0: one file per block; out1: the combined file
    "outputs": [{"type": "netcdf", "variables": ["prim"], "dt": 1.e30, "combine": False},
                {"type": "netcdf", "variables": ["prim"], "dt": 1.e30}],
}


def check(fn, want):
  """the number of lon/lat variables of fn that differ from want"""
  bad = 0
  with netCDF4.Dataset(fn) as nc:
    for k in ("lon", "lat"):
      got = np.asarray(nc.variables[k][0])
      ref = want[k].astype(got.dtype)
      diff = np.abs(got.astype(np.float64) - ref.astype(np.float64)).max()
      print(f"{fn} {k} {got.shape} {got.dtype}: max |file - coordinate| {diff:.3e}")
      bad += not np.array_equal(got, ref)
  return bad


def main() -> int:
  import snapy

  here = os.getcwd()
  with tempfile.TemporaryDirectory(dir=here) as d:
    os.chdir(d)
    with open("deck.yaml", "w") as f:
      yaml.safe_dump(DECK, f)
    mesh = snapy.Mesh(snapy.MeshOptions.from_yaml("deck.yaml"))
    blocks = list(mesh.blocks)
    assert len(blocks) == 2, f"expected 2 blocks, got {len(blocks)}"
    state = []
    for b in blocks:
      w = dict(b.named_buffers())["hydro.D"].detach().clone().zero_()
      w[0], w[4] = 1., 1.e5
      state.append({"hydro_w": w})
    state, _ = mesh.initialize(state)
    mesh.make_outputs(state, 0., False)
    if not any(f.endswith(".nc") for f in os.listdir(".")):
      print("Skipping test_netcdf_cubed_lonlat: this build writes no netcdf")
      return SKIP_CODE

    bad, coords = 0, []
    for b in blocks:
      buf = dict(b.named_buffers())
      ng = DECK["geometry"]["cells"]["nghost"]  # the files hold the interior
      coords.append({k: buf[f"coord.{n}"].squeeze(-1)[ng:-ng, ng:-ng].cpu().numpy()
                     for k, n in (("lon", "longitude"), ("lat", "latitude"))})
      rank = b.get_layout().options.rank()
      bad += check(f"deck.block{rank}.out0.00000.nc", coords[-1])
    for k in ("lon", "lat"):
      assert np.array_equal(coords[0][k], coords[1][k]), f"the blocks' {k} buffers differ"
    bad += check("deck.out1.00000.nc", coords[0])
    os.chdir(here)
  print("FAIL" if bad else "PASS", f"({bad} lon/lat variable(s) differ from the coordinate's)")
  return 1 if bad else 0


if __name__ == "__main__":
  raise SystemExit(main())
