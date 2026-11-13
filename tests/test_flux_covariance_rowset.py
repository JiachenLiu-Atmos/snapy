#!/usr/bin/env python3
"""The flux covariance's nonzero-row set (key dynamics/flux-covariance), Cartesian vs curved, dry vs moist.

_flux_covariance returns only its nonzero rows on a Cartesian grid (IPR and, with species, the tracer
rows and IDN, on the x1-interior cells) and the full flux on a curved one; HydroImpl::forward adds them
to the face flux. One stage from a sheared, stratified state with varying vapor and cloud, switch on vs
off: the rows of the x2 face flux that change, and where, must be exactly
  cartesian dry:    {IPR}
  cartesian moist:  {IDN, IPR, vapor, cloud}
  curved (spherical-polar), dry and moist: every row (the centroid parts reach the momentum rows)
and, in Cartesian, the first and last x1 cells must not change. Each arm runs in its own process (as in
test_flux_covariance_rows: kintera's species table is per process).

  python test_flux_covariance_rowset.py [--device cpu]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_flux_covariance_rows as rows  # noqa: E402  (deck, builders)


def stage_flux(geom, moist, device):
    """x2 face flux after one stage forward from a sheared, stratified state"""
    from snapy import kIPR
    cfg = rows.card(geom, moist=moist)
    _, blocks = rows.blocks_of(cfg, device)
    b = blocks[0]
    nx1 = cfg["geometry"]["cells"]["nx1"]
    z = b.buffer("coord.x1v").cpu()
    x = b.buffer("coord.x2v").cpu()
    zz = (z - z[0]) / (z[-1] - z[0])
    zz, xx = zz[None, None, :], (x - x[0])[None, :, None] / (x[-1] - x[0])

    def shear(block, w):
        w[2] = (20.0 * torch.tanh((zz - 0.5) / 0.15) + 3.0 * torch.sin(2 * math.pi * xx)
                ).expand_as(w[2]).to(w)
        w[1] = (0.5 * torch.sin(2 * math.pi * xx) * torch.sin(math.pi * zz)).expand_as(w[1]).to(w)
        w[3] = (5.0 * torch.cos(2 * math.pi * xx) * zz).expand_as(w[3]).to(w)
        if moist:  # vapor and cloud vary in x1 and x2
            w[kIPR + 1] = (0.02 * torch.exp(-2.0 * zz) * (1.0 + 0.1 * torch.cos(2 * math.pi * xx))
                           ).expand_as(w[kIPR + 1]).to(w)
            w[kIPR + 2] = (1.0e-3 * zz * (1.0 + 0.2 * torch.sin(2 * math.pi * xx))
                           ).expand_as(w[kIPR + 2]).to(w)
    w = rows.balanced_w(b, nx1, wind=shear)
    v, _ = b.initialize({"hydro_w": w.to(device)})
    b.inc_cycle()
    dt = float(b.max_time_step(v))
    b.forward(v, dt, 0)
    return b.buffer("hydro.F2").cpu().clone()


def run(geom, moist, switch, device, tmpdir):
    out = os.path.join(tmpdir, f"{geom}_{int(moist)}_{switch}.pt")
    subprocess.run([sys.executable, __file__, "--child", f"{geom}:{int(moist)}", "--out", out,
                    "--device", device, "--flux-covariance", switch], check=True)
    return torch.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child")
    ap.add_argument("--out")
    ap.add_argument("--flux-covariance", choices=("0", "1"))
    a = ap.parse_args()
    if a.child:
        rows.COV = a.flux_covariance == "1"
        geom, moist = a.child.split(":")
        torch.save(stage_flux(geom, moist == "1", a.device), a.out)
        return 0
    from snapy import kIDN, kIPR
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        for geom in ("cartesian", "spherical-polar"):
            for moist in (False, True):
                on = run(geom, moist, "1", a.device, tmp)
                off = run(geom, moist, "0", a.device, tmp)
                d = (on - off).abs()
                changed = {r for r in range(d.size(0)) if d[r].max() > 0.}
                if geom == "cartesian":
                    want = {kIPR} | ({kIDN, kIPR + 1, kIPR + 2} if moist else set())
                else:
                    want = set(range(d.size(0)))
                tag = f"{geom:15s} {'moist' if moist else 'dry':5s}"
                print(f"{tag} rows {d.size(0)}: changed {sorted(changed)}, expected {sorted(want)}",
                      flush=True)
                if changed != want:
                    failures.append(f"{tag}: changed rows {sorted(changed)} != {sorted(want)}")
                if geom == "cartesian":
                    edge = float(torch.stack([d[..., 0].max(), d[..., -1].max()]).max())
                    if edge != 0.:
                        failures.append(f"{tag}: the first/last x1 cells changed by {edge:.3e}")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("### flux covariance row set test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
