#!/usr/bin/env python3
"""Total energy plus potential energy in a closed box, per gravity-work form (#283).

A moving isentropic column (gamma 1.4, Rd 287, g 9.8, Ts 300 K, density ratio e^3, put
into the scheme's discrete balance with snapy.balance_column, w = w0 sin(pi z / Lz),
w0/c = 1e-2) runs NSTEP explicit rk3 steps (weno5, lmars) between reflecting x1 walls.
E+PE = sum (E + rho g z) dV must stay constant:
  gravity-work: cell + gravity-work-fixer (its default)  |d(E+PE)| / |E+PE| <= TOL, explicit and
                                                         implicit (vic-partial)
  gravity-work: cell, gravity-work-fixer: false            drift > 100 TOL (planted control: the
                                                         defect the fixer removes is real)
The fixer's energy (buffers hydro.gwfix_total + gwfix_pending, printed as fixgrav= in the cycle diagnostics) is
reported. The fixer accepts any impenetrable x1 boundary, including an unnamed one installed from
Python, and runs in float32 (its wall-mass bound scales with the dtype's eps). It refuses a step that
moves mass through an x1 boundary face (outflow, float64 and float32) and a periodic x1 boundary
(also when the layout wraps x1 and a Python bfuncs round trip has cleared the boundary names).
gravity-work: face-wallc is no longer a form and is refused at construction. The new keys
are reachable from Python, where grav2 != 0 with the fixer is refused at
construction. A non-finite E+PE fails an arm; a NaN in a wall cell goes to the redo check.

  python test_gravity_work_fixer.py [--device cpu] [--nstep 200]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

GAMMA, RD, GRAV, TS, PS = 1.4, 287.0, 9.8, 300.0, 1.0e5
CP = GAMMA * RD / (GAMMA - 1.0)
LZ = TS * (1.0 - math.exp(-3.0 * (GAMMA - 1.0))) * CP / GRAV  # rho_top / rho_bot = e^-3
NZ, MACH = 32, 1.0e-2
TOL = 1.0e-12


def config(gravity, x1bc="reflecting", scheme=0):
    rgas = 8.31446261815324
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": LZ, "x2min": 0.0, "x2max": 1.0,
                                "x3min": 0.0, "x3max": 1.0},
                     "cells": {"nx1": NZ, "nx2": 1, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": x1bc, "x1-outer": x1bc,
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9},
        # cell unless an arm says otherwise (the default is face)
        "forcing": {"const-gravity": dict({"grav1": -GRAV, "gravity-work": "cell"}, **gravity)},
    }


def python_reflecting(face):
    """A reflecting x1 wall written in Python, installed without a name."""
    import snapy

    def bc(var, dim, op):
        if var.size(dim) == 1:
            return
        ng, n = op.nghost(), var.size(dim)
        lo = 0 if face == 0 else n - ng
        src = ng if face == 0 else n - 2 * ng
        var.narrow(dim, lo, ng).copy_(var.narrow(dim, src, ng).flip(dim))
        if op.type() in (snapy.kConserved, snapy.kPrimitive):
            var[4 - dim].narrow(dim - 1, lo, ng).mul_(-1)

    return bc


def run(gravity, nstep, device, x1bc="reflecting", scheme=0, pybc=False, dtype=torch.float64, nan=False,
        cubed=False, first_stage=False):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        cfg = config(gravity, x1bc, scheme)
        if cubed:
            cfg["distribute"] = {"layout": "cubed"}
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        opts = MeshBlockOptions.from_yaml(tmp)
    finally:
        os.unlink(tmp)
    if pybc:
        opts.set_bfunc(0, 0, -1, python_reflecting(0))
        opts.set_bfunc(0, 0, 1, python_reflecting(1))
    if cubed:
        opts.bfuncs(opts.bfuncs())  # a Python round trip clears the boundary names
    block = MeshBlock(opts)
    block.to(torch.device(device), dtype)

    ng = 3
    dz = LZ / NZ
    z = (torch.arange(NZ, dtype=torch.float64) + 0.5) * dz
    T = TS - GRAV * z / CP
    p = PS * (T / TS) ** (CP / RD)
    col = torch.zeros(kIPR + 1, 1, 1, NZ, dtype=torch.float64)
    col[kIDN, 0, 0] = p / (RD * T)
    col[kIPR, 0, 0] = p
    wb, _, _ = snapy.balance_column(col, torch.full((NZ,), dz, dtype=torch.float64), GRAV)
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    sl = (Ellipsis, slice(ng, ng + NZ))
    for c in (kIDN, kIPR):
        w[c][sl] = wb[c, 0, 0].to(w)
        w[c][..., :ng] = w[c][..., ng:ng + 1]
        w[c][..., ng + NZ:] = w[c][..., ng + NZ - 1:ng + NZ]
    w[kIV1][sl] = MACH * math.sqrt(GAMMA * RD * TS) * torch.sin(math.pi * z / LZ).to(w)
    block_vars, _ = block.initialize({"hydro_w": w})
    zz = z.to(w.device)

    def epe():
        u = block_vars["hydro_u"][sl].double()
        return ((u[kIPR] + u[kIDN] * GRAV * zz).sum() * dz).item()

    e0 = epe()
    nstage = 1 if first_stage else len(block.module("intg").stages)
    drift = 0.0
    for n in range(nstep):
        if nan and n == nstep - 1:  # a NaN in the bottom wall cell before the last step
            block_vars["hydro_u"][kIDN][..., ng] = float("nan")
        else:
            dt = block.max_time_step(block_vars)
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        d = abs(epe() - e0) / abs(e0)
        drift = max(drift, d) if math.isfinite(d) else math.inf
    buf = dict(block.named_buffers())
    fix = (buf["hydro.gwfix_total"] + buf["hydro.gwfix_pending"]).item()
    if nan:
        return block.check_redo(block_vars)
    return drift, fix, e0, block_vars["hydro_u"][sl].double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nstep", type=int, default=200)
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0
    torch.set_default_dtype(torch.float64)

    failures = []
    arms = (("cell + fixer, implicit", {}, 1, "conserves"),
            ("cell, fixer off", {"gravity-work": "cell", "gravity-work-fixer": False}, 0, "drifts"))
    arms += (("cell + fixer, Python wall", {}, 0, "conserves"),
             ("cell + fixer, float32", {}, 0, "float32"))
    for name, gravity, scheme, want in arms:
        f32 = want == "float32"
        drift, fix, e0, _ = run(gravity, args.nstep, args.device, scheme=scheme,
                                pybc=name.endswith("Python wall"),
                                dtype=torch.float32 if f32 else torch.float64)
        print("%-24s max |d(E+PE)|/|E+PE| over %d steps = %.3e   fixgrav = %.3e J/m2 (%.1e of E+PE)"
              % (name, args.nstep, drift, fix, abs(fix / e0)))
        if not math.isfinite(drift):
            failures.append("%s: E+PE is not finite" % name)
        tol32 = args.nstep * torch.finfo(torch.float32).eps  # one float32 eps per step
        if f32 and not drift <= tol32:
            failures.append("%s: E+PE drifts %.3e > %.1e" % (name, drift, tol32))
        if want == "conserves" and not drift <= TOL:
            failures.append("%s: E+PE drifts %.3e > %.0e" % (name, drift, TOL))
        if want == "drifts" and not drift > 100 * TOL:
            failures.append("%s: drift %.3e <= %.0e, the fixer would not be load-bearing" % (name, drift, 100 * TOL))

    from snapy import kIPR

    refusals = (("outflow x1, float64", "outflow", torch.float64, "crossed an x1 boundary face", False),
                ("outflow x1, float32", "outflow", torch.float32, "crossed an x1 boundary face", False),
                ("periodic x1", "periodic", torch.float64, "non-periodic x1", False),
                ("periodic x1, cubed layout, Python bfuncs round trip", "periodic", torch.float64,
                 "non-periodic x1", True))
    for name, x1bc, dtype, msg, cubed in refusals:
        try:
            run({}, 2, args.device, x1bc=x1bc, dtype=dtype, cubed=cubed)
            failures.append("the fixer accepted %s" % name)
        except RuntimeError as e:
            ok = msg in str(e)
            print("%s + fixer: refused (%s)" % (name, "expected message" if ok else "unexpected message"))
            if not ok:
                failures.append("%s + fixer: unexpected error: %s" % (name, str(e).splitlines()[0]))

    # gravity-work: face-wallc is no longer a form: refused at construction
    try:
        run({"gravity-work": "face-wallc"}, 1, args.device)
        failures.append("gravity-work: face-wallc was accepted")
    except RuntimeError as e:
        ok = "must be cell or face" in str(e)
        print("gravity-work: face-wallc: refused (%s)" % ("expected message" if ok else "unexpected message"))
        if not ok:
            failures.append("face-wallc: unexpected error: %s" % str(e).splitlines()[0])

    # a NaN in a wall cell is left to the redo check (as with the fixer off), not refused
    try:
        redo = run({}, 2, args.device, nan=True)
        print("NaN in a wall cell + fixer: check_redo -> %d" % redo)
        if redo == 0:
            failures.append("NaN in a wall cell: the redo check did not flag the step")
    except RuntimeError as e:
        failures.append("NaN in a wall cell + fixer: %s" % str(e).splitlines()[0])

    # the keys from Python, and the grav2 refusal on that path
    from snapy import MeshBlock, MeshBlockOptions

    def options(cfg):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
            yaml.safe_dump(cfg, f)
            tmp = f.name
        try:
            return MeshBlockOptions.from_yaml(tmp)
        finally:
            os.unlink(tmp)

    nokey = config({})
    del nokey["forcing"]["const-gravity"]["gravity-work"]
    for want, opts in (("face", options(nokey)), ("cell", options(config({})))):
        g = opts.hydro().grav()
        print("python options: gravity_work=%s gravity_work_fixer=%s" % (g.gravity_work(), g.gravity_work_fixer()))
        if g.gravity_work() != want or g.gravity_work_fixer() is not (want == "cell"):
            failures.append("python options do not read the defaults (%s)" % want)
    g.grav2(1.0)
    try:
        MeshBlock(opts)
        failures.append("grav2 != 0 with the fixer was accepted from Python")
    except RuntimeError as e:
        ok = "grav2 = grav3 = 0" in str(e)
        print("python grav2 + fixer: refused (%s)" % ("expected message" if ok else "unexpected message"))
        if not ok:
            failures.append("python grav2 + fixer: unexpected error: %s" % str(e).splitlines()[0])

    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
