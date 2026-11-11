#!/usr/bin/env python3
"""Every MPS path agrees with the CPU at float32.

Each arm runs the same float32 problem on CPU and MPS. err = max|mps - cpu| /
max|cpu| per variable, worst variable; an arm passes when both are finite and
err <= TOL. f64 = the same for CPU float32 against CPU float64, printed for
scale only. Exits 125 when MPS is not available.

The moist column stops before its last stage: the saturation adjustment there
calls the thermodynamics library, which has no MPS kernel.

  python test_mps_matches_cpu.py
"""
import copy
import math
import os
import sys
import tempfile

import torch
import yaml

# Measured on an M3 Ultra (torch 2.10): the worst arm has err = 6.2e-6.
TOL = 1.0e-4

DRY = {
    "geometry": {
        "type": "cartesian",
        "bounds": {"x1min": 0.0, "x1max": 6.4e3, "x2min": 0.0, "x2max": 12.8e3,
                   "x3min": 0.0, "x3max": 12.8e3},
        "cells": {"nx1": 32, "nx2": 32, "nx3": 1, "nghost": 3},
    },
    "distribute": {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False},
    "dynamics": {
        "equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 29.0e-3,
                              "density-floor": 1.0e-10, "pressure-floor": 1.0e-10,
                              "limiter": False},
        "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                        "horizontal": {"type": "weno5", "scale": False, "shock": False}},
        "riemann-solver": {"type": "lmars"},
    },
    "boundary-condition": {"external": {
        f"x{d}-{s}": "reflecting" for d in (1, 2, 3) for s in ("inner", "outer")}},
    "integration": {"type": "rk3", "cfl": 0.9, "implicit-scheme": 0, "nlim": -1,
                    "tlim": 1.0e6, "ncycle_out": 0},
    "forcing": {"const-gravity": {"grav1": -9.8}},
}


def with_(cfg, **kv):
    """A copy of cfg with each key path a/b/c (written a__b__c) set."""
    cfg = copy.deepcopy(cfg)
    for path, value in kv.items():
        *keys, last = path.split("__")
        node = cfg
        for k in keys:
            node = node[k.replace("_", "-")]
        node[last.replace("_", "-")] = value
    return cfg


def make_block(cfg, device, dtype):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)
    block.to(torch.device(device), dtype)
    return block


def bubble_ic(block, dT):
    """Straka's cold bubble on an adiabatic rest state (dT = 0: rest)."""
    from snapy import kIDN, kIPR

    w = dict(block.named_buffers())["hydro.D"].cpu().double().clone().zero_()
    x1 = block.coord.buffer("x1v").cpu().double().view(1, 1, -1)
    x2 = block.coord.buffer("x2v").cpu().double().view(1, -1, 1)
    Rd, cp, Ts = 8.314462618 / 29.0e-3, 3.5 * 8.314462618 / 29.0e-3, 300.0
    temp = Ts - 9.8 * x1 / cp
    w[kIPR] = 1.0e5 * (temp / Ts) ** (cp / Rd)
    L = (((x2 - 6.4e3) / 4.0e3) ** 2 + ((x1 - 3.0e3) / 2.0e3) ** 2).sqrt()
    temp = temp + torch.where(L <= 1, dT * (torch.cos(L * math.pi) + 1) / 2, 0.0)
    w[kIDN] = w[kIPR] / (Rd * temp)
    return {"hydro_w": w}


def column_ic(block, cfg):
    """The moist tracer column, with one vapor-negative cell for fix_vapor."""
    from snapy import kICY, kIDN, kIPR, kIV1

    w = dict(block.named_buffers())["hydro.D"].cpu().double().clone().zero_()
    x1max = float(cfg["geometry"]["bounds"]["x1max"])
    z = block.coord.buffer("x1v").cpu().double()
    p = 1.0e5 * torch.exp(-z * 10.0 / (3900.0 * 320.0))
    w[kIPR], w[kIDN], w[kICY] = p, p / (3900.0 * 320.0), 0.02
    w[kICY + 1] = 1.0e-3  # cloud, so the sedimentation flux is not empty
    w[kICY, :, :, 10] = -1.0e-4
    w[kIV1] = 5.0 * torch.sin(math.pi * z / x1max)
    w[kIV1 + 1] = 10.0 * torch.cos(math.pi * z / x1max)
    return {"hydro_w": w, "scalar_r": torch.full((1,) + w.shape[1:], 1.0e-3)}


def evolve(cfg, ic, device, dtype, nstep, skip_last_stage=False):
    block = make_block(cfg, device, dtype)
    v, _ = block.initialize({k: x.to(device, dtype) for k, x in ic(block).items()})
    for _ in range(nstep):
        dt = block.max_time_step(v)
        n = len(block.intg.stages) - (1 if skip_last_stage else 0)
        for stage in range(n):
            block.forward(v, dt, stage)
    return {k: v[k].detach().cpu().double() for k in ("hydro_u", "scalar_s") if k in v}


def compare(name, run):
    """err and f32 of the worst variable of run(device, dtype)."""
    try:
        cpu, mps, ref = (run(d, t) for d, t in (("cpu", torch.float32),
                                                 ("mps", torch.float32),
                                                 ("cpu", torch.float64)))
    except Exception as e:  # a missing kernel or a float64 on MPS
        return name, math.inf, math.nan, str(e).splitlines()[0][:160]
    err, f64 = [], []
    for k in cpu:
        for i, (a, b, c) in enumerate(zip(cpu[k], mps[k], ref[k])):
            # one variable at a time; momenta on the scale rho * sqrt(E / rho)
            s = a.abs().max()
            if k == "hydro_u" and 1 <= i <= 3:
                s = (cpu[k][0].abs().max() * cpu[k][4].abs().max()).sqrt()
            s = float(s) or 1.0
            err.append(float((b - a).abs().max()) / s)
            f64.append(float((c - a).abs().max()) / s)
    return name, max(err, key=lambda e: (math.isnan(e), e)), max(f64), ""


def recon_arms():
    from snapy import InterpOptions, Reconstruct, ReconstructOptions

    torch.manual_seed(0)
    w = 1.0 + 0.1 * torch.rand(4, 1, 2, 24, dtype=torch.float64)
    w[1] = 0.0  # an all-zero stencil reconstructs to 0
    w[2] = 1.0e-10 * w[3]  # trace values: no absolute floor on the scale
    for kind in ("cp3", "cp5", "weno3", "weno5"):
        for scale in (False, True):
            def run(device, dtype, kind=kind, scale=scale):
                opts = ReconstructOptions().interp(InterpOptions(kind).scale(scale))
                x = w.to(device, dtype)
                x0 = x.clone()
                recon = Reconstruct(opts)
                recon.to(torch.device(device), dtype)
                out = recon.forward(x, 3, False)
                assert torch.equal(x, x0), "the input was modified"
                return {"out": out.cpu().double().flatten(0, 1)}
            yield compare(f"recon {kind} scale={scale}", run)


def solid_arm():
    from snapy import InternalBoundary, InternalBoundaryOptions

    torch.manual_seed(0)
    solid = (torch.rand(1, 12, 16) > 0.6).int()

    def run(device, dtype):
        out, flips = InternalBoundary(InternalBoundaryOptions()).rectify_solid(
            solid.clone().to(device))
        assert flips > 0
        return {"out": torch.cat([out.cpu().double().flatten(), torch.tensor([flips])])[None]}
    yield compare("rectify_solid", run)


def deck_arms():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "test_tracer_dry_convention.yaml")) as f:
        moist = yaml.safe_load(f)
    moist["integration"]["implicit-scheme"] = 0
    dry = {
        "rest": (DRY, 0.0),
        "bubble": (DRY, -15.0),
        "bubble hllc": (with_(DRY, dynamics__riemann_solver__type="hllc"), -15.0),
        "bubble roe": (with_(DRY, dynamics__riemann_solver__type="roe"), -15.0),
        "bubble cp5": (with_(DRY, dynamics__reconstruct__vertical__type="cp5",
                             dynamics__reconstruct__horizontal__type="cp5"), -15.0),
        "bubble weno3 scaled": (with_(DRY, dynamics__reconstruct__vertical={
            "type": "weno3", "scale": True, "shock": False}), -15.0),
        "bubble vic 1": (with_(DRY, integration__implicit_scheme=1), -15.0),
        "bubble vic 9": (with_(DRY, integration__implicit_scheme=9), -15.0),
    }
    for name, (cfg, dT) in dry.items():
        yield compare(name, lambda d, t, c=cfg, dT=dT: evolve(
            c, lambda b: bubble_ic(b, dT), d, t, nstep=5))
    sed = {"radius": {}, "density": {}, "const-vsed": {"cloud": -2.0}}
    column = {
        "column": moist,
        "column vic 1": with_(moist, integration__implicit_scheme=1),
        "column vic 9": with_(moist, integration__implicit_scheme=9),
        "column hllc": with_(moist, dynamics__riemann_solver__type="hllc"),
        "column limiter": with_(moist, dynamics__equation_of_state__limiter=True),
        "column sedimentation": with_(moist, sedimentation=sed),
    }
    for name, cfg in column.items():
        yield compare(name, lambda d, t, c=cfg: evolve(
            c, lambda b: column_ic(b, c), d, t, nstep=1, skip_last_stage=True))


def stretched_refused():
    """A stretched x1 has no MPS reconstruction; it must say so."""
    cfg = with_(DRY, geometry__cells__x1rat=1.02)
    try:
        evolve(cfg, lambda b: bubble_ic(b, 0.0), "mps", torch.float32, nstep=1)
    except Exception as e:
        return "not supported on MPS" in str(e), str(e).splitlines()[0][:160]
    return False, "ran"


def main():
    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 125
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    failures = []
    for name, err, f64, msg in [*recon_arms(), *solid_arm(), *deck_arms()]:
        ok = err <= TOL
        print(f"{'ok  ' if ok else 'FAIL'} {name:28s} err={err:.2e} f64={f64:.2e} {msg}")
        if not ok:
            failures.append(name)
    ok, msg = stretched_refused()
    print(f"{'ok  ' if ok else 'FAIL'} stretched x1 refused: {msg}")
    if not ok:
        failures.append("stretched x1")

    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    print("### MPS matches CPU on every arm. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
