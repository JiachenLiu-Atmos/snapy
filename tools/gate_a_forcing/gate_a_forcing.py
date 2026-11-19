"""Lane A forcing-deck bitwise gate (A-ph1, A.rst G1): 18 arms.

The column, initial state and dt are the #348 harness's (UCzhangxi/snapy
94e4601e tools/harness348/h348.py: config(), initial_w(), the constants),
copied here unchanged. Every lane-A forcing is switched on in some arm. Each
arm runs NSTEP steps and saves hydro_u after every step. See README.md.

usage:
  gate_a_forcing.py --out DIR [--device cpu|cuda] [--moist-deck YAML]
  gate_a_forcing.py --compare DIR_A DIR_B   (exit 0: every arm ran on both
                                             sides and is bitwise identical)
"""
import argparse
import copy
import hashlib
import math
import os
import sys
import tempfile

import torch
import yaml

# ---- from tools/harness348/h348.py at 94e4601e, unchanged ----
GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
RADIUS = 7.e7
CS = math.sqrt(GAMMA * RD * T0)
DT = 20.0 * DZ / CS  # vertical acoustic Courant ~20: the VIC carries the step
NGHOST = 3


def config(scheme, work, geometry):
    rgas = 8.31446261815324
    cfg = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0,
                                "x2max": 31415926.53589793, "x3min": 0.0,
                                "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1,
                               "nghost": NGHOST}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA,
                                  "weight": rgas / RD,
                                  "density-floor": 3.4682e-18,
                                  "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {
                "vertical": {"type": "weno5", "scale": True, "shock": False},
                "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": scheme,
                        "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": work}},
    }
    if work == "face":
        cfg["forcing"]["const-gravity"]["gravity-work-fixer"] = False
    # cell: the fixer key is omitted, so the shipped default (fixer on) runs
    if geometry == "stretched":
        cfg["geometry"]["cells"]["x1rat"] = 1.05
    elif geometry == "spherical-polar":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"]["nx2"] = 1
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min": math.pi / 2 - 0.2, "x2max": math.pi / 2 + 0.2,
            "x3min": 0.0, "x3max": 0.05}
    elif geometry == "gnomonic-equiangle":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"].update(nx2=4, nx3=4)
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min_pi": -0.25, "x2max_pi": 0.25,
            "x3min_pi": -0.25, "x3max_pi": 0.25}
        for side in ("x2-inner", "x2-outer", "x3-inner", "x3-outer"):
            cfg["boundary-condition"]["external"][side] = "reflecting"
    return cfg


def initial_w(block):
    """Isothermal hydrostatic p(z) at the cell centres, density times
    (1 + 1e-3 sin(0.7 i + 1.3 j + 2.1 k)) over the whole array, ghosts included."""
    from snapy import kIDN, kIPR
    bufs = dict(block.named_buffers())
    x1v = bufs["coord.x1v"].cpu()
    z = x1v - x1v[NGHOST] + 0.5 * (x1v[NGHOST + 1] - x1v[NGHOST])
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    w = bufs["hydro.D"].clone().zero_()
    shape = w.shape[1:]
    k = torch.arange(shape[0]).view(-1, 1, 1)
    j = torch.arange(shape[1]).view(1, -1, 1)
    i = torch.arange(shape[2]).view(1, 1, -1)
    pert = 1.e-3 * torch.sin(0.7 * i + 1.3 * j + 2.1 * k)
    rho = (p / (RD * T0)).view(1, 1, -1).expand(shape)
    w[kIDN] = rho.to(w) * (1. + pert.to(w))
    w[kIPR] = p.view(1, 1, -1).expand(shape).to(w)
    return w
# ---- end of the #348 copy ----

NARMS = 18

NSTEP = 6
# the moist arms' card: tests/test_anomaly_storage.yaml of this checkout
MOIST_DECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "tests", "test_anomaly_storage.yaml")
# FP32 storage needs perturbation-flux, which needs HLLC (hydro.cpp)
FP32 = {"precision": "float32", "perturbation-flux": True, "anomaly": True,
        "riemann-solver": {"type": "hllc"}}


def arms(moist_deck=MOIST_DECK):
    base = lambda g, s: config(s, "face", g)  # noqa: E731
    out = {}

    def arm(name, geom, scheme, forcing, dyn=None, cells=None):
        cfg = base(geom, scheme)
        cfg["forcing"].update(copy.deepcopy(forcing))
        if dyn:
            cfg["dynamics"].update(dyn)
        if cells:
            cfg["geometry"]["cells"].update(cells)
        out[name] = cfg

    every = {
        "coriolis": {"omega1": 3.e-4, "omega2": 1.e-4, "omega3": -2.e-4},
        "top-sponge-lyr": {"tau": 2.e4, "width": 3.e5},
        "bot-sponge-lyr": {"tau": 3.e4, "width": 2.e5},
        "relax-bot-velo": {"tau": 1.e4, "bvx": 1., "bvy": -2., "bvz": 0.5},
        "relax-bot-temp": {"tau": 1.e4, "btemp": 800., "at-face": True},
        "body-heat": {"dTdt": 1.e-3, "pmin": 1.e5, "pmax": 5.e6},
        "top-cool": {"flux": -50., "depth": 2},
        "bot-heat": {"flux": 70., "thickness": 9.e4},
    }
    for s in (0, 9):  # explicit, and the VIC carrying the step
        arm(f"cart_every_s{s}", "cartesian", s, every)
        arm(f"stretched_every_s{s}", "stretched", s,
            dict(every, **{"top-cool": {"flux": -50., "thickness": 7.e4}}))
    moist = yaml.safe_load(open(moist_deck))
    moist["integration"]["nlim"] = -1
    mforce = {"relax-bot-comp": {"tau": 50., "species": ["H2O"], "xfrac": [0.01]},
              "relax-bot-temp": {"tau": 50., "btemp": 305.},
              "top-sponge-lyr": {"tau": 30., "width": 2.e3},
              "coriolis": {"omega1": 1.e-3, "omega2": 2.e-3}}
    for tag, dyn in (("fp64", {}), ("fp32", FP32)):
        cfg = copy.deepcopy(moist)
        cfg["forcing"].update(copy.deepcopy(mforce))
        cfg["dynamics"].update(copy.deepcopy(dyn))
        out[f"moist_every_{tag}"] = cfg
    arm("cart_123_2d", "cartesian", 0,
        {"coriolis": {"type": "123", "omega1": 3.e-4, "omega2": 1.e-4,
                      "omega3": -2.e-4},
         "top-cool": {"flux": -50., "thickness": 9.e4},
         "bot-heat": {"flux": 70., "depth": 3}})
    arm("cart_123_3d", "cartesian", 0,
        {"coriolis": {"type": "123", "omega1": 3.e-4, "omega2": 1.e-4,
                      "omega3": -2.e-4}}, cells={"nx3": 4})
    arm("cart_diffusion", "cartesian", 0,
        {"diffusion": {"nu_iso": 2.e5, "kappa_iso": 3.e5}})
    arm("cart_diffusion_dyn", "cartesian", 0,
        {"diffusion": {"nu_iso": 1.e-3, "kappa_iso": 2.e-3, "dynamic": True},
         "coriolis": {"omega1": 3.e-4}})
    for s in (0, 9):  # x1 mass-flux gate: Coriolis before and after the solve
        arm(f"sph_coriolis_s{s}", "spherical-polar", s,
            {"coriolis": {"omega1": 1.e-4, "omega3": 2.e-5},
             "top-sponge-lyr": {"tau": 2.e4, "width": 3.e5}})
        arm(f"gno_coriolis_s{s}", "gnomonic-equiangle", s,
            {"coriolis": {"omega1": 1.e-4},
             "relax-bot-velo": {"tau": 1.e4, "bvy": 3.}})
        arm(f"gno_coriolis_trad_s{s}", "gnomonic-equiangle", s,
            {"coriolis": {"omega1": 1.e-4, "traditional": True}})
    arm("gno_aam_transport", "gnomonic-equiangle", 0,
        {"coriolis": {"omega1": 1.e-4}},
        dyn={"conserve-axial-momentum": "transport"})
    arm("gno_aam_full_closure", "gnomonic-equiangle", 0,
        {"coriolis": {"omega1": 1.e-4}},
        dyn={"conserve-axial-momentum": "full",
             "conserve-axial-momentum-closure": True},
        cells={"interp_order": 4})
    return out


def run_arm(cfg, device):
    from snapy import MeshBlock, MeshBlockOptions
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    if "species" in cfg:  # moist: isothermal 300 K dry column, dt from c_s
        from snapy import kIDN, kIPR
        bufs = dict(block.named_buffers())
        z = bufs["coord.x1v"].cpu()
        p = 1.e5 * torch.exp(-z * 10.44 / (287. * 300.))
        w = bufs["hydro.D"].clone().zero_()
        w[kIDN] = (p / (287. * 300.)).view(1, 1, -1).expand(w.shape[1:]).to(w)
        w[kIDN] *= 1. + 1.e-3 * torch.sin(0.7 * torch.arange(w.shape[-1])).to(w)
        w[kIPR] = p.view(1, 1, -1).expand(w.shape[1:]).to(w)
        dt = 0.3 * (z[1] - z[0]).item() / 400.
    else:
        w = initial_w(block)
        dt = DT if cfg["integration"]["implicit-scheme"] else 0.4 * DZ / CS
    block_vars, _ = block.initialize({"hydro_w": w})
    nstage = len(block.module("intg").stages)
    out = []
    for n in range(NSTEP):
        for stage in range(nstage):
            block.forward(block_vars, dt, stage)
        if block.check_redo(block_vars):
            raise RuntimeError(f"redo at step {n + 1}")
        out.append(block_vars["hydro_u"].detach().cpu().clone())
    return torch.stack(out)


def produce(a):
    torch.set_default_dtype(torch.float64)
    os.makedirs(a.out, exist_ok=True)
    for name, cfg in arms(a.moist_deck).items():
        try:
            u = run_arm(cfg, a.device)
        except Exception as e:  # recorded; must match across builds too
            msg = str(e).strip().splitlines()[0][:200]
            open(os.path.join(a.out, name + ".err"), "w").write(msg + "\n")
            print(f"{name:26s} ERROR {msg}", flush=True)
            continue
        torch.save(u, os.path.join(a.out, name + ".pt"))
        h = hashlib.sha256(u.contiguous().numpy().tobytes()).hexdigest()
        u0 = u[0].abs().max().item()
        print(f"{name:26s} {h[:16]} finite={bool(torch.isfinite(u).all())} "
              f"max|du|/max|u| over steps "
              f"{(u[-1] - u[0]).abs().max().item() / u0:.3e}", flush=True)


def compare(a, b):
    """Pass: NARMS arms, each ran on both sides (a .pt, no .err) and the two
    hydro_u histories are bitwise identical (shape, dtype and bytes)."""
    names = sorted({os.path.splitext(f)[0] for d in (a, b) for f in os.listdir(d)
                    if f.endswith((".pt", ".err"))})
    ok = 0
    for n in names:
        pa, pb = (os.path.join(d, n + ".pt") for d in (a, b))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            errs = [open(os.path.join(d, n + ".err")).read().strip()
                    for d in (a, b) if os.path.exists(os.path.join(d, n + ".err"))]
            print(f"{n:26s} FAILED: did not run on both sides: {errs}")
            continue
        ua, ub = torch.load(pa), torch.load(pb)
        same = ua.shape == ub.shape and ua.dtype == ub.dtype and torch.equal(
            ua.contiguous().view(torch.uint8), ub.contiguous().view(torch.uint8))
        print(f"{n:26s} identical={same}"
              + ("" if same or ua.shape != ub.shape
                 else f" maxabs={(ub - ua).abs().max().item():.3e}"))
        ok += same
    print(f"identical {ok}/{len(names)} (expected arms: {NARMS})")
    return ok == len(names) == NARMS


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--moist-deck", default=MOIST_DECK)
    p.add_argument("--compare", nargs=2)
    a = p.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    produce(a)
