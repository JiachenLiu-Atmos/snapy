#!/usr/bin/env python3
"""Corrected-PE gravity work (derivation docs/derivations/curved-gravity-work-weight.md,
option F).

gravity-work: face adds g1 sigma^2 s[drho] to each cell's x1 gravity work,
sigma^2 = <(x1 - r_c)^2> about the centroid r_c of the cell measure (h^2/12 on a Cartesian grid;
r^2-weighted on spherical-polar and the cubed sphere), s = the slope at x1v of
the quadratic through the cell and its two x1 neighbours (one-sided at the block's x1 ends),
and on a gnomonic-equiangle (cubed-sphere) grid, where x1v is the face midpoint and not the
r^2 centroid r_c, also g1 (r_c - x1v) drho (derivation sec 12), so that E + P is conserved, where
  P = sum V [rho phi(x1v) - g1 (r_c - x1v) rho - g1 sigma^2 s[rho]],  phi = -g1 x1,
is the exact potential energy to O(dx^4) (r_c = x1v except on the cubed sphere). E + PE_d
(PE_d = sum V rho phi(x1v)) is not conserved; its drift is printed. Isentropic column (p0 100,
rho0 1, R 1, g 1, depth 100) between closed (reflecting) x1 walls, seeded u1 = 0.05 c_s
sin(pi z / L) (2-D: times cos(2 pi x2 / L)), weno5, lmars, rk3. Checked:
  1. |d(E + P)| / |E + P| per step <= 1e-14 on a spherical-polar column (x1 in [300, 400]),
     a Cartesian column and a 4x4-column gnomonic-equiangle block (x1 in [300, 400]), explicit
     and VIC (implicit-scheme 9), and on a 2-D Cartesian box with nx3 = 1, explicit and VIC;
  2. the cycle diagnostics (print_cycle_info) log the potential energy the booked work conserves:
     ie= + pe= equals this test's E + P to the printed digits (spherical, Cartesian, 2-D Cartesian
     and gnomonic, after a few explicit steps);
  3. a gnomonic-equiangle (cubed-sphere) block with gravity-work: face sets up and does not warn
     (the work has a form there); a cylindrical block, which has none, warns;
  4. one VIC solve whose availability clamp binds (a momentum kick, as in
     test_implicit_stratified_solid.py's clamp_energy), so the solved density change differs from the raw
     one: its E + P defect is round-off, and the defect the same solve shows if the corrected-PE work had
     used the raw change instead (the mutation, computed from the two changes) is far above it;
  5. the VIC column with a 4-cell immersed solid block: per-step E + P over the fluid cells <= 1e-14,
     and every solve's energy correction in the solid cells is exactly zero, while the corrected-PE work
     g1 sigma^2 s[drho] there, which the solid gate removes, is not.
The diagnostics log is flushed at exit, so each diagnostics case runs in a child process.

  python test_gravity_work_radial_exact.py [--device cpu|cuda]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import torch
import yaml

GAMMA, P0, RHO0, G, DEPTH, NZ, NG = 1.4, 100., 1., 1., 100., 32, 3
NSTEP, EP_TOL = 20, 1.e-14
DIAG_STEPS, DIAG_TOL, DIAG_CASES = 5, 1.e-11, ("sph", "cart", "cart2d", "cs")
CASES = {  # name: (geometry, implicit scheme, nx2)
    "sph": ("spherical-polar", 0, 1), "sph_vic": ("spherical-polar", 9, 1),
    "cart": ("cartesian", 0, 1), "cart_vic": ("cartesian", 9, 1),
    "cart2d": ("cartesian", 0, 16), "cart2d_vic": ("cartesian", 9, 16),
    "cs": ("gnomonic-equiangle", 0, 4), "cs_vic": ("gnomonic-equiangle", 9, 4)}


def config(geometry, scheme, nx2, recon, ncycle_out=0):
    if geometry == "spherical-polar":
        x1min, x2 = 300., (0.5 * math.pi - 0.05, 0.5 * math.pi + 0.05)
        bounds = {"x1min": x1min, "x1max": x1min + DEPTH, "x2min": x2[0], "x2max": x2[1],
                  "x3min": 0.0, "x3max": 0.1}
        x2bc = "reflecting"
    elif geometry == "gnomonic-equiangle":  # one face block, nx2 x nx2 columns
        bounds = {"x1min": 300., "x1max": 300. + DEPTH, "x2min_pi": -0.25, "x2max_pi": 0.25,
                  "x3min_pi": -0.25, "x3max_pi": 0.25}
        x2bc = "reflecting"
    else:
        bounds = {"x1min": 0.0, "x1max": DEPTH, "x2min": 0.0, "x2max": DEPTH, "x3min": 0.0, "x3max": 1.0}
        x2bc = "periodic"
    nx3 = nx2 if geometry == "gnomonic-equiangle" else 1
    x3bc = "reflecting" if geometry == "gnomonic-equiangle" else "periodic"
    return {
        "geometry": {"type": geometry, "bounds": bounds,
                     "cells": {"nx1": NZ, "nx2": nx2, "nx3": nx3, "nghost": NG}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-12, "limiter": True},
            "reconstruct": {"vertical": {"type": recon, "scale": False, "shock": False},
                            "horizontal": {"type": recon, "scale": False, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": x2bc, "x2-outer": x2bc,
                                            "x3-inner": x3bc, "x3-outer": x3bc}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9,
                        "ncycle_out": ncycle_out},
        "forcing": {"const-gravity": {"grav1": -G, "gravity-work": "face"}},
    }


def interior(shape):
    return tuple(slice(NG, -NG) if n > 1 else slice(None) for n in shape)


SOLID = slice(NG + 14, NG + 18)  # the immersed solid block of check 8


def isentrope(z):
    K, ex = P0 / RHO0 ** GAMMA, (GAMMA - 1.) / GAMMA
    p = (P0 ** ex - ex * G * z / K ** (1. / GAMMA)).clamp(min=1.e-3) ** (1. / ex)
    return (p / K) ** (1. / GAMMA), p


def build(case, seed=0.05, recon="weno5", balanced=False, device="cpu", ncycle_out=0, solid=False):
    import snapy
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
    geometry, scheme, nx2 = CASES[case]
    cfg = config(geometry, scheme, nx2, recon, ncycle_out)
    if solid:  # Cartesian: the solid takes the column's state at its lowest cell
        rho_s, p_s = isentrope(torch.tensor((SOLID.start - NG + 0.5) * DEPTH / NZ))
        cfg["boundary-condition"]["internal"] = {"solid-density": float(rho_s), "solid-pressure": float(p_s)}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        b = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    b.to(torch.device(device), torch.float64)
    x1f, x1v, x2v = (b.buffer("coord." + k).cpu() for k in ("x1f", "x1v", "x2v"))
    w = b.buffer("hydro.D").clone().zero_().cpu()
    z = x1v - float(x1f[NG])
    w[kIDN], w[kIPR] = isentrope(z)
    if balanced:  # Cartesian columns only
        I = slice(NG, NG + NZ)
        col = w[:kIPR + 1, ..., I].clone()
        dx1f = (x1f[1:] - x1f[:-1])[I].contiguous()
        wb, _, _ = snapy.balance_column(col.contiguous(), dx1f, G, True, 3e-14, 400)
        w[:kIPR + 1, ..., I] = wb
        for c in (kIDN, kIPR):
            w[c][..., :NG] = w[c][..., NG:NG + 1]
            w[c][..., NG + NZ:] = w[c][..., NG + NZ - 1:NG + NZ]
    u1 = seed * math.sqrt(GAMMA * P0 / RHO0) * torch.sin(math.pi * z / DEPTH)
    u1 = torch.where((z > 0) & (z < DEPTH), u1, torch.zeros_like(u1))
    # gnomonic-equiangle: radial flow only. A lone face block with reflecting x2/x3 walls does not
    # conserve mass to round-off under lateral flow (~1e-11 per step with the switch on or off),
    # which would hide the x1 work's E + P closure this test is about
    if nx2 > 1 and geometry != "gnomonic-equiangle":
        u1 = u1 * torch.cos(2. * math.pi * x2v / DEPTH)[:, None]
    w[kIV1] = u1.expand_as(w[kIV1])
    if not solid:
        v, _ = b.initialize({"hydro_w": w.to(device)})
        return b, v
    mask = torch.zeros_like(w[kIDN], dtype=torch.bool)
    mask[..., SOLID] = True
    w[kIV1][mask] = 0.
    for c, key in ((kIDN, "solid-density"), (kIPR, "solid-pressure")):
        w[c][mask] = cfg["boundary-condition"]["internal"][key]
    v, _ = b.initialize({"hydro_w": w.to(device), "solid": mask.to(device)})
    return b, v


def slope(q, x):
    """s[q] of gravity_work_radial.hpp along the last axis (x = interior x1v)."""
    s = torch.zeros_like(q)
    hm, hp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    s[..., 1:-1] = (-hp / (hm * (hm + hp)) * q[..., :-2] + (hp - hm) / (hm * hp) * q[..., 1:-1]
                    + hm / (hp * (hm + hp)) * q[..., 2:])
    a, c = x[1] - x[0], x[2] - x[1]
    s[..., 0] = -(2 * a + c) / (a * (a + c)) * q[..., 0] + (a + c) / (a * c) * q[..., 1] - a / ((a + c) * c) * q[..., 2]
    a, c = x[-2] - x[-3], x[-1] - x[-2]
    s[..., -1] = c / (a * (a + c)) * q[..., -3] - (a + c) / (a * c) * q[..., -2] + (a + 2 * c) / ((a + c) * c) * q[..., -1]
    return s


def centroid_offset(x1f, x1v):
    """r_c - x1v, written about the cell midpoint: rb h^3 / (6 V_r) + rb - x1v (sec 12)"""
    h, rb = x1f[1:] - x1f[:-1], 0.5 * (x1f[1:] + x1f[:-1])
    return rb * h ** 3 / (6. * (rb * rb * h + h ** 3 / 12.)) + (rb - x1v)


def variance(x1f, spherical):
    """sigma^2 about the r^2 centroid r_c when spherical (h^2/12 otherwise), written about the cell
    midpoint (no cancellation at large x1 / dx)."""
    h, rb = x1f[1:] - x1f[:-1], 0.5 * (x1f[1:] + x1f[:-1])
    if not spherical:
        return h * h / 12.
    vol = rb * rb * h + h ** 3 / 12.
    return (rb * rb * h ** 3 / 12. + h ** 5 / 80.) / vol - (rb * h ** 3 / (6. * vol)) ** 2


class Column:
    def __init__(self, b, case):
        sph = CASES[case][0] == "spherical-polar"
        cs = CASES[case][0] == "gnomonic-equiangle"
        u = b.buffer("hydro.D")
        self.sl = (slice(None),) + interior(u.shape[1:])
        f1, f2, f3 = (b.buffer("coord." + k).cpu() for k in ("x1f", "x2f", "x3f"))
        f2, f3 = (f[NG:-NG] if f.numel() > 2 else f for f in (f2, f3))
        x1f = f1[NG:NG + NZ + 1]
        if sph:  # (r+^3 - r-^3)/3 (cos th- - cos th+) dphi
            rad, lat = (x1f[1:] ** 3 - x1f[:-1] ** 3) / 3., f2[:-1].cos() - f2[1:].cos()
        else:
            rad, lat = x1f[1:] - x1f[:-1], f2[1:] - f2[:-1]
        self.vol = (f3[1:] - f3[:-1])[:, None, None] * lat[None, :, None] * rad[None, None, :]
        self.x = b.buffer("coord.x1v").cpu()[NG:NG + NZ]
        self.var = variance(x1f, sph or cs)
        self.off = centroid_offset(x1f, self.x) if cs else torch.zeros_like(self.x)
        if cs:  # the exact gnomonic solid angle times (r+^3 - r-^3)/3
            self.vol = b.module("coord").cell_volume().cpu()[self.sl[1:]]

    def energies(self, v, fluid=None):
        """(E + PE_d, E + P) of the interior, or of its fluid cells"""
        from snapy import kIDN, kIPR
        u = v["hydro_u"][self.sl].cpu()
        vol = self.vol if fluid is None else self.vol * fluid
        E = (u[kIPR] * vol).sum()
        ped = (u[kIDN] * G * self.x * vol).sum()
        corr = self.var * slope(u[kIDN], self.x) + self.off * u[kIDN]
        return float(E + ped), float(E + ped + G * (corr * vol).sum())


def run(case, steps, seed=0.05, balanced=False, device="cpu", solid=False):
    from snapy import kIDN, kIPR, kIV1
    b, v = build(case, seed=seed, balanced=balanced, device=device, solid=solid)
    col = Column(b, case)
    fluid = None
    if solid:
        fluid = torch.ones(NZ)
        fluid[SOLID.start - NG:SOLID.stop - NG] = 0.
    dz = DEPTH / NZ
    dt = (0.3 if CASES[case][1] == 0 else 1.2) * dz / math.sqrt(GAMMA * P0 / RHO0)
    epd0, ep0 = col.energies(v, fluid)
    dEP = dEPd = umax = gate = ungated = 0.
    for _ in range(steps):
        b.inc_cycle()
        for st in range(len(b.module("intg").stages)):
            b.forward(v, dt, st)
            if solid:  # check 8: this solve's correction, and the work the gate keeps out of the solid
                corr = b.buffer("hydro.icorr.corr").cpu()
                gate = max(gate, float(corr[kIPR][..., SOLID].abs().max()))
                work = G * col.var * slope(corr[kIDN][..., NG:NG + NZ], col.x)
                ungated = max(ungated, float(work[..., SOLID.start - NG:SOLID.stop - NG].abs().max()))
        assert b.check_redo(v) == 0
        epd, ep = col.energies(v, fluid)
        dEP, dEPd = max(dEP, abs(ep - ep0) / abs(ep0)), max(dEPd, abs(epd - epd0) / abs(epd0))
        epd0, ep0 = epd, ep
        w = b.module("hydro.eos").compute("U->W", [v["hydro_u"]])[col.sl].cpu()
        umax = max(umax, float((w[kIV1].abs() / torch.sqrt(GAMMA * w[kIPR] / w[kIDN])).max()))
    return {"dEP": dEP, "dEPd": dEPd, "umax": umax, "gate": gate, "ungated": ungated}, v["hydro_u"].cpu().clone()


def diag(case, out, device="cpu"):
    """a few explicit steps, then the cycle diagnostics on stdout; this test's (E + PE_d, E + P) to out"""
    b, v = build(case, device=device, ncycle_out=1)
    col = Column(b, case)
    dt = 0.3 * DEPTH / NZ / math.sqrt(GAMMA * P0 / RHO0)
    for _ in range(DIAG_STEPS):
        b.inc_cycle()
        for st in range(len(b.module("intg").stages)):
            b.forward(v, dt, st)
    b.print_cycle_info(v, 0., dt)
    json.dump(col.energies(v), open(os.path.join(out, "diag_%s.json" % case), "w"))


def setup_error(geometry):
    """the setup error of a gnomonic-equiangle or cylindrical block with gravity-work: face, or ''
    if it builds, and what the setup wrote to stderr"""
    from snapy import MeshBlock, MeshBlockOptions
    if geometry == "gnomonic-equiangle":
        cfg = config(geometry, 0, 4, "weno5")
    else:  # x1 = R in [300, 400], x2 = phi, x3 = z
        cfg = config("cartesian", 0, 4, "weno5")
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["bounds"] = {"x1min": 300., "x1max": 300. + DEPTH, "x2min": 0., "x2max": 0.1,
                                     "x3min": 0., "x3max": 1.}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    log, fd = tempfile.TemporaryFile("w+"), os.dup(2)
    sys.stderr.flush()
    os.dup2(log.fileno(), 2)  # the C++ warning goes to fd 2
    try:
        MeshBlock(MeshBlockOptions.from_yaml(tmp))
        msg = ""
    except Exception as e:  # noqa: BLE001 -- the message is checked by the caller
        msg = str(e)
    finally:
        sys.stderr.flush()
        os.dup2(fd, 2)
        os.close(fd)
        os.unlink(tmp)
    log.seek(0)
    return msg, log.read()


def clamp_solve(device="cpu"):
    """check 7: one VIC solve with a binding availability clamp; per-solve defects, mutation and clamp size"""
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1
    n = 8
    cfg = config("cartesian", 9, 1, "weno5")
    cfg["geometry"]["cells"]["nx1"] = n
    cfg["geometry"]["bounds"]["x1max"] = float(n)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        b = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    b.to(torch.device(device), torch.float64)
    w = b.buffer("hydro.D").clone().zero_()
    w[kIDN], w[kIPR] = 1., 1.
    du = torch.zeros_like(w)
    du[kIV1, ..., NG + n - 2] = 100.
    b.module("hydro.icorr").forward(du, w, torch.full_like(w[kIDN], GAMMA), 1.0)
    sl = (0, 0, slice(NG, NG + n))
    x, var = torch.arange(n) + 0.5, 1. / 12.
    drho, dE = du[kIDN][sl].cpu(), du[kIPR][sl].cpu()
    raw = b.buffer("hydro.icorr.delta").cpu().reshape(n, -1)[:, 0]
    epd = float((dE + G * x * drho).sum())
    ep = epd + G * var * float(slope(drho, x).sum())
    # had the work used the raw change, dE would differ by g1 sigma^2 (s[raw] - s[solved])
    mut = ep + G * var * float((slope(drho, x) - slope(raw, x)).sum())
    return {"ep": ep, "epd": epd, "mut": mut, "scale": float(dE.abs().sum()),
            "solved_raw": float((drho - raw).abs().max()), "drho": float(drho.abs().max()),
            "clamp": b.buffer("hydro.icorr.dry_clamp_step").item()}


def child(out, device):
    res = {}
    res["setup_error"] = {g: setup_error(g) for g in ("gnomonic-equiangle", "cylindrical")}
    res["clamp_vic"] = clamp_solve(device)
    res["solid_vic"] = run("cart_vic", NSTEP, device=device, solid=True)[0]
    for case in CASES:
        res[case] = run(case, NSTEP, device=device)[0]
    json.dump(res, open(os.path.join(out, "res.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--child", default=None)
    ap.add_argument("--diag", default=None)
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available")
        sys.exit(125)
    torch.set_default_dtype(torch.float64)
    if a.diag:
        diag(a.diag, a.child, a.device)
        return
    if a.child:
        child(a.child, a.device)
        return
    failures = []
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        out = os.path.join(tmp, "run")
        os.makedirs(out)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                        "--child", out], check=True)
        res = json.load(open(os.path.join(out, "res.json")))
        for case in DIAG_CASES:  # one process each: the log is flushed at exit
            log = subprocess.run([sys.executable, os.path.abspath(__file__), "--device", a.device,
                                  "--diag", case, "--child", out], check=True,
                                 capture_output=True, text=True).stdout
            ie, pe = (float(re.search(r" %s=(\S+)" % k, log).group(1)) for k in ("ie", "pe"))
            epd, ep = json.load(open(os.path.join(out, "diag_%s.json" % case)))
            err = abs(ie + pe - ep) / abs(ep)
            print(f"{case:10s} logged ie + pe vs E + P: rel diff {err:.2e} "
                  f"(E + P vs E + PE_d: {abs(ep - epd) / abs(ep):.2e})", flush=True)
            if not err <= DIAG_TOL:
                failures.append(f"{case}: logged ie + pe is not E + P (rel diff {err:.2e})")
    for case in CASES:
        r = res[case]
        print(f"{case:10s} max per-step |d(E+P)|/|E+P| {r['dEP']:.2e}   "
              f"|d(E+PE_d)|/|E+PE_d| {r['dEPd']:.2e}", flush=True)
        if not r["dEP"] <= EP_TOL:
            failures.append(f"{case}: per-step E+P change {r['dEP']:.2e} > {EP_TOL}")
    for geometry, (msg, log) in res["setup_error"].items():
        warned = "has no form" in log
        print(f"{geometry:18s} setup {'error: ' + msg.splitlines()[0] if msg else 'ok'}, "
              f"no-form warning {'yes' if warned else 'no'}", flush=True)
        if msg:
            failures.append(f"{geometry}: the block failed at setup")
        if warned != (geometry == "cylindrical"):  # the cubed sphere has a form
            failures.append(f"{geometry}: the no-form warning {'fired' if warned else 'is missing'}")
    c, sv = res["clamp_vic"], res["solid_vic"]
    print(f"clamp VIC: dry clamp {c['clamp']:.0f}, max|solved - raw drho| {c['solved_raw']:.2e} "
          f"(max|drho| {c['drho']:.2e}); E + P defect {c['ep']:.2e}, E + PE_d {c['epd']:.2e}, "
          f"raw-change mutation {c['mut']:.2e} (sum|dE| {c['scale']:.2e})", flush=True)
    print(f"solid VIC: fluid max per-step |d(E+P)|/|E+P| {sv['dEP']:.2e}, "
          f"|d(E+PE_d)|/|E+PE_d| {sv['dEPd']:.2e}; solid max|dE| {sv['gate']:.2e} "
          f"(ungated work {sv['ungated']:.2e})", flush=True)
    tol = 1.e-13 * c["scale"]
    if not (c["solved_raw"] >= 1.e-3 * c["drho"] and abs(c["ep"]) <= tol and abs(c["mut"]) >= 1.e3 * tol):
        failures.append(f"clamp VIC: solved vs raw {c['solved_raw']:.2e}, E + P defect {c['ep']:.2e}, "
                        f"mutation {c['mut']:.2e}, tolerance {tol:.2e}")
    if not sv["dEP"] <= EP_TOL:
        failures.append(f"solid VIC: fluid per-step E+P change {sv['dEP']:.2e} > {EP_TOL}")
    if not (sv["gate"] == 0. and sv["ungated"] > 0.):  # fluid E+P closes either way
        failures.append(f"solid VIC: solid-cell energy correction {sv['gate']:.2e} (ungated work "
                        f"{sv['ungated']:.2e}); the gate must book none")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
