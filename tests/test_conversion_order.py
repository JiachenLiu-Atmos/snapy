#!/usr/bin/env python3
"""dynamics/conversion-order 4: a fourth-order x1 cell-average <-> point conversion (#339).

Each arm goes RED when one piece of the key-4 path is taken out.
order    right-moving acoustic pulse, c = 1 + 0.03 exp(-((x - 0.3) / 0.07)^2), from the exact u averages, on a
         uniform x1 and on x1 = s + 0.8 sin(2 pi s) / (2 pi), nx1 = 64, 128, 256 against 1024 on the nested faces:
         fitted order >= 3.8 (key 2 printed), a uniform tracer ratio within 1e-13.
radial   the conversion alone on gnomonic-equiangle and spherical-polar blocks (r^2 dr cells), r in [1, 2]: the
         converted velocity average of exact r^2 averages, nx1 = 64, 128, 256, fitted order >= 3.8.
gravity  a column at rest with T = 1 - 0.8 z (g = 1, discretely balanced) and a density/pressure pulse at z = 0.5,
         nx1 = 64, 128, 256 against 1024: fitted order >= 3.8 (key 4 takes the fourth-order face density
         reference, docs/derivations/wb-ref4.md).
theta    a coarse Straka density current (32 x 128) to t = 900 s: theta' stays within its initial range +- 2 K
         (key 4 takes only the face part of that reference).
mass     |dM|/M <= 1e-13 after 40 steps, nx2 = 4: two pulses against a reflecting wall (one block), and an
         x2-uniform 2:1 jump on the seam of two x1 blocks in one process (the converted ghosts at a wall, a seam
         and in the x2 ghost columns); a pulse leaving through x1 outflow faces, 100 steps: dM/M within 1e-10 of
         the run whose every w ghost layer comes from the outflow condition.
floor    limiter on, a 4-cell pressure dip of 0.06 in p = rho = 1: a temperature floor (1e-4) that only the
         point state crosses must not change the solution (bitwise equal to a floor of 1e-6).
slab     moist EOS, limiter on, 32 x 8 with periodic x2 on two x1 blocks in one process: a vapour slab (0.02,
         exactly 0 outside) crossed by w = 25 sin(pi z / 10 km) and u = 20 m/s, 30 steps: dry and water mass and
         the tracer ratio within 1e-13, every seam ghost of w, corners included, bitwise the neighbour's
         converted cell, and the sum of vapour^2 after 30 steps within 1e-10 of the screened run's (u - L u with
         negative vapour falls back before the EOS; without that the state moves by 1.6e-6).
wall     one stage-0 step of rho = T (1 + d), p = T^2 (1 + d)^G, T = 1 - z/2, d = 1e-3 sin^4(pi z), v = .01
         sin(pi z), g = 1 (gravity-only wall path; p_z + g rho = 0 at both walls; WENO unscaled: scaling near v = 0
         costs an order), reflecting walls, nx1 = 64, 128: the mass tendency error in the three cells at each wall
         falls at order >= 2.5 (mirror: ~1).

  python test_conversion_order.py [--device cpu|cuda] [--arms order,...]
"""
import argparse
import os
import tempfile
import time

import numpy as np
import torch
import yaml
from snapy import Mesh, MeshOptions, balance_column, kICY, kIDN, kIPR, kIV1

G, NG, TOL = 1.4, 3, 1.e-13
X, WQ = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(8))
RD = 8.314462618 / 29.e-3


def mesh(key, nx=64, geo="cartesian", x1f=None, nb1=1, nx2=1, limiter=False, moist=False, tracer=False, grav=0.,
         straka=False, tfloor=1.e-12, x1bc="reflecting", scale=True):
    bounds = {"x1min": 0., "x1max": 1., "x2min": 0., "x2max": 1., "x3min": 0., "x3max": 1.}
    bounds.update({"spherical-polar": {"x1min": 1., "x1max": 2., "x2min": 1., "x2max": 2.},
                   "gnomonic-equiangle": {"x1min": 1., "x1max": 2., "x2min": -.05, "x2max": .05,
                                          "x3min": -.05, "x3max": .05}}.get(geo, {}))
    weno = {"type": "weno5", "scale": scale, "shock": False}
    eos = {"type": "ideal-gas", "gammad": G, "limiter": limiter, "density-floor": 1.e-10, "pressure-floor": 1.e-10}
    cfg = {"geometry": {"type": geo, "bounds": bounds, "cells": {"nx1": nx, "nx2": nx2, "nx3": 1, "nghost": NG}},
           "boundary-condition": {"external": {f"x{d}-{s}": x1bc if d == 1 else "periodic"
                                               for d in (1, 2, 3) for s in ("inner", "outer")}},
           "distribute": {"layout": "cubed" if nb1 > 1 else "slab", "nb1": nb1, "nb2": 1, "nb3": 1,
                          "blocks_per_process": nb1},
           "dynamics": {"equation-of-state": eos, "reconstruct": {"vertical": weno, "horizontal": weno},
                        "riemann-solver": {"type": "lmars"}, "conversion-order": key},
           "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9}}
    if limiter:
        eos.update({"density-floor": 1.e-3, "pressure-floor": 1.e-3, "temperature-floor": tfloor})
    if geo != "cartesian":
        cfg["boundary-condition"]["external"] = {"x1-inner": "reflecting", "x1-outer": "reflecting"}
    if x1f is not None:
        cfg["geometry"]["x1f"] = x1f
    if tracer:
        cfg["scalar"] = {"nvar": 1, "names": ["q"], "reconstruct": {"type": "weno5", "shock": False},
                         "riemann-solver": {"type": "upwind"}}
    if straka:
        bounds.update(x1max=6.4e3, x2max=25.6e3)
        cfg["geometry"]["cells"]["nx2"] = 4 * nx
        eos["weight"] = 29.e-3
        cfg["boundary-condition"]["external"] = {f"x{d}-{s}": "reflecting" for d in (1, 2, 3)
                                                 for s in ("inner", "outer")}
        cfg["integration"]["cfl"] = .9
    if grav:
        cfg["forcing"] = {"const-gravity": {"grav1": -grav}}
    if moist:
        cfg["reference-state"] = {"Tref": 0., "Pref": 1.e5}
        cfg["species"] = [{"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
                          {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
                          {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.}]
        cfg["reactions"] = [{"equation": "vapor <=> cloud", "type": "nucleation",
                             "rate-constant": {"formula": "h2o_ideal"}}]
        bounds["x1max"] = bounds["x2max"] = 1.e4
        cfg["dynamics"]["equation-of-state"] = {"type": "ideal-moist", "density-floor": 1.e-10,
                                                "pressure-floor": 1.e-10, "limiter": True}
        cfg["forcing"] = {"const-gravity": {"grav1": -10.}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
    try:
        options = MeshOptions.from_yaml(f.name)
        options.blocks_per_process(nb1)
        return Mesh(options)
    finally:
        os.unlink(f.name)


def start(m, init, device):
    """init(x1f, x1v, x2v column) -> {w row: values}; returns (state, blocks)"""
    blocks, state = list(m.blocks), []
    for b in blocks:
        b.to(torch.device(device), torch.float64)
        w = b.buffer("hydro.D").clone().zero_()
        c = lambda name: b.buffer(f"coord.{name}").double().cpu()
        for row, val in init(c("x1f"), c("x1v"), c("x2v")[:, None]).items():
            w[row] = val.to(w)
        state.append({"hydro_w": w} | ({"scalar_r": torch.full_like(w[:1], .01)}
                                       if "scalar.D" in dict(b.named_buffers()) else {}))
    return m.initialize(state)[0], blocks


def run(m, init, device, nstep=None, dt=None, tend=None, state=None):
    """steps nstep (or to tend) through the driver's redo loop; returns (state, blocks)"""
    blocks = list(m.blocks)
    if state is None:
        state, blocks = start(m, init, device)
    t, steps, redos = 0., 0, 0
    while (steps < nstep) if nstep is not None else (t < tend):
        m.set_cycle(steps + 1)
        h = dt or m.max_time_step(state)
        for st in range(len(blocks[0].module("intg").stages)):
            m.forward(state, h, st)
        err = m.check_redo(state)
        assert err >= 0, f"terminated after {redos} redos at step {steps}"
        redos, steps, t = redos + (err > 0), steps + (err == 0), t + h * (err == 0)
    assert all(torch.isfinite(s["hydro_u"]).all() for s in state), f"non-finite at step {steps}"
    return state, blocks


def interior(state, blocks, name, row):
    return torch.cat([s[name][row][b.part((0, 0, 0), False)[1:]].cpu().flatten() for s, b in zip(state, blocks)])


def averages(xf, rho, v, p, r2=False):
    """the exact (r^2-weighted) averages of rho, rho v, E, as w = W(u)"""
    r = (xf[:-1, None] + xf[1:, None] + (xf[1:, None] - xf[:-1, None]) * X) / 2
    wr = WQ * (r * r if r2 else 1.)
    avg = lambda q: (q(r) * wr).sum(-1) / wr.sum(-1)
    a, m, e = avg(rho), avg(lambda r: rho(r) * v(r)), avg(lambda r: p(r) / (G - 1) + rho(r) * v(r) ** 2 / 2)
    return {kIDN: a, 1: m / a, kIPR: (G - 1) * (e - m * m / a / 2)}, avg(v)


def pulse(x0, amp=.03, sig=.07):
    c = lambda r: 1. + amp * torch.exp(-((r - x0) / sig) ** 2)
    return lambda r: c(r) ** (2 / (G - 1)), lambda r: 2 / (G - 1) * (c(r) - 1.), lambda r: c(r) ** (2 * G / (G - 1)) / G


def fit(err, nx=(64, 128, 256)):
    return -np.polyfit(np.log(nx), np.log(err), 1)[0]


def ratio(state, blocks):
    """max |r/r0 - 1| of the tracer ratio r = scalar / rho"""
    r = interior(state, blocks, "scalar_s", 0) / interior(state, blocks, "hydro_u", kIDN)
    return float((r / .01 - 1).abs().max())


def drift(m, state, blocks, device, rows, **kw):
    """|dM|/M of the summed interior rows (uniform cells) over a run"""
    mass = lambda: float(sum(interior(state, blocks, "hydro_u", r).sum() for r in rows))
    m0 = mass()
    run(m, None, device, state=state, **kw)
    return abs(mass() - m0) / m0


def arm_order(device, a):
    nref, dt = 1024, 0.25 * (1. - a) / 1024
    def mass(nx, key):
        s = np.linspace(0., 1., nx + 1)
        m = mesh(key, nx, x1f=(s + a * np.sin(2 * np.pi * s) / (2 * np.pi)).tolist() if a else None, tracer=True)
        state, blocks = run(m, lambda xf, *_: averages(xf, *pulse(.3))[0], device, round(.12 / dt), dt)
        xf = blocks[0].buffer("coord.x1f").double().cpu()[NG:-NG]
        return interior(state, blocks, "hydro_u", kIDN) * (xf[1:] - xf[:-1]), ratio(state, blocks)
    out = {}
    for key in (2, 4):
        ref, _ = mass(nref, key)
        runs = [mass(nx, key) for nx in (64, 128, 256)]
        err = [float((r - ref.view(len(r), -1).sum(1)).abs().sum()) for r, _ in runs]
        out[key] = (fit(err), max(q for _, q in runs))
        print(f"order {'stretched' if a else 'uniform'} key {key}: L1 {err} order {out[key][0]:.2f} tracer {out[key][1]:.1e}")
    return out[4][0] >= 3.8 and out[4][1] <= TOL


def arm_radial(device):
    ok = True
    for geo in ("spherical-polar", "gnomonic-equiangle"):
        for key in (2, 4):
            err = []
            for nx in (64, 128, 256):
                exact = {}
                def init(xf, *_):
                    w, exact["v"] = averages(xf, lambda r: 1. + .5 * torch.sin(4 * r) * torch.exp(-r / 2),
                                             lambda r: .1 * torch.cos(3 * r), lambda r: 1. + .2 * torch.sin(2 * r), r2=True)
                    return w
                m = mesh(key, nx, geo=geo)
                state, _ = start(m, init, device)
                m.set_cycle(1)
                m.forward(state, 1.e-12, 0)
                s = slice(2 * NG, -2 * NG)
                err.append(float((state[0]["hydro_w"][1, 0, 0].cpu()[s] - exact["v"][s]).abs().max()))
            print(f"radial {geo} key {key}: max err {err} order {fit(err):.2f}")
            ok &= key == 2 or fit(err) >= 3.8
    # the weights follow a rewritten grid: a uniform block, after one stage, takes a stretched block's buffers
    s = np.linspace(0., 1., 65)
    ma, mb = mesh(4, 64), mesh(4, 64, x1f=(s + .8 * np.sin(2 * np.pi * s) / (2 * np.pi)).tolist())
    init = lambda xf, *_: averages(xf, *pulse(.5))[0]
    sa, ba = start(ma, init, device)
    ma.set_cycle(1), ma.forward(sa, 1.e-12, 0)
    sb, bb = start(mb, init, device)
    grid = dict(ba[0].named_buffers())
    for name, y in bb[0].named_buffers():
        if name.startswith("coord."):
            grid[name].copy_(y)
    sa = [{k: v.clone() for k, v in sb[0].items()}]
    for m, st in ((ma, sa), (mb, sb)):
        m.set_cycle(1), m.forward(st, 1.e-12, 0)
    same = torch.equal(sa[0]["hydro_w"], sb[0]["hydro_w"])
    d = float((sa[0]["hydro_w"] - sb[0]["hydro_w"]).abs().max())
    print(f"radial: a rewritten x1f gives the converted w of a fresh block: {same} (max|dw| {d:.1e})")
    return ok and same


def arm_gravity(device):
    temp = lambda r: 1. - .8 * r
    p0, d = lambda r: temp(r) ** 1.25, lambda r: 1.e-3 * torch.exp(-((r - .5) / .1) ** 2)
    rho0 = lambda r: p0(r) / temp(r)
    def run_nx(nx, dt):
        bg = []
        def init(xf, *_):  # the pulse's exact averages on the balanced column
            w0, w1 = averages(xf, rho0, lambda r: 0 * r, p0)[0], \
                averages(xf, lambda r: rho0(r) * (1 + d(r)), lambda r: 0 * r, lambda r: p0(r) * (1 + d(r)) ** G)[0]
            s = slice(NG, -NG)
            col = torch.stack([w0[kIDN][s]] + [0 * w0[kIDN][s]] * 3 + [w0[kIPR][s]]).view(5, 1, 1, -1)
            wb = balance_column(col, (xf[1:] - xf[:-1])[s].contiguous(), 1., rtol=1.e-13)[0]
            wb = torch.cat([wb[..., :1]] * NG + [wb] + [wb[..., -1:]] * NG, -1)[:, 0, 0]
            bg.append(wb[kIDN][s])
            return {kIDN: wb[kIDN] + w1[kIDN] - w0[kIDN], kIPR: wb[kIPR] + w1[kIPR] - w0[kIPR]}
        state, blocks = run(mesh(4, nx, grav=1.), init, device, round(.1 / dt), dt)
        return (interior(state, blocks, "hydro_u", kIDN) - bg[0]) / nx
    dt = .25 / 1024 / G ** .5
    ref = run_nx(1024, dt)
    err = [float((run_nx(nx, dt) - ref.view(nx, -1).sum(1)).abs().sum()) for nx in (64, 128, 256)]
    print(f"gravity key 4: L1 {err} order {fit(err):.2f}")
    return fit(err) >= 3.8


def arm_wall(device):
    T, d = (lambda r: 1. - r / 2.), (lambda r: 1.e-3 * torch.sin(torch.pi * r) ** 4)
    rho, v, p = (lambda r: T(r) * (1. + d(r))), (lambda r: .01 * torch.sin(torch.pi * r)), \
        (lambda r: T(r) ** 2 * (1. + d(r)) ** G)
    err = []
    for nx in (64, 128):
        m = mesh(4, nx, grav=1., scale=False)
        state, blocks = start(m, lambda xf, *_: averages(xf, rho, v, p)[0], device)
        u0, xf = interior(state, blocks, "hydro_u", kIDN), torch.linspace(0., 1., nx + 1, dtype=torch.float64)
        m.set_cycle(1)
        m.forward(state, 1.e-2, 0)
        e = (interior(state, blocks, "hydro_u", kIDN) - u0) / 1.e-2 + (rho(xf) * v(xf)).diff() * nx
        err.append(float(torch.cat([e[:3], e[-3:]]).abs().max()))
    print(f"wall key 4: mass tendency error in the wall cells {err} order {np.log2(err[0] / err[1]):.2f}")
    return np.log2(err[0] / err[1]) >= 2.5


def arm_theta(device):
    g = 9.8
    cp = G * RD / (G - 1)
    def init(xf, z, x):
        big = torch.sqrt((x / 4.e3) ** 2 + ((z - 3.e3) / 2.e3) ** 2)
        temp = 300. - g * z / cp
        p = 1.e5 * (temp / 300.) ** (cp / RD)
        temp = temp + torch.where(big <= 1, -7.5 * (torch.cos(big * np.pi) + 1), 0.)
        return {kIPR: p + 0. * x, kIDN: p / (RD * temp)}
    m = mesh(4, 32, grav=g, straka=True)
    state, blocks = start(m, init, device)
    th = lambda: (lambda w: w[kIPR] / (RD * w[kIDN]) * (1.e5 / w[kIPR]) ** (RD / cp) - 300.)(
        state[0]["hydro_w"][blocks[0].part((0, 0, 0), False)])
    lo, hi = float(th().min()), float(th().max())
    run(m, None, device, tend=900., state=state)
    a, z = float(th().min()), float(th().max())
    print(f"theta key 4: theta' initial [{lo:.2f}, {hi:.2f}] at t = 900 s [{a:.2f}, {z:.2f}]")
    return lo - 2. <= a and z <= hi + 2.


def arm_mass(device):
    ok = True
    wall = lambda xf, z, y: {k: v + 0. * y for k, v in averages(xf, *pulse(.08))[0].items()}
    jump = lambda xf, z, y: {k: torch.where(z < .5, 1., .5).double() + 0. * y for k in (kIDN, kIPR)}
    for name, nb1, init in (("wall", 1, wall), ("seam jump", 2, jump)):
        m = mesh(4, nb1=nb1, nx2=4)
        state, blocks = start(m, init, device)
        dm = drift(m, state, blocks, device, [kIDN], nstep=40)
        print(f"mass {name} key 4: |dM|/M {dm:.2e}")
        ok &= dm <= TOL
        del m, state, blocks  # a mesh still alive stalls the next one's in-process x1 seam
    m = mesh(4, x1bc="outflow")
    state, blocks = start(m, lambda xf, *_: averages(xf, *pulse(.85))[0], device)
    m0 = float(interior(state, blocks, "hydro_u", kIDN).sum())
    run(m, None, device, 100, 1. / 512, state=state)
    out = (float(interior(state, blocks, "hydro_u", kIDN).sum()) - m0) / m0 / -1.687639303786939e-02 - 1.
    print(f"mass outflow key 4: dM/M vs every w ghost layer from the outflow condition {out:.1e}")
    return ok and abs(out) <= 1.e-10


def arm_floor(device):
    init = lambda xf, x, _: {kIDN: 1. + 0. * x, kIPR: torch.where((x > .45) & (x < .45 + 4 / 64), .06, 1.).double()}
    u = [run(mesh(4, limiter=True, tfloor=tf), init, device, 20)[0][0]["hydro_u"].cpu() for tf in (1.e-6, 1.e-4)]
    print(f"floor key 4: max|u(Tf 1e-4) - u(Tf 1e-6)| {float((u[1] - u[0]).abs().max()):.2e}")
    return torch.equal(*u)


def arm_slab(device):
    def init(xf, z, y):
        p = 1.e5 * torch.exp(-z * 10. / (3900. * 320.)) + 0. * y
        q = (((z - 5.e3).abs() < 1.e3) & ((y - 5.e3).abs() < 2.5e3)).double()
        return {kIDN: p / (3900. * 320.), kIPR: p, kICY: .02 * q, kICY + 1: 1.e-3 * q * ((z - 5.e3).abs() < 400.),
                kIV1: 25. * torch.sin(np.pi * z / 1.e4) + 0. * y, kIV1 + 1: 20. + 0. * p}
    m = mesh(4, nx=32, nb1=2, moist=True, tracer=True, nx2=8)
    state, blocks = start(m, init, device)
    water = lambda: float(sum(interior(state, blocks, "hydro_u", r).sum() for r in (kICY, kICY + 1)))
    q0 = water()
    dd = drift(m, state, blocks, device, [kIDN], nstep=30)
    dq, r = abs(water() - q0) / q0, ratio(state, blocks)
    # the species screen: u - L u with negative vapour falls back before the EOS; the state shows it
    # (the screened run's value with the one-sided wall closure; 5.491166165063751e-05 with the mirror ghosts,
    # 2.1e-8 away through the near-wall dynamics, against 1.6e-6 without the screen)
    v2 = float((interior(state, blocks, "hydro_u", kICY) ** 2).sum()) / 5.491166217528447e-05 - 1.
    m.forward(state, m.max_time_step(state), 0)  # check_redo leaves W(u); the stage's w is the converted one
    w0, w1 = (s["hydro_w"] for s in state)  # every seam ghost, corners too, is the neighbour's converted cell
    same = torch.equal(w0[..., 16 + NG:], w1[..., NG:2 * NG]) and torch.equal(w1[..., :NG], w0[..., 16:16 + NG])
    print(f"slab key 4: |dM|/M dry {dd:.2e} water {dq:.2e}, max|r/r0-1| {r:.2e}, seam ghosts = owners {same}, "
          f"sum vapour^2 vs the screened run {v2:.1e}")
    return max(dd, dq, r) <= TOL and same and abs(v2) <= 1.e-10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--arms", default="order,radial,gravity,theta,mass,floor,slab,wall")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(125)
    arms = {"order": lambda d: arm_order(d, 0.) and arm_order(d, .8), "radial": arm_radial, "gravity": arm_gravity,
            "theta": arm_theta, "mass": arm_mass, "floor": arm_floor, "slab": arm_slab,
            "wall": arm_wall}
    bad = []
    for name in a.arms.split(","):
        t0 = time.time()
        try:
            ok = arms[name](a.device)
        except Exception as e:  # an unknown key or a non-finite state fails the arm, not the run
            print(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            ok = False
        print(f"arm {name}: {'PASS' if ok else 'FAIL'} in {time.time() - t0:.1f} s")
        bad += [] if ok else [name]
    assert not bad, bad
