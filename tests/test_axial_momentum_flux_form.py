#!/usr/bin/env python3
"""Axial angular momentum (AAM) in flux form on the cubed sphere, dynamics/conserve-axial-momentum (#314).

Six gnomonic-equiangle panels in one process (Mesh API): a dry isothermal atmosphere on a shell,
constant gravity, closed x1 walls, no Coriolis, no sponge, no drag. With K = r k the rotation field
about z (buffer aam_k_c, contravariant) and m the conserved covariant momentum rows, the AAM is
  L = sum_cells V r (k^2 m_2 + k^3 m_3),
computed here from the coordinate buffers, independently of the option's own arithmetic.
  (a) a solid-body wind u_east = w r cos(lat) with the pressure of a rigidly rotating isothermal gas
      (an exact steady state of the continuum equations, so the tendency is zero up to the
      discrete balance): over NSTEP rk3 steps
        off        |dL| / |L| > 100 TOL   (the drift this option removes is real)
        transport  |dL| / |L| <= TOL;
      the same with a wavenumber-2 perturbation on top (a flow with a tendency), to compare the
      drift of the steady and the evolving flow with the option off;
  (b) a discretely balanced rest state with transport stays at rest (max |v| / c_s <= REST_TOL)
      and keeps L at zero to TOL;
  (c) a longer run of the perturbed jet with transport stays finite with bounded |v| / c_s;
  (d) the key is off by default, and a Cartesian block, where it does nothing, builds with it on;
  (e) an odd grid with a cell on the axis;
  (f)-(h) with a Coriolis force, which stays in cell form, the exchange with the planetary AAM
      is left open and the absolute AAM (relative plus planetary) is not exact. At slow rotation,
      Omega = the wind's rotation rate, transport must drift at least GAIN times less than off:
      explicit, the traditional approximation on a shallow shell, and the vertically implicit
      solve with a vertical wind, dry and with a vapor species. At the deck's Omega, 14 times the
      wind's rate, the two errors of off cancel in part and the drifts are reported only;
  (i) 2D shallow water, the steady zonal flow of Williamson et al. (1992) test 2: exact without
      rotation. With rotation the gain is small here (measured 1.7 at Omega = u0 / a, 1.1 at the
      Earth's rate): the drift must not exceed that of off at Omega = u0 / a and is reported only
      at the Earth's rate. The balanced-flow error must not grow;
  (j) shallow water at rest plus grid-scale noise, NENERGY steps: the noise decays and the kinetic
      energy left stays within KE_BAND of off, at Omega and at 4 Omega. A state of rest is used
      because a background flow hides a gain in its own energy.
Each arm builds its own Mesh and frees it before the next.

  python test_axial_momentum_flux_form.py [--device cuda] [--nstep 20] [--nlong 200] [--nenergy 4000]
"""
import argparse
import gc
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

TOL = 1.0e-12
REST_TOL = 1.0e-6
GAIN = 10.0  # arms (f)-(h): drift of off over drift of transport at slow rotation (measured 83 to 128)
KE_BAND = 2.0  # arm (j): noise energy left against off (measured 1.06 at Omega, 1.37 at 4 Omega)
W_SB = 5.0e-6  # solid-body rate of the test wind, u_east ~ 30 m/s
U_PERT = 10.0  # wavenumber-2 perturbation amplitude (m/s)
W_UP = 5.0  # vertical wind amplitude of the implicit arm (m/s)
QV = 0.01  # bottom vapor mass fraction of the moist implicit arm, unsaturated at the deck's ~280 K
MOIST_SPECIES = [
    {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 3.5},
    {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.0},
]
MOIST_REACTIONS = [
    {"equation": "vapor <=> cloud", "type": "nucleation", "rate-constant": {"formula": "h2o_ideal"}},
]
P0, RHO0 = 1.0e5, 0.1  # bottom pressure and density; RT = P0 / RHO0


def build(cfg, device):
    from snapy import Mesh, MeshOptions

    nx = cfg["geometry"]["cells"]["nx2"]
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
    return mesh, blocks


def interior(block):
    return tuple(block.part((0, 0, 0), False)[1:])


def aam(blocks, mesh_vars, omega=0.0):
    """Global AAM from the coordinate buffers, double precision, interior cells: the relative part
    V r k . m plus, with omega, the planetary part omega rho r^2 cos^2(lat) V (cos^2 = k . k)."""
    from snapy import kIDN, kIV2, kIV3

    total = 0.0
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        u = mesh_vars[ib]["hydro_u"].double()
        k = coord.buffer("aam_k_c").double()  # [3, nc3, nc2, 1], contravariant
        r = coord.buffer("x1v").double()
        vol = coord.cell_volume().double()
        l = vol * r * (k[1] * u[kIV2] + k[2] * u[kIV3])
        if omega != 0.0:
            from snapy import kICY
            rho = u[kIDN] + u[kICY:].sum(0)  # total density: dry plus the species rows
            l = l + omega * vol * r * r * coord.buffer("aam_kk_c").double() * rho
        total += l[interior(block)].sum().item()
    return total


def energy(blocks, mesh_vars):
    from snapy import kIPR

    total = 0.0
    for ib, block in enumerate(blocks):
        vol = block.module("coord").cell_volume().double()
        total += (vol * mesh_vars[ib]["hydro_u"][kIPR].double())[interior(block)].sum().item()
    return total


def initial_state(cfg, blocks, wind, balanced, omega=0.0, w_up=0.0, vapor=0.0):
    """Isothermal column in the scheme's discrete balance, times the rigid-rotation factor
    exp((w^2 + 2 w omega) s^2 / (2 RT)) when `balanced` (the frame rotates at omega), plus an east
    wind u_east(lon, lat, r cos lat)."""
    import snapy
    from snapy import kIDN, kIPR, kIV2, kIV3

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    ng, nx1 = cfg["geometry"]["cells"]["nghost"], cfg["geometry"]["cells"]["nx1"]
    mesh_vars = []
    for block in blocks:
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
        lon = coord.buffer("longitude").to(w)
        lat = coord.buffer("latitude").to(w)
        rc = coord.buffer("x1v").to(w) * torch.cos(lat)  # r cos(lat), [nc3, nc2, nc1]
        if balanced:
            # rigidly rotating isothermal gas: p, rho ~ exp(w^2 s^2 / (2 R T)), s = r cos(lat)
            fac = torch.exp((W_SB * W_SB + 2.0 * W_SB * omega) * rc ** 2 / (2.0 * P0 / RHO0))
            w[kIDN] *= fac
            w[kIPR] *= fac
        if wind is not None:
            k = coord.buffer("aam_k_c").to(w)  # unit-radius rotation field, contravariant
            u_east = wind(lon, lat, rc)
            # velocity of an east wind u_east: u_east / (r cos lat) * r k = u_east k / cos(lat)
            w[kIV2] = u_east * k[1] / torch.cos(lat)
            w[kIV3] = u_east * k[2] / torch.cos(lat)
        if w_up != 0.0:
            # a wavenumber-3 vertical wind, zero at the two walls
            from snapy import kIV1
            z = (coord.buffer("x1v").to(w) - r[0].item() + 0.5 * dr[0].item()) / (dr.sum().item())
            w[kIV1] = w_up * torch.sin(math.pi * z) * torch.cos(lat) * torch.sin(3.0 * lon)
        if vapor != 0.0:
            # vapor falling off linearly with height to a quarter of its bottom value
            from snapy import kICY
            z = (coord.buffer("x1v").to(w) - r[0].item()) / (r[-1].item() - r[0].item())
            w[kICY] = vapor * (1.0 - 0.75 * z.clamp(0.0, 1.0)) + 0.0 * lat
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def run(mesh, blocks, mesh_vars, nstep, omega=0.0):
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    l0, e0 = aam(blocks, mesh_vars, omega), energy(blocks, mesh_vars)
    drift = 0.0
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
        l = aam(blocks, mesh_vars, omega)
        d = abs(l - l0) / abs(l0) if l0 != 0.0 else abs(l)
        drift = max(drift, d) if math.isfinite(d) else math.inf
    l = aam(blocks, mesh_vars, omega)
    final = (l - l0) / l0 if l0 != 0.0 else l
    de = (energy(blocks, mesh_vars) - e0) / e0
    return mesh_vars, l0, drift, final, de


SW_OMEGA, SW_U0, SW_PHI0 = 7.292e-5, 38.6, 2.94e4  # Williamson et al. (1992) test 2


def sw_config(mode, omega=None):
    edge = {"x2min_pi": -0.25, "x2max_pi": 0.25, "x3min_pi": -0.25, "x3max_pi": 0.25}
    return {
        "geometry": {"type": "gnomonic-equiangle",
                     "bounds": dict(x1min=6371000.0, x1max=6371001.0, **edge),
                     "cells": {"nx1": 1, "nx2": 16, "nx3": 16, "nghost": 3}},
        "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6, "verbose": False},
        "dynamics": {"conserve-axial-momentum": mode,
                     "equation-of-state": {"type": "shallow-water"},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False, "shock": False}},
                     "riemann-solver": {"type": "shallow-roe", "dir": "yz"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "custom", "x2-outer": "custom",
                                            "x3-inner": "custom", "x3-outer": "custom"}},
        "integration": {"type": "rk3", "cfl": 0.5, "nlim": 20, "tlim": 1.0e12, "ncycle_out": 0},
        "forcing": {"coriolis": {"type": "xyz", "omega1": SW_OMEGA if omega is None else omega}},
    }


def sw_state(blocks, omega):
    """Steady zonal flow u = u0 cos(lat) with phi = phi0 - (a omega u0 + u0^2 / 2) sin^2(lat)."""
    from snapy import kIDN, kIV2, kIV3

    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        lat = coord.buffer("latitude").to(w)
        a = coord.buffer("x1v").to(w)
        k = coord.buffer("aam_k_c").to(w)
        w[kIDN] = SW_PHI0 - (a * omega * SW_U0 + 0.5 * SW_U0 ** 2) * torch.sin(lat) ** 2
        w[kIV2] = SW_U0 * k[1] + 0.0 * a  # u_east k / cos(lat), u_east = u0 cos(lat)
        w[kIV3] = SW_U0 * k[2] + 0.0 * a
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def sw_kinetic_energy(blocks, mesh_vars):
    """sum V phi |v|^2 / 2 from the covariant momentum rows; the metric's off-diagonal term is the
    cosine of the angle between the panel's coordinate lines."""
    from snapy import kIDN, kIV2, kIV3

    total = 0.0
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        u = mesh_vars[ib]["hydro_u"].double()
        c = coord.buffer("cosine_cell_kj").double()
        m2 = (u[kIV2] ** 2 + u[kIV3] ** 2 - 2.0 * c * u[kIV2] * u[kIV3]) / (1.0 - c * c)
        total += (0.5 * coord.cell_volume().double() * m2 / u[kIDN])[interior(block)].sum().item()
    return total


def run_sw_rest_noise(mode, omega, nstep, device, amp=0.2):
    """Shallow water at rest plus seeded Gaussian noise, 100 amp on phi and 5 amp m/s on both
    velocity rows: the kinetic energy at nstep / 2 and at nstep, over the initial one."""
    from snapy import kIDN, kIV2, kIV3

    mesh, blocks = build(sw_config(mode, omega), device)
    gen = torch.Generator().manual_seed(357)
    mesh_vars = []
    for block in blocks:
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        w[kIDN] = SW_PHI0 + 100.0 * amp * torch.randn(w[kIDN].shape, generator=gen, dtype=torch.float64).to(w)
        for c in (kIV2, kIV3):
            w[c] = 5.0 * amp * torch.randn(w[c].shape, generator=gen, dtype=torch.float64).to(w)
        mesh_vars.append({"hydro_w": w})
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    ke0, ke = sw_kinetic_energy(blocks, mesh_vars), []
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        if cycle in (nstep // 2, nstep):
            ke.append(sw_kinetic_energy(blocks, mesh_vars) / ke0)
    del mesh, blocks, mesh_vars
    gc.collect()
    return ke


def run_sw(mesh, blocks, mesh_vars, nstep, omega):
    from snapy import kIDN

    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    l0 = aam(blocks, mesh_vars, omega)
    phi0 = [mesh_vars[ib]["hydro_u"][kIDN][interior(b)].clone() for ib, b in enumerate(blocks)]
    drift = 0.0
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        d = abs(aam(blocks, mesh_vars, omega) - l0) / abs(l0)
        drift = max(drift, d) if math.isfinite(d) else math.inf
    err = max(((mesh_vars[ib]["hydro_u"][kIDN][interior(b)] - phi0[ib]).abs().max().item())
              for ib, b in enumerate(blocks)) / SW_PHI0
    return drift, err


def vmax_over_cs(blocks, mesh_vars):
    from snapy import kIDN, kIPR

    vmax, cs = 0.0, 0.0
    for ib, block in enumerate(blocks):
        sl = interior(block)
        u = mesh_vars[ib]["hydro_u"]
        v = u[1:4][(slice(None),) + sl] / u[kIDN][sl]
        vmax = max(vmax, v.abs().max().item())
        # the pressure row of the primitives; the conserved row is an energy, which a species'
        # energy offset can make negative
        p = mesh_vars[ib]["hydro_w"][kIPR][sl] if "hydro_w" in mesh_vars[ib] else u[kIPR][sl]
        cs = max(cs, math.sqrt(1.4 * (p / u[kIDN][sl]).abs().max().item()))
    return vmax / cs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--nstep", type=int, default=20)
    ap.add_argument("--nlong", type=int, default=200)
    ap.add_argument("--nenergy", type=int, default=4000)
    ap.add_argument("--yaml", default=str(Path(__file__).resolve().parent / "test_axial_momentum_flux_form.yaml"))
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    torch.set_default_dtype(torch.float64)

    with open(args.yaml) as f:
        base = yaml.safe_load(f)
    failures = []

    def config(mode):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["dynamics"]["conserve-axial-momentum"] = mode
        return cfg

    def solid_body(lon, lat, rc):
        return W_SB * rc

    def jet(lon, lat, rc):
        return W_SB * rc + U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon)

    # (a) steady solid body and perturbed jet, off and transport
    l_scale = None  # |L| of the solid-body wind, the scale for the rest arm
    for name, wind, balanced in (("solid body", solid_body, True), ("perturbed jet", jet, True)):
        for mode in ("off", "transport"):
            cfg = config(mode)
            mesh, blocks = build(cfg, args.device)
            mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, wind, balanced), args.nstep)
            if l_scale is None:
                l_scale = abs(l0)
            v = vmax_over_cs(blocks, mesh_vars)
            del mesh, blocks, mesh_vars
            gc.collect()
            print("(a) %-13s %-9s max |dL| / |L| over %d steps = %.3e (final dL / L = %+.3e)  dE / E = %+.3e  max |v| / c_s = %.3e"
                  % (name, mode, args.nstep, drift, final, de, v))
            if not math.isfinite(drift):
                failures.append("%s, %s: L is not finite" % (name, mode))
            elif mode == "transport" and not drift <= TOL:
                failures.append("%s, transport: L drifts %.3e > %.0e" % (name, drift, TOL))
            elif mode == "off" and not drift > 100 * TOL:
                failures.append("%s, off: drift %.3e <= %.0e, the option would not be load-bearing" % (name, drift, 100 * TOL))

    # (b) rest with transport
    cfg = config("transport")
    mesh, blocks = build(cfg, args.device)
    mesh_vars, _, drift, _, _ = run(mesh, blocks, initial_state(cfg, blocks, None, False), args.nstep)
    v = vmax_over_cs(blocks, mesh_vars)
    del mesh, blocks, mesh_vars
    gc.collect()
    drift /= l_scale  # L starts at zero: measure it against the solid-body wind's AAM
    print("(b) rest + transport: max |v| / c_s after %d steps = %.3e, max |L| / |L_solid body| = %.3e"
          % (args.nstep, v, drift))
    if not v < REST_TOL:
        failures.append("rest state moved with transport: max |v| / c_s = %g" % v)
    if not drift <= TOL:
        failures.append("rest + transport: |L| / |L_solid body| = %.3e > %.0e" % (drift, TOL))

    # (c) a longer perturbed-jet run with transport stays finite and bounded
    if args.nlong > 0:
        cfg = config("transport")
        mesh, blocks = build(cfg, args.device)
        mesh_vars, _, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, jet, True), args.nlong)
        v = vmax_over_cs(blocks, mesh_vars)
        del mesh, blocks, mesh_vars
        gc.collect()
        print("(c) perturbed jet, transport, %d steps: max |dL| / |L| = %.3e, dE / E = %+.3e, max |v| / c_s = %.3e"
              % (args.nlong, drift, de, v))
        if not (math.isfinite(v) and v < 1.0):
            failures.append("long run with transport: max |v| / c_s = %g" % v)
        if not drift <= TOL:
            failures.append("long run with transport: L drifts %.3e > %.0e" % (drift, TOL))

    # (e) an odd grid: one cell per polar panel is centred on the axis; its face AAM budget goes
    #     to its neighbours. Solid body plus a seeded random horizontal wind of about 1 m/s.
    torch.manual_seed(314)
    noise = {}

    def noisy(lon, lat, rc):
        key = tuple(lon.shape)
        if key not in noise:
            noise[key] = torch.randn(lon.shape, dtype=lon.dtype, device=lon.device)
        return W_SB * rc + 1.0 * noise[key]

    for mode in ("off", "transport"):
        cfg = config(mode)
        for key in ("nx2", "nx3"):
            cfg["geometry"]["cells"][key] = 9
        mesh, blocks = build(cfg, args.device)
        mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, noisy, True), args.nstep)
        v = vmax_over_cs(blocks, mesh_vars)
        del mesh, blocks, mesh_vars
        gc.collect()
        print("(e) 9x9 noisy wind %-9s max |dL| / |L| over %d steps = %.3e  dE / E = %+.3e  max |v| / c_s = %.3e"
              % (mode, args.nstep, drift, de, v))
        if mode == "transport" and not drift <= TOL:
            failures.append("9x9, transport: L drifts %.3e > %.0e (pole cell)" % (drift, TOL))
        if mode == "off" and not drift > 100 * TOL:
            failures.append("9x9, off: drift %.3e <= %.0e" % (drift, 100 * TOL))

    # (f)-(h) with a Coriolis force (omega1 about z), which stays in cell form: the absolute AAM,
    #     relative plus planetary, of the perturbed jet in the rotating frame with its balanced
    #     pressure. The exchange with the planetary AAM is not in flux form, so the drift is not
    #     zero; at Omega = the wind's rotation rate it must be GAIN times below off. At the deck's
    #     Omega the two errors of off cancel in part and transport drifts more (reported only).
    OMEGA = 7.0e-5
    RATES = (("Omega = wind rate", W_SB, True), ("Omega = 14 wind rates", OMEGA, False))

    def with_rotation(tag, name, edit, **state):
        for label, om, check in RATES:
            d = {}
            for mode in ("off", "transport"):
                cfg = config(mode)
                cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": om}
                edit(cfg)
                mesh, blocks = build(cfg, args.device)
                _, _, d[mode], _, _ = run(mesh, blocks, initial_state(cfg, blocks, jet, True, om, **state),
                                          args.nstep, om)
                del mesh, blocks
                gc.collect()
            print("(%s) %s, %s, absolute AAM: max |dL| / |L| over %d steps = %.3e with transport, %.3e with off"
                  % (tag, name, label, args.nstep, d["transport"], d["off"]))
            if check and not GAIN * d["transport"] <= d["off"]:
                failures.append("%s, %s: the drift with transport, %.3e, is not %g times below off, %.3e"
                                % (name, label, d["transport"], GAIN, d["off"]))

    # (f) explicit
    with_rotation("f", "Coriolis", lambda cfg: None)

    # (g) the traditional approximation in 3D on a shallow shell (8 m, vertically implicit), where
    #     the cos(lat) terms it leaves out are negligible
    def traditional(cfg):
        cfg["forcing"]["coriolis"]["traditional"] = True
        cfg["geometry"]["bounds"]["x1max"] = float(cfg["geometry"]["bounds"]["x1min"]) + 8.0
        cfg["integration"]["implicit-scheme"] = 9

    with_rotation("g", "traditional Coriolis, 8 m shell", traditional)

    # (h) the production path: vertically implicit solve (implicit-scheme 9) and a vertical wind,
    #     so the x1 mass transfers of the solve carry tangential momentum between cells. The moist
    #     arm adds a vapor species with a vertical gradient: the solve then moves species mass
    #     through the x1 faces and its tangential momentum with it.
    def implicit(cfg):
        cfg["integration"]["implicit-scheme"] = 9

    def implicit_moist(cfg):
        implicit(cfg)
        cfg["species"] = cfg["species"][:1] + MOIST_SPECIES
        cfg["reactions"] = MOIST_REACTIONS
        cfg["dynamics"]["equation-of-state"]["type"] = "ideal-moist"

    with_rotation("h", "implicit 9 + Coriolis + vertical wind", implicit, w_up=W_UP)
    with_rotation("h", "implicit 9 + Coriolis + vertical wind + vapor", implicit_moist, w_up=W_UP, vapor=QV)

    # (i) 2D: shallow water on the six panels (one x1 cell, traditional Coriolis by construction),
    #     the steady zonal flow of Williamson et al. (1992) test 2. Without rotation the AAM is
    #     exact. With rotation the open Coriolis exchange is most of the drift of off here, so the
    #     gain is small: at Omega = u0 / a transport must not drift more than off (measured 0.60 of
    #     it); at the Earth's rate the drifts are reported (0.93). The geopotential error, which
    #     measures the balanced flow, must not grow.
    for label, om, exact, check in (("no rotation", 0.0, True, False),
                                    ("Omega = u0 / a", SW_U0 / 6371000.0, False, True),
                                    ("Omega of the Earth", SW_OMEGA, False, False)):
        d, phi_err = {}, {}
        for mode in ("off", "transport"):
            cfg = sw_config(mode, om)
            mesh, blocks = build(cfg, args.device)
            d[mode], phi_err[mode] = run_sw(mesh, blocks, sw_state(blocks, om), args.nstep, om)
            del mesh, blocks
            gc.collect()
        print("(i) shallow water, Williamson 2, %s: max |dL| / |L| over %d steps = %.3e with transport, %.3e with off; "
              "max |dphi| / phi0 = %.3e and %.3e"
              % (label, args.nstep, d["transport"], d["off"], phi_err["transport"], phi_err["off"]))
        if exact and not d["transport"] <= TOL:
            failures.append("shallow water, %s, transport: L drifts %.3e > %.0e" % (label, d["transport"], TOL))
        if exact and not d["off"] > 100 * TOL:
            failures.append("shallow water, %s, off: drift %.3e <= %.0e" % (label, d["off"], 100 * TOL))
        if check and not d["transport"] <= d["off"]:
            failures.append("shallow water, %s: the drift with transport, %.3e, exceeds off, %.3e"
                            % (label, d["transport"], d["off"]))
        if not phi_err["transport"] <= 1.5 * phi_err["off"]:
            failures.append("shallow water, %s: the balanced-flow error grows with transport, %.3e against %.3e"
                            % (label, phi_err["transport"], phi_err["off"]))

    # (j) the option must not feed grid-scale noise. Shallow water at rest plus noise, so the noise
    #     is all of the kinetic energy (a background flow hides a gain in its own energy). Transport
    #     has the Coriolis force of off: its noise must decay, and what is left stay within KE_BAND
    #     of off.
    if args.nenergy > 0:
        for mult in (1.0, 4.0):
            om = mult * SW_OMEGA
            ke_off = run_sw_rest_noise("off", om, args.nenergy, args.device)
            ke = run_sw_rest_noise("transport", om, args.nenergy, args.device)
            print("(j) noise on rest, transport at %4.2f Omega: KE / KE0 = %.3e at %d steps and %.3e at %d; off %.3e; "
                  "ratio to off %.3f" % (mult, ke[0], args.nenergy // 2, ke[1], args.nenergy, ke_off[1],
                                         ke[1] / ke_off[1]))
            if not ke[1] < ke[0] < 1.0:
                failures.append("noise on rest, transport at %.2f Omega: the kinetic energy does not decay, %.3e then "
                                "%.3e of the initial one" % (mult, ke[0], ke[1]))
            if not ke[1] <= KE_BAND * ke_off[1]:
                failures.append("noise on rest, transport at %.2f Omega: KE is %.3f of off > %.1f"
                                % (mult, ke[1] / ke_off[1], KE_BAND))

    # (d) off by default, and no effect where it is not defined: a Cartesian block builds with it on
    cfg = config("transport")
    cfg["geometry"] = {"type": "cartesian",
                       "bounds": {"x1min": 0., "x1max": 1.e3, "x2min": 0., "x2max": 1.e3, "x3min": 0., "x3max": 1.e3},
                       "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": 3}}
    cfg["distribute"] = {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False}
    cfg["boundary-condition"]["external"] = {k: "periodic" for k in cfg["boundary-condition"]["external"]}
    from snapy import HydroOptions, MeshBlock, MeshBlockOptions

    if HydroOptions().conserve_axial_momentum() != "off":
        failures.append("conserve-axial-momentum is not off by default")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        MeshBlock(MeshBlockOptions.from_yaml(tmp))
        print("(d) default %s; Cartesian: the block builds with the key on" % HydroOptions().conserve_axial_momentum())
    except Exception as exc:  # noqa: BLE001
        failures.append("conserve-axial-momentum: a Cartesian block does not build: " + str(exc)[:200])
    finally:
        os.unlink(tmp)

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
