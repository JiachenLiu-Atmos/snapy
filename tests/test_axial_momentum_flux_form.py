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
        option off   |dL| / |L| > 100 TOL   (the drift this option removes is real)
        option on    |dL| / |L| <= TOL;
      the same with a wavenumber-2 perturbation on top (a flow with a tendency), to compare the
      drift of the steady and the evolving flow with the option off;
  (b) a discretely balanced rest state with the option on stays at rest (max |v| / c_s <= REST_TOL)
      and keeps L at zero to TOL;
  (c) a longer run of the perturbed jet with the option on stays finite with bounded |v| / c_s;
  (d) the key is refused on a Cartesian grid with a clear message;
  (e)-(i) an odd grid with a cell on the axis, Coriolis, the vertically implicit solve with and
      without a vapor species, the traditional approximation, and 2D shallow water.
Each arm builds its own Mesh and frees it before the next.

  python test_axial_momentum_flux_form.py [--device cuda] [--nstep 20] [--nlong 200]
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
TRAD_TOL = 1.0e-10  # traditional approximation on a shallow shell: what the equations leave out
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


def sw_config(on):
    edge = {"x2min_pi": -0.25, "x2max_pi": 0.25, "x3min_pi": -0.25, "x3max_pi": 0.25}
    return {
        "geometry": {"type": "gnomonic-equiangle",
                     "bounds": dict(x1min=6371000.0, x1max=6371001.0, **edge),
                     "cells": {"nx1": 1, "nx2": 16, "nx3": 16, "nghost": 3}},
        "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6, "verbose": False},
        "dynamics": {"conserve-axial-momentum": on,
                     "equation-of-state": {"type": "shallow-water"},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False, "shock": False}},
                     "riemann-solver": {"type": "shallow-roe", "dir": "yz"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "custom", "x2-outer": "custom",
                                            "x3-inner": "custom", "x3-outer": "custom"}},
        "integration": {"type": "rk3", "cfl": 0.5, "nlim": 20, "tlim": 1.0e12, "ncycle_out": 0},
        "forcing": {"coriolis": {"type": "xyz", "omega1": SW_OMEGA}},
    }


def sw_state(blocks):
    """Steady zonal flow u = u0 cos(lat) with phi = phi0 - (a omega u0 + u0^2 / 2) sin^2(lat)."""
    from snapy import kIDN, kIV2, kIV3

    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        lat = coord.buffer("latitude").to(w)
        a = coord.buffer("x1v").to(w)
        k = coord.buffer("aam_k_c").to(w)
        w[kIDN] = SW_PHI0 - (a * SW_OMEGA * SW_U0 + 0.5 * SW_U0 ** 2) * torch.sin(lat) ** 2
        w[kIV2] = SW_U0 * k[1] + 0.0 * a  # u_east k / cos(lat), u_east = u0 cos(lat)
        w[kIV3] = SW_U0 * k[2] + 0.0 * a
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def run_sw(mesh, blocks, mesh_vars, nstep):
    from snapy import kIDN

    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    l0 = aam(blocks, mesh_vars, SW_OMEGA)
    phi0 = [mesh_vars[ib]["hydro_u"][kIDN][interior(b)].clone() for ib, b in enumerate(blocks)]
    drift = 0.0
    for cycle in range(1, nstep + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        d = abs(aam(blocks, mesh_vars, SW_OMEGA) - l0) / abs(l0)
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
    ap.add_argument("--yaml", default=str(Path(__file__).resolve().parent / "test_axial_momentum_flux_form.yaml"))
    args = ap.parse_args(argv)
    if args.device == "cuda" and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    torch.set_default_dtype(torch.float64)

    with open(args.yaml) as f:
        base = yaml.safe_load(f)
    failures = []

    def config(on):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["dynamics"]["conserve-axial-momentum"] = on
        return cfg

    def solid_body(lon, lat, rc):
        return W_SB * rc

    def jet(lon, lat, rc):
        return W_SB * rc + U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon)

    # (a) steady solid body and perturbed jet, option off and on
    l_scale = None  # |L| of the solid-body wind, the scale for the rest arm
    for name, wind, balanced in (("solid body", solid_body, True), ("perturbed jet", jet, True)):
        for on in (False, True):
            cfg = config(on)
            mesh, blocks = build(cfg, args.device)
            mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, wind, balanced), args.nstep)
            if l_scale is None:
                l_scale = abs(l0)
            v = vmax_over_cs(blocks, mesh_vars)
            del mesh, blocks, mesh_vars
            gc.collect()
            print("(a) %-13s option %-5s max |dL| / |L| over %d steps = %.3e (final dL / L = %+.3e)  dE / E = %+.3e  max |v| / c_s = %.3e"
                  % (name, on, args.nstep, drift, final, de, v))
            if not math.isfinite(drift):
                failures.append("%s, option %s: L is not finite" % (name, on))
            elif on and not drift <= TOL:
                failures.append("%s, option on: L drifts %.3e > %.0e" % (name, drift, TOL))
            elif not on and not drift > 100 * TOL:
                failures.append("%s, option off: drift %.3e <= %.0e, the option would not be load-bearing" % (name, drift, 100 * TOL))

    # (b) rest with the option on
    cfg = config(True)
    mesh, blocks = build(cfg, args.device)
    mesh_vars, _, drift, _, _ = run(mesh, blocks, initial_state(cfg, blocks, None, False), args.nstep)
    v = vmax_over_cs(blocks, mesh_vars)
    del mesh, blocks, mesh_vars
    gc.collect()
    drift /= l_scale  # L starts at zero: measure it against the solid-body wind's AAM
    print("(b) rest + option on: max |v| / c_s after %d steps = %.3e, max |L| / |L_solid body| = %.3e"
          % (args.nstep, v, drift))
    if not v < REST_TOL:
        failures.append("rest state moved with the option on: max |v| / c_s = %g" % v)
    if not drift <= TOL:
        failures.append("rest + option on: |L| / |L_solid body| = %.3e > %.0e" % (drift, TOL))

    # (c) a longer perturbed-jet run with the option on stays finite and bounded
    if args.nlong > 0:
        cfg = config(True)
        mesh, blocks = build(cfg, args.device)
        mesh_vars, _, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, jet, True), args.nlong)
        v = vmax_over_cs(blocks, mesh_vars)
        del mesh, blocks, mesh_vars
        gc.collect()
        print("(c) perturbed jet, option on, %d steps: max |dL| / |L| = %.3e, dE / E = %+.3e, max |v| / c_s = %.3e"
              % (args.nlong, drift, de, v))
        if not (math.isfinite(v) and v < 1.0):
            failures.append("long run with the option on: max |v| / c_s = %g" % v)
        if not drift <= TOL:
            failures.append("long run with the option on: L drifts %.3e > %.0e" % (drift, TOL))

    # (e) an odd grid: one cell per polar panel is centred on the axis; its face AAM budget goes
    #     to its neighbours. Solid body plus a seeded random horizontal wind of about 1 m/s.
    torch.manual_seed(314)
    noise = {}

    def noisy(lon, lat, rc):
        key = tuple(lon.shape)
        if key not in noise:
            noise[key] = torch.randn(lon.shape, dtype=lon.dtype, device=lon.device)
        return W_SB * rc + 1.0 * noise[key]

    for on in (False, True):
        cfg = config(on)
        for key in ("nx2", "nx3"):
            cfg["geometry"]["cells"][key] = 9
        mesh, blocks = build(cfg, args.device)
        mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, noisy, True), args.nstep)
        v = vmax_over_cs(blocks, mesh_vars)
        del mesh, blocks, mesh_vars
        gc.collect()
        print("(e) 9x9 noisy wind option %-5s max |dL| / |L| over %d steps = %.3e  dE / E = %+.3e  max |v| / c_s = %.3e"
              % (on, args.nstep, drift, de, v))
        if on and not drift <= TOL:
            failures.append("9x9, option on: L drifts %.3e > %.0e (pole cell)" % (drift, TOL))
        if not on and not drift > 100 * TOL:
            failures.append("9x9, option off: drift %.3e <= %.0e" % (drift, 100 * TOL))

    # (f) with Coriolis (omega1 about z): the absolute AAM, relative plus planetary, with the
    #     Coriolis K component in flux form from the mass fluxes. Solid body in the rotating frame
    #     with its balanced pressure, plus the wavenumber-2 perturbation.
    OMEGA = 7.0e-5
    for on in (False, True):
        cfg = config(on)
        cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": OMEGA}
        mesh, blocks = build(cfg, args.device)
        mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, jet, True, OMEGA),
                                              args.nstep, OMEGA)
        v = vmax_over_cs(blocks, mesh_vars)
        del mesh, blocks, mesh_vars
        gc.collect()
        print("(f) Coriolis, absolute AAM, option %-5s max |dL| / |L| over %d steps = %.3e (final %+.3e)  "
              "dE / E = %+.3e  max |v| / c_s = %.3e" % (on, args.nstep, drift, final, de, v))
        if on and not drift <= TOL:
            failures.append("Coriolis, option on: absolute L drifts %.3e > %.0e" % (drift, TOL))
        if not on and not drift > 100 * TOL:
            failures.append("Coriolis, option off: drift %.3e <= %.0e" % (drift, 100 * TOL))

    # (h) the production path: vertically implicit solve (implicit-scheme 9) with Coriolis and a
    #     vertical wind, so the x1 mass transfers of the solve carry tangential momentum and
    #     planetary AAM between cells. Absolute AAM.
    #     The moist arm adds a vapor species with a vertical gradient: the solve then moves
    #     species mass through the x1 faces and its tangential momentum with it.
    for moist in (False, True):
        for on in (False, True):
            cfg = config(on)
            cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": OMEGA}
            cfg["integration"]["implicit-scheme"] = 9
            if moist:
                cfg["species"] = cfg["species"][:1] + MOIST_SPECIES
                cfg["reactions"] = MOIST_REACTIONS
                cfg["dynamics"]["equation-of-state"]["type"] = "ideal-moist"
            mesh, blocks = build(cfg, args.device)
            mesh_vars, l0, drift, final, de = run(
                mesh, blocks, initial_state(cfg, blocks, jet, True, OMEGA, w_up=W_UP, vapor=QV if moist else 0.0),
                args.nstep, OMEGA)
            v = vmax_over_cs(blocks, mesh_vars)
            del mesh, blocks, mesh_vars
            gc.collect()
            name = "implicit 9 + Coriolis + vertical wind" + (" + vapor" if moist else "")
            print("(h) %s, option %-5s max |dL| / |L| over %d steps = %.3e (final %+.3e)  "
                  "dE / E = %+.3e  max |v| / c_s = %.3e" % (name, on, args.nstep, drift, final, de, v))
            if on and not drift <= TOL:
                failures.append("%s, option on: absolute L drifts %.3e > %.0e" % (name, drift, TOL))
            if not on and not drift > 100 * TOL:
                failures.append("%s, option off: drift %.3e <= %.0e" % (name, drift, 100 * TOL))

    # (g) the traditional approximation in 3D. Its Coriolis force has no w rows, so only the x2
    #     and x3 faces close the planetary AAM, and the absolute AAM changes by the planetary AAM
    #     of the mass moved along x1: the cos(lat) term those equations leave out, of relative
    #     size (shell depth / radius). On the deep shell of the deck that term is the whole change,
    #     option on or off (reported only). On a shallow shell (8 m, vertically implicit) it is
    #     negligible and the option must close the budget.
    for name, depth, check in (("deep shell", None, False), ("8 m shell", 8.0, True)):
        drift_trad = {}
        for on in (False, True):
            cfg = config(on)
            cfg["forcing"]["coriolis"] = {"type": "xyz", "omega1": OMEGA, "traditional": True}
            if depth is not None:
                cfg["geometry"]["bounds"]["x1max"] = float(cfg["geometry"]["bounds"]["x1min"]) + depth
                cfg["integration"]["implicit-scheme"] = 9
            mesh, blocks = build(cfg, args.device)
            mesh_vars, l0, drift, final, de = run(mesh, blocks, initial_state(cfg, blocks, jet, True, OMEGA),
                                                  args.nstep, OMEGA)
            del mesh, blocks, mesh_vars
            gc.collect()
            drift_trad[on] = drift
            print("(g) traditional Coriolis, 3D, %s, absolute AAM, option %-5s max |dL| / |L| over %d steps = %.3e (final %+.3e)"
                  % (name, on, args.nstep, drift, final))
        if check and not drift_trad[True] <= TRAD_TOL:
            failures.append("traditional, %s, option on: absolute L drifts %.3e > %.0e" % (name, drift_trad[True], TRAD_TOL))
        if check and not drift_trad[False] > 100 * TRAD_TOL:
            failures.append("traditional, %s, option off: drift %.3e <= %.0e" % (name, drift_trad[False], 100 * TRAD_TOL))

    # (i) 2D: shallow water on the six panels (one x1 cell, traditional Coriolis by construction),
    #     the steady zonal flow of Williamson et al. (1992) test 2. No mass moves along x1, so the
    #     absolute AAM is exact; the geopotential error measures the balanced flow.
    phi_err = {}
    for on in (False, True):
        cfg = sw_config(on)
        mesh, blocks = build(cfg, args.device)
        drift, err = run_sw(mesh, blocks, sw_state(blocks), args.nstep)
        del mesh, blocks
        gc.collect()
        phi_err[on] = err
        print("(i) shallow water, Williamson 2, option %-5s max |dL| / |L| over %d steps = %.3e  max |dphi| / phi0 = %.3e"
              % (on, args.nstep, drift, err))
        if on and not drift <= TOL:
            failures.append("shallow water, option on: absolute L drifts %.3e > %.0e" % (drift, TOL))
        if not on and not drift > 100 * TOL:
            failures.append("shallow water, option off: drift %.3e <= %.0e" % (drift, 100 * TOL))
    if not phi_err[True] <= 1.5 * phi_err[False]:
        failures.append("shallow water: the balanced-flow error grows with the option on, %.3e against %.3e"
                        % (phi_err[True], phi_err[False]))

    # (d) refused on a Cartesian grid
    cfg = config(True)
    cfg["geometry"] = {"type": "cartesian",
                       "bounds": {"x1min": 0., "x1max": 1.e3, "x2min": 0., "x2max": 1.e3, "x3min": 0., "x3max": 1.e3},
                       "cells": {"nx1": 8, "nx2": 8, "nx3": 8, "nghost": 3}}
    cfg["distribute"] = {"layout": "slab", "nb2": 1, "nb3": 1, "verbose": False}
    cfg["boundary-condition"]["external"] = {k: "periodic" for k in cfg["boundary-condition"]["external"]}
    try:
        from snapy import MeshBlock, MeshBlockOptions

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
            yaml.safe_dump(cfg, f)
            tmp = f.name
        try:
            MeshBlock(MeshBlockOptions.from_yaml(tmp))
        finally:
            os.unlink(tmp)
        failures.append("conserve-axial-momentum accepted on a Cartesian grid")
        print("(d) Cartesian: accepted (unexpected)")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        ok = "conserve-axial-momentum" in msg and "gnomonic-equiangle" in msg
        print("(d) Cartesian: refused, message %s" % ("names the key and the grid" if ok else "unclear: " + msg[:200]))
        if not ok:
            failures.append("Cartesian refusal message is unclear")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
