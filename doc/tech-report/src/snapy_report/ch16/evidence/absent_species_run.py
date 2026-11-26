"""Zero in, zero out in the solver: an absent species under the implicit step.

This is a RUN record, not one of the chapter's checks: it needs a built snapy
in the active Python and is not part of `src/tests`. The committed check
`src/snapy_report/ch16/checks/invariants_zero_check.py` makes the same claim on
a port of the operators and needs nothing but numpy; this one makes it on the
compiled code, with the implicit vertical solve, the moist conserved limiter
and the cross-panel exchange that the port does not carry.

Six gnomonic-equiangular panels of the same shell as `aam_budget_run.py`, with
a four-species moist thermodynamics (dry, NH3, H2S, NH4SH) and the NH3 + H2S
nucleation of `tests/test_condensate_conservation.yaml`. NH3 is PRESENT and is
the control; H2S and its condensate NH4SH are ABSENT, set to exactly zero.
A warm bubble drives vertical motion, so the implicit x1 solve, the positivity
limiter and the species transport all do real work on the present row.

Each cycle it records, over the interiors of all six panels,

  min and count of non-zeros of each absent species row
  min of the present species row  (the limiter's own guarantee)
  the global mass of each species row

The exact answer is that an absent row stays bitwise zero for any number of
steps. A non-zero entry in an absent row is the kintera #147/#151 defect class
appearing in snapy.

What keeps it zero HERE is the zero flux, not the limiter's rectifier, and this
record must not be credited with testing the latter. An identically zero row
reconstructs to zero on both sides of every face, so its species flux is zero,
so `out` and `drain` are zero, so `flux_positivity_theta` returns 1 from the
`else` branch of its `torch::where(drain > 0., ...)` (flux_positivity.cpp:69-71)
and never evaluates the rectified numerator at all. Measured at the pin with the
function instrumented: over the 162 limiter calls of three cycles the two absent
channels have max drain exactly 0.0 and zero cells with drain > 0, while the
present control channel reaches 2.274e+13 over 2456 cells. Consequently, with
`u.relu()` replaced by `u.clamp_min(1e-300)` in the built library this record
still ends GREEN on both CPU and CUDA -- it is insensitive to that term by
construction.

The rectifier is tested instead by `[Z1]` of the committed check, which
prescribes the absent row's face fluxes directly and so can put it in the
draining state the solver will not produce on its own.

Two break arms put the defect into the deck the only way a Python driver can,
through the initial state and the switches:

  --break floor-seed   the two absent rows start at 1e-300 instead of 0, which
                       is what a kintera-style concentration floor would leave
                       behind: the row is no longer absent, so theta is no
                       longer exactly zero and the limiter's bound becomes a
                       relative one at a magnitude where its 4096-eps margin
                       underflows
  --break no-limiter   the same seed with `limiter: false`, so nothing repairs
                       the row at all

A break arm is expected to end RED, and exits nonzero when it does.

  python3 src/snapy_report/ch16/evidence/absent_species_run.py \
      [--device cpu|cuda] [--nx 8] [--cycles 200] [--scheme 9] [--json PATH]
  python3 src/snapy_report/ch16/evidence/absent_species_run.py --break floor-seed

Exits 0 when every absent row is still exactly zero at the end, 1 otherwise;
under `--break` the two are swapped, so a green run of the arm is the failure.
"""
import argparse
import contextlib
import copy
import ctypes
import datetime
import json
import os
import platform
import sys
import tempfile
import time

import torch

from snapy_report.ch16.evidence.aam_budget_run import (
    DECK, P0, RHO0, build, interior, load_cuda_sidecars,
)

PIN = "ccf40f9565642c32be69f4b9ef0cfa6daff3f201"   # doc/tech-report/code_pin.yml

#: dry plus the two vapours and the condensate of tests/test_condensate_conservation.yaml
SPECIES = [
    {"name": "dry", "composition": {"H": 1.5, "He": 0.15}, "cv_R": 2.5},
    {"name": "NH3", "composition": {"N": 1, "H": 3}, "cv_R": 3.5},
    {"name": "H2S", "composition": {"H": 2, "S": 1}, "cv_R": 3.5},
    {"name": "NH4SH", "composition": {"N": 1, "H": 5, "S": 1}, "cv_R": 9.0, "u0_R": -1.2e4},
]
REACTIONS = [{"equation": "NH3 + H2S <=> NH4SH", "type": "nucleation",
              "rate-constant": {"formula": "nh3_h2s_lewis"}}]

PRESENT = 0            # NH3, row ICY + 0: the control, given a non-zero profile
ABSENT = (1, 2)        # H2S and NH4SH, rows ICY + 1 and ICY + 2: exactly zero
NAMES = {0: "NH3 (present control)", 1: "H2S (absent)", 2: "NH4SH (absent)"}


#: the break arms, by name: (what it does, seed shape, seed amplitude, limiter)
BREAKS = {
    "floor-seed": ("the two absent rows start at the 1e-300 a concentration floor "
                   "would leave behind, so they are no longer absent",
                   "uniform", 1.0e-300, True),
    "blob-no-limiter": ("a sharp top-hat of the two rows with the conserved limiter "
                        "switched off, so the reconstruction's undershoot is not repaired",
                        "tophat", 1.0e-6, False),
    "blob-limiter": ("the same sharp top-hat with the limiter on: the control that "
                     "says the limiter, and not the smoothness of the field, is what "
                     "holds the row at or above zero",
                     "tophat", 1.0e-6, True),
}


def deck(nx, scheme, limiter=True):
    cfg = copy.deepcopy(DECK)
    cfg["species"] = copy.deepcopy(SPECIES)
    cfg["reactions"] = copy.deepcopy(REACTIONS)
    # ideal-moist is what gives the hydro array its species rows, and the
    # conserved limiter is the operator under test, so it is ON here
    cfg["dynamics"]["equation-of-state"] = {
        "type": "ideal-moist", "density-floor": 1.0e-20,
        "pressure-floor": 1.0e-20, "limiter": limiter}
    cfg["integration"]["implicit-scheme"] = scheme
    for key in ("nx1", "nx2", "nx3"):
        cfg["geometry"]["cells"][key] = nx
    return cfg


def initial_state(cfg, mesh, blocks, omega, qpresent, dtemp, shape=None, seed=0.0):
    """The balanced isothermal shell of `aam_budget_run.py` plus a warm bubble
    and a non-zero NH3 profile. The two absent rows are left at exactly zero."""
    import snapy
    from snapy import kICY, kIDN, kIPR, kIV1, kIV3
    from snapy.coord import cs_sph_to_contra_

    g = -cfg["forcing"]["const-gravity"]["grav1"]
    ng = cfg["geometry"]["cells"]["nghost"]
    nx1 = cfg["geometry"]["cells"]["nx1"]
    mesh_vars = []
    for face_id, block in enumerate(blocks):
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

        lat = coord.buffer("latitude").to(w)
        lon = coord.buffer("longitude").to(w)
        radius = coord.buffer("x1v").unsqueeze(0).unsqueeze(0).to(w)

        # a warm bubble near the equator, so the implicit x1 solve has work to do
        r0 = 0.5 * (cfg["geometry"]["bounds"]["x1min"] + cfg["geometry"]["bounds"]["x1max"])
        bubble = torch.exp(-((radius - r0) / 1.0e5) ** 2
                           - (lat / 0.3) ** 2 - (lon / 0.3) ** 2)
        w[kIPR] = w[kIPR] * (1.0 + dtemp * bubble)

        # NH3 present everywhere; the two absent rows left at exactly zero,
        # unless a break arm seeds them with what a floor would leave behind
        w[kICY + PRESENT] = qpresent * (1.0 + 0.5 * bubble)
        if shape == "uniform":
            for which in ABSENT:
                w[kICY + which] = seed
        elif shape == "tophat":
            # a discontinuous blob: the stencil of a WENO reconstruction that
            # straddles its edge is exactly where an unlimited face value
            # undershoots below zero
            hat = ((lat.abs() < 0.2) & (lon.abs() < 0.2)
                   & ((radius - r0).abs() < 1.0e5)).to(w) * seed
            for which in ABSENT:
                w[kICY + which] = hat

        vel = torch.zeros_like(w[kIV1:kIV3 + 1])
        vel[2] = omega * radius * torch.cos(lat)
        alpha = coord.buffer("x2v").unsqueeze(0).unsqueeze(-1).to(w)
        beta = coord.buffer("x3v").unsqueeze(-1).unsqueeze(-1).to(w)
        a = alpha.expand_as(vel[0]).contiguous()
        b = beta.expand_as(vel[0]).contiguous()
        cs_sph_to_contra_(vel, a, b, face_id)
        w[kIV1:kIV3 + 1] = vel
        mesh_vars.append({"hydro_w": w})
    mesh_vars, _ = mesh.initialize(mesh_vars)
    return mesh_vars


#: what the thermodynamics prints when it repairs a negative species amount
REPAIR = "Negative concentration"


@contextlib.contextmanager
def capture_native_stdout(sink):
    """Catch what the C++ writes to file descriptor 1.

    A species that goes negative inside a stage is repaired before the step
    ends, so reading the state at the cycle boundary cannot see it. The
    repair announces itself on stdout, from C++, which `contextlib.
    redirect_stdout` does not intercept: only the file descriptor does.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            # the C++ side buffers: without fflush(NULL) its lines reach the real
            # stdout only at interpreter exit, long after fd 1 has been restored
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            sink.append(tmp.read().decode("utf-8", "replace"))


def rows(cfg, blocks, mesh_vars):
    """For each species row: min, count of non-zeros and total mass, over the
    interiors of all six panels."""
    from snapy import kICY
    out = {}
    for which in (PRESENT,) + ABSENT:
        lo, nz, mass = float("inf"), 0, 0.0
        for face_id, block in enumerate(blocks):
            sl = interior(block, cfg)
            vol = block.module("coord").cell_volume()[sl]
            row = mesh_vars[face_id]["hydro_u"][kICY + which][sl]
            lo = min(lo, float(row.min()))
            nz += int(torch.count_nonzero(row))
            mass += float((row * vol).sum())
        out[which] = {"min": lo, "nonzero": nz, "mass": mass}
    return out


def run(args):
    sidecars = load_cuda_sidecars() if args.device == "cuda" else []
    if sidecars:
        print(f"# CUDA sidecars loaded: {', '.join(sidecars)}")
    import snapy
    what, shape, seed, limiter = BREAKS.get(getattr(args, "break"), (None, None, 0.0, True))
    cfg = deck(args.nx, args.scheme, limiter=limiter)
    mesh, blocks = build(cfg, args.device)
    mesh_vars = initial_state(cfg, mesh, blocks, args.omega, args.qpresent,
                              args.dtemp, shape=shape, seed=seed)
    intg = blocks[0].module("intg")

    print(f"# snapy {snapy.__version__} at pin {PIN[:12]}, torch {torch.__version__}, "
          f"python {platform.python_version()}")
    print(f"# device {args.device}, 6 panels, nx = {args.nx}, {args.cycles} cycles, "
          f"implicit-scheme {args.scheme}, moist limiter "
          f"{'on' if limiter else 'OFF'}")
    if what:
        print(f"# break arm {getattr(args, 'break')}: {what}")

    first = rows(cfg, blocks, mesh_vars)
    worst = {w: dict(first[w]) for w in first}
    t0 = time.time()
    cycles_done = 0
    native = []
    with capture_native_stdout(native):
        for cycle in range(1, args.cycles + 1):
            mesh.set_cycle(cycle)
            dt = mesh.max_time_step(mesh_vars)
            for stage in range(len(intg.stages)):
                mesh.forward(mesh_vars, dt, stage)
            if mesh.check_redo(mesh_vars) != 0:
                print(f"# step rejected at cycle {cycle}; stopping")
                break
            cycles_done = cycle
            now = rows(cfg, blocks, mesh_vars)
            for which in now:
                worst[which]["min"] = min(worst[which]["min"], now[which]["min"])
                worst[which]["nonzero"] = max(worst[which]["nonzero"],
                                              now[which]["nonzero"])
            if cycle == 1:
                after_one = {w: dict(now[w]) for w in now}
    wall = time.time() - t0
    final = rows(cfg, blocks, mesh_vars)

    # a species repaired INSIDE a stage is back at or above zero by the time the
    # cycle ends, so the only trace of it is the line the repair prints
    repairs = [ln for ln in "".join(native).splitlines() if REPAIR in ln]
    for line in native:
        sys.stdout.write("".join(ln + "\n" for ln in line.splitlines()
                                 if REPAIR not in ln and ln.strip()))

    print(f"# wall {wall:.1f} s, {cycles_done} cycles")
    print(f"{'species row':24s} {'worst min':>14s} {'max non-zeros':>14s} "
          f"{'after 1 cycle':>14s} {'final mass':>14s}")
    for which in (PRESENT,) + ABSENT:
        print(f"{NAMES[which]:24s} {worst[which]['min']:+14.6e} "
              f"{worst[which]['nonzero']:14d} {after_one[which]['min']:+14.6e} "
              f"{final[which]['mass']:14.6e}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"pin": PIN, "snapy": snapy.__version__, "torch": torch.__version__,
                       "device": args.device, "nx": args.nx, "scheme": args.scheme,
                       "break": getattr(args, "break"), "limiter": limiter,
                       "seed_shape": shape, "seed": seed,
                       "cycles": cycles_done, "wall_s": wall,
                       "worst": {NAMES[w]: worst[w] for w in worst},
                       "final": {NAMES[w]: final[w] for w in final},
                       "when": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                      f, indent=1)

    print(f"# {len(repairs)} repair line(s) of the form "
          f"\"{REPAIR} for species ...\" during the run"
          + (f"; first: {repairs[0].strip()}" if repairs else ""))

    if what:
        # the arm seeds the rows, so "still zero" is not the question any more:
        # the claim it breaks is that the amount never goes NEGATIVE, at a cycle
        # boundary or inside a stage
        lo = min(worst[w]["min"] for w in ABSENT)
        bad = [NAMES[w] for w in ABSENT if worst[w]["min"] < 0.0]
        if bad or repairs:
            where = []
            if bad:
                where.append(f"at a cycle boundary ({', '.join(bad)}, worst {lo:+.6e})")
            if repairs:
                where.append(f"inside a stage ({len(repairs)} repaired negative "
                             f"concentration(s) reported by the thermodynamics)")
            print(f"RED: with the seed, a species amount went negative "
                  f"{' and '.join(where)} over {cycles_done} cycles at implicit-scheme {args.scheme}")
            return 1
        print(f"arm did NOT go red: the seeded rows stayed at or above zero over "
              f"{cycles_done} cycles, worst {lo:+.6e}, with no repair reported")
        return 0

    bad = [NAMES[w] for w in ABSENT
           if worst[w]["min"] != 0.0 or worst[w]["nonzero"] != 0]
    if bad or repairs:
        print(f"RED: {', '.join(bad) or 'a species'} acquired a non-zero or negative "
              f"amount over {cycles_done} cycles at implicit-scheme {args.scheme} "
              f"({len(repairs)} repair line(s))")
        bad = bad or ["(repair reported)"]
    else:
        print(f"GREEN: both absent rows are bitwise zero in every interior cell of all "
              f"6 panels at every one of {cycles_done} cycles at implicit-scheme {args.scheme}, with no "
              f"negative-concentration repair reported at any stage; the present "
              f"control row never goes below {worst[PRESENT]['min']:+.3e}")

    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--nx", type=int, default=8, help="cells per panel edge and in radius")
    p.add_argument("--cycles", type=int, default=200)
    p.add_argument("--scheme", type=int, default=9, help="integration/implicit-scheme")
    p.add_argument("--omega", type=float, default=2.0e-5, help="solid-body rotation [1/s]")
    p.add_argument("--qpresent", type=float, default=1.0e-3, help="NH3 mass fraction")
    p.add_argument("--dtemp", type=float, default=0.1, help="relative warm-bubble amplitude")
    p.add_argument("--break", choices=sorted(BREAKS), default=None,
                   help="run one break arm; it is expected to end RED and exit nonzero")
    p.add_argument("--list-breaks", action="store_true", help="name the break arms and exit")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    if args.list_breaks:
        for name in sorted(BREAKS):
            print(f"{name}\tbreaks [absent-species]\t{BREAKS[name][0]}")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
