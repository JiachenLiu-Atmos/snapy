"""Executable check of the stage increment of the forcing framework (chapter 9, _framework.qmd).

What is checked: the assembly of the stage increment in HydroImpl::forward (src/hydro/hydro_forward.cpp, block
"(6) Calculate external forcing")
at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported on numpy arrays: the divergence term on interior cells,
then each registered module adding dt times its source evaluated on the same W and T, and the dry-density
increment of the forcings kept only when the block carries tracers. Three toy source terms stand in for the
modules. Claims C1-C3.

Run: python3 framework_check.py   (numpy only; exits with the number of failed claims)
"""
import itertools
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


rng = np.random.default_rng(21)
NV, NC = 5, 12                        # variables (IDN, IVX, IVY, IVZ, IPR), cells incl. one ghost each side
W = rng.uniform(0.5, 2.0, (NV, NC))
DIV = rng.normal(size=(NV, NC))
T = 250. + 50. * W[4] / W[0]          # computed once from W, as temp = peos->compute("W->T", {w})
MODULES = [lambda w, t: np.stack([0.1 * w[0], -w[1], np.zeros(NC), np.zeros(NC), 1e-3 * t]),
           lambda w, t: np.stack([np.zeros(NC), np.zeros(NC), 0.3 * w[2], 0.2 * w[3], -0.5 * w[4]]),
           lambda w, t: np.stack([-0.05 * w[0] * t / 300., 0.1 * np.ones(NC), np.zeros(NC), np.zeros(NC),
                                  2. * w[0]])]


def stage_increment(dt, order, track_dry):
    du = np.zeros((NV, NC))
    du[:, 1:-1] = -dt * DIV[:, 1:-1]                 # du.index(interior) = -dt * _div, interior only
    dry_before = du[0].copy() if track_dry else None  # dry_before = du[IDN].clone()
    for m in order:                                   # for (f : forcings) f.forward(du, w, temp, dt)
        du += dt * MODULES[m](W, T)
    dry = du[0] - dry_before if track_dry else None   # _forcing_dry = du[IDN] - dry_before
    return du, dry


def check_C1_loop_order_changes_round_off_only():
    base, _ = stage_increment(0.7, (0, 1, 2), False)
    worst = max(np.abs(stage_increment(0.7, p, False)[0] - base).max() for p in itertools.permutations(range(3)))
    scale = np.abs(base).max()
    return report("C1", worst <= 4 * np.finfo(float).eps * scale, "the 6 orders of 3 modules give the same increment "
                  "up to round-off, since every module reads the same W and T", "max difference %.1e of max |du| %.2f"
                  % (worst, scale))


def check_C2_linear_in_dt():
    d1, _ = stage_increment(0.3, (0, 1, 2), False)
    d2, _ = stage_increment(0.6, (0, 1, 2), False)
    err = np.abs(d2 - 2. * d1).max() / np.abs(d2).max()
    return report("C2", err < 1e-15, "the increment is linear in dt: doubling dt doubles it",
                  "relative difference %.1e" % err)


def check_C3_dry_increment_of_the_forcings():
    du, dry = stage_increment(0.5, (0, 1, 2), True)
    want = 0.5 * sum(m(W, T)[0] for m in MODULES)
    err = np.abs(dry - want).max()
    _, none = stage_increment(0.5, (0, 1, 2), False)
    ok = err < 1e-15 and none is None
    return report("C3", ok, "with tracers, the stored dry increment is the modules' dry-density increment, without the "
                  "divergence term; without tracers it is not formed", "max error %.1e; without tracers: %s"
                  % (err, none))


def main():
    checks = [check_C1_loop_order_changes_round_off_only, check_C2_linear_in_dt, check_C3_dry_increment_of_the_forcings]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
