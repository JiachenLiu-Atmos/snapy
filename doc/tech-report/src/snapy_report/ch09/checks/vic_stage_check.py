"""Executable check of the stage-weighted implicit step (chapter 7, _vic-stage.qmd).

What is checked: dt_corr = wght2 * dt for an integrator of three or more stages (rk3, rk3s4), the full dt for rk1 and
rk2 (HydroImpl::forward, src/hydro/hydro_forward.cpp at snapy@68f4e066974ede87e4379285c4fba895b970b7f1), with the stage update u = w0 u0 + w1 u + w2 du of
pyharp src/integrator/integrator.cpp:101-122 and the VIC solve (I/dt_corr + M) delta = du/dt_corr. A scalar model: the explicit
tendency is L(u) = -lam u and the operator Jacobian is M = lam, so z = lam dt. Claims C1-C5; C5 is exact (sympy).

Run: python3 vic_stage_check.py   (numpy and sympy; exits with the number of failed claims)
"""
import sys

import numpy as np
import sympy as sp


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


# integrator.cpp:35-78, (wght0, wght1, wght2) per stage, as in rk_check.py
WEIGHTS = {
    "rk1": [(0., 1., 1.)],
    "rk2": [(0., 1., 1.), (.5, .5, .5)],
    "rk3": [(0., 1., 1.), (3. / 4., 1. / 4., 1. / 4.), (1. / 3., 2. / 3., 2. / 3.)],
    "rk3s4": [(.5, .5, .5), (0., 1., .5), (2. / 3., 1. / 3., 1. / 6.), (0., 1., .5)],
}


def dt_factor(kind, stage):
    """HydroImpl::forward: an integrator of three or more stages weights the dt of the correction."""
    return WEIGHTS[kind][stage][2] if len(WEIGHTS[kind]) >= 3 else 1.


def amplification(kind, z, weighted=True):
    """One step of u' = -lam u with the VIC correction in every stage; z = lam dt."""
    u0 = u = np.ones_like(z)
    for s, (w0, w1, w2) in enumerate(WEIGHTS[kind]):
        f = dt_factor(kind, s) if weighted else 1.
        du = -z * u                                   # dt L(u)
        delta = du / (1. + f * z)                     # (I + f dt M)^-1 du
        u = w0 * u0 + w1 * u + w2 * delta
    return u


def check_C1_weight_inside_equals_weight_outside():
    rng = np.random.default_rng(3)
    worst = 0.
    for _ in range(100):
        n = 6
        M = rng.normal(size=(n, n)) + 4 * np.eye(n)
        L = rng.normal(size=n)
        dt, w = rng.uniform(.1, 10.), rng.choice([1., .25, 2. / 3.])
        lhs = w * np.linalg.solve(np.eye(n) + w * dt * M, dt * L)
        rhs = np.linalg.solve(np.eye(n) + w * dt * M, w * dt * L)
        worst = max(worst, np.abs(lhs - rhs).max() / np.abs(rhs).max())
    return report("C1", worst < 1e-13,
                  "w2 (I + w2 dt M)^-1 (dt L) = (I + w2 dt M)^-1 (w2 dt L): weighting the solved du is the same "
                  "as weighting the right-hand side, once dt_corr carries w2", "max relative difference %.1e" % worst)


def check_C2_three_or_more_stages_are_weighted():
    f = {k: [dt_factor(k, s) for s in range(len(v))] for k, v in WEIGHTS.items()}
    ok = (f["rk3"] == [1., .25, 2. / 3.] and f["rk3s4"] == [.5, .5, 1. / 6., .5]
          and all(x == 1. for k in ("rk1", "rk2") for x in f[k]))
    return report("C2", ok, "dt_corr / dt per stage: rk3 and rk3s4 are weighted by wght2, rk1 and rk2 use the full dt",
                  "; ".join("%s %s" % (k, ", ".join("%.4g" % x for x in v)) for k, v in f.items()))


def check_C3_stiff_limit():
    z = np.array([1e8])
    rw, rf = amplification("rk3", z)[0], amplification("rk3", z, weighted=False)[0]
    sw, sf = amplification("rk3s4", z)[0], amplification("rk3s4", z, weighted=False)[0]
    ok = (abs(rw - 1. / 12.) < 1e-6 and abs(rf - 1. / 3.) < 1e-6
          and abs(sw) < 1e-6 and abs(sf - 17. / 48.) < 1e-6)
    return report("C3", ok, "on a stiff mode the weighted correction damps rk3 to 1/12 per step and rk3s4 to 0; "
                  "with the full dt they damp to 1/3 and 17/48",
                  "R(z = 1e8): rk3 weighted %.6f, full dt %.6f; rk3s4 weighted %.6f, full dt %.6f"
                  % (rw, rf, sw, sf))


def check_C4_stable_and_first_order_on_the_decay():
    z = np.logspace(-3, 8, 400)
    worst = max(np.abs(amplification(k, z)).max() for k in WEIGHTS)
    errs = []
    for n in (40, 80, 160):
        z1 = 2. / n
        errs.append(abs(amplification("rk3", np.array([z1]))[0]**n - np.exp(-2.)))
    order = np.log2(errs[1] / errs[2])
    ok = worst <= 1. and abs(order - 1.) < 0.1
    return report("C4", ok, "every integrator with the correction keeps |R(z)| <= 1 for z in [1e-3, 1e8]; "
                  "a fully implicit decay converges at first order (the correction is a linearised backward Euler)",
                  "max |R| = %.4f; rk3 errors at 40, 80, 160 steps %s, fitted order %.2f"
                  % (worst, ", ".join("%.2e" % e for e in errs), order))


def check_C5_stiff_limit_steps():
    """Derivation steps 1-4 of _vic-stage.qmd in exact arithmetic: w2 delta as Z -> oo, then the stage products."""
    Z, r, Y = sp.symbols("Z r Y", positive=True)       # r = dt_c / dt
    delta = -Z * Y / (1 + r * Z)                        # step 1: (1 + dt_c G)^-1 (-Z Y)
    R = sp.Rational
    tableaux = {"rk3": [(0, 1, 1), (R(3, 4), R(1, 4), R(1, 4)), (R(1, 3), R(2, 3), R(2, 3))],
                "rk3s4": [(R(1, 2), R(1, 2), R(1, 2)), (0, 1, R(1, 2)), (R(2, 3), R(1, 3), R(1, 6)), (0, 1, R(1, 2))]}
    got = {}
    for kind, stages in tableaux.items():
        for name, ratio in (("weighted", lambda w2: w2), ("full dt", lambda w2: 1)):
            y, ys = sp.Integer(1), []
            for w0, w1, w2 in stages:
                lim = sp.limit(w2 * delta.subs(r, ratio(w2)), Z, sp.oo)   # -Y weighted, -w2 Y with the full dt
                y = sp.nsimplify(w0 + w1 * y + lim.subs(Y, y))
                ys.append(y)
            got[kind, name] = ys
    stated = {("rk3", "weighted"): [0, R(3, 4), R(1, 12)], ("rk3", "full dt"): [0, R(3, 4), R(1, 3)],
              ("rk3s4", "weighted"): [0, 0, R(2, 3), 0],
              ("rk3s4", "full dt"): [R(1, 2), R(1, 4), R(17, 24), R(17, 48)]}
    ok = got == stated
    return report("C5", ok, "as Z -> oo the stage takes w0 Y^n + (w1 - 1) Y weighted and w0 Y^n + (w1 - w2) Y with the "
                  "full dt; the rk3 stages then give 1/12 and 1/3, the rk3s4 stages 0 and 17/48, exactly",
                  "; ".join("%s %s %s" % (k, n, ", ".join(map(str, v))) for (k, n), v in got.items()))


def main():
    checks = [check_C1_weight_inside_equals_weight_outside, check_C2_three_or_more_stages_are_weighted,
              check_C3_stiff_limit, check_C4_stable_and_first_order_on_the_decay,
              check_C5_stiff_limit_steps]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
