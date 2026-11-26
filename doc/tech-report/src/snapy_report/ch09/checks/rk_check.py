"""Executable check of the explicit Runge-Kutta stage update (chapter 7, _rk.qmd).

What is checked: the stage update u <- w0 u0 + w1 u + w2 dU of MeshBlockImpl::advance_local
(src/mesh/meshblock.cpp:755 at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5), with the stage weights of
IntegratorImpl (src/integrator/integrator.cpp:35-78 at pyharp@4721715855e937c1e8b218e964c0655f46e56e29),
ported line for line, and dU = dt L(u) the full-step increment. Claims C1-C7; C5-C7 are exact (sympy).

Run: python3 rk_check.py   (numpy and sympy; exits with the number of failed claims)
"""
import math
import sys

import numpy as np
import sympy as sp

# integrator.cpp:35-78, (wght0, wght1, wght2) per stage
WEIGHTS = {
    "rk1": [(0., 1., 1.)],
    "rk2": [(0., 1., 1.), (.5, .5, .5)],
    "rk3": [(0., 1., 1.), (3. / 4., 1. / 4., 1. / 4.), (1. / 3., 2. / 3., 2. / 3.)],
    "rk3s4": [(.5, .5, .5), (0., 1., .5), (2. / 3., 1. / 3., 1. / 6.), (0., 1., .5)],
}


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def step(kind, u0, rhs, dt):
    """one step: every stage forms w0 u0 + w1 u + w2 dt L(u) (meshblock.cpp:755, integrator.cpp:119-120)"""
    u = u0
    for w0, w1, w2 in WEIGHTS[kind]:
        u = w0 * u0 + w1 * u + w2 * dt * rhs(u)
    return u


def check_C1_stability_polynomials():
    z = np.array([0.1, -0.7, 0.3 + 0.9j, -1.5 + 0.2j])
    want = {"rk1": 1 + z,
            "rk2": 1 + z + z**2 / 2,
            "rk3": 1 + z + z**2 / 2 + z**3 / 6,
            "rk3s4": 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 48}
    worst = max(np.abs(step(k, 1. + 0j, lambda u: u * z, 1.) - want[k]).max() for k in WEIGHTS)
    return report("C1", worst < 1e-14, "amplification factors on u' = lambda u",
                  "rk1 1+z; rk2 to z^2/2; rk3 to z^3/6; rk3s4 to z^3/6 + z^4/48; max error %.1e" % worst)


def check_C2_order_on_a_nonlinear_ode():
    # u' = -u^2 + sin t, autonomous form with t as a second component
    def rhs(y):
        return np.array([-y[0] ** 2 + math.sin(y[1]), 1.])
    T, y0 = 1., np.array([1., 0.])
    def solve(kind, n):
        y = y0.copy()
        for _ in range(n):
            y = step(kind, y, rhs, T / n)
        return y[0]
    ref = solve("rk3s4", 20000)
    out, ok = [], True
    want = {"rk1": 1, "rk2": 2, "rk3": 3, "rk3s4": 3}
    for kind in WEIGHTS:
        errs = [abs(solve(kind, n) - ref) for n in (40, 80, 160)]
        order = np.log2(errs[1] / errs[2])
        ok = ok and abs(order - want[kind]) < 0.1
        out.append("%s %.2f" % (kind, order))
    return report("C2", ok, "fitted order on a nonlinear ODE, n = 40, 80, 160 steps", ", ".join(out))


def check_C3_convex_stages():
    out, ok = [], True
    want = {"rk1": 1., "rk2": 1., "rk3": 1., "rk3s4": 2.}
    for kind, stages in WEIGHTS.items():
        convex = all(abs(w0 + w1 - 1.) < 1e-15 and w0 >= 0 and w1 > 0 for w0, w1, _ in stages)
        # at stage 0 the current state is u0, so the forward-Euler step carries w0 + w1
        c = min(((w0 + w1) if s == 0 else w1) / w2 for s, (w0, w1, w2) in enumerate(stages))
        ok = ok and convex and abs(c - want[kind]) < 1e-15
        out.append("%s C=%g" % (kind, c))
    return report("C3", ok, "each stage is a convex combination of u0 and a forward-Euler step; SSP coefficient C = min over stages of (Euler weight)/w2",
                  ", ".join(out))


def check_C4_positivity_up_to_the_ssp_limit():
    # u' = -u: forward Euler keeps u >= 0 for dt <= 1; the scheme keeps it for dt <= C
    out, ok = [], True
    for kind, c in (("rk3", 1.), ("rk3s4", 2.)):
        pos = all(step(kind, 1., lambda u: -u, dt) >= 0 for dt in np.linspace(0.01, c, 200))
        ok = ok and pos
        out.append("%s nonnegative for dt <= %g" % (kind, c))
    return report("C4", ok, "the stage values stay nonnegative up to dt = C", "; ".join(out))


# integrator.cpp:35-78 in exact rationals, for the claims written as algebra (C5-C7)
R = sp.Rational
EXACT = {
    "rk1": [(0, 1, 1)],
    "rk2": [(0, 1, 1), (R(1, 2), R(1, 2), R(1, 2))],
    "rk3": [(0, 1, 1), (R(3, 4), R(1, 4), R(1, 4)), (R(1, 3), R(2, 3), R(2, 3))],
    "rk3s4": [(R(1, 2), R(1, 2), R(1, 2)), (0, 1, R(1, 2)), (R(2, 3), R(1, 3), R(1, 6)), (0, 1, R(1, 2))],
}


def stage_values(kind, z):
    """On Y' = G Y a stage maps Y -> w0 Y^n + (w1 + w2 Z) Y, from Y^(0) = Y^n = 1 (Derivation steps 1-4)."""
    out, y = [], sp.Integer(1)
    for w0, w1, w2 in EXACT[kind]:
        y = sp.expand(w0 + (w1 + w2 * z) * y)
        out.append(y)
    return out


def fe_fractions(kind):
    """c^(0)_FE = (w0 + w1) / w2 at the first stage (U^(0) = U^n), c^(s)_FE = w1 / w2 after it."""
    return [((w0 + w1) if s == 0 else w1) / sp.Integer(1) / w2 for s, (w0, w1, w2) in enumerate(EXACT[kind])]


def check_C5_stage_products():
    Z = sp.Symbol("Z")
    h = 1 + Z / 2
    stated = {  # the stage values written in Derivation steps 1-4 of _rk.qmd
        "rk1": [1 + Z],
        "rk2": [1 + Z, 1 + Z + Z**2 / 2],
        "rk3": [1 + Z, 1 + Z / 2 + Z**2 / 4, 1 + Z + Z**2 / 2 + Z**3 / 6],
        "rk3s4": [h, h**2, R(2, 3) + h**3 / 3, 1 + Z + Z**2 / 2 + Z**3 / 6 + Z**4 / 48],
    }
    got = {k: stage_values(k, Z) for k in stated}
    ok = all(len(got[k]) == len(v) and all(sp.expand(a - b) == 0 for a, b in zip(got[k], v))
             for k, v in stated.items())
    order = {k: next(n for n in range(1, 6) if got[k][-1].coeff(Z, n) != R(1, sp.factorial(n))) - 1 for k in got}
    ok = ok and order == {"rk1": 1, "rk2": 2, "rk3": 3, "rk3s4": 3}
    return report("C5", ok, "the stage values of Derivation steps 1-4 are the exact stage product; the first power "
                  "of Z whose coefficient differs from e^Z gives the order",
                  "rk3s4 last stage %s; orders %s" % (sp.expand(got["rk3s4"][-1]),
                                                     ", ".join("%s %d" % kv for kv in order.items())))


def check_C6_every_stage_fraction():
    c_fe = {k: fe_fractions(k) for k in EXACT}
    stated = {"rk1": 1, "rk2": 1, "rk3": 1, "rk3s4": 2}   # "every c^(s)_FE is 1 ... every c^(s)_FE is 2 for rk3s4"
    ok = all(all(x == stated[k] for x in c_fe[k]) for k in c_fe)
    return report("C6", ok, "every stage fraction c^(s)_FE is 1 for rk1, rk2, rk3 and 2 for rk3s4, so c_SSP = min_s c^(s)_FE = 1, 1, 1, 2",
                  "; ".join("%s %s" % (k, ", ".join(str(x) for x in v)) for k, v in c_fe.items()))


STEP_RATIO, PER_EVALUATION = 2, R(3, 2)   # Limits 3 of _rk.qmd


def check_C7_rk3s4_against_rk3():
    c = {k: min(fe_fractions(k)) for k in ("rk3", "rk3s4")}
    step = c["rk3s4"] / c["rk3"]
    per_eval = (c["rk3s4"] / len(EXACT["rk3s4"])) / (c["rk3"] / len(EXACT["rk3"]))
    ok = step == STEP_RATIO and per_eval == PER_EVALUATION
    return report("C7", ok, "at the same SSP guarantee rk3s4 takes a step c_SSP = 2 times that of rk3; per operator "
                  "evaluation it advances 2/4 against 1/3, 3/2 times as far",
                  "step ratio %s, per evaluation %s" % (step, per_eval))


def main():
    checks = [check_C1_stability_polynomials, check_C2_order_on_a_nonlinear_ode, check_C3_convex_stages,
              check_C4_positivity_up_to_the_ssp_limit,
              check_C5_stage_products, check_C6_every_stage_fraction, check_C7_rk3s4_against_rk3]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
