"""Executable check of the operator split at the step boundary (chapter 7, _split.qmd).

What is checked: the Lie (sequential) split of MeshBlockImpl::advance_local and the example driver
(src/mesh/meshblock.cpp:795-841, examples/run_hydro.cpp:166-191 at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5): the dynamics advance a whole step, then the saturation
adjustment, then the kinetics, each on the result of the one before. Two non-commuting linear operators stand in
for them. Claims C1-C4; C4 is exact (sympy).

Run: python3 split_check.py   (numpy and sympy; exits with the number of failed claims)
"""
import sys

import numpy as np
import sympy as sp


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def expm(M):
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


A = np.array([[-1.0, 0.4], [0.0, -0.3]])   # stands in for the dynamics
B = np.array([[-0.2, 0.0], [0.7, -2.0]])   # stands in for the step-boundary source
U0 = np.array([1.0, 0.5])
T = 1.0


def split(n, strang=False):
    dt = T / n
    u = U0.copy()
    if strang:
        ea, eb2 = expm(A * dt), expm(B * dt / 2)
        for _ in range(n):
            u = eb2 @ (ea @ (eb2 @ u))
    else:
        ea, eb = expm(A * dt), expm(B * dt)
        for _ in range(n):
            u = eb @ (ea @ u)                  # dynamics first, then the boundary source
    return u


def check_C1_lie_split_is_first_order():
    exact = expm((A + B) * T) @ U0
    errs = [np.abs(split(n) - exact).max() for n in (20, 40, 80)]
    order = np.log2(errs[1] / errs[2])
    return report("C1", abs(order - 1.) < 0.05, "the sequential split converges at first order in dt",
                  "errors %s at 20, 40, 80 steps; fitted order %.3f" % (", ".join("%.2e" % e for e in errs), order))


def check_C2_strang_for_contrast():
    exact = expm((A + B) * T) @ U0
    errs = [np.abs(split(n, True) - exact).max() for n in (20, 40, 80)]
    order = np.log2(errs[1] / errs[2])
    return report("C2", abs(order - 2.) < 0.05, "a symmetric (Strang) split would be second order; snapy does not use it",
                  "fitted order %.3f" % order)


def check_C3_commuting_operators_split_exactly():
    a, b = np.diag([-1.0, -0.3]), np.diag([-0.2, -2.0])
    err = np.abs(expm(b) @ expm(a) @ U0 - expm(a + b) @ U0).max()
    return report("C3", err < 1e-14, "the splitting error is the commutator: commuting operators split exactly",
                  "error %.1e" % err)


def check_C4_commutator_is_the_local_error():
    """Derivation steps 1-2 of _split.qmd on generic 2 x 2 generators, exponentials to third order."""
    h = sp.Symbol("h")
    Gd = sp.Matrix(2, 2, sp.symbols("a0:4"))   # dynamics, applied first
    Gk = sp.Matrix(2, 2, sp.symbols("b0:4"))   # kinetics, applied last

    def exp3(X):
        return sp.eye(2) + X + X * X / 2 + X * X * X / 6

    diff = (exp3(h * Gk) * exp3(h * Gd) - exp3(h * (Gk + Gd))).applyfunc(sp.expand)
    low = [diff.applyfunc(lambda e, n=n: e.coeff(h, n)) for n in (0, 1)]
    c2 = diff.applyfunc(lambda e: e.coeff(h, 2))
    stated = (Gk * Gd - Gd * Gk) / 2          # step 2: (h^2 / 2) [G_kin, G_dyn]
    ok = all(m == sp.zeros(2) for m in low) and (c2 - stated).applyfunc(sp.expand) == sp.zeros(2) \
        and c2 != sp.zeros(2)
    return report("C4", ok, "e^{h G_kin} e^{h G_dyn} - e^{h (G_kin + G_dyn)} = (h^2/2) [G_kin, G_dyn] + O(h^3) on generic "
                  "2 x 2 generators", "orders h^0 and h^1 vanish; the h^2 term equals the commutator over 2 entry by entry")


def main():
    checks = [check_C1_lie_split_is_first_order, check_C2_strang_for_contrast,
              check_C3_commuting_operators_split_exactly,
              check_C4_commutator_is_the_local_error]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
