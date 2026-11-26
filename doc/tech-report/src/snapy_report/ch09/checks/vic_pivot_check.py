"""Executable check of the LU pivot tolerance (chapter 7, _vic-pivot.qmd).

What is checked: ludcmp's rejection rule |p_j| / s_j <= 8 N eps (src/math/ludcmp.h:29-94 at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5; docs/derivations/290-lu-pivot-tolerance.md), ported in
vic_port.py for float32 and float64. Claims C1-C5.

Run: python3 vic_pivot_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vic_port as vp  # noqa: E402


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def tol(dtype, N):
    return dtype(8 * N) * np.finfo(dtype).eps


def near_singular(eps, dtype):
    """Rows (1,0,0), (0,1,0), (0.5,0.8,eps): the last pivot is eps against the row's original magnitude 0.8."""
    return np.array([[1., 0., 0.], [0., 1., 0.], [.5, .8, eps]], dtype=dtype)


def check_C1_tolerance_table():
    want = {(np.float32, 3): 2.86102294921875e-6, (np.float32, 5): 4.76837158203125e-6,
            (np.float64, 3): 5.329070518200751e-15, (np.float64, 5): 8.881784197001252e-15}
    got = {k: float(tol(*k)) for k in want}
    ok = all(got[k] == want[k] for k in want)
    return report("C1", ok, "tau_N = 8 N eps reproduces the table of the derivation note",
                  "float N=3 %.15g, N=5 %.15g; double N=3 %.16g, N=5 %.16g" % tuple(got[k] for k in want))


def check_C2_threshold():
    rows = []
    ok = True
    for dt in (np.float32, np.float64):
        t = float(tol(dt, 3))
        below = vp.ludcmp(near_singular(dt(0.8 * t * 0.99), dt), dt)[0]
        above = vp.ludcmp(near_singular(dt(0.8 * t * 1.01), dt), dt)[0]
        ok &= below == 0 and above != 0
        rows.append("%s: ratio 0.99 tau -> %d, 1.01 tau -> %d" % (np.dtype(dt).name, below, above))
    return report("C2", ok, "a pivot ratio just under tau_N is refused (0) and just over it accepted (+-1)", "; ".join(rows))


def check_C3_row_scaling_does_not_change_the_decision():
    rng = np.random.default_rng(9)
    flips2 = flips10 = n = 0
    for _ in range(400):
        A = rng.normal(size=(5, 5))
        if rng.random() < 0.5:
            A[4] = A[:4].T @ rng.normal(size=4) + rng.normal(size=5) * 10.**rng.uniform(-17, -12)
        d0 = vp.ludcmp(A)[0] != 0
        S2 = np.diag(2.**rng.integers(-100, 101, 5))
        S10 = np.diag(10.**rng.uniform(-30, 30, 5))
        flips2 += d0 != (vp.ludcmp(S2 @ A)[0] != 0)
        flips10 += d0 != (vp.ludcmp(S10 @ A)[0] != 0)
        n += not d0
    return report("C3", flips2 == 0, "scaling rows by powers of two (exact in floating point) never changes accept or "
                  "refuse: the ratio is per row, so the units of a VIC row drop out",
                  "400 matrices, %d refused; decisions changed: %d with factors 2^-100..2^100, %d with decimal "
                  "factors 1e-30..1e30 (rounding of the scaled entries at the margin)" % (n, flips2, flips10))


def check_C4_float32_is_stricter_and_nonfinite_is_refused():
    A = near_singular(1e-6, np.float64)
    d64 = vp.ludcmp(A, np.float64)[0]
    d32 = vp.ludcmp(A, np.float32)[0]
    nan = np.eye(3)
    nan[1, 2] = np.nan
    zero = np.eye(3)
    zero[2] = 0.
    dn, dz = vp.ludcmp(nan)[0], vp.ludcmp(zero)[0]
    ok = d64 != 0 and d32 == 0 and dn == 0 and dz == 0
    return report("C4", ok, "a pivot ratio of 1.25e-6 passes in double and is refused in float32; a NaN entry or a zero "
                  "row is refused", "double %d, float32 %d, NaN %d, zero row %d" % (d64, d32, dn, dz))


def longest(N):
    return N - 1   # step 1 of _vic-pivot.qmd: the longest Crout inner product has N - 1 terms


def check_C5_crout_inner_products_and_the_bound():
    """ludcmp.h:50-61: entry (i, j) above the diagonal sums k < i, on and below it k < j. With eps_m/2 per operation
    the computed factors satisfy L U = A + dA, |dA| <= zeta_N |L| |U|, zeta_N = N (eps_m/2) / (1 - N eps_m/2)."""
    rows, ok = [], True
    for N in (3, 5):
        lengths = []
        for j in range(N):
            lengths += [i for i in range(j)]          # for (k = 0; k < i; k++)
            lengths += [j for _ in range(j, N)]       # for (k = 0; k < j; k++)
        n = max(lengths)
        ok = ok and n == longest(N)
        for dtype in (np.float32, np.float64):
            u = float(np.finfo(dtype).eps) / 2
            zeta = N * u / (1 - N * u)
            factor = float(tol(dtype, N)) / zeta
            ok = ok and 15.99 < factor <= 16.
            rows.append("N=%d %s: longest %d terms, tolerance / zeta_N = %.6f" % (N, np.dtype(dtype).name, n, factor))
    return report("C5", ok, "the longest inner product of the Crout loops has N - 1 terms, and the tolerance 8 N eps_m "
                  "is 16 (1 - N eps_m / 2) times the backward-error bound zeta_N", "; ".join(rows))


def main():
    checks = [check_C1_tolerance_table, check_C2_threshold, check_C3_row_scaling_does_not_change_the_decision,
              check_C4_float32_is_stricter_and_nonfinite_is_refused,
              check_C5_crout_inner_products_and_the_bound]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
