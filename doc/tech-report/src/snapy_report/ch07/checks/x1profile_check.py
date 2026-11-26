"""Executable check of the x1 coefficient profiles and the mean-of-products face coefficient
(chapter 9, _x1profile.qmd).

What is checked: face_scaled_coefficient (:168-194) and profile_from_table (:216-225) of
src/forcing/diffusion.cpp at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported line for
line, and the closed forms of docs/derivations/diffusion-face-coefficient.md at the same sha.
Claims C1-C5. Writes data/x1profile_tendency.csv, read by fig_x1profile_tendency.py.

Run: python3 x1profile_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

NG = 2
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the chapter directory


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def column(nx):
    dx = 1. / nx
    return (np.arange(nx + 2 * NG) - NG + 0.5) * dx, dx


def rho_of(x):
    return (1.25 - x)**1.5


def face_coeff(s, q, form):
    """interior x1 faces, walls extrapolated (uniform mesh: t = 1/2), as the code does for x1"""
    a, b = slice(NG - 1, -NG), slice(NG, len(q) - NG + 1)
    if form == "MP":  # face_coefficient(value * scale, ...), diffusion.cpp:172-175
        c = 0.5 * (s[a] * q[a] + s[b] * q[b])
    else:  # the product of means, the form before docs/derivations/diffusion-face-coefficient.md
        c = 0.5 * (s[a] + s[b]) * 0.5 * (q[a] + q[b])
    p = s * q
    c[0] = 1.5 * p[NG] - 0.5 * p[NG + 1]
    c[-1] = 1.5 * p[-NG - 1] - 0.5 * p[-NG - 2]
    return c


def tendency(s, q, T, dx, form):
    a, b = slice(NG - 1, -NG), slice(NG, len(T) - NG + 1)
    F = -face_coeff(s, q, form) * (T[b] - T[a]) / dx
    return -(F[1:] - F[:-1]) / dx


def profile_from_table(table, xv):  # diffusion.cpp:216-225
    x, s = table
    i = np.clip(np.searchsorted(x, xv, side="right") - 1, 0, len(x) - 2)
    f = np.clip((xv - x[i]) / (x[i + 1] - x[i]), 0., 1.)
    return (1. - f) * s[i] + f * s[i + 1]


def check_C1_covariance_identity():
    rng = np.random.default_rng(5)
    sa, sb, qa, qb = rng.random(4) + 0.5
    pm = 0.5 * (sa + sb) * 0.5 * (qa + qb)
    mp = 0.5 * (sa * qa + sb * qb)
    err = abs(pm - (mp - 0.25 * (sb - sa) * (qb - qa)))
    return report("C1", err < 1e-15, "product of means = mean of products - (1/4) ds dq",
                  "identity error %.1e" % err)


def check_C2_constant_dynamic_coefficient():
    rows, prev, ok = [], None, True
    data = None
    for nx in (16, 32, 64, 128):
        x, dx = column(nx)
        q = rho_of(x)
        s, T = 1. / q, 300. + 50. * x
        mp = tendency(s, q, T, dx, "MP")
        pm = tendency(s, q, T, dx, "PM")
        ok = ok and np.abs(mp).max() < 1e-12 * 50. / dx  # same bound as the docs script
        cur = (np.abs(pm).max(), np.abs(pm[1:-1]).max())
        if prev is not None:
            rows.append("n1=%d wall %.2f inner %.2f" % (nx, np.log2(prev[0] / cur[0]),
                                                       np.log2(prev[1] / cur[1])))
        prev = cur
        if nx == 32:
            data = np.column_stack([x[NG:-NG], mp, pm])
    np.savetxt(os.path.join(HERE, "data", "x1profile_tendency.csv"), data, delimiter=",",
               header="x1,tendency_mean_of_products,tendency_product_of_means "
                      "(c = s q = 1, rho = (1.25-x1)^1.5, T = 300 + 50 x1, n1 = 32)",
               fmt="%.10e")
    o_wall = float(rows[-1].split()[2])
    o_in = float(rows[-1].split()[4])
    ok = ok and o_wall > 0.85 and o_in > 1.7
    return report("C2", ok, "constant s q: mean of products gives no tendency, product of means does",
                  "MP max|du| < 1e-12 F/dx at n1 16..128; PM orders %s" % "; ".join(rows))


def check_C3_smooth_profile_second_order():
    errs = {"MP": [], "PM": []}
    for nx in (32, 64, 128, 256):
        x, dx = column(nx)
        q = rho_of(x)
        s = 1. + 0.5 * np.cos(3. * x)
        T = 300. + 50. * np.sin(2. * x)
        xf = (np.arange(nx + 1)) * dx
        exact = -(1. + 0.5 * np.cos(3. * xf)) * rho_of(xf) * 100. * np.cos(2. * xf)
        a, b = slice(NG - 1, -NG), slice(NG, len(T) - NG + 1)
        for form in errs:
            F = -face_coeff(s, q, form) * (T[b] - T[a]) / dx
            errs[form].append(np.abs(F - exact).max() / np.abs(exact).max())
    orders = {k: np.log2(v[-2] / v[-1]) for k, v in errs.items()}
    ok = min(orders.values()) > 1.85
    return report("C3", ok, "smooth profile: both forms give a second-order face flux",
                  "orders n1 128 -> 256: mean of products %.2f, product of means %.2f"
                  % (orders["MP"], orders["PM"]))


def check_C4_table_interpolation():
    table = (np.array([0., 1., 3.]), np.array([1., 2., 0.5]))
    xv = np.array([-1., 0., 0.5, 1., 2., 3., 4.])
    got = profile_from_table(table, xv)
    want = np.array([1., 1., 1.5, 2., 1.25, 0.5, 0.5])
    err = np.abs(got - want).max()
    return report("C4", err == 0., "table profile: linear between knots, held beyond, exact on knots",
                  "max error %.1e at x1 = %s" % (err, xv.tolist()))


def check_C5_x2_x3_faces_round_off():
    rng = np.random.default_rng(7)
    q = rng.random((2, 1000)) + 0.5
    s = rng.random(1000) + 0.5  # both cells of an x2/x3 face share one x1 position
    a = 0.5 * (q[0] + q[1]) * s  # diffusion.cpp:178-179
    b = 0.5 * (q[0] * s + q[1] * s)
    rel = np.abs(a - b).max() / np.abs(a).max()
    return report("C5", 0. < rel < 1e-15, "x2/x3 faces: the two forms agree to round-off only",
                  "max relative difference %.1e (1000 random faces), not zero" % rel)


def main():
    checks = [check_C1_covariance_identity, check_C2_constant_dynamic_coefficient,
              check_C3_smooth_profile_second_order, check_C4_table_interpolation,
              check_C5_x2_x3_faces_round_off]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
