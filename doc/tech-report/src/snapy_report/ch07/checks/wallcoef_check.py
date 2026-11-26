"""Executable check of the diffusion face coefficient at walls (chapter 9, _wallcoef.qmd).

What is checked: extrapolate_to_wall and face_coefficient of src/forcing/diffusion.cpp
:110-159 at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported line for line, against the
two-cell average that reads the mirror ghost; and the arithmetic of the immersed-solid
placeholder state of sources/canoe__diffusion_ghost_TECH_REPORT.md section 3. Claims C1-C5.

Run: python3 wallcoef_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def extrapolate_to_wall(value, width, upper):  # diffusion.cpp:114-136, one column
    near, nxt = (-1, -2) if upper else (0, 1)
    t = width[near] / (width[near] + width[nxt])
    wall = (1. + t) * value[near] - t * value[nxt]
    return wall if wall > 0. else value[near]


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def centres(width, x0=0.):
    faces = x0 + np.concatenate([[0.], np.cumsum(width)])
    return 0.5 * (faces[1:] + faces[:-1]), faces


def check_C1_exact_for_linear_on_stretched_cells():
    rng = np.random.default_rng(3)
    worst = 0.
    for _ in range(100):
        width = 0.5 + rng.random(8)
        xc, xf = centres(width)
        a, b = 2. + rng.random(), 0.1 * (rng.random() - 0.5)  # q > 0 on the column
        q = a + b * xc
        lo = extrapolate_to_wall(q, width, False)
        hi = extrapolate_to_wall(q, width, True)
        worst = max(worst, abs(lo - (a + b * xf[0])), abs(hi - (a + b * xf[-1])))
    return report("C1", worst < 1e-13, "q_f = (1+t) q_a - t q_b is exact for a linear q",
                  "max error %.1e over 100 random stretched columns" % worst)


def check_C2_mirror_first_order_extrapolation_second():
    H = 1.
    rows, e_m, e_x = [], [], []
    for n in (8, 16, 32, 64):
        h = H / n
        width = np.full(n, h)
        xc, xf = centres(width)
        rho = np.exp(-xc / H)
        mirror = 0.5 * (rho[0] + rho[0])  # ghost = mirror of cell 0
        ratio = mirror / np.exp(-xf[0] / H)
        e_m.append(abs(ratio - 1.))
        e_x.append(abs(extrapolate_to_wall(rho, width, False) - 1.))
        rows.append("h/H=1/%d mirror/exact=%.10f (exp(-h/2H)=%.10f)" % (n, ratio, np.exp(-h / (2 * H))))
    o_m = np.log2(e_m[-2] / e_m[-1])
    o_x = np.log2(e_x[-2] / e_x[-1])
    ok = abs(ratio - np.exp(-h / (2 * H))) < 1e-14 and abs(o_m - 1.) < 0.05 and abs(o_x - 2.) < 0.05
    return report("C2", ok, "mirror average is O(h), wall extrapolation O(h^2), on rho = exp(-x1/H)",
                  "%s; orders %.3f (mirror), %.3f (extrapolated), h/H 1/32 -> 1/64"
                  % (rows[-1], o_m, o_x))


def check_C3_positivity_fallback():
    width = np.ones(4)
    q = np.array([0.1, 1.0, 2.0, 3.0])  # 1.5*0.1 - 0.5*1.0 < 0
    got = extrapolate_to_wall(q, width, False)
    return report("C3", got == q[0], "a non-positive extrapolation falls back to the wall cell",
                  "1.5*0.1 - 0.5*1.0 = %.2f -> %.2f" % (1.5 * 0.1 - 0.5 * 1.0, got))


def check_C4_reflecting_wall_parities():
    h = 0.25
    T_a, v1_a = 290., 0.4
    dTdn = (T_a - T_a) / h  # ghost T mirrored even
    dv1dn = (v1_a - (-v1_a)) / h  # ghost v1 mirrored odd
    ok = dTdn == 0. and abs(dv1dn - 2. * v1_a / h) < 1e-15
    return report("C4", ok, "at a reflecting wall dT/dn = 0 and dv1/dn = 2 v1/h",
                  "dT/dn = %.1f, dv1/dn = %.3f = 2 v1/h" % (dTdn, dv1dn))


def check_C5_solid_placeholder_factor():
    rho_s, p_s = 1.e3, 1.e9  # placeholder defaults (source section 3)
    rho_f, p_f = 1., 1.e5
    face_factor = 0.5 * (rho_s + rho_f) / rho_f
    T_ratio = (p_s / rho_s) / (p_f / rho_f)
    ok = face_factor == 500.5 and T_ratio == 10.
    return report("C5", ok, "a fluid face beside a solid placeholder",
                  "rho cv at the face = %.1f x the fluid value; T_solid/T_fluid = %.1f"
                  % (face_factor, T_ratio))


def main():
    checks = [check_C1_exact_for_linear_on_stretched_cells,
              check_C2_mirror_first_order_extrapolation_second, check_C3_positivity_fallback,
              check_C4_reflecting_wall_parities, check_C5_solid_placeholder_factor]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
