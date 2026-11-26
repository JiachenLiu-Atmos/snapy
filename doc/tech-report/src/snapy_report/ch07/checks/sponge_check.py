"""Executable check of the sponge layers (chapter 9, _sponge.qmd).

What is checked: TopSpongeLyrImpl::forward (src/forcing/top_sponge_lyr.cpp:47-76),
BotSpongeLyrImpl::forward (src/forcing/bot_sponge_lyr.cpp:47-76) and coord_vec_lower_impl
(src/coord/coord_utils_impl.h:19-24) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported
line for line. Claims C1-C4. Also writes data/sponge_profile.csv for fig_sponge_profile.py.

Run: python3 sponge_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the chapter directory


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def eta_top(x1f, width):  # top_sponge_lyr.cpp:58-61, lower face of every cell
    return np.clip((width - (x1f[-1] - x1f[:-1])) / width, 0., 1.)


def eta_bot(x1f, width):  # bot_sponge_lyr.cpp:58-61
    return np.clip((width - (x1f[:-1] - x1f[0])) / width, 0., 1.)


def lower(v2, v3, cth):  # coord_utils_impl.h:19-24
    return v2 + v3 * cth, v3 + v2 * cth


def check_C1_profile_ends():
    x1f = np.linspace(0., 10., 21)  # 20 cells of 0.5
    top, bot = np.sin(np.pi / 2 * eta_top(x1f, 2.))**2, np.sin(np.pi / 2 * eta_bot(x1f, 2.))**2
    ok = bot[0] == 1. and abs(top[-1] - np.sin(np.pi / 2 * 0.75)**2) < 1e-15
    np.savetxt(os.path.join(HERE, "data", "sponge_profile.csv"),
               np.column_stack([x1f[:-1], x1f[1:], top, bot]), delimiter=",", fmt="%.10e",
               header="x1f_lower,x1f_upper,top_scale,bot_scale (20 cells of 0.5 on [0,10], width 2)")
    return report("C1", ok, "the sin^2 profile is read at each cell's lower face",
                  "bottom cell scale %.3f; top cell scale %.4f = sin^2(pi/2 (1 - dz/width)), dz/width = 0.25"
                  % (bot[0], top[-1]))


def check_C2_drag_heats():
    rho, v, s, tau, dt = 1.2, np.array([3., -4., 1.]), 0.8, 100., 5.
    a = dt * s / tau
    m_new = rho * v - a * rho * v  # top_sponge_lyr.cpp:71-73, orthogonal grid
    dKE = 0.5 * (m_new @ m_new) / rho - 0.5 * rho * (v @ v)
    dIE = -dKE  # E is not touched
    want = 0.5 * rho * (v @ v) * (1. - (1. - a)**2)
    return report("C2", abs(dIE - want) < 1e-12 and dIE > 0.,
                  "E is untouched, so the kinetic energy removed becomes internal energy",
                  "dIE = %.6f J m^-3 = rho |v|^2 (1 - (1 - a)^2)/2, a = dt s/tau = %.3f" % (dIE, a))


def check_C3_lowered_drag_brakes():
    rng = np.random.default_rng(12)
    worst_cross, worst_power = 0., -np.inf
    for _ in range(1000):
        c = -0.5 * rng.random()  # cos(theta) on a cubed-sphere panel, -1/2 at a corner
        v2, v3 = rng.standard_normal(2)
        f2, f3 = lower(-v2, -v3, c)  # force ~ -v, lowered
        m2, m3 = lower(v2, v3, c)  # covariant momentum per unit rho
        worst_cross = max(worst_cross, abs(f2 * m3 - f3 * m2))
        worst_power = max(worst_power, v2 * f2 + v3 * f3)  # contravariant v times covariant f
    ok = worst_cross < 1e-14 and worst_power < 0.
    return report("C3", ok, "the lowered drag is antiparallel to the covariant momentum and does negative work",
                  "max |f x m| = %.1e, max v.f = %.3e < 0 over 1000 draws" % (worst_cross, worst_power))


def check_C4_unlowered_drag_rotates():
    c, v2, v3 = -0.5, 1.0, 0.3
    m2, m3 = lower(v2, v3, c)
    f2, f3 = -v2, -v3  # the form before the fix: contravariant force added to covariant du
    cross = f2 * m3 - f3 * m2
    want = c * (v3**2 - v2**2)
    return report("C4", abs(cross - want) < 1e-15 and cross != 0.,
                  "unlowered drag is not antiparallel: cross = cos(theta) (v3^2 - v2^2)",
                  "cross = %.4f at a corner (cos = -1/2, v = (1, 0.3))" % cross)


def main():
    checks = [check_C1_profile_ends, check_C2_drag_heats, check_C3_lowered_drag_brakes,
              check_C4_unlowered_drag_rotates]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
