"""Executable check of the time-step bound (chapter 7, _cfl.qmd).

What is checked: HydroImpl::max_time_step (src/hydro/hydro.cpp:311-398) and the Courant factor and redo
halving of MeshBlockImpl::max_time_step (src/mesh/meshblock.cpp:510-529) at
snapy@e894700ff7aee30b52882e5202b16461413780b0, ported line for line on one column of cells (x1 is the
last axis, x2 the middle one). Claims C1-C5.

Run: python3 cfl_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def hydro_max_time_step(v1, v2, cs, dx1, dx2, cfl, scheme=None, adv_cfl=1., shear_cfl=0.):
    """hydro.cpp:326-398 on a (nc2, nc1) slab; scheme None = no implicit options (the else branch)"""
    nc2, nc1 = cs.shape
    dt = [1e9, 1e9]
    if scheme is not None:
        adv = cfl / adv_cfl                                        # :332
        if nc1 > 1:
            one_d = nc2 == 1                                       # :335, nc3 == 1 here
            denom1 = (np.abs(v1) + cs) if (not (scheme & 1) or one_d) else np.abs(v1) * adv
            with np.errstate(divide="ignore"):
                dt[0] = np.min(dx1 / denom1)
        if nc2 > 1:
            denom2 = (np.abs(v2) + cs) if not ((scheme >> 1) & 1) else np.abs(v2) * adv
            dt[1] = np.min(dx2 / denom2)
        if shear_cfl > 0. and nc1 > 1 and nc2 > 1:                 # :354-374
            csf = 0.5 * (cs[:, :-1] + cs[:, 1:])
            scale = shear_cfl / cfl
            lo, hi = v2[:, :-1], v2[:, 1:]
            prod = np.maximum(np.abs(lo) * np.abs(hi), 1e-30)
            dts = np.where(np.abs(hi - lo) >= csf, scale * csf * dx2 / prod, 1e9)
            dt[0] = min(dt[0], dts.min())
    else:
        dt[0] = np.min(dx1 / (np.abs(v1) + cs))                    # :377-381
        if nc2 > 1:
            dt[1] = np.min(dx2 / (np.abs(v2) + cs))
    return min(dt)


def block_max_time_step(dt_local, cfl, redo):  # meshblock.cpp:528, after the MIN allreduce
    return 2. ** (-redo) * cfl * dt_local


def column(nc2=8, nc1=16, v1=100., v2=0., cs=340.):
    return (np.full((nc2, nc1), v1), np.full((nc2, nc1), v2), np.full((nc2, nc1), cs))


def check_C1_explicit_acoustic_courant():
    v1, v2, cs = column()
    dx1, dx2, cfl = 10., 50., 0.9
    dt = block_max_time_step(hydro_max_time_step(v1, v2, cs, dx1, dx2, cfl), cfl, 0)
    courant = dt * (100. + 340.) / dx1
    return report("C1", abs(courant - cfl) < 1e-14, "explicit: the acoustic Courant number of the binding direction is cfl",
                  "(|v1|+c_s) dt/dx1 = %.15f for cfl 0.9" % courant)


def check_C2_implicit_advective_courant_is_advection_cfl():
    v1, v2, cs = column()
    out, ok = [], True
    for cfl in (0.5, 0.9):
        for adv_cfl in (0.25, 1.0, 2.5):
            dt = block_max_time_step(hydro_max_time_step(v1, v2, cs, 10., 1e6, cfl, scheme=9, adv_cfl=adv_cfl), cfl, 0)
            courant = dt * 100. / 10.
            ok = ok and abs(courant - adv_cfl) < 1e-14
            out.append("cfl %.1f adv %.2f -> %.15f" % (cfl, adv_cfl, courant))
    return report("C2", ok, "implicit x1: |v1| dt/dx1 equals advection-cfl whatever cfl is", "; ".join(out))


def check_C3_one_dimensional_column_keeps_the_acoustic_bound():
    v1, v2, cs = column(nc2=1)
    dt = block_max_time_step(hydro_max_time_step(v1, v2, cs, 10., 50., 0.9, scheme=9, adv_cfl=1.), 0.9, 0)
    courant = dt * (100. + 340.) / 10.
    return report("C3", abs(courant - 0.9) < 1e-14, "a column with nc2 = nc3 = 1 runs at the acoustic bound even with an implicit scheme",
                  "(|v1|+c_s) dt/dx1 = %.15f" % courant)


def check_C4_shear_bound():
    cs0, dx2, cfl, sh = 340., 50., 0.9, 0.2
    v1, v2, cs = column(v1=0.)
    v2[:, :8], v2[:, 8:] = 3 * cs0, -3 * cs0         # a 6 c_s reversal across one x1 face
    dt = block_max_time_step(hydro_max_time_step(v1, v2, cs, 10., dx2, cfl, scheme=9, shear_cfl=sh), cfl, 0)
    want = sh * cs0 * dx2 / (3 * cs0) ** 2
    v2c = np.where(v2 > 0, 0.3 * cs0, -0.3 * cs0)       # jump 0.6 c_s: does not qualify
    dtc = block_max_time_step(hydro_max_time_step(v1, v2c, cs, 10., dx2, cfl, scheme=9, shear_cfl=sh), cfl, 0)
    wantc = cfl * dx2 / (0.3 * cs0 + cs0)
    ok = abs(dt / want - 1) < 1e-14 and abs(dtc / wantc - 1) < 1e-14
    return report("C4", ok, "shear bound shear-cfl c_f dx2/(|v2_i||v2_i+1|), only where the jump is at least c_f",
                  "dt = %.6e s (= %.6e); a 0.6 c_s jump leaves the x2 acoustic bound %.6e s" % (dt, want, dtc))


def check_C5_redo_halving():
    out = [block_max_time_step(1.0, 0.9, r) for r in range(4)]
    ok = all(abs(out[r] - 0.9 * 2. ** -r) < 1e-15 for r in range(4))
    return report("C5", ok, "each redo halves the step", "dt / dt_bound for redo 0..3: %s" % ", ".join("%.4f" % x for x in out))


def main():
    checks = [check_C1_explicit_acoustic_courant, check_C2_implicit_advective_courant_is_advection_cfl,
              check_C3_one_dimensional_column_keeps_the_acoustic_bound, check_C4_shear_bound, check_C5_redo_halving]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
