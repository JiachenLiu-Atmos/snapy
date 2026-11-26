"""Executable check of the VIC clamp meter (chapter 13, _vicclamp.qmd).

What is checked: the clamp residual of ImplicitHydroImpl::forward_masked (src/implicit/implicit_hydro.cpp:358-378)
at snapy@e894700ff7aee30b52882e5202b16461413780b0, computed on the output of the chapter 7 port of
vic_constituent_column (src/snapy_report/ch09/vic_port.py). Claims C1-C3.

Run: python3 vicclamp_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ch09"))
import vic_port as vp  # noqa: E402


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def clamp_residual(M, mass, vol, floor=1e-300):
    """implicit_hydro.cpp:358-378 for one column: per cell, |sum_ch MASS V - (M_i - M_(i+1))| over
    max(|M_i|, |M_(i+1)|), floored, then the maximum."""
    Mu = np.append(M[1:], 0.)
    lhs = (mass * vol[:, None]).sum(1)
    scale = np.maximum(np.maximum(np.abs(M), np.abs(Mu)), floor)
    return (np.abs(lhs - (M - Mu)) / scale).max()


def two_cells(starve_species):
    y = np.array([[0.01, 0.02], [0.01, 0.02]])
    rho = np.array([1.0, 0.8])
    w = np.concatenate([rho[:, None], y], 1)
    vol = np.array([1.0, 1.0])
    du = np.zeros((2, 3))
    if starve_species:
        du[:, 1:] = -rho[:, None] * y            # the explicit update empties the species of both cells
    delta0 = du.sum(1) + np.array([-0.1, 0.1])  # the solve moves 0.1 of mass upward through the one face
    return du, w, delta0, vol


def check_C1_zero_when_nothing_clamps():
    du, w, d0, vol = two_cells(False)
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    r = clamp_residual(M, mass, vol)
    return report("C1", r < 1e-15, "with no clamp the cell changes are the face divergence: the meter reads 0",
                  "face transfer %.3g, residual %.1e" % (M[1], r))


def check_C2_starved_species_give_the_species_fraction():
    du, w, d0, vol = two_cells(True)
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    r = clamp_residual(M, mass, vol)
    ok = abs(r - 0.03) < 1e-12 and not mark.any()
    return report("C2", ok, "when every species of the donor is starved, only the dry fraction moves, so each cell misses "
                  "M sum(y): the meter reads sum(y) = 0.03", "residual %.15f; dry clamp marks %d" % (r, int(mark.sum())))


def check_C3_dtype_safe_floor():
    f32 = np.float32(1e-300)
    floor32 = float(np.finfo(np.float32).tiny)
    ok = f32 == 0. and floor32 > 0. and floor32 == 1.1754943508222875e-38
    return report("C3", ok, "the floor of the scale is the smallest normal float in float32, since 1e-300 rounds to 0 "
                  "there; float64 keeps 1e-300", "float32(1e-300) = %g; float32 floor %.6e" % (f32, floor32))


def main():
    checks = [check_C1_zero_when_nothing_clamps, check_C2_starved_species_give_the_species_fraction,
              check_C3_dtype_safe_floor]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
