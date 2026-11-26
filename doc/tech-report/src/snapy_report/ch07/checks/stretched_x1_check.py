"""Executable check of the diffusion stencils on a stretched x1 (chapter 9, _viscosity.qmd, _wallcoef.qmd,
_csdiffusion.qmd).

What is checked, ported line for line from snapy@68f4e066974ede87e4379285c4fba895b970b7f1:
  face_mean and centered_derivative of src/forcing/diffusion.cpp on a stretched x1; the wall extrapolation
  (1 + g) k_near - g k_next with g = h_near / (h_near + h_next) of extrapolate_to_wall, of the cubed-sphere face
  coefficient (DiffusionImpl::forward_cubed_sphere) and of DiffusionImpl::radial_conduction_step; and the four-point
  Lagrange weights of `weights` in src/forcing/diffusion_cubed_sphere.cpp, on the actual x1 centres or on the index.
Claims C1-C4; C1-C3 are exact (sympy), C4 is double precision.

Run: python3 stretched_x1_check.py   (numpy and sympy; exits with the number of failed claims)
"""
import sys

import numpy as np
import sympy as sp


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def check_C1_face_mean():
    """face_mean: (hu * below + hd * above) / (hd + hu), hd and hu the widths of the cells below and above."""
    a, b, xf, hd, hu = sp.symbols("a b x_f h_d h_u", positive=True)
    q = lambda x: a + b * x
    below, above = q(xf - hd / 2), q(xf + hu / 2)       # cell-centre values of a linear q
    mean = (hu * below + hd * above) / (hd + hu)
    exact = sp.simplify(mean - q(xf)) == 0
    uniform = sp.simplify((mean - (below + above) / 2).subs(hu, hd)) == 0
    return report("C1", exact and uniform, "the distance-weighted x1 face mean is the face value of any linear q, "
                  "and the two-cell average on equal widths", "mean - q(x_f) = 0: %s; hd = hu gives (q_i-1 + q_i)/2: %s"
                  % (exact, uniform))


def check_C2_centred_difference():
    """centered_derivative on a stretched x1: (q_{i+1} - q_{i-1}) / (x_{i+1} - x_{i-1})."""
    a, b, x0, d1, d2, h = sp.symbols("a b x_0 d_1 d_2 h", positive=True)
    q = lambda x: a + b * x
    xm, xp = x0 - d1, x0 + d2
    exact = sp.simplify((q(xp) - q(xm)) / (xp - xm) - b) == 0
    uniform = sp.simplify(((q(x0 + h) - q(x0 - h)) / (2 * h)) - (q(x0 + h) - q(x0 - h)) / ((x0 + h) - (x0 - h))) == 0
    return report("C2", exact and uniform, "the x1 centred difference over the centre distance is exact for a linear q "
                  "on any spacing and is the 2h form on equal widths", "exact: %s; equal widths: %s" % (exact, uniform))


def check_C3_wall_extrapolation():
    """(1 + g) k_near - g k_next, g = h_near / (h_near + h_next): linear in position to the wall face."""
    a, b, xw, hn, hx = sp.symbols("a b x_w h_n h_x", positive=True)
    k = lambda x: a + b * x
    near, nxt = k(xw + hn / 2), k(xw + hn + hx / 2)      # wall at x_w, near cell, then next cell
    g = hn / (hn + hx)
    wall = (1 + g) * near - g * nxt
    exact = sp.simplify(wall - k(xw)) == 0
    half = sp.simplify(g.subs(hx, hn)) == sp.Rational(1, 2)
    hn_, hx_ = 1.0, 1.5
    lin = lambda x: 1. + 2. * x
    old = 1.5 * lin(hn_ / 2) - 0.5 * lin(hn_ + hx_ / 2)
    return report("C3", exact and half, "the wall coefficient is linear in position over the two widths, exact for a "
                  "linear k, and g = 1/2 (1.5, -0.5) on equal widths",
                  "exact: %s; g(h, h) = 1/2: %s; k = 1 + 2x, widths 1 and 1.5: the 1.5/-0.5 form gives %.4f at the "
                  "wall for 1" % (exact, half, old))


def lagrange_weights(x, t, derivative):
    """weights() of diffusion_cubed_sphere.cpp for one row: four nodes x, evaluation point t."""
    w = np.zeros(4)
    for j in range(4):
        if not derivative:
            w[j] = np.prod([(t - x[k]) / (x[j] - x[k]) for k in range(4) if k != j])
        else:
            for k in range(4):
                if k == j:
                    continue
                term = 1. / (x[j] - x[k])
                for l in range(4):
                    if l != j and l != k:
                        term *= (t - x[l]) / (x[j] - x[l])
                w[j] += term
    return w


def check_C4_lagrange_on_the_radii():
    faces = np.cumsum(np.r_[10., 0.05 * 1.3 ** np.arange(5)])   # five cells growing by 1.3
    xv = 0.5 * (faces[:-1] + faces[1:])
    cubic = lambda x: 1. + x - 0.3 * x ** 2 + 0.02 * x ** 3
    dcubic = lambda x: 1. - 0.6 * x + 0.06 * x ** 2
    t = faces[2]                                             # the face between cells 1 and 2
    nodes, idx = xv[:4], np.arange(4.)
    tq = 1.5                                                 # that face in index space
    h = (faces[-1] - faces[0]) / 5                           # a mean width, the index-space scale
    on_radii = abs(lagrange_weights(nodes, t, False) @ cubic(nodes) - cubic(t))
    d_radii = abs(lagrange_weights(nodes, t, True) @ cubic(nodes) - dcubic(t))
    on_index = abs(lagrange_weights(idx, tq, False) @ cubic(nodes) - cubic(t))
    d_index = abs(lagrange_weights(idx, tq, True) @ cubic(nodes) / h - dcubic(t))
    ok = on_radii < 1e-12 and d_radii < 1e-10 and on_index > 1e-6 and d_index > 1e-3
    return report("C4", ok, "four-point weights on the actual x1 centres reproduce a cubic and its derivative at a face; "
                  "index-space weights on the same cells do not",
                  "errors on the radii: value %.1e, derivative %.1e; in index space: value %.2e, derivative %.2e"
                  % (on_radii, d_radii, on_index, d_index))


def main():
    checks = [check_C1_face_mean, check_C2_centred_difference, check_C3_wall_extrapolation,
              check_C4_lagrange_on_the_radii]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
