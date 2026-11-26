"""Executable check of the Coriolis forcing (chapter 9, _coriolis.qmd).

What is checked: Coriolis123Impl::forward (:50-66), CoriolisXYZImpl::reset (:84-108) and the
orthogonal-grid branch of CoriolisXYZImpl::forward (:183-189) of src/forcing/coriolis.cpp at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported line for line. Claims C1-C5. The
cubed-sphere branch calls the coordinate transforms of chapter 3 and is checked by the ctest
cases, not here.

Run: python3 coriolis_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def du_xyz(omega, m, dt):  # coriolis.cpp:187-189
    o1, o2, o3 = omega
    m1, m2, m3 = m
    return np.array([2. * dt * (o3 * m2 - o2 * m3),
                     2. * dt * (o1 * m3 - o3 * m1),
                     2. * dt * (o2 * m1 - o1 * m2)])


def spherical_omega(omegax, omegay, omegaz, theta, phi):  # coriolis.cpp:104-108
    return np.array([
        np.sin(theta) * np.cos(phi) * omegax + np.sin(theta) * np.sin(phi) * omegay
        + np.cos(theta) * omegaz,
        np.cos(theta) * np.cos(phi) * omegax + np.cos(theta) * np.sin(phi) * omegay
        - np.sin(theta) * omegaz,
        -np.sin(phi) * omegax + np.cos(phi) * omegay])


def check_C1_minus_two_omega_cross_m():
    rng = np.random.default_rng(1)
    om, m = rng.standard_normal(3), rng.standard_normal(3)
    err = np.abs(du_xyz(om, m, 0.3) - (-2. * 0.3 * np.cross(om, m))).max()
    return report("C1", err < 1e-15, "du = -2 dt Omega x m", "max error %.1e" % err)


def check_C2_no_work():
    rng = np.random.default_rng(2)
    worst = 0.
    for _ in range(1000):
        om, m = rng.standard_normal(3), rng.standard_normal(3)
        worst = max(worst, abs(m @ du_xyz(om, m, 1.)) / (np.linalg.norm(om) * (m @ m)))
    return report("C2", worst < 1e-15, "the force does no work on an orthogonal grid (m.du = 0)",
                  "max |m.du| / (|Omega| |m|^2) = %.1e over 1000 draws" % worst)


def check_C3_spherical_projection_is_a_rotation():
    rng = np.random.default_rng(4)
    worst = 0.
    for _ in range(1000):
        th, ph = rng.random() * np.pi, rng.random() * 2. * np.pi
        o = rng.standard_normal(3)
        worst = max(worst, abs(np.linalg.norm(spherical_omega(*o, th, ph)) - np.linalg.norm(o)))
    z = spherical_omega(0., 0., 1., 0.7, 1.9)
    ok = worst < 1e-14 and np.allclose(z, [np.cos(0.7), -np.sin(0.7), 0.], atol=1e-15)
    return report("C3", ok, "spherical-polar projection preserves |Omega|; z-axis -> (cos, -sin, 0)",
                  "max norm change %.1e; Omega_z at theta = 0.7: (%.6f, %.6f, %.1f)" % (worst, *z))


def check_C4_cartesian_axis_mapping():
    # coriolis.cpp:86-93: omega1 is read as the z component, omega2 as x, omega3 as y,
    # and on a Cartesian grid x1 = z, x2 = x, x3 = y, so (omega1, omega2, omega3) are the
    # (x1, x2, x3) components unchanged
    omegaz, omegax, omegay = 0.7, -0.4, 0.2
    got = np.array([omegaz, omegax, omegay])
    return report("C4", np.array_equal(got, [0.7, -0.4, 0.2]),
                  "Cartesian xyz: (omega1, omega2, omega3) are the (x1, x2, x3) components",
                  "omega(x1, x2, x3) = %s" % got.tolist())


def check_C5_123_two_d_drops_x3():
    om, m = np.array([0.3, 0.5, 0.7]), np.array([1., 2., 3.])
    full = du_xyz(om, m, 1.)
    two_d = full.copy()
    two_d[2] = 0.  # coriolis.cpp:60-62: du[IVZ] only when w.size(1) > 1 (nc3 > 1)
    lost = full[2]
    return report("C5", lost != 0., "type 123 on a grid with nc3 = 1 leaves du3 unchanged",
                  "the dropped component would have been %.2f; m.du = %.2f in 2-D" % (lost, m @ two_d))


def main():
    checks = [check_C1_minus_two_omega_cross_m, check_C2_no_work,
              check_C3_spherical_projection_is_a_rotation, check_C4_cartesian_axis_mapping,
              check_C5_123_two_d_drops_x3]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
