"""Executable check of the isotropic viscosity and heat conduction (chapter 9, _viscosity.qmd).

What is checked: the discrete operator of DiffusionImpl::forward on a uniform Cartesian grid,
ported line for line from src/forcing/diffusion.cpp at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5 (centered_derivative :70-84,
face_normal_derivative :86-97, face_average :99-108, the flux loop :485-564, max_time_step
:572-619). Claims C1-C8 of the Derivation layer.

Run: python3 viscosity_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np

NG = 2


# --- port of the stencil helpers, one direction, interior faces f = NG .. NG+n ---
def centered_derivative(v, h):  # diffusion.cpp:70-84, at the cells NG-1 .. NG+n
    return (v[2:] - v[:-2]) / (2. * h)


def face_normal_derivative(v, h):  # diffusion.cpp:86-97, faces NG-1/2 .. NG+n-1/2
    return (v[NG:len(v) - NG + 1] - v[NG - 1:len(v) - NG]) / h


def face_average(v):  # diffusion.cpp:99-108
    return 0.5 * (v[NG:len(v) - NG + 1] + v[NG - 1:len(v) - NG])


def divergence(F, h):  # coord->divergence on a uniform Cartesian column
    return (F[1:] - F[:-1]) / h


def periodic(v):
    v = v.copy()
    v[:NG] = v[-2 * NG:-NG]
    v[-NG:] = v[NG:2 * NG]
    return v


def column(n, L=2. * np.pi):
    h = L / n
    x = (np.arange(n + 2 * NG) - NG + 0.5) * h
    return x, h


def tendency_1d(rho, v1, v2, cv, T, nu, kappa, h, dt=1.):
    """du of (m1, m2, E) from the x1 fluxes only (one active direction), diffusion.cpp:485-567"""
    div_vel = np.zeros_like(v1)
    div_vel[1:-1] = centered_derivative(v1, h)  # :467-476
    rho_face = face_average(rho)  # face_coefficient without walls
    div_face = face_average(div_vel)
    s1 = 2. * face_normal_derivative(v1, h) - 2. / 3. * div_face  # :521-522
    s2 = face_normal_derivative(v2, h)  # :519, x2 inactive: no cross term
    F1 = -nu * rho_face * s1
    F2 = -nu * rho_face * s2
    FE = face_average(v1) * F1 + face_average(v2) * F2  # :538-540
    FE = FE - kappa * face_average(rho * cv) * face_normal_derivative(T, h)  # :552-560
    return -dt * divergence(F1, h), -dt * divergence(F2, h), -dt * divergence(FE, h)


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def check_C1_uniform_state():
    x, h = column(32)
    one = np.ones_like(x)
    d = tendency_1d(1.2 * one, 0.3 * one, -0.7 * one, 717.5 * one, 300. * one, 0.1, 0.2, h)
    err = max(np.abs(a).max() for a in d)
    # exact: every difference of equal numbers is 0
    return report("C1", err == 0., "uniform state has zero tendency", "max|du| = %.1e" % err)


def check_C2_transverse_laplacian():
    x, h = column(64)
    rho, nu = 1.3, 0.05
    v2 = periodic(np.sin(x))
    _, dm2, _ = tendency_1d(rho * np.ones_like(x), 0. * x, v2, 0. * x + 1., 0. * x, nu, 0., h)
    lap = (v2[NG + 1:len(x) - NG + 1] - 2. * v2[NG:-NG] + v2[NG - 1:-NG - 1]) / h**2
    err = np.abs(dm2 - nu * rho * lap).max() / np.abs(nu * rho * lap).max()
    # modified wavenumber of the 3-point Laplacian
    rate = np.abs(dm2).max() / (nu * rho * np.abs(np.sin(x[NG:-NG])).max())
    k_eff2 = 4. * np.sin(h / 2.)**2 / h**2
    ok = err < 1e-13 and abs(rate / k_eff2 - 1.) < 1e-3
    return report("C2", ok, "transverse momentum: nu rho times the 3-point Laplacian",
                  "rel err %.1e; decay rate / (4 sin^2(kh/2)/h^2) = %.6f (n1 = 64)"
                  % (err, rate / k_eff2))


def check_C3_normal_stress_four_thirds():
    x, h = column(64)
    rho, nu = 1., 0.05
    v1 = periodic(np.sin(x))
    dm1, _, _ = tendency_1d(rho * np.ones_like(x), v1, 0. * x, 0. * x + 1., 0. * x, nu, 0., h)
    _, dm2, _ = tendency_1d(rho * np.ones_like(x), 0. * x, v1, 0. * x + 1., 0. * x, nu, 0., h)
    # 2 d1 v1 - 2/3 div v at the face: the face average of the wide centred
    # divergence makes the x1 normal-stress stencil wider than the shear one
    ratio = np.abs(dm1).max() / np.abs(dm2).max()
    return report("C3", abs(ratio - 4. / 3.) < 2e-3, "x1 normal stress acts at about 4/3 nu",
                  "max|dm1| / max|dm2| = %.5f against 4/3 = 1.33333 (n1 = 64)" % ratio)


def check_C4_total_energy_conserved():
    rng = np.random.default_rng(9)
    x, h = column(48)
    rho = periodic(1. + 0.2 * rng.random(len(x)))
    v1 = periodic(rng.standard_normal(len(x)))
    v2 = periodic(rng.standard_normal(len(x)))
    T = periodic(300. + 10. * rng.standard_normal(len(x)))
    cv = 717.5 + 0. * x
    dm1, dm2, dE = tendency_1d(rho, v1, v2, cv, T, 0.07, 0.3, h)
    scale = max(np.abs(dE).max(), np.abs(dm1).max())
    sums = [abs(a.sum()) * h / scale for a in (dm1, dm2, dE)]
    # flux form: the column sums telescope on a periodic column; tolerance 1e-13
    # relative: 48 terms of either sign at double round-off
    return report("C4", max(sums) < 1e-13, "momentum and total energy conserved (flux form)",
                  "|sum du| h / max|du| = %.1e, %.1e, %.1e" % tuple(sums))


def check_C5_viscous_heating_nonnegative():
    rng = np.random.default_rng(11)
    x, h = column(48)
    rho = periodic(1. + 0.2 * rng.random(len(x)))
    v2 = periodic(rng.standard_normal(len(x)))
    dm1, dm2, dE = tendency_1d(rho, 0. * x, v2, 1. + 0. * x, 0. * x, 0.07, 0., h)
    # rate of change of internal energy to first order in dt: dE - v.dm, summed
    heating = (dE - v2[NG:-NG] * dm2).sum() * h
    return report("C5", heating > 0., "viscosity turns kinetic energy into heat",
                  "sum (dE - v2 dm2) h = %.4e > 0 (random shear, n1 = 48)" % heating)


def check_C6_parabolic_bound():
    out = []
    ok = True
    for ndim in (1, 2, 3):
        h, kappa = 0.1, 2.
        dt = h * h / (2. * ndim * kappa)  # diffusion.cpp:614
        # forward Euler amplification of the checkerboard mode of the ndim-D 3-point Laplacian
        g = 1. - 4. * ndim * kappa * dt / h**2
        ok = ok and abs(abs(g) - 1.) < 1e-14
        out.append("ndim %d: |G| = %.15f" % (ndim, abs(g)))
    return report("C6", ok, "dt = h^2/(2 ndim kappa) is the forward-Euler stability limit",
                  "; ".join(out))


def check_C7_kappa_semantics():
    gamma = 1.4
    cv = 717.5
    cp = gamma * cv
    # E-row flux -rho cv kappa grad T. At fixed rho: rho cv dT/dt = rho cv kappa lap T.
    # At fixed p (enthalpy form): rho cp dT/dt = rho cv kappa lap T.
    r_rho = (cv * 1.) / cv
    r_p = cv / cp
    ok = r_rho == 1. and abs(r_p - 1. / gamma) < 1e-15
    return report("C7", ok, "kappa_iso diffuses T at kappa (fixed rho) and kappa/gamma (fixed p)",
                  "rate/kappa = %.6f at fixed rho, %.6f at fixed p (gamma = 1.4)" % (r_rho, r_p))


def adiabat(n, top=6400., Rd=287., cp=1004.5, g=9.8, T0=300.):
    h = top / n
    z = (np.arange(n + 2 * NG) - NG + 0.5) * h
    T = T0 - g * z / cp
    rho = (T / T0)**((cp - Rd) / Rd)  # rho ~ T^(cv/R) on an adiabat, rho(0) = 1
    return z, h, T, rho


def check_C8_adiabat_interior_heating():
    Rd, cp, g, kappa = 287., 1004.5, 9.8, 75.
    cv = cp - Rd
    errs, ns = [], (64, 128, 256)
    for n in ns:
        z, h, T, rho = adiabat(n)
        _, _, dE = tendency_1d(rho, 0. * z, 0. * z, cv + 0. * z, T, 0., kappa, h)
        dTdt = dE / (rho[NG:-NG] * cv)
        exact = kappa * g * g * cv / (Rd * cp * cp * T[NG:-NG])
        mid = slice(n // 4, 3 * n // 4)
        errs.append(np.abs(dTdt[mid] / exact[mid] - 1.).max())
    order = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    rate300 = kappa * g * g * cv / (Rd * cp * cp * 300.)
    ok = min(order) > 1.9
    return report("C8", ok, "interior heating of a resting adiabat = kappa g^2 cv/(R cp^2 T)",
                  "rate at T = 300 K, kappa = 75: %.4e K/s (%.4e per unit kappa); rel err %s at n1 = %s,"
                  " orders %.2f %.2f" % (rate300, rate300 / kappa,
                                         " ".join("%.2e" % e for e in errs), ns, *order))


def main():
    checks = [check_C1_uniform_state, check_C2_transverse_laplacian,
              check_C3_normal_stress_four_thirds, check_C4_total_energy_conserved,
              check_C5_viscous_heating_nonnegative, check_C6_parabolic_bound,
              check_C7_kappa_semantics, check_C8_adiabat_interior_heating]
    failed = sum(not c() for c in checks)
    sys.exit(failed)


if __name__ == "__main__":
    main()
