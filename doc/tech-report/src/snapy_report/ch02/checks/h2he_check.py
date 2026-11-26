"""H2/He deck arithmetic and a float64 port of the kintera v2.6.13 rotor.

Independent calculations, not a snapy/kintera runtime test.
Sources: snapy 73c45a7c8c991209924e1137992352c335575062 examples/jupiter_crm.yaml;
kintera 0429d795f79a0665afeca565e5a05c87ab4ba1e2 (v2.6.13) src/constants.h:6 (Rgas),
src/thermo/h2_rotation.h:10-43 (rotor); pyharp 4721715855e937c1e8b218e964c0655f46e56e29
src/element.cpp:54-69. All numeric inputs are in SI except the element weights (g/mol).
Run: python src/snapy_report/ch02/checks/h2he_check.py from the report directory.
"""
import numpy as np
import sympy as sp


def rotor_cp_R(temperature, normal=False):
    """Port of h2_rot_reduce/eval_h2cp; levels 0..40, theta_rot = 87.55 K."""
    temperature = np.asarray(temperature, dtype=np.float64)
    levels = np.arange(41, dtype=np.float64)
    m = levels * (levels + 1)
    degeneracy = 2 * levels + 1
    odd = levels % 2 == 1

    def variance(weights):
        wx = weights * np.exp(-87.55 * m / temperature[..., None])
        norm = wx.sum(axis=-1)
        mean = (wx * m).sum(axis=-1) / norm
        second = (wx * m * m).sum(axis=-1) / norm
        return (87.55 / temperature) ** 2 * (second - mean * mean)

    if normal:
        rotation = .25 * variance(degeneracy * ~odd) + .75 * variance(degeneracy * odd)
    else:
        rotation = variance(degeneracy * np.where(odd, 3., 1.))
    return 2.5 + rotation


def main():
    checks = []

    def check(name, value, passed):
        checks.append(bool(passed))
        print(f'[{"PASS" if passed else "FAIL"}] {name}: {value}')

    mu = (1.5 * 1.008 + .15 * 4.002602) * 1e-3
    rgas = 8.31446261815324  # kintera src/constants.h:6
    rd = rgas / mu
    cv = 2.5 * rd
    cp = 3.5 * rd
    # Bounds are half the last displayed decimal unit of each quoted value.
    check('C1 mu_dry [kg/mol]', mu, abs(mu - .002112390) < .5e-9)
    check('C2 Rd [J/(kg K)]', rd, abs(rd - 3936.04) < .005)
    check('C3 cv_d [J/(kg K)]', cv, abs(cv - 9840.11) < .005)
    check('C4 cp_d [J/(kg K)]', cp, abs(cp - 13776.16) < .005)
    gamma = 1 + 1 / sp.Rational(5, 2)
    kappa = 1 / sp.Rational(7, 2)
    check('C5 gamma_d (exact)', gamma, gamma == sp.Rational(7, 5))
    check('C6 R/cp (exact)', kappa, kappa == sp.Rational(2, 7))
    wrong_mu = 2 * 1.008 + 15.999
    nh3_mu = 14.007 + 3 * 1.008
    error = 100 * (wrong_mu / nh3_mu - 1)
    check('C7 water-formula molar mass [g/mol]', wrong_mu, abs(wrong_mu - 18.015) < .0005)
    check('C8 NH3 molar mass [g/mol]', nh3_mu, abs(nh3_mu - 17.031) < .0005)
    check('C9 NH3 molar-mass error [%]', error, abs(error - 5.78) < .005)
    # This molecular interpretation is a hypothesis; no deck establishes its intent.
    physical_mu = (1.5 * 1.008 + .15 * 4.002602) / (.75 + .15)
    he_fraction = sp.Rational(15, 100) / sp.Rational(90, 100)
    check('C10 hypothesized molecular mixture [g/mol], He mole fraction',
          (physical_mu, he_fraction), abs(physical_mu - 2.347) < .0005 and he_fraction == sp.Rational(1, 6))
    high = float(rotor_cp_R(1000.))
    low = float(rotor_cp_R(20., normal=True))
    # These are deliberately approximate limiting-value checks, not truncation bounds.
    check('C11 equilibrium cp/R at 1000 K', high, abs(high - 3.5) < .01)
    check('C12 normal cp/R at 20 K', low, abs(low - 2.5) < .01)
    grid = np.arange(40., 120. + .25, .5)
    values = rotor_cp_R(grid)
    peak = int(np.argmax(values))
    check('C13 sampled equilibrium peak cp/R, T [K]',
          (float(values[peak]), float(grid[peak])), abs(values[peak] - 4.57) < .005 and abs(grid[peak] - 50.) <= 1.)
    normal_max = float(rotor_cp_R(np.arange(20., 200. + .25, .5), normal=True).max())
    check('C14 sampled normal maximum cp/R, 20..200 K in 0.5 K steps', normal_max, normal_max <= 3.26)
    print(f'{sum(checks)}/{len(checks)} PASS')
    return int(not all(checks))


if __name__ == '__main__':
    raise SystemExit(main())
