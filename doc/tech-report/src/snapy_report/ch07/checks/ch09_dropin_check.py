"""D1-D6: float64 ports against rational stored-input references, seed 9.

Sources at snapy 6fbb39750f62b83df096058a4746e77d0a634bc5:
coriolis.cpp:183-189; top_cool.cpp:52-54; relax_bot_temp.cpp:98-100;
diffusion.cpp:130-134, 173, 545-567; coordinate.cpp:535-552 (src/forcing
except coordinate.cpp in src/coord). Bounds are first-order, with finite normal
inputs and no overflow/underflow. D4 treats already formed shared face products
as inputs. D6 tests interior faces; wall extrapolation is covered by D3b.

Run directly for GREEN, or --mutate D1 (etc.) for a deliberate wrong operator.
The mutations change an operator/coefficient, never an assertion or tolerance.
"""
import argparse
from fractions import Fraction as Q
import numpy as np
import sympy as sp

U = np.finfo(float).eps / 2
TAGS = ('D1', 'D2', 'D3a', 'D3b', 'D4', 'D5', 'D5n', 'D6')


def rational(x):
    return Q(float(x))


def check(mutate=None):
    rng = np.random.default_rng(9)
    results = {}
    ratios = []
    for _ in range(20000):
        omega, m = rng.normal(size=(2, 3))
        dt = float(rng.uniform(.01, 2))
        o, p = list(map(rational, omega)), list(map(rational, m))
        du = np.array([2 * dt * (omega[(i+2)%3]*m[(i+1)%3] -
                                omega[(i+1)%3]*m[(i+2)%3]) for i in range(3)])
        if mutate == 'D1':
            du[0] *= -1  # wrong sign in one cyclic component
        work = abs(sum(p[i]*rational(du[i]) for i in range(3)))
        bound = 6*np.sqrt(3)*U*dt*np.linalg.norm(omega)*np.dot(m, m)
        ratios.append(float(work)/bound)
    results['D1'] = (max(ratios) <= 1, 'work/bound max=%.6f (20000 draws)' % max(ratios))

    ratios = []
    for _ in range(20000):
        dt, flux, dz = rng.uniform(.01, 100, 3)
        depth = int(rng.integers(1, 32))
        exact = rational(dt)*rational(flux)/(rational(dz)*depth)
        got = dt*flux/(dz*(depth+1 if mutate == 'D2' else depth))
        ratios.append(float(abs(rational(got)-exact)/abs(exact))/(3*U))
    results['D2'] = (max(ratios) <= 1, 'relative error/(3u) max=%.6f' % max(ratios))

    for tag, n in [('D3a', 3), ('D3b', 2)]:
        ratios = []
        for _ in range(20000):
            h0, h1 = rng.uniform(.01, 10, 2)
            q0, q1 = rng.uniform(200, 400, 2)
            if tag == 'D3a':
                xf = float(rng.uniform(-10, 10)); x0 = xf+h0/2; x1 = x0+(h0+h1)/2
                a = (x0-xf)/(x1-x0)
                aq = (rational(x0)-rational(xf))/(rational(x1)-rational(x0))
            else:
                a = h0/(h0+h1); aq = rational(h0)/(rational(h0)+rational(h1))
            exact = (1+aq)*rational(q0)-aq*rational(q1)
            got = (1+a)*q0 + (a*q1 if mutate == tag else -a*q1)
            bound = U*(n*float(aq)*abs(q0-q1)+2*(1+float(aq))*abs(q0)+float(aq)*abs(q1)+abs(float(exact)))
            ratios.append(float(abs(rational(got)-exact))/bound)
        results[tag] = (max(ratios) <= 1, 'extrapolation error/bound max=%.6f (20000 draws)' % max(ratios))

    ratios = []
    for _ in range(200):
        af = rng.normal(size=33); vol = rng.uniform(.1, 2, 32); dt = float(rng.uniform(.01, 2))
        af[-1] = af[0]
        du = -dt*(np.diff(af)/vol)
        if mutate == 'D4':
            du[0] *= -1  # reverse one cell's divergence
        defect = abs(sum(rational(v)*rational(d) for v, d in zip(vol, du)))
        differences = [rational(b)-rational(a) for a, b in zip(af[:-1], af[1:])]
        bound = 3*U*dt*float(sum(map(abs, differences)))
        ratios.append(float(defect)/bound)
    results['D4'] = (max(ratios) <= 1, 'conservation defect/bound max=%.6f (200 columns, 32 cells)' % max(ratios))

    z, t0, g, cp, cv, r, k = sp.symbols('z T0 g cp cv R kappa', positive=True)
    temp = t0-g*z/cp
    rho = temp**(cv/r)
    flux_factor = temp if mutate == 'D5' else rho
    derived = sp.diff(k*flux_factor*cv*sp.diff(temp, z), z)/(rho*cv)
    expected = k*g*g*cv/(r*cp*cp*temp)
    results['D5'] = (sp.simplify(derived-expected) == 0, 'symbolic resting dry adiabat at fixed density')
    value = float(expected.subs({k:1, g:9.8, cp:1004.5, cv:717.5, r:287, t0:300, z:0}))
    if mutate == 'D5n':
        value *= 1.4  # substitute cp for cv
    results['D5n'] = (abs(value-7.9318e-7) < 5e-12, 'value per unit kappa at 300 K=%.9e K/s' % value)

    x = (np.arange(36)-1.5)/32
    q = (1.25-x)**1.5; scale = 1/q; temp = 300+50*x
    h = 1/32; dt = .125; kappa = .25
    flux, exact_flux = [], []
    for i in range(2, 33):
        coef = .5*(scale[i]*q[i]+scale[i+1]*q[i+1])
        if mutate == 'D6':
            coef = .25*(scale[i]+scale[i+1])*(q[i]+q[i+1])
        flux.append(-kappa*coef*((temp[i+1]-temp[i])/h))
        exact_coef = (rational(scale[i])*rational(q[i])+rational(scale[i+1])*rational(q[i+1]))/2
        exact_flux.append(-rational(kappa)*exact_coef*(rational(temp[i+1])-rational(temp[i]))/rational(h))
    ratios, residuals = [], []
    for i in range(len(flux)-1):
        exact = -rational(dt)*(exact_flux[i+1]-exact_flux[i])/rational(h)
        got = -dt*((flux[i+1]-flux[i])/h)
        bound = 7*U*dt/h*float(abs(exact_flux[i+1])+abs(exact_flux[i]))+3*U*abs(float(exact))
        ratios.append(float(abs(rational(got)-exact))/bound)
        residuals.append(abs(float(exact))/(dt*kappa*50/h))
    results['D6'] = (max(ratios) <= 1, 'tendency error/bound max=%.6f; stored residual/(dt F/h)=%.3e' % (max(ratios), max(residuals)))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mutate', choices=TAGS)
    args, _ = parser.parse_known_args(argv)  # runpy harness retains pytest arguments
    results = check(args.mutate)
    for tag, (ok, detail) in results.items():
        print('[%s] %s %s' % (tag, 'PASS' if ok else 'FAIL', detail))
    failed = sum(not ok for ok, _ in results.values())
    print('%d passed, %d failed' % (len(results)-failed, failed))
    return failed


if __name__ == '__main__':
    raise SystemExit(main())
