"""Check the chapter-2 relations of the V3 sections (laws, ideal gas, ideal moist, other closures) in sympy.
Run: python3 v3_relations_check.py, from any directory. Independent algebra, not snapy runtime.
Each relation is the one the chapter derives and reads from the code at the snapy pin of the chapter's
PROVENANCE blocks; the code sites are named in the chapter. A check passes when sympy simplifies the
difference of the two sides to 0. [V10] holds under the assumption it states.
"""
import sympy as sp


def _zero(expr):
    return sp.simplify(sp.together(sp.expand(expr))) == 0


def _report(tag, label, parts):
    ok = all(_zero(e) for e in parts)
    print(f'[{tag}] {"PASS" if ok else "FAIL"} {label} ({len(parts)} identities)')
    return ok


def _metric():
    c, rho, v1, v2, v3 = sp.symbols('c rho v1 v2 v3', real=True)
    s = 1 - c**2
    m2, m3 = rho * (v2 + c * v3), rho * (v3 + c * v2)            # coord_vec_lower_impl
    r2, r3 = (m2 - c * m3) / s, (m3 - c * m2) / s                  # coord_vec_raise_impl
    g = sp.Matrix([[1, 0, 0], [0, 1, c], [0, c, 1]])
    v = sp.Matrix([v1, v2, v3])
    m = sp.Matrix([rho * v1, m2, m3])
    ke_metric = sp.Rational(1, 2) * rho * (v.T * g * v)[0]
    ke_prim = sp.Rational(1, 2) * (v.T * m)[0]                     # kinetic_energy_(prim, mom)
    ke_cons = sp.Rational(1, 2) * (m.T * g.inv() * m)[0] / rho     # cons_kinetic_energy_
    return [r2 - rho * v2, r3 - rho * v3, ke_prim - ke_metric, ke_cons - ke_metric, g.det() - s]


def check_V1():
    x, t, gam = sp.symbols('x t gamma', positive=True)
    r, u, p = [sp.Function(n)(x, t) for n in ('r', 'u', 'p')]
    E = p / (gam - 1) + r * u**2 / 2
    mass = sp.diff(r, t) + sp.diff(r * u, x)
    mom = sp.diff(r * u, t) + sp.diff(r * u**2 + p, x)
    ene = sp.diff(E, t) + sp.diff((E + p) * u, x)
    rt = sp.solve(mass, sp.diff(r, t))[0]
    ut = sp.solve(mom.subs(sp.diff(r, t), rt), sp.diff(u, t))[0]
    pt = sp.solve(ene.subs({sp.diff(r, t): rt}).subs({sp.diff(u, t): ut}).doit(), sp.diff(p, t))[0]
    return _report('V1', 'conservative Euler -> primitive velocity and pressure equations',
                   [ut + u * sp.diff(u, x) + sp.diff(p, x) / r, pt + u * sp.diff(p, x) + gam * p * sp.diff(u, x)])


def check_V2():
    return _report('V2', 'metric lower/raise inverse, kinetic energy in three forms, det g', _metric())


def check_V3():
    rho, p, gam, R, K = sp.symbols('rho p gamma R K', positive=True)
    ke = sp.Symbol('KE')
    E = p / (gam - 1) + ke                                         # IdealGasImpl::_prim2cons
    cv = R / (gam - 1)                                             # IdealGasImpl::species_cv_ref
    return _report('V3', 'ideal gas: (gamma-1)(E-KE)=p, UT->I = W->I, c_s^2 = gamma p/rho',
                   [(gam - 1) * (E - ke) - p, rho * cv * (p / (rho * R)) - p / (gam - 1),
                    sp.diff(K * rho**gam, rho) - gam * K * rho**gam / rho])


def _moist():
    S = sp.symbols
    Rd, cvd, R1, R2, cv1, cv2, cc, rho, T = S('R_d c_vd R_1 R_2 c_v1 c_v2 c_c rho T', positive=True)
    y1, y2, y3, u0d, u01, u02, u03, ke = S('y1 y2 y3 u0d u01 u02 u03 KE', real=True)
    ys, Rs, cvs, u0s = [y1, y2, y3], [R1, R2, 0], [cv1, cv2, cc], [u01, u02, u03]   # two vapours, one condensate
    yd = 1 - sum(ys)
    f_eps = 1 + y1 * (R1 / Rd - 1) + y2 * (R2 / Rd - 1) - y3       # IdealMoistImpl::f_eps
    f_sig = 1 + sum(ys[i] * (cvs[i] / cvd - 1) for i in range(3))  # IdealMoistImpl::f_sig
    gd = (cvd / Rd + 1) / (cvd / Rd)                               # gammad_
    Rmix = yd * Rd + sum(ys[i] * Rs[i] for i in range(3))
    cvmix = yd * cvd + sum(ys[i] * cvs[i] for i in range(3))
    p = rho * Rd * T * f_eps                                       # _prim2temp inverted
    off_w = rho * yd * u0d + rho * sum(ys[i] * u0s[i] for i in range(3))            # _prim2intEng
    ie = p * f_sig / f_eps / (gd - 1) + off_w
    off_u = (rho * yd) * u0d + sum((rho * ys[i]) * u0s[i] for i in range(3))        # _cons2prim, conserved rows
    ut_i = off_u + ((rho * yd) * cvd + sum((rho * ys[i]) * cvs[i] for i in range(3))) * T   # _temp2intEng
    gam = 1 + (gd - 1) * f_eps / f_sig                             # "W->A"
    h = [u0s[i] + (cvs[i] + Rs[i]) * T + ke for i in range(3)]     # species_enthalpy
    hd = u0d + (cvd + Rd) * T + ke
    return dict(yd=yd, f_eps=f_eps, f_sig=f_sig, gd=gd, Rd=Rd, cvd=cvd, Rmix=Rmix, cvmix=cvmix, p=p, ie=ie,
                off_u=off_u, ut_i=ut_i, gam=gam, h=h, hd=hd, ys=ys, rho=rho, T=T, ke=ke)


def check_V4():
    d = _moist()
    return _report('V4', 'ideal-moist f_eps, f_sig and gamma_d from the gas and heat-capacity sums',
                   [d['f_eps'] - d['Rmix'] / d['Rd'], d['f_sig'] - d['cvmix'] / d['cvd'],
                    d['gd'] - (1 + d['Rd'] / d['cvd'])])


def check_V5():
    d = _moist()
    return _report('V5', 'ideal-moist round trip, UT->I = W->I, UT->I affine in T, gamma = c_p/c_v',
                   [(d['gd'] - 1) * (d['ie'] - d['off_u']) * d['f_eps'] / d['f_sig'] - d['p'],
                    d['ut_i'] - d['ie'], sp.diff(d['ut_i'], d['T'], 2),
                    d['gam'] - (d['cvmix'] + d['Rmix']) / d['cvmix']])


def check_V6():
    d = _moist()
    total = d['yd'] * d['hd'] + sum(d['ys'][i] * d['h'][i] for i in range(3))
    return _report('V6', 'species enthalpy split sums to (E+p)/rho',
                   [total - (d['ie'] + d['rho'] * d['ke'] + d['p']) / d['rho']])


def check_V7():
    u_i, u_p, r_i, r_p = sp.symbols('uI uP rI rP', real=True)
    return _report('V7', 'anomaly rows: total_u(sub_reference_(u)) = u', [((u_i - r_i) + r_i) - u_i, ((u_p - r_p) + r_p) - u_p])


def check_V8():
    d = _moist()
    rho_back = d['rho'] * d['yd'] + sum(d['rho'] * y for y in d['ys'])
    return _report('V8', 'dry plus species densities give rho', [rho_back - d['rho']])


def check_V9():
    th0, g0, c0, dth, dg, dc, rc, rr, T = sp.symbols('th0 g0 c0 dth dg dc rho_cv rho_R T', real=True)
    p0 = th0 * g0 / c0
    dp = (dth * (g0 + dg) + th0 * dg - p0 * dc) / (c0 + dc)        # kintera eval_const_cv_pressure_delta
    return _report('V9', 'perturbation pressure is the exact difference; p = Theta G / C_v',
                   [dp - ((th0 + dth) * (g0 + dg) / (c0 + dc) - p0), (rc * T) * rr / rc - rr * T])


def check_V10():
    c1, c2, R, T, lam = sp.symbols('c1 c2 R T lambda', positive=True)
    z1, z2 = sp.Function('Z1')(c1), sp.Function('Z2')(c2)
    p = R * T * (z1 * c1 + z2 * c2)
    dpdl = sp.diff(p.subs({c1: lam * c1, c2: lam * c2}, simultaneous=True), lam).subs(lam, 1)
    code = R * T * ((z1 + sp.diff(z1, c1) * c1) * c1 + (z2 + sp.diff(z2, c2) * c2) * c2)
    return _report('V10', 'moist-mixture c_T^2 rho = isothermal dp/dlambda, if Z_n depends on c_n only',
                   [dpdl - code])


def main():
    checks = [check_V1, check_V2, check_V3, check_V4, check_V5, check_V6, check_V7, check_V8, check_V9, check_V10]
    return sum(not check() for check in checks)


if __name__ == "__main__":
    raise SystemExit(main())
