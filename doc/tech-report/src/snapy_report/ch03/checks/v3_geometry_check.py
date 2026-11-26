"""Check the chapter-3 relations of the V3 sections (Cartesian and stretched x1, gnomonic panels, panel rotation)
in sympy. Run: python3 v3_geometry_check.py, from any directory. Independent algebra, not snapy runtime.
Each relation is the one the chapter derives and reads from the code at the snapy pin of the chapter's
PROVENANCE blocks; the code sites are named in the chapter. A check passes when sympy simplifies the
difference of the two sides to 0.
"""
import sympy as sp


def _zero(expr):
    return sp.simplify(expr) == 0 or sp.simplify(sp.together(sp.expand(expr))) == 0


def _report(tag, label, parts):
    ok = all(_zero(e) for e in parts)
    print(f'[{tag}] {"PASS" if ok else "FAIL"} {label} ({len(parts)} identities)')
    return ok


def check_G1():
    a, b, q = sp.symbols('x1min x1max q', positive=True)
    k, n = sp.symbols('k N', positive=True, integer=True)
    lq = sp.log(q)
    face = lambda kk: a + (b - a) * (sp.exp(kk * lq) - 1) / (sp.exp(n * lq) - 1)   # from_yaml, x1rat (expm1)
    w = lambda kk: face(kk + 1) - face(kk)
    return _report('G1', 'x1rat faces: first face x1min, last x1max, width ratio q',
                   [face(0) - a, face(n) - b, sp.simplify(w(k + 1) / w(k)) - q])


def check_G2():
    f0, f1, f2 = sp.symbols('f0 f1 f2', real=True)
    g1, g2 = 2 * f0 - f1, 2 * f0 - f2                                       # CoordinateOptionsImpl::global_x1f
    return _report('G2', 'mirrored ghost faces repeat the interior widths in reverse',
                   [(f0 - g1) - (f1 - f0), (g1 - g2) - (f2 - f1)])


def check_G3():
    d1, d2, d3, h = sp.symbols('dx1 dx2 dx3 h', positive=True)
    vol = sp.integrate(sp.integrate(sp.integrate(1, (sp.Symbol('s1'), 0, d1)), (sp.Symbol('s2'), 0, d2)),
                       (sp.Symbol('s3'), 0, d3))
    return _report('G3', 'Cartesian volume is the product of the widths', [vol - d1 * d2 * d3])


def _gno():
    X, Y = sp.symbols('X Y', real=True)
    C, D, Q = sp.sqrt(1 + X**2), sp.sqrt(1 + Y**2), sp.sqrt(1 + X**2 + Y**2)
    u = sp.Matrix([1, X, Y]) / Q
    return X, Y, C, D, Q, u


def check_G4():
    X, Y, C, D, Q, u = _gno()
    tx, ty = u.diff(X), u.diff(Y)
    cos = (tx.dot(ty)) / sp.sqrt(tx.dot(tx) * ty.dot(ty))
    return _report('G4', 'panel angle: cos = -XY/(CD), sin = Q/(CD), |du/dX| = D/Q^2, |du/dY| = C/Q^2',
                   [cos - (-X * Y / (C * D)), (-X * Y / (C * D))**2 + (Q / (C * D))**2 - 1,
                    tx.dot(tx) - D**2 / Q**4, ty.dot(ty) - C**2 / Q**4])


def check_G5():
    X, Y, C, D, Q, u = _gno()
    F = sp.atan(X * Y / Q)
    corner = lambda x, y: sp.atan(x * y / sp.sqrt(1 + x**2 + y**2))
    panel = corner(1, 1) - corner(1, -1) - corner(-1, 1) + corner(-1, -1)
    dA2 = u.diff(X).cross(u.diff(Y)).dot(u.diff(X).cross(u.diff(Y)))
    return _report('G5', 'solid angle: d2/dXdY atan(XY/Q) = 1/Q^3 = |du/dX x du/dY|; a panel is 2 pi/3',
                   [sp.diff(F, X, Y) - 1 / Q**3, dA2 - 1 / Q**6, 6 * panel - 4 * sp.pi])


def check_G6():
    rm, rp = sp.symbols('r_m r_p', positive=True)
    r = sp.Symbol('r', positive=True)
    return _report('G6', 'radial measures: volume dr(rp^2+rp rm+rm^2)/3, x2/x3 face rbar dr',
                   [sp.integrate(r**2, (r, rm, rp)) - (rp - rm) * (rp**2 + rp * rm + rm**2) / 3,
                    sp.integrate(r, (r, rm, rp)) - (rp + rm) / 2 * (rp - rm)])


def check_G7():
    X, Y, C, D, Q, u = _gno()
    # x2 face at fixed X: arc rate |du/dY| = C/Q^2 times the face sine Q/(CD) gives 1/(QD);
    # its X derivative over the solid-angle density 1/Q^3, per unit radius, is the source coefficient
    coef = sp.diff((C / Q**2) * (Q / (C * D)), X) / (1 / Q**3)
    return _report('G7', 'x2 source coefficient Delta_j(A sin psi)/V tends to -X/(r D) (per 1/r)',
                   [coef - (-X / D)])


def check_G8():
    X, Y, C, D, Q, u = _gno()
    vz, vx, vy = sp.symbols('vz vx vy', real=True)
    # cs_contra_to_cart_at_ then cs_cart_to_contra_at_, panel frame (identity permutation)
    cz = (vz - vx * X / D - vy * Y / C) / Q
    cx = (vz * X + vx * D - vy * X * Y / C) / Q
    cy = (vz * Y + vy * C - vx * X * Y / D) / Q
    back1 = (cz + X * cx + Y * cy) / Q
    back2 = D / Q * (cx - X * cz)
    back3 = C / Q * (cy - Y * cz)
    return _report('G8', 'contra -> Cartesian -> contra is the identity on a panel',
                   [back1 - vz, back2 - vx, back3 - vy])


def main():
    checks = [check_G1, check_G2, check_G3, check_G4, check_G5, check_G6, check_G7, check_G8]
    return sum(not check() for check in checks)


if __name__ == "__main__":
    raise SystemExit(main())
