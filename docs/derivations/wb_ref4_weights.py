"""Exact checks behind docs/derivations/wb-ref4.md (SNAP_WB_REF4), in sympy.

  1. the base binomial B = (1, 4, 6, 4, 1)/16 has 2nd moment 1;
  2. the face value of cell averages from the quartic primitive: interior (-1, 7, 7, -1)/12, wall rows
     (25, -23, 13, -3)/12 and (3, 13, -5, 1)/12, exact for cubic point profiles;
  3. the base face-density offset delta rho_f = dz^2/12 (p' R' + 2 p R'') + O(dz^4), R = rho/p.
Every check is an exact (symbolic or rational) identity; the script raises on the first failure.

  python3 docs/derivations/wb_ref4_weights.py
"""
import sympy as sp

z, h, x = sp.symbols("z h x", real=True)
R = sp.Rational
B = [R(1, 16), R(4, 16), R(6, 16), R(4, 16), R(1, 16)]


def check(cond, what):
    assert cond, what
    print("ok:", what)


# 1. moments
mom = lambda w, k: sum(wm * m ** k for wm, m in zip(w, range(-2, 3)))
check(mom(B, 2) == 1 and mom(B, 1) == 0, "B: 2nd moment 1 (bias dz^2/2 q'')")

# 2. face value from cell averages via the quartic primitive
def face_weights(face, start):
    """weights on cells start..start+3 (uniform, unit width) of P'(face)"""
    nodes = list(range(start, start + 5))
    w = []
    for k in range(4):
        tot = 0
        for j in range(k + 1, 5):
            L = sp.prod([(x - nodes[m]) / (nodes[j] - nodes[m]) for m in range(5) if m != j])
            tot += sp.diff(L, x).subs(x, face)
        w.append(sp.nsimplify(tot))
    return w


check(face_weights(2, 0) == [R(-1, 12), R(7, 12), R(7, 12), R(-1, 12)], "interior face (-1, 7, 7, -1)/12")
check(face_weights(0, 0) == [R(25, 12), R(-23, 12), R(13, 12), R(-3, 12)], "wall face (25, -23, 13, -3)/12")
check(face_weights(1, 0) == [R(3, 12), R(13, 12), R(-5, 12), R(1, 12)], "next face (3, 13, -5, 1)/12")
a = sp.symbols("a0:4")
q = sum(ak * x ** k for k, ak in enumerate(a))  # cubic point profile
avg = [sp.integrate(q, (x, i, i + 1)) for i in range(4)]
for f in (0, 1, 2):
    check(sp.expand(sum(w * v for w, v in zip(face_weights(f, 0), avg)) - q.subs(x, f)) == 0,
          f"face {f}: exact for a cubic profile")

# 3. base offset: rho_sf - I[rho_ref], smooth p(z), rho(z), to O(h^2)
p0, p1, p2, R0, R1, R2 = sp.symbols("p0 p1 p2 R0 R1 R2")
P = p0 + p1 * x + p2 * x ** 2 / 2          # around the face, x in units of length
Rr = R0 + R1 * x + R2 * x ** 2 / 2
rho = sp.expand(P * Rr)
cavg = lambda f, i: sp.integrate(f, (x, i * h, (i + 1) * h)) / h   # cell i spans [i h, (i+1) h], face at 0
rbar = lambda i: cavg(rho, i)
pbar = lambda i: cavg(P, i)
ratio = lambda i: sp.series(rbar(i) / pbar(i), h, 0, 3).removeO()
rs = lambda i: sp.expand(sum(wb * ratio(i + m) for wb, m in zip(B, range(-2, 3))))
# p_ref = p_bar to high order (wb-ref4.md section 2); I[q]_face = q(0) + O(h^4) for cell averages of q
dref = lambda i: sp.series(pbar(i) * rs(i), h, 0, 3).removeO()
# the cell field dref has cell averages D(z) = rho + h^2 c(z) + ...; its face value I[dref] equals the
# (-1,7,7,-1)/12 combination to O(h^4)
I_dref = sp.expand(sum(w * dref(i) for w, i in zip([R(-1, 12), R(7, 12), R(7, 12), R(-1, 12)], (-2, -1, 0, 1))))
dsf = sp.expand(p0 * (rs(-1) + rs(0)) / 2)
off = sp.simplify(sp.series(dsf - I_dref, h, 0, 3).removeO())
want = h ** 2 / 12 * (p1 * R1 + 2 * p0 * R2)
check(sp.simplify(off - want) == 0, "base face-density offset = dz^2/12 (p'R' + 2 p R'')")

print("all checks passed")
