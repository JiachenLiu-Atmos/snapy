"""Independent algebra for chapter 15; no compiled snapy execution.

Source correspondence: coordinate divergence, wall-saturation totals and wall
reference order at snapy 6fbb39750f62b83df096058a4746e77d0a634bc5.
"""
import sympy as s


def main():
    volumes = s.symbols('V0:3', positive=True)
    transfers = s.symbols('AF0:4', real=True)
    tendencies = [(transfers[i] - transfers[i + 1]) / volumes[i] for i in range(3)]
    weighted = sum(v * drho for v, drho in zip(volumes, tendencies))
    assert s.expand(weighted - transfers[0] + transfers[3]) == 0
    assert s.expand(weighted.subs({transfers[0]: 0, transfers[3]: 0})) == 0
    # An internal transfer on unequal volumes conserves mass but not sum(density).
    counterexample = sum(tendencies).subs(dict(zip(volumes, (1, 2, 3))))
    counterexample = counterexample.subs(dict(zip(transfers, (0, 1, 0, 0))))
    assert counterexample == -s.Rational(1, 2)
    print('[C1] PASS volume-weighted telescoping; closed walls zero; unweighted counterexample -1/2')
    total = s.Symbol('total', real=True)
    oracle = s.Symbol('oracle', positive=True)
    assert s.simplify(s.Abs(total / oracle - 1) - s.Abs(total - oracle) / oracle) == 0
    print('[C2] PASS relative-drift identity for a nonzero positive oracle')
    coefficient, width, order = s.symbols('psi h zeta', positive=True)
    ratio = s.simplify(coefficient * width**order / (coefficient * (width / 2)**order))
    assert s.simplify(ratio - 2**order) == 0
    assert s.simplify(s.log(ratio) / s.log(2) - order) == 0
    print('[C3] PASS single-power refinement ratio and logarithmic order')
    print('3 claims pass')


if __name__ == '__main__':
    main()
