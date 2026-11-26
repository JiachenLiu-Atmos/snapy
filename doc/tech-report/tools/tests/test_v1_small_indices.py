"""Regressions for the consolidated appendix corrections (issue #126)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_derivation_index as g
import gen_notation_index as n

def test_dropin_checks_are_equation_specific():
    rows = [r for _, _, rows in g.collect(g.BOOK, g.load_statements(g.STATEMENTS), g.CHECKS) for r in rows]
    selected = {r.equation: r.check for r in rows if r.equation and r.equation.startswith('eq-ch07-')}
    assert 'ch09_dropin_check.py' in selected['eq-ch07-coriolis-force']
    assert 'ch09_dropin_check.py' not in selected['eq-ch07-coriolis-sph']
    assert 'ch09_dropin_check.py' not in selected['eq-ch07-viscosity-dt']

def test_notation_generator_uses_current_pin():
    text = n.fragment(n.NOTATION.read_text(), n.BOOK)
    assert 'e894700ff7aee30b52882e5202b16461413780b0' not in text
    assert '919936da07d80a310b98ab11da184972a3203274' in text
