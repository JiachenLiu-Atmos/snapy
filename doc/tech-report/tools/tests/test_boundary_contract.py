"""Source drift and generated recipe drift must fail the N6 boundary gate."""
import sys
import os
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_boundary_contract as c
import code_anchors

# the contract's chapter is legacy (tools/code-anchors-legacy.txt): these tests run again once its line is deleted
_LEGACY = code_anchors.legacy_chapter_of(c.TARGET)
pytestmark = pytest.mark.skipif(bool(_LEGACY), reason=f"{_LEGACY} is in tools/code-anchors-legacy.txt")


def test_clean_boundary_contract():
    report = Path(os.environ.get("BOUNDARY_CONTRACT_REPORT", c.REPORT))
    assert c.check(report) == 0


@pytest.mark.parametrize("old,new", [
    ("dx1: int", "dx1: float"),
    ("self->bfuncs()[face] = func;", "self->bfuncs()[face] = nullptr;"),
    ("nx1() + 2 * nghost()", "nx1() + nghost()"),
    ('ix1 += "_inner";', 'ix1 += "_bad";'),
    ("bc/*.cpp", "bc/*.cc"),
])
def test_boundary_source_drift(monkeypatch, capsys, old, new):
    read = c.read_source
    seen = []
    def mutated(*args):
        text = read(*args)
        if old in text:
            seen.append(True)
        return text.replace(old, new)
    monkeypatch.setattr(c, "read_source", mutated)
    assert c.check() == 1
    assert seen
    assert "[N6 boundary contract] 1 problem(s)" in capsys.readouterr().out
    monkeypatch.setattr(c, "read_source", read)
    assert c.check() == 0


def test_boundary_recipe_drift(tmp_path):
    target = tmp_path / c.TARGET
    target.parent.mkdir(parents=True)
    (tmp_path / "code_pin.yml").write_text((c.REPORT / "code_pin.yml").read_text())
    clean = c.generate(c.REPORT)
    target.write_text(clean.replace("target.copy_(", "target.add_("))
    assert c.check(tmp_path) == 1
    target.write_text(clean)
    assert c.check(tmp_path) == 0


@pytest.mark.parametrize("required", [
    'BC_FUNCTION(zero_gradient_inner, var, dim, op)',
    'BC_FUNCTION(zero_gradient_outer, var, dim, op)',
    "Boundary function 'zero_gradient_inner' is not defined.",
    'src/CMakeLists.txt#L35',
    'x1-inner: zero_gradient',
    'name="zero_gradient_inner"',
    '@sec-ch11-registry',
])
def test_native_selection_contract(required):
    assert required in c.generate(c.REPORT)
