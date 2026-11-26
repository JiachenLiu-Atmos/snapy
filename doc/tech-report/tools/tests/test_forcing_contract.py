"""N6 contract drift must fail for signature, ghosts and registration."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_forcing_contract as c
import code_anchors

# the contract's chapter is legacy (tools/code-anchors-legacy.txt): these tests run again once its line is deleted
_LEGACY = code_anchors.legacy_chapter_of(c.TARGET)
pytestmark = pytest.mark.skipif(bool(_LEGACY), reason=f"{_LEGACY} is in tools/code-anchors-legacy.txt")


def test_clean_contract():
    assert c.check() == 0


@pytest.mark.parametrize('old,new', [
    ('dt: float,', 'dt: int,'),
    ('nx1() + 2 * nghost()', 'nx1() + nghost()'),
    ('block.set_user_stage_forcings([str(forcing_path)])', 'block.set_user_stage_forcings([])'),
])
def test_contract_drift(tmp_path, old, new):
    target = tmp_path / c.TARGET
    target.parent.mkdir(parents=True)
    (tmp_path / 'code_pin.yml').write_text((c.REPORT / 'code_pin.yml').read_text())
    clean = c.generate(c.REPORT)
    assert old in clean
    target.write_text(clean.replace(old, new))
    assert c.check(tmp_path) == 1
    target.write_text(clean)
    assert c.check(tmp_path) == 0
