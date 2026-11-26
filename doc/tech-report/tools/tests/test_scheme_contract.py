"""N6 scheme contract drift must fail for the interface, faces, ghost width, dispatch and the recipe."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import code_anchors
import gen_scheme_contract as c

# the contract's chapter is legacy (tools/code-anchors-legacy.txt): these tests run again once its line is deleted
_LEGACY = code_anchors.legacy_chapter_of(c.TARGET)
pytestmark = pytest.mark.skipif(bool(_LEGACY), reason=f"{_LEGACY} is in tools/code-anchors-legacy.txt")


def test_clean_contract():
    assert c.check() == 0


@pytest.mark.parametrize('old,new', [
    ('torch::Tensor forward(torch::Tensor w, int dim, bool floor = true);',
     'torch::Tensor forward(torch::Tensor w, int dim);'),
    ('int nghost = pinterp1->stencils() / 2 + 1;', 'int nghost = pinterp1->stencils() / 2;'),
    ('interp.type("dc")', 'interp.type("plm")'),
    ('enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };', 'enum { DIM1 = 1, DIM2 = 2, DIM3 = 3 };'),
    ('18 set `shock: false`', '19 set `shock: false`'),
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


def test_shock_sentence_agrees_with_the_decks_at_the_pin():
    """Count the example decks' reconstruct `shock:` values at the pin, independently of the generator, and
    require the committed contract block to state that split."""
    repo, sha = code_anchors.read_pin(c.REPORT / 'code_pin.yml')
    git = code_anchors.find_commit(repo, sha)
    names = [n for n in subprocess.run(['git', '-C', str(git), 'ls-tree', '--name-only', sha, 'examples/'],
                                       check=True, capture_output=True, text=True).stdout.split()
             if n.endswith('.yaml')]
    counts = {'true': 0, 'false': 0}
    for name in names:
        text = subprocess.run(['git', '-C', str(git), 'show', f'{sha}:{name}'],
                              check=True, capture_output=True, text=True).stdout
        values = set(re.findall(r'shock:\s*(true|false)', text))
        assert len(values) == 1, f'{name}: shock values {values}'
        counts[values.pop()] += 1
    block = (c.REPORT / c.TARGET).read_text()
    assert f'of the {len(names)} decks in `examples/`' in block
    assert f'{counts["false"]} set `shock: false`' in block
    assert f'{counts["true"]} set `shock: true`' in block
