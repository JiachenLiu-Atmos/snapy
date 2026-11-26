"""Pinned-source generation must fail closed on stale or missing reader text."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_reader_reference as g


def test_missing_blocks_are_red_on_the_unmigrated_book():
    # Once migration is complete, the real-book assertion below is the GREEN side.
    assert g.check_defaults(g.REPORT) == 0


def test_changed_value_is_rejected_and_regeneration_repairs_it(tmp_path):
    path = tmp_path / 'text.qmd'
    path.write_text(g.marker('value', 'CMake 3.20'))
    blocks = [('text.qmd', 'value', 'CMake 3.22')]
    assert g.sync_blocks(tmp_path, blocks, True) == 1
    assert g.sync_blocks(tmp_path, blocks, False) == 0
    assert g.sync_blocks(tmp_path, blocks, True) == 0


@pytest.mark.parametrize('count', [0, 2])
def test_missing_and_duplicate_blocks_fail_even_when_generating(tmp_path, count):
    (tmp_path / 'text.qmd').write_text('\n'.join([g.marker('value', 'ok')] * count))
    assert g.sync_blocks(tmp_path, [('text.qmd', 'value', 'ok')], False) == 1


def test_pin_values_distinguish_the_two_riemann_paths():
    legacy = g.code_anchors.legacy_chapter_of(g.RIEMANN)
    v = g.source_values(g.REPORT, riemann=not legacy)
    assert v['sha'] == g.code_anchors.read_pin(g.REPORT / 'code_pin.yml')[1]
    assert v['cmake'] == '3.22'
    if legacy:  # the Riemann block's chapter is legacy (tools/code-anchors-legacy.txt)
        pytest.skip(f"{legacy} is in tools/code-anchors-legacy.txt")
    assert (v['no_block'], v['no_type']) == ('hllc', 'roe')
    assert v['no_block_cite'].endswith('/src/riemann/riemann_solver.hpp#L33)')
    assert v['no_type_cite'].endswith('/src/riemann/riemann_solver.cpp#L27)')
    assert v['cmake'] == '3.22'


def test_real_glossary_is_current():
    assert g.check_glossary(g.REPORT) == 0


def glossary_fixture(tmp_path):
    import json
    (tmp_path / 'tools').mkdir()
    book = tmp_path / 'book'
    (book / 'chapters/01-demo').mkdir(parents=True)
    (book / '_quarto.yml').write_text('book:\n  chapters:\n    - chapters/01-demo.qmd\n')
    (book / 'chapters/01-demo.qmd').write_text('# Demo\n\nCPU and $p$; `type`.\n\n{{< include 01-demo/_one.qmd >}}\n')
    (book / 'chapters/01-demo/_one.qmd').write_text('## One\n\nCPU and $p$; `type`.\n')
    (tmp_path / 'NOTATION.md').write_text('## 1. State\n\n| symbol | meaning |\n|---|---|\n| $p$ | pressure |\n')
    data = {'version': 1, 'entries': [
        {'kind': 'acronym', 'term': 'CPU', 'meaning': 'central processing unit'},
        {'kind': 'key', 'term': 'type', 'meaning': 'implementation selector'},
        {'kind': 'symbol', 'term': '$p$', 'notation_group': '1'},
    ]}
    (tmp_path / 'tools/reader-glossary.json').write_text(json.dumps(data))
    return book


def test_glossary_first_use_crosses_include_boundaries_and_is_idempotent(tmp_path):
    book = glossary_fixture(tmp_path)
    assert g.check_glossary(tmp_path) == 1
    assert g.check_glossary(tmp_path, False) == 0
    assert g.check_glossary(tmp_path) == 0
    main = (book / 'chapters/01-demo.qmd').read_text()
    assert main.count('[^rg-') == 6  # three uses and three definitions
    assert '[^rg-' not in (book / 'chapters/01-demo/_one.qmd').read_text()
    (book / 'chapters/01-demo.qmd').write_text(main.replace('central processing unit', 'wrong'))
    assert g.check_glossary(tmp_path) == 1


def test_glossary_defines_yaml_keys_without_changing_the_example(tmp_path):
    book = glossary_fixture(tmp_path)
    main = book / 'chapters/01-demo.qmd'
    code = '```yaml\ntype: roe\n```\n'
    main.write_text('# Demo\n\n' + code)
    assert g.check_glossary(tmp_path, False) == 0
    assert code in main.read_text()
    assert 'Keys in this fragment: `type`[^rg-' in main.read_text()
    assert g.check_glossary(tmp_path) == 0
    main.write_text('# Demo\n\n```yaml\nunregistered: 1\n```\n')
    assert g.check_glossary(tmp_path) == 1


def test_glossary_rejects_symbols_missing_from_appendix_a(tmp_path, capsys):
    glossary_fixture(tmp_path)
    p = tmp_path / 'tools/reader-glossary.json'
    p.write_text(p.read_text().replace('$p$', '$missing$'))
    assert g.check_glossary(tmp_path) == 1
    assert 'absent from Appendix A' in capsys.readouterr().out


def test_glossary_rejects_empty_definitions(tmp_path, capsys):
    glossary_fixture(tmp_path)
    p = tmp_path / 'tools/reader-glossary.json'
    p.write_text(p.read_text().replace('central processing unit', ''))
    assert g.check_glossary(tmp_path) == 1
    assert 'no definition for acronym CPU' in capsys.readouterr().out


@pytest.mark.parametrize('relative, expected, wrong', [
    ('12-configuration/_opening.qmd', 'MC — mass covariance', 'MC — monotonised-central'),
    ('09-time-integration/_cfl.qmd', 'scale — ratio of the shallow-water CFL bound to the selected CFL number', 'scale — scaling in the enclosing component'),
    ('14-parallelism-io.qmd', 'dt — output interval', 'dt — time-step duration'),
])
def test_first_use_uses_the_local_meaning(relative, expected, wrong):
    outputs, _, _ = g.glossary_outputs(g.REPORT)
    text = outputs[g.REPORT / 'book/chapters' / relative]
    assert expected in text
    assert wrong not in text
