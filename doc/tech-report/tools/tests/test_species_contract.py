"""N6 contract drift must fail for the species keys, the state rows and the condensate reaction."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_species_contract as c
import code_anchors

# a target in a legacy chapter (tools/code-anchors-legacy.txt): its tests run again once the chapter's line is deleted
_LEGACY = {k: code_anchors.legacy_chapter_of(t) for k, t in c.TARGETS.items()}
_ALL_LEGACY = all(_LEGACY.values())


@pytest.mark.skipif(_ALL_LEGACY, reason=f"{sorted({v for v in _LEGACY.values() if v})} in tools/code-anchors-legacy.txt")
def test_clean_contract():
    assert c.check() == 0


@pytest.mark.parametrize('key,old,new', [
    ('species', 'vapour["phase"] = "gas"', 'vapour["phase"] = "liquid"'),
    ('species', 'assert (n_dry, n_listed, n_gas) == (5, 5, 6)', 'assert (n_dry, n_listed, n_gas) == (5, 6, 6)'),
    ('species', 'src/species.cpp#L175', 'src/species.cpp#L174'),
    ('condensate', '"formula": "h2o_bryan"', '"formula": "h2o_ideal"'),
    ('condensate', 'hydro_u.narrow(0, ICY, ny) = yfrac * rho;', 'hydro_u.narrow(0, IDN, ny) = yfrac * rho;'),
    ('condensate', 'h_idx.push_back(id + 1);', 'h_idx.push_back(id);'),
])
def test_contract_drift(tmp_path, key, old, new):
    if _LEGACY[key]:
        pytest.skip(f"{_LEGACY[key]} is in tools/code-anchors-legacy.txt")
    (tmp_path / 'code_pin.yml').write_text((c.REPORT / 'code_pin.yml').read_text())
    legacy = c.REPORT / 'tools' / code_anchors.LEGACY_FILE.name  # the same live targets in tmp_path
    (tmp_path / 'tools').mkdir(exist_ok=True)
    (tmp_path / 'tools' / legacy.name).write_text(legacy.read_text())
    clean = c.generate(c.REPORT, [k for k in c.TARGETS if not _LEGACY[k]])
    for k, text in clean.items():
        (tmp_path / c.TARGETS[k]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / c.TARGETS[k]).write_text(text)
    assert c.check(tmp_path) == 0
    assert old in clean[key]
    (tmp_path / c.TARGETS[key]).write_text(clean[key].replace(old, new))
    assert c.check(tmp_path) == 1


KINTERA_BLOB = 'https://github.com/chengcli/kintera/blob/c55b13b2204997d2d09e04498558ab9495d8ee77/'


def _code_lines(text):
    inside, lines = False, []
    for line in text.splitlines():
        if line.strip().startswith('```'):
            inside = not inside
        elif inside:
            lines.append(line)
    return lines


@pytest.mark.parametrize('key', sorted(c.TARGETS))
def test_every_code_line_fits_the_page(key):
    """The committed block's code lines stay within 80 characters, so the PDF does not cut them."""
    text = (c.REPORT / c.TARGETS[key]).read_text()
    wide = [line for line in _code_lines(text) if len(line) > 80]
    assert wide == [], wide


def test_the_native_alternative_gives_the_formula_tables_and_rebuild():
    """The native C++ alternative of the condensate block: the generated signature, a formula and its _ddT pair,
    the CPU table and name list, the CUDA table one position on, the lookups and the rebuild, each linked at the
    kintera pin, and the measured effect of a name without its table entry."""
    text = (c.REPORT / c.TARGETS['condensate']).read_text()
    native = text[text.index('**Native C++ alternative.**'):]
    for piece in ('using user_func1 = double (*)(double);', 'inline double h2o_bryan(double T)',
                  'inline double h2o_bryan_ddT(double T)', 'user_func1 func1_table_cpu[] = {',
                  'std::vector<std::string> func1_names = {', '__device__ user_func1 func1_table_cuda[] = {',
                  'nullptr,', 'funcs.push_back(func_table[id]);', 'h_idx.push_back(id + 1);', 'name += "_ddT";',
                  'cmake --build build --parallel', 'pip install .',
                  'run: `python3 src/snapy_report/ch10/checks/func_table_check.py`'):
        assert piece in native, piece
    for path in ('src/utils/user_funcs.hpp', 'src/vapors/vapor_functions.h', 'src/func_table.cpp',
                 'src/func_table.cu', 'src/utils/user_funcs.cu', 'src/thermo/log_svp.hpp', 'README.md'):
        assert KINTERA_BLOB + path + '#L' in native, path


def test_the_function_table_check_output_is_committed():
    out = (c.REPORT / 'src/snapy_report/ch10/checks/func_table_check.out').read_text()
    assert '5 of 5 passed' in out
    assert all(f'[C{n}] PASS' in out for n in range(1, 6))
