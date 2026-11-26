"""The concept registry (N1/N4): defining sections, the "not defined here" mark and its ratchet, the glossary
appendix, the topic-index line of every concept and the rendered-book gate. Each gate fails on a deliberately
broken input and passes on the real book.

Run from the report directory: `python3 -m pytest tools/tests`.
"""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
REPORT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import check_rendered_glossary as rendered  # noqa: E402
import gen_glossary_appendix as appendix  # noqa: E402
import glossary_registry as registry  # noqa: E402

FIX = Path(__file__).resolve().parent / 'fixtures' / 'rendered-glossary'


@pytest.fixture(scope='module')
def data():
    return json.loads((REPORT / registry.REGISTRY).read_text(encoding='utf-8'))


def _broken(data, term, **change):
    broken = copy.deepcopy(data)
    item = next(e for e in broken['entries'] if e['term'] == term)
    for key, value in change.items():
        if value is None:
            item.pop(key, None)
        else:
            item[key] = value
    return broken


# --- the registry: every concept has a defining section, or is marked "not defined here" -------------------------

def test_every_concept_has_a_defining_section(data):
    assert registry.registry_problems(REPORT, data) == []
    assert all(e.get('section') for e in data['entries'])
    assert len(data["entries"]) == 374  # 338 acronyms, keys and symbols, 17 named concepts and 19 reader terms


@pytest.mark.parametrize('term,change,message', [
    ('CPU', {'section': None}, 'acronym CPU: no defining section'),
    ('CPU', {'section': 'sec-ch01-nowhere'}, 'acronym CPU: defining section sec-ch01-nowhere is not a label'),
    ('$x_1, x_2, x_3$', {'section': 'sec-appa-2'}, 'defined by Appendix A group 1, not sec-appa-2'),
    ('riemann-solver', {'section': 'sec-ch01-driver'}, 'key riemann-solver: not defined here, and section '
                                                       'sec-ch01-driver does not use it in prose'),
    ('EOS', {'section': 'sec-ch01-driver'}, 'acronym EOS: section sec-ch01-driver does not define it'),
    ('CPU', {'not_defined_here': False}, 'acronym CPU: section sec-ch01-driver does not define it'),
    ('CPU', {'not_defined_here': 'yes'}, 'acronym CPU: not_defined_here must be true or false'),
])
def test_a_broken_defining_section_fails(data, term, change, message):
    problems = registry.registry_problems(REPORT, _broken(data, term, **change))
    assert any(message in p for p in problems), problems


@pytest.mark.parametrize('kind,term,meaning,paragraph,expected', [
    ('acronym', 'EOS', 'equation of state', 'The equation of state (EOS) converts the state.', True),
    ('acronym', 'WENO', 'weighted essentially non-oscillatory',
     'Acronyms: WENO, weighted essentially non-oscillatory; TVD, total-variation diminishing.', True),
    ('acronym', 'CPU', 'central processing unit', 'The run used one CPU and took 3 s.', False),
    ('key', 'cfl', 'Courant number', '| `integration/cfl` | 0.9 | the Courant number |', True),
    ('key', 'nb2', 'blocks', '| `distribute/layout`, `nb1`, `nb2`, `nb3` | `slab`, 1, 1, 1 | decomposition |', True),
    ('key', 'cfl', 'Courant number', '| block: reduce, `cfl`, redo | `max_time_step` |', False),
    ('key', 'cfl', 'Courant number', 'The step is multiplied by `cfl`, and halved for each redo.', False),
    ('acronym', 'MC', 'monotonised-central limiter', 'TVD, total-variation diminishing; MC, the monotonised-central limiter.',
     True),
    ('key', 'limiter', 'positivity limiter', registry.KEY_TABLE + '| EOS `limiter` | false | positivity correction |', True),
    ('key', 'disable-flux-x3', 'switch', registry.KEY_TABLE + '| `dynamics/disable-flux-x1`, `-x2`, `-x3` | false | gates |',
     True),
    ('key', 'cfl', 'Courant number', '| block: reduce, `cfl`, redo | `max_time_step` |', False),
    ('key', 'nu_iso', 'viscosity', 'With the default `dynamic: false`, `nu_iso` is a kinematic viscosity.', True),
    ('key', 'bot-heat', 'flux', '`body-heat` is a volumetric rate; `top-cool` and `bot-heat` stand for a surface flux.', True),
    ('key', 'gammad', 'ratio', 'the last term; `gammad` is $\\gamma_d$ from the card.', True),
    ('key', 'nx1', 'cells', 'The anchor `nx1` is in the hydro reset.', False),
    ('key', 'uv-solver', 'solver', '**Activation.** `uv-solver` is `auto` (default), `kkt` or `partition`.', False),
    ('concept', 'Riemann solver', '', 'A Riemann solver is a function of the two face states.', True),
    # a named concept that is also a class name is defined in code font (section 1.2)
    ('concept', 'MeshBlock', '', 'A `MeshBlock` is the unit that owns a grid patch and every module that acts on it.',
     True),
    ('concept', 'MeshBlock', '', 'Each `MeshBlock` holds its own stage buffers.', False),
    ('concept', 'MeshBlock', '', 'A `MeshBlockOptions` is a tree of options.', False),
])
def test_the_definition_rule(kind, term, meaning, paragraph, expected):
    assert registry.defines(dict(kind=kind, term=term, meaning=meaning), paragraph) is expected


def test_a_generated_first_use_note_is_not_a_definition():
    """CPU's meaning appears in the book only in generated first-use notes, so the book does not define it."""
    order, paras = registry.paragraphs(REPORT)
    cpu = dict(kind='acronym', term='CPU', meaning='central processing unit')
    assert not any(registry.defines(cpu, p) for s in order for p in paras.get(s, []))
    assert 'central processing unit' in (REPORT / 'book/chapters/01-overview/_driver.qmd').read_text()


# The definitions the earlier recognisers missed (the review of the registry): under an unlabelled heading or in an
# included preamble (Appendix A), in an acronym list, in the configuration key table and in prose.
MISSED = {('acronym', 'PDE'): 'sec-appa-index', ('acronym', 'SI'): 'sec-appa-index',
          ('acronym', 'BLAS'): 'sec-appa-index', ('acronym', 'MC'): 'sec-ch04-plm',
          ('key', 'disable-flux-x2'): 'sec-ch12-yaml', ('key', 'disable-flux-x3'): 'sec-ch12-yaml',
          ('key', 'limiter'): 'sec-ch12-yaml', ('key', 'vertical'): 'sec-ch12-yaml',
          ('key', 'horizontal'): 'sec-ch12-yaml', ('key', 'scale'): 'sec-ch12-yaml', ('key', 'shock'): 'sec-ch12-yaml',
          ('key', 'x1-inner'): 'sec-ch12-yaml', ('key', 'x3-outer'): 'sec-ch12-yaml',
          ('key', 'const-gravity'): 'sec-ch06-cell', ('key', 'nu_iso'): 'sec-ch07-viscosity',
          ('key', 'kappa_iso'): 'sec-ch07-viscosity', ('key', 'body-heat'): 'sec-ch07-heating',
          ('key', 'top-cool'): 'sec-ch07-heating', ('key', 'bot-heat'): 'sec-ch07-heating',
          ('key', 'gammad'): 'sec-ch02-ideal-moist'}


def test_definitions_the_book_gives_are_not_marked(data):
    entries = {(e['kind'], e['term']): e for e in data['entries']}
    assert {k: entries[k][registry.MARK] for k in MISSED} == {k: False for k in MISSED}
    assert {k: entries[k]['section'] for k in MISSED if k[0] == 'acronym'} == {
        k: s for k, s in MISSED.items() if k[0] == 'acronym'}


def test_the_mark_follows_the_book(data):
    """No entry is marked while the book defines it, and the acronyms and keys marked are at most the reviewed 125."""
    marked = [e for e in data['entries'] if e.get(registry.MARK) and e['kind'] in ('acronym', 'key')]
    assert len(marked) <= 125
    assert not [e['term'] for e in data['entries'] if e.get('notation_only') and e.get(registry.MARK)]
    assert 'notation only; see @sec-appa. Not defined here' not in appendix.FRAGMENT.read_text(encoding='utf-8')


def test_no_defining_section_is_a_roll_up(data):
    book_labels = registry.labels(REPORT / 'book')
    assert [e['term'] for e in data['entries'] if registry.rollup(e['section'], book_labels)] == []
    own = {'PLM': 'sec-ch04-plm', 'PPM': 'sec-ch04-ppm', 'WENO': 'sec-ch04-weno', 'LMARS': 'sec-ch04-lmars',
           'HLLC': 'sec-ch04-hllc', 'LLF': 'sec-ch04-roe', 'PVRS': 'sec-ch04-hllc', 'TVD': 'sec-appa-index',
           'LF': 'sec-ch04-riemann'}
    assert {e['term']: e['section'] for e in data['entries'] if e['term'] in own} == own


def test_a_roll_up_section_fails_the_registry(data):
    problems = registry.registry_problems(REPORT, _broken(data, 'PLM', section='sec-ch04'))
    assert 'acronym PLM: section sec-ch04 is a chapter or appendix heading, not a section of its own' in problems


# Reader terms (the Q326 and q333 held-out lookups): concepts a reader looks up that are not eponymous method names, so
# `concept_candidates` does not find them; each is defined in the section it routes to.
READER_TERMS = {'MeshBlock': 'sec-ch01-construction', 'tensor layout': 'sec-ch01-layout',
                'grid type': 'sec-ch03-choosing',
                'ghost exchange': 'sec-ch03-exchange', 'halo exchange': 'sec-ch03-exchange',
                'steady state': 'sec-ch09-vic-steady',
                'ghost zone': 'sec-ch04-framework', 'latent heat': 'sec-ch10-satadj',
                'hydrostatic primitive-equation core': 'sec-ch02-pecore',
                'mixing ratio': 'sec-ch10-tracer', 'Python bindings': 'sec-ch01-python',
                'options object': 'sec-ch01-options', 'module': 'sec-ch01-options',
                'functional': 'sec-ch01-options', 'deck': 'sec-preface-install',
                'case': 'sec-ch01-initialization', 'restart file': 'sec-ch14-restart',
                'conserved variables': 'sec-ch14-netcdf',
                'primitive variables': 'sec-ch14-netcdf'}


def test_named_concepts_are_in_the_registry(data):
    concepts = sorted(e['term'] for e in data['entries'] if e['kind'] == 'concept' and e['term'] not in READER_TERMS)
    assert concepts == registry.concept_candidates(REPORT) and len(concepts) == 17
    assert all(e.get('section') and isinstance(e.get(registry.MARK), bool) for e in data['entries']
               if e['kind'] == 'concept')
    assert '## Named concepts {.unnumbered .unlisted}' in appendix.FRAGMENT.read_text(encoding='utf-8')


def test_reader_terms_are_defined_where_they_route(data):
    """Each reader term is in the registry, unmarked, with the section that defines it (H03, H04, H20, H24, H34,
    H36), and the re-routed entries land on the section that answers the question (H03, H16, H24)."""
    have = {e['term']: e for e in data['entries'] if e['kind'] == 'concept'}
    assert {t: have[t]['section'] for t in READER_TERMS if t in have} == READER_TERMS
    assert not any(have[t][registry.MARK] for t in READER_TERMS)
    defs = registry.definitions(REPORT, data)
    assert all(READER_TERMS[t] in defs[('concept', t)] for t in READER_TERMS)
    routes = {(e['kind'], e['term']): (e['section'], e.get(registry.MARK)) for e in data['entries']}
    assert routes[('key', 'nghost')] == ('sec-ch04-framework', False)
    assert routes[('acronym', 'VIC')] == ('sec-ch09-vic-options', False)  # Q333 H17
    assert routes[('acronym', 'CFL')] == ('sec-ch09-cfl', False)  # Q333 H24
    assert routes[('key', 'uv-solver')] == ('sec-ch10-satadj', False)


def test_a_use_under_an_unlabelled_section_has_no_section(data):
    """A use under a `##` heading with no label (such as the first section of chapter 15) belongs to no section,
    so no concept's defining section is a label that merely covers it."""
    _, uses = registry.prose_uses(REPORT)
    assert any(s is None for occurrences in uses.values() for _, s in occurrences)
    assert all(e['section'] for e in data['entries'])


# --- the "not defined here" ratchet ------------------------------------------------------------------------------

def test_the_baseline_is_the_count_of_marked_entries(data):
    base = json.loads((REPORT / registry.BASELINE).read_text())
    assert base == {'not_defined_here': registry.marked(data)}
    assert registry.ratchet_problems(REPORT) == []


def test_a_new_not_defined_here_entry_fails(data):
    problems = registry.ratchet_problems(REPORT, _broken(data, 'EOS', not_defined_here=True))
    n = registry.marked(data)
    assert problems == [f'{n + 1} entries marked "not defined here", baseline {n}: define the new concepts instead']


def test_an_unrecorded_fall_fails(data):
    problems = registry.ratchet_problems(REPORT, _broken(data, 'CPU', not_defined_here=False))
    assert len(problems) == 1 and 'record the verified fall' in problems[0]


def test_a_missing_baseline_fails(tmp_path):
    assert registry.ratchet_problems(REPORT, baseline=tmp_path / 'none.json')[0].endswith('--update')


# --- the glossary appendix: generated from the registry, every link resolves ------------------------------------

def test_the_committed_glossary_appendix_is_fresh_and_its_links_resolve():
    assert appendix.problems(REPORT) == []


def test_the_glossary_says_not_defined_here(data):
    text = appendix.FRAGMENT.read_text(encoding='utf-8')
    assert text.lower().count('not defined here') == registry.marked(data)
    assert '- **EOS**: equation of state. Defined in' in text


def test_a_glossary_link_that_does_not_resolve_fails(data):
    problems = appendix.problems(REPORT, _broken(data, 'CPU', section='sec-ch01-nowhere'))
    assert problems == ["acronym CPU: link target 'sec-ch01-nowhere' is not a section of the book"]


def test_a_stale_glossary_appendix_fails():
    fragment = REPORT / appendix.FRAGMENT.relative_to(REPORT)
    original = fragment.read_text(encoding='utf-8')
    try:
        fragment.write_text(original.replace('Defined in', 'Defined at', 1))
        assert appendix.problems(REPORT) == [f'{fragment.name} is stale; run python3 tools/gen_glossary_appendix.py']
    finally:
        fragment.write_text(original)


# --- the topic index: one line per concept, to its own defining section -----------------------------------------

def test_every_concept_has_its_own_topic_index_line(data):
    """Source preflight of the rendered check: Appendix F has a line per concept, to its defining section."""
    lines = rendered.index_lines(REPORT / 'book/chapters/17-appendices/_appf-topic-index.qmd')
    for e in data['entries']:
        line = lines[f'`{e["term"]}`' if e['kind'] == 'key' else e['term']]
        assert f'@{e["section"]}' in line or f'(#{e["section"]})' in line, line


def test_a_concept_that_is_also_an_alias_keeps_its_pointer():
    lines = rendered.index_lines(REPORT / 'book/chapters/17-appendices/_appf-topic-index.qmd')
    assert lines['CFL'].endswith('See also **time step**.')


# --- the rendered book: every section resolves in the .aux, every concept is reachable ---------------------------

def _rendered(tmp_path, aux=None, toc=None, index=None):
    root = tmp_path / 'report'
    shutil.copytree(FIX, root)
    for name, text in (('book/index.aux', aux), ('book/index.toc', toc), ('book/_appf.qmd', index)):
        if text is not None:
            (root / name).write_text(text)
    return rendered.problems(root / 'book', root, root / 'tools/reader-glossary.json', root / 'book/_apph.qmd',
                             root / 'book/_appf.qmd')


def test_the_rendered_fixture_passes(tmp_path):
    problems, stats = _rendered(tmp_path)
    assert problems == []
    assert stats == dict(concepts=3, via_toc=2, via_topic_index=1, index_entries=3, roll_ups=0, appendix_links=3,
                         unresolved=0)


def test_a_section_missing_from_the_aux_fails(tmp_path):
    aux = (FIX / 'book/index.aux').read_text().replace('\\newlabel{sec-ch01-driver}', '\\newlabel{sec-ch01-x}')
    problems, stats = _rendered(tmp_path, aux=aux)
    assert 'acronym CPU: defining section sec-ch01-driver does not resolve in index.aux' in problems
    assert 'glossary appendix link #sec-ch01-driver does not resolve in index.aux' in problems
    assert stats['unresolved'] == 2


def test_a_concept_without_its_own_index_line_is_unreachable(tmp_path):
    index = (FIX / 'book/_appf.qmd').read_text().replace('**`type`**: Conservation laws, @sec-ch02-laws.\n', '')
    problems, _ = _rendered(tmp_path, index=index)
    assert problems == ['key type: no topic-index line of its own refers to its defining section sec-ch02-laws',
                        'key type: defining section sec-ch02-laws is neither in the table of contents nor in the '
                        'topic index']


def test_an_index_line_to_a_roll_up_fails(tmp_path):
    index = (FIX / 'book/_appf.qmd').read_text().replace('**CPU**: Driver loop, @sec-ch01-driver.',
                                                          '**CPU**: Overview, @sec-ch01.')
    problems, stats = _rendered(tmp_path, index=index)
    assert problems == ['acronym CPU: no topic-index line of its own refers to its defining section sec-ch01-driver']
    assert stats['index_entries'] == 2


def test_a_rendered_roll_up_fails_even_when_registry_and_index_agree(tmp_path):
    registry_text = (FIX / 'tools/reader-glossary.json').read_text().replace('"section": "sec-ch01-driver"',
                                                                            '"section": "sec-ch01"')
    index = (FIX / 'book/_appf.qmd').read_text().replace('**CPU**: Driver loop, @sec-ch01-driver.',
                                                          '**CPU**: Overview, @sec-ch01.')
    root = tmp_path / 'report'
    shutil.copytree(FIX, root)
    (root / 'tools/reader-glossary.json').write_text(registry_text)
    (root / 'book/_appf.qmd').write_text(index)
    problems, stats = rendered.problems(root / 'book', root, root / 'tools/reader-glossary.json',
                                        root / 'book/_apph.qmd', root / 'book/_appf.qmd')
    assert problems == ['acronym CPU: defining section sec-ch01 is a chapter or appendix heading (chapter.1), a roll-up']
    assert stats['roll_ups'] == 1 and stats['index_entries'] == 3


def test_the_rendered_gate_needs_a_render(tmp_path):
    (tmp_path / 'book').mkdir()
    problems, _ = rendered.problems(tmp_path / 'book', tmp_path)
    assert problems and 'not found' in problems[0]


@pytest.mark.skipif(not (REPORT / 'book/index.aux').exists(), reason='needs a local PDF render of the book')
def test_the_real_render_passes():
    problems, stats = rendered.problems(REPORT / 'book')
    assert problems == [] and stats['unresolved'] == 0 and stats['index_entries'] == stats['concepts']
