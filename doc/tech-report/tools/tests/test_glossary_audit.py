"""Regression cases from the 055481aa audit, including file-local resolution."""
import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_reader_reference as g
REPORT = Path(os.environ.get('GLOSSARY_AUDIT_REPORT', g.REPORT))

meanings={
'boundary-condition':'boundary options: `external` boundary-function names for the domain faces and `internal` solid-cell settings',
'double_precision':'whether an output stream writes NetCDF variables as `NC_DOUBLE` instead of `NC_FLOAT` (default false)',
'forcing':'forcing options block: a map with one sub-block per forcing component',
'gravity-work':'form of the $x_1$ gravity work: `face` (default), `face-wallc` or `cell`',
'gravity-work-fixer':'whether the gravity-work fixer runs (a boolean; default true with `gravity-work: cell`, and it needs `cell`)',
'ncycle_out':'number of cycles between printed cycle lines; 0 turns the line off',
'names':'names of the passive scalars, one per scalar (in the `scalar` block)',
'omega1':r'first rotation-rate component: $\Omega_z$ of the Cartesian frame under the default `type: xyz`, the $x_1$ component under `type: 123`',
}
local={
'type':{'chapters/03-grids-geometry.qmd':'`SyncOptions` field choosing which variables a ghost exchange syncs (default conserved)'},
'depth':{'chapters/01-overview/_layout.qmd':'`PartOptions` field: how many cells deep a selected slab reaches (default 99)'},
'radius':{'chapters/03-grids-geometry/_gnomonic.qmd':'C++ local holding the cell-centre radius `x1v` in the gnomonic source terms'},
'density':{'chapters/08-positivity/_wbface.qmd':'C++ local in `HydroImpl::forward`: the $x_1$ density tensor shifted by edge replication in the well-balanced face fallback'},
'cells':{'chapters/08-positivity/_scalar.qmd':'C++ local: the interior part of the block'},
'$C$':{'chapters/05-well-balanced/_balcol.qmd':"the free gauge constant of the balanced column: the constant pressure perturbation read at the top cell [Pa]"},
'$z$':{f'chapters/{f}':"the $z$ axis of the Cartesian frame (the polar axis), not the grid's $x_1$" for f in ['03-grids-geometry/_csinterp.qmd','07-diffusion-forcing/_coriolis.qmd']},
'$k = 1..N_r$':{f'chapters/{f}':m for f,m in [
('03-grids-geometry/_x1seams.qmd','scalar row index in the seam tag list'),
('05-well-balanced/_wallclamp.qmd','the $x_3$ cell index'),
('09-time-integration/_vic-pivot.qmd','the $x_3$ cell index'),
('07-diffusion-forcing/_framework.qmd','summation index of the bound'),
('10-moist-coupling/_redo.qmd','the $x_3$ cell index')]},
'$l$':{'chapters/05-well-balanced/_wallclamp.qmd':'cell index counted from the wall ($l=0$ the wall cell)'},
r'$\rho_n$':{f'chapters/{f}/_scalar.qmd':r'density of passive scalar $n$, $\rho_{\mathrm d} r_n$ (`scalar_s`)' for f in ['01-overview','08-positivity']},
r'$M$, $M_{\mathrm{wall}}$':{'chapters/07-diffusion-forcing/_framework.qmd':'the number of installed forcing modules'},
}

@pytest.mark.parametrize('term, expected', meanings.items())
def test_audited_definition(term, expected):
    entries = {e['term']: e for e in g.glossary_entries(REPORT)}
    assert entries[term]['meaning'] == expected

@pytest.mark.parametrize('term, uses', local.items())
def test_audited_local_sense(term, uses):
    entries = {e['term']: e for e in g.glossary_entries(REPORT)}
    for file, expected in uses.items():
        assert entries[term]['meaning_by_file'].get(file) == expected
        outputs, _, _ = g.glossary_outputs(REPORT)
        assert expected in outputs[REPORT / 'book' / file]


def test_unsourced_audit_disposition():
    import json
    data = json.loads((REPORT / 'tools/reader-glossary.json').read_text())['entries']
    linked = [e for e in data if e.get('source')]
    notation = [e for e in data if e.get('notation_only')]
    assert len(linked) == 42
    assert len(notation) == 27
    entries = {e['term']: e for e in g.glossary_entries(REPORT)}
    for e in linked:
        assert e['source'] in entries[e['term']]['meaning']
    for e in notation:
        assert 'notation only; see @sec-appa' in entries[e['term']]['meaning']


@pytest.mark.parametrize('metadata', [
    {'source': 'https://github.com/chengcli/snapy/blob/main/src/snap.h#L1'},
    {'source': 'https://github.com/chengcli/snapy/blob/919936da07d80a310b98ab11da184972a3203274/src/snap.h#L1',
     'notation_only': True},
])
def test_glossary_source_refuses_ambiguous_provenance(tmp_path, metadata):
    import json
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'NOTATION.md').write_text((REPORT / 'NOTATION.md').read_text())
    data = {'version': 1, 'entries': [dict(kind='acronym', term='CPU', meaning='central processing unit', **metadata)]}
    (tmp_path / 'tools/reader-glossary.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='invalid pinned glossary source'):
        g.glossary_entries(tmp_path)


@pytest.mark.parametrize('term', [r'$\theta_p$', r'$\theta_v$'])
def test_theta_diagnostic_source(term):
    entries = {e['term']: e for e in g.glossary_entries(REPORT)}
    meaning = entries[term]['meaning']
    assert ('https://github.com/chengcli/snapy/blob/'
            '919936da07d80a310b98ab11da184972a3203274/'
            'src/output/load_diag_output_data.cpp#L98-L104') in meaning
    assert 'notation only' not in meaning
    if term == r'$\theta_p$':
        assert 'the book uses dry constants' in meaning
        assert 'code computes `theta` from entropy' in meaning
