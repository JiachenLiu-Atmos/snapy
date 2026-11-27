"""Tests of tools/gen_notation_index.py, the Appendix A generator.

Run from the report directory: `python3 -m pytest tools/tests`.
"""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import gen_notation_index as g  # noqa: E402

NOTATION = r"""# Notation for the snapy Technical Report

Intro, see chapter 10.7 and Appendix E.

## 1. Coordinates and grids

| symbol | meaning | code | source letter |
|---|---|---|---|
| $\rho$ | total density [kg m$^{-3}$] | `w[IDN]` | |
| $y_n$ | mass fraction, dimensionless [-]; per total mass | `w[ICY+n-1]` | $q_n$ |
| $N_v, N_y$ | vapours and species (count) | `nvapor` | |
| $\mathbf U$ | conserved state vector $(\rho, m)$ | `hydro_u` | |
| $\lvert a\rvert$ | a norm, in $[0,1]$ | | |

## 2. Operators

| symbol | meaning |
|---|---|
| $\mathsf S$ | slope matrix |
| $R_{\mathrm{d}}$ | dry gas constant |

Reserved: never reuse.
"""

CHAPTER = r"""# Thermodynamics {#sec-ch02}

{{< include 02-x/_gas.qmd >}}
"""

SCHEME = r"""## Gas {#sec-ch02-gas}

Density $\rho$ and $R_{\mathrm d}$ appear here, and $\mathsf S q$.
"""


def make_book(tmp_path):
    book = tmp_path / "book"
    (book / "chapters" / "02-x").mkdir(parents=True)
    (book / "chapters" / "02-thermo.qmd").write_text(CHAPTER)
    (book / "chapters" / "02-x" / "_gas.qmd").write_text(SCHEME)
    return book


def test_units_move_to_their_column():
    assert g.split_units("total density [kg m$^{-3}$]", r"$\rho$") == ("total density", "[kg m$^{-3}$]")
    assert g.split_units("mass fraction, dimensionless [-]; per total mass", "$y_n$") == \
        ("mass fraction; per total mass", "[-]")
    assert g.split_units("vapours (count)", "$N_v$") == ("vapours", "count")
    assert g.split_units("state vector", r"$\mathbf U$") == ("state vector", "vector")
    assert g.split_units("slope matrix", r"$\mathsf S$") == ("slope matrix", "matrix")
    assert g.split_units("flux Jacobian", r"$\partial\mathbf F/\partial\mathbf U$") == ("flux Jacobian", "")
    # a bracket inside maths is not a unit
    assert g.split_units("a norm, in $[0,1]$", r"$\lvert a\rvert$") == ("a norm, in $[0,1]$", "")


def test_cells_split_outside_maths_and_code():
    assert g._cells(r"| $\lvert a\rvert$ | x `a|b` | y |") == [r"$\lvert a\rvert$", "x `a|b`", "y"]


def test_rows_first_use_and_references(tmp_path):
    frag = g.fragment(NOTATION, make_book(tmp_path))
    assert "7 symbols in 2 sections" in frag
    assert "@sec-ch08-drycarry and @sec-appe" in frag                     # no hand-typed chapter number
    row = next(l for l in frag.splitlines() if l.startswith(r"| $\rho$"))
    assert row == r"| $\rho$ | total density | [kg m$^{-3}$] | @sec-ch02-gas | `w[IDN]` |"
    y = next(l for l in frag.splitlines() if l.startswith("| $y_n$"))
    assert "(source notes: $q_n$)" in y and "| — |" in y                 # y_n is not in the chapter's maths
    # `\mathrm{d}` in NOTATION matches `\mathrm d` in the chapter
    assert next(l for l in frag.splitlines() if l.startswith(r"| $R_{\mathrm{d}}$")).count("@sec-ch02-gas") == 1
    assert "{#tbl-appa-1}" in frag and "{#sec-appa-2 .unnumbered}" in frag and "Reserved: never reuse." in frag


def test_the_committed_fragment_depends_on_notation_only(tmp_path):
    """First uses and counts are render-time values: a chapter edit changes no committed line of Appendix A."""
    book = make_book(tmp_path)
    text, values = g.outputs(NOTATION, book)
    assert "@sec-ch02-gas" not in text and "{{< generated appa symbols >}} symbols" in text
    rho = next(l for l in text.splitlines() if l.startswith(r"| $\rho$"))
    assert rho == r"| $\rho$ | total density | [kg m$^{-3}$] | {{< generated appa %s >}} | `w[IDN]` |" % g.first_key(r"\rho")
    assert values[g.first_key(r"\rho")] == "@sec-ch02-gas" and values["symbols"] == "7"
    scheme = next(book.rglob("_*.qmd"))
    scheme.write_text(scheme.read_text().replace("\\rho", "\\varrho"))   # the chapter stops using rho
    assert g.outputs(NOTATION, book)[0] == text


def test_check_mode_flags_a_stale_fragment(tmp_path):
    book = make_book(tmp_path)
    (tmp_path / "N.md").write_text(NOTATION)
    frag = tmp_path / "frag.qmd"
    args = ["--notation", str(tmp_path / "N.md"), "--book", str(book), "--fragment", str(frag)]
    assert g.main(args + ["--check"]) == 1
    assert g.main(args) == 0 and g.main(args + ["--check"]) == 0
    (tmp_path / "N.md").write_text(NOTATION.replace("total density", "density"))
    assert g.main(args + ["--check"]) == 1


def test_every_render_regenerates_appendix_a():
    import yaml
    book = TOOLS.parent / "book"
    pre = yaml.safe_load((book / "_quarto.yml").read_text())["project"]["pre-render"]
    pre = [pre] if isinstance(pre, str) else pre
    assert any((book / p).resolve() == Path(g.__file__).resolve() for p in pre)
    include = f"{{{{< include {g.FRAGMENT.relative_to(book / 'chapters').as_posix()} >}}}}"
    assert include in (book / "chapters" / "appa-notation.qmd").read_text()


def test_the_committed_appendix_a_matches_a_fresh_generation():
    assert g.main(["--check"]) == 0, "run python3 tools/gen_notation_index.py and commit the fragment"
