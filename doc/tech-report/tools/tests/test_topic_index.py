"""Gate of the topic index (N1): tools/topic-index.yaml against the book, and the Appendix F generator.

The gate fails when a label in the data file resolves to no `#` or `##` heading, when a labelled `##` section has no
index term, and when the committed Appendix F fragment is not a fresh generation. The fixture book under
fixtures/topic-index shows each failure on a small case.

Run from the report directory: `python3 -m pytest tools/tests`.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import gen_topic_index as g  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "topic-index"


def _run(data: str, tmp_path, *extra):
    return g.main(["--data", str(FIX / data), "--book", str(FIX / "book"),
                   "--fragment", str(tmp_path / "_appf.qmd"), *extra])


# --- the gate on the fixture book --------------------------------------------------------------------------------

def test_the_gate_passes_on_a_complete_index(tmp_path):
    assert _run("good.yaml", tmp_path) == 0
    assert _run("good.yaml", tmp_path, "--check") == 0


def test_the_gate_fails_on_a_label_that_does_not_resolve(tmp_path, capsys):
    assert _run("broken-label.yaml", tmp_path) == 1
    err = capsys.readouterr().err
    assert "sec-ch01-cfI: no `#` or `##` heading of the book has this label" in err


def test_the_gate_fails_on_a_section_with_no_entry(tmp_path, capsys):
    assert _run("missing-section.yaml", tmp_path) == 1
    err = capsys.readouterr().err
    assert "section sec-ch01-cfl" in err and "has no index entry" in err


def test_the_gate_fails_on_an_unlisted_unlabelled_heading(tmp_path, capsys):
    book = tmp_path / "book"
    (book / "chapters").mkdir(parents=True)
    (book / "chapters" / "01-x.qmd").write_text("# X {#sec-ch01}\n\n## Labelled {#sec-ch01-a}\n\n## No label\n")
    data = tmp_path / "t.yaml"
    data.write_text("sections:\n  sec-ch01-a:\n    - a topic\n")
    rc = g.main(["--data", str(data), "--book", str(book), "--fragment", str(tmp_path / "f.qmd")])
    assert rc == 1 and "`## No label` has no label" in capsys.readouterr().err
    data.write_text("sections:\n  sec-ch01-a:\n    - a topic\nunlabeled:\n  - file: chapters/01-x.qmd\n"
                    "    heading: No label\n    covered-by: sec-ch01\n")
    assert g.main(["--data", str(data), "--book", str(book), "--fragment", str(tmp_path / "f.qmd")]) == 0


def test_a_chapter_label_may_carry_a_term_but_need_not(tmp_path):
    """An appendix with no `##` sections (the test index) can be an index destination; a chapter label stays
    optional, so every other chapter heading still needs no entry."""
    book = tmp_path / "book"
    (book / "chapters").mkdir(parents=True)
    (book / "chapters" / "01-x.qmd").write_text("# X {#sec-ch01}\n\n## Labelled {#sec-ch01-a}\n")
    (book / "chapters" / "appc-t.qmd").write_text("# Test index {#sec-appc}\n\n### Summary\n")
    data = tmp_path / "t.yaml"
    data.write_text("sections:\n  sec-ch01-a:\n    - test problems\n  sec-appc:\n    - test problems\n")
    frag = tmp_path / "f.qmd"
    assert g.main(["--data", str(data), "--book", str(book), "--fragment", str(frag)]) == 0
    assert "**test problems**: Labelled, @sec-ch01-a; Test index, @sec-appc." in frag.read_text()


def test_an_alias_may_not_also_be_a_term(tmp_path, capsys):
    data = tmp_path / "t.yaml"
    data.write_text((FIX / "good.yaml").read_text().replace("[CFL, dt]", "[CFL, implicit scheme]"))
    rc = g.main(["--data", str(data), "--book", str(FIX / "book"), "--fragment", str(tmp_path / "f.qmd")])
    assert rc == 1 and "alias 'implicit scheme'" in capsys.readouterr().err


def test_the_fragment_merges_shared_terms_and_points_aliases(tmp_path):
    assert _run("good.yaml", tmp_path) == 0
    text = (tmp_path / "_appf.qmd").read_text()
    assert "**implicit scheme**: The implicit solve, @sec-ch01-implicit." in text
    assert "**implicit scheme, time step**: The time step, @sec-ch01-cfl." in text
    assert "*CFL*, see **time step**." in text
    # an unnumbered section is linked, not cross-referenced by number
    assert "**install**: [Install and run](#sec-preface-install)." in text
    # a heading inside a code fence is not a section, and a part divider needs no entry
    assert "not a heading" not in text


def test_the_reader_accepts_what_a_yaml_reader_reads():
    yaml = pytest.importorskip("yaml")
    for path in [g.DATA, *FIX.glob("*.yaml")]:
        text = path.read_text()
        full = yaml.safe_load(text)
        mine = g.load(text)
        assert mine["sections"] == full["sections"], path.name
        assert mine["unlabeled"] == (full.get("unlabeled") or []), path.name


# --- the gate on the real book -----------------------------------------------------------------------------------

def test_every_section_of_the_book_has_an_index_entry_and_every_label_resolves():
    data = g.load(g.DATA.read_text())
    assert g.problems(data, g.headings(g.BOOK)) == []


def test_the_committed_appendix_f_matches_a_fresh_generation():
    assert g.main(["--check"]) == 0, "run python3 tools/gen_topic_index.py and commit the fragment"


def test_every_render_regenerates_appendix_f():
    yaml = pytest.importorskip("yaml")
    book = TOOLS.parent / "book"
    pre = yaml.safe_load((book / "_quarto.yml").read_text())["project"]["pre-render"]
    pre = [pre] if isinstance(pre, str) else pre
    assert any((book / p).resolve() == Path(g.__file__).resolve() for p in pre)
    include = f"{{{{< include {g.FRAGMENT.relative_to(book / 'chapters').as_posix()} >}}}}"
    assert include in (book / "chapters" / "appf-topic-index.qmd").read_text()


def test_implicit_scheme_leads_to_the_vic_sections():
    data = g.load(g.DATA.read_text())
    labels = [lab for lab, ents in data["sections"].items() if "implicit scheme" in [t for t, _ in g.items(ents)]]
    assert labels and all(lab.startswith("sec-ch09-vic-") for lab in labels)


# --- appendix F does not count as prose for the ratchet's uncited-section check ---------------------------------

def _uncited(counts_detail):
    _, detail = counts_detail
    return [d for d in detail if d.endswith("is not cited from another chapter")]


def test_appendix_f_citations_do_not_make_a_section_cited(tmp_path):
    import ratchet_gate as rg
    book = tmp_path / "book"
    (book / "chapters" / "17-appendices").mkdir(parents=True)
    (book / "chapters" / "01-a.qmd").write_text("# A {#sec-ch01}\n\n## One {#sec-ch01-one}\n\nSee @sec-ch02-two.\n")
    (book / "chapters" / "02-b.qmd").write_text("# B {#sec-ch02}\n\n## Two {#sec-ch02-two}\n\nSee @sec-ch01.\n")
    (book / "chapters" / "appf-topic-index.qmd").write_text(
        "# Topic index\n\n{{< include 17-appendices/_appf-topic-index.qmd >}}\n")
    (book / "chapters" / "17-appendices" / "_appf-topic-index.qmd").write_text(
        "**one**: One, @sec-ch01-one.\n\n**two**: Two, @sec-ch02-two.\n")
    uncited = _uncited(rg.xref_counts(book))
    # sec-ch01-one is cited only from the topic index: still uncited; sec-ch02-two is cited from ch01 prose
    assert "ch01: section sec-ch01-one is not cited from another chapter" in uncited
    assert not any("sec-ch02-two" in d for d in uncited)


def test_the_real_uncited_counts_ignore_the_topic_index():
    import ratchet_gate as rg
    counts, detail = rg.xref_counts(TOOLS.parent / "book")
    uncited = _uncited((counts, detail))
    assert sum(d.startswith("appe:") for d in uncited) == 1
    assert sum(d.startswith("ch16:") for d in uncited) == 4
