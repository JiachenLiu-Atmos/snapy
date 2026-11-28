"""Gate of the task map (N2): tools/task-map.yaml against the question bank and the book, and Appendix G.

The gate fails when a question of the bank range has no record (or two), when a destination is not a label of the
book, when a status is not one of the five, and when the committed Appendix G fragment is not a fresh generation.
The fixtures under fixtures/task-map run against the topic-index fixture book.

Run from the report directory: `python3 -m pytest tools/tests`.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import gen_task_map as g  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "task-map"
BOOK = Path(__file__).resolve().parent / "fixtures" / "topic-index" / "book"


def _run(data, tmp_path, *extra):
    return g.main(["--data", str(data), "--book", str(BOOK), "--fragment", str(tmp_path / "_appg.qmd"), *extra])


# --- the gate on fixtures ----------------------------------------------------------------------------------------

def test_the_gate_passes_on_a_complete_map(tmp_path):
    assert _run(FIX / "good.yaml", tmp_path) == 0
    assert _run(FIX / "good.yaml", tmp_path, "--check") == 0


def test_the_gate_fails_on_a_missing_bank_id(tmp_path, capsys):
    assert _run(FIX / "missing-id.yaml", tmp_path) == 1
    assert "Q-002: bank question has no record" in capsys.readouterr().err


def test_the_gate_fails_on_a_destination_that_does_not_resolve(tmp_path, capsys):
    assert _run(FIX / "broken-dest.yaml", tmp_path) == 1
    assert "Q-002: dest sec-ch01-cfI is not a label of the book" in capsys.readouterr().err


@pytest.mark.parametrize("edit, message", [
    (("status: answered", "status: done"), "status 'done' is not one of"),
    (("  missing: the largest stable CFL per integrator\n", "  missing: the largest stable CFL per integrator\n"
      "  not_provided: both\n"), "needs exactly one of dest and not_provided"),
    (("- id: Q-003", "- id: Q-001"), "Q-001: listed 2 times"),
    (("bank_ids: Q-001..Q-003", "bank_ids: Q-001..Q-002"), "Q-003: not an id of the bank range"),
])
def test_the_gate_fails_on_other_record_errors(tmp_path, capsys, edit, message):
    data = tmp_path / "t.yaml"
    data.write_text((FIX / "good.yaml").read_text().replace(*edit, 1))
    assert _run(data, tmp_path) == 1
    assert message in capsys.readouterr().err


def test_the_fragment_counts_statuses_and_points_to_sections(tmp_path):
    assert _run(FIX / "good.yaml", tmp_path) == 0
    text = (tmp_path / "_appg.qmd").read_text()
    assert "| answered | 1 |" in text and "| partly documented | 1 |" in text and "| outside scope | 1 |" in text
    assert "| total | 3 |" in text
    # an unnumbered destination is linked; a numbered one is cross-referenced; the gap is stated
    assert "See [Install and run](#sec-preface-install)." in text
    assert ("See The time step, @sec-ch01-cfl. Not covered: the largest stable CFL per integrator." in text)
    assert "Not provided: a plotting tutorial is outside scope." in text


def test_the_reader_accepts_what_a_yaml_reader_reads():
    yaml = pytest.importorskip("yaml")
    for path in [g.DATA, *FIX.glob("*.yaml")]:
        text = path.read_text()
        full, mine = yaml.safe_load(text), g.load(text)
        assert mine["tasks"] == full["tasks"] and mine["bank_ids"] == full["bank_ids"], path.name


# --- the gate on the real book -----------------------------------------------------------------------------------

def test_every_bank_question_has_one_record_and_every_destination_resolves():
    data = g.load(g.DATA.read_text())
    assert g.problems(data, g.labels(g.BOOK)) == []
    assert len(data["tasks"]) == len(g.bank_range(data["bank_ids"])) == 95


def test_the_status_counts_are_those_of_the_lookup():
    # answered 18, partly documented 59, undocumented 10, unsupported by the code 0, outside scope 8 (total 95)
    assert g.counts(g.load(g.DATA.read_text())) == {
        "answered": 18, "partly documented": 59, "undocumented": 10, "unsupported by the code": 0, "outside scope": 8}


def test_the_committed_appendix_g_matches_a_fresh_generation():
    assert g.main(["--check"]) == 0, "run python3 tools/gen_task_map.py and commit the fragment"


def test_every_render_regenerates_appendix_g():
    yaml = pytest.importorskip("yaml")
    book = TOOLS.parent / "book"
    pre = yaml.safe_load((book / "_quarto.yml").read_text())["project"]["pre-render"]
    pre = [pre] if isinstance(pre, str) else pre
    assert any((book / p).resolve() == Path(g.__file__).resolve() for p in pre)
    # Appendix G prints the curated reader-task list (tools/gen_reader_tasks.py); the bank keeps its lookup grades
    import gen_reader_tasks
    include = f"{{{{< include {gen_reader_tasks.FRAGMENT.relative_to(book / 'chapters').as_posix()} >}}}}"
    assert include in (book / "chapters" / "appg-task-map.qmd").read_text()


def test_task_map_citations_do_not_make_a_section_cited(tmp_path):
    import ratchet_gate as rg
    book = tmp_path / "book"
    (book / "chapters" / "17-appendices").mkdir(parents=True)
    (book / "chapters" / "01-a.qmd").write_text("# A {#sec-ch01}\n\n## One {#sec-ch01-one}\n")
    (book / "chapters" / "appg-task-map.qmd").write_text("# Task map\n")
    (book / "chapters" / "17-appendices" / "_appg-task-map.qmd").write_text("**Q-001.** Why? See One, @sec-ch01-one.\n")
    _, detail = rg.xref_counts(book)
    assert "ch01: section sec-ch01-one is not cited from another chapter" in detail


def test_the_render_gate_runs_the_task_map_gate(tmp_path):
    import shutil
    import render_gate as rg
    report = tmp_path / "report"
    shutil.copytree(BOOK, report / "book")
    (report / "tools").mkdir()
    data = report / "tools" / "task-map.yaml"
    shutil.copy(FIX / "good.yaml", data)
    frag = report / "book" / "chapters" / "17-appendices" / "_appg-task-map.qmd"
    assert g.main(["--data", str(data), "--book", str(report / "book"), "--fragment", str(frag)]) == 0
    assert rg.gate_task_map(report / "book") == 0
    shutil.copy(FIX / "missing-id.yaml", data)
    assert rg.gate_task_map(report / "book") == 1
    shutil.copy(FIX / "broken-dest.yaml", data)
    assert rg.gate_task_map(report / "book") >= 1
    data.unlink()
    assert rg.gate_task_map(report / "book") == 0   # not the report layout: skipped
