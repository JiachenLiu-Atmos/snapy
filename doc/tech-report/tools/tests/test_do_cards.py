"""Static check of the runnable cards (N8): tools/do-cards.yaml against the chapter sources (tools/check_do_cards.py).

The check fails on a card that names no deck, on a deck that does not resolve, when the sources and the data file
disagree on the card set, and on a stale named exception. The fixture repository under fixtures/do-cards shows each
failure on a two-card book; the last tests run the check on the real book and data file.

Run from the report directory: `python3 -m pytest tools/tests`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import check_do_cards as c  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "do-cards"
REPORT = FIX / "repo" / "doc" / "report"


def _check(name):
    return c.check(REPORT, FIX / "data" / name)


# --- the fixture book ---------------------------------------------------------------------------------------------

def test_the_fixture_book_has_two_cards():
    assert [x["anchor"] for x in c.source_cards(REPORT / "book")] == ["sec-ch01-one", "sec-ch01-two"]


def test_the_check_passes_when_every_card_has_a_deck_that_resolves():
    assert _check("good.yaml") == (0, [])


def test_the_check_fails_on_a_card_that_names_no_deck():
    status, msgs = _check("no-deck.yaml")
    assert status == 1
    assert msgs == ["ch01-two: names no deck"]


def test_a_named_exception_lets_exactly_that_card_pass():
    assert _check("no-deck-excepted.yaml") == (0, [])


def test_an_exception_fails_once_its_card_is_fixed():
    status, msgs = _check("stale-exception.yaml")
    assert status == 1
    assert msgs == ["ch01-two: stale exception: the card now passes, remove it from exceptions"]


def test_the_check_fails_on_a_deck_that_does_not_resolve():
    status, msgs = _check("bad-deck.yaml")
    assert status == 1
    assert msgs == ["ch01-two: deck examples/missing.yaml does not resolve in the repository"]


def test_the_check_fails_on_a_card_with_no_record():
    status, msgs = _check("missing-record.yaml")
    assert status == 1
    assert any("card under sec-ch01-two" in m and "has no record" in m for m in msgs)


def test_the_check_fails_on_a_record_with_no_card():
    status, msgs = _check("extra-record.yaml")
    assert status == 1
    assert any(m.startswith("ch01-three: record for sec-ch01-three, but no card") for m in msgs)


def test_the_loader_refuses_another_layout():
    import pytest
    with pytest.raises(ValueError):
        c.load("cards:\n  - id: ch01-one\n")


# --- the needs-build-option class: a card that runs only in a build with a CMake option -------------------------

NEEDS = ('"class": "CPU-seconds"', '"class": "needs-build-option", "build_option": {"name": "PNETCDF", "value": "ON"}')


def _variant(tmp_path, *edits):
    """good.yaml with its second record edited; written to tmp_path (fixture data stays out of data/ directories)."""
    lines = (FIX / "data" / "good.yaml").read_text().splitlines()
    for old, new in edits:
        lines[-1] = lines[-1].replace(old, new, 1)
    path = tmp_path / "variant.yaml"
    path.write_text("\n".join(lines) + "\n")
    return c.check(REPORT, path)


def test_a_needs_build_option_card_naming_a_cmake_option_passes(tmp_path):
    assert _variant(tmp_path, NEEDS) == (0, [])


def test_a_needs_build_option_card_must_name_its_option(tmp_path):
    status, msgs = _variant(tmp_path, ('"class": "CPU-seconds"', '"class": "needs-build-option"'))
    assert status == 1
    assert msgs == ["ch01-two: class needs-build-option needs build_option {\"name\": <CMake option>, \"value\": \"ON\"|\"OFF\"}"]


def test_the_build_option_must_be_an_option_of_the_cmake_files(tmp_path):
    status, msgs = _variant(tmp_path, NEEDS, ('"PNETCDF"', '"PNETCDF_OUTPUT"'))
    assert status == 1
    assert msgs == ["ch01-two: build_option PNETCDF_OUTPUT is not an option() of the repository's CMake files"]


def test_the_build_option_value_is_on_or_off(tmp_path):
    status, msgs = _variant(tmp_path, NEEDS, ('"value": "ON"', '"value": "yes"'))
    assert status == 1
    assert msgs == ["ch01-two: build_option value 'yes' is not ON or OFF"]


def test_only_a_needs_build_option_card_names_a_build_option(tmp_path):
    status, msgs = _variant(tmp_path, ('"class": "CPU-seconds"', '"class": "CPU-seconds", "build_option": {"name": "PNETCDF", "value": "ON"}'))
    assert status == 1
    assert msgs == ["ch01-two: build_option is only for class needs-build-option, not CPU-seconds"]


def test_the_runner_skips_with_a_reason_when_the_build_lacks_the_option():
    rec = {"id": "ch14-pnetcdf", "class": "needs-build-option", "build_option": {"name": "PNETCDF", "value": "ON"}}
    off = c.missing_build_option(rec, "NETCDF:BOOL=ON\nPNETCDF:BOOL=OFF\n")
    assert off == "ch14-pnetcdf needs build option PNETCDF=ON; this build has PNETCDF=OFF (CMakeCache.txt)"
    unset = c.missing_build_option(rec, "NETCDF:BOOL=ON\n")
    assert unset == "ch14-pnetcdf needs build option PNETCDF=ON; this build does not set PNETCDF (CMakeCache.txt)"
    assert c.missing_build_option(rec, "PNETCDF:BOOL=ON\n") is None          # the option is on: the card runs
    assert c.missing_build_option({"id": "x", "class": "CPU-seconds"}, "") is None


# CMake's if(): a cached value is true unless it is a false constant (0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, the empty
# string, a *-NOTFOUND or a zero number), case-insensitively.
CMAKE_TRUE = ["ON", "on", "1", "TRUE", "true", "YES", "yes", "Y", "y", "2", "0.5"]
CMAKE_FALSE = ["OFF", "off", "0", "FALSE", "false", "NO", "no", "N", "n", "IGNORE", "NOTFOUND", "PNETCDF-NOTFOUND",
               "pnetcdf-notfound", "", "0.0"]


@pytest.mark.parametrize("cached", CMAKE_TRUE)
def test_every_cmake_true_value_counts_as_on(cached):
    rec = {"id": "ch14-pnetcdf", "class": "needs-build-option", "build_option": {"name": "PNETCDF", "value": "ON"}}
    assert c.missing_build_option(rec, f"PNETCDF:BOOL={cached}\n") is None
    off = dict(rec, build_option={"name": "PNETCDF", "value": "OFF"})
    assert c.missing_build_option(off, f"PNETCDF:BOOL={cached}\n") is not None


@pytest.mark.parametrize("cached", CMAKE_FALSE)
def test_every_cmake_false_value_counts_as_off(cached):
    rec = {"id": "ch14-pnetcdf", "class": "needs-build-option", "build_option": {"name": "PNETCDF", "value": "ON"}}
    why = c.missing_build_option(rec, f"PNETCDF:BOOL={cached}\n")
    assert why == f"ch14-pnetcdf needs build option PNETCDF=ON; this build has PNETCDF={cached} (CMakeCache.txt)"
    off = dict(rec, build_option={"name": "PNETCDF", "value": "OFF"})
    assert c.missing_build_option(off, f"PNETCDF:BOOL={cached}\n") is None


def test_a_build_with_the_option_runs_the_card_and_reports_its_failure(tmp_path):
    """PNETCDF:BOOL=1 is on: the runner must run the card, so a driver that always fails fails the card (no skip)."""
    pytest.importorskip("yaml")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/nonexistent\nPNETCDF:BOOL=1\n")
    driver = build / "bin" / "straka.release"
    driver.write_text("#!/bin/sh\necho the fixture driver always fails\nexit 3\n")
    driver.chmod(0o755)
    env = dict(os.environ, SNAP_PIN_BUILD=str(build), PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                          str(TOOLS / "tests" / "test_do_cards_run.py"), "-k", "ch14-pnetcdf"],
                         cwd=tmp_path, env=env, capture_output=True, text=True).stdout
    assert "1 failed" in out and "skipped" not in out, out[-600:]
    assert "the fixture driver always fails" in out


def test_ch14_pnetcdf_is_a_needs_build_option_card():
    data = c.load((TOOLS / "do-cards.yaml").read_text())
    card = next(r for r in data["cards"] if r["id"] == "ch14-pnetcdf")
    assert card["class"] == "needs-build-option"
    assert card["build_option"] == {"name": "PNETCDF", "value": "ON"}
    assert card["expect"] == "exit 0"


# --- the real book and data file ----------------------------------------------------------------------------------

def test_the_real_data_file_matches_the_sources():
    report = TOOLS.parent
    status, msgs = c.check(report)
    assert (status, msgs) == (0, [])
    data = c.load((report / "tools" / "do-cards.yaml").read_text(encoding="utf-8"))
    assert len(data["cards"]) == len(c.source_cards(report / "book")) == 77


def test_the_construction_card_has_no_runnable_claim_or_exception():
    data = c.load((TOOLS / "do-cards.yaml").read_text())
    card = next(r for r in data['cards'] if r['id'] == 'ch04-sphsrc')
    assert card['class'] == 'not runnable'
    assert card['expect'] == 'nothing to run'
    assert card['run']['kind'] == 'none'
    assert card['id'] not in {e['id'] for e in data['exceptions'] or []}
