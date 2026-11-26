"""Check the shipped bounds, and ensure wrong operators are detected."""
from snapy_report.ch07.checks.ch09_dropin_check import TAGS, check
import pytest


def test_green():
    results = check()
    assert all(ok for ok, _ in results.values()), results


@pytest.mark.parametrize('tag', TAGS)
def test_mutation_is_red(tag):
    results = check(tag)
    assert not results[tag][0], results[tag]
    assert all(ok for name, (ok, _) in results.items() if name != tag), results
