"""Run the chapter 15 invariant checks and ensure their committed evidence is current."""
from pathlib import Path
import subprocess
import sys
import pytest

CHECKS = sorted((Path(__file__).parents[1] / 'snapy_report/ch16/checks').glob('*_check.py'))


def _arms(script):
    """`--list-breaks` prints one `<arm>\\tbreaks [Cn]\\t<what it does>` line per arm. A check that does not
    know the flag ignores it and prints its ordinary claims instead, so only lines of that shape count."""
    out = subprocess.run([sys.executable, str(script), '--list-breaks'], text=True, capture_output=True)
    if out.returncode:
        return []
    return [line.split('\t')[0] for line in out.stdout.splitlines() if '\tbreaks [' in line]


BREAKS = [(s, arm) for s in CHECKS for arm in _arms(s)]


@pytest.mark.parametrize('script', CHECKS, ids=lambda p: p.stem)
def test_ch15_invariant(script):
    result = subprocess.run([sys.executable, str(script)], text=True, capture_output=True, check=True)
    assert result.stdout == script.with_suffix('.out').read_text()


@pytest.mark.parametrize('script,arm', BREAKS, ids=lambda a: a if isinstance(a, str) else a.stem)
def test_ch15_break_arm_is_red(script, arm):
    """A claim that cannot be made to fail is not a check: every arm must exit nonzero."""
    result = subprocess.run([sys.executable, str(script), '--break', arm], text=True, capture_output=True)
    assert result.returncode != 0, f'{script.stem} --break {arm} passed'
    assert 'FAIL' in result.stdout
