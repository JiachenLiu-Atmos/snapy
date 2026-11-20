"""V3 X5 (gw1): the global gravity-work fixer is removed.

forcing/const-gravity/gravity-work-fixer: true is refused, from YAML and from
Python, with a message naming the removal; false (and no key) still loads, so
decks that switched the fixer off keep working.
"""
import os
import tempfile

import yaml

from gravity_box import config


def build(gravity, python_fixer=None):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(gravity), f)
        tmp = f.name
    try:
        opts = MeshBlockOptions.from_yaml(tmp)
        if python_fixer is not None:
            opts.hydro().grav().gravity_work_fixer(python_fixer)
        return MeshBlock(opts)
    finally:
        os.unlink(tmp)


def refused(fn):
    try:
        fn()
    except RuntimeError as e:
        assert "removed in V3" in str(e), str(e)
        return True
    return False


def main():
    failures = []
    if not refused(lambda: build({"gravity-work": "face", "gravity-work-fixer": True})):
        failures.append("YAML gravity-work-fixer: true was accepted")
    if not refused(lambda: build({"gravity-work": "face"}, python_fixer=True)):
        failures.append("Python gravity_work_fixer(True) was accepted")
    build({"gravity-work": "face", "gravity-work-fixer": False})
    build({"gravity-work": "face"})
    assert not failures, failures
    print("gravity-work-fixer: true refused (YAML, Python); false and no key load")


if __name__ == "__main__":
    main()
