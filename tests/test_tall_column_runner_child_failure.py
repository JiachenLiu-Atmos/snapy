#!/usr/bin/env python3
"""The tall-column runner fails when a ladder child fails (#299).

test_implicit_gravity_tall_column.py runs its face ladder in a child process and judges the
JSON rows they print. A child that dies (an exception exits 1, as a failing rung did) or one that
prints an incomplete ladder must fail the parent. A sitecustomize.py on PYTHONPATH injects the
fault into the --ladder children only; the parent is untouched:
  raise:  the children cannot import snapy (an uncaught exception, exit 1, no rows);
  silent: the children exit 0 before printing any row.
Each must make the parent exit non-zero with a FAIL line that names the child. Before the fix the
parent accepted exit 1 and never checked that the ladder was complete, so it exited 0.

  python test_tall_column_runner_child_failure.py
"""
import os
import subprocess
import sys
import tempfile

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_implicit_gravity_tall_column.py")
FAULTS = {
    "raise": 'import sys\nif "--ladder" in sys.argv:\n    sys.modules["snapy"] = None\n',
    "silent": 'import os, sys\nif "--ladder" in sys.argv:\n    os._exit(0)\n',
}


def main():
    failures = []
    for name, code in FAULTS.items():
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as inject:
            with open(os.path.join(inject, "sitecustomize.py"), "w") as f:
                f.write(code)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([inject] + [p for p in [env.get("PYTHONPATH")] if p])
            out = subprocess.run([sys.executable, RUNNER, "--device", "cpu", "--nstep", "2",
                                  "--geometry", "cartesian", "--courants", "6.6"],
                                 env=env, capture_output=True, text=True, cwd=os.getcwd())
        fails = [line for line in out.stdout.splitlines() if line.startswith("FAIL")]
        named = [line for line in fails if "child" in line]
        print(f"{name:6s}: parent exit {out.returncode}, FAIL lines {len(fails)}, naming the child "
              f"{len(named)}", flush=True)
        for line in fails:
            print("   ", line, flush=True)
        if out.returncode == 0 or not named:
            failures.append(f"{name}: the parent exited {out.returncode} with {len(named)} child FAIL lines")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
