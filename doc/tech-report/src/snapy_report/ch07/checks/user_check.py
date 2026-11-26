"""Executable check of the user stage forcings (chapter 9, _user.qmd).

What is checked: the addition of user hydro_du to the native stage increment in MeshBlockImpl::advance_local
(src/mesh/meshblock.cpp:689-742) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported on numpy arrays, and
how a user increment enters the step through the stage weights of pyharp
(src/integrator/integrator.cpp:33-85 at pyharp@4721715855e937c1e8b218e964c0655f46e56e29, as in chapter 7's
rk_check.py). Claims C1-C3.

Run: python3 user_check.py   (numpy only; exits with the number of failed claims)
"""
import itertools
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


WEIGHTS = {
    "rk1": [(0., 1., 1.)],
    "rk2": [(0., 1., 1.), (.5, .5, .5)],
    "rk3": [(0., 1., 1.), (3. / 4., 1. / 4., 1. / 4.), (1. / 3., 2. / 3., 2. / 3.)],
    "rk3s4": [(.5, .5, .5), (0., 1., .5), (2. / 3., 1. / 3., 1. / 6.), (0., 1., .5)],
}
rng = np.random.default_rng(5)
SHAPE = (5, 8)


def add_user(native, inputs, modules):
    """meshblock.cpp:731-737: the inputs are built once, then each module's hydro_du is added in turn."""
    du = native.copy()
    for mod in modules:
        result = mod(inputs)
        for key in result:
            if key not in ("hydro_du", "scalar_ds"):
                raise ValueError("unsupported key '%s'" % key)
        if "hydro_du" in result:
            if result["hydro_du"].shape != du.shape:
                raise ValueError("hydro_du shape")
            du += result["hydro_du"]
    return du


MODS = [lambda v: {"hydro_du": 0.1 * v["hydro_w"]}, lambda v: {"hydro_du": -0.2 * v["hydro_w"] ** 2},
        lambda v: {"hydro_du": np.full(SHAPE, 0.05)}]


def check_C1_sum_independent_of_module_order():
    native = rng.normal(size=SHAPE)
    inputs = {"hydro_w": rng.uniform(0.5, 1.5, SHAPE)}
    base = add_user(native, inputs, MODS)
    worst = max(np.abs(add_user(native, inputs, [MODS[i] for i in p]) - base).max()
                for p in itertools.permutations(range(3)))
    want = native + sum(m(inputs)["hydro_du"] for m in MODS)
    ok = worst <= 4 * np.finfo(float).eps * np.abs(base).max() and np.abs(base - want).max() < 1e-15
    return report("C1", ok, "the stage hands the integrator the native increment plus every user hydro_du, whatever "
                  "the module order, since all modules read the same inputs", "max order difference %.1e" % worst)


def check_C2_bad_results_are_refused():
    refused = 0
    for mod in (lambda v: {"hydro_du": np.zeros((5, 7))}, lambda v: {"hydro_dv": np.zeros(SHAPE)}):
        try:
            add_user(np.zeros(SHAPE), {"hydro_w": np.ones(SHAPE)}, [mod])
        except ValueError:
            refused += 1
    return report("C2", refused == 2, "a hydro_du of the wrong shape and an unsupported key are refused",
                  "refused %d of 2" % refused)


def check_C3_constant_increment_enters_once_per_step():
    out = []
    for kind, stages in WEIGHTS.items():
        u0 = 0.
        u = u0
        for w0, w1, w2 in stages:          # a user increment G = 1 in every stage, no native tendency
            u = w0 * u0 + w1 * u + w2 * 1.
        out.append((kind, u))
    ok = all(abs(u - 1.) < 1e-15 for _, u in out)
    return report("C3", ok, "a constant user increment G per stage adds exactly G per step with every integrator, "
                  "through the stage weights w2", ", ".join("%s %.15f" % kv for kv in out))


def main():
    checks = [check_C1_sum_independent_of_module_order, check_C2_bad_results_are_refused,
              check_C3_constant_increment_enters_once_per_step]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
