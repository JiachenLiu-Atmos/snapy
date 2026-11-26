"""Executable check of the VIC options and activation (chapter 7, _vic-options.qmd).

What is checked: ImplicitOptionsImpl::from_yaml, type() and size() (src/implicit/implicit_hydro.cpp:24-98,
src/implicit/implicit_hydro.hpp:33-44), the reset that calls type() (implicit_hydro.cpp:107-110) and the
nb1 guard (src/hydro/hydro.cpp:411-416) at snapy@e894700ff7aee30b52882e5202b16461413780b0, ported on
Python dicts in place of YAML nodes. Claims C1-C4.

Run: python3 vic_options_check.py   (numpy only; exits with the number of failed claims)
"""
import math
import sys


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


class Unsupported(Exception):
    pass


def type_of(scheme):
    names = {0: "none", 1: "vic-partial", 9: "vic-full"}
    if scheme not in names:
        raise Unsupported("Unsupported implicit scheme")
    return names[scheme]


def size_of(scheme):
    return 5 if (scheme >> 3) & 1 else 3


def from_yaml(config):
    """Returns None (explicit) or a dict of options; raises ValueError where snapy's TORCH_CHECK fires."""
    if "integration" not in config:
        return None
    intg = config["integration"]
    s = intg.get("implicit-scheme")
    if not s:                                     # absent, or 0: a true explicit spelling
        if "implicit-advection-cfl" in intg or "shear-cfl" in intg:
            raise ValueError("implicit-advection-cfl and shear-cfl bound an implicit direction")
        return None
    op = {"scheme": s}
    for key, fallback, ok in (("implicit-advection-cfl", 1.0, lambda v: v > 0.),
                              ("shear-cfl", 0.0, lambda v: v >= 0.)):
        v = float(intg.get(key, fallback))
        if not math.isfinite(v) or not ok(v):
            raise ValueError("%s = %r" % (key, v))
        op[key] = v
    return op


def build(config, nb1=1):
    """HydroImpl: picorr exists iff options hold one; reset() calls type(). The nb1 guard fires at the first
    correction (hydro.cpp:411-416); it is modelled here at construction."""
    op = from_yaml(config)
    if op is None:
        return None
    type_of(op["scheme"])
    if nb1 != 1:
        raise ValueError("implicit scheme requires nb1 = 1")
    return op


def check_C1_scheme_values():
    rows, ok = [], True
    for s in range(0, 11):
        try:
            op = build({"integration": {"implicit-scheme": s}})
            got = "explicit" if op is None else "%s, N = %d" % (type_of(s), size_of(s))
        except Unsupported:
            got = "error"
        rows.append("%d %s" % (s, got))
        want = {0: "explicit", 1: "vic-partial, N = 3", 9: "vic-full, N = 5"}.get(s, "error")
        ok &= got == want
    return report("C1", ok, "only 0 (explicit), 1 (partial, 3 x 3) and 9 (full, 5 x 5) build; any other value stops "
                  "at construction", "; ".join(rows))


def check_C2_cfl_keys():
    cases = [({}, (1.0, 0.0)), ({"implicit-advection-cfl": 4.}, (4.0, 0.0)), ({"shear-cfl": 0.5}, (1.0, 0.5))]
    ok, rows = True, []
    for extra, want in cases:
        op = from_yaml({"integration": dict({"implicit-scheme": 1}, **extra)})
        got = (op["implicit-advection-cfl"], op["shear-cfl"])
        ok &= got == want
        rows.append("%s -> %s" % (extra or "defaults", got))
    bad = 0
    for extra in ({"implicit-advection-cfl": 0.}, {"implicit-advection-cfl": float("inf")}, {"shear-cfl": -1.},
                  {"shear-cfl": float("nan")}):
        try:
            from_yaml({"integration": dict({"implicit-scheme": 1}, **extra)})
        except ValueError:
            bad += 1
    ok &= bad == 4
    return report("C2", ok, "implicit-advection-cfl defaults to 1 and must be finite and > 0; shear-cfl defaults to 0 "
                  "and must be finite and >= 0", "%s; refused %d of 4 bad values" % ("; ".join(rows), bad))


def check_C3_bounds_without_a_scheme_are_refused():
    refused = 0
    for intg in ({"implicit-advection-cfl": 2.}, {"shear-cfl": 1.}, {"implicit-scheme": 0, "shear-cfl": 1.}):
        try:
            from_yaml({"integration": intg})
        except ValueError:
            refused += 1
    plain = from_yaml({"integration": {"type": "rk3", "cfl": 0.9}}) is None and from_yaml({}) is None
    return report("C3", refused == 3 and plain, "the implicit time-step keys without an implicit scheme are an error; "
                  "a plain explicit block is accepted", "refused %d of 3; explicit block accepted: %s" % (refused, plain))


def check_C4_activation_needs_one_x1_block():
    op1 = build({"integration": {"implicit-scheme": 9}}, nb1=1)
    try:
        build({"integration": {"implicit-scheme": 9}}, nb1=2)
        refused = False
    except ValueError:
        refused = True
    explicit_nb2 = build({"integration": {"implicit-scheme": 0}}, nb1=2) is None
    ok = op1 is not None and refused and explicit_nb2
    return report("C4", ok, "a VIC run needs nb1 = 1; implicit-scheme 0 runs at any nb1",
                  "nb1 = 1 builds: %s; nb1 = 2 refused: %s; scheme 0 at nb1 = 2 explicit: %s"
                  % (op1 is not None, refused, explicit_nb2))


def main():
    checks = [check_C1_scheme_values, check_C2_cfl_keys, check_C3_bounds_without_a_scheme_are_refused,
              check_C4_activation_needs_one_x1_block]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
