"""Executable check of VIC rejection and rollback (chapter 7, _vic-reject.qmd).

What is checked: the control flow of ImplicitHydroImpl::forward_masked (src/implicit/implicit_hydro.cpp:160-529),
its latch (implicit_hydro.hpp:97-102), the stage-zero reset (src/mesh/meshblock.cpp:616-619) and the redo flag
(src/mesh/meshblock.cpp:1324-1326, MeshBlockImpl::local_redo_flag_tensor) at the code pin, ported as a small state
machine; the solve itself is the vic_port.py sweep. Claims C1-C5.

Run: python3 vic_reject_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vic_port as vp  # noqa: E402


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


GUARDS = ("inputs", "sentinel", "results", "correction")
COS, SIN = 0.6, 0.8                                        # cosine_cell_kj and its sine, one value for the block


def finite_columns(x):
    """finite_columns: every value of a column (axis 0 here) finite."""
    return np.isfinite(x).all(axis=tuple(range(1, x.ndim)))


class Corrector:
    """forward_masked at the pin with its four guards behind two screens: the inputs (du, w) and the solve are
    screened by one total after the solve (227-229, 377-386), the results and the correction by a second total at
    the end (511-526). Only a nonfinite total runs the per-column guards, in the order inputs, sentinel, results,
    correction. `poison` injects a nonfinite value after the entry copies at the sentinel, results or correction
    guard; a nonfinite input is passed in by the caller."""

    def __init__(self):
        self.solve_failed = False
        self.solves = 0
        self.guard = None                                  # the guard that refused, for the report

    def forward_masked(self, du, w, dt, a, b, c, poison=None):
        if self.solve_failed:                              # 176-180: the latch
            return np.zeros_like(du)
        du0 = du.copy()                                    # 195: _du0
        w0 = w[..., vp.IVY:vp.IVZ + 1].copy()              # 200: only the two rows the projection edits

        def reject(guard):                                 # 202-220
            self.solve_failed, self.guard = True, guard
            du[...] = du0
            w[..., vp.IVY:vp.IVZ + 1] = w0
            return np.zeros_like(du)

        screen = [du.sum(), w.sum()]                       # 227-229 (gamma is not modelled)
        w[..., vp.IVY] += w[..., vp.IVZ] * COS             # 232-233: the frame projection, in place
        w[..., vp.IVZ] *= SIN
        self.solves += 1
        delta = np.zeros_like(du)
        for k in range(du.shape[0]):                       # one column at a time, as the dispatch does
            aa = a[k].copy()
            if poison == "sentinel" and k == du.shape[0] // 2:
                aa[0] = 0.                                 # a zero block: ludcmp returns 0
            _, af, d = vp.forward_sweep(aa, b[k], c[k], du[k] / dt)
            delta[k] = vp.backward_substitute(af, d)
        screen.append(delta.sum())                         # 377
        if not np.isfinite(np.sum(screen)):                # 378-386: the first screen
            w_in = w.copy()
            w_in[..., vp.IVY:vp.IVZ + 1] = w0              # the inputs, as they were on entry
            if (~(finite_columns(du0) & finite_columns(w_in))).any():
                return reject("inputs")
            if (~finite_columns(delta)).any():
                return reject("sentinel")
        du[...] = delta                                    # the redistribution, modelled as the solved increment
        w[..., vp.IVZ] /= SIN                              # 453-454: back to the global frame
        w[..., vp.IVY] -= w[..., vp.IVZ] * COS
        if poison == "results":
            du[0, 1, 1] = np.inf
        corr = du - du0                                    # 511-512
        if poison == "correction":
            corr[0, 2, 2] = np.nan
        if not np.isfinite(du.sum() + corr.sum()):         # 514-517: the second screen
            bad_results, bad_correction = ~finite_columns(du), ~finite_columns(corr)
            if (bad_results | bad_correction).any():       # 519-525
                return reject("results" if bad_results.any() else "correction")
        return corr


rng = np.random.default_rng(13)
N_COL, N_LAYER = 4, 8
SHAPE = (N_COL, N_LAYER, 5)


def system():
    a = np.eye(5) * 2. + rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1
    return a, rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1, rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1


def check_C1_every_guard_restores_bitwise():
    rows, ok = [], True
    for g in GUARDS:
        p = Corrector()
        du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
        if g == "inputs":
            du[0, 0, 0] = np.nan                           # a nonfinite input, as the caller hands it in
        du_in, w_in = du.copy(), w.copy()
        corr = p.forward_masked(du, w, 0.5, *system(), poison=None if g == "inputs" else g)
        same = (np.array_equal(du, du_in, equal_nan=True) and np.array_equal(w, w_in) and not corr.any()
                and p.solve_failed and p.guard == g)
        ok &= same
        rows.append("%s %s by %s" % (g, "restored" if same else "NOT restored", p.guard))
    return report("C1", ok, "each of the four guards refuses its own fault, restores du and w bit for bit, returns a "
                  "zero correction and latches", ", ".join(rows))


def check_C2_latch_skips_later_stages():
    p = Corrector()
    a, b, c = system()
    out = []
    for stage in range(3):
        du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
        corr = p.forward_masked(du, w, 0.5, a, b, c, poison="sentinel" if stage == 0 else None)
        out.append(float(np.abs(corr).max()))
    ok = p.solves == 1 and out == [0., 0., 0.]
    return report("C2", ok, "after a failure at stage 0 the later stages of the step return zero without solving",
                  "solves run %d of 3 stages; max |correction| per stage %s" % (p.solves, out))


def redo_flags(p):
    """meshblock.cpp:860 and 1324-1326: the VIC failure is the sixth flag of local_redo_flag_tensor, cause 32."""
    return 32 if p.solve_failed else 0


def check_C3_failure_requests_a_redo_and_the_retry_starts_clean():
    p = Corrector()
    a, b, c = system()
    p.forward_masked(rng.normal(size=SHAPE), rng.normal(size=SHAPE), .5, a, b, c, poison="sentinel")
    mask = redo_flags(p)
    p.solve_failed = False                                 # meshblock.cpp:616-619 at stage 0 of the retry
    corr = p.forward_masked(rng.normal(size=SHAPE), rng.normal(size=SHAPE), .25, a, b, c)
    ok = mask == 32 and redo_flags(p) == 0 and np.abs(corr).max() > 0
    return report("C3", ok, "a failed step raises redo cause 32; the stage-0 reset lets the retry solve again",
                  "cause mask %d; retry mask %d, retry max |correction| %.3g" % (mask, redo_flags(p), np.abs(corr).max()))


def check_C4_one_bad_column_rolls_back_the_block():
    p = Corrector()
    a, b, c = system()
    du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
    du_in = du.copy()
    good = Corrector().forward_masked(du.copy(), w.copy(), .5, a, b, c)
    corr = p.forward_masked(du, w, .5, a, b, c, poison="sentinel")
    ok = np.abs(good).max() > 0 and not corr.any() and np.array_equal(du, du_in)
    return report("C4", ok, "one failed column of %d returns a zero correction for every column of the block; "
                  "no column is partly updated" % N_COL,
                  "max |correction| without the failure %.3g, with it %.1g; du unchanged: %s"
                  % (np.abs(good).max(), np.abs(corr).max(), np.array_equal(du, du_in)))


def check_C5_screen_overflow_is_not_a_refusal():
    p = Corrector()
    a, b, c = system()
    du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
    w[:, :2, vp.IDN] = 1e308                               # finite values whose sum overflows
    total = w.sum()
    corr = p.forward_masked(du, w, .5, a, b, c)
    ok = not np.isfinite(total) and not p.solve_failed and np.isfinite(corr).all() and np.abs(corr).max() > 0
    return report("C5", ok, "a stage whose screen total overflows with every value finite is not refused: the "
                  "per-column guards decide",
                  "screen total %s; refused: %s; max |correction| %.3g" % (total, p.solve_failed, np.abs(corr).max()))


def main():
    checks = [check_C1_every_guard_restores_bitwise, check_C2_latch_skips_later_stages,
              check_C3_failure_requests_a_redo_and_the_retry_starts_clean, check_C4_one_bad_column_rolls_back_the_block,
              check_C5_screen_overflow_is_not_a_refusal]
    with np.errstate(over="ignore"):                      # C5 overflows a sum on purpose
        sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
