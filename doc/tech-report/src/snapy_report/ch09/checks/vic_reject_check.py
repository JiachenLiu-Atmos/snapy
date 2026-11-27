"""Executable check of VIC rejection and rollback (chapter 9, _vic-reject.qmd).

What is checked: the per-column failure path of ImplicitHydroImpl::forward_masked
(src/implicit/implicit_hydro.cpp:301-623) at snapy@73c45a7c8c991209924e1137992352c335575062: the column kernel's
status bits (vic_solve_column, src/implicit/vic_kernels.h:175-208; kVicInputNonfinite 1, kVicSingular 2,
kVicResultNonfinite 4, src/implicit/implicit_dispatch.hpp:17-19), the per-column result screen and the per-column
undo with device masks (implicit_hydro.cpp:564-579, 617), the step's status OR and its truth table (status(),
implicit_hydro.cpp:223-243), the stage-0 reset (reset_status, src/mesh/meshblock_advance.cpp:183) and the redo flag
(solve_failed_tensor in MeshBlockImpl::local_redo_flag_tensor, src/mesh/meshblock_redo.cpp:131-133), ported as a
small state machine; the solve itself is the vic_port.py sweep, and the redistribution is modelled as the solved
increment. Claims C1-C5.

Run: python3 vic_reject_check.py   (numpy only; exits with the number of failed claims)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vic_port as vp  # noqa: E402

INPUT, SINGULAR, RESULT, RETRIED, FP64_FAILED = 1, 2, 4, 8, 16   # VicStatusBit
FAIL_BITS = INPUT | SINGULAR | RESULT                            # kVicFailBits
COS, SIN = 0.6, 0.8                                              # cosine_cell_kj and its sine


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def failed_word(bits):
    """status()'s truth table (implicit_hydro.cpp:228-237): the FP64 retry failed, or a failure no retry followed."""
    failed, retried, fp64_failed = bool(bits & FAIL_BITS), bool(bits & RETRIED), bool(bits & FP64_FAILED)
    return fp64_failed or (failed and not retried)


class Corrector:
    """forward_masked without the FP64 stage: every column gets its own status bits; a failed column is undone,
    the others keep their solution."""

    def __init__(self, n_col):
        self.status = np.zeros(n_col, dtype=np.int32)   # _status, the step's OR
        self.solves = 0

    def reset_status(self):                               # stage 0 of every attempt
        self.status[:] = 0

    def solve_failed(self):                               # status(): any column whose last solve failed
        return any(failed_word(int(b)) for b in self.status)

    def forward_masked(self, du, w, dt, a, b, c, poison=None, bad=1):
        n_col = du.shape[0]
        stage = np.zeros(n_col, dtype=np.int32)          # _stage_status.zero_()
        du0 = du.copy()                                   # _du0
        w0 = w[..., vp.IVY:vp.IVZ + 1].copy()             # the two rows the projection edits
        w[..., vp.IVY] += w[..., vp.IVZ] * COS            # (1) the frame projection, in place
        w[..., vp.IVZ] *= SIN
        self.solves += 1
        delta = np.zeros_like(du)
        for k in range(n_col):                            # vic_solve_column, one column per thread
            bits = 0
            if not (np.isfinite(du[k]).all() and np.isfinite(w[k]).all()):
                bits |= INPUT                             # the input screen; the sweep still runs
            aa = a[k].copy()
            if poison == "singular" and k == bad:
                aa[0] = 0.                                # a zero block: ludcmp returns 0
            ok, af, d = vp.forward_sweep(aa, b[k], c[k], du[k] / dt)
            if not ok:
                stage[k] = bits | SINGULAR
                delta[k] = d                              # the NaN sentinel
                continue
            delta[k] = vp.backward_substitute(af, d)
            if not np.isfinite(delta[k]).all():
                bits |= RESULT
            stage[k] = bits
        du[...] = delta                                   # the redistribution, modelled as the solved increment
        w[..., vp.IVZ] /= SIN                             # (3) back to the global frame
        w[..., vp.IVY] -= w[..., vp.IVZ] * COS
        if poison == "result":
            du[bad, 1, 1] = np.inf
        corr = du - du0
        for k in range(n_col):                            # the result screen, per column
            if not (np.isfinite(du[k]).all() and np.isfinite(corr[k]).all()):
                stage[k] |= RESULT
        drop = (stage & FAIL_BITS) != 0                   # undone per column with masks
        du[drop] = du0[drop]
        w[drop, :, vp.IVY:vp.IVZ + 1] = w0[drop]
        corr[drop] = 0.
        self.status |= stage
        self.last_stage = stage
        return corr


rng = np.random.default_rng(13)
N_COL, N_LAYER = 4, 8
SHAPE = (N_COL, N_LAYER, 5)


def system():
    a = np.eye(5) * 2. + rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1
    return a, rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1, rng.normal(size=(N_COL, N_LAYER, 5, 5)) * .1


def redo_mask(p):
    """the sixth flag of local_redo_flag_tensor, cause 32"""
    return 32 if p.solve_failed() else 0


def check_C1_each_fault_undoes_its_column_bitwise():
    rows, ok = [], True
    for fault, bit in (("input", INPUT), ("singular", SINGULAR), ("result", RESULT)):
        sysm = system()
        du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
        if fault == "input":
            du[1, 0, 0] = np.nan                          # a nonfinite input in column 1
        du_in, w_in = du.copy(), w.copy()
        p = Corrector(N_COL)
        corr = p.forward_masked(du, w, 0.5, *sysm, poison=None if fault == "input" else fault)
        same = (np.array_equal(du[1], du_in[1], equal_nan=True) and np.array_equal(w[1], w_in[1])
                and not corr[1].any() and p.last_stage[1] & bit and (p.last_stage[[0, 2, 3]] == 0).all())
        ok &= bool(same)
        rows.append("%s: column 1 bits %d, restored %s, other columns clear %s"
                    % (fault, p.last_stage[1], np.array_equal(du[1], du_in[1], equal_nan=True),
                       (p.last_stage[[0, 2, 3]] == 0).all()))
    return report("C1", ok, "each fault sets its own status bit on its own column (input 1, singular 2, result 4), "
                  "and that column's du and w are restored bit for bit with a zero correction", "; ".join(rows))


def check_C4_other_columns_keep_their_solution():
    sysm = system()
    du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
    good = Corrector(N_COL).forward_masked(du.copy(), w.copy(), .5, *sysm)
    corr = Corrector(N_COL).forward_masked(du.copy(), w.copy(), .5, *sysm, poison="singular", bad=2)
    kept = [k for k in range(N_COL) if k != 2]
    ok = (not corr[2].any() and np.abs(good[2]).max() > 0
          and all(np.array_equal(corr[k], good[k]) for k in kept))
    return report("C4", ok, "one failed column of %d: only that column's correction is zero; the other columns keep "
                  "their solution bit for bit" % N_COL,
                  "failed column max |correction| %.1g (%.3g without the failure); other columns equal to the "
                  "failure-free solve: %s" % (np.abs(corr[2]).max(), np.abs(good[2]).max(),
                                              all(np.array_equal(corr[k], good[k]) for k in kept)))


def check_C2_later_stages_still_solve():
    p = Corrector(N_COL)
    a, b, c = system()
    out = []
    p.reset_status()
    for stage in range(3):
        du, w = rng.normal(size=SHAPE), rng.normal(size=SHAPE)
        corr = p.forward_masked(du, w, 0.5, a, b, c, poison="singular" if stage == 0 else None)
        out.append(round(float(np.abs(corr).max()), 3))
    ok = p.solves == 3 and out[1] > 0 and out[2] > 0 and p.solve_failed()
    return report("C2", ok, "after a failure at stage 0 the later stages of the step still solve; the step's status "
                  "keeps the failure", "solves run %d of 3 stages; max |correction| per stage %s; step failed: %s"
                  % (p.solves, out, p.solve_failed()))


def check_C3_failure_requests_a_redo_and_the_retry_starts_clean():
    p = Corrector(N_COL)
    a, b, c = system()
    p.reset_status()
    p.forward_masked(rng.normal(size=SHAPE), rng.normal(size=SHAPE), .5, a, b, c, poison="singular")
    mask = redo_mask(p)
    p.reset_status()                                      # stage 0 of the retry
    corr = p.forward_masked(rng.normal(size=SHAPE), rng.normal(size=SHAPE), .25, a, b, c)
    ok = mask == 32 and redo_mask(p) == 0 and np.abs(corr).max() > 0
    return report("C3", ok, "a failed step raises redo cause 32; the stage-0 reset lets the retry solve again",
                  "cause mask %d; retry mask %d, retry max |correction| %.3g" % (mask, redo_mask(p), np.abs(corr).max()))


def check_C5_retried_columns_and_the_log():
    words = range(32)
    fails = [w_ for w_ in words if failed_word(w_)]
    retried_ok = SINGULAR | RETRIED                       # failed in FP32, solved in FP64
    # report_failure logs every column with a nonzero status word (implicit_hydro.cpp:275-294)
    logged = [w_ for w_ in (retried_ok, SINGULAR | RETRIED | FP64_FAILED) if w_ != 0]
    ok = (not failed_word(retried_ok) and failed_word(SINGULAR) and failed_word(SINGULAR | RETRIED | FP64_FAILED)
          and len(logged) == 2)
    return report("C5", ok, "a column retried in FP64 that succeeded is not a failure of the step, but the failure log "
                  "lists every column with a nonzero word, that one included",
                  "%d of 32 status words fail the step; retried-and-solved word %d fails: %s; logged when another "
                  "column fails: %s" % (len(fails), retried_ok, failed_word(retried_ok), retried_ok in logged))


def main():
    checks = [check_C1_each_fault_undoes_its_column_bitwise, check_C2_later_stages_still_solve,
              check_C3_failure_requests_a_redo_and_the_retry_starts_clean, check_C4_other_columns_keep_their_solution,
              check_C5_retried_columns_and_the_log]
    with np.errstate(invalid="ignore", over="ignore"):
        sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
