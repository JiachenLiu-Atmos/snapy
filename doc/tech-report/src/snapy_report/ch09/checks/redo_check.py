"""Executable check of step acceptance and redo (chapter 9, _redo.qmd).

What is checked: the cause encoding of MeshBlockImpl::decide_retry (src/mesh/meshblock_redo.cpp:168-198),
the retry and termination logic of MeshBlockImpl::apply_redo (:32-88) and the redo factor of
MeshBlockImpl::global_time_step (src/mesh/meshblock_advance.cpp:71) at
snapy@73c45a7c8c991209924e1137992352c335575062, with the default max_redo = 5 of
pyharp@4721715855e937c1e8b218e964c0655f46e56e29 (src/integrator/integrator.hpp:58), ported line for line.
C5 and C6 port MeshBlockImpl::read_redo_flags (meshblock_redo.cpp:147-162): the seventh entry, the step's
vapour-repair failures, and the eighth, the heat-capacity latch, join the limiter flag; a ninth entry would set
the seventh flag (bit 64), but local_redo_flag_tensor (:90-145) stacks eight. Claims C1-C6.

Run: python3 redo_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np

NAMES = ["floor", "clamp", "limiter", "nan", "saturation", "vic-solve", "seventh"]


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def reduce_redo_flags(per_rank):  # decide_retry, meshblock_redo.cpp:187-197: MAX over ranks, then a bitmask
    flag = np.max(np.array(per_rank, dtype=float), axis=0)
    return sum(int(flag[i] > 0.) << i for i in range(len(flag)))


class Block:
    """the state apply_redo touches: current_redo, a restored flag, the decision (meshblock_redo.cpp:32-88)"""

    def __init__(self, max_redo=5):
        self.max_redo, self.current_redo, self.restored = max_redo, 0, 0

    def apply_redo(self, causes):
        if causes:
            self.current_redo += 1
            if self.current_redo > self.max_redo and not (causes & 32):
                return -1                      # terminate, state not restored
            self.restored += 1                 # flux register rolled back, hydro_u <- u0, splits undone, primitives recomputed
            if self.current_redo > self.max_redo:
                return -1                      # VIC failure: terminate after the restore
            return 1
        self.current_redo = 0
        return 0


def dt_factor(current_redo):  # meshblock_advance.cpp:71, the factor in front of cfl * dt
    return 2. ** -current_redo


def check_C1_cause_bitmask():
    ranks = [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
    causes = reduce_redo_flags(ranks)
    names = [n for i, n in enumerate(NAMES) if causes >> i & 1]
    ok = causes == 1 + 4 + 32 and names == ["floor", "limiter", "vic-solve"]
    return report("C1", ok, "causes are ORed over ranks into bits 1, 2, 4, 8, 16, 32",
                  "rank flags -> causes = %d (%s); every rank sees the same mask" % (causes, " ".join(names)))


def check_C2_retry_schedule():
    b = Block()
    factors, decisions = [], []
    while True:
        factors.append(dt_factor(b.current_redo))
        d = b.apply_redo(1)
        decisions.append(d)
        if d < 0:
            break
    ok = (len(factors) == 6 and factors[-1] == 1. / 32. and decisions[-1] == -1
          and decisions[:-1] == [1] * 5)
    return report("C2", ok, "a step that always fails is tried 1 + max_redo times, the last at dt/32, then the run stops",
                  "dt factors %s; decisions %s" % (", ".join("%g" % f for f in factors), decisions))


def check_C3_acceptance_resets():
    b = Block()
    for _ in range(3):
        b.apply_redo(4)
    d = b.apply_redo(0)
    ok = d == 0 and b.current_redo == 0 and dt_factor(b.current_redo) == 1.
    return report("C3", ok, "an accepted step resets the redo count, so the next step starts at the full bound",
                  "after 3 redos and an acceptance: current_redo = %d, factor = %g" % (b.current_redo, dt_factor(0)))


def check_C4_vic_failure_restores_before_stopping():
    a, v = Block(), Block()
    for _ in range(6):
        ra, rv = a.apply_redo(1), v.apply_redo(32)
    ok = ra == -1 and rv == -1 and a.restored == 5 and v.restored == 6
    return report("C4", ok, "past max_redo a VIC solve failure (32) still restores the step; other causes stop without it",
                  "restores: floor-only %d, vic-solve %d, of 6 attempts" % (a.restored, v.restored))


def read_redo_flags(entries):  # meshblock_redo.cpp:147-162: seven flags from the device entries
    out = [a > 0. for a in entries[:6]] + [False]
    if len(entries) > 6 and entries[6] > 0.:   # vapour-repair failures
        out[2] = True
    if len(entries) > 7 and entries[7] > 0.:   # heat-capacity latch
        out[2] = True
    if len(entries) > 8 and entries[8] > 0.:   # a ninth entry, never stacked at the pin
        out[6] = True
    return out


def check_C5_vapour_failure_is_a_limiter_redo():
    alone = reduce_redo_flags([read_redo_flags([0, 0, 0, 0, 0, 0, 1])])
    with_floor = reduce_redo_flags([read_redo_flags([1, 0, 0, 0, 0, 0, 1]), read_redo_flags([0] * 7)])
    healthy = reduce_redo_flags([read_redo_flags([0] * 7)])
    b = Block()
    decisions = [b.apply_redo(alone), b.apply_redo(healthy)]
    ok = alone == 4 and with_floor == 5 and healthy == 0 and decisions == [1, 0]
    return report("C5", ok, "a nonzero vapour-repair count sets the limiter flag (4), so the step is redone, not raised",
                  "causes: vapour alone %d, vapour and a floor on one of two ranks %d, none %d; decisions %s"
                  % (alone, with_floor, healthy, decisions))


def check_C6_heat_capacity_latch_is_a_limiter_redo():
    heat = reduce_redo_flags([read_redo_flags([0, 0, 0, 0, 0, 0, 0, 1])])
    eight = read_redo_flags([1] * 8)
    ok = heat == 4 and eight[6] is False and reduce_redo_flags([eight]) == 63
    return report("C6", ok, "the eighth entry, the heat-capacity latch, sets the limiter flag; eight entries never set bit 64",
                  "causes: heat latch alone %d, all eight entries set %d" % (heat, reduce_redo_flags([eight])))


def main():
    checks = [check_C1_cause_bitmask, check_C2_retry_schedule, check_C3_acceptance_resets,
              check_C4_vic_failure_restores_before_stopping, check_C5_vapour_failure_is_a_limiter_redo,
              check_C6_heat_capacity_latch_is_a_limiter_redo]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
