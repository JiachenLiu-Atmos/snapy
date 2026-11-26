#!/usr/bin/env python3
"""A rejected step leaves no count in the block's EfStatus meters.

The census counts accepted steps only (flux_register.hpp, RegisterMonitor::
rollback). The positivity limiter's meters (hydro.positivity_hits,
positivity_severe, positivity_min, lim_cut, lim_flux, conversion_trips and
scalar.positivity_hits; flux_positivity.cpp census_ and post_) are slots of
the block's EfStatus, so a redo must restore them with the register.

Two eager blocks (the moist column of test_graph_positivity_meters_match_eager,
flux-covariance on, so the limiter runs and lim_flux > 0) step the same dt.
Block B is forced into one redo: a NaN density planted in hydro_u after its
first attempt (limiter off, so only the detector sees it). It then repeats
the step at the same dt, which starts from the restored state, so the two
states are bitwise equal. Its meters must equal A's, bitwise, after the
accepted step and after a second one.

RED at 73c45a7c: apply_redo (meshblock_redo.cpp) rolls back only the flux
register; B's lim_flux is twice A's (the rejected attempt counted too).

  python test_redo_positivity_meters.py
"""
import sys

import torch

from snapy import kIDN

from test_graph_chemistry_slot import assert_equal
from test_graph_positivity_meters_match_eager import DT, make, meters


def step(block, state, dt):
    for stage in range(len(block.intg.stages)):
        block.forward(state, dt, stage)


def compare(a, b, where):
    ma, mb = meters(a), meters(b)
    assert ma.keys() == mb.keys(), (where, ma.keys(), mb.keys())
    bad = ["%s: no redo %s, after a redo %s" % (k, ma[k].tolist(), mb[k].tolist())
           for k in ma if not torch.equal(ma[k], mb[k])]
    assert not bad, (where, bad)
    return ma


def run(precision):
    (a, sa), (b, sb) = make(precision, torch.device("cpu")), make(precision, torch.device("cpu"))
    dt = DT["moist"]
    step(a, sa, dt)
    assert a.check_redo(sa) == 0, (precision, "the reference step was rejected")

    step(b, sb, dt)
    u = sb["hydro_u"]
    k, j, i = u.size(1) // 2, u.size(2) // 2, u.size(3) // 2
    u[kIDN, k, j, i] = float("nan")
    assert b.check_redo(sb) == 1, (precision, "the planted NaN did not force a redo")
    step(b, sb, dt)
    assert b.check_redo(sb) == 0, (precision, "the repeated step was rejected")

    assert_equal(sb, sa, (precision, "state after the redo"))
    me = compare(a, b, (precision, "after the accepted step"))
    assert me["hydro.lim_flux"].item() > 0, (precision, "vacuous: lim_flux is zero", me)

    for blk, st in ((a, sa), (b, sb)):
        step(blk, st, dt)
        assert blk.check_redo(st) == 0, (precision, "second step rejected")
    assert_equal(sb, sa, (precision, "state after the second step"))
    me = compare(a, b, (precision, "after the second step"))
    print(precision, "meters after a redo equal those without:",
          {k: v.tolist() for k, v in me.items()})


def main():
    torch.set_num_threads(1)
    for precision in ("float64", "float32"):
        run(precision)
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
