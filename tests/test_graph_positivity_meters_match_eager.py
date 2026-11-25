#!/usr/bin/env python3
"""V3: the block's EfStatus meters under cuda-graph equal eager, bitwise.

The capture of a graph segment runs the segment eagerly three times (the
warm-ups) before it records it, and restores the state the StageExecutor saved
(StageExecutor::save_list) after them. A meter the segment adds to that is not
on the save list keeps the warm-ups' additions: every capture counts the
segment four times.

The moist column of test_graph_chemistry_slot.py (H2O, H2O(l)) with
flux-covariance on, so the tracer flux-positivity limiter
(flux_positivity.hpp) runs inside the captured hydro forward (the EOS limiter,
the other switch, keeps a case eager) and adds to the block's EfStatus meters
at every stage: hydro.positivity_hits, positivity_severe, positivity_min,
lim_cut and lim_flux (and hydro.conversion_trips if the deck has one), and the
scalar's positivity_hits where the scalar has one. A cuda-graph block (not
strict) and an eager block step the same dt schedule (1.0, then 0.5, a rebind
that recaptures, an in-place edit) in FP64 and FP32. After every step, from
the first (the capture) on, every meter, check_redo (the limiter marks and the
vapor-failure count), hydro_u, hydro_w, scalar_s and scalar_r must match
bitwise, and so must the budget's cycle line (print_cycle_info: limcut,
thetamin, thetasevere). The case does something: the eager lim_flux is
positive after step 1, the eager positivity_hits and lim_cut (and in FP64
positivity_severe) are positive at the end, and the graph block captured and
replayed with no eager segment. On CPU the cuda-graph setting runs eager and
must equal eager.

A second case, co4, adds conversion order 4 and one conserved tracer
(unlimited cp5), and starts with an isothermal dip (rho and p x 0.3) next to
the lower x1 wall: the wall's one-sided ghosts are not positive there, so the
conversion falls back (hydro.conversion_trips, a module buffer outside the
EfStatus) and the scalar's positivity census (scalar.positivity_hits) counts.
dt 0.05 (0.025 mid-run). Its eager conversion_trips and scalar
positivity_hits must be positive at the end too.

RED at 939f7928: after the first step (three captures) the graph block's
positivity_hits, positivity_severe, positivity_min, lim_cut and lim_flux
differ from eager's in both cases, and in co4 also conversion_trips (96 vs
24) and scalar.positivity_hits; the warm-ups of each capture are counted,
and they ran on states past the stage's. At e119b3f9 (the EfStatus slots
saved) only conversion_trips still differs.

  python test_graph_positivity_meters_match_eager.py
"""
import ctypes
import os
import sys
import tempfile

import torch

from snapy import kICY, kIDN, kIPR, kIV1, kIV2

from cuda_build import build_has_cuda, skip
from test_graph_chemistry_slot import (QVAPOR, assert_equal, build, config,
                                       forward_both, moist_state)

METERS = ("positivity_hits", "positivity_severe", "positivity_min",
          "lim_cut", "lim_flux", "conversion_trips")


DT = {"moist": 1.0, "co4": 0.05}  # the step's dt (halved mid-run)


def make(precision, device, execution="eager", case="moist"):
    cfg = config(precision, execution)
    cfg["dynamics"]["flux-covariance"] = True  # runs the positivity limiter
    cfg["integration"]["ncycle_out"] = 1  # the budget's cycle line, every cycle
    if case == "co4":
        # conversion order 4 (its fallback count, hydro.conversion_trips) and
        # one conserved tracer, unlimited cp5, so the scalar's positivity
        # census runs too (flux-covariance; an upper-bound needs the EOS
        # limiter, which keeps a case eager)
        cfg["dynamics"]["conversion-order"] = 4
        cfg["scalar"] = {"nvar": 1, "names": ["tracer"],
                         "reconstruct": {"type": "cp5", "shock": False},
                         "riemann-solver": {"type": "upwind"}}
    block = build(cfg, device)
    return block, (co4_state(block) if case == "co4" else moist_state(block))


def co4_state(block):
    """moist_state's column with an isothermal dip (rho and p x 0.3) in the
    first two cells above the lower x1 wall, so the wall's one-sided ghosts
    (Conversion::fill_walls) are not positive there and the conversion keeps
    the old ghosts, a trip; and a tracer top-hat in x2."""
    thermo = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")
    u = block.module("hydro").buffer("D").double().clone().zero_()
    z = block.module("coord").buffer("x1v").view(1, 1, -1).expand_as(u[kIDN])
    x2 = block.module("coord").buffer("x2v").view(1, -1, 1).expand_as(u[kIDN])
    g, t0, gam, p0, rd = 10.44, 300.0, 6.5e-3, 1.0e5, 287.0
    temp = (t0 - gam * z).clamp_min(200.0)
    pres = p0 * (temp / t0) ** (g / (rd * gam))
    rho = pres / (rd * temp)
    y = torch.zeros_like(u[kICY:])
    y[0] = QVAPOR
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    u[kIDN] = rho * (1.0 - y.sum(0))
    u[kICY:] = rho * y
    u[kIPR] = ie
    w = eos.compute("U->W", [u.to(block.module("hydro").buffer("D").dtype)])
    w[kIV2] = 0.5 * torch.sin(2.0 * torch.pi * x2 / 4.0e3)
    w[kIV1] = 0.2 * torch.cos(2.0 * torch.pi * x2 / 4.0e3) * torch.sin(torch.pi * z / 1.0e4)
    il = 3  # the first interior x1 cell (nghost 3)
    w[kIDN, :, 8:10, il:il + 2] *= 0.3
    w[kIPR, :, 8:10, il:il + 2] *= 0.3
    ratio = (((x2 / 4.0e3) % 1.0) < 0.5).to(w.dtype).unsqueeze(0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return state


def meters(block):
    out = {}
    for module in ("hydro", "scalar"):
        bufs = dict(block.module(module).named_buffers())
        for name in METERS:
            if name in bufs:
                out[module + "." + name] = bufs[name].clone()
    return out


LIBC = ctypes.CDLL(None)


def cycle_line(block, state, dt):
    """print_cycle_info's line, read from the C++ stdout (fd 1)."""
    sys.stdout.flush()
    LIBC.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+") as f:
        os.dup2(f.fileno(), 1)
        try:
            block.print_cycle_info(state, 0.0, dt)
            LIBC.fflush(None)  # std::cout shares the C stdout buffer
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        return f.read()


def assert_meters(pair, dt, where):
    (block, state), (ref, expected) = pair
    mg, me = meters(block), meters(ref)
    assert mg.keys() == me.keys(), (where, mg.keys(), me.keys())
    bad = ["%s graph %s eager %s" % (k, mg[k].tolist(), me[k].tolist())
           for k in mg if not torch.equal(mg[k], me[k])]
    assert not bad, (where, bad)
    lg, le = cycle_line(block, state, dt), cycle_line(ref, expected, dt)
    assert lg == le, (where, "budget line", lg, le)
    assert "limcut=" in le, (where, "the budget line has no limcut", le)
    return me


def run(precision, device, nsteps, case="moist"):
    pair = (make(precision, device, "cuda-graph", case),
            make(precision, device, case=case))
    (block, state), (ref, expected) = pair
    for step in range(nsteps):
        dt = DT[case] * (0.5 if nsteps // 3 <= step < 2 * nsteps // 3 else 1.0)
        if step == nsteps - 3:  # a rebind between steps recaptures
            for fields in (state, expected):
                for k in ("hydro_u", "scalar_s"):
                    if k in fields:
                        fields[k] = fields[k].clone()
        if step == nsteps - 2:  # an in-place edit reaches the captured inputs
            for fields in (state, expected):
                fields["hydro_u"][kIPR, 0, 4, 10] += 1.0e-4
        forward_both(pair, dt)
        ra, rb = block.check_redo(state), ref.check_redo(expected)
        assert ra == rb == 0, (precision, step, "check_redo", ra, rb)
        assert_equal(state, expected, (precision, step))
        me = assert_meters(pair, dt, (case, precision, device.type, step))
        if step == 0:
            assert me["hydro.lim_flux"].item() > 0, (
                case, precision, "the limiter was offered no flux; vacuous", me)
            print(case, precision, device.type, "eager meters after step 1:",
                  {k: v.tolist() for k, v in me.items()})
    # positivity_severe is above FP32 round-off only in FP64 here
    nonzero = ["hydro.positivity_hits", "hydro.lim_cut"]
    if precision == "float64":
        nonzero.append("hydro.positivity_severe")
    if case == "co4":
        nonzero += ["hydro.conversion_trips", "scalar.positivity_hits"]
    for k in nonzero:
        assert k in me and me[k].item() > 0, (
            case, precision, "vacuous: eager %s is zero" % k, me)
    print(case, precision, device.type, "eager meters at the end:",
          {k: v.tolist() for k, v in me.items()})
    return block.execution_report()


def main():
    torch.set_num_threads(1)
    for case in ("moist", "co4"):
        for precision in ("float64", "float32"):
            report = run(precision, torch.device("cpu"), 3 if case == "moist" else 9, case)
            assert report["captures"] == 0 and report["replays"] == 0, report
            print(case, precision, "CPU: cuda-graph runs eager and its meters equal eager:",
                  report)
    if not build_has_cuda():
        skip("CUDA unavailable (graph part)")
    failed = []
    for case in ("moist", "co4"):
        for precision in ("float64", "float32"):
            try:
                report = run(precision, torch.device("cuda"), 9, case)
                assert report["captures"] > 0 and report["replays"] > 0, report
                assert report["eager"] == 0, report
                print(case, precision, "graph meters and budget line equal eager:", report)
            except (AssertionError, RuntimeError) as e:
                failed.append("%s %s: %s" % (case, precision, str(e).strip()))
    for f in failed:
        print("FAIL", f)
    print("PASSED" if not failed else "FAILED (%d)" % len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
