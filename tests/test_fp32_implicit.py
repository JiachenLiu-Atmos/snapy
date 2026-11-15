#!/usr/bin/env python3
"""FP32 VIC accuracy at large dt, FP64 recovery, and rejection without it."""

import torch
from snapy import balance_column, kIDN, kIPR, kIV1
from test_perturbation_flux import make
from test_anomaly_storage import initial_state, run_steps, interior


def column(precision, gravity=True, fallback=True):
    b = make(
        True,
        precision="float32",
        scheme=9,
        one_column=True,
        gravity=gravity,
        implicit_precision=precision,
    )
    b.module("hydro.icorr").options.fallback_fp64(fallback)
    w = initial_state(b)["hydro_w"]
    w[1:4].zero_()
    w[5:].zero_()
    z = b.module("coord").buffer("x1v")
    if gravity:
        w[..., 3:-3] = balance_column(
            w[..., 3:-3].contiguous(),
            b.module("coord").buffer("dx1f")[3:-3].contiguous(),
            10.44,
            rtol=1.0e-13,
            max_iter=400,
        )[0]
        w[kIV1] = 1.0e-3 * torch.sin(torch.pi * z / 1.0e4)
    else:
        w[kIDN].fill_(1.0)
        w[kIPR].fill_(1.0e5)
        w[kIV1] = 1.0e-10 * torch.sin(torch.pi * z / 1.0e4)
    return b, b.initialize({"hydro_w": w})[0]


def main():
    torch.set_num_threads(1)
    for dt in (1.0, 10.0, 100.0, 1000.0):
        states = []
        for precision in ("float32", "float64"):
            b, v = column(precision)
            run_steps(b, v, 3, dt=dt)
            assert b.module("hydro.icorr").buffer("fp64_fallbacks").item() == 0
            expected = torch.float32 if precision == "float32" else torch.float64
            assert b.module("hydro.icorr").buffer("a").dtype == expected
            states.append(interior(b, b.total_u(v)))
        scale = states[1].abs().amax((1, 2, 3), keepdim=True).clamp_min(1.0)
        error = ((states[0] - states[1]).abs() / scale).max().item()
        print(f"VIC dt={dt}: FP32/FP64 difference={error:.3e}", flush=True)
        assert error < 2.0e-6
    # Deliberately ill-conditioned regularization: FP32 loses pivots in the
    # near-null acoustic mode; FP64 can resolve them. This is a recovery test,
    # not a production timestep recommendation.
    b, v = column("float32", gravity=False)
    run_steps(b, v, 1, dt=1.0e9)
    assert b.module("hydro.icorr").buffer("fp64_fallbacks").item() > 0
    assert torch.isfinite(v["hydro_u"]).all()
    b, v = column("float32", gravity=False, fallback=False)
    for stage in range(len(b.intg.stages)):
        b.forward(v, 1.0e9, stage)
    expected = b.buffer("u0").clone()
    assert b.check_redo(v) == 1
    assert torch.equal(v["hydro_u"], expected)
    print("FP64 fallback and FP32-only retry passed", flush=True)


if __name__ == "__main__":
    main()
