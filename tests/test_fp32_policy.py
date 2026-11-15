#!/usr/bin/env python3
"""FP32 policy boundaries, numerical agreement, rest, restart and rollback."""

import glob
import os
import tempfile
import torch
from snapy import balance_column, kIPR, kIV1
from test_perturbation_flux import make
from test_anomaly_storage import initial_state, interior, run_steps


def init(block, rest=False):
    w = initial_state(block)["hydro_w"]
    if rest:
        w[1:4].zero_()
        w[5:].zero_()
        dx = block.module("coord").buffer("dx1f")[3:-3].contiguous()
        w[..., 3:-3] = balance_column(
            w[:, :1, :1, 3:-3].contiguous(), dx, 10.44, rtol=1.0e-13, max_iter=400
        )[0]
    return block.initialize({"hydro_w": w})[0]


def main():
    torch.set_num_threads(1)
    block = make(True, precision="float32")
    rounded = initial_state(block)["hydro_w"].float()
    bv, _ = block.initialize({"hydro_w": rounded})
    assert bv["hydro_w"].dtype == torch.float64
    assert bv["hydro_u"].dtype == torch.float32
    run_steps(block, bv, 1, dt=0.01)
    for order, stretched, outflow in (
        (2, False, False),
        (4, False, False),
        (2, True, False),
        (4, True, False),
        (2, False, True),
    ):
        states = []
        for precision in ("float32", "float64"):
            block = make(
                True, order=order, precision=precision, stretched=stretched, outflow=outflow
            )
            bv = init(block)
            run_steps(block, bv, 10, dt=0.05)
            states.append(interior(block, block.total_u(bv)))
        scale = states[1].abs().amax((1, 2, 3), keepdim=True).clamp_min(1.0)
        error = ((states[0] - states[1]).abs() / scale).max().item()
        print(
            f"order={order}, stretched={stretched}, outflow={outflow}: error={error:.3e}",
            flush=True,
        )
        assert error < 2.0e-5
    for scheme in (0, 9):
        block = make(True, precision="float32", scheme=scheme)
        grid = block.module("coord").buffer("x1f").clone()
        block.to(torch.float32)
        assert torch.equal(grid, block.module("coord").buffer("x1f"))
        assert block.module("coord").buffer("x1f").dtype == torch.float64
        bv = init(block, rest=True)
        run_steps(block, bv, 100, dt=0.05)
        noise = interior(block, bv["hydro_u"])[kIV1].abs().max().item()
        print(f"rest scheme={scheme}: momentum={noise:.3e}", flush=True)
        assert noise < 2.0e-8
        # Retry restores the exact FP32 input, including the dtype.
        saved = bv["hydro_u"].clone()
        for stage in range(len(block.intg.stages)):
            block.forward(bv, 0.05, stage)
            if stage == 0:
                expected = block.buffer("u0").clone()
                assert torch.equal(interior(block, expected), interior(block, saved))
        bv["hydro_u"][kIPR, 3, 3, 10] = float("nan")
        assert block.check_redo(bv) == 1
        assert bv["hydro_u"].dtype == torch.float32
        assert torch.equal(bv["hydro_u"], expected)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            outputs = [{"type": "restart", "dt": 1.0e9}]
            block = make(True, precision="float32", scheme=9, outputs=outputs)
            bv = init(block)
            run_steps(block, bv, 3, dt=0.05)
            block.make_outputs(bv, 0.15, True)
            (path,) = glob.glob("*.final.restart")
            run_steps(block, bv, 3, dt=0.05)
            resume = make(True, precision="float32", scheme=9, outputs=outputs)
            rv, _ = resume.initialize_from_restart(path)
            assert rv["hydro_u"].dtype == torch.float32
            run_steps(resume, rv, 3, dt=0.05)
            assert torch.equal(bv["hydro_u"], rv["hydro_u"])
            print("FP32 restart: bit-identical", flush=True)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
