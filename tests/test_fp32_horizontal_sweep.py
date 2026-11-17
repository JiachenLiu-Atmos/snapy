#!/usr/bin/env python3
"""Pair the covariance-off FP32 horizontal sweep with the FP64 reference path.

Both x2 and x3 are tested independently with pressure variation along both the
active and disabled horizontal axes, so an inactive sweep cannot add a source.
The FP64 arm takes the full-state reconstruction path, so a mistake in FP32
gauge restoration/background subtraction cannot cancel between the two arms.
"""

import argparse
import os
import sys

import torch

from snapy import kIDN, kIPR, kIV1
from test_anomaly_storage import initial_state, interior, run_steps
from test_perturbation_flux import make


NSTEP = 20
DT = 0.05
VELOCITY_TOL = 5.0e-5  # m/s; detects the half-background mutation by a wide margin
FIELD_TOL = 2.0e-5


def compare(dim):
    disabled = (1, 3 if dim == 2 else 2)
    blocks = {
        precision: make(
            True,
            precision=precision,
            gravity=False,
            disabled_dims=disabled,
            flux_covariance=False,
        )
        for precision in ("float64", "float32")
    }
    # Build a single FP64 initial condition for both precisions. This isolates
    # evolution differences from rounding in initial-condition construction.
    reference = blocks["float64"]
    w = initial_state(reference)["hydro_w"]
    w[5:].zero_()  # dry limit isolates reconstruction from saturation adjustment
    w[kIV1 : kIV1 + 3].zero_()
    coord = reference.module("coord")
    x = coord.buffer(f"x{dim}v")
    shape = [1, 1, 1]
    shape[3 - dim] = -1
    phase = 2.0 * torch.pi * x.view(shape) / 4000.0
    w[kIPR] *= 1.0 + 0.01 * torch.cos(phase)
    # A uniform disabled direction would hide a spurious background force if
    # reconstruct_sweep stopped checking disable_flux_x2/disable_flux_x3.
    other_dim = disabled[1]
    other_shape = [1, 1, 1]
    other_shape[3 - other_dim] = -1
    other_x = coord.buffer(f"x{other_dim}v").view(other_shape)
    w[kIPR] *= 1.0 + 0.01 * torch.cos(2.0 * torch.pi * other_x / 4000.0)
    velocity_row = kIV1 + dim - 1
    w[velocity_row] = 0.5 * torch.sin(phase)
    initial_velocity = interior(reference, w)[velocity_row].clone()

    final = {}
    for precision, block in blocks.items():
        hydro = block.module("hydro")
        assert hydro.options.flux_covariance() is False
        assert not getattr(hydro.options, f"disable_flux_x{dim}")()
        assert getattr(hydro.options, f"disable_flux_x{other_dim}")()
        state, _ = block.initialize({"hydro_w": w.clone()})
        assert state["hydro_u"].dtype == getattr(torch, precision)
        assert state["hydro_u"].device.type == os.environ["SNAP_TEST_DEVICE"]
        run_steps(block, state, NSTEP, dt=DT)
        hydro_eos = block.module("hydro.eos")
        hydro_eos.compute("U'->W", [state["hydro_u"], state["hydro_w"]])
        final[precision] = interior(block, state["hydro_w"]).clone()
        assert torch.isfinite(final[precision]).all()
        assert (final[precision][kIDN] > 0).all()

    expected, actual = final["float64"], final["float32"]
    activity = (expected[velocity_row] - initial_velocity).abs().max().item()
    assert activity > 1.0e-3, (dim, "horizontal evolution was not exercised", activity)
    velocity_error = (actual[velocity_row] - expected[velocity_row]).abs().max().item()
    scale = expected.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    field_error = ((actual - expected).abs() / scale).max().item()
    print(
        f"x{dim} covariance=false, {NSTEP} steps: "
        f"velocity error={velocity_error:.6e} m/s, "
        f"scaled primitive error={field_error:.6e}, activity={activity:.6e} m/s",
        flush=True,
    )
    return velocity_error <= VELOCITY_TOL and field_error <= FIELD_TOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and (
        os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()
    ):
        print("SKIP: CUDA unavailable")
        return 125
    os.environ["SNAP_TEST_DEVICE"] = args.device
    torch.set_num_threads(1)
    passed = [compare(dim) for dim in (2, 3)]
    assert all(passed), f"FP32 covariance-off horizontal sweep failed: x2/x3={passed}"
    print(f"PASS: FP32 covariance-off x2/x3 sweeps agree with FP64 on {args.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
