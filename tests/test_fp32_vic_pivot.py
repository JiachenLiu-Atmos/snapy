#!/usr/bin/env python3
"""FP32 vertical implicit solve accuracy on a moving 2-D moist column set.

implicit-precision auto (FP32 solve) vs float64 (FP64 solve), both with FP32
dynamics, on the moist wind state of test_anomaly_storage.initial_state (32
columns, Cartesian face gravity work, so the Woodbury corners are on). With
ludcmp's implicit diagonal-block pivot scaling the FP32 solve drifts to a w gap
of 1.0e-3 after 100 steps; with block-row scaling and partial pivoting it stays
at 2.9e-6, the FP32-dynamics level. The E+P budget drift under auto must also
stay at the FP64-solve level.
"""
import argparse
import torch
from test_perturbation_flux import make
from test_anomaly_storage import initial_state, interior, run_steps
from snapy import kIDN, kIPR, kIV1

W_GAP = 5.0e-5  # m/s, max |w_auto - w_fp64|


def energy_plus_potential(b, bv, g=10.44):
    """E + P as test_conservation_budget.cpp books face work (Cartesian)."""
    u = interior(b, b.total_u(bv)).double()
    co = b.module("coord")
    x1f = co.buffer("x1f").double()[3:-3]
    z = co.buffer("x1v").double()[3:-3]
    vol = interior(b, co.cell_volume().unsqueeze(0).expand_as(b.total_u(bv)))[0].double()
    rho = u[kIDN] + u[5:].sum(0)
    slope = torch.gradient(rho, spacing=(z,), dim=-1, edge_order=2)[0]
    pe = rho * g * (z - x1f[0]) + g * (x1f[1:] - x1f[:-1]) ** 2 / 12.0 * slope
    return ((u[kIPR] + pe) * vol).sum().item()


def run(implicit_precision, steps, dt):
    b = make(True, scheme=9, anomaly=True, precision="float32",
             implicit_precision=implicit_precision)
    bv, _ = b.initialize({"hydro_w": initial_state(b)["hydro_w"]})
    ep0 = energy_plus_potential(b, bv)
    run_steps(b, bv, steps, dt=dt)
    icorr = b.module("hydro.icorr")
    assert icorr.buffer("fp64_fallbacks").item() == 0
    w = interior(b, bv["hydro_w"])[kIV1].double()
    return w, (energy_plus_potential(b, bv) - ep0) / abs(ep0), icorr.buffer("a").dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    w32, ep32, a32 = run("auto", args.steps, args.dt)
    w64, ep64, a64 = run("float64", args.steps, args.dt)
    assert a32 == torch.float32 and a64 == torch.float64, (a32, a64)
    gap = (w32 - w64).abs().max().item()
    print(f"steps={args.steps} dt={args.dt}: w gap auto vs FP64 solve={gap:.3e} "
          f"(limit {W_GAP:.1e}); E+P drift auto={ep32:.3e}, FP64 solve={ep64:.3e}",
          flush=True)
    assert gap < W_GAP, gap
    assert abs(ep32) < 2.0 * abs(ep64) + 1.0e-12, (ep32, ep64)
    print("FP32 VIC pivoting passed")


if __name__ == "__main__":
    main()
