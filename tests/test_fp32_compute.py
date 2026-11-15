#!/usr/bin/env python3
"""M4 compute dtype and reduced moist evolution acceptance."""

import argparse
import torch
from test_perturbation_flux import make
from test_anomaly_storage import initial_state, interior, run_steps
from snapy import kIDN, kIPR, kIV1


def case(scheme, steps, sediment=True):
    block = make(True, scheme=scheme, sediment=sediment, precision="float32")
    w = initial_state(block)["hydro_w"]
    if sediment:
        w[6].fill_(0.001)
    bv, _ = block.initialize({"hydro_w": w})
    assert bv["hydro_u"].dtype == torch.float32
    assert bv["hydro_w"].dtype == torch.float64
    before = interior(block, block.total_u(bv)).clone()
    for step in range(steps):
        # A small heat source stays representable in anomaly space.
        bv["hydro_u"][kIPR, :, :, 15:18] += 1.0e-4
        run_steps(block, bv, 1, dt=0.01)
        assert bv["hydro_u"].dtype == torch.float32
        if step == 0 or (step + 1) % 100 == 0:
            buffers = dict(block.named_buffers())
            for name in (
                "u0",
                "hydro.D",
                "hydro.F1",
                "hydro.F2",
                "hydro.F3",
                "hydro.riemann.elr",
                "hydro.riemann.clr",
                "hydro.riemann.glr",
            ):
                assert buffers[name].dtype == torch.float32, (name, buffers[name].dtype)
            for name in (
                "coord.x1f",
                "coord.dx1f",
                "hydro.ref.u0",
                "hydro.ref.species0",
                "hydro.P1",
            ):
                assert buffers[name].dtype == torch.float64, (name, buffers[name].dtype)
            if scheme:
                assert buffers["hydro.icorr.a"].dtype == torch.float32
                assert buffers["hydro.icorr.fp64_fallbacks"].item() == 0
            # Every nonempty FP64 buffer must belong to a declared boundary:
            # immutable masters, total face pressure, or small diagnostics.
            prefixes = (
                "coord.",
                "hydro.ref.",
                "hydro.eos.",
                "hydro.recon1.",
                "hydro.recon23.",
                "hydro.sed.sedvel.",
            )
            diagnostics = {
                "hydro.P1",
                "hydro.positivity_min",
                "hydro.lim_cut",
                "hydro.lim_flux",
                "hydro.gwfix_d",
                "hydro.gwfix_total",
                "hydro.gwfix_pending",
                "hydro.gwfix_wall",
                "hydro.icorr.clamp_residual",
                "hydro.icorr.dry_clamp_step",
            }
            unexpected = [
                n
                for n, t in buffers.items()
                if t.numel()
                and t.dtype == torch.float64
                and not n.startswith(prefixes)
                and n not in diagnostics
            ]
            assert not unexpected, unexpected
            assert torch.isfinite(bv["hydro_u"]).all()
            print(
                f"scheme={scheme}, step={step + 1}, max momentum={bv['hydro_u'][kIV1].abs().max().item():.3e}",
                flush=True,
            )
    after = interior(block, block.total_u(bv))
    mass0 = before[kIDN].sum() + before[5:].sum()
    mass1 = after[kIDN].sum() + after[5:].sum()
    drift = abs((mass1 - mass0) / mass0).item()
    print(f"scheme={scheme}, mass drift={drift:.3e}", flush=True)
    assert drift < 2.0e-5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(1)
    original_dtype = torch.get_default_dtype()
    for scheme in (0, 9):
        case(scheme, args.steps)
    assert torch.get_default_dtype() == original_dtype
