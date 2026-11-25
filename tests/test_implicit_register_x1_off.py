#!/usr/bin/env python3
"""Audit-3 lead 4: with the x1 hydro flux off, the vertical implicit solve
still books its face transfers in the register, its owner. The six-panel
sphere of test_cubed_sphere_capture (solid-body wind, warm bubble, radial
gravity) with implicit-scheme 1 and disable-flux-x1 runs one RK3 step eagerly
on CUDA. After every stage the solve has moved mass through the x1 faces, so
on some block its kept total ("vic.mass_total" of x1, the buffer
hydro.K_vic_mass_total_1) is nonzero, and the disabled x1 total (rewritten
each stage) holds exactly its dry-mass post: row IDN of hydro.F1 equals the
kept "vic.mass_dry" part, bit for bit.

RED before: the hydro handed the solve a null register when the x1 flux was
off, so nothing was kept or posted and the x1 mass-flux term rebuilt the
rate from the solve's own correction buffer.

  python test_implicit_register_x1_off.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cuda_build import build_has_cuda, skip  # noqa: E402
from snapy import kIDN  # noqa: E402


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    import test_cubed_sphere_capture as cs

    base = cs.deck

    def deck(*args):
        d = base(*args)
        d["integration"]["implicit-scheme"] = 1
        d["dynamics"]["disable-flux-x1"] = True
        return d

    cs.deck = deck
    try:
        mesh, blocks, mvars = cs.make("eager", False, False, False)
    finally:
        cs.deck = base
    dt = 0.5 * float(mesh.max_time_step(mvars))
    for s in range(3):
        mesh.forward(mvars, dt, s)
        kept = 0.0
        for i, b in enumerate(blocks):
            bufs = dict(b.named_buffers())
            kept = max(kept, bufs["hydro.K_vic_mass_total_1"].abs().max().item())
            assert torch.equal(bufs["hydro.F1"][kIDN],
                               bufs["hydro.K_vic_mass_dry_1"][kIDN]), (s, i)
        print(f"stage {s}: max|kept vic.mass_total| over blocks = {kept:.6g}",
              flush=True)
        assert kept > 0.0, (s, "the solve booked no transfer with the x1 flux off")
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
