#!/usr/bin/env python3
"""snapy.balance_column at conversion_order 4 from Python, on test_balance_column's moist card
(vapour falling with height), 64 cells, x1rat 1 and 1.05, and at 1.05 with the vapour 0 above 5 km
(u - L u has negative vapour there, where the solver keeps W(u)). The column it returns needs no
further pass: a second call takes 0 passes and changes nothing beyond 1e-12. It is at rest under
key 4: 10 explicit steps leave max|v1|/c below 1e-13. Order 4 without the run's eos raises."""
import os
import tempfile

import torch
import yaml
from snapy import MeshBlock, MeshBlockOptions, balance_column, kICY, kIDN, kIPR, kIV1

g, nx, ng = 10.44, 64, 3
for x1rat, qtop in ((1., 1.e9), (1.05, 1.e9), (1.05, 5.e3)):
    cfg = yaml.safe_load(open("test_balance_column_moist.yaml"))
    cfg["geometry"]["cells"].update({"nx1": nx, **({"x1rat": x1rat} if x1rat != 1. else {})})
    cfg["dynamics"]["conversion-order"] = 4
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
    block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    os.unlink(f.name)
    block.to(torch.device("cpu"), torch.float64)
    eos, xf = block.module("hydro.eos"), block.buffer("coord.x1f")[ng:-ng]
    dx1f, z = (xf[1:] - xf[:-1]).contiguous(), (xf[1:] + xf[:-1]) / 2.
    w = torch.zeros(eos.nvar(), 1, 1, nx, dtype=torch.float64)
    w[kICY] = torch.where(z < qtop, 1.e-2 * torch.exp(-z / 1.5e3), 0.)
    w[kIDN], w[kIPR] = 1., 1.
    rt = (300. - 6.5e-3 * z) / eos.compute("W->T", [w.clone()])  # R_mix T
    w[kIPR] = 1.e5 * torch.exp(-g * torch.cumsum(dx1f / rt[0, 0], 0))
    w[kIDN] = w[kIPR] / rt

    args = (dx1f, g, True, 1.e-13, 400, "cartesian", x1rat != 1., 4)
    wb = balance_column(w, *args, eos=eos)[0]
    w2, _, passes = balance_column(wb, *args, eos=eos)
    assert passes == 0 and ((w2 - wb).abs() / wb.abs().clamp_min(1.e-300)).max() <= 1.e-12, passes
    v, _ = block.initialize({"hydro_w": torch.cat([wb[..., :1]] * ng + [wb] + [wb[..., -1:]] * ng, -1)})
    c = (eos.compute("W->A", [wb]) * wb[kIPR] / wb[kIDN]).sqrt()
    for _ in range(10):
        for stage in range(len(block.intg.stages)):
            block.forward(v, .8 * float((dx1f / c).min()), stage)
    wc = float((eos.compute("U->W", [v["hydro_u"].clone()])[kIV1, ..., ng:-ng] / c).abs().max())
    print(f"x1rat {x1rat}, vapour below {qtop:g} m: max|v1|/c after 10 key-4 steps {wc:.3e}")
    assert wc < 1.e-13, wc
try:
    balance_column(w, *args)
    raise AssertionError("conversion_order 4 without eos did not raise")
except RuntimeError as e:
    assert "needs the run's eos" in str(e), e
print("balance_column order 4: OK")
