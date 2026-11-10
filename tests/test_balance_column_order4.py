#!/usr/bin/env python3
"""snapy.balance_column at conversion_order 4 from Python, on test_balance_column's moist card
(vapour falling with height). The column it returns needs no further pass: a second call takes 0
passes and changes nothing beyond 1e-12. Order 4 without the run's eos raises."""
import torch
from snapy import MeshBlock, MeshBlockOptions, balance_column, kICY, kIDN, kIPR

block = MeshBlock(MeshBlockOptions.from_yaml("test_balance_column_moist.yaml"))
block.to(torch.device("cpu"), torch.float64)
eos, g, nx = block.module("hydro.eos"), 10.44, 32
dx1f, z = torch.full((nx,), 1.e4 / nx, dtype=torch.float64), (torch.arange(nx) + .5) * 1.e4 / nx
w = torch.zeros(eos.nvar(), 1, 1, nx, dtype=torch.float64)
w[kICY] = 1.e-2 * torch.exp(-z / 1.5e3)
w[kIDN], w[kIPR] = 1., 1.
rt = (300. - 6.5e-3 * z) / eos.compute("W->T", [w.clone()])  # R_mix T
w[kIPR] = 1.e5 * torch.exp(-g * torch.cumsum(dx1f / rt[0, 0], 0))
w[kIDN] = w[kIPR] / rt

args = (dx1f, g, True, 1.e-13, 400, "cartesian", False, 4)
wb = balance_column(w, *args, eos=eos)[0]
w2, _, passes = balance_column(wb, *args, eos=eos)
assert passes == 0 and ((w2 - wb).abs() / wb.abs().clamp_min(1.e-300)).max() <= 1.e-12, passes
try:
    balance_column(w, *args)
    raise AssertionError("conversion_order 4 without eos did not raise")
except RuntimeError as e:
    assert "needs the run's eos" in str(e), e
print("balance_column order 4: balanced in", balance_column(w, *args, eos=eos)[2], "passes; OK")
