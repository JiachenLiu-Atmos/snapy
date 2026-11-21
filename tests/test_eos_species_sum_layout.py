#!/usr/bin/env python3
"""EF-13: the ideal-moist EOS species sums must not round by memory position.

f_sig, f_eps and the species offset in _prim2intEng sum over the species axis
of `prim.narrow(0, ICY, ny)`. In a block's (nvar, nc3, nc2, nc1) layout the
species stride is nc3*nc2*nc1, not 1, so the operand is a strided view. A
matmul on that view rounded by memory position: identical inputs in different
x2 columns came out 1 ulp apart (the S-K4 step-28 two-block vs one-block
difference). With an explicit multiply-and-sum the result depends on the
values only.

The gate: a state that is identical in every x2 column (it varies over x3 and
x1, and every species is nonzero) goes through "W->A" (f_eps / f_sig),
"W->T" (f_eps), "W->I" (f_sig, f_eps and the offset sum), "W->U" and
"U->W" (both directions of the conversion). Every output must be bitwise
equal across the x2 columns, in FP64 and FP32 storage. RED before EF-13.

  python test_eos_species_sum_layout.py [--device cuda]
"""

import argparse
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_sk4_chemistry_source import config  # noqa: E402

from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1  # noqa: E402

KEYS = ("W->A", "W->T", "W->I", "W->U", "U->W")


def make(precision, device):
    cfg = config(precision)
    cfg["geometry"]["cells"].update({"nx1": 24, "nx2": 8, "nx3": 4})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(path))
    finally:
        os.unlink(path)
    block.to(device)
    eos = block.module("hydro.eos")
    return block, eos


def x2_uniform_state(block, seed):
    """Random over (x3, x1), copied to every x2 column; contiguous layout."""
    w = block.module("hydro").buffer("D").clone().zero_()
    nvar, n3, n2, n1 = w.shape
    ny = nvar - kICY
    g = torch.Generator().manual_seed(seed)
    col = torch.zeros(nvar, n3, 1, n1, dtype=torch.float64)
    col[kIDN] = 0.1 + torch.rand(n3, 1, n1, generator=g, dtype=torch.float64)
    col[kIPR] = 1.0e5 * (0.5 + torch.rand(n3, 1, n1, generator=g, dtype=torch.float64))
    col[kIV1:kIV1 + 3] = torch.rand(3, n3, 1, n1, generator=g, dtype=torch.float64) - 0.5
    col[kICY:] = 1.0e-2 * torch.rand(ny, n3, 1, n1, generator=g, dtype=torch.float64)
    w.copy_(col.expand(nvar, n3, n2, n1))
    return w


def nonuniform(x):
    """Entries that differ (bitwise) from the first x2 column of the same (x3, x1)."""
    return int((x != x[..., :1, :]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; skipping")
        return 0
    device = torch.device(args.device)
    torch.set_num_threads(1)
    failures = []
    for precision in ("float64", "float32"):
        block, eos = make(precision, device)
        for seed in range(args.seeds):
            w = x2_uniform_state(block, seed).to(device)
            assert nonuniform(w) == 0
            ny = w.size(0) - kICY
            yu = w.narrow(0, kICY, ny).unfold(0, ny, 1)
            assert not yu.is_contiguous() and yu.stride(-1) == w.stride(0) != 1, (
                "the species operand should be the strided block layout, got strides %s" % (yu.stride(),))
            u = eos.compute("W->U", [w])
            out = {k: eos.compute(k, [u if k == "U->W" else w]) for k in KEYS}
            bad = {k: nonuniform(v) for k, v in out.items() if nonuniform(v)}
            print("%s seed %d: w %s %s, species stride %d; x2-nonuniform entries %s"
                  % (precision, seed, str(w.dtype).replace("torch.", ""), tuple(w.shape),
                     w.stride(0), bad if bad else "none"))
            if bad:
                failures.append("%s seed %d: identical x2 columns give different results %s"
                                % (precision, seed, bad))
    for f in failures:
        print("FAIL:", f)
    print("PASSED" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
