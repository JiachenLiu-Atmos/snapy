"""V3 S-K4 (K.rst 3.5): Mesh.set_chemistry runs the chemistry step in the
Mesh's global phase (the blocks' runs are deferred, the step runs after the
last stage before the final exchange). A two-block slab Mesh with the
chemistry registered equals the one-block MeshBlock with the same
registration on every owned cell, bitwise, over several steps, in FP64 and
FP32. The control: the same Mesh against a MeshBlock without chemistry
differs, so the Mesh really ran the step. RED before S-K4: Mesh has no
set_chemistry.

RULINGS S-K4.fp32_seam: on this base the FP32 arm is an expected failure.
Two blocks and one block differ at the x2 seam from step 1 in FP32 with the
chemistry off as well (the base's S step 4 rounds a seam sum by block
position). The FP64 arm is unchanged by this mark: bitwise for 100 steps on
x86 CPU and the H100 here, while on arm64 it differs from step 28 on this
base, as it did on the old base. The mark is strict: the FP32 arm fails if
it passes, so the base fix (a W4 step) removes this mark as its GREEN.

  python test_sk4_mesh_chemistry.py [--device cuda] [--steps 3]
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

import kintera as kt  # noqa: E402
from snapy import Mesh, MeshBlock, MeshBlockOptions, MeshOptions, kICY, kIDN, kIPR, kIV1  # noqa: E402

NGHOST = 3


def deck(precision, nb2):
    cfg = config(precision)
    cfg["geometry"]["cells"]["nx2"] = 8  # two slabs of four cells, wider than the ghost zone
    cfg["distribute"]["nb2"] = nb2
    return cfg


def fill(block):
    w = block.module("hydro").buffer("D").clone().zero_()
    ty = block.module("hydro.eos.thermo")
    z = block.module("coord").buffer("x1v")
    temp = (400.0 - 1.0e-3 * z).clamp_min(150.0)
    rd = 8.31446261815324 * ty.buffer("inv_mu")[0]
    pressure = 1.0e7 * (temp / 400.0) ** (24.79 / (rd * 0.001))
    w[kIPR] = pressure
    w[kIDN] = pressure / (rd * temp)
    w[kICY].fill_(0.01)
    w[kICY + 1].fill_(0.001)
    w[kICY + 3].fill_(1.0e-5)
    w[kICY + 5].fill_(1.0e-6)
    w[kIV1] = 0.01 * torch.sin(torch.pi * z / 2.0e5)
    return {"hydro_w": w, "scalar_r": torch.full_like(w[:1], 0.001)}


def make(precision, device, block_chemistry):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(precision, 1), f)
        one = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(precision, 2), f)
        two = f.name
    try:
        kin = kt.KineticsOptions.from_yaml(one)
        block = MeshBlock(MeshBlockOptions.from_yaml(one))
        options = MeshOptions.from_yaml(two)
        options.blocks_per_process(2)
        mesh = Mesh(options)
        mesh.set_chemistry(kin)
        if block_chemistry:
            block.set_chemistry(kin)
    finally:
        os.unlink(one)
        os.unlink(two)
    block.to(device)
    blocks = list(mesh.blocks)
    assert len(blocks) == 2, "the slab Mesh should hold two blocks, got %d" % len(blocks)
    for b in blocks:
        b.to(device)
    bv, _ = block.initialize(fill(block))
    mv, _ = mesh.initialize([fill(b) for b in blocks])
    return block, bv, mesh, mv


def owned_diffs(bv, mv):
    """(cells, max |d|) per Mesh block on the owned cells, against the one-block state."""
    out = []
    for ib, v in enumerate(mv):
        n2 = v["hydro_u"].shape[2] - 2 * NGHOST
        j0 = NGHOST + ib * n2
        own = v["hydro_u"][:, NGHOST:-NGHOST, NGHOST:NGHOST + n2, NGHOST:-NGHOST]
        ref = bv["hydro_u"][:, NGHOST:-NGHOST, j0:j0 + n2, NGHOST:-NGHOST]
        d = (own.double() - ref.double()).abs()
        out.append((int((d > 0).sum()), d.max().item()))
    return out


def run(precision, device, steps, block_chemistry):
    block, bv, mesh, mv = make(precision, device, block_chemistry)
    dt = 0.1
    worst = []
    for step in range(steps):
        for stage in range(len(block.intg.stages)):
            block.forward(bv, dt, stage)
            mesh.forward(mv, dt, stage)
        rb, rm = block.check_redo(bv), mesh.check_redo(mv)
        assert rb == 0 and rm == 0, "%s step %d: check_redo %d (block) %d (mesh)" % (precision, step, rb, rm)
        worst.append(owned_diffs(bv, mv))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; skipping")
        return 0
    device = torch.device(args.device)
    torch.set_num_threads(1)
    failures = []
    for precision in ("float64", "float32"):
        diffs = run(precision, device, args.steps, True)
        bad = [(s, d) for s, d in enumerate(diffs) if any(n for n, _ in d)]
        if precision == "float32":
            # RULINGS S-K4.fp32_seam: expected failure on this base, strict
            if bad:
                print("float32: two-block Mesh differs from the one-block MeshBlock, first at step %d: %s;"
                      " expected failure on this base (RULINGS S-K4.fp32_seam: the x2 seam rounds by"
                      " block position in FP32 since the base's S step 4, chemistry off as well)"
                      % (bad[0][0], bad[0][1]))
            else:
                failures.append("float32: the two-block Mesh equals the one-block MeshBlock, so the"
                                " S-K4.fp32_seam expected-failure mark must come out")
        elif bad:
            failures.append("%s: two-block Mesh differs from the one-block MeshBlock, first at step %d: %s"
                            % (precision, bad[0][0], bad[0][1]))
        print("%s: %d steps, Mesh.set_chemistry equals the block's registered step on the owned cells %s"
              % (precision, args.steps, "bitwise" if not bad else "NOT"))
        control = run(precision, device, 1, False)
        if not any(n for n, _ in control[0]):
            failures.append("%s: control failed, the Mesh with chemistry equals a block without it" % precision)
        else:
            print("%s: control, Mesh with chemistry vs block without: %s cells differ"
                  % (precision, [n for n, _ in control[0]]))
    for f in failures:
        print("FAIL:", f)
    print("PASSED" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
