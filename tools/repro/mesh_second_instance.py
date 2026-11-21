"""Repro for S-L1 (mesh.second_instance): a second Mesh in the same process
must not change the first one.

What it checks
  Mesh A is the S-K4 two-block slab deck (examples/jupiter_crm.yaml cut down
  as in tests/test_sk4_chemistry_source.py, nx2=8, nb2=2, FP64, no chemistry
  registered). A steps N times (default 6) alone, and again with a Mesh B of
  the same deck built in the same process. Each run is its own subprocess, so
  process-wide state starts clean. The hydro_u of A (ghosts included) after
  every step must equal the A-alone run bitwise.

  Variants (B built at the start of step 1 or step 3, after A has stepped):
    ctor@1 init@1 step@1 ctor@3 init@3 step@3
      ctor = B constructed only, init = B constructed and initialized,
      step = B initialized and stepped alongside A
  Controls (expected bitwise before and after the fix):
    before:ctor before:init before:step  B fully built before A
    nb1:ctor@1 nb1:step@1                one-block A and one-block B
                                         (no same-process exchange)

How to run (the built snapy of the tree under test on PYTHONPATH, or --pkg):
  python tools/repro/mesh_second_instance.py [--pkg DIR] [--steps 6]
         [--variants ctor@1,step@3,...] [--no-controls]

RED / GREEN
  RED   = some variant's A differs from A-alone (exit status 1). Expected on
          7c7743517e68 (EF-13) and its parent 6760831d: the process-wide
          g_local_layouts (src/layout/layout.cpp:102) is keyed by
          (process_rank, local_block_index) without the Mesh, so B's blocks
          replace A's and A's halo exchange copies B's buffers.
          ctor@1 shows max abs 1.6056 at step 1.
  GREEN = every variant and control is bitwise (exit status 0).
  A control that differs is also reported and also exits 1.
  mesh_second_instance.rereg-hook.patch is the diagnostic hook (not a fix):
  built with it and run with SNAPY_DIAG_REREG=1 this script reports GREEN.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "..", "..", "examples", "jupiter_crm.yaml")
VARIANTS = ["ctor@1", "init@1", "step@1", "ctor@3", "init@3", "step@3"]
CONTROLS = ["before:ctor", "before:init", "before:step", "nb1:ctor@1", "nb1:step@1"]


def deck(nb2):
    """The S-K4 slab deck (tests/test_sk4_chemistry_source.py config + test_sk4_mesh_chemistry deck)."""
    import yaml

    with open(DECK) as f:
        cfg = yaml.safe_load(f)
    names = {"dry", "H2O", "NH3", "H2O(l)", "H2O(l,p)", "NH3(s)", "NH3(s,p)"}
    cfg["species"] = [s for s in cfg["species"] if s["name"] in names]
    cfg["reactions"] = [r for r in cfg["reactions"]
                        if "H2S" not in r["equation"] and "NH4SH" not in r["equation"]]
    cfg["geometry"] = {
        "type": "cartesian",
        "bounds": {"x1min": 0.0, "x1max": 2.0e5, "x2min": 0.0, "x2max": 2.0e4,
                   "x3min": 0.0, "x3max": 2.0e4},
        "cells": {"nx1": 24, "nx2": 8, "nx3": 4, "nghost": 3},
    }
    cfg["distribute"] = {"layout": "slab", "nb2": nb2, "nb3": 1}
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True, "precision": "float64",
                            "riemann-solver": {"type": "hllc"}})
    cfg["integration"].update({"implicit-scheme": 9, "cfl": 1.0, "ncycle_out": 0})
    cfg["forcing"] = {"const-gravity": {"grav1": -24.79},
                      "bot-heat": {"flux": 100.0, "depth": 3},
                      "top-cool": {"flux": -100.0, "depth": 3},
                      "diffusion": {"kappa_iso": 10.0, "nu_iso": 10.0}}
    cfg["sedimentation"] = {"radius": {}, "density": {},
                            "const-vsed": {"H2O(l,p)": -20.0, "NH3(s,p)": -20.0}}
    cfg["outputs"] = []
    cfg.pop("problem", None)
    return cfg


def fill(block):
    import torch
    from snapy import kICY, kIDN, kIPR, kIV1

    w = block.module("hydro").buffer("D").clone().zero_()
    z = block.module("coord").buffer("x1v")
    temp = (400.0 - 1.0e-3 * z).clamp_min(150.0)
    rd = 8.31446261815324 * block.module("hydro.eos.thermo").buffer("inv_mu")[0]
    pressure = 1.0e7 * (temp / 400.0) ** (24.79 / (rd * 0.001))
    w[kIPR] = pressure
    w[kIDN] = pressure / (rd * temp)
    w[kICY].fill_(0.01)
    w[kICY + 1].fill_(0.001)
    w[kICY + 3].fill_(1.0e-5)
    w[kICY + 5].fill_(1.0e-6)
    w[kIV1] = 0.01 * torch.sin(torch.pi * z / 2.0e5)
    return {"hydro_w": w, "scalar_r": torch.full_like(w[:1], 0.001)}


def make(nb2, init):
    import yaml
    from snapy import Mesh, MeshOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(nb2), f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(nb2)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    return mesh, (mesh.initialize([fill(b) for b in mesh.blocks])[0] if init else None)


def child(args):
    """One process: A (nb2) steps; B (mode, built at step `when`, -1 = before A)."""
    import torch

    torch.set_num_threads(1)
    mode, when = args.mode, args.when
    B = None
    if mode != "none" and when < 0:
        B = make(args.nb, mode != "ctor")
    A, av = make(args.nb, True)
    out = []
    for s in range(args.steps):
        if mode != "none" and B is None and s == when:
            B = make(args.nb, mode != "ctor")
        for stage in range(3):
            A.forward(av, 0.1, stage)
            if mode == "step" and B is not None:
                B[0].forward(B[1], 0.1, stage)
        A.check_redo(av)
        if mode == "step" and B is not None:
            B[0].check_redo(B[1])
        out.append(torch.stack([v["hydro_u"].clone() for v in av]))
    torch.save(out, args.out)


def parse(name):
    """'ctor@1' -> (nb, mode, when); 'before:step' -> when -1; 'nb1:ctor@1' -> nb 1."""
    nb = 2
    if name.startswith("nb1:"):
        nb, name = 1, name[4:]
    if name.startswith("before:"):
        return nb, name[7:], -1
    mode, when = name.split("@")
    return nb, mode, int(when)


def run(nb, mode, when, steps, tmp, env):
    out = os.path.join(tmp, "%d_%s_%d.pt" % (nb, mode, when))
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--nb", str(nb), "--mode", mode,
           "--when", str(when), "--steps", str(steps), "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit("run %s failed:\n%s" % (" ".join(cmd), r.stderr[-4000:]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help="comma list from %s (controls are added unless --no-controls)" % VARIANTS)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--pkg", help="directory holding the built snapy package (prepended to PYTHONPATH)")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--nb", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument("--mode", default="none", help=argparse.SUPPRESS)
    p.add_argument("--when", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        return child(args)

    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    if args.pkg:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.abspath(args.pkg), env.get("PYTHONPATH")]))
    import torch

    info = subprocess.run([sys.executable, "-c", "import snapy; print(snapy.__file__)"],
                          env=env, check=True, capture_output=True, text=True).stdout.strip()
    print("snapy: %s  torch %s  steps %d  SNAPY_DIAG_REREG=%s" %
          (info, torch.__version__, args.steps, env.get("SNAPY_DIAG_REREG", "")))
    names = [v for v in args.variants.split(",") if v]
    if not args.no_controls:
        names += CONTROLS
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        alone = {}
        for name in names:
            nb, mode, when = parse(name)
            if nb not in alone:
                alone[nb] = torch.load(run(nb, "none", 0, args.steps, tmp, env))
            got = torch.load(run(nb, mode, when, args.steps, tmp, env))
            d = [(x - y).abs().max().item() for x, y in zip(got, alone[nb])]
            first = next((s for s, v in enumerate(d) if v > 0), None)
            kind = "variant" if name in VARIANTS else "control"
            if first is None:
                print("%-12s %-7s BITWISE over %d steps" % (name, kind, args.steps))
            else:
                bad.append(name)
                print("%-12s %-7s DIFFERS first at step %d: max abs %.6g there, %.6g over all steps"
                      % (name, kind, first, d[first], max(d)))
    print("RED: A changed by a second Mesh in %s" % ", ".join(bad) if bad else
          "GREEN: A bitwise to A-alone in every run")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
