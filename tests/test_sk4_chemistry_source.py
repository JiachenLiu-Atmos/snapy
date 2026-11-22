#!/usr/bin/env python3
"""V3 lane K, S-K4 (K.rst 3.6, row S-K4): the chemistry step registered on a
block (MeshBlock.set_chemistry) equals the caller's own step at the pin,
bitwise.

The reduced Jupiter CRM of tests/test_fp32_jupiter.py. Block A registers the
reaction options; block B is stepped by hand with the pin's chemistry
(examples/run_hydro.cpp:171-198, as test_fp32_jupiter.py:116-129 did before
S-K4): the EOS cons2prim, FP64 rates, Jacobian, one implicit step, and the
species increment on hydro_u. After every step A's hydro_u and hydro_w must
equal B's bitwise, in FP64 and FP32. RED before S-K4: MeshBlock has no
set_chemistry.

  python test_sk4_chemistry_source.py [--device cuda] [--steps 5]
"""
import argparse
import os
import sys
import tempfile

import torch
import yaml

import kintera as kt
from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

HERE = os.path.dirname(os.path.abspath(__file__))


def config(precision):
    with open(os.path.join(HERE, "../examples/jupiter_crm.yaml")) as f:
        cfg = yaml.safe_load(f)
    names = {"dry", "H2O", "NH3", "H2O(l)", "H2O(l,p)", "NH3(s)", "NH3(s,p)"}
    cfg["species"] = [s for s in cfg["species"] if s["name"] in names]
    cfg["reactions"] = [
        r for r in cfg["reactions"] if "H2S" not in r["equation"] and "NH4SH" not in r["equation"]
    ]
    cfg["geometry"] = {
        "type": "cartesian",
        "bounds": {"x1min": 0.0, "x1max": 2.0e5, "x2min": 0.0, "x2max": 2.0e4,
                   "x3min": 0.0, "x3max": 2.0e4},
        "cells": {"nx1": 24, "nx2": 4, "nx3": 4, "nghost": 3},
    }
    cfg["distribute"] = {"layout": "slab", "nb2": 1, "nb3": 1}
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True,
                            "precision": precision, "riemann-solver": {"type": "hllc"}})
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


def make(precision, device, register, execution=None):
    cfg = config(precision)
    if execution:  # the executor's strict refusal is tested below
        cfg["integration"]["execution"] = execution
        cfg["integration"]["execution-strict"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        options = MeshBlockOptions.from_yaml(path)
        kin_options = kt.KineticsOptions.from_yaml(path)
        tx_options = kt.ThermoOptions.from_yaml(path)
        block = MeshBlock(options)
        if register:
            block.set_chemistry(kin_options)
        kin = kt.Kinetics(kin_options)
        tx = kt.ThermoX(tx_options)
    finally:
        os.unlink(path)
    block.to(device)
    kin.to(device)
    tx.to(device)
    ty = block.module("hydro.eos.thermo")
    w = block.module("hydro").buffer("D").clone().zero_()
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
    ratio = torch.full_like(w[:1], 0.001)
    bv, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    return block, bv, kin, tx


def pin_chemistry(block, bv, kin, tx, dt):
    """The caller's own step at the pin (test_fp32_jupiter.py:116-129 before S-K4)."""
    eos = block.module("hydro.eos")
    ty = block.module("hydro.eos.thermo")
    eos.compute("U'->W", [bv["hydro_u"], bv["hydro_w"]])
    w = bv["hydro_w"]
    t, p = eos.compute("W->T", [w]), w[kIPR]
    x = ty.compute("Y->X", [w[kICY:]])
    conc = tx.compute("TPX->V", [t, p, x])
    cv = tx.compute("TV->cp", [t, conc])
    c = conc[..., 1:]
    rate, dc, dT, kf = kin.forward_with_rate_constants(t, p, c)
    jac = kin.jacobian(t, c, cv, rate, dc, dT, kf)
    delta = kt.evolve_implicit(rate, kin.buffer("stoich"), jac, dt)
    assert delta.dtype == torch.float64 and jac.dtype == torch.float64
    bv["hydro_u"][kICY:] += (delta / ty.buffer("inv_mu")[1:]).permute(3, 0, 1, 2)
    # the registered step applies the boundaries after its chemistry; the pin
    # does the same, so the ghost cells are derived from the owned cells on
    # both sides (on arm64 a ghost cell's own chemistry differs by 1 ulp)
    block.apply_boundaries(bv, bv["hydro_u"], bv.get("scalar_s"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; skipping")
        return 0
    device = torch.device(args.device)
    torch.set_num_threads(1)
    failures = []
    for precision in ("float64", "float32"):
        a, av, _, _ = make(precision, device, True)
        b, bvv, kin, tx = make(precision, device, False)
        assert torch.equal(av["hydro_u"], bvv["hydro_u"]), "the two blocks start differently"
        dt = 0.1
        for step in range(args.steps):
            for stage in range(len(a.intg.stages)):
                a.forward(av, dt, stage)
                b.forward(bvv, dt, stage)
            pin_chemistry(b, bvv, kin, tx, dt)
            sa, sb = a.check_redo(av), b.check_redo(bvv)
            if sa != 0 or sb != 0:
                failures.append("%s step %d: check_redo %d (registered) vs %d (pin)" % (precision, step, sa, sb))
                break
            for k in ("hydro_u", "hydro_w"):
                if not torch.equal(av[k], bvv[k]):
                    d = (av[k].double() - bvv[k].double()).abs()
                    failures.append("%s step %d: %s differs, max |d| %.3e in %d of %d"
                                    % (precision, step, k, d.max().item(), int((d > 0).sum()), d.numel()))
                    break
            else:
                continue
            break
        changed = int((av["hydro_u"][kICY:] != bvv["hydro_u"][kICY:]).sum()) if failures else 0
        print("%s: %d steps, registered step equals the pin's %s" % (
            precision, args.steps, "bitwise" if not failures else "NOT (%d species cells differ)" % changed))
    # a registered chemistry step is a capture reason: the strict cuda-graph
    # executor refuses the block before its first launch and names it
    c, cv, _, _ = make("float64", device, True, execution="cuda-graph")
    try:
        c.forward(cv, 0.1, 0)
        failures.append("strict cuda-graph with a chemistry step was not refused")
    except RuntimeError as e:
        if "a chemistry step is registered" not in str(e):
            failures.append("refusal does not name the chemistry step: %s" % str(e).splitlines()[0])
        else:
            print("strict cuda-graph: refused, 'a chemistry step is registered'")
    for f in failures:
        print("FAIL:", f)
    print("PASSED" if not failures else "FAILED (%d)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
