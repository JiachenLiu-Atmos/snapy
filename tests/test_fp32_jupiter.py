#!/usr/bin/env python3
"""Reduced H2O/NH3 CRM: FP32 dynamics/VIC, FP64 saturation and kinetics."""

import argparse
import os
import tempfile
import torch
import yaml
import kintera as kt
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kICY, kIV1


def run(steps, precision="float32"):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "../examples/jupiter_crm.yaml")) as f:
        cfg = yaml.safe_load(f)
    names = {"dry", "H2O", "NH3", "H2O(l)", "H2O(l,p)", "NH3(s)", "NH3(s,p)"}
    cfg["species"] = [s for s in cfg["species"] if s["name"] in names]
    cfg["reactions"] = [
        r for r in cfg["reactions"] if "H2S" not in r["equation"] and "NH4SH" not in r["equation"]
    ]
    cfg["geometry"] = {
        "type": "cartesian",
        "bounds": {
            "x1min": 0.0,
            "x1max": 2.0e5,
            "x2min": 0.0,
            "x2max": 2.0e4,
            "x3min": 0.0,
            "x3max": 2.0e4,
        },
        "cells": {"nx1": 24, "nx2": 4, "nx3": 4, "nghost": 3},
    }
    cfg["distribute"] = {"layout": "slab", "nb2": 1, "nb3": 1}
    cfg["dynamics"].update(
        {
            "anomaly": True,
            "perturbation-flux": True,
            "precision": precision,
            "riemann-solver": {"type": "hllc"},
        }
    )
    cfg["integration"].update({"implicit-scheme": 9, "cfl": 1.0, "ncycle_out": 0})
    cfg["forcing"] = {
        "const-gravity": {"grav1": -24.79},
        "bot-heat": {"flux": 100.0, "depth": 3},
        "top-cool": {"flux": -100.0, "depth": 3},
        "diffusion": {"kappa_iso": 10.0, "nu_iso": 10.0},
    }
    cfg["sedimentation"] = {
        "radius": {},
        "density": {},
        "const-vsed": {"H2O(l,p)": -20.0, "NH3(s,p)": -20.0},
    }
    cfg.pop("outputs", None)
    cfg.pop("problem", None)
    cfg["scalar"] = {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "weno5"}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(path))
        kin = kt.Kinetics(kt.KineticsOptions.from_yaml(path))
        tx = kt.ThermoX(kt.ThermoOptions.from_yaml(path))
    finally:
        os.unlink(path)
    eos = block.module("hydro.eos")
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
    sl = block.part((0, 0, 0), False)
    initial = block.total_u(bv)[sl].clone()
    inv_mu = ty.buffer("inv_mu")[1:]
    stoich = kin.buffer("stoich")
    assert stoich.dtype == torch.float64
    dt = 0.1
    for step in range(steps):
        for stage in range(len(block.intg.stages)):
            block.forward(bv, dt, stage)
        # Same FP64 chemistry boundary as examples/run_hydro.cpp. Apply only
        # its species increment to the stored state, never FP32 total energy.
        eos.compute("U'->W", [bv["hydro_u"], bv["hydro_w"]])
        w = bv["hydro_w"]
        t, p = eos.compute("W->T", [w]), w[kIPR]
        x = ty.compute("Y->X", [w[kICY:]])
        conc = tx.compute("TPX->V", [t, p, x])
        cv = tx.compute("TV->cp", [t, conc])
        c = conc[..., 1:]
        rate, dc, dT, kf = kin.forward_with_rate_constants(t, p, c)
        jac = kin.jacobian(t, c, cv, rate, dc, dT, kf)
        delta = kt.evolve_implicit(rate, stoich, jac, dt)
        assert delta.dtype == torch.float64 and jac.dtype == torch.float64
        bv["hydro_u"][kICY:] += (delta / inv_mu).permute(3, 0, 1, 2)
        status = block.check_redo(bv)
        assert status == 0, (step, status)
        assert bv["hydro_u"].dtype == (torch.float32 if precision == "float32" else torch.float64)
        assert bv["scalar_s"].dtype == bv["hydro_u"].dtype
        assert torch.isfinite(bv["scalar_s"]).all()
        assert torch.isfinite(bv["hydro_u"]).all()
        if (step + 1) % 100 == 0 or step == 0:
            print(
                f"Jupiter {precision} step={step + 1}, fallbacks={block.module('hydro.icorr').buffer('fp64_fallbacks').item()}",
                flush=True,
            )
    final = block.total_u(bv)[sl].clone()
    mass0 = initial[kIDN].sum() + initial[kICY:].sum()
    mass1 = final[kIDN].sum() + final[kICY:].sum()
    drift = abs((mass1 - mass0) / mass0).item()
    print(f"Jupiter {precision}: {steps} steps, mass drift={drift:.3e}", flush=True)
    assert drift < (2.0e-5 if precision == "float32" else 1.0e-11)
    assert block.module("hydro.icorr").buffer("fp64_fallbacks").item() == 0
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(1)
    run(args.steps)
