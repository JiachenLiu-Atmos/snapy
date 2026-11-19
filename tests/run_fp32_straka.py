#!/usr/bin/env python3
"""Paired Straka experiment for the supported HLLC/ideal-moist FP32 path.

Run each precision separately, with CUDA_VISIBLE_DEVICES selecting its GPU.
Uses the shipped Straka geometry/initial condition and forcing (no added
viscosity); problem/K in the shipped deck is not consumed by straka.cpp.
This is a precision comparison, not the LMARS reference regression.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR
from test_fp32_science import memory_bytes


def make_case(precision, device, execution="eager", strict=False):
    os.environ["DEVICE"] = device.type
    os.environ["DEVICE_ID"] = "0"
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "examples/straka.yaml").read_text())
    moist = yaml.safe_load((root / "tests/test_anomaly_storage.yaml").read_text())
    cfg["reference-state"] = moist["reference-state"]
    cfg["species"] = moist["species"][:1]
    cfg["reactions"] = []
    cfg["distribute"]["nb2"] = 1
    cfg["outputs"] = []
    cfg["dynamics"].update(
        {
            "anomaly": True,
            "perturbation-flux": True,
            "precision": precision,
            "riemann-solver": {"type": "hllc"},
        }
    )
    cfg["dynamics"]["equation-of-state"] = {
        "type": "ideal-moist",
        "limiter": False,
        "density-floor": 1.0e-10,
        "pressure-floor": 1.0e-10,
    }
    cfg["integration"]["ncycle_out"] = 0
    cfg["integration"]["execution"] = execution
    cfg["integration"]["execution-strict"] = strict
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(f.name))
    finally:
        os.unlink(f.name)
    block.to(device)
    thermo = block.module("hydro.eos.thermo")
    rd = 8.31446261815324 * thermo.buffer("inv_mu")[0].item()
    cp = 3.5 * rd
    z = block.module("coord").buffer("x1v").view(1, 1, -1)
    x = block.module("coord").buffer("x2v").view(1, -1, 1)
    w = block.module("hydro").buffer("D").double().clone().zero_()
    temp = 300.0 - 9.8 * z / cp
    w[kIPR] = 1.0e5 * (temp / 300.0) ** (cp / rd)
    radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
    temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
    w[kIDN] = w[kIPR] / (rd * temp)
    ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": ratio})
    expected = getattr(torch, precision)
    assert state["hydro_u"].dtype == state["scalar_s"].dtype == expected
    assert block.module("hydro").buffer("F1").dtype == expected
    return block, state, cfg, rd, cp, x, z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precision", choices=["float64", "float32"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tlim", type=float, default=900.0)
    ap.add_argument("--execution", choices=["eager", "cuda-graph"], default="eager")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--profile", action="store_true", help="Profile five extra steps after saving the run"
    )
    args = ap.parse_args()
    if args.dt <= 0 or args.tlim <= 0:
        ap.error("dt and tlim must be positive")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    os.environ["DEVICE"] = device.type
    os.environ["DEVICE_ID"] = "0"
    if args.execution == "cuda-graph" and device.type != "cuda":
        ap.error("cuda-graph execution requires CUDA")
    block, state, cfg, rd, cp, x, z = make_case(args.precision, device, args.execution)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg))
    sl = tuple(block.part((0, 0, 0), False))
    vol = block.module("coord").cell_volume()[sl[1:]].double()
    mass0 = (block.total_u(state)[sl][kIDN].double() * vol).sum().item()
    history = []

    def snapshot(t):
        block.module("hydro.eos").compute("U'->W", [state["hydro_u"], state["hydro_w"]])
        prim = state["hydro_w"][sl].double()
        theta = prim[kIPR] / (rd * prim[kIDN]) * (1.0e5 / prim[kIPR]) ** (rd / cp) - 300.0
        total = block.total_u(state)[sl].double()
        drift = ((total[kIDN] * vol).sum().item() - mass0) / mass0
        data = {
            "time": t,
            "mass_drift": drift,
            "theta_min": theta.min().item(),
            "theta_max": theta.max().item(),
        }
        history.append(data)
        np.savez_compressed(
            args.output / f"state-{t:g}.npz",
            theta=theta.cpu().numpy(),
            primitive=prim.cpu().numpy(),
            total=total.cpu().numpy(),
            scalar=state["scalar_s"][sl].double().cpu().numpy(),
            x=x.flatten()[3:-3].cpu().numpy(),
            z=z.flatten()[3:-3].cpu().numpy(),
        )
        print(data, flush=True)

    def advance(dt):
        for stage in range(len(block.intg.stages)):
            block.forward(state, dt, stage)

    snapshot(0.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    current, steps = 0.0, 0
    next_output = min(300.0, args.tlim)
    while current < args.tlim - 1.0e-9:
        dt = min(args.dt, args.tlim - current, next_output - current)
        allowed = block.max_time_step(state)
        assert dt <= allowed * (1.0 + 1.0e-10), (current, dt, allowed)
        advance(dt)
        assert block.check_redo(state) == 0, (current, "step rejected")
        assert torch.isfinite(state["hydro_u"]).all(), current
        steps += 1
        current += dt
        if current >= next_output - 1.0e-9:
            snapshot(next_output)
            current = next_output
            next_output = min(next_output + 300.0, args.tlim)
        if steps % 1000 == 0:
            print(f"{args.precision} step={steps} time={current:.3f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "precision": args.precision,
        "execution": args.execution,
        "graph_captures": block.execution_report()["captures"],
        "graph_replays": block.execution_report()["replays"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "snapy": importlib.metadata.version("snapy"),
        "kintera": importlib.metadata.version("kintera"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_libraries_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(snapy.__file__).parent / "lib").glob("libsnap*")
            if p.is_file()
        },
        "rd": rd,
        "cp": cp,
        "steps": steps,
        "dt": args.dt,
        "elapsed_seconds_including_diagnostics": time.perf_counter() - start,
        "storage": memory_bytes(block, state),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "history": history,
    }
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            for _ in range(5):
                advance(args.dt)
                assert block.check_redo(state) == 0
            if device.type == "cuda":
                torch.cuda.synchronize()
        report["profile_extra_steps"] = 5
        report["profile"] = {
            e.key: {
                "calls": e.count,
                "self_cpu_ms": e.self_cpu_time_total / 1000.0,
                "self_device_ms": e.self_device_time_total / 1000.0,
            }
            for e in prof.key_averages()
        }
        prof.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
