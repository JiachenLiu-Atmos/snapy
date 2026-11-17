#!/usr/bin/env python3
"""M6 reduced moist budgets, paired seeded trajectories, and measured step costs.

Engineering gates on closed Cartesian columns, not production climate validation.
All diagnostic sums promote before reduction. Corrected PE uses the independent
quadrature formula already exercised by test_gravity_work_radial_exact.
"""

import argparse
import gc
import hashlib
import os
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import snapy
from snapy import kIDN, kIPR, kIV1, kICY
from test_anomaly_storage import initial_state, run_steps
from test_gravity_work_radial_exact import slope, variance
from test_perturbation_flux import make

G = 10.44
# Provisional reduced-case engineering limits, not scientific equivalence bounds.
LIMITS = {"mass": 2.0e-5, "dry_mass": 2.0e-5, "water_mass": 2.0e-4, "energy": 2.0e-5}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare(precision, scheme, seed, case, device):
    block = make(True, precision=precision, scheme=scheme, sediment=case == "sediment")
    block.to(device)
    w = initial_state(block)["hydro_w"]
    generator = torch.Generator().manual_seed(seed)
    phases = 2.0 * math.pi * torch.rand(2, generator=generator, dtype=torch.float64)
    z = block.module("coord").buffer("x1v")
    x = block.module("coord").buffer("x2v").view(1, -1, 1)
    w[kIV1] += (
        0.03
        * torch.sin(math.pi * z / 1.0e4)
        * torch.cos(2.0 * math.pi * x / 4000.0 + phases[0].item())
    )
    w[kIV1 + 2] += 0.03 * torch.sin(2.0 * math.pi * x / 4000.0 + phases[1].item())
    if case == "sediment":
        w[kICY + 1].fill_(0.001)
    state, _ = block.initialize({"hydro_w": w})
    return block, state


class Diagnostics:
    def __init__(self, block):
        self.block = block
        self.sl = tuple(block.part((0, 0, 0), False))
        self.vol = block.module("coord").cell_volume()[self.sl[1:]].double()
        x = block.module("coord").buffer("x1v")
        self.z = x[3:-3].double()
        self.var = variance(block.module("coord").buffer("x1f")[3:-3].double(), False)

    def read(self, state):
        u = self.block.total_u(state)[self.sl].double()
        rho = u[kIDN] + u[kICY:].sum(0)
        pe = G * (rho * self.z + self.var * slope(rho, self.z))
        energy = ((u[kIPR] + pe) * self.vol).sum().item()
        mass = (rho * self.vol).sum().item()
        dry = (u[kIDN] * self.vol).sum().item()
        water = (u[kICY:].sum(0) * self.vol).sum().item()
        kinetic = (0.5 * u[kIV1 : kIV1 + 3].square().sum(0) / rho * self.vol).sum().item()
        vrms = ((u[kIV1] / rho).square() * self.vol).sum().div(self.vol.sum()).sqrt().item()
        vapor = (u[kICY] * self.vol).sum().item()
        return {
            "mass": mass,
            "dry_mass": dry,
            "water_mass": water,
            "energy": energy,
            "kinetic_energy": kinetic,
            "vertical_velocity_rms": vrms,
            "vapor_mass": vapor,
        }


def memory_bytes(block, state):
    # Deduplicate aliased storages. Unregistered scratch/caches are NOT included.
    seen, totals = set(), {}
    for tensor in list(dict(block.named_buffers()).values()) + list(state.values()):
        if not isinstance(tensor, torch.Tensor) or not tensor.numel():
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            name = str(tensor.dtype)
            totals[name] = totals.get(name, 0) + storage.nbytes()
    return {"by_dtype": totals, "total": sum(totals.values())}


def fallbacks(block, scheme):
    return int(block.module("hydro.icorr").buffer("fp64_fallbacks").item()) if scheme else 0


def evolution(precision, scheme, seed, case, args):
    block, state = prepare(precision, scheme, seed, case, args.device)
    diag = Diagnostics(block)
    initial = diag.read(state)
    samples, source, requested = [], 0.0, 0.0
    worst = dict.fromkeys(LIMITS, 0.0)
    for step in range(args.steps):
        if case == "heated":
            # Known external source in the owned cells; record actual FP32
            # addition separately from requested heat so rounding is visible.
            cell = state["hydro_u"][kIPR, 3:-3, 3:-3, 15:18]
            before = cell.double().clone()
            cell += 1.0e-3
            vol = diag.vol[..., 12:15]
            source += ((cell.double() - before) * vol).sum().item()
            requested += (torch.ones_like(before) * 1.0e-3 * vol).sum().item()
        run_steps(block, state, 1, dt=args.dt)
        if (step + 1) % args.sample_every == 0 or step + 1 == args.steps:
            values = diag.read(state)
            assert all(math.isfinite(x) for x in values.values()), values
            for key in LIMITS:
                residual = values[key] - initial[key] - (source if key == "energy" else 0.0)
                error = abs(residual) / abs(initial[key])
                worst[key] = max(worst[key], error)
                limit = LIMITS[key] if precision == "float32" else 2.0e-11
                assert error < limit, (case, scheme, seed, precision, step + 1, key, error, limit)
            requested_error = abs(values["energy"] - initial["energy"] - requested) / abs(
                initial["energy"]
            )
            worst["energy_with_requested_source"] = max(
                worst.get("energy_with_requested_source", 0.0), requested_error
            )
            assert requested_error < (LIMITS["energy"] if precision == "float32" else 2.0e-11)
            samples.append({"time": (step + 1) * args.dt, **values})
    count = fallbacks(block, scheme)
    assert count == 0, (case, scheme, seed, "unexpected FP64 fallback", count)
    final = block.total_u(state)[diag.sl].double().cpu()
    tail = samples[len(samples) // 2 :]
    metrics = ("kinetic_energy", "vertical_velocity_rms", "vapor_mass")
    result = {
        "precision": precision,
        "scheme": scheme,
        "seed": seed,
        "case": case,
        "max_relative_budget_residual": worst,
        "source_energy_actual": source,
        "source_energy_requested": requested,
        "relative_source_rounding_error": (source - requested) / requested if requested else 0.0,
        "fp64_fallbacks": count,
        "late_time_means": {k: statistics.mean(v[k] for v in tail) for k in metrics},
        "samples": samples,
    }
    print(f"{case} scheme={scheme} seed={seed} {precision}: budgets={worst}", flush=True)
    return result, final


def benchmark(precision, scheme, args):
    gc.collect()
    block, state = prepare(precision, scheme, 0, "closed", args.device)
    run_steps(block, state, 5, dt=args.dt)
    synchronize(args.device)
    resident = memory_bytes(block, state)
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run_steps(block, state, args.benchmark_steps, dt=args.dt)
        synchronize(args.device)
        times.append((time.perf_counter() - start) / args.benchmark_steps)
    peak = torch.cuda.max_memory_allocated(args.device) if args.device.type == "cuda" else None
    result = {
        "precision": precision,
        "scheme": scheme,
        "seconds_per_step": times,
        "median_seconds_per_step": statistics.median(times),
        "registered_buffer_and_state_bytes": resident,
        "cuda_peak_allocated_bytes": peak,
        "fp64_fallbacks": fallbacks(block, scheme),
    }
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            run_steps(block, state, 5, dt=args.dt)
            synchronize(args.device)
        events = prof.key_averages()
        result["profile_self_cpu_ms"] = {v.key: v.self_cpu_time_total / 1000.0 for v in events}
        trace = args.output.parent / f"fp32-profile-{precision}-scheme{scheme}.json"
        prof.export_chrome_trace(str(trace))
        result["trace_file"] = trace.name
    print(
        f"benchmark scheme={scheme} {precision}: {result['median_seconds_per_step']:.6f} s/step",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--schemes", type=int, nargs="+", choices=(0, 9), default=[0, 9])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("closed", "heated", "sediment"),
        default=["closed", "heated", "sediment"],
    )
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--jupiter-steps", type=int, default=0)
    parser.add_argument(
        "--source-revision", help="fixture revision for a source snapshot without .git"
    )
    parser.add_argument("--output", type=Path, default=Path("fp32-science.json"))
    args = parser.parse_args()
    if (
        min(args.steps, args.seeds, args.sample_every, args.repeats) <= 0
        or args.dt <= 0
        or not math.isfinite(args.dt)
        or args.benchmark_steps < 0
        or args.jupiter_steps < 0
    ):
        parser.error(
            "step/seed/sample/repeat counts and dt must be positive; benchmark steps nonnegative"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no CPU substitution")
    if args.profile and not args.benchmark_steps:
        parser.error("--profile requires --benchmark-steps")
    torch.set_num_threads(1)
    args.device = torch.device(args.device)
    os.environ["DEVICE"] = args.device.type
    os.environ["DEVICE_ID"] = "0"
    os.environ["SNAP_TEST_DEVICE"] = str(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        modified = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        )
    except subprocess.CalledProcessError:
        revision, modified = args.source_revision, None
    report = {
        "schema_version": 1,
        "scope": "reduced Cartesian moist engineering validation",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "kintera": importlib.metadata.version("kintera"),
        "git_revision": revision,
        "working_tree_modified": modified,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapy": importlib.metadata.version("snapy"),
        "native_libraries_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(snapy.__file__).parent / "lib").glob("libsnap*")
            if p.is_file()
        },
        "device": str(args.device),
        "threads": torch.get_num_threads(),
        "steps": args.steps,
        "dt": args.dt,
        "seeds": args.seeds,
        "benchmark_steps": args.benchmark_steps,
        "benchmark_repeats": args.repeats,
        "benchmark_warmup_steps": 5,
        "sample_every": args.sample_every,
        "budget_limits_fp32": LIMITS,
        "runs": [],
        "comparisons": [],
        "benchmarks": [],
        "ensembles": [],
    }
    if args.device.type == "cuda":
        report["device_name"] = torch.cuda.get_device_name(args.device)
    for case in args.cases:
        for scheme in args.schemes:
            paired = {p: [] for p in ("float64", "float32")}
            for seed in range(args.seeds):
                outputs = {}
                for precision in paired:
                    result, outputs[precision] = evolution(precision, scheme, seed, case, args)
                    paired[precision].append(result["late_time_means"])
                    report["runs"].append(result)
                oracle = outputs["float64"]
                scale = oracle.abs().amax((1, 2, 3), keepdim=True).clamp_min(1.0)
                error = ((outputs["float32"] - oracle).abs() / scale).amax().item()
                assert error < 2.0e-4, (case, scheme, seed, "paired state error", error)
                report["comparisons"].append(
                    {
                        "case": case,
                        "scheme": scheme,
                        "seed": seed,
                        "max_normalized_state_error": error,
                    }
                )
            for metric in paired["float64"][0]:
                values = {p: [v[metric] for v in paired[p]] for p in paired}
                means = {p: statistics.mean(v) for p, v in values.items()}
                report["ensembles"].append(
                    {
                        "case": case,
                        "scheme": scheme,
                        "metric": metric,
                        "means": means,
                        "sample_stdevs": {
                            p: statistics.stdev(v) if len(v) > 1 else None
                            for p, v in values.items()
                        },
                        "paired_mean_difference": means["float32"] - means["float64"],
                    }
                )
    if args.benchmark_steps:
        for scheme in args.schemes:
            for precision in ("float64", "float32"):
                report["benchmarks"].append(benchmark(precision, scheme, args))
    # Preserve completed measurements even if a later acceptance gate fails.
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.jupiter_steps:
        from test_fp32_jupiter import run as jupiter

        states = {p: jupiter(args.jupiter_steps, p) for p in ("float64", "float32")}
        oracle = states["float64"]["total"]
        delta = states["float32"]["total"] - oracle
        scale = oracle.abs().amax((1, 2, 3), keepdim=True).clamp_min(1.0)
        error = (delta.abs() / scale).amax().item()
        families = {"water": [kICY, kICY + 2, kICY + 3], "ammonia": [kICY + 1, kICY + 4, kICY + 5]}
        differences = {}
        for name, rows in families.items():
            base = oracle[rows].sum().item()
            differences[name] = delta[rows].sum().item() / base
        scalar64 = states["float64"]["scalar_s"].double()
        scalar_error = (
            (states["float32"]["scalar_s"].double() - scalar64).abs().max() / scalar64.abs().max()
        ).item()
        report["jupiter"] = {
            "steps": args.jupiter_steps,
            "dt": 0.1,
            "max_normalized_state_error": error,
            "relative_family_mass_difference": differences,
            "relative_scalar_error": scalar_error,
            "passed": error < 2.0e-4
            and all(abs(v) < 2.0e-4 for v in differences.values())
            and scalar_error < 2.0e-4,
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        assert report["jupiter"]["passed"], report["jupiter"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
