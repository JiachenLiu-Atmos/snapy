#!/usr/bin/env python3
"""G7 benchmark (V3 X7): the four precision/execution combinations on the
dry Straka case, one block on one GPU.

Each combination runs NSTEP fixed-dt steps after WARM untimed ones, REPEATS
times, with the run order alternated between repeats on the same GPU. Per
combination: median and spread of the wall time per step, the graph
captures and replays (execution_report), and the peak device memory. The
graph rows are compared with the eager row of the same precision, and FP32
with FP64 of the same execution (X.rst section 10; V3_ROADMAP.md G7).

  python tools/bench_execution.py [--nstep 50] [--repeats 5] [--warm 3]

Run from tests/ (it imports run_fp32_straka.make_case).
"""
import argparse
import statistics
import time

import torch

from run_fp32_straka import make_case

COMBOS = [("float64", "eager"), ("float64", "cuda-graph"),
          ("float32", "eager"), ("float32", "cuda-graph")]


def run(precision, execution, nstep, warm, dt):
    torch.cuda.reset_peak_memory_stats()
    block, state, *_ = make_case(precision, torch.device("cuda"), execution)
    nstage = len(block.intg.stages)

    def step():
        for stage in range(nstage):
            block.forward(state, dt, stage)

    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(nstep):
        step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / nstep
    rep = block.execution_report()
    return wall, rep, torch.cuda.max_memory_allocated()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nstep", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.1)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "the G7 benchmark needs a CUDA device"
    times = {c: [] for c in COMBOS}
    last = {}
    for r in range(a.repeats):
        order = COMBOS if r % 2 == 0 else list(reversed(COMBOS))
        for prec, exe in order:
            wall, rep, mem = run(prec, exe, a.nstep, a.warm, a.dt)
            times[(prec, exe)].append(wall)
            last[(prec, exe)] = (rep, mem)
    print(f"device {torch.cuda.get_device_name()}, {a.repeats} repeats x "
          f"{a.nstep} steps (+{a.warm} warm), order alternated")
    print(f"{'precision':9} {'execution':10} {'ms/step':>9} {'spread':>8} "
          f"{'captures':>8} {'replays':>8} {'peak MiB':>9} {'vs eager':>9}")
    for prec, exe in COMBOS:
        t = times[(prec, exe)]
        med = statistics.median(t)
        spread = (max(t) - min(t)) / med
        rep, mem = last[(prec, exe)]
        base = statistics.median(times[(prec, "eager")])
        print(f"{prec:9} {exe:10} {1e3 * med:9.3f} {100 * spread:7.1f}% "
              f"{rep['captures']:8d} {rep['replays']:8d} {mem / 2**20:9.1f} "
              f"{base / med:8.2f}x")


if __name__ == "__main__":
    main()
