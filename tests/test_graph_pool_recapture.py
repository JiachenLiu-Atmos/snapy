"""A block's graphs share one memory pool (V3 A3a); recapturing after every
graph of the block was dropped must work.

Dropping all of a block's graphs (MeshBlock.to(), a restart, a schedule
rebuild) can leave tensors that were allocated in the pool alive, for
instance scalar_r, which a replay sets to a graph output. Capturing into the
same pool handle after its last graph is gone then trips the CUDA caching
allocator ("use_count > 0" internal assert). The executor must start a fresh
pool for such a block. Checked on the dry Straka deck (it carries a tracer):
graph steps, then block.to(cuda) (drops the graphs), then more graph steps,
bitwise to an eager block throughout, FP64 and FP32.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fp32_straka import make_case  # noqa: E402

KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def step(pair, dt):
    for block, state in pair:
        for s in range(len(block.intg.stages)):
            block.forward(state, dt, s)


def check(precision):
    cuda = torch.device("cuda")
    graph, gstate, *_ = make_case(precision, cuda, "cuda-graph", True)
    eager, estate, *_ = make_case(precision, cuda)
    pair = ((graph, gstate), (eager, estate))
    dt = 0.5 * eager.max_time_step(estate)
    nst = len(graph.intg.stages)
    for n in range(2):
        step(pair, dt)
    assert graph.execution_report()["graphs"] == nst
    graph.to(cuda)  # drops every graph of the block
    assert graph.execution_report()["graphs"] == 0
    for n in range(3):
        step(pair, dt)
        for k in KEYS:
            if k in gstate:
                assert torch.equal(gstate[k], estate[k]), (precision, n, k)
    r = graph.execution_report()
    assert r["captures"] == 2 * nst and r["graphs"] == nst, r
    # a bare block's exchanges involve no other block: no cut (V3 B3), its
    # tracer's theta sync stays inside the one graph per stage
    assert r["cuts"] == 0, r
    print(precision, "recaptured after the block's graphs were dropped:", r)


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return
    for precision in ("float64", "float32"):
        check(precision)
    print("PASSED")


if __name__ == "__main__":
    main()
