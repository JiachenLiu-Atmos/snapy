"""V3 A3b: a user TorchScript stage forcing under integration/execution: cuda-graph.

A user forcing is opaque, so it is a per-op refusal: the stage's forcing op
runs eagerly between two graphs (the tendencies, then the combine and the
end-of-stage slots), and the tendencies cross the boundary in the block's
fixed stage.du / stage.ds storage. A strict case is refused before its first
launch, naming the forcing segment. A non-strict case equals an eager block
bitwise, FP64 and FP32, with dt changing every step, including after the
forcing is set on a block that has already captured (the schedule is rebuilt
and the block's graphs dropped). The case is the dry Straka deck of
run_fp32_straka.make_case, whose tracer exercises the user dry carry.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fp32_straka import make_case  # noqa: E402
from cuda_build import build_has_cuda, skip  # noqa: E402

REASON = "a user TorchScript stage forcing is set"
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


class _Heating(torch.nn.Module):
    """A density-weighted heating and a small dry source, both state-dependent."""

    def forward(
        self, variables: Dict[str, torch.Tensor], dt: float, stage: int
    ) -> Dict[str, torch.Tensor]:
        hydro_u = variables["hydro_u"]
        du = torch.zeros_like(hydro_u)
        du[4] = 2.0e-3 * hydro_u[0] * (1.0 + 0.1 * stage)
        du[0] = -1.0e-7 * hydro_u[0]
        return {"hydro_du": du}


def assert_equal(a, b, where):
    for k in KEYS:
        if k in a:
            assert torch.equal(a[k], b[k]), (where, k)


def step(pair, dt):
    for block, state in pair:
        for s in range(len(block.intg.stages)):
            block.forward(state, dt, s)


def check(precision, path):
    cuda = torch.device("cuda")
    # strict: refused before its first launch, naming the forcing segment
    block, state, *_ = make_case(precision, cuda, "cuda-graph", True)
    block.set_user_stage_forcings([path])
    c = block.execution_capability(state)
    assert REASON in c["eager_reasons"] and not c["graph_fp64"] and not c["graph_fp32"], c
    before = {k: state[k].clone() for k in KEYS if k in state}
    try:
        block.forward(state, 0.1, 0)
    except RuntimeError as e:
        msg = str(e)
        assert "cannot run as a CUDA graph" in msg and "(forcing)" in msg, msg
        assert REASON in msg, msg
    else:
        raise AssertionError("a strict cuda-graph case with a user forcing ran")
    for k, v in before.items():
        assert torch.equal(state[k], v), (precision, "strict touched", k)

    # not strict: one graph step without the forcing, then with it
    graph, gstate, *_ = make_case(precision, cuda, "cuda-graph")
    eager, estate, *_ = make_case(precision, cuda)
    pair = ((graph, gstate), (eager, estate))
    dt0 = eager.max_time_step(estate)
    nst = len(graph.intg.stages)
    step(pair, 0.5 * dt0)
    assert_equal(gstate, estate, (precision, "before the forcing"))
    r = graph.execution_report()
    assert r["captures"] == nst and r["eager"] == 0 and r["graphs"] == nst, r

    graph.set_user_stage_forcings([path])
    eager.set_user_stage_forcings([path])
    rows = graph.execution_capability(gstate)["segments"]
    assert [row["ops"] for row in rows] == [
        ["exchange"], ["advance"], ["forcing"],
        ["combine", "saturation", "chemistry", "boundaries"]], rows
    assert rows[1]["capturable"] and rows[3]["capturable"], rows
    assert not rows[1]["reads_host_dt"] and not rows[3]["reads_host_dt"], rows
    assert not rows[2]["capturable"] and rows[2]["reasons"] == [REASON], rows
    nsteps = 5
    for n in range(nsteps):
        step(pair, 0.5 * dt0 * (1.0 + 0.1 * ((3 * n) % 4)))
        assert_equal(gstate, estate, (precision, "step", n))
    r = graph.execution_report()
    # one eager forcing segment per stage; two graphs per stage, captured
    # once after the rebuild and replayed as dt changes (dt is a graph input,
    # V3 D; before D both read the host dt and recaptured every step)
    assert r["eager"] == nsteps * nst, r
    assert r["graphs"] == 2 * nst, r
    assert r["captures"] == nst + 2 * nst, r
    assert r["replays"] == nst + 2 * nst * nsteps, r
    assert r["host"] == (nsteps + 1) * nst, r
    # the forcing did something
    ref, rstate, *_ = make_case(precision, cuda)
    for n in range(nsteps + 1):
        dt = 0.5 * dt0 if n == 0 else 0.5 * dt0 * (1.0 + 0.1 * ((3 * (n - 1)) % 4))
        for s in range(nst):
            ref.forward(rstate, dt, s)
    assert not torch.equal(rstate["hydro_u"], estate["hydro_u"]), "the forcing did nothing"
    print(precision, "user forcing: strict refused; graph + eager forcing + graph "
          "equals eager over", nsteps + 1, "steps:", r)


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    with TemporaryDirectory(prefix="snapy-graph-forcing-") as d:
        path = str(Path(d) / "heating.pt")
        torch.jit.script(_Heating().eval()).save(path)
        for precision in ("float64", "float32"):
            check(precision, path)
    print("PASSED")


if __name__ == "__main__":
    main()
