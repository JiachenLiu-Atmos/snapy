"""#314: an outflow boundary under integration/execution: cuda-graph.

The outflow (radiating) ghost fill checks its inputs on the host, which a CUDA
graph capture cannot do. The pre-capture check (the block's capability row)
names the outflow boundary, so a strict case is refused before its first
launch and a non-strict case runs eagerly. The case is the dry Straka deck of
run_fp32_straka.make_case with x2 outflow; the same deck with its reflecting
walls is the control. The capability row is checked on every build (CPU
too); the launch part needs a CUDA device and is skipped without one.
"""

import os
import tempfile
from pathlib import Path

import torch
import yaml

from snapy import MeshBlock, MeshBlockOptions

REASON = "an outflow boundary is set"
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def make(device, outflow, execution="cuda-graph", strict=False):
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
    cfg["dynamics"]["riemann-solver"] = {"type": "hllc"}
    cfg["dynamics"]["equation-of-state"] = {"type": "ideal-moist", "limiter": False}
    if outflow:
        cfg["boundary-condition"]["external"]["x2-inner"] = "outflow"
        cfg["boundary-condition"]["external"]["x2-outer"] = "outflow"
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
    w = block.module("hydro").buffer("D").clone().zero_()
    z = block.module("coord").buffer("x1v").view(1, 1, -1)
    w[0] = 1.0 + 0.0 * z
    w[4] = 1.0e5 * (1.0 - z / 2.0e4)
    state, _ = block.initialize({"hydro_w": w, "scalar_r": torch.zeros_like(w[:1])})
    return block, state


def check_row(device):
    for outflow in (False, True):
        block, state = make(device, outflow)
        reasons = block.execution_capability(state)["eager_reasons"]
        assert (REASON in reasons) is outflow, (device, outflow, reasons)


def step(block, state):
    for stage in range(len(block.intg.stages)):
        block.forward(state, 0.1, stage)


def check_launch():
    cuda = torch.device("cuda")
    # strict: refused before its first launch, the state untouched
    block, state = make(cuda, True, strict=True)
    c = block.execution_capability(state)
    assert REASON in c["eager_reasons"] and not c["graph_fp64"], c
    before = {k: state[k].clone() for k in KEYS if k in state}
    try:
        block.forward(state, 0.1, 0)
    except RuntimeError as e:
        assert "cannot run as a CUDA graph" in str(e) and REASON in str(e), str(e)
    else:
        raise AssertionError("a strict cuda-graph case with an outflow boundary ran")
    for k, v in before.items():
        assert torch.equal(state[k], v), k
    # not strict: eager, bitwise to an eager block
    block, state = make(cuda, True)
    ref, expected = make(cuda, True, execution="eager")
    step(block, state)
    step(ref, expected)
    for k in KEYS:
        if k in state:
            assert torch.equal(state[k], expected[k]), k
    report = block.execution_report()
    assert report["captures"] == 0 and report["eager"] == 3, report
    # the reflecting control still captures
    block, state = make(cuda, False)
    assert block.execution_capability(state)["eager_reasons"] == []
    step(block, state)
    report = block.execution_report()
    assert report["captures"] == 3 and report["eager"] == 0, report


def main():
    check_row(torch.device("cpu"))
    print("capability row: an outflow boundary is named on CPU")
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (launch part)")
        return
    check_row(torch.device("cuda"))
    check_launch()
    print("cuda-graph: outflow refused before launch (strict) or eager; reflecting captures")


if __name__ == "__main__":
    main()
