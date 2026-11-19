"""V3 X3: the stage executor (integration.execution: cuda-graph) on the dry
Straka case against eager, bitwise, in FP64 and FP32.

Like kintera's K-G3 gate, every forward() of the graph block runs under
torch.cuda.set_sync_debug_mode("error"): its captures and replays must not
synchronize the host (the eager warm-up inside a capture is exempt, as eager
stages are: a block's first eager stage syncs once). The eager reference runs
under the default mode. The run changes dt, rebinds hydro_u/scalar_s between
steps, edits the state in place and forces a retry. On CPU only the setup
rules are checked: eager plus strict is refused at setup, and a strict
cuda-graph case that cannot be captured is refused before its first launch.
"""

import torch

from run_fp32_straka import make_case

KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def expect_error(fn, text):
    try:
        fn()
    except RuntimeError as e:
        assert text in str(e), str(e)
        return str(e)
    raise AssertionError(f"no error containing {text!r}")


def check_setup():
    cpu = torch.device("cpu")
    expect_error(lambda: make_case("float64", cpu, "eager", True), "execution-strict")

    block, state, *_ = make_case("float64", cpu, "cuda-graph", True)
    before = {k: state[k].clone() for k in KEYS}
    msg = expect_error(lambda: block.forward(state, 0.1, 0), "cannot run as a CUDA graph")
    assert "not on a CUDA device" in msg, msg
    for k in KEYS:  # refused before its first launch
        assert torch.equal(state[k], before[k]), k

    # without strict the same case runs eager
    block, state, *_ = make_case("float64", cpu, "cuda-graph")
    ref, expected, *_ = make_case("float64", cpu)
    for stage in range(len(block.intg.stages)):
        block.forward(state, 0.1, stage)
        ref.forward(expected, 0.1, stage)
    for k in KEYS:
        assert torch.equal(state[k], expected[k]), k
    report = block.execution_report()
    assert report["captures"] == 0 and report["eager"] == 3, report
    print("setup: eager+strict refused; strict cuda-graph on CPU refused before launch")


def forward_both(pair, dt):
    (block, state), (ref, expected) = pair
    for stage in range(len(block.intg.stages)):
        torch.cuda.set_sync_debug_mode("error")
        try:
            block.forward(state, dt, stage)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        ref.forward(expected, dt, stage)


def assert_equal(state, expected, where):
    for k in KEYS:
        assert torch.equal(state[k], expected[k]), (where, k)


def check_capture(precision):
    cuda = torch.device("cuda")
    block, state, *_ = make_case(precision, cuda, "cuda-graph", True)
    ref, expected, *_ = make_case(precision, cuda)
    pair = ((block, state), (ref, expected))
    nstages = len(block.intg.stages)

    recaptures, last_dt = 0, None
    for step in range(40):
        dt = 0.05 if 10 <= step < 20 else 0.1
        if step == 25:  # a rebind between steps: X2 adopts, the graph recaptures
            for fields in (state, expected):
                fields["hydro_u"] = fields["hydro_u"].clone()
                fields["scalar_s"] = fields["scalar_s"].clone()
            recaptures += 1
        elif dt != last_dt:
            recaptures += 1
        if step == 30:  # an in-place edit reaches the captured inputs
            for fields in (state, expected):
                fields["hydro_u"][0, 0, 20, 20] += 1.0e-4
        last_dt = dt
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0
        assert_equal(state, expected, (precision, step))

    # a forced retry restores the step; the smaller dt recaptures
    before = {k: state[k].clone() for k in ("hydro_u", "scalar_s")}
    forward_both(pair, 0.1)
    for fields in (state, expected):
        fields["hydro_u"][0, 0, 20, 20] = float("nan")
    assert block.check_redo(state) == ref.check_redo(expected) == 1
    for k, v in before.items():
        assert torch.equal(state[k], v), (precision, "restore", k)
        assert torch.equal(expected[k], v), (precision, "restore eager", k)
    for dt in (0.05, 0.05):
        forward_both(pair, dt)
        assert block.check_redo(state) == ref.check_redo(expected) == 0
        assert_equal(state, expected, (precision, "after retry"))
    recaptures += 1

    report = block.execution_report()
    assert report["eager"] == 0, report
    assert report["captures"] == nstages * recaptures, (report, recaptures)
    assert report["replays"] == nstages * 43, report
    print(precision, "graph equals eager over 43 steps:", report)


def main():
    check_setup()
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable (capture part)")
        return
    for precision in ("float64", "float32"):
        check_capture(precision)


if __name__ == "__main__":
    main()
