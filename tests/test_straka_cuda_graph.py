"""CUDA replay correctness, invalidation and rejection for the Straka fixture."""

import copy

import torch

from run_fp32_straka import make_case
from straka_cuda_graph import StrakaGraphStep


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return
    for precision in ("float64", "float32"):
        block, state, cfg, *_ = make_case(precision, torch.device("cuda"))
        ref, expected, *_ = make_case(precision, torch.device("cuda"))
        for field, value in (("limiter", True),):
            bad = copy.deepcopy(cfg)
            bad["dynamics"]["equation-of-state"][field] = value
            try:
                StrakaGraphStep(block, state, 0.1, bad)
            except ValueError:
                pass
            else:
                raise AssertionError("unsupported limiter accepted")
        graph = StrakaGraphStep(block, state, 0.1, cfg)
        for step in range(100):
            dt = 0.05 if 20 <= step < 40 else 0.1
            if step == 50:
                for fields in (state, expected):
                    fields["hydro_u"] = fields["hydro_u"].clone()
                    fields["scalar_s"] = fields["scalar_s"].clone()
            if step == 60:
                # An in-place edit must reach the captured inputs too.
                for fields in (state, expected):
                    fields["hydro_u"][0, 0, 20, 20] += 0.0001
            graph(dt)
            for stage in range(len(ref.intg.stages)):
                ref.forward(expected, dt, stage)
            assert block.check_redo(state) == ref.check_redo(expected) == 0
            for key in ("hydro_u", "hydro_w", "scalar_s", "scalar_r"):
                assert torch.equal(state[key], expected[key]), (precision, step, key)
        assert graph.captures == 4, graph.captures
        # The replay's native RK backup must restore the immediately preceding
        # accepted state, not the warmup/capture state or a stale first step.
        for attempt in range(6):
            before = {k: state[k].clone() for k in ("hydro_u", "scalar_s")}
            graph(0.1 / (2**attempt))
            state["hydro_u"][0, 0, 20, 20] = float("nan")
            status = block.check_redo(state)
            if attempt == 5:
                # The fixture uses the default five-retry budget. Warmup must
                # not reset current_redo and permit retries indefinitely.
                assert status == -1, (precision, "retry budget", status)
            else:
                assert status == 1
                for k, v in before.items():
                    assert torch.equal(state[k], v), (precision, "restore", k)
        print(
            precision, "100 matching replays; dt/storage/in-place changes; retry restoration passed"
        )


if __name__ == "__main__":
    main()
