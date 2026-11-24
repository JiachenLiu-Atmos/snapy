"""V3 X7: a block's row of the capability matrix (X.rst section 10).

The dry Straka case in FP64 and FP32: on CPU it runs eager only and says why;
on CUDA it can run as CUDA graphs in its own precision, and a solid mask in
its variables makes it eager again, with the reason named.
"""

import torch

from run_fp32_straka import make_case
from cuda_build import build_has_cuda, skip


def check(precision, device, capturable):
    block, state, *_ = make_case(precision, device, "cuda-graph")
    c = block.execution_capability(state)
    fp32 = precision == "float32"
    assert c["eager_fp32"] is fp32 and c["eager_fp64"] is (not fp32), c
    assert c["graph_fp32"] is (capturable and fp32), c
    assert c["graph_fp64"] is (capturable and not fp32), c
    assert (c["eager_reasons"] == []) is capturable, c
    if not capturable:
        assert any("not on a CUDA device" in r for r in c["eager_reasons"]), c
    solid = dict(state)
    solid["solid"] = torch.zeros_like(state["hydro_w"][0], dtype=torch.bool)
    s = block.execution_capability(solid)
    assert not s["graph_fp64"] and not s["graph_fp32"], s
    assert "a solid mask is set" in s["eager_reasons"], s
    return c


def main():
    for precision in ("float64", "float32"):
        check(precision, torch.device("cpu"), False)
    print("capability matrix rows: CPU eager only")
    if not build_has_cuda():
        skip("CUDA unavailable (CUDA rows)")
    for precision in ("float64", "float32"):
        check(precision, torch.device("cuda"), True)
    print("capability matrix rows: CUDA graph in the block's precision")


if __name__ == "__main__":
    main()
