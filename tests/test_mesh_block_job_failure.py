"""A block job that fails on one block of a Mesh never hangs the other blocks.

The six-panel cubed sphere of test_default_execution with 2 x 2 blocks per
panel (24 blocks): its stages hold panel-seam exchanges, so the blocks meet
at the same-process waits (the rendezvous, the seam mail). Each arm caps the
caching allocator (set_per_process_memory_fraction) after the Mesh is built
and runs in a child process under a timeout, so a hang fails the test:

  capture  default and cuda-graph execution, the cap 20 MiB above the built
           Mesh: a capture runs out of device memory, and the caller gets one
           out-of-memory error that names execution: eager
  eager    eager execution, 256 x 256 cells per block: after one step the
           cache is emptied and capped 40 MiB above what stays, so some blocks
           run out of memory while the others go on to the seam exchanges;
           the caller gets that error as it is

  python test_mesh_block_job_failure.py
"""
import os
import subprocess
import sys

import torch

from cuda_build import build_has_cuda, skip

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 120


def child(arm):
    sys.path.insert(0, HERE)
    import test_default_execution as t

    execution = {"default": None, "cuda-graph": "cuda-graph", "eager": "eager"}[arm]
    mesh, blocks, mvars = t.cubed(2, execution, 256 if arm == "eager" else 8)
    if arm == "eager":
        for s in range(3):
            mesh.forward(mvars, 1.0e-6, s)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    cap = torch.cuda.memory_reserved() + (40 if arm == "eager" else 20) * 2**20
    torch.cuda.set_per_process_memory_fraction(
        cap / torch.cuda.get_device_properties(0).total_memory, 0)
    try:
        for _ in range(2):
            for s in range(3):
                mesh.forward(mvars, 1.0e-6, s)
    except Exception as e:
        print("RAISED " + type(e).__name__ + ": " + str(e).replace("\n", " "), flush=True)
        return
    print("FINISHED " + str(mesh.execution_report()), flush=True)


def run(arm):
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", arm],
                             capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{arm}: the Mesh hung, no result within {TIMEOUT} s")
    lines = [l for l in out.stdout.splitlines() if l.startswith(("RAISED", "FINISHED"))]
    assert out.returncode == 0 and lines, (arm, out.returncode, out.stdout[-2000:],
                                           out.stderr[-2000:])
    print(arm + ": " + lines[-1][:400], flush=True)
    return lines[-1]


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return 0
    if not build_has_cuda():
        skip("CUDA unavailable")
    for arm in ("default", "cuda-graph"):
        line = run(arm)
        assert line.startswith("RAISED OutOfMemoryError") and "execution: eager" in line, line
    line = run("eager")
    assert line.startswith("RAISED OutOfMemoryError") and "execution: eager" not in line, line
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
