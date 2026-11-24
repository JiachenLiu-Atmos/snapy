"""The python tests' check for a CUDA part: a CUDA build of snapy and a visible
CUDA device. tests/CMakeLists.txt sets SNAPY_BUILD_CUDA=0 for every test of a
build without CUDA, so a CPU build skips the CUDA part even when a GPU is
visible (torch's own availability is not the build, as in cuda_test_gate.hpp).
skip() ends the test with SKIP_RETURN_CODE, which ctest reports as skipped.
"""
import os
import sys

import torch

SKIP_RETURN_CODE = 125


def build_has_cuda():
    return os.environ.get("SNAPY_BUILD_CUDA", "1") != "0" and torch.cuda.is_available()


def skip(why):
    print("SKIP:", why, flush=True)
    sys.exit(SKIP_RETURN_CODE)
