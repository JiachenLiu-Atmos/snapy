#!/usr/bin/env python3
"""Gate G-6 (lane O, O2): the deleted int-only flip_zero against
bc/flip_zero_impl.h<int>, exhaustively (1,179,576 cases), on cpu or cuda.

  g6_flip_zero.py --repo SNAPY [--old-ref REF] [--new-ref REF]
                  [--device cpu|cuda] [--work DIR] [--cxx C++] [--nvcc NVCC]

--old-ref must still have src/utils/flip_zero.cpp (main; default
730b35007d16389c64116e4bf9d2fe93707e29e9). --new-ref is the step whose
src/bc/flip_zero_impl.h is checked (default HEAD of --repo). The two files
are taken with `git show`, so no checkout or build of snapy is needed.
Exit status 0 pass, 1 differences, 2 setup error, 3 CUDA error.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = "730b35007d16389c64116e4bf9d2fe93707e29e9"

CONFIGURE_H = """#pragma once
// DISPATCH_MACRO as snapy's configure.h.in defines it
#ifdef __CUDACC__
#define DISPATCH_MACRO __host__ __device__
#else
#define DISPATCH_MACRO
#endif
"""


def git_show(repo, ref, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="a snapy git checkout")
    p.add_argument("--old-ref", default=MAIN)
    p.add_argument("--new-ref", default="HEAD")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--work", help="build directory (default: a temporary one)")
    p.add_argument("--cxx", default="c++")
    p.add_argument("--nvcc", default="nvcc")
    a = p.parse_args()

    work = Path(a.work or tempfile.mkdtemp(prefix="g6_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        old_sha = subprocess.run(["git", "-C", a.repo, "rev-parse", a.old_ref],
                                 capture_output=True, text=True, check=True).stdout.strip()
        new_sha = subprocess.run(["git", "-C", a.repo, "rev-parse", a.new_ref],
                                 capture_output=True, text=True, check=True).stdout.strip()
        (work / "old_flip_zero.cpp").write_text(
            git_show(a.repo, a.old_ref, "src/utils/flip_zero.cpp"))
        (work / "flip_zero_impl.h").write_text(
            git_show(a.repo, a.new_ref, "src/bc/flip_zero_impl.h"))
    except subprocess.CalledProcessError as e:
        print(f"setup: {e.stderr.strip() or e}")
        return 2
    (work / "configure.h").write_text(CONFIGURE_H)
    shutil.copy(HERE / "g6_flip_zero.cpp", work / "g6_flip_zero.cpp")
    print(f"old: {old_sha}:src/utils/flip_zero.cpp")
    print(f"new: {new_sha}:src/bc/flip_zero_impl.h")

    exe = work / f"g6_{a.device}"
    if a.device == "cpu":
        cmd = [a.cxx, "-std=c++17", "-O2", f"-I{work}", str(work / "g6_flip_zero.cpp"),
               "-o", str(exe)]
    else:
        cmd = [a.nvcc, "-std=c++17", "-O2", "-x", "cu", f"-I{work}",
               str(work / "g6_flip_zero.cpp"), "-o", str(exe)]
    print("+", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        return 2
    return subprocess.run([str(exe)]).returncode


if __name__ == "__main__":
    sys.exit(main())
