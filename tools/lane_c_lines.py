#!/usr/bin/env python3
"""Count lane C's source lines against its ceiling (V3 C.rst section 9).

Lane C is every file under src/recon and src/riemann plus the well-balanced
reference files in src/hydro; documentation moved to docs/ does not count.
Lines are counted as ``wc -l`` does. Exits 1 above the ceiling.

    python3 tools/lane_c_lines.py [--root DIR] [--ceiling N] [--files]
"""

import argparse
import pathlib
import sys

CEILING = 4672  # 20% below the 5,841-line lane C baseline
DIRS = ("src/recon", "src/riemann")
HYDRO = (
    "balance_column.cpp",
    "balance_column.hpp",
    "hydro_dispatch.cpp",
    "hydro_dispatch.cu",
    "hydro_dispatch.hpp",
    "hydro_ref_x1_impl.h",
    "hydro_reference.hpp",
    "split_reference_scalar.hpp",
    "wb_ref4.cpp",
    "wb_ref4.hpp",
)


def lane_files(root):
    files = [f for d in DIRS for f in (root / d).iterdir() if f.is_file()]
    files += [root / "src/hydro" / n for n in HYDRO]
    return sorted(f for f in files if f.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=pathlib.Path(__file__).parents[1])
    ap.add_argument("--ceiling", type=int, default=CEILING)
    ap.add_argument("--files", action="store_true", help="per-file counts")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    total = 0
    files = lane_files(root)
    for f in files:
        n = f.read_bytes().count(b"\n")
        total += n
        if args.files:
            print(f"{n:6d} {f.relative_to(root)}")
    verdict = "under" if total <= args.ceiling else "OVER"
    print(f"lane C: {total} lines in {len(files)} files; ceiling "
          f"{args.ceiling}: {verdict} by {abs(args.ceiling - total)}")
    return 0 if total <= args.ceiling else 1


if __name__ == "__main__":
    sys.exit(main())
