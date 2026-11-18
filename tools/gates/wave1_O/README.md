# Wave 1 lane O gates: G-1 and G-6

Gate tooling for lane O's wave 1 steps (O1, O2, O2b, O3, O4; O.rst section 5 at
JiachenLiu-Atmos/snapy `36b61c401fd5f5d2a6917b9e05dcce3a00022815`). No snapy
source is changed. Base: chengcli/snapy main
`730b35007d16389c64116e4bf9d2fe93707e29e9`. Lane O head:
JiachenLiu-Atmos/snapy `v3/wave1-O` `239b98147e6ecf1889e2bf64348ebb98ac63c802`.

| file | gate |
|---|---|
| `g1_decks.py` | G-1: example decks, 3 steps, sha256 of every written file, base vs step, with a base vs base control |
| `g6_flip_zero.py`, `g6_flip_zero.cpp` | G-6: the deleted `src/utils/flip_zero.cpp` against `src/bc/flip_zero_impl.h<int>`, 1,179,576 cases |

Both take `--device cpu|cuda`. A CUDA run compares CUDA results only with CUDA
results: G-1 runs the base and the step on the same device, and G-6 runs both
implementations in the same CUDA kernel.

## Requirements

- Python 3 with PyYAML. G-1 also needs `ninja` on PATH, because it rebuilds a
  seeded `run_hydro` from the tree's own compile and link lines.
- G-1: two snapy checkouts, each built with
  `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DBUILD_EXAMPLES=ON [-DCUDA=ON]`
  and `cmake --build build`. One is the base (main) and one is the step (e.g.
  `239b9814`). Configure both builds the same way. For `--device cuda`, both
  builds need CUDA on.
- G-6: one git checkout of snapy that has both refs (it reads the two files
  with `git show`; nothing is built), a C++17 compiler, and `nvcc` for
  `--device cuda`.

## G-1

```
python tools/gates/wave1_O/g1_decks.py gate \
    --base /path/to/snapy-main --step /path/to/snapy-239b9814 \
    --out /tmp/g1_cpu --device cpu

python tools/gates/wave1_O/g1_decks.py gate \
    --base /path/to/snapy-main --step /path/to/snapy-239b9814 \
    --out /tmp/g1_cuda --device cuda
```

- `--base-build` and `--step-build` point at build dirs other than `<tree>/build`.
- `--build-type` is the target suffix (default `release`).
- `--multiprocess` adds the 12 decks that need 2-6 processes. They run through
  `torchrun --standalone`, so they need local sockets and, on CUDA, enough
  GPUs. Not run by the author; see below.
- `run` and `compare` are the two halves, for runs made separately:

```
python g1_decks.py run --tree TREE --out RUN --device cpu|cuda [deck ...]
python g1_decks.py compare RUN_A RUN_B [deck ...]
```

What it does, per deck:

- It copies the deck with `integration/nlim: 3` and a restart output with
  `dt: 1e-9`, which writes a restart dump every step.
- It runs the deck's own driver with `OMP_NUM_THREADS=1`, `DEVICE=<device>` and
  `BACKEND=gloo`.
- run_hydro decks (`earth_crm`, `uranus`) use `bin/run_hydro_seeded.release`.
  That's the tree's `examples/run_hydro.cpp` plus `torch::manual_seed(1234)`,
  built from run_hydro's own ninja compile and link lines. run_hydro's
  `rand_like` noise is unseeded otherwise.
- The three convection benchmarks stop at an end time, not at nlim, so they run
  with fixed short arguments. `planar_convection` and `spherical_convection` get
  `cuda:0` under `--device cuda`. `cubed_sphere_convection` has no device
  argument and runs on the CPU under either device.
- It records the sha256 of every file the run writes.

Compare rule:

- Every file must be bitwise equal.
- The one exception is a JSON file whose only difference is the wall-clock field
  `seconds` (the convection benchmarks' `stats.csv.json`). It is compared with
  that field removed.
- The run's exit codes and file sets must match.

`gate` runs the base twice (A and B) and the step once.

- Control: base A vs base B.
- Gate: base A vs step.
- Exit status:
  - 0: pass;
  - 1: a file differs from the base, and it was stable in the control;
  - 2: inconclusive, because the control itself differs;
  - 3: a run or build failed.
- `summary.json` in `--out` lists the result per deck.

Expected output on CPU, for the base and each of O1, O2, O2b, O3 and O4 (macOS,
Apple clang 21, torch 2.10.0, kintera 2.6.10), in both the control and the gate:

```
total: 13 decks, 63 files, 61 bitwise, 2 equal without the wall-clock field, 0 decks differ or failed
G-1 PASS (cpu): 13 decks equal to the base; control equal
```

The 2 files are `planar_convection/stats.csv.json` and
`spherical_convection/stats.csv.json`. On CUDA the numbers come from the CUDA
run itself. If the control differs (non-deterministic kernels), the gate
reports it as inconclusive (exit 2), not as a pass.

The single-process deck list: bryan, cubed_sphere_convection, earth_crm,
jupiter_evap_precip_1d, planar_convection, shallow_splash_mesh6,
shallow_xy_mesh4, shallow_xy_single, shock, spherical_convection,
straka_mesh2, straka_single, uranus.

Multi-process decks (`--multiprocess`), with their process counts:

- 4: jupiter_crm, jupiter_crm_dry, shallow_xy, shallow_xy_proc4
- 6: jupiter_gcm, jupiter_gcm_dry, shallow_splash
- 2: shallow_splash_proc2_mesh3, shallow_xy_proc2_mesh2, straka, straka_proc2
- 3: shallow_splash_proc3_mesh2

The author's machine blocks local sockets, so these were never run there.

## G-6

```
python tools/gates/wave1_O/g6_flip_zero.py --repo /path/to/snapy \
    --old-ref 730b35007d16389c64116e4bf9d2fe93707e29e9 \
    --new-ref 239b98147e6ecf1889e2bf64348ebb98ac63c802 --device cpu

python tools/gates/wave1_O/g6_flip_zero.py --repo /path/to/snapy \
    --old-ref 730b35007d16389c64116e4bf9d2fe93707e29e9 \
    --new-ref 239b98147e6ecf1889e2bf64348ebb98ac63c802 --device cuda
```

- `--repo` is any snapy git checkout that contains both refs. Fetch
  JiachenLiu-Atmos/snapy `v3/wave1-O` for `239b9814`.
- `--work DIR` keeps the build directory.
- `--cxx` and `--nvcc` choose the compilers.

The script writes the two files next to `g6_flip_zero.cpp`, with a `configure.h`
that defines `DISPATCH_MACRO` as snapy does. It compiles them into one program,
each inside its own namespace, and runs every case:

- sequence lengths 1-14, all bit patterns;
- minRun0 and minRun1 in 1-3;
- both flip modes;
- stride 1 and 2.

Each case compares the minimum flip count and the fixed sequence. The new copy
with `dir = -1` on the reversed input must give the same minimum count.

Under `--device cuda`, nvcc builds both copies as `__host__ __device__` and runs
one case per CUDA thread. The host only tallies the results.

Expected output (both devices):

```
dir=+1: 1179576 cases (64 impossible), 0 mismatches (count and sequence)
dir=-1 on reversed input: 589788 cases, 0 min-flip mismatches
G-6 PASS
```

Exit status: 0 pass, 1 differences, 2 setup or compile error, 3 CUDA error.
To check that the harness can fail, change `len >= needed` to
`len >= needed - 1` in the extracted `flip_zero_impl.h` (`--work DIR`, then
rebuild by hand). That gives 825,394 mismatches.

The CUDA paths of both scripts were written for this gate but not run by the
author, who has no CUDA.
