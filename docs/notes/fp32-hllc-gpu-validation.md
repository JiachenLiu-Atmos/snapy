# HLLC GPU validation — 2026-11-16

## Scope and provenance

Test source commit `2f321dc` on dungeon3 GPU 0 (RTX 5090), using an isolated
build at `/tmp/snapy-hllc-2f321dc`. Native libraries and Python bindings were
rebuilt together; `/proc/self/maps` confirms both snapy libraries load from
`package/snapy/lib` in that directory. Installed snapy and the existing remote
checkout were not changed. This build includes the HLLC fix **and** the earlier
pressure-row conversion/forcing-scratch changes. Timing against the older
installed build does not isolate HLLC's contribution.

Build: Release, CUDA 13.1.115, sm_120, Torch 2.10.0+cu128, kintera 2.6.10.
UCX, NetCDF and PNetCDF remain enabled. Existing dependency sources were reused
read-only. The only scratch source adjustment restricts CMake's architecture
list to sm_120. The linker reports the CUDA 12/13 runtime combination already
present in this environment; GPU execution is validated below, not general
compatibility with other CUDA installations. Kintera source/API is unchanged.
The runner's `snapy` version field comes from installed distribution metadata;
use the source commit and native SHA256 hashes to identify this tested build.

Repository copies of the build/test scripts and raw results have been removed.
Commands below run on
dungeon3 after extracting `git archive 2f321dc` into the isolated `src` directory.

```sh
bash /tmp/snapy-hllc-2f321dc/configure-build.sh
bash /tmp/snapy-hllc-2f321dc/snapy-hllc-package.sh
```

The recorded configure script is named `snapy-hllc-configure.sh` in the evidence
directory. It configures CMake and runs:

```sh
cmake --build build --parallel 12 --target snap_release snap_cuda_release test_riemann.release
```

The package script builds the Python extension into the isolated package, stages
the source Python files and native libraries, verifies loaded library paths,
then runs these checks from the isolated source directory:

```sh
export PATH=/home/chengcli/pyenv/bin:$PATH
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_riemann.release$'
for precision in float64 float32; do
  CUDA_VISIBLE_DEVICES=0 python -u tests/run_fp32_straka.py \
    --precision "$precision" --device cuda --tlim 10 --profile \
    --output "/tmp/snapy-hllc-2f321dc/results/smoke-$precision"
done
```

Native/Python builds passed. CTest passed **1/1 executable, 11/11 GoogleTests**,
including `cuda_Float` and `cuda_Double` HLLC pressure-split dispatch tests (no
CUDA skips). Both 100-step GPU smoke runs and their five extra profiled steps
passed. The shock/expansion/extreme-boost scalar sweep also passed in this suite;
it is a CPU test of the shared implementation, not a separate CUDA sweep.

## CUDA code generation and profile

```sh
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-hllc-sass.py
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-hllc-profile-summary.py
```

SASS verification **passed**: both float HLLC specializations (N=7 and N=8)
contain zero DADD/DMUL/DFMA/DSETP instructions and zero F32↔FP64 conversions.
The old float N=8 kernel had 15 DADD, 50 DMUL, 101 DFMA and 59 F32→FP64
conversions. These are static instruction counts, not dynamic execution counts.

The matched five-step profiles, taken after 10 simulated seconds:

| Metric | FP64 | FP32 |
|---|---:|---:|
| HLLC kernel time, 30 calls | 0.209247 ms | 0.079583 ms |
| Sum of GPU kernel intervals | 13.697911 ms | 14.847970 ms |
| CUDA kernel launches | 8665 | 9780 |
| `aten::_to_copy` calls | 105 | 790 |
| `aten::copy_` calls | 1925 | 2650 |
| `aten::empty_strided` calls | 1005 | 1700 |

HLLC itself is **2.63× faster in FP32 than FP64** in this short profile.
Compared with the historical installed FP32 profile, its time fell from
0.212545 ms to 0.079583 ms. HLLC is now only about **0.54%** of summed FP32 GPU
kernel time. Overall FP32 still performs more launches and casts/copies. Kernel
interval sums are taken only from trace events with `cat=kernel`; CPU operator
and CUDA kernel times are not added together. Short profiles explain where work
occurs; repeated full runs determine the throughput result.

## Full Straka comparison

Run the existing HLLC dry-limit Straka experiment: 64×256 cells, WENO5, RK3,
explicit dt=0.1, 9000 steps to 900 s, tracer transport, unchanged forcing,
CFL/rejection/finite-state checks each step, snapshots at 0/300/600/900 s.
Use GPU 0 sequentially, three pairs, with pair 2 reversing precision order.
Timing includes diagnostics and snapshots and excludes initialization/profile.
GPU 1 remained idle. The host had substantial unrelated CPU load (load average
20.14 during pair 2, about 20 other busy Python processes). This limits comparisons
with historical wall times; use the alternating current-build pairs for the
FP64/FP32 comparison. No unrelated processes were stopped.

```sh
bash /tmp/snapy-hllc-2f321dc/snapy-hllc-benchmark.sh
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-hllc-summarize.py
```

**6/6 full runs passed**, each completing 9000 steps with no rejected/nonfinite
state. The comparison and repeated-field verification commands passed too.

| Wall time including diagnostics | FP64 | FP32 |
|---|---:|---:|
| Pair 1 | 68.999842 s | 72.648737 s |
| Pair 2 (FP32 first) | 69.057985 s | 76.279721 s |
| Pair 3 | 68.341718 s | 76.260171 s |
| Median | **68.999842 s** | **76.260171 s** |
| Sample standard deviation | 0.397816 s | 2.090728 s |

FP32 is **10.522% slower end to end** (0.905× FP64 speed) in this environment.
The 2× target is **not met**. Half the current FP64 median is 34.4999 s; retain
the original 31.7241 s target too. Background host load and source/build changes
make historical absolute-wall-time comparisons uncontrolled. No overall speed
gain over the installed binary is claimed from these measurements.

At 900 s:

- FP32−FP64 potential-temperature error: **0.00165068 K RMS**, **0.0338796 K
  maximum**, relative L2 **0.125724%**.
- Cold-front location: **15.35 km in both precisions**.
- Relative mass drift: FP64 **1.30648e-16**, FP32 **6.46765e-10**. Maximum
  observed FP32 drift at saved times is **1.26373e-9** (600 s).
- All saved arrays (theta, primitive, total conserved, tracer and coordinates)
  at 0/300/600/900 s are **bit-identical across same-precision repeats**.
- CUDA peak allocated memory: FP64 **15,606,784 bytes**, FP32 **12,379,648 bytes**
  (about 20.7% lower). Retained forcing scratch increases the new FP32 peak
  versus the historical 11,710,976 bytes; registered-buffer counts alone miss it.

These are observed numerical differences, not a new scientific-equivalence
threshold. Repository copies of the raw reports, hashes and comparison plot
have been removed. The installed library hashes matched their pre-test values.

## Next milestone

H1 is now validated on GPU. The typed HLLC change delivers the expected kernel
improvement, but HLLC occupies too little of the full step to reach the target.
Continue H2 with fused dry face pressure/EOS/HLLC work and fewer intermediates,
then H3 dispatch/synchronization reductions. The new profile still has **1956
FP32 launches per step versus 1733 FP64**, and many more casts/copies. Preserve
this numerical comparison and the per-step safety checks for subsequent work.
No H2 implementation was made during this GPU-validation step.

Historical local evidence checks (Python compileall, Ruff F401/F821 and
`git diff --check`) **passed** before the raw artifacts were removed. Kintera is unchanged and compatibility was
exercised with installed 2.6.10; other dependency combinations are not covered
by this run.
