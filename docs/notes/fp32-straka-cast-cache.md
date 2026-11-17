# Straka dtype-conversion caching

2026-11-16. Implements the first two targets from the
[graph profile diagnosis](fp32-straka-graph-profile.md).

## Change

- Uniform WENO5/Center5 reconstruction now caches coefficients in the working
  dtype. Their existing nonuniform coefficient path uses the same helper.
- Divergence caches converted full face-area and volume tensors, then slices
  those tensors instead of converting each slice on every call.
- The helper tracks source identity, mutation version, storage and metadata,
  and target device/dtype. A detached metadata snapshot detects `set_data`
  changes that preserve the original TensorImpl/version. Edited cached values
  also invalidate. Inference tensors/mode and autograd sources bypass caching.
- These are derived caches, not registered buffers. Restart fields, equations,
  pressure-cancellation precision, CFL/retry/finite checks and the graph
  executor are unchanged. Geometry/coefficient edits still require a new graph
  executor, as documented by the existing fixture-specific graph interface.

## Validation

The isolated dungeon3 build and matching bindings were rebuilt. Kintera 2.6.10
is unchanged; the installed snapy package and user's remote checkout were not
modified. Native class layouts changed: consumers/bindings must be rebuilt
together rather than mixing old binaries with new headers/libraries.

From `/tmp/snapy-hllc-2f321dc/src`, with the existing Python/CUDA toolchain:

```sh
cmake --build build --parallel 12 --target snap_release snap_cuda_release test_weno.release test_coordinate.release test_x1_coeffs.release
MAX_JOBS=8 python setup.py build_ext --build-lib /tmp/snapy-hllc-2f321dc/package --build-temp /tmp/snapy-hllc-2f321dc/extension-build
cp build/lib/libsnap*so ../package/snapy/lib/
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_(weno|coordinate|x1_coeffs|straka_cuda_graph_python)(.release)?$'
```

Build/bindings passed. **4/4 CTests passed**, 10.50 s. New CPU/CUDA tests cover
cache reuse and invalidation, in-place coefficient/grid edits, source storage
replacement, metadata and dtype/device changes, inference-mode fallback, and
agreement with uncached reconstruction/analytic linear-flux divergence. Existing
nonuniform reconstruction and graph parity/rollback tests also passed.

## Benchmark method

The previous complete package was preserved as `package-before-cast-cache`.
Three repeats of baseline/optimized FP64/FP32 run sequentially on dungeon3
RTX 5090 GPU 0; trial 2 reverses the order. Every run uses CUDA graph execution,
the fixed Straka grid, 9000 steps of nominal dt 0.1, unchanged checks and saved
fields at 0/300/600/900 s. All timing claims use these fresh paired runs.

Each run also profiles five extra checked steps after saving the final state.
The first profiling step can recapture when the final clipped dt changes back
to 0.1. Kernel analysis therefore counts only kernels correlated with the five
`cudaGraphLaunch` calls, excluding capture warmup and outside-graph checks.
These are summed captured GPU kernel durations, distinct from full-run wall
time. Unlike the earlier short profile, reported kernel totals here exclude
outside-graph work. Category medians need not sum to the median total.

Commands:

```sh
bash /tmp/snapy-hllc-2f321dc/snapy-cast-cache-benchmark.sh
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-cast-cache-summary.py
```

Both commands passed: **12/12 full runs**, with all fields at 0/300/600/900 s
bit-identical to the same-precision baseline across all repeats. Every run had
9000 replays and four captures. Full traces/snapshots remain on dungeon3 in
`/tmp/snapy-hllc-2f321dc/results/cast-cache/`.

## Results

| Measurement | Before | After |
|---|---:|---:|
| FP32 captured dtype-copy kernels / timestep | 177 | 99 |
| FP32 captured dtype-copy time / five steps | 1.39236 ms | 0.89626 ms |
| FP32 captured total kernel time / five steps | 13.90260 ms | 13.40345 ms |
| FP64 captured total kernel time / five steps | 12.86437 ms | 12.86938 ms |
| FP32 full-run median | 31.89151 s | 30.05118 s |
| FP64 full-run median | 28.91095 s | 30.23014 s |
| FP32 peak allocated CUDA memory | 14,952,448 bytes | 15,179,776 bytes |

Kernel durations and full-run times are medians of three trials. Conversion
counts are identical across repeats. **78 conversions per timestep removed**,
a 44.1% reduction in captured dtype-copy kernels; conversion GPU time falls
**35.6%**, and total captured FP32 GPU kernel time falls **3.59%**. FP64 captured
kernel counts are unchanged and its GPU time differs by only 0.04%.

FP32 full-run median speedup is **1.061×**, or 5.77% less time. The individual
paired speedups are **1.111× / 1.034× / 1.032×**. Timings in trial order:

| Mode | Trial 1 | Trial 2 | Trial 3 |
|---|---:|---:|---:|
| Before FP64 | 29.08114 s | 28.09039 s | 28.91095 s |
| After FP64 | 30.23014 s | 32.47595 s | 27.48074 s |
| Before FP32 | 33.39289 s | 31.89151 s | 29.71130 s |
| After FP32 | 30.05118 s | 30.84608 s | 28.78214 s |

FP64 wall times have substantial variation: its median increases 4.56% despite
unchanged kernel count and nearly unchanged GPU time. Consequently, the after
FP32/FP64 wall medians being within 0.6% is **not evidence of precision parity**.
The captured GPU profile still shows FP32 **4.15% slower than FP64**, down from
8.07% before caching. This is a measured improvement, not a 2× precision speedup.

The retained casts increase peak FP32 allocation by **227,328 bytes (1.52%)**.
FP64 peak allocation stays at 19,572,224 bytes. The remaining copies include
full-field hydro/EOS conversions; reducing those would require a separate
numerical review rather than more constant caching.

Repository copies of the kernel/run summaries, scripts and CTest log have
been removed. The results and commands above are retained.
Local Google clang-format checking, Python compilation, `bash -n` on the
benchmark script and `git diff --check` passed. No kintera or restart-schema
changes; rebuild native snapy consumers/bindings together.
