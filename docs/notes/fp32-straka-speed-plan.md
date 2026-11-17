# Straka GPU throughput target: 2× FP64

## Target and measurement contract

Target optimized FP32 throughput at least **2× the FP64 baseline**, on the same
RTX 5090 and the existing 64×256, 9000-step, dt=0.1 Straka experiment. The measured
FP64 median is 63.4482 s: the initial acceptance target is **≤31.7241 s**. This is
also a 2.21× improvement over current installed FP32 (70.2006 s). If the FP64
implementation changes, repeat its measurement and retain both comparisons.
This is a goal, not a predicted result from the HLLC fix.

Keep HLLC, WENO5, RK3, resolution, forcing, scalar transport, timestep schedule,
and validation/output schedule identical. Do not meet the target by lowering
accuracy, skipping physical work, hiding rejected steps or excluding validation
from only one side. Run three or more sequential alternating pairs on one GPU;
report median, range, variation, hashes, end-to-end time and separately kernel
intervals. No CPU timing contributes to this target. Preserve numerical gates
and report fields/budgets at 300/600/900 s. No regridding work.

## H1: true FP32 HLLC arithmetic and safe acoustic differences

Replace implicitly double literals with T-typed constants; assert representative
intermediate types at compile time. For float, form acoustic wave/flow differences
directly, honoring the clipped-wave branches separately. Retain FP64 arithmetic
order and shifted-pressure semantics. Remove the unused middle velocity estimate.

Validate shocks, rarefactions, both boost signs, all directions, species flux and
large pressure backgrounds against double precision. Exercise ordinary states
and extreme boosted states where float subtraction erases an acoustic speed.
Run existing Riemann, hydro, restart and implicit suites plus Straka. On an
updated GPU binary inspect SASS: avoid unintended double arithmetic in the float
kernel, then time the kernel and full case. This stage cannot deliver 2× alone:
HLLC is only ~1.4% of currently summed FP32 GPU kernel duration.

## H2: fuse high-frequency work around existing kernels

Use existing traces to rank total per-step cost and launch counts. Target:

1. Fuse face pressure gauge operations and dry face EOS preparation with the
   HLLC wrapper/kernel; keep sensitive pressure differences in FP64 locally.
2. Fuse flux divergence, gravity/source increments and RK update where dependency
   order permits. Preserve shared-face conservation and source rounding contracts.
3. Cache immutable geometry/background coefficients with explicit invalidation;
   eliminate redundant whole-state conversions and temporary face arrays.

Keep a tested fallback for unsupported geometry, covariance or moist cases.
Measure each change independently. A working launch-count goal is ≤500 kernels
per step versus the current ~1953, but only measured runtime establishes success.
Retained scratch can increase resident memory; track CUDA peak allocation rather
than claiming memory savings from registered buffers alone.

## H3: reduce host dispatch and synchronization

After buffers and shapes stabilize, assess CUDA graph capture for safe regional
or full-stage execution. Inventory host scalar reads, retry decisions, CFL checks
and dynamic allocations first. Keep failure detection and retry semantics intact;
do not bypass them to enable capture. Compare graph and eager paths numerically.
If capture is unsuitable, batch independent operations and fuse reductions or
validation to reduce CPU dispatch. End-to-end target remains 3.525 ms/step; host
and device budgets must be measured as overlapping timelines, not simply added.

## H4: prove scope, accuracy and throughput

Re-run the exact fixed-grid Straka benchmark with the optimized GPU binary.
Require the 2× target across repeated trials with numerical acceptance; record
any regression, not just favorable runs. Sweep larger grids separately to locate
launch-limited versus device-work-limited regimes. A larger-grid speedup does not
satisfy the fixed-grid target. Recheck moist/Jupiter, restart and multi-block
regressions before enabling changes beyond the validated dry Cartesian path.

An optional FP64 evolving-anomaly/RK accumulation plus FP32 compute experiment
addresses accuracy and the Kratos precision-policy distinction. It is separate
from the throughput target and must not silently change existing storage/restart
contracts. No kintera API changes are currently required.

## Execution status

H1 implementation, local validation, CUDA codegen verification and six full
GPU Straka runs are complete. Results are tracked in
[the HLLC GPU report](fp32-hllc-gpu-validation.md). H2–H4 are planned.
The user requested GPU testing before further implementation. An isolated
dungeon3 build now contains the current source; the installed package remains
untouched. Historical installed-package measurements remain separate.

### H1 local validation completed

The shared CPU/CUDA HLLC implementation is fixed. Compile-time assertions prevent
representative float intermediates from silently becoming double. The new test
covers 120 combinations of direction, shock/expansion/equal state, signed boost
and reference pressure, including ±1e8 boosts and a transported species.

Commands and results:

```sh
PYTHONPATH=/tmp/snapy-straka-opt-python cmake --build build --parallel 4 --target snap_release test_riemann.release
ctest --test-dir build --output-on-failure -L 'fp32-(foundations|science|acceptance)'
PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py \
  --precision float32 --device cpu --output /tmp/straka-hllc-fix-900
```

Build passed; **40/40 CTests passed** (158.18 s). Straka completed **9000 steps**,
with no rejected/nonfinite steps and relative mass drift **1.1391e-9**, below the
existing reduced-case 2e-5 engineering mass limit. At 900 s, theta differs from
the old CPU FP32 run by **0.0005673 K RMS / 0.01535 K maximum**; changing arithmetic
precision is not expected to preserve bitwise trajectories. These differences
are observations, not a new scientific-equivalence claim. Repository copies
of the raw correctness data and binary hashes have been removed.

Google clang-format and `git diff --check` passed. No kintera changes. Subsequent
GPU results are recorded in [the HLLC GPU report](fp32-hllc-gpu-validation.md);
CPU runtime is not used to infer GPU gains. H2/H3 remain planned, not implemented.

### H1 GPU validation completed

Three alternating GPU-only Straka pairs passed (9000 steps each). Median FP64
**69.00 s**, FP32 **76.26 s**: FP32 remains **10.5% slower overall**. Background
CPU load limits historical wall-time comparisons. The HLLC kernel itself is
**2.63× faster in FP32** in the matched short profile and both compiled float
specializations contain no FP64 arithmetic/conversions. Final theta error is
**0.00165 K RMS**, FP32 mass drift **6.47e-10**. Full evidence and commands are in
the [GPU report](fp32-hllc-gpu-validation.md).

The 2× end-to-end target remains open. H2 should reduce the measured 1956 FP32
launches/step and conversion/copy overhead; H3 addresses remaining host dispatch.
CUDA peak allocation, including retained scratch, is the memory acceptance
metric. Preserve the fixed-grid experiment and all safety checks.


### Graph-enabling cache/check change

The user selected graph-first execution to minimize code changes. Full-step
capture initially failed on per-call EOS constant uploads. FP64 constant caching
and deferred capacity validation now allow warmed-up Straka capture to pass.
Both full GPU precision trajectories remain bit-identical to the pre-cache
results; see [the cache/check report](fp32-cache-validation.md).

Next: validate repeated replay with stable state storage and correct replay of
limiter/retry bookkeeping before exposing an execution option. The separate dry
HLLC fusion experiment failed parity checks and was removed from the active
change; do not count it as implemented or validated.


### Straka graph replay and throughput validation completed

The runner now supports opt-in `--execution cuda-graph` through a small Python
executor, with no new native kernels. Replay correctness, changed timesteps and
storage, caller edits, rollback and retry-budget exhaustion are tested. Ten full
GPU runs passed; every saved graph field matches eager execution bit for bit.

Median graph FP32 **31.1725 s** versus matched eager FP64 **66.5690 s** gives
**2.136×**, or **2.120×** with a conservative setup-inclusive bound. This meets
the fixed-case paired throughput goal. Graph FP64 is faster still (**29.4328 s**):
the gain comes from graph replay, not an FP32 arithmetic advantage. Median meets
the original 31.7241 s threshold, though the slowest trial does not. Full detail:
[graph replay report](fp32-straka-graph-replay.md).

Stop expanding native fusion for this benchmark now that its throughput goal is
met. General graph support for moist/implicit/distributed/limiter-enabled models
and larger-grid scaling remain separate future work; eager remains the default.

### Follow-up: identify the remaining FP32 graph overhead

At the user's request, profiled the remaining precision gap before further
implementation. [GPU diagnosis](fp32-straka-graph-profile.md): FP32 has 6.59%
more summed GPU kernel time and 196 additional kernels per timestep. Converting
copies and mixed arithmetic outweigh the now-faster WENO/HLLC kernels.

Next candidates, in order of implementation scope:

1. Cache uniform WENO5/Center5 coefficients in the working dtype, consolidating
   with the existing nonuniform coefficient-cache pattern (48 casts/step).
2. Cache converted face areas/volume used by divergence (30 casts/step).
3. Investigate full-state hydro/EOS precision round trips, preserving FP64
   cancellation-sensitive expressions and checks.

Steps 1 and 2 need correct invalidation on coefficient/geometry/device/dtype
changes. Measure each with the same graph profiles and full Straka runs; require
same-precision snapshot equality for caching-only changes. These are proposals,
not implemented gains. No kintera changes are indicated. The current evidence
supports targeting the remaining roughly 6% gap, not claiming 2× over FP64 graphs.

### Coefficient and geometry cast caching completed

Implemented follow-up candidates 1 and 2 with one shared cache, also replacing
the older nonuniform coefficient-cache helper. Invalidation covers source edits,
storage/metadata replacement and device/dtype changes; inference/autograd inputs
bypass caching. No equations or cancellation-sensitive precision were changed.

Four regression suites and twelve full paired GPU runs passed. Saved fields are
bit-identical within each precision. **78 conversion kernels/step removed**;
captured dtype-copy GPU time falls **35.6%** and total captured GPU time **3.59%**.
Full FP32 median improves **31.89151→30.05118 s (1.061×)**. FP64 wall-time
variation is substantial; FP32 still uses **4.15% more captured GPU time** than
FP64. Retained caches add 227,328 bytes to peak FP32 allocation.

[Validation, commands and evidence](fp32-straka-cast-cache.md). The remaining
candidate is reducing full-field hydro/EOS dtype round trips; that is a separate
numerical change and has not been implemented. Kintera remains unchanged.
