# Straka FP32 graph slowdown diagnosis

2026-11-16. Diagnostic work only; no solver, graph executor or kintera changes.

FP32 saves time in WENO and HLLC but spends more on dtype conversions and
mixed-type arithmetic. Graph capture removes Python/native dispatch overhead;
it still replays every captured conversion. Caching EOS constants enabled
capture, but did not remove the remaining per-stage conversions.

## Measurements

Dungeon3, RTX 5090 GPU 0, sequential runs, same isolated native libraries as the
[graph replay validation](fp32-straka-graph-replay.md), no rebuild. Three
alternating FP64/FP32 graph pairs, second pair reversed, plus one eager profile
per precision. Each run advances 100 checked steps, then profiles five more.
The diagnostic wrapper adds the normal CFL and finite checks to the existing
profile loop. All eight runs completed successfully.

Times below are medians of **summed GPU kernel durations over five checked
timesteps**, not application wall time. Only GPU kernel trace events are
counted, avoiding double-counting their CPU operator parents. Graph-launch
correlation IDs distinguish captured kernels from checks and input copies.

| GPU work | FP64 ms | FP32 ms | FP32 minus FP64 ms |
|---|---:|---:|---:|
| Dtype-converting copies | 0.096 | 1.415 | +1.320 |
| Elementwise add/multiply/divide/subtract, including cast-capable kernels | 5.741 | 6.379 | +0.638 |
| WENO5 | 1.471 | 0.724 | −0.746 |
| HLLC kernel | 0.207 | 0.077 | −0.130 |
| Hydrostatic x1 reference scan | 0.819 | 0.819 | approximately zero |
| All kernels | 14.081 | 15.009 | +0.928 |

The table omits smaller categories; complete categories are in the JSON
evidence. Category medians need not sum exactly to the median total.
Cast-capable arithmetic alone rises by 1.807 ms, but much of that replaces
ordinary arithmetic. The **net arithmetic increase is 0.638 ms**, not 1.807 ms.
Kernel-symbol classification identifies conversion/cast-capable implementations;
the separate dispatch probe confirms mixed input dtypes and explicit casts.

Captured timestep kernels account for 13.494 / 14.412 ms (FP64 / FP32).
Outside-graph kernels account for only 0.587 / 0.597 ms. Thus the precision
gap is overwhelmingly inside the graph, rather than in CFL/retry/finite checks.
There are 9,510 / 10,490 kernels per five steps: **196 extra kernels per FP32
step**, of which 195 are inside the graph. Kernel counts are identical across
all three repeats. FP32 GPU kernel time is 6.59% higher, consistent with the
earlier full-run graph wall-time gap of 5.91%.

The profiler perturbs launch timing; its CPU `cudaGraphLaunch` duration is not
an unprofiled launch-overhead measurement. Use the earlier 9000-step benchmark
for end-to-end speed claims.

## Callers responsible and next priorities

A separate eager `TorchDispatchMode` probe records C++ backtraces, shapes and
dtypes during one warmed-up RK3 timestep. These are **call counts, not per-caller
GPU timings**; graph kernel symbols alone do not uniquely identify their native
call site. Mixed scalar operands also do not automatically imply promotion of
an entire array.

1. **Cache uniform reconstruction coefficients in the working dtype.**
   FP32 performs 36 WENO5 coefficient conversions and 12 Center5 conversions
   per timestep; FP64 performs none. Uniform `left`/`right` paths repeatedly
   call `cm.to(w.options())` / `cp.to(w.options())` in
   `src/recon/weno5.cpp` and `src/recon/cp5.cpp`. The nonuniform path already
   has `x1_coeffs_for` caching: consolidate the uniform path with that pattern,
   including invalidation when coefficients/device/dtype change. These arrays
   are small, so the expected benefit is fewer GPU kernels, not saved bandwidth.

2. **Cache dtype-specific geometry for divergence.**
   `CoordinateImpl::divergence` (`src/coord/coordinate.cpp:706`) performs
   **30 FP64→FP32 conversions per timestep**: four face-area slices and a
   volume tensor on each of six calls. These contain roughly 17–18k cells each.
   Cache converted full areas/volume and slice those, preserving existing
   arithmetic and invalidating on geometry/device/dtype changes. This is more
   data movement than the coefficient casts; both are low-scope candidates.
   Together these two targets account for **78 explicit conversions per step**.
   Their individual GPU savings still require an A/B experiment.

3. **Reduce full-field dtype round trips at the hydro/EOS boundary.**
   The probe finds another 18 explicit casts in `HydroImpl::forward`, nine in
   `IdealMoistImpl::perturbations`, four in `HydroImpl::total_u`, and 30 mixed
   dtype copies in hydro per timestep. Reconstruction includes conversions of
   full five-component cell and two-sided face states. EOS promotes stored
   anomalies to FP64 and returns FP32 deltas; hydro combines those with FP64
   backgrounds. These are candidates for retaining FP32 working arrays and
   narrowing FP64 operations to cancellation-sensitive expressions, but carry
   more numerical risk than caching immutable tensors. Do not demote pressure
   cancellation or remove validation checks merely to remove casts.

HLLC kernel optimization is a low priority now: the FP32 kernel is already
2.69× faster and uses only 0.51% of total profiled FP32 kernel time. Its wrapper
still has six explicit casts per step, separate from the timed HLLC kernel.
The hydrostatic scan is a shared cost, not the cause of the precision gap.
Further EOS constant caching does not address the leading conversions found
here. No kintera change is indicated by this profile.

Start with priorities 1 and 2, then repeat graph profiling and full 9000-step
FP64/FP32 comparisons with same-precision snapshot equality. The measured gap
suggests an opportunity to recover the remaining roughly 6%; it does **not**
establish that these small changes can deliver 2× over FP64 graphs. The earlier
2× result was FP32 graphs versus eager FP64.

## Evidence and validation

Repository copies of the diagnostic scripts, kernel/caller summaries and run
reports have been removed. Full traces and full C++ backtraces were stored on
dungeon3 under
`/tmp/snapy-hllc-2f321dc/results/graph-profile/`. The run reports include native
library hashes; installed package version metadata alone does not identify the
isolated native build. No installed package or remote checkout was changed.

Commands on dungeon3, all exit 0:

```sh
bash /tmp/snapy-hllc-2f321dc/profile-straka-checked.sh
# 8/8 runs completed, each 100 checked steps plus 5 profiled checked steps.
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/analyze-straka-graph.py
# Six graph traces parsed; exactly five graph launches in every trace.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package /home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/straka-op-origin.py
# Both precision probes completed, native retry checks passed.
```

Cross-repo compatibility: none introduced; kintera 2.6.10 and native libraries
are unchanged. Future caches must preserve coefficient/geometry updates and
graph recapture semantics. No new behavioral tests are needed for this
diagnostic-only change.

Historical local artifact checks (Python compileall, `bash -n` on the benchmark
script and `git diff --check`) passed before the raw artifacts were removed.
