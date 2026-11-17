# FP64 EOS constant cache and deferred capacity checks

The perturbation EOS previously recreated three constant tensors from CPU
vectors on every call. A warmed-up full-Straka CUDA capture failed in that
upload (`TensorDataContainer::convert_to_tensor` → `_to_copy` →
`copy_kernel_cuda`). Caching these tensors removes that confirmed blocker.

The private cache retains double-valued coefficients independently of module
FP32 conversion. Device, molecular weights, reference heat capacities, energy
offsets and gas/cloud counts form its key. Reset discards it. Cache tensors are
not restart state; a restarted or moved block rebuilds them before capture.
Configuration changes require recapturing any future execution graph.

Direct `perturbations()` calls retain immediate exceptions for invalid capacity.
Hydro stages use deferred checking: a device scalar latches nonfinite/nonpositive
background or total heat capacity. Guarded capacity denominators allow the stage
to finish without division by zero in the pressure formula. The existing batched
`check_redo` readback rejects the step; a later valid stage cannot erase the flag.
Stage zero clears it for the next attempt. The internal device flag vector gains
an eighth entry, folded into the existing six host retry causes; process-wide
retry reduction and restart formats are unchanged.

A separate experimental dry-HLLC fusion failed parity testing and was removed
from the active change. Its unfinished patch is preserved locally at
`/tmp/snapy-dry-hllc-wip.patch`. It is not required for caching or graph capture.

## Validation

Tests run in `/tmp/snapy-hllc-2f321dc/src` on dungeon3, using the isolated package
at `/tmp/snapy-hllc-2f321dc/package` and installed kintera 2.6.10. Native libraries
and matching Python bindings were rebuilt. Installed snapy is unchanged.

```sh
cmake --build build --parallel 12 --target snap_release snap_cuda_release test_eos.release test_riemann.release
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_(eos|riemann).release$'
bash /tmp/snapy-hllc-2f321dc/snapy-cache-validate.sh
```

The native suites passed **2/2**, including CPU/CUDA float/double capacity tests.
New regressions exercise cached equality, host-option refresh, immediate invalid
capacity errors, deferred failure persistence and routing to retry detection.

The validation script runs four Python regression fixtures, a warmed-up full
step capture probe, short FP64/FP32 profiles, and two full 9000-step GPU Straka
runs. Its final comparison checks every saved array at 0/300/600/900 seconds
against the pre-cache same-precision run. All four Python fixtures passed; capture
passed; both 9000-step GPU runs passed. Every saved array is **bit-identical** to
the pre-cache same-precision result at all four output times.

Five-step profiles in both precisions show `aten::item` calls **15→0** and
`cudaStreamSynchronize` **65→5**. FP32 `_to_copy` calls **790→750**, `copy_`
**2650→2610**, launches **9780→9775**. FP64 corresponding counts are **105→65**,
**1925→1885**, **8665→8660**. This is a reduction in synchronization and transfers,
not a large kernel-count reduction.

Single full-run wall times including diagnostics were FP64 **62.9569 s**, FP32
**69.5430 s**. Historical runs had different host load; these are not controlled
measurements of the cache's standalone speedup and do not establish the 2× goal.
Source tests and compiled-library hashes are retained in the evidence directory.

The capture probe now passes. This establishes capture feasibility only; it does
not establish correct repeated replay, stable RK input/output storage, or replay
of host-managed limiter bookkeeping. Those remain required before exposing a
production graph-execution option. No 2× speed claim is made from capture alone.

Repository copies of the raw artifacts have been removed; the results and
validation commands above are retained.

Additional CUDA moist regression:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package \
  /home/chengcli/pyenv/bin/python tests/test_fp32_science.py --device cuda \
  --steps 30 --output /tmp/snapy-hllc-2f321dc/results/cache-science.json
```

Passed (exit 0), covering closed, heated and sedimenting cases with the default
explicit/implicit schemes. Kintera 2.6.10 is unchanged. Formatting, Python
compileall, Ruff F401/F821 and `git diff --check` passed.


## Fresh post-cache GPU repeats — 2026-11-16

Command: `bash /tmp/snapy-hllc-2f321dc/snapy-cache-repeat.sh`: **6/6 passed**.
Summary validation: `/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-cache-repeat-summary.py`: **passed**.

Three alternating pairs on dungeon3 RTX 5090 GPU 0, 9000 steps each, using the
cache/check-validated binaries without recompiling. Both GPUs were initially
idle; recorded host load average was 0.60. Eager execution, with graph replay
not enabled. All CFL, rejection and finite-state checks and snapshots retained.

| Trial | FP64 seconds | FP32 seconds |
|---|---:|---:|
| 1 | 63.095297 | 69.435885 |
| 2 | 62.854693 | 69.855218 |
| 3 | 62.708276 | 70.016083 |
| Median | **62.854693** | **69.855218** |
| Sample standard deviation | 0.195411 | 0.299541 |

FP32 is **11.138% slower** (0.900× FP64 speed). The 2× target remains open.
The current measurements do not isolate caching's performance contribution.

Every saved array at 0/300/600/900 s is bit-identical across same-precision
repeats and to the pre-cache same-precision run. At 900 s, FP32−FP64 theta error
is **0.00165068 K RMS / 0.0338796 K maximum**, relative L2 **0.125724%**. Both
cold fronts are at **15.35 km**. FP32 relative mass drift is **6.46765e-10**.

Repository copies of the raw reports, plot and benchmark script have been
removed. Installed snapy and kintera 2.6.10 are unchanged; no source/API
changes were made during these repeats. Evidence compileall, Ruff F401/F821
and `git diff --check` passed.
