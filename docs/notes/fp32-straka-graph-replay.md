# Straka CUDA graph replay

The Straka runner now accepts `--execution cuda-graph`; eager execution remains
the default. This is a fixture-specific experiment, not a general graph API for
all snapy models. It requires the dry Cartesian HLLC, explicit RK3 case with
limiter disabled and one block. Chemistry, VIC, distributed exchange and host
limiter queues are outside the supported configuration.

No native kernels or kintera code changed. The executor captures the existing
three-stage timestep. Detached input handles retain the storage that the graph
reads even when native RK updates call `set_`/`set_data`. Before each replay,
current state is copied into those inputs, including caller edits and restored
retry state. A changed timestep or tensor storage triggers recapture. Changing
configuration or hydrostatic references requires constructing a new executor.

Warmup uses zero-duration stages and restores the original state, without
accepting synthetic steps or resetting the integrator's rejection counter.
CFL, rejection, finite-state checks and snapshots remain outside the graph and
retain their original frequency. The existing runner aborts on a rejected step;
the regression separately checks native rollback and retry-budget exhaustion.

## Correctness checks

On dungeon3, from `/tmp/snapy-hllc-2f321dc/src`:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package \
  /home/chengcli/pyenv/bin/python tests/test_straka_cuda_graph.py
```

This compares 100 graph/eager steps in each precision, including timestep
changes, replaced tensors and an in-place perturbation. It checks hydro and
tracer conserved/primitive fields at every step. Injected NaNs exercise native
rollback; successive rejected attempts must exhaust the retry budget rather
than being reset by capture warmup. Unsupported limiter configuration is rejected.
The fixture is registered in CTest only for CUDA builds.

## Full-run measurement

```sh
bash /tmp/snapy-hllc-2f321dc/snapy-replay-benchmark.sh
/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-replay-summary.py
```

Three trials of eager FP64, graph FP64 and graph FP32 run sequentially on the
same RTX 5090 GPU 0, with reversed order in trial 2. An additional eager FP32 run
checks the refactored fixture. Each run uses 9000 steps at dt=0.1 with the same
CFL/retry/finite checks and 0/300/600/900-second snapshots. No recompilation was
needed; the isolated cache/check-validated libraries and kintera 2.6.10 are reused.

The timed interval excludes initialization and initial capture. Any recapture
during the run is included. Total graph setup time is reported separately;
adding all setup time to the measured interval is a conservative upper bound
that counts in-run setup twice. This bound is also compared with eager FP64.
Peak CUDA allocation includes retained graph input/output and work storage.

Repository copies of the raw reports, hashes, scripts and host-load observations
have been removed. All ten full runs and all-field comparisons passed; the
results below are retained.


## Results

| Mode | Trials | Median seconds | Range seconds | Speed versus eager FP64 |
|---|---:|---:|---:|---:|
| Eager FP64 | 3 | 66.5690 | 66.5574–68.0403 | 1.000× |
| Eager FP32 | 1 | 72.2040 | 72.2040–72.2040 | 0.922× |
| Graph FP64 | 3 | 29.4328 | 28.2130–29.5516 | 2.262× |
| Graph FP32 | 3 | 31.1725 | 29.1774–31.8344 | 2.136× |

FP32 graph replay achieves **2.136×** the matched eager FP64 median. Adding
all recorded graph setup time gives a conservative **31.4056 s / 2.120×** result.
All three paired trials exceed 2× their corresponding eager FP64 run even using
this conservative bound. The median also meets the original **31.7241 s** target;
the slowest FP32 graph trial (31.8344 s, excluding initial setup) does not meet
that historical absolute threshold. Host load varied (recorded load average
5.63); the alternating current-build pairs are the primary comparison.

**This is graph acceleration, not a 2× FP32 arithmetic advantage.** Graph FP64
is faster than graph FP32; FP32 is about 5.9% slower when both use graphs. The
fixed-grid target versus eager FP64 is met, but no broader moist/implicit or
larger-grid performance claim is made.

All graph runs completed **9000 replays**, with **four captures** per run.
Floating-point timestep truncation at output boundaries causes recapture; the
exact eager timestep schedule is preserved. Total setup per graph run is
0.217–0.242 s, including the in-run captures already counted in runtime.

Every saved theta, primitive, total conserved, tracer and coordinate array at
0/300/600/900 s is **bit-identical to its same-precision eager reference** across
all trials. The first full graph runs also matched the pre-graph reference
snapshots. FP32 science differences from FP64 are therefore unchanged.

Peak CUDA allocation:

| Mode | Bytes |
|---|---:|
| Eager FP64 | 15,314,944 |
| Graph FP64 | 19,572,224 |
| Eager FP32 | 12,087,808 |
| Graph FP32 | 14,952,448 |

Retaining graph storage increases memory within either precision. FP32 graph
peak is about 23.6% below FP64 graph peak.

Final commands/results:

- `bash /tmp/snapy-hllc-2f321dc/snapy-replay-benchmark.sh`: **10/10 full runs passed**.
- `/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-replay-summary.py`:
  **passed**, including exact equality of all saved fields.
- `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_straka_cuda_graph_python$'`:
  **1/1 passed**, 3.76 s, including both precisions, invalidation, rollback and
  exhaustion of the five-retry budget.
- Python compileall, Ruff F401/F821 and `git diff --check`: **passed**.

No native library, dependency or restart-format changes were required for this
step. Compatibility was exercised with Torch 2.10.0+cu128 and kintera 2.6.10;
other combinations are untested. Keep graph execution opt-in and restricted to
this fixture until broader models' host-side bookkeeping is explicitly handled.
