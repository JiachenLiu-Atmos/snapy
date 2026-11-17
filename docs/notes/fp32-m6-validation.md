# M6 validation: reduced moist cases

M6 remains in progress. Regridding is excluded. This harness supplies reduced
engineering checks and measurements; it does not establish production climate
statistics or production GPU throughput.

`tests/test_fp32_science.py` compares FP32 and FP64 using deterministic velocity
seeds, closed moist columns, prescribed heating and sedimentation. Diagnostics
promote before reduction and include dry mass, water-family mass and corrected
potential energy. Heating records actual rounded increments separately from the
requested source. Late-window means and sample standard deviations are reported,
without treating three seeds as a statistical equivalence test.

The optional Jupiter comparison includes H2O/NH3 chemistry, heating/cooling,
diffusion, sedimentation and a passive scalar. Its provisional normalized state,
family-mass and scalar limits are 2e-4. Failed Jupiter comparisons retain the JSON
measurements and return a nonzero exit status.

Timing synchronizes CUDA and reports repeated step times after warmup. Storage
counts deduplicate registered buffers and state; they exclude unregistered
scratch and allocator overhead. CUDA peak allocated bytes are separately
reported. Profiler traces expose operators but copy costs cannot all be assigned
to precision conversion. Small grids can be dominated by launch/dispatch costs.

## Reproduction

No native compilation was performed for M6 validation. Local tests use the
existing isolated installation. On dungeon3, fixtures were copied into
`/tmp/snapy-m6-hsaZRT`; its source checkout and installed packages were unchanged.
The test fixture revision is `cbf8683` plus the M6 Python edits. JSON records
fixture script and native-library SHA256 hashes separately from package versions.

Local short checks:

```sh
ctest --test-dir build --output-on-failure -L fp32-science
ctest --test-dir build --output-on-failure -L fp32-foundations
```

Results: science **1/1 passed** (8.11 s); foundations **37/37 passed** (74.41 s).

On dungeon3, from the isolated fixture directory:

```sh
export CUDA_VISIBLE_DEVICES=0 DEVICE=cuda DEVICE_ID=0 SNAP_TEST_DEVICE=cuda
PY=/home/chengcli/pyenv/bin/python
$PY -u tests/test_fp32_science.py --device cuda --steps 30 \
  --benchmark-steps 5 --repeats 1 --jupiter-steps 10 \
  --source-revision cbf8683 --output results/gpu-smoke.json
$PY -u tests/test_fp32_restart.py
$PY -u tests/test_fp32_science.py --device cuda --steps 1000 \
  --seeds 3 --schemes 9 --sample-every 50 --benchmark-steps 100 \
  --repeats 3 --profile --jupiter-steps 1000 \
  --source-revision cbf8683 --output results/gpu-implicit.json
```

The GPU is an RTX 5090, with Torch 2.10.0+cu128, snapy 2.13.27 and
kintera 2.6.10. The smoke command passed all 12 trajectories and both Jupiter
precisions. Maximum normalized column state difference was 1.1403e-6;
10-step Jupiter difference was 4.0441e-6. VIC fallback counts were zero.

The GPU restart command **failed** with missing `hydro_restart_metadata`.
The installed library therefore does not implement the merged M5 restart
contract tested here. Compute results are evidence for these installed binaries,
not validation of M5 GPU restart support. The user requested no recompilation;
the installation was left untouched. No kintera source/API changes were needed.

## Remaining acceptance work

Investigate the 1000-step Jupiter paired-state discrepancy: the initial CPU run
measured 4.6658e-4 against the provisional 2e-4 limit, although its standalone
mass and zero-fallback checks passed. Do not loosen this gate merely to obtain
a pass. Repeat M5 GPU restart validation with a matching installed binary when
available. Production-size timing, long-time science statistics, boundary-flux
budgets and actual distributed-rank validation remain open.

## Completed 1000-step measurements

Each platform completed 18 column trajectories (3 cases × 3 seeds × 2
precisions) at dt=0.05 and the Jupiter pair at dt=0.1. Both complete commands
returned **failure solely at the final Jupiter paired-state gate**. Column
budgets, paired column states, Jupiter family/scalar comparisons, standalone
Jupiter mass checks and zero-fallback checks passed. The raw CPU/GPU reports
recorded that failure; repository copies of those reports have been removed.

| Measurement | CPU (M1 Max) | GPU (RTX 5090) |
| --- | ---: | ---: |
| Worst FP32 relative mass residual | 1.520e-7 | 1.521e-7 |
| Worst FP32 relative water-family residual | 3.154e-5 | 3.152e-5 |
| Worst FP32 relative energy residual | 1.557e-8 | 1.521e-8 |
| Maximum paired column state error | 2.618e-5 | 2.600e-5 |
| Jupiter paired state error (limit 2e-4) | **4.666e-4** | **4.669e-4** |
| Jupiter water-family difference | 2.159e-5 | 2.145e-5 |
| Jupiter ammonia-family difference | 1.977e-5 | 1.977e-5 |
| Jupiter scalar difference | 1.651e-5 | 1.651e-5 |
| FP64 VIC median seconds/step | 0.018425 | 0.017175 |
| FP32 VIC median seconds/step | 0.018742 | 0.021717 |

Timing uses three repetitions of 100 steps after five warmup steps. The reduced
column has 32×8×4 active cells. These measurements show no FP32 speedup; GPU FP32
latency is approximately 26% higher in this experiment. CPU and GPU use different
native builds, so the table is not a controlled hardware comparison.

Registered buffers/state shrink from 3,941,104 to 2,145,104 bytes (45.6%) for VIC.
CUDA peak allocation shrinks from 16,562,688 to 14,665,728 bytes (11.5%). These
are distinct measures; neither is total process/GPU memory usage. In the five-step
GPU profiler samples, CPU self-time for `aten::_to_copy` is 0.215 ms (FP64) versus
1.723 ms (FP32), and `aten::copy_` is 3.816 versus 5.573 ms. These are host operator
measurements, not CUDA kernel durations or a complete attribution of the slowdown.
Chrome traces remain alongside the scratch reports; their paths in committed
JSON are artifact names, not checked-in files.

Local long command:

```sh
PYTHONPATH=/tmp/snapy-rebase-python python -u tests/test_fp32_science.py \
  --steps 1000 --seeds 3 --schemes 9 --sample-every 50 \
  --benchmark-steps 100 --repeats 3 --profile --jupiter-steps 1000 \
  --output /tmp/snapy-m6-cpu-implicit-final.json
```

Additional dungeon3 commands using the exports above:

```sh
$PY -u tests/test_fp32_implicit.py
$PY -u tests/test_fp32_policy.py
```

Both **passed** with the installed binaries. Local science CTest was rerun after
the report-on-failure change: **1/1 passed**, 8.38 s. Python compileall,
`ruff check --no-cache --select F401,F821` on the four changed Python fixtures,
and `git diff --check` passed. The neighboring kintera 2.6.10 source was consulted;
no cross-repo source changes were made. The installed snapy restart contract
mismatch remains the material compatibility issue.

## Straka on two GPUs

The user requested simultaneous Straka FP64/FP32 runs. GPU 0 ran FP64 and GPU 1
ran FP32, both RTX 5090s initially idle, using the same installed libraries above.
No recompilation was performed. `tests/run_fp32_straka.py` uses the shipped
64×256×1, 100 m grid, reflecting walls, cold-bubble initialization and passive
tracer, with one block per process. Both runs take exactly 9000 explicit RK3
steps of 0.1 s through 900 s, checking the CFL bound, finite state and zero redo
status each step. State, scalar and flux buffer dtypes are verified.

This tests the supported FP32 anomaly path: **HLLC with the ideal-moist EOS in
its dry-only limit**, in both precisions. It is not the shipped ideal-gas/LMARS
reference regression. Dry gas constants are Rd=286.9998232 and cp=1004.4993812
J/kg/K (the existing anomaly fixture's dry composition). No viscosity is added:
the shipped `problem/K: 75` is not consumed by `examples/straka.cpp`. Thus these
results do not establish agreement with a viscous Straka reference solution.

From `/tmp/snapy-m6-hsaZRT` on dungeon3, run these commands concurrently:

```sh
CUDA_VISIBLE_DEVICES=0 /home/chengcli/pyenv/bin/python -u tests/run_fp32_straka.py \
  --precision float64 --output results/straka-float64
CUDA_VISIBLE_DEVICES=1 /home/chengcli/pyenv/bin/python -u tests/run_fp32_straka.py \
  --precision float32 --output results/straka-float32
```

Both commands **passed**, exit 0. Initial saved fields were identical.
At 900 seconds:

| Measurement | FP64 | FP32 |
| --- | ---: | ---: |
| Relative mass drift | 1.31e-16 | 1.09e-9 |
| Minimum potential temperature anomaly (K) | -14.35264 | -14.35652 |
| Cold-front position (km) | 15.35 | 15.35 |
| Wall time including diagnostics (s) | 63.63 | 70.30 |
| Registered buffer/state bytes | 6,623,984 | 3,983,024 |
| CUDA peak allocated bytes | 15,606,784 | 11,710,976 |

The front is the furthest cell center with theta anomaly below -1 K in the
lowest 1 km, so matching positions mean agreement to the 100 m grid resolution.
The FP32−FP64 theta RMS difference is **0.001764 K**, maximum **0.03645 K**,
and relative L2 difference **0.1344%**. Maximum vertical/horizontal velocity
differences are **0.03481/0.02893 m/s**, and maximum pressure difference is
**0.9112 Pa**. No scientific-equivalence threshold was imposed after observing
these results. FP32 saved 39.9% registered storage and 25.0% CUDA peak allocation,
but took 10.5% longer in this single simultaneous-run measurement. Shared-host
contention and run-to-run variance prevent treating this as an isolated benchmark.

Repository copies of the comparison JSON and field plot have been removed.
Raw NPZ snapshots and emitted configurations remain under each remote output directory
and in `/tmp/snapy-m6-results/` locally. Reproduce the comparison after retrieval:

```sh
MPLCONFIGDIR=/tmp/snapy-mpl python tests/compare_fp32_straka.py \
  /tmp/snapy-m6-results/straka-float64 /tmp/snapy-m6-results/straka-float32 \
  --output /tmp/snapy-m6-comparison/straka
```

The comparison command passed and checks matched coordinates, timesteps, steps,
gas constants and native-library hashes. New Python scripts passed compileall,
Ruff F401/F821 and `git diff --check`. No kintera modifications were required;
the EOS/solver adaptation and previously identified installed M5 restart mismatch
remain compatibility limitations.

## Straka profiling and native optimization

Baseline profiles are Straka-specific and were run sequentially on GPU 0:

```sh
CUDA_VISIBLE_DEVICES=0 /home/chengcli/pyenv/bin/python -u tests/run_fp32_straka.py \
  --precision float64 --tlim 10 --profile --output results/straka-profile-float64
CUDA_VISIBLE_DEVICES=0 /home/chengcli/pyenv/bin/python -u tests/run_fp32_straka.py \
  --precision float32 --tlim 10 --profile --output results/straka-profile-float32
```

Both passed. The five profiled extra steps occur after saved final fields and
capture of elapsed time/peak memory. The profile includes stage advancement and
redo checks, excluding the runner's CFL/finite checks and snapshots.

| Five-step operator count | FP64 | FP32 |
| --- | ---: | ---: |
| `aten::_to_copy` | 105 | 805 |
| `aten::copy_` | 1925 | 2665 |
| `aten::empty_strided` | 1005 | 1730 |
| `cudaLaunchKernel` | 8665 | 9765 |

The FP64/FP32 profiles recorded CPU/device operator times, counts and
native-library hashes; repository copies have been removed. The results above
confirm overhead in the dry
Straka path without invoking chemistry as an explanation.

The proposed native changes target horizontal face-state promotion/demotion and
forcing scratch allocation. Pressure cancellation remains FP64 with the same
operation order; covariance consumers keep the existing path. Reused forcing
buffers preserve FP64 accumulation and explicit FP32 rounding before addition.
They are private scratch, not restart fields, and are excluded from the existing
registered-buffer-only storage metric. Updated GPU peak memory and throughput
must be measured with a rebuilt binary before claiming any benefit.

### Implemented changes and CPU equivalence

With `dynamics/flux-covariance: false`, `hydro_forward.cpp` now keeps active
horizontal FP32 face density/velocity/species in FP32 and batches the two
pressure rows for FP64 gauge restoration, pressure flooring and background
subtraction. That branch avoids a full FP64 face-state copy and conversion back
to FP32. The gravity-balanced vertical sweep already kept faces in FP32.
Covariance consumers and disabled sweeps retain their previous paths.
Flux covariance defaults to true, and the Straka runner used below does not
override it: those Straka runs did **not** exercise this new horizontal branch.
The reconstruction input is copied directly to FP32, with its pressure anomaly
computed in FP64 before assignment.

Forcing uses reusable FP64 accumulation and FP32 rounded-increment buffers.
Accumulation and rounding order are preserved; buffers are zeroed each stage
and recreated on module reset or shape/device changes. This reduces allocation
churn, not necessarily peak memory: the buffers now remain allocated between
steps. They are unregistered scratch, excluded from registered-only storage
counts. Added tests check that prior forcing cannot affect a zero-duration stage
and that disabled fluxes produce no pressure source.

Local build and targeted checks:

```sh
PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target snap_release
PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/test_fp32_compute.py
PYTHONPATH=/tmp/snapy-rebase-python python -u tests/run_fp32_straka.py \
  --precision float32 --device cpu --profile --output /tmp/straka-baseline-900
PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py \
  --precision float32 --device cpu --profile --output /tmp/straka-optimized-900
```

The first build attempt failed because the temporary kintera installation lacked
headers. Headers were restored from matching local kintera 2.6.10 source; the
build then passed. The optimized library was copied to a separate temporary
snapy package, preserving the baseline installation. The 15 unique foundation
C++ executables were rebuilt against the changed Hydro class layout. Python CTest
was configured with `-DSNAPY_TEST_PYTHONPATH=/tmp/snapy-straka-opt-python`.

The two covariance-on 900-second runs passed and saved **bit-identical** theta,
primitive, conserved and tracer arrays at 0, 300, 600 and 900 seconds. These
results validate the shared-path/forcing changes, not the covariance-off
horizontal sweep. Repository copies of the raw CPU equivalence/profile data
have been removed.
The five-step CPU profiles show `_to_copy` calls 2270→2255, `copy_` 4265→4250,
`empty_strided` 3195→3165, `clone` 330→315 and `empty` 1260→1245. Operator counts
do not measure the inactive covariance-off branch's face-state conversion saving.
CPU elapsed times overlapped other builds/tests and are not a controlled speed claim.

The long optimized run preceded the final guard for disabled horizontal sweeps;
the covariance-off branch was inactive in Straka both before and after that guard.
The final guarded binary was checked with the then-complete FP32 regression
suite and a separate covariance-on 30-second Straka comparison. Neither provided
coverage of the new covariance-off branch; PR #368 review identified that gap.
At this historical checkpoint, dungeon3 remained on the original installed
binaries: no optimized GPU speedup or memory result was claimed before rebuilding.
Kintera source/API is unchanged.

Final guarded-build validation:

```sh
ctest --test-dir build --output-on-failure -L 'fp32-(foundations|science|acceptance)'
PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py \
  --precision float32 --device cpu --tlim 30 --output /tmp/straka-final-30
```

**40/40 CTests passed**, 163.30 s: 37 foundations, two acceptance tests and one
science smoke test. This includes restart/multi-block, VIC, moist 1000-step and
standalone Jupiter checks; it does not clear the separate long paired Jupiter
science gate. The final 30-second Straka run passed and was bit-identical to the
original binary for all four saved fields. Repository copies of the raw checks
and final binary hashes have been removed. Google clang-format verification,
Python compileall/Ruff F401/F821
and `git diff --check` passed. Optimized GPU timing remains unmeasured.

### PR #368 review follow-up — covariance-off coverage (2026-11-16)

The historical 40/40 result above did not cover the new horizontal sweep.
`tests/test_fp32_horizontal_sweep.py` now explicitly sets
`dynamics/flux-covariance: false` and tests x2 and x3 independently against the
FP64 full-state reconstruction path. Both precisions start from the same FP64
initial condition, with a horizontally varying pressure background and velocity.
The other flux directions and gravity are disabled; zero species fractions
isolate the dry limit. Each arm runs 20 RK3 steps at dt=0.05 s and asserts the
requested storage dtype/device, active direction, finite state, positive density
and nonzero reference evolution.

Acceptance requires maximum active-velocity error <=5e-5 m/s and maximum
per-field scaled primitive error <=2e-5. The latter divides each field's absolute
error by its FP64 maximum magnitude, floored at one. CPU and CUDA CTest entries
carry the `fp32-foundations` and `fp32-compute` labels; the CUDA entry is registered
in CUDA builds and returns skip code 125 when CUDA is unavailable.

The local CPU library and Python bindings were rebuilt from the reviewed head.
Dungeon3 reused the matching isolated head library on RTX 5090 with PyTorch
2.10.0+cu128 and kintera 2.6.10. Its core library SHA-256 was
`8790a6a528893137ef209e0c8083976fd4260d67d3d734dacc9afa8fe3dd628c`.
Separate temporary libraries changed only `pressure -= background` to
`pressure -= 0.5 * background` inside the FP32 horizontal branch. This mutation
leaves the FP64 reference path intact; the working source and normal libraries
were not modified.

| Case | x2 max velocity error (m/s) | x3 max velocity error (m/s) | Result |
| --- | ---: | ---: | --- |
| Local macOS CPU, reviewed implementation | 1.221002e-6 | 7.557630e-7 | Pass |
| Dungeon3 CPU, reviewed implementation | 1.220634e-6 | 8.022738e-7 | Pass |
| CUDA, reviewed implementation | 1.221002e-6 | 7.557630e-7 | Pass |
| Local macOS CPU, half-background mutation | 4.607246e-1 | 2.284097e-1 | Fails both directions |
| CUDA, half-background mutation | 4.607246e-1 | 2.284097e-1 | Fails both directions |

The normal scaled primitive errors on local CPU and CUDA were 7.710517e-7 (x2)
and 6.785613e-7 (x3); dungeon3 CPU gave 7.708191e-7 and 7.203210e-7.
These are focused regression results, not new covariance-off Straka science or
performance measurements. The separate long paired Jupiter gate remains open.

Validation commands (local CTest uses `/tmp/snapy-pr368-python`):

```sh
ctest --test-dir build --output-on-failure \
  -R '^test_(fp32_(horizontal_sweep|compute)|perturbation_flux)_python$'
# 3/3 passed on CPU.

# On dungeon3, using the matching isolated package:
ctest --test-dir /tmp/snapy-hllc-2f321dc/src/build --output-on-failure \
  -R '^test_fp32_horizontal_sweep(_cuda)?_python$'
# 2/2 passed: CPU and CUDA arms on dungeon3.

PYTHONPATH=/tmp/pr368-mutant-cpu/package python -u \
  tests/test_fp32_horizontal_sweep.py --device cpu
# Expected exit 1: x2/x3=[False, False].

# On dungeon3, from /tmp/pr368-review:
PYTHONPATH=/tmp/pr368-review/mutant/package /home/chengcli/pyenv/bin/python -u \
  tests/test_fp32_horizontal_sweep.py --device cuda
# Expected exit 1: x2/x3=[False, False].
```

The cast-cache header now documents its restriction against independently
versioned aliases and unversioned writes, including raw-pointer writes and CUDA
graph replay. No solver, provider API or dependency version changed; this review
follow-up introduces no cross-repository compatibility change.

### PR #368 review follow-up — disabled-axis guard (2026-11-17)

Review of head `fb408de867c5fc904e6fe5b8b192c4afcdaf3bb8` identified another
coverage gap: the paired test's disabled horizontal axis had uniform pressure,
and the all-disabled check retained covariance on. Neither detected removal of
the `disable_flux_x2`/`disable_flux_x3` checks in `active_horizontal`.

The paired test now includes a 1% pressure cosine along the disabled horizontal
axis as well as the active axis, and asserts that the other axis is disabled.
Its existing all-field comparison catches the resulting unwanted transverse
motion. `test_fp32_compute.py` now runs its all-disabled, at-rest check with
covariance both on and off, requiring bitwise unchanged conserved state after
two steps at dt=0.05 s. Existing tolerances are unchanged.

In separate temporary CPU/CUDA libraries, replaced only the guard with
`bool active_horizontal = dim == DIM2 || dim == DIM3;`, leaving sweep arithmetic
intact. Both modified tests pass with the normal library and fail (exit 1) with
this mutation on local CPU and dungeon3 CUDA (RTX 5090):

| Check | Normal library | Guard mutation |
| --- | ---: | ---: |
| Paired sweep, x2 active: scaled primitive error | 6.633744e-7 | 6.169843e-1 |
| Paired sweep, x3 active: scaled primitive error | 7.771514e-7 | 1.137464 |
| All disabled, covariance off: max conserved change | 0 (bitwise equal) | 0.1290788 |
| All disabled, covariance on: max conserved change | 0 (bitwise equal) | 0 (bitwise equal) |

The paired test's scaled-error limit is 2e-5. Active-velocity error alone does
not detect this mutation; checking all primitive fields is essential. This
follow-up tests the disabled-axis guard, not the separate non-blocking mutation
that removes only `!flux_covariance()`.

Validation commands:

```sh
ctest --test-dir build --output-on-failure \
  -R '^test_fp32_(horizontal_sweep|compute)_python$'
# Local CPU: 2/2 passed, using /tmp/snapy-pr368-python.

# On dungeon3:
ctest --test-dir /tmp/snapy-hllc-2f321dc/src/build --output-on-failure \
  -R '^test_fp32_(horizontal_sweep(_cuda)?|compute)_python$'
# 3/3 passed, including the CUDA paired sweep.
cd /tmp/snapy-hllc-2f321dc/src
SNAP_TEST_DEVICE=cuda PYTHONPATH=/tmp/snapy-hllc-2f321dc/package \
  /home/chengcli/pyenv/bin/python -u tests/test_fp32_compute.py
# Passed, including both covariance settings for the all-disabled check.

# Local CPU guard mutant: each command must exit 1 at the numerical assertion.
PYTHONPATH=/tmp/pr368-guard-mutant-cpu/package python -u \
  tests/test_fp32_horizontal_sweep.py --device cpu
PYTHONPATH=/tmp/pr368-guard-mutant-cpu/package python -u \
  tests/test_fp32_compute.py

# On dungeon3, from /tmp/snapy-hllc-2f321dc/src: each command exited 1 as expected.
PYTHONPATH=/tmp/pr368-guard/mutant/package /home/chengcli/pyenv/bin/python -u \
  tests/test_fp32_horizontal_sweep.py --device cuda
SNAP_TEST_DEVICE=cuda PYTHONPATH=/tmp/pr368-guard/mutant/package \
  /home/chengcli/pyenv/bin/python -u tests/test_fp32_compute.py
```

Normal binaries were reused; only the isolated guard-mutant libraries were
built. Working solver source and normal libraries were unchanged. No provider
API/dependency changes or new cross-repository compatibility risks.

## Repeated GPU-only Straka comparison (installed binary)

At the user's request, compare FP64 and FP32 only on dungeon3 GPU 0 (RTX 5090),
with three sequential pairs and reversed precision order in pair 2. Each run
uses the same 64×256 Straka case, dt=0.1 and 9000 steps to 900 s. Timing includes
step validation and snapshots as in the original runner; it is end-to-end time
for GPU execution, not CUDA-kernel-only time. No CPU simulation timings enter
this comparison.

The installed native hashes were checked before starting and still match the
pre-optimization GPU reports. These trials therefore characterize the installed
snapy 2.13.27 binary, **not the new native optimizations**. The isolated optimized
GPU build remains pending clarification of the earlier no-recompilation request.

From `/tmp/snapy-m6-hsaZRT`:

```sh
mkdir -p results/straka-gpu-repeat
for trial in 1 2 3; do
  if [ "$trial" = 2 ]; then
    precisions="float32 float64"
  else
    precisions="float64 float32"
  fi
  for precision in $precisions; do
    CUDA_VISIBLE_DEVICES=0 /home/chengcli/pyenv/bin/python -u tests/run_fp32_straka.py \
      --precision "$precision" --output "results/straka-gpu-repeat/$trial-$precision" \
      > "results/straka-gpu-repeat/$trial-$precision.log" 2>&1 || exit 1
  done
done
```

Repeated GPU results: **6/6 runs passed**, each completing 9000 steps. FP64 times
were 63.4482, 63.0573 and 63.5365 s; FP32 times were 69.9918, 70.2651 and
70.2006 s. Medians are **63.4482 / 70.2006 s**, so FP32 latency is **10.6423%**
higher. Sample standard deviations are 0.2550 / 0.1428 s. Final theta, primitive,
conserved and tracer fields are bit-identical across repetitions of each dtype.

See [the Kratos-based diagnosis](fp32-kratos-diagnosis.md) for subsequent analysis
of the precision policy, implicit C++ promotions and GPU kernel attribution.


## Updated HLLC GPU validation — 2026-11-16

The isolated current-source GPU build and six full paired Straka runs are now
complete. See [the HLLC GPU validation report](fp32-hllc-gpu-validation.md) for
exact commands, hashes, correctness, memory and timing results. HLLC's FP32 kernel
is 2.63× faster than FP64 in the short profile, but full-run medians are 69.00 s
FP64 / 76.26 s FP32 (10.5% slower FP32). The 2× end-to-end goal remains open.
Kintera and dungeon3's installed snapy were unchanged; historical timings above
remain measurements of the previous installed binary.
