# V3 project progress: FP32 and CUDA graphs

Updated: 2026-11-17. Plan: [V3_ROADMAP.md](V3_ROADMAP.md).

This tracker combines FP32 numerical implementation/science milestones M0–M6
with CUDA graph milestones G0–G7 across snapy, kintera, pyharp and UCX/commux.
M6 remains in progress; the general graph milestones remain planned beyond the
validated dry Straka prototype. Historical checkpoints below are dated records,
not replacements for the current milestone tables.

## Milestones

### FP32 numerical implementation and science

| Milestone | Status | Evidence / next action |
|---|---|---|
| M0 Baseline and records | Merged baseline | PR #326 merged as 2d92c92; M3a validation below |
| M1 Reference consolidation | Merged (#326) | Shared operator, cache invalidation, composition restart; CPU and reviewer CUDA evidence below |
| M2 Thermodynamic perturbations | Merged (#326) | Delta thermodynamics, exact total reads and FP32 floors; hydro wiring follows |
| M3 Perturbation dynamics | Merged (#360) | 31/31 foundation tests pass after main rebase; split reference, flux/source coupling, seams and restart; CUDA execution pending |
| M4 True FP32 | Merged (#361) | 35/35 foundations after main rebase; 1000-step moist/Jupiter acceptance; FP32 VIC with checked FP64 fallback; GPU validation pending |
| M5 Coupled workflow/restart | CPU implementation merged (#367) | 37/37 foundations, exact coupled/bundled continuation; external driver/distributed/GPU validation pending |
| M6 Science/performance | In progress | Budget/seeded-comparison harness; GPU smoke passed; Jupiter comparison and installed GPU restart compatibility remain open |

### CUDA graph execution

| ID | Deliverable | Owner(s) | Dependencies | Status |
|---|---|---|---|---|
| Existing | Dry Cartesian Straka graph prototype and cast caches | snapy | — | Validated for the restricted fixture; evidence below |
| G0 | Version/capability inventory, baselines and capture probes | snapy + providers | — | Plan/source inventory recorded; runtime audit/probes pending |
| G1 | General segmented executor, stable RK storage, device diagnostics | snapy + pyharp | G0 | Planned |
| G2 | Capture-safe moist thermodynamics and source terms | kintera + snapy | G1 | Planned |
| G3 | VIC graphs with stage-local FP64 fallback and retry | snapy | G1; G2 for moist gate | Planned |
| G4 | Local multi-block and UCX distributed Cartesian execution | snapy; commux if needed | G1; G2/G3 for combined gate | Planned |
| G5 | Eager FP32 cubed-sphere numerical extension | snapy | G0 numerical baselines | Planned |
| G6 | Cubed-sphere graph execution and full combined matrix | all providers as needed | G2–G5 | Planned |
| G7 | Production API, compatible releases, CI and performance acceptance | snapy + providers | G6 | Planned |

## Decisions

- UCX remains the communication backend. Capture GPU computation/packing and
  unpacking in segments; initially keep UCX posting/progress/completion and
  global CFL/retry/correction reductions outside capture.
- One GPU per process initially; support multiple blocks on that GPU. No
  regridding. VIC retains nb1=1, with horizontal block/process decomposition.
- Preserve physical operators, precision policy, diagnostics and rollback.
  Stable storage is preferred; blanket `set_`→`copy_` replacement is not the plan.
- Coordinate snapy, kintera and pyharp interfaces; modify commux only when its
  inspected completion/stream API or measured synchronization cost requires it.
- FP32 cubed-sphere numerics must be implemented and validated eagerly before
  claiming FP32 cubed-sphere graph support. Current guards cannot just be removed.

## Existing evidence

- [Straka graph replay](docs/notes/fp32-straka-graph-replay.md): eager/graph parity,
  storage/dt changes, rollback and retry exhaustion; full 9000-step comparisons.
- [Dtype-copy caching](docs/notes/fp32-straka-cast-cache.md): 12 paired full runs,
  bit-identical same-precision fields, 78 conversions/step removed. Captured
  FP32 GPU time remains 4.15% above FP64; no general 2× precision claim.
- [Capture-enabling EOS changes](docs/notes/fp32-cache-validation.md): cached
  constants and deferred heat-capacity validation.
- These results do not validate moist, VIC, multiple-block or distributed graph
  execution. They also do not establish eager FP32 cubed-sphere support.

## Work log

### 2026-11-16 — Cross-repository plan recorded

- Inspected snapy's RK state updates, implicit failure/fallback path, local-block
  worker/exchange orchestration, cached halo buffers, UCX adapter and global
  retry handling; kintera v2.6.10 temperature/equilibrium/table/failure paths;
  and pyharp's allocating RK stage combination.
- Identified separate numerical guards for Cartesian-only anomaly storage and
  FP32 VIC, temperature-dependent FP32 perturbation EOS, and cubed-sphere full
  axial-momentum mode. They have explicit prerequisites/capability gates in the
  roadmap rather than being treated as graph-capture fixes.
- Full commux implementation/version audit remains G0 work. No changes were
  made to provider code or snapy runtime in this planning step.
- Documentation validation: `git diff --check` passed (exit 0). A Python
  consistency check passed for all 30 local Markdown links in the four touched
  documents and the G0–G7 milestones in both plan and tracker. No runtime tests
  were run for this documentation-only change.
- Cross-repository compatibility requirements are recorded in the roadmap:
  additive provider APIs, pinned revision sets, matching native/binding rebuilds,
  and graph invalidation after restart/configuration changes. No new runtime
  dependency or compatibility change is introduced by these documents.

### 2026-11-16 — V3 documentation consolidated

- Combined FP32 progress/history and CUDA graph tracking in this file, and both
  roadmaps in [V3_ROADMAP.md](V3_ROADMAP.md). Removed the four predecessor files
  and updated repository references, including README and the CTest comment.
- Preserved M0–M6 and G0–G7 milestones, numerical requirements, historical
  implementation/validation records and cross-repository ownership.
- Documentation paths in historical command listings now use the consolidated
  filenames; other command arguments and recorded results are retained.
- Validation: `git diff --check` passed. Content-preservation checks passed;
  all 27 checked local links resolve, M0–M6/G0–G7 appear in both documents, and
  no predecessor filenames remain in repository references. No runtime or
  dependency changes; runtime tests are not needed for this documentation/comment-only edit.

### 2026-11-16 — PR #368 covariance-off review coverage

- Added a paired FP32/FP64 regression with `flux-covariance=false`, independent
  x2/x3 sweeps, a nonuniform pressure background and 20 RK3 steps per arm.
  Registered CPU and CUDA variants under `fp32-foundations`/`fp32-compute`.
- Corrected the M6 note: historical Straka runs and the old 40/40 suite kept
  covariance enabled and did not exercise the new horizontal branch. Detailed
  commands, tolerances and results are in the
  [review follow-up](docs/notes/fp32-m6-validation.md#pr-368-review-follow-up--covariance-off-coverage-2026-11-16).
- Rebuilt local CPU core/bindings from reviewed head
  `3289c456404ecd50ba0ecf7947e545940c66a65f`; reused the matching isolated CUDA
  build on dungeon3 (RTX 5090, kintera 2.6.10). Local CPU and CUDA maximum
  velocity errors were 1.221002e-6 m/s (x2) and 7.557630e-7 m/s (x3).
- Mutation check: separate temporary libraries subtracting half the horizontal
  pressure background failed both directions on CPU and CUDA (expected exit 1).
  Errors increased to 0.4607246 and 0.2284097 m/s against a 5e-5 m/s limit.
  No mutation was applied to the working source or normal libraries.
- Local validation:
  `ctest --test-dir build --output-on-failure -R '^test_(fp32_(horizontal_sweep|compute)|perturbation_flux)_python$'`
  passed **3/3**. On dungeon3:
  `ctest --test-dir /tmp/snapy-hllc-2f321dc/src/build --output-on-failure -R '^test_fp32_horizontal_sweep(_cuda)?_python$'`
  passed **2/2**, including CUDA.
- `python -m compileall -q tests/test_fp32_horizontal_sweep.py tests/test_perturbation_flux.py`,
  `ruff check --no-cache --select F401,F821 tests/test_fp32_horizontal_sweep.py tests/test_perturbation_flux.py`,
  `clang-format --style=Google --lines=9:16 src/utils/tensor_cast_cache.hpp | diff -u src/utils/tensor_cast_cache.hpp -`,
  and `git diff --check` passed.
- Documented the cast-cache restriction on independently versioned aliases and
  unversioned writes (including CUDA graph replay). No solver, provider API or
  dependency changes; no new cross-repository compatibility risk. M6 remains
  open for the existing science/performance gates.

### 2026-11-17 — PR #368 disabled-axis guard coverage

- Added pressure variation along the disabled horizontal axis in the paired
  sweep regression. The all-disabled FP32 check now runs with covariance both
  on and off, retaining its bitwise-equality requirement.
- Tested the exact reviewer mutation
  `bool active_horizontal = dim == DIM2 || dim == DIM3;` in separate temporary
  libraries. Both tests pass with the normal library and fail with the mutation
  on local CPU and dungeon3 CUDA. Paired scaled errors become 0.6169843/1.137464
  against 2e-5; the all-disabled covariance-off state changes by 0.1290788 instead
  of remaining bitwise unchanged. These errors are from intentionally broken
  temporary builds, not the working solver.
- Validation:
  `ctest --test-dir build --output-on-failure -R '^test_fp32_(horizontal_sweep|compute)_python$'`
  passed **2/2** locally.
  `ctest --test-dir /tmp/snapy-hllc-2f321dc/src/build --output-on-failure -R '^test_fp32_(horizontal_sweep(_cuda)?|compute)_python$'`
  passed **3/3** on dungeon3.
  `SNAP_TEST_DEVICE=cuda PYTHONPATH=/tmp/snapy-hllc-2f321dc/package /home/chengcli/pyenv/bin/python -u tests/test_fp32_compute.py`
  also passed from the isolated dungeon3 source directory.
- Python compileall, Ruff F401/F821 and format checks, and `git diff --check`
  passed. Detailed commands and mutation evidence are in the
  [validation note](docs/notes/fp32-m6-validation.md#pr-368-review-follow-up--disabled-axis-guard-2026-11-17).
- No solver/provider changes or new cross-repository compatibility risks.
  M6's outstanding science/performance gates are unchanged. The separate
  non-blocking covariance-condition mutation has not been tested here.

## FP32 implementation and validation history

Historical foundation baseline (2026-11-08): snapy work was recorded on
`fp32-step1-anomaly`, based on `fe003dd`.
The paired kintera change is commit `80ac117` on `fp32-perturbation-thermo`, based
on user-rebased `v2.6.8` / `e646fe5`. Fresh CPU validation uses the paired source
builds, not the globally installed packages.

### Validation log

- Prior inspection: `ctest --test-dir build --output-on-failure -R
  'test_(balance_column|hydro_ref_x1)\.release$'`: 2/2 passed using existing
  binaries. Registry lacked newer anomaly tests; not current-source validation.
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON`: passed;
  resolved installed kintera 2.6.6.
- `cmake --build build --parallel 4 --target test_balance_column.release test_hydro_ref_x1.release test_fp32_storage.release`:
  failed before project compilation because cached Xcode SDK was missing
  (`cstddef`, `stdio.h` not found).
- Reconfigured successfully with
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`.
  Retried the same build; in progress. Logs: `/tmp/snapy-fp32-build.log`.
- User rebased kintera: source `v2.6.8`, commit `e646fe5`; installed Python package
  still 2.6.6. Isolated source copy at `/tmp/snapy-fp32-kintera-2.6.8` configured
  successfully using the installed SDK; building `kintera_release`.

### Decisions — 2026-11-08

- Jupiter Cartesian ideal-moist first; perturbation kernels before true FP32.
- Initial balancing remains opt-in; preserve unbalanced-state residual forces.
- Reuse well-balanced machinery; distinguish static and dynamic references.
- Kintera source edits authorized for moist physics. Keep repository responsibilities.
- Record exact commands/results and cross-repo compatibility at each checkpoint.

### Environment and outstanding work

Local host is macOS/CPU; CUDA and production GPU validation require another host.
The initial 2026-11-08 validation pinned the isolated kintera v2.6.8 source
build plus its paired change. Those historical Python tests used `/tmp/snapy-fp32-python` (kintera 2.6.8+fp32
and the rebuilt snapy extension/library). Global Python packages were not replaced.
Coupled Jupiter cases live outside this repository and were not found in the
local workspace search. The temporary build/install paths are local validation
artifacts; a clean host must build/install the paired sources itself.

### Implementation checkpoint — 2026-11-08

- Added `HydroReferenceX1Operator` shared by runtime and `balance_column`.
  Retains legacy nonuniform vs stretched operator choice, runtime seam ordering,
  and wb_ref4 placement. Geometry cache checks identity/version/dtype/device.
- Added tests comparing shared evaluation with existing dispatch for float/double,
  uniform/nonuniform/stretched grids and cache invalidation on edits/conversion.
- Added FP64 species background and optional `hydro_ref_species` restart field.
  Legacy restarts remain supported for storage; absent composition is explicitly
  unavailable, never inferred from evolved species or stale initialization.
- Added species background conversion and restart checks to anomaly tests.
- Compatibility: new restart field is additive. Older writers lack composition
  needed by future perturbation thermodynamics; existing anomaly storage does not
  require it. No kintera source changes yet.

### Thermodynamics checkpoint — 2026-11-08

- Kintera source at v2.6.8 now has `eval_const_cv_pressure_delta` and
  `test_pressure_delta`: direct constant-cv mixture pressure increments, with
  tests for sub-ulp heating and composition changes. Changes are in the original
  neighboring checkout; the temporary build copy contains the same edits.
- `cmake --build /tmp/snapy-fp32-kintera-2.6.8/build --parallel 4 --target test_pressure_delta.release`:
  passed.
- `ctest --test-dir /tmp/snapy-fp32-kintera-2.6.8/build --output-on-failure -R '^test_pressure_delta.release$'`:
  1/1 passed (two gtest cases).
- Snapy now has an internal ideal-moist perturbation result carrying background
  and delta density/pressure. Uses source double coefficients, species energy
  offsets and kinetic energy; does not mutate or limit input. Not yet connected
  to the hydro sweep. Variable heat capacities are explicitly unsupported here.
- FP32 total reconstruction promotes before addition. Active FP32 anomaly floors
  round upward when needed; inactive cells stay encoded as supplied.
- Added direct snapy EOS tests for these contracts; validation pending.
- Compatibility: this branch now requires the paired kintera header helper;
  CMake checks capability explicitly instead of pretending an unreleased API is
  present in stock 2.6.8. Python extension headers follow the imported kintera
  package so isolated paired builds do not silently use global-install headers.
- Snapy reconfiguration selects `/tmp/snapy-fp32-kintera-2.6.8` headers and its
  `build/lib/libkintera_release.dylib` via `_KINTERA_HEADER_DIR`,
  `KINTERA_INCLUDE_DIR`, and `KINTERA_LIBRARY`; rebuilding EOS and reference tests.

### Gravity-work checkpoint — 2026-11-08

- Added a shared local-potential-difference face-work helper. Cartesian explicit
  hydro and the implicit post-solve correction now use it; non-Cartesian explicit
  behavior remains on its existing formula. Physical wall and radial correction
  handling stays in the callers.
- Tests cover FP64 energy identity, conservation with closed walls, and FP32
  translation invariance with large-origin FP64 coordinates. Validation pending.
- Kintera changes are on `fp32-perturbation-thermo`, based on user-rebased v2.6.8.
- Isolated kintera wheel `2.6.8+fp32` built and installed at
  `/tmp/snapy-fp32-python`; import check confirms that path and version.
- `git diff --check` and `python -m py_compile tests/test_anomaly_storage.py setup.py`:
  passed. These are static checks, not numerical validation.

### CPU validation checkpoint — 2026-11-08

- First expanded EOS run exposed mixed-dtype assumptions in coordinate vector
  transforms and species repair. Fixed explicit geometry/repair compute views;
  rerun passed (no tolerance relaxation).
- `ctest --test-dir build --output-on-failure -R 'test_(eos|coordinate|balance_column|hydro_ref_x1|fp32_storage|gravity_work_cache_to|implicit_face_work_jacobian)\.release$'`:
  **7/7 passed**. CUDA parameterizations skip on this CPU build.
- `PYTHONPATH=/tmp/snapy-fp32-python python tests/test_anomaly_storage.py`:
  **PASS**, all seven arms. Includes species-background storage and restart;
  anomaly on/off maximum relative total-state difference 5.678e-13 over 10 steps;
  same-dtype restart is bit-identical with anomaly both on and off.
- HLLC now has a separate C++ `forward_perturbations` contract. The shared scalar
  kernel accepts face pressure backgrounds, preserves small pressure differences,
  and uses total pressure in energy transport. CPU, CUDA and tensor dispatch
  are wired; normal `forward` retains its total-state contract.
- `ctest --test-dir build --output-on-failure -R '^test_riemann.release$'`:
  **1/1 passed**, including all direction/contact-branch comparisons, sub-ulp
  pressure differences, moist species fluxes and CPU-vs-tensor fallback parity.
  CUDA tests were skipped; CUDA source is not compiled on this host.
- Final configure selects paired kintera **2.6.8** (source plus local changes),
  and `SNAPY_TEST_PYTHONPATH=/tmp/snapy-fp32-python` ensures Python CTests use
  the rebuilt branch instead of the globally installed snapy.

### Remaining implementation (not completed by the foundation work)

1. Split the shared hydrostatic reference's linear scan from its nonlinear
   anchor/guards/density operator. Cache static background evaluations and
   invalidate them on reference/restart/geometry changes.
2. Wire pressure and density perturbations into reconstruction and the new HLLC
   entry point in all directions. Preserve static residual forces, wall parity,
   distributed anchors, seam face pressures, and positivity fallback semantics.
   A balanced reference's constant pressure gauge must not reintroduce a large
   FP32 reconstruction offset. Nonlinear reconstruction must not be treated as
   additive for arbitrary nonuniform background residuals.
3. Add the opt-in dynamics switch only with the complete reference/flux/source
   contract. It is deliberately not exposed as a working hydro mode yet.
4. Implement M4 dtype plumbing/FP64 implicit and chemistry boundaries, then M5
   coupled driver/regrid/dtype-aware restart and M6 science/performance gates.
5. CUDA host and reduced Jupiter workspace location requested asynchronously;
   neither is available locally. CUDA/production acceptance remains pending.

### Historical storage-only results

These are reported by the original roadmap, **not rerun in this work**. Errors
are maximum velocity differences from its FP64 total-storage baseline.

| Reduced case | FP32 total storage | FP32 anomaly storage |
|---|---:|---:|
| Rest, 200 steps | 2.6e-5 m/s | 3.1e-7 m/s |
| Bottom heating, 200 steps | 2.6e-5 m/s | 3.9e-7 m/s |
| Full physics, 500 steps | 6.8e-4 m/s | 1.1e-5 m/s |

These storage-emulation measurements do not demonstrate true FP32 arithmetic.

### Reproducing the current foundation gate

Current CMake configuration was generated with:

```sh
PYTHONPATH=/tmp/snapy-fp32-python cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  -D_KINTERA_HEADER_DIR=/tmp/snapy-fp32-kintera-2.6.8/kintera \
  -DKINTERA_INCLUDE_DIR=/tmp/snapy-fp32-kintera-2.6.8 \
  -DKINTERA_LIBRARY=/tmp/snapy-fp32-kintera-2.6.8/build/lib/libkintera_release.dylib \
  -DKINTERA_DATA_DIR=/tmp/snapy-fp32-python/kintera/data \
  -DSNAPY_TEST_PYTHONPATH=/tmp/snapy-fp32-python
```

The final C++ build command (passed):

```sh
cmake --build build --parallel 4 --target \
  test_eos.release test_coordinate.release test_fp32_storage.release \
  test_balance_column.release test_hydro_ref_x1.release \
  test_gravity_work_cache_to.release test_implicit_face_work_jacobian.release \
  test_riemann.release test_stretched_x1_ref.release \
  test_hydro_ref_x1_rf_wall.release test_wb_ref_wall.release \
  test_wb_wall_corner.release test_x1_seam_split.release
```

The local Python extension was built with:

```sh
PYTHONPATH=/tmp/snapy-fp32-python \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=4 \
python -m pip install --no-deps --no-build-isolation \
  --target /tmp/snapy-fp32-python .
```

After the final library rebuild, refreshed its packaged copy:
`cp build/lib/libsnap_release.dylib /tmp/snapy-fp32-python/snapy/lib/libsnap_release.dylib`.
Use matching include/library/install paths on another host; the SDK option is
macOS-specific. `fp32-foundations` labels the 26 C++/environment cases and seven
Python integration cases for the work implemented so far. It is not a substitute
for M3 hydro integration or the M4–M6 acceptance gates.

### Final validated checkpoint — 2026-11-08

- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **33/33 passed**, 57.69 seconds. Covers the current EOS/reference/HLLC/gravity
  changes, wall and stretched-grid variants, in-process split columns, and seven
  Python integrations using this branch. CUDA parameterizations are skipped;
  no production-GPU or remote-process validation is claimed.
- `ctest --test-dir /tmp/snapy-fp32-kintera-2.6.8/build --output-on-failure -R '^test_pressure_delta.release$'`:
  **1/1 passed** (two numerical cases).
- `git diff --check` in snapy and kintera, and
  `python -m py_compile tests/test_anomaly_storage.py setup.py`: **passed**.
- The overall overhaul is **not complete**: M3 reference/reconstruction/source
  integration and M4–M6 remain open as listed above. No true-FP32 Jupiter run or
  performance claim is made. CUDA host/workspace information is still pending.
- Cross-repo requirement: build the paired kintera helper with this snapy branch;
  stock kintera 2.6.8 lacks the new header API. New restart composition metadata
  is additive; legacy anomaly storage still works, but perturbation EOS requires
  the original composition rather than guessing it from an evolved restart.

### PR #326 rebase and CI repair — 2026-11-09

- Rebased onto snapy main `3c891cb` (#337 radial wall refresh and #338 species
  positivity limiter). The rebase applied without textual conflicts.
- Kintera #155 merged as `b8e726ce2fb1538aa15f0787fdcef732d9cdc997`, tagged
  `v2.6.9`. Raised the CMake, Python build/runtime, and release-workflow minimum
  to 2.6.9; removed the temporary unreleased-header probe.
- The failed macOS CI job installed kintera 2.6.6 and stopped in CMake. PyPI does
  not yet offer 2.6.9, so CI builds the exact merged/tagged commit from source
  before configuring snapy. Package release builds require that dependency to
  be published (or preinstalled from source); a version bump alone is insufficient.
- Resolved the semantic interaction with #338: species positivity now uses
  total dry density through `dry_density`, including in anomaly mode. Added
  2-D/3-D anomaly cases to `test_tracer_ratio.py` and its CPU test to the
  `fp32-foundations` gate. Scalar transport already used total dry density.

Validation uses a fresh source build of kintera v2.6.9 in
`/tmp/snapy-326-kintera` and an isolated install at `/tmp/snapy-326-python`.
The earlier 2.6.8 paired-build results above remain historical checkpoints.

Reproduction (macOS SDK flag is host-specific):

```sh
git clone --branch v2.6.9 --depth 1 https://github.com/chengcli/kintera.git /tmp/snapy-326-kintera
cmake -S /tmp/snapy-326-kintera -B /tmp/snapy-326-kintera/build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON -DBUILD_TESTS=OFF -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
cmake --build /tmp/snapy-326-kintera/build --parallel 4
(cd /tmp/snapy-326-kintera && SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=4 python -m pip install --no-deps --no-build-isolation --target /tmp/snapy-326-python .)
PYTHONPATH=/tmp/snapy-326-python cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -D_KINTERA_HEADER_DIR=/tmp/snapy-326-python/kintera/kintera -DKINTERA_INCLUDE_DIR=/tmp/snapy-326-python/kintera -DKINTERA_LIBRARY=/tmp/snapy-326-python/kintera/lib/libkintera_release.dylib -DKINTERA_DATA_DIR=/tmp/snapy-326-python/kintera/data -DSNAPY_TEST_PYTHONPATH=/tmp/snapy-326-python
cmake --build build --parallel 4 --target test_eos.release test_coordinate.release test_fp32_storage.release test_balance_column.release test_hydro_ref_x1.release test_gravity_work_cache_to.release test_implicit_face_work_jacobian.release test_riemann.release test_stretched_x1_ref.release test_hydro_ref_x1_rf_wall.release test_wb_ref_wall.release test_wb_wall_corner.release test_x1_seam_split.release test_cubed_sphere_exchange.release
PYTHONPATH=/tmp/snapy-326-python SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=4 python -m pip install --no-deps --no-build-isolation --target /tmp/snapy-326-python .
```

Validation results:

- The dependency configure/build/install and snapy configure/build/install
  commands above all passed; CMake detected kintera **2.6.9**.
- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **34/34 passed**, 62.83 seconds, including the new tracer-ratio cases.
- `ctest --test-dir build --output-on-failure -R '^test_cubed_sphere_exchange.release$'`:
  **1/1 passed**, 6.14 seconds, including main's radial wall refresh regression.
- `PYTHONPATH=/tmp/snapy-326-python python tests/test_tracer_ratio.py`:
  **passed**; both anomaly cases preserve ratios within 7e-16.
- `git diff --check`, `python -m py_compile tests/test_tracer_ratio.py`, and
  `pre-commit run --files V3_PROGRESS.md V3_ROADMAP.md .github/workflows/ci.yml .github/workflows/release.yml CMakeLists.txt cmake/modules/FindKintera.cmake pyproject.toml src/hydro/hydro_forward.cpp tests/CMakeLists.txt tests/test_tracer_ratio.py`:
  **passed**.
- Local validation is macOS CPU only. CUDA and production FP32 acceptance
  remain outstanding; milestone scope is unchanged.

#### Linux CI follow-up

CI run `42645426999` passed pre-commit and macOS (4/4 tests). Linux passed its
build/install but failed one CTest: `test_uranus_cycle1_abort.release`. The dry
abort fixture has no reacting species; kintera 2.6.9's Jacobian attempted to
slice a zero-length species axis and threw before snapy's intended redo/exit
handling. This was reproduced locally with the tagged dependency.

`run_hydro` now skips the chemistry update when its stoichiometry is empty.
The existing abort assertion is unchanged, and failed assertions now include
the captured subprocess log. The two moist Uranus cases continue to execute
chemistry. No kintera source modification is needed for this caller fix.

- `cmake --build build --parallel 4 --target test_uranus_cycle1_abort.release run_hydro.release`: passed.
- `ctest --test-dir build --output-on-failure -R '^test_uranus_cycle1_abort.release$'`:
  **failed before the fix; passed after**, 1/1 CTest containing three cases,
  1.42 seconds.
- `pre-commit run --files examples/run_hydro.cpp tests/test_uranus_cycle1_abort.cpp`
  and `git diff --check`: passed.
- `ctest --test-dir build --output-on-failure -R '^test_example_ic_stretched_python$'`:
  **1/1 passed**, 2.73 seconds.

### PR326_REVIEWS.md follow-up — 2026-11-09

- Restore #332's deficit-based redo criterion with `species_repair_mark` and
  pre-repair species density; use total dry density in anomaly mode. A new
  regression distinguishes a small receiving deficit from a relatively large
  donor change, and checks both storage modes plus a real deficit that must redo.
- Retain the shared Cartesian face-gravity-work calculation. Its different
  arithmetic grouping changes default FP64 results at round-off: reviewers
  measured about **2e-9 relative velocity difference** against main in Straka
  after 40 steps, with density and pressure identical at output precision.
  They isolated the difference to this branch; disabling it restored bitwise
  equality. This is reviewer-provided evidence, not a new local measurement.
  Default FP64 behavior is numerically equivalent, not promised bitwise identical
  to main. This does not change the same-dtype restart-continuation requirement.
- Update all README dependency references to kintera >=2.6.9 and explain source
  installation while wheels are unavailable. No dependency API change is needed.
- Register a CUDA twin of `test_anomaly_storage` with the same assertions and
  a no-device skip. Zoey's review reports successful CUDA compilation and tests
  at cf14479 on RTX 5090; these are external results, not local GPU validation.
- Check finite positive background and total volumetric heat capacity before
  evaluating perturbation pressure. Tests cover zero/negative total capacity
  and zero background capacity.
- Preserve FP32 stored anomalies on restart instead of copying them into the
  file's FP64 totals tensor. The new storage-only restart case failed before
  the fix and verifies dtype, exact bits, and FP64 reference metadata. This is
  serialization coverage, not a claim that full FP32 evolution works.
- Full-model FP32 dtype failures remain M4 work; coupled/solid-boundary restart
  acceptance remains M5. Earlier-round comments and validation are historical;
  the active round-2 requests above are the scope of this follow-up.

Review validation (macOS CPU, tagged kintera 2.6.9):

- `PYTHONPATH=/tmp/snapy-326-python cmake --build build --parallel 4 --target test_eos.release test_fix_vapor_deficit.release test_forcing.release test_uranus_cycle1_abort.release run_hydro.release`:
  passed. The PYTHONPATH is needed for CMake's dependency-version probe; the
  global environment still has kintera 2.6.6.
- Refreshed the isolated Python installation's shared library using
  `cp build/lib/libsnap_release.dylib /tmp/snapy-326-python/snapy/lib/libsnap_release.dylib`.
  Python binding sources are unchanged.
- `ctest --test-dir build --output-on-failure -R '^test_(eos|fix_vapor_deficit|forcing|uranus_cycle1_abort)\.release$'`:
  **4/4 passed**, 3.45 seconds.
- `PYTHONPATH=/tmp/snapy-326-python python tests/test_anomaly_storage.py`:
  **passed**, including the new FP32 serialization case (which failed against
  the previous shared library).
- `PYTHONPATH=/tmp/snapy-326-python python tests/test_anomaly_storage.py --device cuda`:
  returned **125**, the expected no-GPU skip. This is not a CUDA execution result.
- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **34/34 passed**, 58.86 seconds.
- `git diff --check`, `python -m py_compile tests/test_anomaly_storage.py`, and
  `pre-commit run --files README.md V3_PROGRESS.md V3_ROADMAP.md src/eos/equation_of_state.cpp src/eos/ideal_moist.cpp src/mesh/meshblock.cpp tests/test_eos.cpp tests/test_fix_vapor_deficit.cpp tests/test_anomaly_storage.py tests/CMakeLists.txt`:
  passed.

### Rebase onto main 919936d — 2026-11-12

- Rebased the seven PR commits onto main `919936da07d80a310b98ab11da184972a3203274`
  (#340 Roe face frames, #334 CUDA reconstruction launch sizing, #339 optional
  fourth-order x1 conversion). Resolved hydro option conflicts by retaining
  both `conversion-order` and `anomaly`, including cloning and YAML parsing.
- Adapted the new conversion-order 4 path to use total conserved cell averages
  before applying the spatial conversion, then explicitly use the total-state
  EOS entry point. Applying it to stored anomalies would omit background
  curvature and test anomaly density against a total-density floor. The
  anomaly-on/off regression now exercises conversion orders 2 and 4.
- Prior temporary validation packages were no longer present. Built the clean
  neighboring kintera v2.6.10 checkout (`3f5ff938181351a84cb129a9e6314fb89b669b71`)
  and installed it in `/tmp/snapy-rebase-python`; the global package is 2.6.6
  and a package-index install of `kintera>=2.6.9` still found no match.
  No kintera source change or minimum-version change is required.

Validation uses `/tmp/snapy-rebase-python` for the Python package, kintera
headers/library/data, and `SNAPY_TEST_PYTHONPATH`. CMake configured successfully
with kintera 2.6.10; the foundation executables and fresh snapy Python extension
built successfully. Local validation is CPU only; upstream CUDA reconstruction
changes require CUDA CI/hardware.

- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **34/34 passed**, 62.29 seconds, including anomaly on/off with conversion
  orders 2 and 4.
- `ctest --test-dir build --output-on-failure -R '^test_(conversion_order|wb_ref4_order|roe_cubed_sphere_rest)_python$'`:
  **3/3 passed**, 40.42 seconds.
- `pre-commit run --files src/hydro/hydro.hpp src/hydro/hydro_options.cpp src/hydro/hydro_forward.cpp tests/test_anomaly_storage.py V3_PROGRESS.md`,
  `python -m py_compile tests/test_anomaly_storage.py`, and `git diff --check`:
  passed.

### Next milestone: M3a signed pressure scan — 2026-11-12

- Start `fp32-step2-pressure-scan` from merged main `2d92c92` (PR #326).
- Implement the first of three M3 stages listed in V3_ROADMAP.md. The new
  internal API requires an explicit anchor increment and scans mixture-density
  increments directly; it must not derive an exponential anchor, clamp signed
  deltas, or subtract rounded total pressures.
- Consolidate the CPU/CUDA recurrence with the existing guarded total-reference
  scan, retaining default arithmetic and floor behavior. Add a tensor backend
  and CPU/CUDA parameter coverage. Verify vertical relay and halo orientations
  independently of distributed transport.
- No public perturbation-flux switch, background cache, or new restart schema
  in M3a. Nonlinear reference splitting and its cache invalidation belong to M3b;
  hydro reconstruction/flux/source integration belongs to M3c. Kintera's existing
  >=2.6.9 pressure helper is sufficient; no cross-repo source change is needed.


#### M3a implementation and validation

- Added `HydroReferenceX1Operator::scan_pressure(rho, grav, anchor)`, returning
  lower/upper face pressure on each cell, ghosts included. It requires an
  explicit `[nc3,nc2,1]` anchor at the upper face of cell `iu`, accepts signed
  mixture-density/anchor increments, and validates shape/dtype before dispatch.
- CPU/CUDA share `hydro_ref_x1_pressure_scan_impl<T, Clamp>` with the default
  reference evaluator. `Clamp=false` retains signed/zero increments;
  `Clamp=true` retains the existing floor and nonlinear anchor wrapper.
  The tensor fallback uses running faces rather than subtracting cumulative
  totals. No Python API, output/restart schema, or dependency minimum changes.
- Configured against isolated kintera **2.6.10**, using the existing
  `/tmp/snapy-rebase-python` environment. Global Python still imports 2.6.6.
- `PYTHONPATH=/tmp/snapy-rebase-python cmake -S . -B build` and
  `PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target test_pressure_scan_x1.release test_hydro_ref_x1.release test_balance_column.release`:
  passed. Refreshed the packaged library with
  `cp build/lib/libsnap_release.dylib /tmp/snapy-rebase-python/snapy/lib/libsnap_release.dylib`.
- `ctest --test-dir build --output-on-failure -L fp32-pressure-scan`:
  **1/1 passed**, 1.28 seconds (9 CPU cases passed, 8 CUDA cases skipped).
  Includes an independent FP64 summation oracle, float/double tensor-backend
  parity, noncontiguous inputs, signed and sub-ulp increments, linearity,
  zero/reversed gravity, irregular widths, default floor behavior, and a
  two-section column relay including ghosts. Relay coverage is arithmetic,
  not a new multi-process communication test.
- CUDA source is implemented but cannot be compiled/run on this CPU host.
  Tensor fallback arithmetic was tested on CPU; actual MPS execution is untested.

- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **35/35 passed**, 54.89 seconds (including existing wall/stretched-grid/seam,
  EOS, HLLC, gravity, anomaly storage and restart regressions).
- `pre-commit run --files src/hydro/hydro_reference.hpp src/hydro/hydro_ref_x1_impl.h src/hydro/hydro_dispatch.hpp src/hydro/hydro_dispatch.cpp src/hydro/hydro_dispatch.cu tests/test_pressure_scan_x1.cpp tests/CMakeLists.txt V3_ROADMAP.md V3_PROGRESS.md`
  and `git diff --check`: passed.
- M3a implementation is ready for review with CPU evidence. GPU execution and
  actual MPS validation remain outstanding. M3b and M3c are not implemented;
  the new scan is an internal building block, not an enabled hydro mode.

#### M3b/M3c implementation in progress

- Reused the complete CPU/CUDA reference kernel with finite-increment FP64
  scalar pairs, including the M3a recurrence. Nonlinear branches inspect totals;
  the base follows the selected branch, rather than a second independently
  guarded background solution. Exponential/logarithmic increments use
  `expm1`/`log1p`. Floors discard the increment when active.
- Simplified caching: retain the existing validated geometry/stencil cache only.
  Recompute branch-dependent values each sweep; there is no derived background
  cache to invalidate on reference replacement, restart, or thermodynamic change.
- Extended existing anchor relay and owner-cell ghost exchange to carry pressure
  pairs. wb_ref4 retains its existing ordering and total density semantics.
- Wiring opt-in `dynamics: {anomaly: true, perturbation-flux: true}` for Cartesian
  constant-cv ideal-moist HLLC. Reconstruction and sensitive reference operations
  remain FP64 in M3; true FP32 computation remains M4. Immersed solids and MPS
  are outside this initial path. No kintera API/dependency change.
- Pending: compile, nonlinear reference tests, full dynamics equivalence,
  balanced/unbalanced rest, heating, split-column and conservation checks.


#### M3b/M3c completion — 2026-11-12

- M3b and M3c implementation is complete for the documented Cartesian,
  constant-cv ideal-moist HLLC scope. Numerical contract and remaining M4
  boundaries: [fp32-perturbation-dynamics.md](docs/derivations/fp32-perturbation-dynamics.md).
- The full paired reference shares existing CPU/CUDA scan and cell kernels;
  FP32 or FP64 input increments are evaluated in FP64. Current-state guards
  select both components. Tests cover uniform, stretched and legacy log-mean
  grids, finite changes, sub-FP64-ulp increments, and active pressure floors.
- Hydro now carries x1 pressure increments through reconstruction and HLLC;
  horizontal/gravity-free sweeps use static face gauges after FP64
  reconstruction. All directions retain background gradients. Vertical source
  coupling uses mixture-density increments and preserves floor residuals and
  unbalanced initial profiles. Total face pressure remains available to implicit
  and energy consumers. The existing gravity energy operator is reused.
- Anchor/owner-cell exchange carries paired pressures; remote seam flux payloads
  also carry the background with the corresponding source correction. Same-process
  four-block column tests pass for smooth and discontinuous tracers. No extra
  mutable background cache or restart schema was introduced.
- Built all 15 C++ executables in the foundation label from current source via
  `PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target test_balance_column.release test_coordinate.release test_eos.release test_fp32_storage.release test_gravity_work_cache_to.release test_hydro_ref_x1.release test_hydro_ref_x1_rf_wall.release test_implicit_face_work_jacobian.release test_pressure_scan_x1.release test_riemann.release test_split_reference.release test_stretched_x1_ref.release test_wb_ref_wall.release test_wb_wall_corner.release test_x1_seam_split.release`:
  passed. The final mixed-dtype reference test was rebuilt with
  `PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target test_split_reference.release`.
- Rebuilt Python bindings using
  `PYTHONPATH=/tmp/snapy-rebase-python SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=4 python -m pip install --no-deps --no-build-isolation --upgrade --target /tmp/snapy-rebase-python .`;
  refreshed the final library with
  `cp build/lib/libsnap_release.dylib /tmp/snapy-rebase-python/snapy/lib/libsnap_release.dylib`.
- `ctest --test-dir build --output-on-failure -L fp32-dynamics`:
  **4/4 passed**. Includes the wb_ref4 plus mass-covariance configuration.
- Final `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **39/39 passed**, 67.86 seconds. Paired reference tests: 6 CPU cases passed,
  6 CUDA cases skipped. The final run includes FP32 increments compared with
  correctly promoted FP64 oracle values (an earlier test harness comparison
  rejected mixed dtypes; fixed without changing numerical tolerances).
- Full dynamics comparisons: normalized differences no greater than
  **3.63e-14**, including conversion-order 4, stretched grids, VIC scheme 9,
  partial hydrostatic correction, heating and moist sedimentation. Deliberately
  unbalanced rest still accelerates. Balanced rest with FP32 storage rounding
  for 20 steps has maximum vertical momentum **1.27e-13**. Closed-domain mass
  and gravity-free total energy are conserved within **2e-13** relative.
  Same-dtype restart continuation is **bit-identical**.
- Cross-repo compatibility: tested against isolated **kintera 2.6.10**; minimum
  remains **2.6.9**. No kintera source/API changes. Legacy restarts without the
  species background cannot enable perturbation fluxes. No claim of true FP32
  compute, GPU performance, or production Jupiter acceptance is made here.
  CUDA source is implemented but uncompiled/unexecuted on this CPU-only host;
  MPS and immersed solids are explicitly outside the enabled scope.
- Next implementation milestone: **M4**, including reconstruction dtype policy,
  FP64 implicit/chemistry boundaries, and full moist GPU acceptance. M3 CPU
  implementation is no longer blocking that work.

#### Rebase onto main aced752 — 2026-11-13

- Rebased PR #342 onto main's consolidated hydrostatic density reference,
  reconstruction, VIC gravity work, moist EOS base and deck-option defaults.
  Kept the main implementation of each operator and layered paired reference
  evaluation onto it. The earlier commit IDs above describe historical runs.
- Replaced the obsolete wb_ref4/environment-switch test variant with an explicit
  `--x1-mass-covariance` fixture using the current dynamics deck key. Conversion
  order 4 retains main's unconditional face-density correction.
- Updated main's new wall-test member-pointer call for the extended reference
  signature. Freshly rebuilt the foundation C++ executables plus
  `test_pref_local_seam.release`, `test_options_defaults.release`,
  `test_eos_io_core.release` and `test_lu_failure.release`; rebuilt bindings with
  `PYTHONPATH=/tmp/snapy-rebase-python SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=2 python -m pip install --no-deps --no-build-isolation --upgrade --target /tmp/snapy-rebase-python .`.
- `ctest --test-dir build --output-on-failure -L fp32-foundations`: **31/31 passed**,
  56.76 seconds. Main removed duplicate variants, accounting for the smaller
  count. `ctest --test-dir build --output-on-failure -R 'test_(options_defaults|eos_io_core|lu_failure|pref_local_seam)\.release'`:
  **4/4 passed**, 3.02 seconds. Kintera remains 2.6.10; no cross-repo API changes.
  CUDA remains untested on this CPU host.
#### M4 started — 2026-11-12

- Branch `fp32-step3-compute`, based on M3 `9f18a2d` (PR #342).
- M4a: explicit precision policy, FP32 state/RK/flux/tendency buffers, FP64
  geometry/background/EOS masters, and matching kernel coefficient views.
- M4b: cancellation-safe FP32 reconstruction, FP64 implicit and chemistry
  boundaries, forcing/sedimentation/scalar/retry audits.
- M4c: dtype allowlist tests, rest/heating/moist comparisons, and a 1000-step
  reduced full-physics acceptance run. Record unsupported hardware separately.
- Initial audit finds snapy-side dtype assumptions; no necessary kintera source
  change identified yet. Existing FP64 chemistry APIs will be exercised first.

#### M4a checkpoint

- Added `dynamics/precision: float32`, requiring the M3 anomaly/HLLC path.
  FP32 state/RK buffers, reconstruction, HLLC scratch/fluxes and divergence are
  now active. Geometry, total primitive/EOS boundary values and static references
  stay FP64. Kernel coefficient/geometry views match compute dtype.
- First smoke tests passed with real FP32 storage, explicit and FP64-implicit
  dynamics, moist saturation and sedimentation. Both completed **1000 steps**
  with finite states, no redo errors, and mass drift about **1.7e-7** relative.
  This is an intermediate result, not final M4 acceptance.
- User requested FP32 vertical implicit correction. Implementing
  `integration/implicit-precision: auto|float32|float64`; auto follows M4 compute
  dtype. FP32 solves retain the existing pivot/nonfinite checks. Since the
  current solver rejects whole blocks, fallback retries the rejected block
  from its original RHS in FP64 and counts fallbacks explicitly.
- No kintera source change required by the smoke/1000-step saturation runs.

#### M4b/M4c implementation checkpoint

- FP32 vertical implicit assembly/solve is the default under the explicit FP32
  dynamics policy. `implicit-precision` supports auto/float32/float64; pivot and
  nonfinite checks remain active. Rejected FP32 blocks retry from the original
  RHS in FP64; `fp64_fallbacks` counts these retries. Disabling fallback retains
  normal rejection and exact rollback.
- Pressure gauges are removed before FP32 reconstruction; x1 gauges travel with
  the existing anchor relay. Total face pressure, EOS/geometry/reference masters,
  forcing increments and chemistry remain declared FP64 boundaries. No kintera
  source changes; tested with isolated kintera 2.6.10 (minimum 2.6.9 unchanged).
- Added dtype allowlist, FP64 agreement, stretched/outflow, rest, exact restart,
  rollback, implicit stress/fallback, split-column and reduced Jupiter tests.
  The Jupiter fixture uses repository H2O/NH3 kinetics, saturation, sedimentation,
  heating/cooling, diffusion and passive scalar transport. It is not the external
  production deck. CUDA and production GPU validation remain outstanding.
- Preliminary 1000-step explicit/FP32-VIC moist runs passed with mass drift
  1.673e-7; reduced Jupiter passed with drift 2.330e-7 and zero VIC fallbacks.
  Final fresh-build gates follow below.
- Parent PR #342 CI exposed a stale member-pointer call in
  `test_pref_local_seam.cpp`; pass the new optional reference arguments explicitly
  because default arguments do not apply through member pointers.

#### M4 final fresh-build validation

- Built all 15 foundation C++ targets using the exact M3 build command above,
  adding `test_lu_failure.release`; exit 0. Built the corrected seam executable:
  `PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target test_pref_local_seam.release`:
  passed. Refreshed the package library with
  `cp build/lib/libsnap_release.dylib /tmp/snapy-rebase-python/snapy/lib/libsnap_release.dylib`.
- Configured the final test registration with
  `PYTHONPATH=/tmp/snapy-rebase-python cmake -S . -B build`: passed. The new split
  test must be registered before the existing PYTHONPATH setup; its first run
  imported the old global snapy package. Corrected registration and reran.
- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **43/43 passed**, 75.87 seconds. The dtype allowlist, initial FP32 primitive
  promotion, stretched/outflow comparisons, rest, exact restart/rollback,
  FP32 VIC and same-process split-column checks all pass.
- `ctest --test-dir build --output-on-failure -R 'test_(fp32_split_python|lu_failure|pref_local_seam)'`:
  **4/4 passed**, 4.54 seconds. Parent PR #342 CI call-site fix pushed as
  `00335c5`. CUDA-gated cases remain skipped on this CPU-only host.
- FP32/FP64 VIC differences for dt=1,10,100,1000 are at most **4.089e-9**
  normalized; these runs use zero fallbacks. An ill-conditioned dt=1e9 fixture
  exercises successful FP64 recovery and exact rollback with fallback disabled.
- `pre-commit run`:
  passed. `git diff --check`: passed. Python unused-name checks passed via
  `ruff check --no-cache --select F401,F821 tests/test_fp32_compute.py tests/test_fp32_policy.py tests/test_fp32_implicit.py tests/test_fp32_jupiter.py`.

- `ctest --test-dir build --output-on-failure -L fp32-acceptance`:
  **2/2 passed**, 75.16 seconds. Both explicit and FP32-VIC moist cases completed
  1000 steps (relative mass drift **1.673e-7**); the reduced H2O/NH3 Jupiter
  case completed 1000 steps (drift **2.330e-7**). All used finite FP32 states,
  no retry exhaustion and **zero FP64 VIC fallbacks**.
- **M4a–M4c implementation and CPU acceptance are complete.** Configuration and
  numerical boundaries are documented in
  [fp32-compute-policy.md](docs/derivations/fp32-compute-policy.md).
  Default FP64 behavior remains covered by the foundation suite. CUDA kernels
  were not compiled/executed here; production decks and GPU performance remain
  unvalidated. M5 covers broader coupled workflow/restart integration; M6 retains
  production science/performance validation. No new kintera dependency or API.

#### M4 rebase validation — 2026-11-13

- Rebased PR #343 onto validated M3 `337b157`, based on main `aced752`.
  Preserved main's consolidated reconstruction, moist EOS and VIC operators,
  including corrected-PE wall coupling, while retaining FP32 views/workspaces
  and checked FP64 fallback. The stack contains M3 once, followed by M4.
- Freshly rebuilt the same C++ target set and Python bindings as the M3 rebase
  above. `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **35/35 passed**, 72.03 seconds. Main removed eight duplicate variants from
  the previous 43-test gate. Formatting and `git diff --check` pass.
- `ctest --test-dir build --output-on-failure -R 'test_(options_defaults|eos_io_core|lu_failure|pref_local_seam)\.release'`:
  **4/4 passed**, 2.91 seconds.
- `ctest --test-dir build --output-on-failure -L fp32-acceptance`:
  **2/2 passed**, 89.09 seconds. The FP32 VIC moist case has relative mass drift
  **1.674e-7** and reduced Jupiter **2.347e-7**, after 1000 steps; zero FP64
  implicit fallbacks. The stress test separately confirms fallback recovery.
- Cross-repo compatibility unchanged: kintera 2.6.10 tested, minimum 2.6.9,
  no kintera edits/API changes. CUDA and production GPU validation remain pending.

#### M5 started — 2026-11-16

- Branch `fp32-step4-restart`, from main `3fe398a`. M3 and M4 entered main via
  #360/#361; original #342/#343 were closed. Their Linux/macOS and bitwise CPU
  CI passed. No CUDA execution evidence was added by those CPU runs.
- M5a: additive version/dtype metadata and per-block background-grid identity;
  preserve exact stored anomalies and reject unknown/corrupt representations.
- M5b: destination-policy dtype selection for perturbation-mode restarts,
  legacy compatibility, changed-grid rejection, and FP64 outflow references.
  Audit found FP32-to-FP64 kept the saved FP32 dtype and FP32 outflow resume
  demoted the boundary reference; both are corrected in this stage.
- M5c: uninterrupted/resumed rest/heating, four-block bundles and reduced
  H2O/NH3 chemistry/scalar continuation with freshly constructed modules.
- No kintera source/API changes. External production/paddle integration and
  distributed/GPU validation remain separate, unverified work.
- Contract: [fp32-restart.md](docs/derivations/fp32-restart.md).

#### M5 CPU implementation and validation — 2026-11-16

- Version-1 anomaly metadata records saved storage precision and grid faces;
  unknown versions, inconsistent dtype/shape, missing backgrounds and changed
  grids are rejected. Pre-metadata files remain readable with their existing
  restrictions. Exact anomalies and FP64 backgrounds are retained.
- Destination precision selects perturbation-mode hydro/scalar storage; legacy
  storage-emulation mode keeps its original behavior. Characteristic outflow
  references follow FP64 primitive precision rather than FP32 anomaly precision.
- The four-block continuation test exposed a layout lifetime bug: destroying
  the old mesh could erase the replacement's process-local registration and
  stall exchange. Destruction now unregisters only the pointer it owns. The
  same replacement fixture passes with this fix.
- Fresh core build: `PYTHONPATH=/tmp/snapy-rebase-python cmake --build build --parallel 4 --target snap_release`:
  passed; rebuilt all 15 foundation C++ executables listed in M3's build command.
  Python rebuild: `PYTHONPATH=/tmp/snapy-rebase-python SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk MAX_JOBS=2 python -m pip install --no-deps --no-build-isolation --upgrade --target /tmp/snapy-rebase-python .`:
  passed. Refreshed the final library after the layout fix with
  `cp build/lib/libsnap_release.dylib /tmp/snapy-rebase-python/snapy/lib/libsnap_release.dylib`.
- `PYTHONPATH=/tmp/snapy-rebase-python python -u tests/test_fp32_restart.py`:
  passed. Same-dtype rest/heating/outflow, four-block FP32 bundles and H2O/NH3
  hydro/scalar/kinetics continuation in FP32 and FP64 are **bit-identical**.
  Both precision-switch directions pass; largest normalized error **1.456e-7**.
  Early fixture-only buffer-access/archive-alias errors were corrected; the
  stalled multi-block case was resolved by the layout fix above.
- `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **37/37 passed**, 72.40 seconds, including the new restart test (6.31 seconds).
- `ctest --test-dir build --output-on-failure -L fp32-acceptance`:
  **2/2 passed**, 78.57 seconds; 1000-step moist explicit/implicit and reduced
  Jupiter runs retain zero FP64 VIC fallbacks.
- `pre-commit run` and `git diff --check`: passed. Kintera **2.6.10** tested;
  no source/API change and minimum remains 2.6.9. Remaining M5 integration work:
  external paddle/production decks, actual distributed ranks, explicit remapping
  workflows and GPU execution. M6 science/performance work is not claimed here.

#### M5 merge confirmed — 2026-11-16

- PR #367 merged as `cbf8683009de2d4c211e65e5daeaf1ad1e05c80d`.
  Linux/macOS build-and-test, bitwise CPU and pre-commit CI all passed.
- The CPU restart implementation is integrated. Remaining validation is CUDA,
  production Jupiter/paddle workflows and actual distributed ranks, followed by
  M6 scientific budgets/statistics and GPU memory/throughput measurements.
- No new kintera changes are required by this merge. Changed-grid restart still
  requires explicit remapping and background reconstruction; it is rejected by
  the current loader.

#### M6 started — 2026-11-16

- Branch `fp32-step5-validation`, based on main `cbf8683`. User explicitly
  excluded regridding; changed-grid rejection remains the supported behavior.
- M6a: closed Cartesian moist mass/dry-mass/water-family/energy budgets,
  including the existing corrected potential energy and actual/requested heat
  increments. Pair FP32/FP64 trajectories over several deterministic seeds.
- M6b: late-window statistics and reduced full-physics Jupiter comparisons.
  Engineering tolerances are provisional reduced-case gates, not evidence of
  production climate equivalence or long-time statistical convergence.
- M6c: synchronized step timing, deduplicated registered-buffer/state storage,
  CUDA peak allocation when available, and optional operator profiler traces.
  No speedup assertion in CI. This host reports no available CUDA or MPS device.
- Restored missing pure-Python files and metadata in the isolated kintera
  install from the clean local 2.6.10 source; no kintera source modification.

#### M6 GPU validation without recompilation — 2026-11-16

- Used installed packages on dungeon3 GPU 0 (RTX 5090): Torch 2.10.0+cu128,
  snapy 2.13.27, kintera 2.6.10. Copied Python fixtures into an isolated
  `/tmp/snapy-m6-hsaZRT` directory; no checkout, library or package changes.
- Made existing column, Jupiter and restart Python fixtures device-aware.
  CPU defaults are retained. No native source or kintera API changes.
- `ctest --test-dir build --output-on-failure -L fp32-science`: **1/1 passed**,
  8.11 s. `ctest --test-dir build --output-on-failure -L fp32-foundations`:
  **37/37 passed**, 74.41 s, after the device-aware fixture changes.
- GPU 30-step column smoke and 10-step Jupiter pair passed, with zero VIC
  fallbacks. Exact commands, provenance, and remaining gates are in
  [the M6 validation report](docs/notes/fp32-m6-validation.md).
- GPU `test_fp32_restart.py` failed: installed native restart output lacks
  `hydro_restart_metadata`, required by merged M5. Honored the explicit
  no-recompilation instruction; this is an installed-library compatibility
  blocker, not a completed M5 GPU restart validation.
- Initial 1000-step CPU Jupiter paired-state error was **4.6658e-4**, exceeding
  the provisional **2e-4** gate. Standalone mass/fallback checks passed.
  M6 remains open; no tolerance was loosened. JSON now preserves measurements
  before a Jupiter comparison failure.
- Long runs finished on both CPU and GPU: all 18 column trajectories per host
  passed 1000-step budgets and paired-state gates. GPU worst water-family
  residual **3.152e-5**; zero VIC fallbacks. Jupiter state gate failed on both
  CPU (**4.666e-4**) and GPU (**4.669e-4**) against **2e-4**; family/scalar
  comparisons passed. Raw JSON and exact commands are retained in the report.
- GPU `test_fp32_implicit.py` and `test_fp32_policy.py`: **passed** using installed
  libraries. VIC registered storage fell **45.6%**, CUDA peak allocation **11.5%**.
  Small-grid GPU median step time rose from **17.175 ms** to **21.717 ms**;
  no performance gain is claimed. Production-size profiling remains open.
- Final science CTest after JSON failure-reporting edits: **1/1 passed**, 8.38 s.
  Compileall, Ruff F401/F821 and `git diff --check`: passed.

#### M6 Straka paired GPUs — 2026-11-16

- User requested FP64 on one GPU and FP32 on the other. Ran dungeon3 GPU 0/1
  simultaneously with installed binaries; no recompilation. Added reproducible
  run/comparison scripts and saved JSON plus a field-comparison plot.
- Both 64×256 explicit Straka runs completed **9000 steps / 900 s**, dt=0.1,
  without rejected/nonfinite steps. Used matching HLLC/ideal-moist dry-limit
  configurations required by the FP32 path, not the shipped LMARS regression.
  No explicit viscosity; the shipped problem/K is unused by straka.cpp.
- At 900 s: theta RMS difference **0.001764 K**, maximum **0.03645 K**, relative
  L2 **0.1344%**. Both fronts at **15.35 km** (100 m resolution). FP32 relative
  mass drift **1.09e-9**. No post-hoc equivalence threshold imposed.
- FP64/FP32 wall times **63.63/70.30 s**, including diagnostics; FP32 saved
  **39.9%** registered storage and **25.0%** CUDA peak allocation. This single
  simultaneous pair does not demonstrate a speedup or isolated timing variance.
- Exact commands, results and scope are in the Straka section of the
  [M6 report](docs/notes/fp32-m6-validation.md). Both GPU run commands and the
  comparison passed. Compileall, Ruff F401/F821 and `git diff --check` passed.
  Kintera remains unchanged. Jupiter/M5 restart/production gates remain open.

#### M6 Straka optimization work — 2026-11-16

- User authorized prioritizing face-state conversions and forcing allocations.
  Dungeon3 binaries remain unchanged; new native code is validated with a local
  isolated build before any optimized GPU performance claim.
- Added `--profile` to the Straka runner: profiles five extra steps after saving
  final fields and capturing elapsed time/peak allocation, so the profile does
  not alter the reported trajectory or timed interval.
- Sequential baseline profiles on dungeon3 GPU 0, each after 10 simulated
  seconds: FP64/FP32 `_to_copy` calls **105/805**, `copy_` calls **1925/2665**,
  `empty_strided` calls **1005/1730**, CUDA launches **8665/9765** over five steps.
- Horizontal FP32 reconstruction now promotes only the two pressure rows for
  gauge restoration/flooring/background subtraction. Density/velocity/species
  faces remain FP32; covariance consumers retain the existing total-state path.
  The gravity-balanced vertical sweep already reconstructed FP32 faces.
- Reuse FP64 forcing accumulation and FP32 rounded-increment scratch buffers.
  Reset contents each stage; invalidate on module reset, shape/device changes.
  Preserve accumulation order and rounding before addition to the hydro update.
  Scratch is not restart state; registered-buffer-only memory reports exclude it.
- Added a zero-duration-stage regression after nonzero gravity forcing to detect
  stale forcing increments. Preparing before/after local Straka comparison and
  foundation/acceptance validation. No kintera source/API changes.
- Native build passed after restoring missing matching headers in the temporary
  kintera installation. Created `/tmp/snapy-straka-opt-python` for optimized tests;
  retained `/tmp/snapy-rebase-python` as baseline. Rebuilt the 15 foundation C++
  executables for the changed Hydro layout.
- Before/after FP32 CPU Straka at 0/300/600/900 s: **bit-identical** temperature,
  primitive, total conserved and tracer arrays. Five-step CPU profiles reduce
  `_to_copy` 2270→2255, `copy_` 4265→4250, `empty_strided` 3195→3165,
  `clone` 330→315 and `empty` 1260→1245. No controlled timing gain claimed.
- Reviewed disabled-flux behavior and added a guard plus a no-pressure-source
  regression. The long run predates this guard; final guarded binary is checked
  separately. Reused scratch persists between stages and is excluded from the
  registered-only memory metric; no new GPU memory-saving claim is made.
- Final guarded build validation:
  `ctest --test-dir build --output-on-failure -L 'fp32-(foundations|science|acceptance)'`:
  **40/40 passed**, 163.30 s (37 foundations, 2 acceptance, 1 science), including
  restart, multi-block, VIC, moist 1000-step and standalone Jupiter tests.
  The separate long paired Jupiter science discrepancy remains open.
- Final guarded Straka command:
  `PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py --precision float32 --device cpu --tlim 30 --output /tmp/straka-final-30`:
  passed; all saved fields bit-identical to the original binary's 30-second run.
- Google clang-format verification, Python compileall/Ruff F401/F821 and
  `git diff --check` passed. Native optimization is implemented and locally
  validated; optimized CUDA measurements remain pending the user's no-remote-
  rebuild constraint. Dungeon3 installation and kintera sources are unchanged.

#### GPU repeats and Kratos-based diagnosis — 2026-11-16

- Completed the exact three-pair GPU-only Straka loop recorded in the M6 report:
  **6/6 passed**, 9000 steps each. Same GPU, sequential, reversed order in pair 2.
  Medians FP64 **63.4482 s**, FP32 **70.2006 s**: **10.6423% slower** FP32.
  Same-precision final arrays are bit-identical across all three repeats.
  Installed binary hashes still predate the native optimizations.
- Read Wang (2025), DOI 10.3847/1538-4365/adbdb6, using indexed publisher text
  and arXiv v2 when the PDF was unavailable. Recorded source-linked findings and
  revised priorities in `docs/notes/fp32-kratos-diagnosis.md`.
- Verified unintended FP64 arithmetic inside the installed float HLLC CUDA
  specialization using cuobjdump (no compilation). `.5`/`1.0` plus `auto`
  promote intermediates. Static float N=8 kernel contains 101 DFMA/50 DMUL and
  59 F32→F64 conversions; static counts are not runtime instruction counts.
- Kernel-only trace analysis: HLLC ~1.4% of FP32 summed GPU kernel time, so fixing
  it alone cannot explain/remove the observed slowdown. WENO speeds up, but
  launch counts rise 8665→9765 and casts/copies consume the saving. Do not claim
  the entire hydrodynamics pipeline is pure FP32 merely from buffer dtypes.
- Prioritize typed, cancellation-safe scalar HLLC arithmetic and fewer pipeline
  intermediates/launches. Consider an explicit FP64 accumulation/FP32 compute
  experiment while preserving the hydrostatic anomaly design and current mode.
  This diagnosis changes no native code or kintera API. Optimized GPU build
  permission remains pending; all new GPU timings use installed libraries.

#### HLLC fix and 2× Straka target — 2026-11-16

- User requested HLLC first and a plan for 2× speed. Recorded staged work and
  measurement contract in `docs/notes/fp32-straka-speed-plan.md`.
- Target: same-GPU FP32 median ≤31.7241 s for the existing 9000-step Straka case
  (2× FP64 baseline, 2.21× old FP32), preserving physics/numerics/diagnostics.
  HLLC alone cannot meet it; subsequent work targets fusion and launch overhead.
- HLLC now uses typed constants with compile-time intermediate-type assertions.
  Float acoustic differences avoid subtracting close velocities, including the
  correct branches for clipped waves. Double arithmetic order is retained.
  Added shock/expansion/boost/species/pressure-background comparisons in the
  Riemann suite. Local build/validation in progress; no GPU speed claim yet.
- HLLC build command:
  `PYTHONPATH=/tmp/snapy-straka-opt-python cmake --build build --parallel 4 --target snap_release test_riemann.release`: passed.
- `ctest --test-dir build --output-on-failure -L 'fp32-(foundations|science|acceptance)'`:
  **40/40 passed**, 158.18 s, including the new 120-case Riemann sweep.
- `PYTHONPATH=/tmp/snapy-straka-opt-python python -u tests/run_fp32_straka.py --precision float32 --device cpu --output /tmp/straka-hllc-fix-900`:
  completed 9000 steps without rejection/nonfinite state; relative mass drift
  **1.1391e-9**. Theta change versus old CPU FP32 at 900 s: **0.0005673 K RMS**,
  **0.01535 K maximum**. CPU run used for correctness only, not GPU speed claims.
- Google clang-format and `git diff --check`: passed. Kintera API/source unchanged.
  H1 locally validated; GPU codegen/timing pending isolated build. H2/H3 fusion
  and dispatch work are planned to pursue the 2× target, not yet implemented.

#### H1 GPU validation — 2026-11-16 (complete)

- User requested GPU FP64/FP32 testing before further fusion work. H2 is paused.
- Testing source commit `2f321dc` in `/tmp/snapy-hllc-2f321dc` on dungeon3;
  installed snapy and the existing checkout remain untouched. Rebuilding the
  native libraries and Python bindings together avoids mixing incompatible
  Hydro layouts. Kintera remains installed version 2.6.10, unchanged.
- Isolated build uses Release, CUDA 13.1 and sm_120 only (a scratch CMake
  architecture override), with the existing dependency sources reused read-only.
  Planned checks: GPU Riemann tests, short paired Straka profiles, then three
  sequential alternating FP64/FP32 9000-step pairs on GPU 0 and SASS inspection.

- Native and matching Python bindings built successfully. Loaded-library paths
  and hashes verify the isolated build; installed library hashes are unchanged.
- `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_riemann.release$'`:
  **1/1 executable, 11/11 GoogleTests passed**, including CUDA float/double dispatch.
- `bash /tmp/snapy-hllc-2f321dc/snapy-hllc-benchmark.sh`: **6/6 full GPU runs
  passed**, 9000 steps each, no rejected/nonfinite state. Same-precision saved
  fields are bit-identical across all repeats. Both short GPU profiles passed.
- Median FP64 **68.9998 s**, FP32 **76.2602 s**: FP32 **10.522% slower**. Ranges
  68.3417–69.0580 / 72.6487–76.2797 s. Host background CPU load was substantial;
  historical absolute wall times are not a controlled before/after comparison.
- HLLC short-profile GPU time: FP64 **0.209247 ms**, FP32 **0.079583 ms**, over
  30 calls: **2.63× faster for HLLC itself**. SASS verification passed: both float
  specializations have no FP64 arithmetic/conversions. Full FP32 still has
  **9780 launches per five steps**, versus FP64 8665. The end-to-end 2× goal is
  not met; H2/H3 remain necessary and no new fusion was implemented in this step.
- At 900 s: theta error **0.00165068 K RMS / 0.0338796 K max**, identical 15.35 km
  front location, FP32 relative mass drift **6.46765e-10**. CUDA peak FP64/FP32:
  **15,606,784 / 12,379,648 bytes**; include retained scratch in memory accounting.
- Reproducible commands, source/native hashes, full trial reports, profiler/SASS
  evidence and plot: [HLLC GPU validation](docs/notes/fp32-hllc-gpu-validation.md).
  Evidence Python compileall, Ruff F401/F821 and `git diff --check`: passed.
  Kintera 2.6.10 was used unchanged; other dependency combinations were not tested.

#### H2 constant cache and capture-safe capacity validation — 2026-11-16

- Full-step CUDA capture probe warmed up successfully, then failed in
  `IdealMoistImpl::perturbations`: C++ stack traced `torch::tensor` through
  `_to_copy`/`copy_kernel_cuda`. Per-call CPU constant uploads are the first
  confirmed blocker; the synchronous capacity check is another known blocker.
- Cache FP64 cv, energy-offset and gas coefficients privately, keyed by device,
  original double-valued host options and gas/cloud counts. Reset clears the
  cache; module FP32 conversion cannot round these coefficients. No restart
  format or kintera API change.
- During hydro stages, latch invalid background/total heat capacity on-device,
  guard invalid denominators, and include the latch in the existing batched
  check_redo readback. Invalid steps are rejected even if a later stage becomes
  valid. Stage zero clears the latch; direct EOS calls still throw immediately.
- Added regression checks for cached repeat equality, host-option refresh,
  immediate invalid-capacity exceptions, deferred failure persistence, and
  routing the failure to the existing retry decision. Validation underway.
- The separate dry HLLC fusion experiment failed parity tests and was removed
  from this change; its patch is preserved at `/tmp/snapy-dry-hllc-wip.patch`.
  Initial cache/capture probes disabled fusion explicitly. Full warmed-up
  Straka capture now passes; safe repeated replay remains a separate milestone.

- Final cache/check-only native build and bindings passed. On dungeon3:
  `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_(eos|riemann).release$'`:
  **2/2 passed**, including CPU/CUDA float/double regression coverage.
- `bash /tmp/snapy-hllc-2f321dc/snapy-cache-validate.sh`: passed. Four Python
  compute/restart/perturbation fixtures, the warmed-up CUDA capture probe, both
  short GPU profiles and **2/2 full 9000-step GPU Straka runs** passed. All saved
  arrays at 0/300/600/900 s are **bit-identical** to the pre-cache results.
- Five-step profiles: `aten::item` **15→0**, stream synchronizations **65→5** in
  both precisions. FP32 `_to_copy` **790→750**, launches **9780→9775**. Single-run
  times FP64/FP32 **62.9569 / 69.5430 s**; no standalone speedup claim against
  historical runs with different host load. The 2× target remains open.
- [Cache/check report and evidence](docs/notes/fp32-cache-validation.md).
  Kintera 2.6.10 unchanged. Native libraries/bindings require a matching rebuild;
  restart formats are unchanged. Production graph replay still needs storage
  and host-bookkeeping validation despite successful capture.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package /home/chengcli/pyenv/bin/python tests/test_fp32_science.py --device cuda --steps 30 --output /tmp/snapy-hllc-2f321dc/results/cache-science.json`:
  passed (exit 0), closed/heated/sedimenting cases with explicit/implicit schemes.
  Google clang-format, evidence compileall/Ruff F401/F821 and `git diff --check`
  passed. Cache/check milestone complete; 2× graph-replay milestone remains open.

#### Fresh post-cache FP64/FP32 GPU repeats — 2026-11-16 (complete)

- User requested fresh FP64/FP32 testing. Reusing the validated isolated native
  build on dungeon3 (hashes match the cache/check report), without recompiling.
- Three sequential alternating pairs on RTX 5090 GPU 0, second pair reversed;
  full 9000-step Straka, identical diagnostics and timestep checks. Eager
  execution; production CUDA graph replay is not enabled.
- Command: `bash /tmp/snapy-hllc-2f321dc/snapy-cache-repeat.sh`.
  New outputs live in `results/cache-repeat/`, preserving all earlier results.

- Benchmark and summary commands passed: **6/6 runs**, 9000 steps each. Median
  FP64 **62.8547 s**, FP32 **69.8552 s**: FP32 **11.138% slower**. Ranges
  62.7083–63.0953 / 69.4359–70.0161 s; sample stddev 0.1954 / 0.2995 s.
- All saved fields are bit-identical across same-precision repeats and against
  pre-cache same-precision results. Final theta error **0.00165068 K RMS**,
  maximum **0.0338796 K**, FP32 relative mass drift **6.46765e-10**.
- [Detailed results and reproduction](docs/notes/fp32-cache-validation.md).
  No native changes or kintera changes. The eager 2× target remains unmet;
  production graph replay remains the next milestone. Evidence compileall,
  Ruff F401/F821 and `git diff --check` passed.

#### H3 CUDA graph replay — 2026-11-16 (Straka benchmark validated)

- Added opt-in `--execution cuda-graph` to the existing Straka benchmark,
  preserving eager as the default. No native kernel or kintera changes.
- The fixture-specific executor keeps detached input storage alive across native
  RK `set_`/`set_data`, copies current outputs/caller edits into those inputs
  before replay, and recaptures on timestep or state-storage changes.
- Warmup uses zero-duration stages and restores all state, without accepting
  synthetic steps or resetting the integrator's retry counters. Capture setup
  time, capture count and replay count are reported separately. Recaptures during
  the simulation remain inside its timed interval.
- Restrict to the dry Cartesian explicit RK3 Straka fixture with limiter off;
  moist chemistry, implicit solves, host limiter queues and distributed exchange
  are not supported. Configuration/reference changes require a new executor.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package /home/chengcli/pyenv/bin/python tests/test_straka_cuda_graph.py`:
  passed for FP64 and FP32: 100 bit-identical eager/replay steps, changed dt,
  replaced storage, in-place perturbation, and native rollback after injected
  NaNs. Added CUDA-only CTest registration.
- Running three full eager-FP64/graph-FP64/graph-FP32 trials, reversed order in
  trial 2, plus one fresh eager-FP32 run. CFL, retry and finite checks and snapshot
  schedules are unchanged. Target remains 2× eager FP64 on the fixed grid.

- `bash /tmp/snapy-hllc-2f321dc/snapy-replay-benchmark.sh`: **10/10 full runs passed**.
  Three trials each: eager FP64 median **66.5690 s**, graph FP64 **29.4328 s**,
  graph FP32 **31.1725 s**; one eager FP32 run **72.2040 s**.
- FP32 graph versus eager FP64: **2.136×**; adding all graph setup conservatively
  gives **31.4056 s / 2.120×**. All three matched pairs exceed 2×. Median meets
  the original 31.7241 s target; slowest trial does not meet that historical
  absolute threshold. Graph FP64 remains faster: this gain is from replay.
- Summary script passed: all saved arrays at 0/300/600/900 s are bit-identical
  to same-precision eager references across all repeats. Each graph run executes
  9000 replays and four captures; in-run recapture stays in the timing.
- `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_straka_cuda_graph_python$'`:
  **1/1 passed**, 3.76 s, including final retry-budget exhaustion regression.
- Peak graph FP64/FP32 allocation **19,572,224 / 14,952,448 bytes**. Graph storage
  increases memory versus eager. Python compileall/Ruff and `git diff --check`
  passed. [Report and artifacts](docs/notes/fp32-straka-graph-replay.md).
- No native or kintera changes in this step. The fixed Straka throughput target
  versus eager FP64 is met; this optional fixture path does not enable general
  moist, VIC, distributed or limiter-enabled graph execution. Broader M6 gates
  remain separate.

#### FP32 graph slowdown diagnosis — 2026-11-16 (complete)

- Profiled three alternating graph FP64/FP32 pairs and one eager run per
  precision on dungeon3 GPU 0, reusing the validated build. Each run includes
  100 checked steps and five profiled steps with CFL/retry/finite checks.
- Median summed GPU kernel time over five steps: FP64 **14.081 ms**, FP32
  **15.009 ms** (+6.59%). FP32 has **196 extra kernels per step**, 195 inside
  the graph. Outside-graph GPU work differs by only 0.010 ms per five steps.
- Dtype-copy overhead rises **1.320 ms**, net elementwise arithmetic rises
  **0.638 ms**, outweighing WENO/HLLC savings of **0.876 ms** per five steps.
- C++ dispatch backtraces identify **48 uniform reconstruction coefficient
  casts and 30 divergence geometry casts per FP32 timestep**. Cache these
  first; then inspect full-field hydro/EOS round trips. Caller counts are not
  per-caller timings; A/B validation is still needed to establish savings.
- `bash /tmp/snapy-hllc-2f321dc/profile-straka-checked.sh`: **8/8 passed**.
  `/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/analyze-straka-graph.py`:
  passed, six traces, five graph launches each.
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/snapy-hllc-2f321dc/package /home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/straka-op-origin.py`:
  both precision probes passed.
- [Diagnosis and ranked next steps](docs/notes/fp32-straka-graph-profile.md).
  No solver/native/kintera changes or new cross-repo compatibility risks.
  This identifies candidates; no additional speedup is claimed.

#### Dtype-conversion cache implementation — 2026-11-16 (complete)

- Added a shared derived-tensor cast cache for uniform/nonuniform WENO5/Center5
  coefficients and divergence face areas/volume. Source edits, storage/metadata
  replacement and target device/dtype changes invalidate the cached conversion.
  Inference tensors and autograd inputs bypass caching; no restart buffers added.
- Added CPU/CUDA regression coverage for mutation/storage/device/dtype changes,
  coefficient results and divergence after grid edits. Isolated dungeon3 native
  library and matching Python bindings rebuilt successfully; kintera unchanged.
- `CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_(weno|coordinate|x1_coeffs|straka_cuda_graph_python)(.release)?$'`:
  **4/4 passed**, 10.50 s.
- `bash /tmp/snapy-hllc-2f321dc/snapy-cast-cache-benchmark.sh` and
  `/home/chengcli/pyenv/bin/python /tmp/snapy-hllc-2f321dc/snapy-cast-cache-summary.py`:
  **12/12 full runs passed**, all saved fields bit-identical within each
  precision. Preserved previous package supplies the fresh before/after control.
- FP32 captured dtype-copy kernels **177→99 per timestep**; GPU copy time
  **1.39236→0.89626 ms per five steps (−35.6%)**. Total captured GPU time
  **13.90260→13.40345 ms (−3.59%)**. FP64 GPU time/count nearly unchanged.
- Full FP32 median **31.89151→30.05118 s (1.061×)**; all three paired speedups
  positive. FP64 wall-time variability prevents a precision-parity claim:
  FP32 still costs **4.15% more captured GPU time** than FP64.
- Retained cache memory increases FP32 peak allocation by **227,328 bytes**.
  Local formatting, diagnostic-script syntax and `git diff --check` passed.
  [Detailed results and evidence](docs/notes/fp32-straka-cast-cache.md).
  No kintera/restart changes; native ABI consumers/bindings require rebuilding.

## Record for each implementation milestone

Append: affected repositories and commit/version set; concrete changes; exact
build/test/benchmark commands and results (including skips); capture/replay and
failure-path evidence; same-precision and FP32/FP64 science comparisons; setup,
runtime and peak-memory measurements; compatibility/restart implications;
remaining blockers; and the next milestone. Update the table only when the
stated exit gate passes. Proposed APIs/test names are not implemented features.
