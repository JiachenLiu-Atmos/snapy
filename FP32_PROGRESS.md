# FP32 project progress

Updated: 2026-11-12. Roadmap: [FP32_ROADMAP.md](FP32_ROADMAP.md).

## Milestones

| Milestone | Status | Evidence / next action |
|---|---|---|
| M0 Baseline and records | Merged baseline | PR #326 merged as 2d92c92; M3a validation below |
| M1 Reference consolidation | Merged (#326) | Shared operator, cache invalidation, composition restart; CPU and reviewer CUDA evidence below |
| M2 Thermodynamic perturbations | Merged (#326) | Delta thermodynamics, exact total reads and FP32 floors; hydro wiring follows |
| M3 Perturbation dynamics | M3a–M3c implemented, CPU validated | 39/39 foundation tests pass; split reference, flux/source coupling, seams and restart; CUDA execution pending |
| M4 True FP32 | Planned | Follow perturbation kernels |
| M5 Coupled workflow/restart | Planned | Validate actual kintera build |
| M6 Science/performance | Planned | Requires production GPU and coupled cases |

Historical foundation baseline (2026-11-08): snapy work was recorded on
`fp32-step1-anomaly`, based on `fe003dd`.
The paired kintera change is commit `80ac117` on `fp32-perturbation-thermo`, based
on user-rebased `v2.6.8` / `e646fe5`. Fresh CPU validation uses the paired source
builds, not the globally installed packages.

## Validation log

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

## Decisions — 2026-11-08

- Jupiter Cartesian ideal-moist first; perturbation kernels before true FP32.
- Initial balancing remains opt-in; preserve unbalanced-state residual forces.
- Reuse well-balanced machinery; distinguish static and dynamic references.
- Kintera source edits authorized for moist physics. Keep repository responsibilities.
- Record exact commands/results and cross-repo compatibility at each checkpoint.

## Environment and outstanding work

Local host is macOS/CPU; CUDA and production GPU validation require another host.
The initial 2026-11-08 validation pinned the isolated kintera v2.6.8 source
build plus its paired change. Those historical Python tests used `/tmp/snapy-fp32-python` (kintera 2.6.8+fp32
and the rebuilt snapy extension/library). Global Python packages were not replaced.
Coupled Jupiter cases live outside this repository and were not found in the
local workspace search. The temporary build/install paths are local validation
artifacts; a clean host must build/install the paired sources itself.

## Implementation checkpoint — 2026-11-08

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

## Thermodynamics checkpoint — 2026-11-08

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

## Gravity-work checkpoint — 2026-11-08

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

## CPU validation checkpoint — 2026-11-08

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

## Remaining implementation (not completed by the foundation work)

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

## Historical storage-only results

These are reported by the original roadmap, **not rerun in this work**. Errors
are maximum velocity differences from its FP64 total-storage baseline.

| Reduced case | FP32 total storage | FP32 anomaly storage |
|---|---:|---:|
| Rest, 200 steps | 2.6e-5 m/s | 3.1e-7 m/s |
| Bottom heating, 200 steps | 2.6e-5 m/s | 3.9e-7 m/s |
| Full physics, 500 steps | 6.8e-4 m/s | 1.1e-5 m/s |

These storage-emulation measurements do not demonstrate true FP32 arithmetic.

## Reproducing the current foundation gate

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

## Final validated checkpoint — 2026-11-08

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

## PR #326 rebase and CI repair — 2026-11-09

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
  `pre-commit run --files FP32_PROGRESS.md FP32_ROADMAP.md .github/workflows/ci.yml .github/workflows/release.yml CMakeLists.txt cmake/modules/FindKintera.cmake pyproject.toml src/hydro/hydro_forward.cpp tests/CMakeLists.txt tests/test_tracer_ratio.py`:
  **passed**.
- Local validation is macOS CPU only. CUDA and production FP32 acceptance
  remain outstanding; milestone scope is unchanged.

### Linux CI follow-up

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

## PR326_REVIEWS.md follow-up — 2026-11-09

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
  `pre-commit run --files README.md FP32_PROGRESS.md FP32_ROADMAP.md src/eos/equation_of_state.cpp src/eos/ideal_moist.cpp src/mesh/meshblock.cpp tests/test_eos.cpp tests/test_fix_vapor_deficit.cpp tests/test_anomaly_storage.py tests/CMakeLists.txt`:
  passed.

## Rebase onto main 919936d — 2026-11-12

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
- `pre-commit run --files src/hydro/hydro.hpp src/hydro/hydro_options.cpp src/hydro/hydro_forward.cpp tests/test_anomaly_storage.py FP32_PROGRESS.md`,
  `python -m py_compile tests/test_anomaly_storage.py`, and `git diff --check`:
  passed.

## Next milestone: M3a signed pressure scan — 2026-11-12

- Start `fp32-step2-pressure-scan` from merged main `2d92c92` (PR #326).
- Implement the first of three M3 stages listed in FP32_ROADMAP.md. The new
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


### M3a implementation and validation

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
- `pre-commit run --files src/hydro/hydro_reference.hpp src/hydro/hydro_ref_x1_impl.h src/hydro/hydro_dispatch.hpp src/hydro/hydro_dispatch.cpp src/hydro/hydro_dispatch.cu tests/test_pressure_scan_x1.cpp tests/CMakeLists.txt FP32_ROADMAP.md FP32_PROGRESS.md`
  and `git diff --check`: passed.
- M3a implementation is ready for review with CPU evidence. GPU execution and
  actual MPS validation remain outstanding. M3b and M3c are not implemented;
  the new scan is an internal building block, not an enabled hydro mode.

### M3b/M3c implementation in progress

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


### M3b/M3c completion — 2026-11-12

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

### Rebase onto main aced752 — 2026-11-13

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
