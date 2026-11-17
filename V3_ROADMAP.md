# V3 roadmap: FP32 and CUDA graph execution

Updated: 2026-11-16. Status and validation evidence:
[V3_PROGRESS.md](V3_PROGRESS.md).

This roadmap combines FP32 numerical foundations/science (M0–M6) with general
CUDA graph execution (G0–G7) across snapy, kintera, pyharp and UCX/commux.
The current execution plan and cross-repository contracts come first; the
[FP32 foundations](#fp32-foundations-and-science-roadmap) preserve the numerical
requirements and earlier implementation stages. Dated implementation claims in
that section describe earlier checkpoints; the progress tracker records current
status. Planning does not imply that a configuration is implemented or validated.

## Objective and scope

Provide opt-in CUDA graph execution for FP32 and FP64 dry/moist dynamics,
explicit and vertical implicit integration, Cartesian and cubed-sphere grids,
and multiple local blocks and distributed GPU processes. Keep **UCX** as the
communication backend. Start with one GPU per process and allow several blocks
on that GPU. Regridding is out of scope.

Support means correct repeated execution, diagnostics, rejection/retry,
restart, and communication across the declared configuration matrix. A timestep
may consist of several captured GPU segments separated by host-controlled UCX
communication or solver fallback. One monolithic graph is not a requirement.
Eager execution remains available as the reference and default during rollout.

The first physics scope is ideal-moist/HLLC, existing RK schemes, H2O/NH3 phase
equilibrium, heating, sedimentation, positivity protection and the current VIC
schemes. Existing FP64 cancellation-sensitive arithmetic remains FP64 unless a
separate numerical change is validated. Additional EOSs, arbitrary Python
callbacks and temperature-dependent thermodynamics require explicit capability
entries; do not silently claim that every physical option is supported.

## Starting point and prerequisites

| Area | Current evidence or gap |
|---|---|
| Dry single-block Cartesian | Optional Straka graph runner validated in FP32/FP64; not a general Mesh execution API |
| Constants and checks | EOS constants cached; heat-capacity errors deferred to device flags; coefficient/geometry casts cached |
| State storage | RK returns fresh tensors; snapy rebinds state with `set_`; the Straka wrapper retains aliases and copies inputs before replay |
| Moisture | Kintera equilibrium iterations already run inside CUDA kernels; temperature inversion, table setup and failure-counter lifecycle still need work |
| VIC | GPU solve/workspace exists; host status reads and conditional FP64 fallback prevent full capture |
| Local blocks | Worker threads, host rendezvous and CUDA events coordinate existing exchanges |
| UCX | Cached exchange buffers exist; request posting, waits and stream synchronization remain host-managed |
| Cubed sphere | Internal face-state exchanges, metric/frame transformations, corner treatment and global corrections add graph boundaries |
| FP32 cubed sphere | **A numerical prerequisite, not just capture work:** anomaly storage and FP32 VIC currently reject non-Cartesian geometry; full axial-momentum mode also restricts FP32/perturbation flux |

Relevant sources: snapy's `tests/straka_cuda_graph.py`, `src/mesh/meshblock.cpp`,
`src/mesh/mesh.cpp`, `src/layout/layout.cpp`, `src/layout/process_group*.cpp`,
`src/hydro/hydro.cpp`, `src/hydro/hydro_forward.cpp`, `src/implicit/implicit_hydro.cpp`
and `src/eos/equation_of_state.hpp`; kintera's `src/thermo/thermo_y.cpp`,
`src/thermo/thermo_dispatch.cu`, `src/thermo/log_svp.cpp`, `src/utils/user_funcs.cu`
and `src/loops.cuh`; pyharp's `src/integrator/integrator.cpp`.

The source inspection used kintera v2.6.10. Audit actual deployed revisions of
all four projects before implementation; commux internals still need inspection.
Previous measured speedups apply only to the recorded Straka configurations.

## Repository ownership and interface contracts

| Repository | Owns | Proposed deliverable |
|---|---|---|
| snapy | Execution orchestration, state, physics ordering, VIC, halo packing/unpacking, global retry and capability checks | Segmented graph executor integrated with Mesh/MeshBlock; lifecycle/invalidation; full validation matrix |
| kintera | Temperature inversion, phase equilibrium, thermodynamic tables, reaction active sets and thermo failures | Capture-safe thermodynamic entry points with persistent device tables/workspace and failure counters |
| pyharp | RK coefficients and stage combination | Additive output-buffer RK API, with documented input/output alias rules and preserved mixed-precision arithmetic |
| commux, if needed | UCX requests, progress and CUDA stream/buffer ownership | Document existing completion semantics; only add event/completion APIs if required for safe overlap or removal of excess synchronization |

Do not duplicate RK mathematics in snapy or move thermodynamic kernels out of
kintera. Keep pyharp's existing allocating API working. Kintera should expose
device failure access/reset/consume operations so snapy no longer manipulates
the lifetime of `nfail_` directly. Readback remains an explicit host operation.
Persistent tables must be keyed by device and relevant species/reaction options
and must live as long as every graph referencing them.

Every integrated test records provider commits/versions, native library hashes,
Torch/CUDA/compiler versions, GPU model, and UCX/commux capabilities. Merge
provider interfaces before dependent snapy changes, or test exact paired
revisions until then. Rebuild affected native libraries and bindings together.
Derived caches and graph executables are never serialized into restart files.

## Execution design

The host advances an ordered schedule of GPU segments and external operations:

```text
global CFL and timestep selection
  for each RK stage:
    replay compute/pack segment(s)
    establish pack completion -> post/progress/complete UCX exchange
    establish receive completion -> replay unpack/compute segment(s)
    perform any additional within-stage exchange or solver fallback boundary
    apply required local/global corrections in their original order
global failure decision -> accept all blocks, or restore/retry all blocks
```

Segment boundaries follow real dependencies. Cubed-sphere face-state exchange,
fourth-order conversion/reference exchange, and configured global correction
reductions may occur inside a stage. Do not assume `advance_local()` is always
communication-free. Preserve each exchange's variables, peers, tags, phase and
corner/rotation rules. Global CFL/retry decisions and output stay outside graphs.

Use the existing cached halo buffers. Maintain these ordering guarantees:

- GPU packing completes before UCX reads a send buffer.
- UCX completion makes receive data safe for the consuming GPU stream.
- A send buffer is not reused until its request completes; a receive buffer is
  not overwritten while the GPU still consumes it.
- All ranks execute compatible communication order and a common accepted dt;
  a failure on any rank triggers the existing collective rollback decision.

Start with the existing conservative waits. Introduce overlap only after
validation: run work independent of incoming halos while UCX progresses, then
join before boundary-dependent work. Retain UCX; no backend replacement is part
of this plan. Direct capture of UCX requests is not assumed or required.

Each graph records its device, shapes/strides/dtypes, referenced storage,
physics/configuration generation, RK stage, precision mode and scalar parameters
such as dt. Validate all relevant graph-owned state/workspace/halo addresses,
not just public state tensors. Ordinary state evolution is not invalidation.
Buffer replacement, shape/device/dtype changes, restart, reference/geometry or
reaction-option changes invalidate affected graphs and derived caches.

Initially recapture for changed dt; coordinate distributed schedules and ensure
outstanding work completes before destroying referenced buffers. Parameterizing
dt on-device is a later optimization, not a reason to quantize or alter the
timestep schedule. Warmup/capture must restore state, active-set history,
failure flags, meters and retry bookkeeping. Zero-dt Straka warmup is not a
general VIC/moist warmup strategy.

## Milestones and dependencies

G1 follows G0. G2 and G3 build on G1. G4 starts with dry explicit G1 and integrates
G2/G3 before acceptance. G5 can develop against eager execution independently
of graph support. G6 joins G2–G5; G7 is the production acceptance gate.

### G0 — Capability inventory and reproducible baselines

Owners: snapy, with provider maintainers.

- Pin the four-repository revision set and audit actual UCX/CUDA synchronization
  and progress semantics. Enumerate host reads, dynamic allocations, counters,
  callbacks, communication phases and existing unsupported physics combinations.
- Turn the Straka executor's restrictions into an explicit capability matrix;
  distinguish numerical support from graph support.
- Save eager FP64/FP32 baselines, configurations, restart states, budgets and
  performance traces for each target family. Define tolerances before numerical
  changes, including the new cubed-sphere FP32 work.
- Add small capture probes for temperature inversion, saturation, VIC and each
  exchange segment. A successful capture is only a probe, not a replay gate.

Exit: revision manifest, predeclared numerical baselines and reproducible failure
probes. The plan/source inventory is recorded; these execution gates are pending.

### G1 — General executor, stable storage and device diagnostics

Owners: snapy + pyharp.

- Extract reusable lifecycle/segmentation from the test-only executor. Keep
  capability checks explicit and reject unsupported strict graph requests before
  launching work. Report segment count, captures, eager boundaries and reasons.
- Add pyharp's output-buffer stage combination; preserve existing evaluation
  order, intermediate precision and output rounding. Keep backup state separate.
  Switch hot hydro/scalar state updates to fixed storage and stabilize primitive
  tracer buffers where needed. Do not replace every `set_` with a temporary plus
  `copy_`; that can add full-field copies at every RK stage.
- Retain the existing alias/copy approach until fixed-storage equivalence passes.
  Fusing RK arithmetic is a separate optimization with its own numerical gate.
- Replace snapy's host limiter-mark queues with replay-correct persistent device
  diagnostics, or explicitly schedule their aggregation outside captured segments.
  Preserve vapor repair, heat-capacity checks, limiter behavior and retry limits.
- Support checkpoint/restart invalidation and externally edited/replaced state.
  Initialization/reset-only `set_data` calls need not be removed wholesale.

Exit: dry single-block graph/eager parity in both precisions, supported RK
schemes, dt changes, alias tests, limiter enabled, injected NaNs, rollback and
retry exhaustion. No unexplained eager or FP64 performance regression.

### G2 — Moist thermodynamics and source terms

Owners: kintera + snapy.

- First run explicit Cartesian moist dynamics with suitable compute segments
  captured and thermodynamics eager, preserving the final-stage adjustment order.
- Make `ThermoY::_intEng_to_temp` capture-safe; its host convergence `.item()`
  currently runs even for constant heat capacity. Preserve physical tolerances
  and failure reporting. Keep equilibrium's existing device-side iteration;
  variable iteration count inside a CUDA kernel does not require host control.
- Cache saturation-pressure specifications and function-pointer/derivative
  tables. Current per-call host uploads and temporary Thrust device vectors need
  persistent storage; audit shared-memory kernel setup and all lazy initialization
  before capture. Cache energy/capacity constants where profitable.
- Give saturation failures a persistent device counter with in-place reset and
  accumulation. Replace snapy's stage-zero drain and post-step handle deletion
  with a lifecycle that executes correctly on every replay.
- Preserve reaction-set/nactive warm starts across steps and restore appropriate
  state after warmup/retry. Test active-set changes during replay, not only an
  equilibrium state with an unchanged active set.
- Include heating, sedimentation, vapor repair and species positivity in the
  coupled tests. Keep latent-energy and species bookkeeping unchanged.
- Audit temperature-dependent heat capacities and optional chemistry separately.
  Capture-safe inversion alone does not enable them in FP32: the perturbation EOS
  currently rejects NASA9/H2 heat capacities. Required extensions need their own
  eager numerical gate before enabling graph support; remaining options must be
  explicitly unsupported, not silently replaced by constant heat capacity.

Exit: single-block explicit dry/moist FP64/FP32 replay passes closed, heated,
condensing, evaporating and sedimenting cases; water/energy budgets, convergence,
failure propagation and rollback match the corresponding eager reference.

### G3 — Vertical implicit correction and precision fallback

Owner: snapy; reuse kintera/pyharp interfaces as needed.

- Start with dry Cartesian FP64 VIC. Convert per-stage host finite/singularity
  screens and host failure booleans into device status, safe masked correction
  handling, and the existing global rejection mechanism. A failed column must
  not contaminate redistribution or diagnostics before rejection.
- Add FP32 VIC with stage-local FP64 fallback. Initially split graphs around the
  status/fallback decision, preserving the current stage RHS and fallback policy.
  Keep FP32 and FP64 workspaces/storage distinct where graphs retain addresses.
  Do not unconditionally compute both solves or change stage-local fallback into
  a whole-step FP64 rerun merely to ease capture.
- Keep fallback counters, clamp meters, mass transfers and gravity work correct
  on every replay. Use valid warmup states/dt and restore all bookkeeping.
- Validate moist VIC after G2. Radial/cubed-sphere FP32 VIC follows G5's numerical
  extension rather than bypassing the current Cartesian guard.

Exit: explicit/VIC parity at existing tolerances, partial/full schemes, long
acoustic timesteps, singular/near-singular/nonfinite columns, forced FP64
fallback, subsequent-stage failure persistence and exhausted-retry restoration.
Keep **nb1=1** for VIC: horizontal distribution is supported; a vertically
distributed implicit column solver is a separate numerical/communication project.

### G4 — Multiple Cartesian blocks and UCX processes

Owners: snapy; commux only if a demonstrated interface gap requires it.

- First support two/many local blocks on one GPU. Use fixed per-block compute
  graphs and host-driven exchange phases rather than capturing thread scheduling,
  mutexes or condition-variable state. Preserve CUDA stream/event dependencies.
- Support two GPU processes with UCX, then several local blocks per process.
  Build segment boundaries for every enabled within-stage communication path.
  Warm up all ranks consistently; preserve tags and request grouping.
- Keep global CFL, retry, conservation corrections and reductions outside capture
  initially. All blocks/ranks must restore on a remote rejection, including when
  one rank exhausts its retry budget.
- Exercise unequal block contents, uneven ownership where supported, repeated
  replay, changed dt, restart and stable buffer lifetime under communication.
- Inspect commux before any changes. Add producer/consumer event or completion
  contracts only if existing waits cannot express correctness or measured overlap.
  Correct segmented execution is not gated on a speculative commux rewrite.

Exit: same-decomposition eager/graph parity and decomposition-invariance budgets
on one GPU and two UCX GPU processes, including moist explicit/VIC cases from
G2/G3. No hangs, stale halos, missed failure flags or buffer reuse races.

### G5 — Eager FP32 cubed-sphere numerical support

Owner: snapy. This is a prerequisite with its own review and acceptance.

- Extend anomaly/reference representation through panel interpolation, vector
  frame transforms and corner fills. Interpolating stored anomalies and reference
  fields must reproduce the intended total-state boundary operator; do not simply
  delete the Cartesian guard or independently round cancelling total pressures.
- Extend perturbation HLLC pressure gauges and flux/source balance to curved
  metrics, radial gravity work, stretched x1 and seam/covariance operators.
- Resolve the current full axial-angular-momentum FP32/perturbation restriction
  with a validated formulation where that mode is required. Track enabled modes
  explicitly; graph support must not mask an unsupported numerical combination.
- Extend FP32 VIC's geometry/metric treatment and fallback to the radial column
  formulation, preserving nb1=1 and all existing FP64 behavior.
- Reuse existing geometry, reference and seam operators. Establish FP32 budgets
  against eager FP64 for rest balance, transport, moist budgets, positivity,
  gravity work and angular momentum before graph integration.

Exit: eager FP32 Cartesian/cubed-sphere dry/moist explicit/VIC matrix passes its
numerical gates, including panel edges/corners and distributed decomposition.
This milestone cannot be marked complete using FP64 cubed-sphere graph tests.

### G6 — Cubed-sphere graphs and combined configurations

Owners: snapy + kintera + pyharp; commux as required by G4.

- Capture compute/pack/unpack segments around reconstructed face-state exchange,
  primitive/reference exchange, panel rotations/interpolation and corner filling.
- Preserve global angular-momentum/gravity-work correction ordering and reduction
  semantics. Device flags and solver fallback remain correct across all segments.
- Run six-panel cases on one GPU, then across two UCX processes; include multiple
  tiles per panel and several blocks per process. Exercise panel seams/corners,
  both precisions, moisture/sedimentation, VIC and forced fallback/rejection.

Exit: the complete target matrix below passes. Physics combinations must pass
jointly; isolated moist, VIC and exchange tests are insufficient evidence.

### G7 — Production rollout and performance acceptance

Owner: snapy, coordinated provider releases.

- Expose a documented execution selection at Mesh/driver level. The existing
  Straka CLI is only a prototype; choose final YAML/Python names at G0/G1.
  Explain required eager communication/fallback boundaries and reject unsupported
  strict requests clearly. Never silently run an entire unsupported case eager
  while reporting it as graph-enabled.
- Document feature/version requirements, invalidation, restart behavior, supported
  geometries/physics and UCX synchronization. Keep eager fallback explicit.
- Benchmark all four precision/execution combinations with fixed grids, physics,
  diagnostics and checks. Alternate run order on the same GPU hardware; report
  repeats, spread, kernel time, wall time, graph setup/recapture, peak memory,
  communication/reduction time and fallback frequency.
- Retain 2× versus matched eager FP64 as the historical Straka target. Report
  graph FP32 versus graph FP64 separately. Do not promise 2× for compute-bound
  thermodynamics or communication-bound cases. Define workload-specific targets
  from G0 baselines before optimizing; investigate any material regression.

Exit: merged compatible provider versions, reproducible CUDA/UCX CI, complete
support matrix and user documentation, with remaining exclusions explicit.

## Acceptance matrix and regression policy

Each row must exercise FP64/FP32, dry/moist, and explicit/VIC combinations; VIC
retains nb1=1. Add limiter-enabled and relevant source/correction variants.

| Geometry and distribution | Required coverage |
|---|---|
| Cartesian, one block/GPU | Straka baseline; closed/heated/sedimenting moist column; VIC failure/fallback |
| Cartesian, multiple local blocks | Halo parity, local stream ordering, block-wide rollback |
| Cartesian, two or more UCX GPU processes | Multiple blocks/process, global dt/retry, restart and cross-process conservation |
| Cubed sphere, six panels on one GPU | All panel edges/corners, frame transforms, rest state and angular momentum |
| Cubed sphere, distributed UCX | Six panels plus tiled panels, moist/VIC coupling, remote rejection and corrections |

For execution/storage-only changes, require bit-identical same-precision eager
and graph outputs where arithmetic order is unchanged. If a numerical extension
or fusion changes rounding, document the reason and use predeclared scientific
tolerances; never loosen tolerances merely to make a graph test pass.
Compare different decompositions using the established conservation/error
budgets where reduction order precludes bitwise equality.

All families include many replays, changing state/active sets, dt changes,
in-place edits and replacement, checkpoint/restart recapture, failed solves,
nonconvergence, a failure on only one block/rank, rollback and retry exhaustion.
CUDA/UCX tests that skip do not count as acceptance. Add multiprocess timeouts
and failure diagnostics so communication bugs do not hang CI indefinitely.

Reuse and extend existing suites: `test_straka_cuda_graph`, `test_fp32_compute`,
`test_fp32_restart`, `test_fp32_science`, `test_fp32_jupiter`,
`test_fp32_vic_pivot`, `test_lu_failure`, `test_vic_moist_device`,
`test_mesh_multi_block`, `test_mesh_exchange`, `run_exchange_backend.py`,
`test_cubed_sphere_exchange`, `test_cubed_sphere_seams`,
`test_flux_positivity_cubedsphere_moist` and `test_sedimentation_cubed_seam`.
Add provider-local kintera capture/replay/failure tests and pyharp output-alias
tests. New graph test names/CTest labels are to be added during implementation.

Existing baseline commands (not new support-validation claims):

```sh
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_straka_cuda_graph_python$'
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -L fp32-foundations
python tests/run_exchange_backend.py --build-dir build --build-type release --backend ucx --device cuda
```

Use the pinned matching package/build environment. Record exact commands, exit
status, skipped tests, hardware/revisions and evidence paths in the progress
file for every milestone. Broad M6 science gates remain separate unless their
actual acceptance tests are run; graph implementation alone does not close M6.

## FP32 foundations and science roadmap

The original M0–M6 design and validation requirements remain part of V3. The G
milestones extend their execution support and add the separate eager FP32
cubed-sphere numerical prerequisite; they do not supersede open M6 science gates.

### Design and scope

First supported case: the Cartesian ideal-moist Jupiter workflow, using WENO5,
HLLC, RK3, vertical implicit correction, H2O/NH3 chemistry and sedimentation.
Production driver/deck: `run_jupiter_moist_beta.py` with
`jup_crm3d_H2O-NH3_beta.yaml` (100 x 400 x 200 cells). Reduced acceptance decks
from `workspace/UM-JUPITER/reduced/`: `jup_a_rest.yaml`,
`jup_b_bot_heat.yaml`, `jup_c_full_short.yaml` (100 x 16 x 8 cells).
Complete perturbation kernels before enabling true FP32. Preserve default FP64
behavior. Discrete initial-profile balancing is an explicit opt-in operation.

Reuse the existing well-balanced operator, not a second hydrostatic solver:
`_hydro_ref_x1` already owns wall treatment, anchors, stretched quadrature and
vertical seam exchange. `balance_column` uses the same numerical kernel.
The static storage background and the per-sweep dynamic hydrostatic reference
serve different purposes, but share operator configuration and evaluation.
Boundary references and horizontal-mean diagnostic anomalies remain separate.

The pressure scan is linear only with fixed anchors and inactive clamps.
Density references, wall continuation, anchor construction and guards require
total-state information. Gravity uses mixture density, not stored dry density.
Ideal-moist pressure conversion is algebraic; cancellation, rather than an EOS
iteration tolerance, is the first problem to solve.

### Milestones

#### M0 — Baseline and project records

Record current branch/revisions and freshly build relevant tests. Existing
storage emulation and anomaly storage are implemented, but require fresh-build
validation. Keep unprojected and discretely balanced rest cases separate.

#### M1 — Consolidate reference machinery

Share grid classification, stretched stencil construction and reference
allocation/dispatch between initialization and runtime. Preserve runtime seam
communication and wb_ref4 ordering. Extend the existing Fp64Store background
with species information when thermodynamic perturbations need it. Keep u0 and
restart compatibility. Derived caches must follow geometry, dtype and device.
Do not project or horizontally average supplied initial conditions implicitly.
Gate: FP64 anomaly, balance, wall, stretched-grid, seam and restart regressions.

#### M2 — Cancellation-safe thermodynamics

Preserve public total-state EOS keys and anomaly helpers. Add internal total
primitive plus pressure/mixture-density perturbation results. Compute pressure
changes directly from energy and species differences, including kinetic energy
and species energy offsets; avoid subtraction of rounded total pressures or
composition factors. Explicit total conversions use FP64. Active floors round
toward representable anomalies satisfying the total-state lower bound, without
changing inactive cells. Keep thermodynamic primitives in kintera where they
belong to its existing abstractions.
Gate: dry/moist sub-ulp perturbations, changing composition, kinetic energy,
round trips and active density/temperature floors against FP64 calculations.

#### M3 — Perturbation dynamics

Opt-in dynamics/perturbation-flux requires anomaly storage and Cartesian
ideal-moist HLLC. Extend the shared reference evaluator with background plus
perturbations; retain FP64 nonlinear anchor/guard/fallback calculations.
Preserve wall parity, wb_ref4 and distributed seams. Add optional shared face
pressure backgrounds to HLLC on CPU, CUDA and tensor backends. Pressure
subtractions and momentum flux use perturbations; wave speeds and energy
transport use totals. Include horizontal sweeps. Keep total and perturbation
face-pressure semantics explicit for downstream consumers. Preserve static
background residual forces, including for unbalanced initial states.
Consolidate explicit/implicit Cartesian gravity work using local potential
differences; preserve sedimentation and wall corrections.
The shared Cartesian face-work arithmetic is not bitwise identical to main:
reviewers measured about 2e-9 relative velocity change in FP64 Straka at 40
steps (density/pressure unchanged at output precision). Require numerical
equivalence and conservation; retain exact same-dtype restart continuation.
Gate: FP64 equivalence, conservation, storage-emulated rest noise and split
columns. A velocity target alone cannot distinguish initialization imbalance.

##### M3 delivery stages after merged PR #326

1. **M3a — Explicit signed pressure scan (implemented).** Extract the
   top-to-bottom/bottom-to-top recurrence from the existing reference kernel.
   Accept an explicit anchor increment and mixture-density increments; return
   signed lower/upper face-pressure increments without floors or a total-state
   subtraction. Share the recurrence with the default CPU/CUDA path, retaining
   its anchor calculation and floors. Gate: default-reference regression,
   sub-ulp signed increments, linearity, zero gravity, irregular widths,
   ghost-face orientation, and vertically split-column anchor relay. Keep this
   internal; do not expose a working perturbation-flux mode yet.
2. **M3b — Split complete reference evaluation (implemented).** Use M3a with background plus
   deltas; evaluate anchor changes, pressure floors, pressure quadrature guards,
   and rho/p wall continuation from FP64 totals. Define guard/fallback behavior
   and retain the static residual force for unbalanced backgrounds. Cache only
   geometry/stencil setup; rebuild branch-dependent references each sweep. This
   avoids a second derived-background cache and its restart invalidation. Preserve wb_ref4/conversion-order 4 and seams.
3. **M3c — Wire reconstruction, HLLC, and sources together (implemented).** Carry pressure
   deltas through all sweeps and the existing perturbation HLLC entry point;
   separate pressure-delta momentum flux from total energy pressure. Add the
   opt-in configuration only when flux/source, positivity, boundaries, and
   explicit/implicit pressure consumers have matching contracts. Validate
   balanced and deliberately unbalanced rest, heating, and split columns before
   advancing to M4.

M3b/c reuse the complete reference kernel with finite-increment scalar pairs.
The opt-in mode now covers flux/source coupling and horizontal sweeps, while
FP64 reconstruction and nonlinear decisions remain explicit boundaries.
See [the M3 numerical contract](docs/derivations/fp32-perturbation-dynamics.md)
for branch semantics, static residual forces, face-pressure consumers and
supported scope. CPU validation is recorded in V3_PROGRESS.md; CUDA execution
and performance still require suitable hardware. True FP32 computation remains
M4, including split reconstruction or an explicit FP64 reconstruction boundary.

#### M4 — True FP32 with explicit boundaries (implemented; CPU validated)

FP32 storage, reconstruction, fluxes and tendencies; FP64 background/geometry
masters, sensitive nonlinear reference calculations, reductions and time.
Kernel geometry views match compute dtype. Lazy buffers follow their compute
policy. Cast before reductions. Assemble/solve vertical implicit columns in
FP32 under the FP32 policy, retaining pivot/nonfinite checks and retrying rejected
blocks from the original RHS in FP64. Allow an explicit FP64 solve override and
count fallbacks. Keep chemistry/saturation in FP64, applying their increments
without a lossy FP32 total-state round trip. Audit forcing, diffusion, sedimentation, boundaries,
scalar transport and rollback. Test dtype contracts with an explicit allowlist.
Keep Python's global default unchanged; driver precision is explicit.
Gate: reduced full physics completes 1000 steps without dtype/nonfinite failures
or precision-induced retry exhaustion. CPU results and retained hardware limits
are recorded in V3_PROGRESS.md; see the
[compute policy](docs/derivations/fp32-compute-policy.md) for configuration.

#### M5 — Restart and coupled workflow

Retain totals plus exact stored-anomaly/reference restart fields; version added
background metadata and record dtype. Same-dtype continuation is exact;
cross-dtype continuation has numerical tolerances. Retain legacy restrictions.
Regridding is excluded from the requested scope; changed-grid restarts remain
rejected. Coordinate snapy and kintera
changes and track paddle driver/forcing integration separately.
Gate: uninterrupted/resumed and reduced rest/heating/full-moist workflows.
See [the restart contract](docs/derivations/fp32-restart.md) for versioned
metadata, destination precision and explicit changed-grid rejection. External
paddle/production integration remains separately tracked.

#### M6 — Science validation and performance

The Straka performance work now targets **2× GPU throughput versus eager FP64** on
the fixed reference case. See the [staged speed plan](docs/notes/fp32-straka-speed-plan.md)
for HLLC arithmetic, kernel fusion, dispatch reduction and measured acceptance.

Check mass/species/energy budgets including sources and boundary fluxes. Compare
long-run statistics against multiple FP64 seeds as well as deterministic tests.
The reduced-case harness is `tests/test_fp32_science.py`; measurements and open
acceptance issues are recorded in [the M6 validation report](docs/notes/fp32-m6-validation.md).
Profile production GPU memory, throughput and FP64/casting costs. Lower retained
FP64 precision only with regression evidence. Publish measured supported scope
and gains; BF16/stochastic rounding are separate research, not delivery gates.

### Validation and cross-repo compatibility

Run relevant neighboring tests per milestone, including CUDA and distributed
seams on suitable hardware. The implemented foundation gate is `ctest --test-dir build --output-on-failure
-L fp32-foundations`; build/install details and results are in V3_PROGRESS.md.
Broader regression selection as later milestones land:

```sh
ctest --test-dir build --output-on-failure -R 'test_(anomaly_storage|fp32_storage|balance_column|hydro_ref_x1|stretched_x1|wb_|riemann|gravity_work|implicit|restart|x1_seam)'
```

Kintera owns thermodynamics/chemistry; source changes there are authorized for
this effort. The merged pressure helper requires kintera >= 2.6.9; until its
wheel is published, build tag v2.6.9 from source (as CI does). Validate against
the actual library and record both revisions. Main compatibility risks are stored hydro_u
interpretation, chemistry/forcing total-state writebacks, and kintera
dtype assumptions. Other EOS/geometries remain FP64 regression obligations;
LMARS perturbation support is deferred.
