# FP32 roadmap for snapy and kintera

Status and validation evidence: [FP32_PROGRESS.md](FP32_PROGRESS.md).

## Design and scope

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

## Milestones

### M0 — Baseline and project records

Record current branch/revisions and freshly build relevant tests. Existing
storage emulation and anomaly storage are implemented, but require fresh-build
validation. Keep unprojected and discretely balanced rest cases separate.

### M1 — Consolidate reference machinery

Share grid classification, stretched stencil construction and reference
allocation/dispatch between initialization and runtime. Preserve runtime seam
communication and wb_ref4 ordering. Extend the existing Fp64Store background
with species information when thermodynamic perturbations need it. Keep u0 and
restart compatibility. Derived caches must follow geometry, dtype and device.
Do not project or horizontally average supplied initial conditions implicitly.
Gate: FP64 anomaly, balance, wall, stretched-grid, seam and restart regressions.

### M2 — Cancellation-safe thermodynamics

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

### M3 — Perturbation dynamics

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

#### M3 delivery stages after merged PR #326

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
supported scope. CPU validation is recorded in FP32_PROGRESS.md; CUDA execution
and performance still require suitable hardware. True FP32 computation remains
M4, including split reconstruction or an explicit FP64 reconstruction boundary.

### M4 — True FP32 with explicit boundaries (implemented; CPU validated)

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
are recorded in FP32_PROGRESS.md; see the
[compute policy](docs/derivations/fp32-compute-policy.md) for configuration.

### M5 — Restart and coupled workflow

Retain totals plus exact stored-anomaly/reference restart fields; version added
background metadata and record dtype. Same-dtype continuation is exact;
cross-dtype continuation has numerical tolerances. Retain legacy restrictions.
Regridding invalidates all background-derived data. Coordinate snapy and kintera
changes and track paddle driver/forcing integration separately.
Gate: uninterrupted/resumed and reduced rest/heating/full-moist workflows.

### M6 — Science validation and performance

Check mass/species/energy budgets including sources and boundary fluxes. Compare
long-run statistics against multiple FP64 seeds as well as deterministic tests.
Profile production GPU memory, throughput and FP64/casting costs. Lower retained
FP64 precision only with regression evidence. Publish measured supported scope
and gains; BF16/stochastic rounding are separate research, not delivery gates.

## Validation and cross-repo compatibility

Run relevant neighboring tests per milestone, including CUDA and distributed
seams on suitable hardware. The implemented foundation gate is `ctest --test-dir build --output-on-failure
-L fp32-foundations`; build/install details and results are in FP32_PROGRESS.md.
Broader regression selection as later milestones land:

```sh
ctest --test-dir build --output-on-failure -R 'test_(anomaly_storage|fp32_storage|balance_column|hydro_ref_x1|stretched_x1|wb_|riemann|gravity_work|implicit|restart|x1_seam)'
```

Kintera owns thermodynamics/chemistry; source changes there are authorized for
this effort. The merged pressure helper requires kintera >= 2.6.9; until its
wheel is published, build tag v2.6.9 from source (as CI does). Validate against
the actual library and record both revisions. Main compatibility risks are stored hydro_u
interpretation, chemistry/forcing total-state writebacks, regridding, and kintera
dtype assumptions. Other EOS/geometries remain FP64 regression obligations;
LMARS perturbation support is deferred.
