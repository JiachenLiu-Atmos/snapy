# M4 FP32 compute policy

Enable the Cartesian constant-cv ideal-moist HLLC path explicitly:

```yaml
dynamics:
  anomaly: true
  perturbation-flux: true
  precision: float32
  riemann-solver:
    type: hllc
integration:
  implicit-scheme: 9
  implicit-precision: auto
  implicit-fallback-fp64: true
```

The default dynamics policy remains `float64`. `implicit-precision: auto`
follows the dynamics policy; `float32` and `float64` explicitly select the
vertical solve precision. Vertical implicit correction still requires one
block along x1. FP32 implicit correction supports Cartesian ideal-moist EOS.

## Dtype boundaries

Stored anomalies, species, passive scalars, RK work arrays, reconstruction,
HLLC scratch/fluxes, divergence and tendencies use FP32. Uniform and stretched
reconstruction coefficients and kernel geometry views match their inputs.
Geometry and reference masters remain FP64. Under this explicit policy,
`MeshBlock.to(dtype)` preserves the configured mixed precision; device moves
remain supported. Python's global default dtype is unchanged.

Total primitive/EOS values, nonlinear hydrostatic reference decisions, pressure
gauges, total face pressure and sensitive reductions remain FP64. Explicit
forcing is evaluated into an FP64 increment buffer and only its increment is
added to the FP32 tendency. Chemistry and saturation operate on FP64 totals;
apply species increments without a round trip through FP32 total energy.
Kintera's existing APIs suffice; no new kintera API or source change is required.

Before x1 FP32 reconstruction, subtract a constant pressure gauge from the
paired hydrostatic residual. Relay that gauge with the existing reference
anchor across vertical seams. Horizontal and gravity-free sweeps subtract a
static pressure gauge before FP32 reconstruction and restore it in FP64 before
forming the shared HLLC face gauge. The existing total-pressure energy and
wave-speed semantics are retained. Static background residual forces remain
active, including deliberately unbalanced initial states.

## Vertical implicit correction

Assembly, block tridiagonal solve, correction and constituent transport use
FP32 when selected. Energy-offset removal/restoration uses FP64. Existing
pivot and nonfinite checks remain active, and screening reductions promote
before summation. On a rejected FP32 solve, retry the entire block in FP64
from the original RHS. The existing failure contract rejects a whole block,
so this does not retain partially accepted columns. The registered int64
`hydro.icorr.fp64_fallbacks` buffer counts retries. A failed FP64 retry retains
the normal timestep rejection. Disabling fallback also retains normal rejection
and exact rollback. A fallback may temporarily leave implicit workspace FP64;
the next FP32 solve resizes it to the selected compute dtype.

## Validation scope

`ctest --test-dir build --output-on-failure -L fp32-foundations` includes dtype
contracts, FP64 comparisons, balanced rest, stretched grids, outflow, exact
restart/rollback, vertical seam comparisons and implicit stress/fallback tests.
`ctest --test-dir build --output-on-failure -L fp32-acceptance` runs 1000-step
moist explicit/implicit cases and a reduced Jupiter fixture with H2O/NH3
kinetics, sedimentation, heat/cooling, diffusion and scalar transport.

The Jupiter fixture derives from the repository example; it is not the
external production deck. CPU results and exact commands are recorded in
[FP32_PROGRESS.md](../../FP32_PROGRESS.md). CUDA compilation/execution and
production GPU performance remain unverified on the development host. MPS,
immersed solids and other EOS/geometries are outside the FP32 dynamics scope.
