# Perturbation pressure fluxes (M3)

Enable with:

```yaml
dynamics:
  anomaly: true
  perturbation-flux: true
  equation-of-state: {type: ideal-moist}
  riemann-solver: {type: hllc}
```

The initial supported path is Cartesian, constant heat capacity ideal-moist,
without immersed solids. It requires the composition background written by new
anomaly restarts. Reconstruction, nonlinear decisions and background arithmetic
remain FP64 on CPU/CUDA. This switch does **not** enable model FP32; M4 owns that
compute policy. MPS cannot provide the FP64 arithmetic this implementation needs.
The default dynamics path is unchanged.

## Reference arithmetic

`HydroReferenceX1Operator::evaluate_split` invokes the same templated kernel as
ordinary reference evaluation. Each scalar stores `(base, delta)`, with delta a
finite increment, not a derivative. For example:

```
(a*b).delta = a.base*b.delta + a.delta*(b.base+b.delta)
(a/b).delta = (a.delta - (a.base/b.base)*b.delta)/(b.base+b.delta)
```

Exponentials and logarithms use `expm1` and `log1p`. The shared signed scan thus
never forms and subtracts two large face pressures to obtain an increment.
The exponential top anchor, log mean, wall rho/p continuation, quadrature and
fallback all use the existing operator. Decisions compare FP64 totals. A floor
returns `(minimum_positive, 0)`; a quadrature fallback selects both components
of the fallback. Consequently the base follows the **current selected branch**,
not a separately evaluated and possibly differently guarded background.

Only geometry/stencil setup is cached. Branch-dependent values are reevaluated
every sweep, so reference replacement and restart cannot retain stale derived
backgrounds. The existing geometry cache checks identity, version, device, dtype
and wall/grid configuration. Vertical seams relay both pressure components and
exchange owner-cell reference components before reconstruction. Conversion order 4
corrects total face density after the existing owner-cell exchange. The removed
`SNAP_WB_REF4` switch is not used; the interior density reference follows main
and uses each cell's own rho/p.

## Reconstruction and flux/source contract

For x1, reconstruct the pressure residual

```
(p0 - reference.cell0) + (dp - reference.cell_delta).
```

Keep the background residual even when the supplied initial state is not
balanced. Restore only `reference.face_delta` into the reconstructed pressure
row. HLLC receives `reference.face0` separately. Its wave speeds and energy flux
use total pressures; its normal momentum flux and internal face-pressure result
exclude this shared face background. Positivity decisions use FP64 totals and
retain the existing reference fallback and wall parity.

Horizontal and gravity-free sweeps retain their existing FP64 reconstruction.
Their shared face gauge is the adjacent-cell average of the static EOS pressure
background. Subtract it in FP64 before perturbation HLLC and retain its gradient
in the source. This keeps nonlinear WENO decisions identical to total-state
reconstruction. M4 must retain this FP64 boundary or implement split WENO before
lowering the reconstruction dtype; casting total primitives first is not safe.

Let `B` be the x1 face background, `D` the Cartesian face divergence, `h` the
non-hydrostatic fraction and `Gdelta` the existing hydrostatic correction formed
from pressure-delta face states. The vertical momentum source paired with the
pressure-delta flux is

```
h * [g*drho + (g*rho0 - D(B))] + (1-h)*Gdelta.
```

For an unclamped scan, `D(B)=g*rho0` algebraically: cancel this pair directly,
without introducing a roundoff residual from differencing large pressures.
Where a pressure floor breaks that identity, evaluate and retain the residual
in FP64. The source never removes the reconstructed initial-profile imbalance.
Remote seam averaging includes `B`; its change is accounted for in the source
so the sum of flux plus background remains single-valued. In-process seams use
the existing owner-cell/anchor exchange.

`face_pressure1()` remains **total** pressure for implicit and energy consumers.
`flux1/2/3()` normal momentum excludes the shared background in this opt-in mode;
mass, species and energy fluxes retain total semantics. Scalar transport reads
only unchanged mass fluxes. Sedimentation and positivity limiting retain their
existing ordering. Constant gravity uses density increments for vertical
momentum but totals for energy; the existing shared Cartesian face-work operator
then supplies explicit/implicit gravitational energy work.

## Validation scope

`ctest --test-dir build --output-on-failure -L fp32-dynamics` covers paired
reference arithmetic (including increments below FP64 ulp), current-state
floors/guards, uniform/stretched/log-mean grids, full dynamics equivalence,
conversion order 4, implicit correction, partial hydrostatic correction, moist
heating/sedimentation, deliberately unbalanced and balanced rest, FP32 storage
rounding, mass/energy checks, exact restart continuation and same-process split
columns. The foundation label also protects the default path. CUDA variants of
the reference tests are registered but skip when CUDA is unavailable. CPU
validation does not establish GPU performance or a true FP32 dtype contract.

M4 adds explicit `dynamics/precision: float32` on top of this mode. See the
[compute policy](fp32-compute-policy.md) for FP32 reconstruction, vertical
implicit correction and retained FP64 boundaries. The default remains FP64.
