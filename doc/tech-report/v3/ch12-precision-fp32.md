# 12 Precision: the FP32 policy

<!-- PROVENANCE
pin: snapy@73c45a7c8c991209924e1137992352c335575062 (UCzhangxi/snapy v3/final at the code pin; code_pin.yml)
outline: doc/tech-report/STRUCTURE.md at UCzhangxi/snapy tech-report ca48c817e9d8d51bdb7cb7e42662d199318a30cd, chapter 12
sources: src/mesh/precision_policy.hpp, src/mesh/meshblock.cpp, src/utils/tensor_cast_cache.hpp, src/hydro/hydro.cpp,
         src/hydro/hydro.hpp, src/hydro/hydro_forward.cpp, src/hydro/sweep.cpp, src/hydro/fp64_store.hpp,
         src/eos/ideal_moist.cpp, src/riemann/riemann_kernels.h, src/implicit/implicit_hydro.cpp,
         src/implicit/implicit_hydro.hpp, src/implicit/forward_sweep_impl.h, src/options/hydro.yaml and the tests named
         in the Tests layers, all read at the pin
form: pre-conversion draft (STYLE.md section 0): `path:lines@sha` code locations; section 10 (Quarto markup) does not bind it yet
status: draft
author: Xinyue Wang
reviewer: (checker from tools/chapter-owners.tsv: cshsgy)
-->

> **Draft conventions** (as in the chapter 4, 5 and 17 drafts). Every code location is `path:lines@73c45a7c`, the
> code pin of `code_pin.yml`. A bare `:N` refers to the file of the last full citation before it. A claim not read in
> the code at the pin is marked **unverified**. No test was run for this draft: the Tests layers report what each
> test asserts in its source, and every section says **Run evidence at the pin: missing evidence**.

The book chapter `12-configuration.qmd` keeps the switch and configuration reference. The outline moves that
material to chapter 15 and appendices D and I. This draft is the outline's chapter 12, the FP32 precision policy:
which dtype each quantity is stored and computed in, how FP32 state stays well balanced, and the FP32 vertical solve.
kintera is not cited here. The one call into it (the constant-$c_v$ pressure increment in 12.3) is named and not
described.

---

## 12.1 The precision policy

### Summary

One table decides the dtype of each quantity. `dynamics/precision` sets it to `float64` (the default) or `float32`,
and `integration/implicit-precision` sets the implicit row. In FP32 the evolved state, its backup, the scalars,
the fluxes and the implicit solve are FP32. References and geometry are stored FP64 and cast to FP32 per use,
and the primitives stay FP64. The policy has one
rule for the widest dtype a device holds: FP64, except FP32 on MPS. So on MPS every FP64 row of the table holds FP32.
The block rebuilds its policy for the device it is on when it is set up and whenever it is moved.

::: {.callout-note title="At a glance"}
**Purpose.** One dtype contract per quantity, for both precisions and every device.

**Data.** `dynamics/precision`, `integration/implicit-precision` and the block's device in; a storage and a compute
dtype per quantity out.

**Placement.** Built at block setup and rebuilt on every device move.

**Activation.** Always. `float64` is the default. `float32` narrows four rows entirely, plus the compute dtype of
the reference and implicit rows.

**Guarantee / limits.** References and geometry are stored FP64 on CPU and CUDA. MPS has no FP64, so the policy holds FP32
there.

**Evidence.** `test_precision_policy.release`, `test_fp32_policy_python`, `test_mps_matches_cpu_python`.
:::

### Derivation

None. The policy is a table, not a scheme. Let $w$ be the device's wide dtype, FP32 on MPS and FP64 elsewhere
(`PrecisionPolicy::wide`, `src/mesh/precision_policy.hpp:51-54@73c45a7c`). Let $n$ be FP32 under `float32` and
$w$ otherwise (`:37-38`). The constructor fills every row with $(w, w)$, gives five rows $(n, n)$, sets the reference
row's compute dtype to $n$, and sets the implicit row's storage back to $w$ (`:39-45`):

| quantity (`Quantity`) | what it holds (`:13-21`) | storage | compute |
|---|---|---|---|
| `kHydroState` | `hydro_u`, anomaly form | $n$ | $n$ |
| `kBackup` | `u0`, `s0` | $n$ | $n$ |
| `kScalarState` | `scalar_s` | $n$ | $n$ |
| `kFlux` | fluxes, `du` | $n$ | $n$ |
| `kReference` | reference and geometry masters | $w$ | $n$ |
| `kPrimitive` | `hydro_w` totals | $w$ | $w$ |
| `kImplicit` | the vertical implicit solve | $w$ (its FP64 stage) | $n$, or as `implicit-precision` sets it |

`implicit-precision` other than `auto` overrides the implicit compute dtype: `float32` gives FP32 and `float64`
gives $w$ (`:46-48`). The key accepts only `auto`, `float32` and `float64`, and `implicit-fallback-fp64` defaults to
true (`src/implicit/implicit_hydro.cpp:45-51@73c45a7c`; `src/implicit/implicit_hydro.hpp:56-57@73c45a7c`).
`dynamics/precision` accepts only `float64` and `float32`, and defaults to `float64`
(`src/hydro/hydro.cpp:34-35@73c45a7c`; `src/options/hydro.yaml:61@73c45a7c`).

### Numerical method

`MeshBlockImpl::_reset_policy` builds the policy from the hydro's precision, the given device and the implicit
precision (`src/mesh/meshblock.cpp:38-42@73c45a7c`). `reset` calls it on the factories' default device (`:189`).
`to(device, dtype)` and `to(device)` call it after the move (`:44-53`, `:60-65`). In FP32, `to(device, dtype)` moves
the device and ignores the dtype, and `to(dtype)` returns at once (`:46-47`, `:54-55`). So converting an FP32 block
to another dtype does not change its policy. `PrecisionPolicy::cast` and `store` convert a tensor to a row's compute
or storage dtype, and `Tensor::to` returns the tensor itself when it already has that dtype
(`src/mesh/precision_policy.hpp:60-66@73c45a7c`; asserted in `tests/test_precision_policy.cpp:65-75@73c45a7c`).

### Code

| what | code at the pin | symbol |
|---|---|---|
| the quantities | `src/mesh/precision_policy.hpp:13-22@73c45a7c` | `Quantity` |
| the table | `src/mesh/precision_policy.hpp:31-71@73c45a7c` | `PrecisionPolicy` |
| the wide-dtype rule | `src/mesh/precision_policy.hpp:51-54@73c45a7c` | `PrecisionPolicy::wide` |
| the block's policy | `src/mesh/meshblock.cpp:38-42@73c45a7c` | `MeshBlockImpl::_reset_policy` |
| the precision key check | `src/hydro/hydro.cpp:34-35@73c45a7c` | `HydroImpl::reset` |
| the implicit keys | `src/implicit/implicit_hydro.cpp:45-51@73c45a7c` | `ImplicitOptionsImpl::from_yaml` |

### Tests

| test (ctest name) | what it asserts | tolerance | source |
|---|---|---|---|
| `test_precision_policy.release` | every row in `float64` and `float32`; on an MPS device both precisions give FP32 in every row; `cast` and `store` return the tensor itself when its dtype already matches | exact (`EXPECT_EQ`, `torch::equal`) | `tests/test_precision_policy.cpp:26-75@73c45a7c` |
| `test_precision_policy.release` (`BlockPolicyFollowsTheBlockDevice`) | the block's policy is built for the block's device by `reset` and `to` | exact | `tests/test_precision_policy.cpp:77-80@73c45a7c` and below |
| `test_fp32_policy_python` | `hydro_u` is FP32 and `hydro_w` FP64; FP32 vs FP64 after 10 steps over five arms (conversion order 2 and 4, stretched, outflow) `< 2.0e-5` relative; `x1f` stays FP64 and unchanged through `block.to(float32)`; rest noise in the $x_1$ momentum `< 2.0e-8` after 100 steps (schemes 0 and 9); a redo restores the FP32 input bit for bit; an FP32 restart continues bit for bit | `2.0e-5`, `2.0e-8`, `torch.equal` | `tests/test_fp32_policy.py:25-95@73c45a7c` |
| `test_mps_matches_cpu_python` | every MPS path agrees with CPU at FP32: $\mathrm{err} \le \max(\mathrm{TOL}, 3 f_{64})$, with $\mathrm{TOL} = 10^{-4}$; the reconstruction arms match exactly; exits 125 (skip) without MPS | `TOL = 1.0e-4` | `tests/test_mps_matches_cpu.py:2-15@73c45a7c`, `:27` |

**Run evidence at the pin: missing evidence.** The MPS test runs only on an MPS device.

### Limits

1. MPS has no FP64, so on MPS the FP64 rows hold FP32 (`src/mesh/precision_policy.hpp:28-30@73c45a7c`). Anomaly
   storage, which keeps its reference in FP64, refuses MPS (12.3).
2. The figure the outline names (`src/snapy_report/ch12/figures/tab_policy_rows.py`) is not written for this draft.
   The table above is written from the constructor by hand.

---

## 12.2 FP32 storage, typed constants and the horizontal sweep

### Summary

In FP32 the stored state is FP32, but the primitives the reconstruction reads are FP64 (`kPrimitive`). The WENO
and LMARS float instantiations compute in FP32 throughout, because their constants are typed: no untyped constant
promotes them to FP64 (as the test that pins it states, `tests/test_fp32_typed_constants.cpp:1-9@73c45a7c`; the kernels'
constants themselves were not re-read for this draft). Constant tables that a kernel reads in another dtype or device are
converted once and cached. Before an FP32 sweep reconstructs, it removes a per-sweep pressure gauge, so the FP32
face states carry a pressure difference and not a large total.

::: {.callout-note title="At a glance"}
**Purpose.** FP32 compute without a silent promotion to FP64, and FP32 face states that do not lose the pressure
increment.

**Data.** FP64 primitives and the FP64 background pressure in; FP32 or FP64 face states out.

**Placement.** Each reconstruction call of the hydro sweep, when `dynamics/precision` is `float32`.

**Activation.** `dynamics/precision: float32`.

**Guarantee / limits.** The cast cache is valid only for writes that bump the tensor's version counter.

**Evidence.** `test_fp32_typed_constants.release`, `test_fp32_horizontal_sweep_python`, `test_fp32_compute_python`.
:::

### Derivation

Let $p_0$ be the FP64 background pressure of the anomaly reference (12.3), and $\delta p$ the stored pressure
perturbation. Along a sweep in direction $d$, the gauge is the midrange
$G = \tfrac12\,(\min_d p_0 + \max_d p_0)$ (`src/hydro/sweep.cpp:50-57@73c45a7c`). The FP32 reconstruction input
replaces the pressure row with $(p_0 - G) + \delta p$, rounded to FP32 (`:58-60`). For a column that is uniform along
the sweep, $G$ equals the column's value exactly in every block. Otherwise $|p_0 - G|$ is at most half the range.
A mean, by contrast, is a rounded sum over the block's own cells: it can be one ulp off by block extent and make the
FP32 face states depend on the decomposition (`:50-55`).

### Numerical method

An FP64 run takes the plain path: reconstruct `w`, then post the face floors (`src/hydro/sweep.cpp:45-49@73c45a7c`).
In FP32, after the gauged reconstruction:

- **An active horizontal sweep with `flux-covariance` off** (`:70-89`). It adds the gauge back in FP64, floors
  pressure and density, and subtracts the static face gauge, leaving FP32 face perturbations for the Riemann solve.
- **Every other sweep** (`:90-102`). It converts the face states to FP64, adds the gauge back, and floors both sides.
  The covariance consumers need total face states (`:71`).

In both paths the solve gets FP32 states, and `forward_perturbations` runs on them (`:130-131`). The face pressure
the solve returns gets the background added back, so its consumers see totals (`:133-135`).

The HLLC kernel has FP32 branches. For the contact wave, FP32 forms the side velocities relative to the outer wave
speeds directly, as $c_L q_L$ and $-c_R q_R$, and does not subtract large velocities. FP64 keeps its established
arithmetic (`src/riemann/riemann_kernels.h:109-118@73c45a7c`). The flux step simplifies only its acoustic branches
(`:136-143`).

`TensorCastCache::get` returns the source tensor when device and dtype match. Otherwise it returns a contiguous
converted copy, and reuses it until the source's identity, version, data pointer, device, dtype, sizes or strides
change, or the copy itself is written (`src/utils/tensor_cast_cache.hpp:19-46@73c45a7c`). Inference tensors,
tensors that need gradients, and inference mode bypass the cache (`:23-27`). The cache holds the area and volume
conversions of the coordinates and the reconstruction's coefficient tables
(`src/coord/coordinate.hpp:420@73c45a7c`; `src/recon/reconstruct.hpp:72@73c45a7c`).

### Code

| what | code at the pin | symbol |
|---|---|---|
| the gauged FP32 reconstruction | `src/hydro/sweep.cpp:37-103@73c45a7c` | `reconstruct_faces` |
| the FP32 Riemann call | `src/hydro/sweep.cpp:105-136@73c45a7c` | `riemann_solve` |
| HLLC FP32 branches | `src/riemann/riemann_kernels.h:109-118@73c45a7c`, `:136-143` | `hllc_impl` |
| the cast cache | `src/utils/tensor_cast_cache.hpp:17-51@73c45a7c` | `TensorCastCache` |

### Tests

| test (ctest name) | what it asserts | tolerance | source |
|---|---|---|---|
| `test_fp32_typed_constants.release` | the WENO3/5 and LMARS float paths match a float-only reference bit for bit (within `kUlps` when the library fuses multiply-adds); the uniform `weno5` table holds its weights in its dtype; the FP64 paths keep their formulas | bitwise, or `kUlps` | `tests/test_fp32_typed_constants.cpp:1-19@73c45a7c`, tests at `:255-342` |
| `test_fp32_horizontal_sweep_python` | the covariance-off FP32 $x_2$ and $x_3$ sweeps track the FP64 path over 20 steps; activity `> 1.0e-3` | `VELOCITY_TOL = 5.0e-5` m/s, `FIELD_TOL = 2.0e-5` | `tests/test_fp32_horizontal_sweep.py:1-7@73c45a7c`, `:21-24`, `:78-91` |
| `test_fp32_compute_python` | under `float32` `hydro_u` is FP32 and `hydro_w` FP64; the moist evolution is accepted | see the source | `tests/test_fp32_compute.py:1-20@73c45a7c` |

**Run evidence at the pin: missing evidence.**

### Limits

1. The cast cache's aliasing contract: a write that bypasses the version counter (a raw pointer, `set_data` on an
   alias, or a CUDA-graph replay) can leave a stale conversion (`src/utils/tensor_cast_cache.hpp:10-16@73c45a7c`).
2. **Unverified:** the outline's "FP64-accumulated sums" as a general rule. At the pin I found FP64 arithmetic in the
   gauge restore (above) and in the perturbation split (12.3), not a general accumulation rule.

---

## 12.3 Anomaly storage and perturbation flux

### Summary

FP32 cannot hold a hydrostatic pressure and its small increment in one number. So FP32 dynamics store `hydro_u` rows
IDN and IPR as anomalies about a static FP64 reference, and solve for the perturbation flux. The hydrostatic part
then cancels in FP64. The chain is enforced at setup:

- `float32` requires `perturbation-flux`;
- `perturbation-flux` requires anomaly storage and HLLC;
- anomaly storage requires the `ideal-moist` EOS and a Cartesian grid.

::: {.callout-note title="At a glance"}
**Purpose.** Keep the FP32 state small, so that a perturbation is not rounded away against its background.

**Data.** The stored anomalies and the FP64 reference (`hydro.ref.u0`, `species0`) in; FP64 background and FP32
perturbation density and pressure out.

**Placement.** At every hydro call that has `perturbation-flux`, before the reconstruction.

**Activation.** `dynamics/anomaly: true` (or `SNAP_ANOMALY=1`) and `dynamics/perturbation-flux: true`. FP32
requires both.

**Guarantee / limits.** In FP64, anomaly storage is an exact rewrite. It is limited to Cartesian `ideal-moist`,
with no immersed solids and no MPS.

**Evidence.** `test_anomaly_storage_python`, `test_perturbation_flux_python`, `test_perturbation_flux_split_python`.
:::

### Derivation

Let $u_{\mathrm{ref}}$ be the reference rows {IDN, IPR} and $u$ the conserved total. The stored rows are
$u' = u - u_{\mathrm{ref}}$ (`HydroImpl::sub_reference_`, `src/hydro/hydro.cpp:626-631@73c45a7c`). `total_u`
promotes an FP32 $u'$ to FP64 *before* it adds the reference. Rounding the total first could not recover the
increment (`:633-638`).

`IdealMoistImpl::perturbations` splits the state into background and perturbation in FP64. Index $k$ runs over the
mass rows, the dry row and then each species (`src/eos/ideal_moist.cpp:79-157@73c45a7c`):

- $\rho_0$ is the reference dry density and the reference species, and $\delta\rho$ is the stored dry anomaly and
  the species minus their reference (`:96-97`).
- $c_{v,0} = \sum_k \rho_{0,k} c_{v,k}$ and $\delta c_v = \sum_k \delta\rho_k c_{v,k}$, likewise for the gas
  constant, with the cloud rows' gas constant zero (`:101-124`, `:134-135`).
- The background pressure is $p_0 = e_0\, R_0 / c_{v,0}$, where $e_0$ is the reference energy row minus
  $\sum_k \rho_{0,k}$ times the energy offsets, and $R_0 = \sum_k \rho_{0,k} R_k$ (`:132`, `:134`, `:152`).
- The pressure increment comes from kintera's `eval_const_cv_pressure_delta`, called with the background, the
  energy increment and the capacity increments (`:153-154`). Its body is not read here.
- $p_0$ and $\rho_0$ are returned in FP64, and $\delta\rho$ and $\delta p$ in the stored dtype (`:155-156`).

The split requires an anomaly background, the original species background and constant heat capacities
(`:81-89`). It also requires finite positive background and total heat capacities. The hydro step defers that check:
a failure latches the `EfStatus` heat-capacity flag, which rejects the step (`:137-151`).

### Numerical method

The hydro step calls `perturbations` when `perturbation-flux` is on. It refuses immersed solids and a primitive that
is not FP64 (`src/hydro/hydro_forward.cpp:353-363@73c45a7c`). In FP32 the primitives are FP64 (`kPrimitive`, 12.1),
so the check holds. The sweep then works on $(p_0 - G) + \delta p$ (12.2). Gravity-free and horizontal sweeps
subtract a static face gauge, so the Riemann flux carries the physical pressure increments, and the full gradient of
the gauge stays in the momentum source (`src/hydro/sweep.cpp:120-129@73c45a7c`).

`Fp64StoreImpl` holds the reference. It ignores dtype conversions and honours device moves
(`src/hydro/fp64_store.hpp:15-22@73c45a7c`, `:42-52`), and stores each buffer as FP64 (`:68-70`). Its `to(device)`
refuses MPS: "anomaly storage keeps its reference in float64, which MPS does not have" (`:49-50`).

### Code

| what | code at the pin | symbol |
|---|---|---|
| FP32 requires perturbation-flux | `src/hydro/hydro.cpp:38-39@73c45a7c` | `HydroImpl::reset` |
| anomaly: `ideal-moist` only, Cartesian only | `src/hydro/hydro.cpp:136-149@73c45a7c` | `HydroImpl::reset` |
| perturbation-flux: anomaly and HLLC | `src/hydro/hydro.cpp:152-155@73c45a7c` | `HydroImpl::reset` |
| the key and its environment override | `src/options/hydro.yaml:33-41@73c45a7c` | `anomaly` (`env: SNAP_ANOMALY`) |
| the FP64 holder | `src/hydro/fp64_store.hpp:23-80@73c45a7c` | `Fp64StoreImpl` |
| the split | `src/eos/ideal_moist.cpp:79-157@73c45a7c` | `IdealMoistImpl::perturbations` |
| the call | `src/hydro/hydro_forward.cpp:353-363@73c45a7c` | `HydroImpl::forward` |

### Tests

| test (ctest name) | what it asserts | tolerance | source |
|---|---|---|---|
| `test_anomaly_storage_python` | the FP64 reference exists and survives `block.to(float32)`; on vs off agree to round-off; the anomaly keys; `SNAP_ANOMALY=1` turns it on and an unsupported EOS refuses it; restart bit for bit; FP32 stored anomalies keep dtype and bits; the limiter clips in anomaly space to the same totals | round-off; bitwise | `tests/test_anomaly_storage.py:2-22@73c45a7c` |
| `test_perturbation_flux_python` | full RHS, unbalanced rest, heating, conversion and VIC with `perturbation-flux` | `2.e-13`, `2.e-10`, `2.e-9` | `tests/test_perturbation_flux.py:129-156@73c45a7c` |
| `test_perturbation_flux_split_python` | the coupled scalar and species seam checks with anomaly, HLLC and WENO5 | pass/fail | `tests/test_perturbation_flux_split.py:1-18@73c45a7c` |
| `test_mps_matches_cpu_python` | a deck with anomaly storage is refused on MPS with "MPS does not have" | string match | `tests/test_mps_matches_cpu.py:251-257@73c45a7c` |

**Run evidence at the pin: missing evidence.**

### Limits

1. Cartesian only: the curved grids interpolate ghosts, so the reference's ghosts would not match
   (`src/hydro/hydro.cpp:141-148@73c45a7c`). FP32 on the cubed sphere is therefore refused (17.1).
2. `ideal-moist` with constant heat capacities only (`:136-140`; `src/eos/ideal_moist.cpp:87-89@73c45a7c`).
3. No immersed solids with `perturbation-flux` (`src/hydro/hydro_forward.cpp:356-358@73c45a7c`).
4. No MPS (`src/hydro/fp64_store.hpp:49-50@73c45a7c`).
5. kintera's pressure increment is cited by name only.

---

## 12.4 The FP32 implicit solve (pointer to 6.3)

### Summary

The vertical implicit solve takes its dtype from the policy's `kImplicit` row: it computes in the row's compute
dtype, and its FP64 stage is the row's storage. When the two differ and `implicit-fallback-fp64` is on, the solve
retries in FP64 only the columns that failed. In FP32 the block-tridiagonal elimination scales each block row by a
power of two before it pivots. Chapter 6.3 owns the solve. This section keeps only its precision rules.

::: {.callout-note title="At a glance"}
**Purpose.** An FP32 vertical solve that stays at the FP32-dynamics error, with an FP64 retry per failed column.

**Data.** The policy's `kImplicit` row and `implicit-fallback-fp64` in; the solve's dtype and retry dtype out.

**Placement.** `HydroImpl::prepare_implicit` before the stage; `_apply_implicit_correction` in it.

**Activation.** An implicit scheme with `dynamics/precision: float32`, or `integration/implicit-precision: float32`.

**Guarantee / limits.** An FP32 solve requires Cartesian `ideal-moist`.

**Evidence.** `test_fp32_vic_pivot_python`, `test_fp32_implicit_python`, `test_fp32_restart_python`.
:::

### Derivation

None here (6.3 derives the solve). The precision rules:

- The solve's dtype is `row.compute`. The retry dtype is `row.storage`, used only when the two differ and
  `fallback_fp64()` is true (`src/hydro/hydro.cpp:421-429@73c45a7c`).
- In `_apply_implicit_correction`, an FP32 solve ("narrower than its FP64 stage") requires `ideal-moist` on a
  Cartesian grid (`:457-463`).
- On the boundary (FP32 dynamics or an FP32 solve), `du` is taken to the row's storage dtype. An FP32 solve keeps a
  storage-dtype copy of the right-hand side and of `wi`, and solves on FP32 copies (`:464-471`).
- On MPS both rows are FP32, so no retry is set up (`:458`, and 12.1).

### Numerical method

For FP32 and block size $N > 4$, the forward sweep scales each block row by a power of two of its $[B\ A\ C]$
maximum, then pivots on the scaled rows. FP64 keeps `ludcmp`'s implicit diagonal-block scaling
(`src/implicit/forward_sweep_impl.h:47-58@73c45a7c`, `:81-82`, `:98`). The predicated FP64 stage solves only the
columns in `retry`. Every other column carries `kVicSkipColumn` (`src/implicit/implicit_hydro.cpp:338-342@73c45a7c`).
A failed retry sets `kVicFp64Failed` (`src/implicit/implicit_dispatch.hpp:21@73c45a7c`).

### Code

| what | code at the pin | symbol |
|---|---|---|
| the solve and retry dtypes | `src/hydro/hydro.cpp:421-429@73c45a7c` | `HydroImpl::prepare_implicit` |
| the FP32 solve's refusal and boundary | `src/hydro/hydro.cpp:431-471@73c45a7c` | `HydroImpl::_apply_implicit_correction` |
| block-row scaling | `src/implicit/forward_sweep_impl.h:47-58@73c45a7c` | `rowscale` |
| the FP64 retry stage | `src/implicit/implicit_hydro.cpp:338-342@73c45a7c` | `ImplicitHydroImpl` |

### Tests

| test (ctest name) | what it asserts | tolerance | source |
|---|---|---|---|
| `test_fp32_vic_pivot_python` | FP32 vs FP64 solve under FP32 dynamics on a 32-column moist set; the docstring records a $w$ gap of 1.0e-3 without block-row scaling and 2.9e-6 with it | see the source | `tests/test_fp32_vic_pivot.py:1-10@73c45a7c` |
| `test_fp32_implicit_python` | FP32 VIC accuracy at large dt, FP64 recovery, and rejection without it | see the source | `tests/test_fp32_implicit.py:1@73c45a7c` |
| `test_fp32_restart_python` | versioned anomaly restart; exact same-dtype and cross-precision continuation | bitwise; see the source | `tests/test_fp32_restart.py:1@73c45a7c` |

The two gap values in the `test_fp32_vic_pivot` row are the test's own docstring, not a run of mine.
**Run evidence at the pin: missing evidence.**

### Limits

1. FP32 solve: Cartesian `ideal-moist` only (`src/hydro/hydro.cpp:460-463@73c45a7c`).
2. The implicit solve refuses an $x_1$ decomposition (`:442-446`; 17.1).
3. The tolerances of the three tests were not re-read line by line for this draft.
