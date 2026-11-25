# snapy Technical Report: structure proposal for the V3 update

> Status: structure **approved** at `0a802f6a`; the section outline is in this file (sections 6 and 7), **for
> refutation** before any writer starts. Binding style: `STYLE.md` (every scheme section keeps the six layers:
> Summary, Derivation, Numerical method, Code, Tests, Limits).
>
> Code pin: snapy `v3/final` = `4c023985af5f80cd6b20ce9d5f9b6acfc7873d6a` (UCzhangxi/snapy). It lands the closing batch on
> `939f7928398f302080cc34aee6df059e883e830d`: #316, the execution default, the wall-batch slots with the MPS
> wide-dtype operands, and the EfStatus meters (12 commits, patch-identical to the checked commits but for one
> mechanical merge in `tests/CMakeLists.txt`). The structure was approved at `70f6e15e`; every anchor in this file,
> section 6 included, is at `4c023985`.
> V3 lane map: `docs/v3/DESIGN.rst` on happysky19/snapy `v3-design/DESIGN` at
> `449636b3e1963b5fc7d44407cb7d6db437e8b016`, Table 1 (lane heads, lines 36-80) and Table 7 (lane scopes and
> skeletons, lines 209-275). Lanes: C reconstruction/Riemann/well-balanced reference; R flux register, monitor,
> hydro stage; EF EOS, floors, positivity, tracers, sedimentation; A rotation, axial momentum, sources,
> diffusion; S seams, coordinates, layout, exchange, boundaries; I vertical implicit solver, gravity work;
> X execution (precision, CUDA graphs, mesh, integrator); O output, input, diagnostics, options schema;
> K kintera thermodynamics and chemistry contracts. Closing work named here: B3 (multi-block capture, exchange
> cuts, mesh-agree), S4 (eager-path launch reduction, spread over the chapters that own the kernels it changed; the execution
default), seams (panel-seam register and
> rotation cache), capture (the capture rules and allow-lists).

## 1. Principle

The pre-V3 book follows the physics of one step and scatters the machinery that runs it (stages, retries,
budgets, GPU) over chapters 7, 13 and 14. V3 makes that machinery explicit and shared: every stage runs through
one stage executor (`src/mesh/stage_executor.*`, `segment.*`), every face flux goes through a flux register
with a monitor (`src/hydro/flux_register.*`: one register per module, the hydro's and the scalar's, sharing one
wall batch per block), state lives in fixed storage (`src/mesh/state_store.hpp`), dt is a
device value in a captured stage (`src/utils/step_dt.*`), and precision is one policy
(`src/mesh/precision_policy.hpp`). The proposal therefore has four parts: **I. the framework** (what a step is
and what it runs on), **II. the spatial operator** (lane by lane, in the order a stage calls it), **III. time
stepping and bookkeeping**, **IV. execution, parallelism and I/O**, then **V. verification and limits**. Each
chapter is owned by one lane where possible, so a chapter's sources are one lane's design document plus its
code.

## 2. Proposed chapters

| # | proposed title | status | lanes | why |
|---|---|---|---|---|
| 1 | Overview: the V3 step and the code map | kept, rewritten (old 1) | all (map) | readers need the step anatomy (stage, segments, register, exchange) before any scheme |
| 2 | Governing equations and thermodynamics | kept (old 2) | EF, K | EOS and the kintera contracts are the equations every later chapter discretises |
| 3 | Grids, geometry and cubed-sphere seams | kept, extended (old 3; seam material from old 5 and 14.A) | S, seams | the panel-seam register and its rotation are geometry; one home for all seam math |
| 4 | Reconstruction, Riemann solvers and the hydro sweep | kept (old 4.A and 4.B) | C, R (sweep), S4 | C owns reconstruction and solvers; the sweep is where they meet the register; S4's floor and event count posts live here |
| 5 | Hydrostatic reference and well-balanced treatment | kept (old 5, seam parts moved to 3) | C | unchanged scope, lane C |
| 6 | Gravity work and the vertical implicit correction | merged (old 6 + old 7.B) | I, S4 | lane I owns both; the VIC energy row is the face gravity work; S4's status truth table lives here |
| 7 | Rotation, axial momentum, diffusion and forcing | kept (old 9, Coriolis and axial momentum gathered) | A | lane A's scope as one chapter |
| 8 | Moist physics, positivity, floors and limiters | merged (old 8 + old 10) | EF, K | EF owns EOS, floors, positivity, tracers and sedimentation together; the positivity census is the limiter's own record |
| 9 | Time integration: RK stages, dt and step acceptance | split (from old 7.A) | X (integrator), I (FP64 retry flag), S4 | the stage loop and redo decision, without the bookkeeping now in 10; S4's padded tracer-transfer shifts live here |
| 10 | The flux register, monitor amounts and conservation budgets | merged (old 13 + new register material) | R, O (budget series), S4 | budgets are read off the register in V3; one home for posts, prescribed faces, the wall batch, amounts, monitor counts and their rollback (order in section 4.1) |
| 11 | Execution: the stage executor and CUDA graphs | new | X, B3, capture, S4 (catalogue and launch accounting) | the central V3 addition; one home for the execution model, the refusal list and the graph rules; of S4 it holds a short cost catalogue that points to the code folds in 4/6/9/10, and the measured eager-graph gap, cited (section 4.2); a redo replays the captured stages, and how the register's rollback survives that is a pointer to 10 |
| 12 | Precision: the FP32 policy | new | X (policy), I, EF, C (their rows) | one policy table governs every FP32 decision; today it would be scattered |
| 13 | Mesh, layout, exchange and multi-block runs | split (from old 14.A) | S, X (Mesh), B3 | layout, exchange and the same-process rendezvous are the Mesh's own model |
| 14 | Boundary conditions and immersed solids | kept (old 11) | S | which faces are prescribed and with what; how the register books them is 10 (pointer to the wall batch in 10) |
| 15 | Configuration, output and restart | merged (old 12 + old 14.B) | O | lane O owns the options schema, writer core and restart together |
| 16 | Verification and reproducibility | merged (old 15 + old 14.C) | all, capture | the bitwise gates are the verification catalogue's own rules; holds the capture sweep and its table, the launch-counting procedure that 11 cites, and the regression provenance (each regression test with the issue and RED/GREEN commits it came from) |
| 17 | Known limits | kept (old 16) | all | the limits themselves, as pointer rows (each names its home chapter); no numbers repeated and no regression history (that is 16's) |
| A | Notation | kept | - | |
| B | Derivation index | kept | - | |
| C | Test index | kept | - | |
| D | Environment-variable switches | kept | X, O | |
| E | External components: kintera, pyharp, pydisort, commux | kept, extended | K | gains the kintera V3 contract surface (K) |
| F | Topic index | kept | - | |
| G | V3 lane map and design-document index | merged (old G task map) | all | replaces the task map with the lane -> chapter -> design-document map |
| H | Glossary | kept | - | adds V3 terms (segment, cut, capture, fold) and two separate entries: "positivity census" (8) and "monitor counts" (10) |
| I | Deck key reference | kept | O | generated from the options schema |
| J | Source tree map | kept, rewritten | all | the V3 tree (mesh split files, register, executor) |

## 3. Where every old chapter goes

| old | goes to |
|---|---|
| 1 Overview and code map | 1 (rewritten around the V3 step) |
| 2 Governing equations and thermodynamics | 2 |
| 3 Grids and geometry | 3 |
| 4.A Reconstruction, Riemann, divergence, sources | 4 (divergence and register posts: 4; source terms: 7) |
| 4.B Face-average corrections | 4 |
| 5 Hydrostatic and well-balanced | 5 (seam relay and panel parts: 3) |
| 6 Gravity and energy | 6 |
| 7.A Explicit stages, time step, step acceptance | 9 (stage loop, dt, redo decision); rollback of amounts: 10 |
| 7.B Vertical implicit correction | 6 |
| 8 Positivity, floors and limiters | 8 |
| 9 Diffusion, viscosity and forcing | 7 |
| 10 Moist physics coupling | 8 (EOS side, saturation, sedimentation); chemistry slot: 11 (as a segment), kintera contract: E |
| 11 Boundary conditions and immersed solids | 14 |
| 12 Switches and configurations | 15 (YAML keys), D (environment), I (deck keys) |
| 13 Conservation budgets and diagnostics | 10 |
| 14.A Parallel execution and GPU | 13 (layout, exchange, Mesh, ranks); GPU execution and graphs: 11 |
| 14.B Output and restart | 15 |
| 14.C Reproducibility | 16 |
| 15 Verification catalogue | 16 |
| 16 Known limits | 17 |
| Appendices A-F, H-J | kept (A-F, H-J); G task map is replaced by the lane map (G) |

Nothing is dropped. Old 4.A's source-term entries and old 10's chemistry material are the only splits across
more than one new chapter; each piece has the single home given above.

## 4. One home for each new V3 topic

| topic | home | code at the pin |
|---|---|---|
| stage executor, segments, execution model | 11 | `src/mesh/stage_executor.*`, `src/mesh/segment.*` |
| capture rules and allow-lists (EOS ideal-gas/ideal-moist, Riemann lmars/hllc), the refusal list with its named reasons | 11 | `StageExecutor::config_reasons` in `src/mesh/stage_executor.cpp` (lines 579-643 at the pin) |
| capture sweep (which shipped tests capture, and the table of results) | 16 | the sweep's committed record, cited when it lands |
| S4 cost catalogue: one short list of the eager-path folds, each pointing to its home (4, 6, 9, 10), plus the counting method's home (16) | 11 | pointers only; the code is described in the home chapters |
| the execution default (cuda-graph non-strict on CUDA, eager on CPU, explicit eager honoured) | 11.4 | `StageExecutor::mode` in `src/mesh/stage_executor.cpp` (lines 107-115); see H1 |
| graph = eager bitwise rule; why eager is launch-bound and graph faster | 11 | launches per step and timings as measured, cited from their committed source (see 4.2); no H100 headline |
| launch-counting procedure (profiler trace; kernel, memcpy and memset events over N steps) | 16 | `docs/v3/closing/tools/speed/bench.py` (`--profile`, `SPEED_PROF_TRACE`) at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan) |
| graph-safe coding (no host reads, no host-to-device copies inside a captured segment, fixed shapes, device dt, host state replay does not refresh) | 11.5 | `src/utils/step_dt.*`, `src/mesh/state_store.hpp`; the copy rule holds at the pin for the stretched x1 reference, whose rows are built on the grid's device (`src/recon/x1_reference.hpp:62-66`, #316), see 4.3 |
| mesh-agree (a Mesh whose blocks disagree runs eager) | 13 | `StageExecutor::agree` in `src/mesh/stage_executor.cpp` |
| exchange cuts and multi-block capture (B3) | 13 | `src/mesh/exchange_cut.hpp`, `src/layout/exchange.*` |
| flux register and monitor amounts: posts, prescribed faces, wall batch, stage fold, rollback/redo | 10 | `FluxRegister`, `RegisterMonitor`, `WallBatch` in `src/hydro/flux_register.hpp` |
| monitor counts (face floors, `implicit.fp64_columns`, events) and their rollback | 10 | `RegisterMonitor::count`, `flush_counts`, `rollback` |
| rollback through graph replay (the redo replays the captured stages; the host stage bookkeeping survives `FluxRegister::rollback`) | 10, pointer from 11 | `FluxRegister::rollback` in `src/hydro/flux_register.cpp`; `StageExecutor::replay` in `src/mesh/stage_executor.cpp` |
| prescribed-face reapplication after the implicit correction | 10 | `FluxRegister::apply_prescribed`; the call in `HydroImpl::forward` (`src/hydro/hydro_forward.cpp`) |
| the settling carry on a prescribed face (#315) | 8, pointer from 10 | `flux_positivity_carry_` in `src/hydro/flux_positivity.cpp` |
| regression provenance (which issue and RED/GREEN commits each regression test pins) | 16 | the tests' own headers and the issues they cite |
| seam rotation cache | 3 | `SeamRotationCache` in `src/layout/seam_register.hpp` |
| positivity census (EfStatus `kPosHits`, `kPosSevere` (theta < 0.9 and withheld above round-off, #256), `kPosMin`, `kScalarPosHits`) | 8.3 | `PositivityLimiter::census_` in `src/hydro/flux_positivity.cpp` |
| the EfStatus slots (marks, vapor failures, heat-capacity latch, meters) | 8.6 | `EfStatus` in `src/eos/ef_status.hpp`; saved across capture warm-ups by `StageExecutor::save_list` |
| the device dt (the executor's per-block 0-d FP64 dt and `StepDt`) | 11.5; pointers from 6.4 and 11.2 | `StageExecutor::device_dt`, `StepDt` in `src/utils/step_dt.*` |
| eager ops inside a captured stage (user forcing, chemistry segment) | 11.7 | `StageExecutor::build_schedule`, `StageExecutor::chemistry_slot` |
| MPS precision (wide dtype FP32, no FP64) | 12.1; MPS limits as rows in 17 | `PrecisionPolicy` in `src/mesh/precision_policy.hpp` |
| S4 status truth table | 6 | `ImplicitHydroImpl::status` in `src/implicit/implicit_hydro.cpp` |
| S4 padded tracer-transfer shifts (`r_below`, `P_above`) | 9 | `MeshBlockImpl::stage_advance` in `src/mesh/meshblock_advance.cpp` |
| S4 floor and event count posts | 4 | `src/recon/reconstruct.*`, `src/riemann/riemann.*`, `src/hydro/sweep.cpp` |
| FP32 precision policy | 12 | `PrecisionPolicy` in `src/mesh/precision_policy.hpp` |
| redo decision and step acceptance | 9 | `src/mesh/meshblock_redo.cpp` |

Every row was found by `git grep` on `70f6e15e` and again at the pin `4c023985` (file and symbol exist). Their behaviour is not
re-read here; that is the section outline's job.

### 4.1 Chapter 10's order and test layer

Chapter 10 covers, in this order: the posts (`add`, `scale`, `overwrite_`; plain adds of the rows, no
`index_add_` at the pin); prescribed faces (`apply_prescribed`, the written mark `written_since_prescribed`, and the
post-implicit reapplication, which runs when the implicit correction ran or the mark is set; the settling
carry that sets the mark, #315, is chapter 8's and is pointed to from here); the wall batch (one `WallBatch` per block, shared by the hydro
and scalar registers, flushed in the stage combine); the amounts (FP64, FP32 on MPS; the stage fold); rollback
and redo (the device slots and counters are zeroed, the host stage bookkeeping is kept, because a graph replay
does not rerun `begin_stage`/`end_stage`); and the budget series O reads after the step is accepted. Its Tests
layer cites `test_prescribed_after_sources`, `test_flux_positivity_carry` and `test_graph_monitor_amounts`
(including `GraphSaveList.monitor_amounts_match_eager_after_a_rollback`). Chapter 14 says which faces are
prescribed and with what; chapter 10 says how the register books them.

### 4.2 Measured numbers

The S4 table (main eager, the default, explicit eager; ms/step and launches/step, with the GPUs and their notes)
and the bound as ruled are committed in `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51`
(`v3/closing-dryrun-plan`), not in this report. Chapter 11 (11.6) cites that file at that commit for the numbers and
the bound, chapter 16 (16.3) cites it for the procedure's results, and neither repeats a number of its own. The
counting procedure itself (profiler trace; kernel, memcpy and memset events over N steps, with
`docs/v3/closing/tools/speed/bench.py` at the same commit) is described in 16.3.

### 4.3 Writing around a fix that has not landed

#316 has landed on `v3/final` (RED `213189c4`, fix `069769ba`): the stretched x1 reference rows are built on the
grid's device (`src/recon/x1_reference.hpp:62-66`), so no host-to-device copy runs inside a captured segment.
Chapter 11 (11.5) states the rule and cites that mechanism; the chapter 17 row for the stretched-x1 capture is
closed. The rule stays: no chapter is written against a fix that has not landed on the pin.

### 4.4 The pin

One pin per package (STYLE.md 3.1, `code_pin.yml`): the whole report moves to the commit that lands the
execution default, by `tools/code_anchors.py bump`; chapter 11 is not pinned separately. That commit is
`v3/final` = `4c023985`, and this file is pinned there. From `939f7928` to `4c023985` the outline's anchors in
`src/mesh/stage_executor.cpp` move by +20 to +37 (e.g. `StageExecutor::config_reasons` `:558` is `:579`), those in
`src/hydro/flux_register.cpp` by +15 to +87 where they move, and those in `src/mesh/meshblock_advance.cpp` (after
line ~160) and `src/mesh/meshblock_options.cpp` (after line ~58) by +1; no other file's anchors move. Section 6
gives every anchor at the pin.

## 5. Hypotheses (not checked in code)

- **H1** (confirmed in code at the pin). The S4 default (cuda-graph, non-strict, on CUDA; eager on CPU; an
  explicit `eager` honoured) is on `v3/final` at `4c023985af5f80cd6b20ce9d5f9b6acfc7873d6a` (RED `53c46723`, fix `02e4ef05`,
  patch-identical to `ea3e5777`/`8854ccd4`, whose non-author check passed on CUDA sm_61 and CPU):
  `integration.execution` defaults to `auto` (`src/mesh/meshblock.hpp:98`), which `StageExecutor::mode`
  (`src/mesh/stage_executor.cpp:107-115`) resolves to cuda-graph on a CUDA block and eager otherwise. Before it, at
  `70f6e15e` and `939f7928`, the default was `eager`. Chapter 11 is written against this pin and cites the bound as
  ruled from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b70` (section 4.2).
- **H2.** Old 10's chemistry material splits cleanly into the chemistry segment (11) and the kintera contract
  (E); not checked against the old chapter's sections.
- **H3** (resolved in review; two meanings). The positivity census (`PositivityLimiter::census_`,
  `src/hydro/flux_positivity.cpp:207`) writes the EfStatus slots `kPosHits`, `kPosSevere`, `kPosMin`
  (`src/eos/ef_status.hpp:27-29`) and, for scalars, `kScalarPosHits` (`:32`); at the pin they are read by the
  budget diagnostics (`src/diagnostics/budget.cpp:345-346`, through `HydroImpl::positivity_*`,
  `src/hydro/hydro.cpp:280-288`), not by `check_redo`, whose limiter cause is the EOS `limiter_marks()`
  (`src/mesh/meshblock_redo.cpp:118-120`). It lives in 8. The "census" of `src/mesh/meshblock_redo.cpp:56` is
  the monitor's counters (`face_floor.*` from `src/hydro/sweep.cpp:78-90`, `implicit.fp64_columns` from
  `src/hydro/hydro.cpp:492`), zeroed by `RegisterMonitor::rollback()` (`src/hydro/flux_register.cpp:46-65`);
  they live in 10.
- **H4.** Precision warrants its own chapter (12): assumed from the single `PrecisionPolicy` class and the
  number of lanes with FP32 rows; the size of the material is not measured.
- **H5.** The lane owning each new chapter matches the lane scopes in DESIGN.rst Table 7 (confirmed in review
  for R and S4); ownership of the other closing work (B3, seams, capture) is taken from the integration
  history, not a document.

## 6. Section outline

Code anchors are `path:line` at snapy `v3/final` = `4c023985af5f80cd6b20ce9d5f9b6acfc7873d6a` (function or class name in the same cell). That head lands the closing batch on `939f7928` (#316, the execution default, the wall-batch slots and the MPS wide-dtype operands, the EfStatus meters), so no section carries a re-pin marker; what the batch changed is in the sections' Code and Tests lines at this pin. Anchors, test files, ctest names and tolerances were resolved by reading the code at that commit; tolerances are quoted from the test source ("bitwise" where the test compares with `torch.equal`/`torch::equal`); `allclose`/`isclose`/`assert_allclose` calls are read argument by argument (positional rtol/atol included, the library default stated where none is given), and a Tests cell lists every tolerance found in that file, so the writer quotes the one on the assertion the section relies on. Figures marked **new** are to be written; the others are existing scripts under `doc/tech-report/`, to be re-checked at the pin. A section with no test or no figure says so.

### Chapter 1. Overview: the V3 step and the code map

#### 1.1 The V3 step: Mesh, block, stage, segments

- **Derivation:** none (control flow): one step = rk3 stages; each stage = the block's segment schedule run eagerly or replayed
- **Code:** `src/mesh/mesh.cpp:367` `MeshImpl::forward`; `src/mesh/meshblock_advance.cpp:88` `MeshBlockImpl::forward`; `src/mesh/stage_executor.cpp:778` `StageExecutor::run_stage`; `src/mesh/segment.cpp:40` `StageSchedule::add`
- **Tests:** `test_mesh_executor.release` (bitwise (equal x2)); `test_stage_schedule.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch01/figures/fig_stage_flow.py` (deck: none (diagram)); **new** `src/snapy_report/ch01/figures/fig_v3_step.py` (deck: none (diagram))

#### 1.2 Construction and dependencies

- **Derivation:** none (control flow): options -> MeshBlock -> modules (hydro, scalar, coord, EOS, kintera thermo)
- **Code:** `src/mesh/meshblock.cpp:107` `MeshBlockImpl::reset`; `src/hydro/hydro.cpp:21` `HydroImpl::HydroImpl`
- **Tests:** `test_uniform_module_tree.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch01/figures/fig_construction_flow.py` (deck: none (diagram)); `src/snapy_report/ch01/figures/fig_dependencies_flow.py` (deck: none (diagram))

#### 1.3 Code map

- **Derivation:** none: directory -> lane -> chapter table (lane scopes from DESIGN.rst Table 7)
- **Code:** none in `src/` (method or record section)
- **Tests:** none (no test covers this section at the pin)
- **Figures/tables:** `src/snapy_report/ch01/figures/fig_overview_dependencies.py` (deck: none (diagram))

### Chapter 2. Governing equations and thermodynamics

#### 2.1 Conservation laws and variables

- **Derivation:** Euler equations for (rho, rho v, E, rho q_i); conserved <-> primitive maps; the anomaly form u - u_ref
- **Code:** `src/eos/equation_of_state.cpp:222` `EquationOfStateImpl::compute`; `src/eos/equation_of_state.cpp:297` `EquationOfStateImpl::forward`
- **Tests:** `test_eos.release` (bitwise (equal x15); allclose(rtol=1.E-6, atol=1.E-6) x5; allclose(rtol=1.E-6, atol=1e-08 (default)); allclose(rtol=2.e-6, atol=1.e-10); allclose(rtol=1.e-14, atol=1.e-8); allclose(rtol=2.e-6, atol=1.e-7); allclose(rtol=2.e-6, atol=1.e-12); allclose(rtol=2.e-7, atol=1.e-10) x2; allclose(rtol=1.e-6, atol=1.e-8); allclose(rtol=1.E-12, atol=1.E-12) x2; allclose(rtol=1.E-2, atol=1.E-2); allclose(rtol=1.E-9, atol=1.E-9); allclose(rtol=1.E-4, atol=1.E-4); EXPECT_NEAR 1.e-25); `test_anomaly_storage_python` (bitwise (equal x7); constants tol=1.0e-10)
- **Figures/tables:** `src/snapy_report/ch02/figures/fig_laws_flow.py` (deck: none (diagram))

#### 2.2 Ideal-gas EOS

- **Derivation:** p = rho R T, e = cv T; cons2prim from E - KE
- **Code:** `src/eos/ideal_gas.cpp:73` `IdealGasImpl::_cons2prim`; `src/eos/ideal_gas.cpp:54` `IdealGasImpl::_prim2cons`
- **Tests:** `test_eos.release` (bitwise (equal x15); allclose(rtol=1.E-6, atol=1.E-6) x5; allclose(rtol=1.E-6, atol=1e-08 (default)); allclose(rtol=2.e-6, atol=1.e-10); allclose(rtol=1.e-14, atol=1.e-8); allclose(rtol=2.e-6, atol=1.e-7); allclose(rtol=2.e-6, atol=1.e-12); allclose(rtol=2.e-7, atol=1.e-10) x2; allclose(rtol=1.e-6, atol=1.e-8); allclose(rtol=1.E-12, atol=1.E-12) x2; allclose(rtol=1.E-2, atol=1.E-2); allclose(rtol=1.E-9, atol=1.E-9); allclose(rtol=1.E-4, atol=1.E-4); EXPECT_NEAR 1.e-25); `test_eos_temp2inteng_python` (compared < 1.0e-12, 1.0e-10)
- **Figures/tables:** `src/snapy_report/ch02/figures/fig_ideal_gas_flow.py` (deck: none (diagram))

#### 2.3 Ideal-moist EOS and the kintera thermodynamics

- **Derivation:** mixture cv and molecular weight from species fractions; internal-energy offset; T from (rho, e) by the kintera ThermoY contract
- **Code:** `src/eos/ideal_moist.cpp:202` `IdealMoistImpl::_cons2prim`; `src/eos/ideal_moist.cpp:65` `IdealMoistImpl::internal_energy_offset`; `src/eos/ideal_moist.cpp:79` `IdealMoistImpl::perturbations`
- **Tests:** `test_eos_species_sum_layout_python` (no numeric tolerance in the source (pass/fail checks only)); `test_eos_temp2inteng_python` (compared < 1.0e-12, 1.0e-10); `test_sk1_failure_counter_python` (bitwise (equal x1))
- **Figures/tables:** `src/snapy_report/ch02/figures/fig_ideal_moist_flow.py` (deck: none (diagram))

#### 2.4 Other equations of state (moist-mixture, aneos, shallow-water)

- **Derivation:** their closures in one paragraph each; capture refuses them (pointer to 11.3)
- **Code:** `src/eos/moist_mixture.cpp:48` `MoistMixtureImpl::compute`; `src/eos/shallow_water.cpp:14` `ShallowWaterImpl::compute`
- **Tests:** `test_aneos_gamma.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch02/figures/fig_moist_mixture_flow.py` (deck: none (diagram))

### Chapter 3. Grids, geometry and cubed-sphere seams

#### 3.1 Cartesian and stretched x1 grids

- **Derivation:** cell volume and face areas from face positions; stretched x1 widths and their reference rows
- **Code:** `src/coord/coordinate.cpp:552` `CoordinateImpl::cell_volume`; `src/coord/coordinate.cpp:484` `CoordinateImpl::center_width1`
- **Tests:** `test_coordinate.release` (bitwise (equal x4); allclose(rtol=1.e-12, atol=1.e-12) x11; allclose(rtol=1e-05 (default), atol=1e-08 (default)) x5; allclose(rtol=1.e-4, atol=1.e-5) x5; allclose(rtol=1.e-8, atol=1.e-8); allclose(rtol=1.e-6, atol=1.e-6) x4; allclose(rtol=2.e-5, atol=2.e-5); EXPECT_NEAR 1.e-15); `test_stretched_x1.release` (bitwise (equal x12); EXPECT_NEAR 1e-13 * 16.)
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_cartesian_cell.py` (deck: none (diagram))

#### 3.2 Gnomonic equiangle panels

- **Derivation:** x = tan(alpha), y = tan(beta); delta = sqrt(1+x^2+y^2), C, D; cell volume and face areas from the metric
- **Code:** `src/coord/gnomonic_equiangle.cpp:281` `GnomonicEquiangleImpl::cell_volume`; `src/coord/gnomonic_equiangle.cpp:273` `GnomonicEquiangleImpl::face_area2`; `src/coord/cubed_sphere_utils.cpp:200` `cs_metric_at`
- **Tests:** `test_cubed_sphere_cell_volume_python` (constants REST_TOL=1e-6); `test_radial_face_moments.release` (EXPECT_NEAR 1e-14)
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_gnomonic_cosine.py` (deck: none (analytic))

#### 3.3 Panel rotation and the seam register

- **Derivation:** face fluxes rotated covariant face basis -> Cartesian -> neighbour basis; the owner's rows replace the neighbour's on the seam
- **Code:** `src/coord/coord_rotation.cpp:79` `PanelRotation::face_rows_to_cart_`; `src/layout/seam_register.cpp:36` `reconcile_panel_seams`
- **Tests:** `test_panel_rotation.release` (bitwise (equal x1)); `test_cubed_sphere_seams.release` (bitwise (equal x3))
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_thetacopy_seam.py` (deck: none (diagram))

#### 3.4 The seam rotation cache

- **Derivation:** the rotation operands depend on the grid alone; built once per side, rebuilt when a grid tensor's storage, version, dtype, device or shape changes
- **Code:** `src/layout/seam_register.hpp:41` `SeamRotationCache`; `src/coord/coord_rotation.cpp:72` `PanelRotation::face_operands`
- **Tests:** `test_seam_rotation_cache_python` (bitwise (equal x3)). Test_seam_rotation_cache sets `execution: eager` (it counts the kernels each stage launches)
- **Figures/tables:** **new** `src/snapy_report/ch03/figures/fig_seam_cache.py` (deck: test_seam_rotation_cache's six-panel deck)

### Chapter 4. Reconstruction, Riemann solvers and the hydro sweep

#### 4.1 Reconstruction (PLM, WENO5) and the conversion order

- **Derivation:** piecewise-linear and WENO5 face states from cell averages; fourth-order point/average conversion
- **Code:** `src/recon/reconstruct.cpp:257` `ReconstructImpl::forward`; `src/recon/reconstruct.cpp:119` `InterpImpl::forward`
- **Tests:** `test_plm.release` (allclose(rtol=1e-05 (default), atol=1e-08 (default)); EXPECT_NEAR 1.E-10, 1.E-6, 2.E-6); `test_weno.release` (bitwise (equal x4); EXPECT_NEAR 1.E-10, 1.E-6, 2.E-6); `test_conversion_order_python` (bitwise (equal x4); compared < 1e-13, 1.e-10)
- **Figures/tables:** `src/snapy_report/ch04/figures/fig_recon_stencil.py` (deck: none (diagram)); **new** `src/snapy_report/ch04/checks/recon_order_run.py + fig_recon_order.py` (deck: a 1-D smooth profile through ReconstructImpl at the pin (replaces recon_check.py, a port of src/recon/interp_simple.hpp, deleted by 231dc228))

#### 4.2 Riemann solvers

- **Derivation:** HLLC (three waves) and LMARS (low-Mach pressure and velocity at the face); roe, shallow-roe and upwind in one paragraph
- **Code:** `src/riemann/riemann.cpp:156` `RiemannSolverImpl::forward`; `src/riemann/riemann.cpp:219` `RiemannSolverImpl::solve`
- **Tests:** `test_riemann.release` (bitwise (equal x2); allclose(rtol=1.e-10, atol=1.e-10) x4; allclose(rtol=2.e-6, atol=1.e-12) x2; allclose(rtol=2.e-5, atol=1.e-6); allclose(rtol=2.e-5, atol=1.e-7); EXPECT_NEAR 1.e-7, 1.e-8f; compared < 1e-13); `test_roe_cubed_sphere_rest_python` (compared < 1.0e-9)
- **Figures/tables:** `src/snapy_report/ch04/figures/fig_riemann_face.py` (deck: none (diagram))

#### 4.3 The sweep and its register posts

- **Derivation:** none (control flow): reconstruct -> face floors -> solve -> post to the register; S4: one reduced count post per floor or event that ran (the `post_events` lambda inside `riemann_solve`)
- **Code:** `src/hydro/sweep.cpp:31` `reconstruct_faces`; `src/hydro/sweep.cpp:23` `post_face_floors`; `src/hydro/sweep.cpp:99` `riemann_solve`; `src/recon/reconstruct.cpp:246` `ReconstructImpl::floor_counts`
- **Tests:** `test_face_floor.release` (EXPECT_NEAR 1.e-3, 1.e-5 * 1.28524e-8); `test_fp32_face_floor_counts_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch04/figures/fig_stage_pipeline.py` (deck: none (diagram))

#### 4.4 Face-average corrections at O(dx^2)

- **Derivation:** flux covariance from the in-cell product expansion; x1 centroid shift on curved faces
- **Code:** `src/hydro/hydro_forward.cpp:58` `HydroImpl::_flux_covariance`
- **Tests:** `test_flux_covariance_rows_python` (constants CART_TOL=2e-3, OFFSET_TOL=1e-10, REST_TOL=1e-9, UNIFORM_TOL=1e-13); `test_horizontal_flux_covariance_python` (bitwise (equal x2); constants EP_TOL=1.e-13); `test_x1_centroid_rest_python` (constants TOL_OFF=1.0e-8, TOL_ON=1.0e-10)
- **Figures/tables:** `src/snapy_report/ch04/figures/fig_x1cov_order.py` (deck: data/x1cov_orders.json from ch04/checks/covariance_check.py, a port of _flux_covariance at e894700f and the x1 mass-flux covariance at 6fbb3975; re-port at the pin)

### Chapter 5. Hydrostatic reference and well-balanced treatment

#### 5.1 The x1 hydrostatic reference

- **Derivation:** reference pressure and density from dp/dz = -rho g integrated per column; face states of the reference subtracted before the solve
- **Code:** `src/hydro/hydro.cpp:531` `HydroImpl::_hydro_ref_x1`; `src/recon/x1_reference.hpp:41` `HydroReferenceX1Operator`
- **Tests:** `test_hydro_ref_x1.release` (bitwise (equal x16); allclose(rtol=atol=2e-12 FP64, 2e-5 FP32) through expect_close (tests/test_hydro_ref_x1.cpp:203-204), GPU arm 2e-12 (:267-269); compared < 1e-6); `test_hydro_ref_x1_rf_wall.release` (no numeric tolerance in the source (pass/fail checks only)); `test_stretched_x1_ref.release` (EXPECT_NEAR 1.e-12)
- **Figures/tables:** `src/snapy_report/ch05/figures/fig_refkernel_scan.py` (deck: none (diagram))

#### 5.2 Well-balanced face states

- **Derivation:** face states that keep a hydrostatic column at rest to round-off
- **Code:** `src/recon/well_balanced_x1.cpp:16` `WellBalancedX1::forward`
- **Tests:** `test_hydrostatic.release` (no numeric tolerance in the source (pass/fail checks only)); `test_wb_ref_wall.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch05/figures/fig_wbx1_ghosts.py` (deck: none (diagram))

#### 5.3 Balance column and the fourth-order reference

- **Derivation:** discrete balance column solved for the initial state; fourth-order reference stencils and their order
- **Code:** `src/hydro/balance_column.cpp:18` `balance_column`
- **Tests:** `test_balance_column.release` (bitwise (equal x14); constants rtol=1.e-10); `test_balance_column_order4_python` (compared < 1.e-13, 1.e-12); `test_wb_ref4_order_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch05/figures/fig_balcol_passes.py` (deck: data/bryan_balance_ic.json printed by ctest test_bryan_balance_ic (bryan example, CPU, FP64; recorded at 68f4e066, rerun at the pin)); `src/snapy_report/ch05/figures/fig_wbref4_order.py` (deck: data/wb_ref4_order.json from two runs of tests/test_wb_ref4_order.py (CPU, FP64, recorded at e894700f, rerun at the pin))

### Chapter 6. Gravity work and the vertical implicit correction

#### 6.1 Face gravity work and the corrected potential energy

- **Derivation:** energy source g * mass flux at faces; corrected PE with the exact 3-point slope (derivation sec 7 of the gravity-work note)
- **Code:** `src/forcing/const_gravity.cpp:47` `ConstGravityImpl::forward`; `src/hydro/gravity_work.hpp:217` `corrected_pe_work`
- **Tests:** `test_gravity_work_radial_exact_python` (constants tol=1.e-13; compared < 1e-14); `test_gravity_work_rows.release` (bitwise (equal x1))
- **Figures/tables:** `src/snapy_report/ch06/figures/fig_dwork_stencil.py` (deck: none (diagram)); `src/snapy_report/ch06/figures/fig_face_order.py` (deck: data/face_order.csv from ch06/checks/cellface_check.py, a port of the x1 gravity-work block of hydro_forward.cpp at e894700f; re-port at the pin)

#### 6.2 The VIC: assembly and column solve

- **Derivation:** (I + b dt J') delta = b dt L per column; block tridiagonal LU with the wall third points by Woodbury
- **Code:** `src/implicit/implicit_hydro.cpp:301` `ImplicitHydroImpl::forward_masked`; `src/implicit/vic_kernels.h:176` `vic_solve_column`
- **Tests:** `test_implicit_face_work_jacobian.release` (no numeric tolerance in the source (pass/fail checks only)); `test_implicit_gravity_tall_column_python` (constants W_TOL=1.0e-7); `test_lu_failure.release` (bitwise (equal x14))
- **Figures/tables:** `src/snapy_report/ch07/figures/fig_vic_stencil.py` (deck: none (diagram)); `src/snapy_report/ch07/figures/fig_vic_sweep.py` (deck: none (diagram))

#### 6.3 Column status, the FP64 retry and the S4 status truth table

- **Derivation:** status bits per column; the predicated FP64 stage solves only failed columns; status() as one table lookup (S4)
- **Code:** `src/implicit/implicit_hydro.cpp:223` `ImplicitHydroImpl::status`; `src/hydro/hydro.cpp:436` `HydroImpl::_apply_implicit_correction`
- **Tests:** `test_fp32_vic_pivot_python` (no numeric tolerance in the source (pass/fail checks only)); `test_fp32_implicit_python` (bitwise (equal x1); constants rtol=1.0e-13; compared < 2.0e-6)
- **Figures/tables:** `src/snapy_report/ch07/figures/fig_vic_reject.py` (deck: none (diagram))

#### 6.4 The implicit solve on the device dt (pointer to 11.5)

- **Derivation:** pointer: the solve's dt_corr and I/dt rows read the device dt; the mechanism (StepDt) is derived once, in 11.5
- **Code:** `src/implicit/implicit_hydro.cpp:301` `ImplicitHydroImpl::forward_masked`
- **Tests:** `test_graph_implicit_dt_python` (bitwise (equal x3)); `test_graph_implicit_python` (bitwise (equal x3))
- **Figures/tables:** none

### Chapter 7. Rotation, axial momentum, diffusion and forcing

#### 7.1 Coriolis

- **Derivation:** -2 Omega x v projected on the local frame; the x1 mass-flux w terms from the stage's face transfers
- **Code:** `src/forcing/coriolis.cpp:110` `CoriolisImpl::add`; `src/forcing/x1_momentum.cpp:48` `X1Momentum::after_solve`
- **Tests:** `test_forcing.release` (bitwise (equal x3); allclose(rtol=1e-05 (default), atol=1e-08 (default)) x9; allclose(rtol=1.e-12, atol=1.e-12); allclose(rtol=1.e-13, atol=0.) x2; allclose(rtol=1.e-8, atol=1.e-8); allclose(rtol=1.e-10, atol=1.e-10); allclose(rtol=1.e-10, atol=1.e-8); allclose(rtol=1.e-10, atol=0.); EXPECT_NEAR 1.e-9)
- **Figures/tables:** `src/snapy_report/ch09/figures/fig_coriolis_projection.py` (deck: none (diagram))

#### 7.2 Axial angular momentum: flux form and closure

- **Derivation:** AAM conservation by the flux form of the azimuthal momentum; the global closure fixer
- **Code:** `src/forcing/axial_momentum.cpp:137` `AxialMomentum::flux_form`; `src/forcing/axial_momentum.cpp:297` `AxialClosureFixer::apply`
- **Tests:** `test_axial_momentum.release` (bitwise (equal x2)); `test_axial_momentum_flux_form_python` (constants REST_TOL=1.0e-6, TOL=1.0e-12); `test_global_phase_closure.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** none

#### 7.3 Diffusion and viscosity

- **Derivation:** second-order diffusion operators on Cartesian, cubed-sphere and radial grids; the diffusive time-step bound
- **Code:** `src/forcing/diffusion.cpp:551` `DiffusionImpl::add`; `src/forcing/diffusion.cpp:717` `DiffusionImpl::max_time_step`
- **Tests:** `test_diffusion.release` (bitwise (equal x1); allclose(rtol=1e-05 (default), atol=1e-08 (default)) x3; allclose(rtol=1.e-6, atol=1.e-6); allclose(rtol=1.e-5, atol=1.e-5) x2; allclose(rtol=3.e-4, atol=3.e-4); allclose(rtol=1.e-10, atol=1.e-10); allclose(rtol=1.e-12, atol=0.) x4; EXPECT_NEAR 1.e-12, 1.e-3, 1.e-4 * full, 1.e-6, dtype == torch::kFloat64 ? 1.e-9 : 1.e-3); `test_diffusion_cubed_sphere.release` (bitwise (equal x1); allclose(rtol=0., atol=1.e-12); allclose(rtol=1.e-11, atol=1.e-12); allclose(rtol=1.e-8, atol=1.e-10); allclose(rtol=1.e-12, atol=1.e-9); allclose(rtol=1.e-8, atol=1.e-8); EXPECT_NEAR 0.02 * analytical, 1.e-10, 1.e-11, 1.e-12 * energy0); `test_diffusion_moist.release` (bitwise (equal x1); allclose(rtol=1.e-5, atol=1.e-5); allclose(rtol=1.e-12, atol=1.e-12); allclose(rtol=1.e-6, atol=1.e-6); EXPECT_NEAR 1.e-12 * expected, 1.e-3)
- **Figures/tables:** `src/snapy_report/ch09/figures/fig_viscosity_stencil.py` (deck: none (diagram)); `src/snapy_report/ch09/figures/fig_csdiffusion_stencil.py` (deck: none (diagram))

#### 7.4 Other forcings and the user stage forcing

- **Derivation:** heating, relaxation and sponge terms (one table); the TorchScript stage forcing and its species clip
- **Code:** `src/mesh/meshblock_advance.cpp:240` `MeshBlockImpl::stage_forcing`
- **Tests:** `test_jit_user_forcing_python` (assert_close(rtol=dtype default (default), atol=dtype default (default))); `test_forcing_species_clip_python` (constants tol=1.e-14); `test_source_registry.release` (bitwise (equal x2); allclose(rtol=1.e-12, atol=1.e-12))
- **Figures/tables:** `src/snapy_report/ch09/figures/fig_user_forcing.py` (deck: none (diagram)); `src/snapy_report/ch09/figures/fig_sponge_profile.py` (deck: none (analytic))

### Chapter 8. Moist physics, positivity, floors and limiters

#### 8.1 Saturation adjustment and the vapor repair

- **Derivation:** kintera equilibrium per cell at constant (rho, e); vapor deficit repair by column borrowing
- **Code:** `src/mesh/meshblock_advance.cpp:358` `MeshBlockImpl::stage_saturation`
- **Tests:** `test_fix_vapor_deficit.release` (EXPECT_NEAR 1.e-15); `test_condensate_conservation.release` (bitwise (equal x1); allclose(rtol=1.e-12, atol=1.e-12); allclose(rtol=1.e-6, atol=1.e-6) x2); `test_wall_saturation.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch10/figures/fig_satadj_timeline.py` (deck: none (diagram)); `src/snapy_report/ch10/figures/fig_fixvapor_column.py` (deck: data/fixvapor_column.csv from ch10/evidence/fixvapor_column_run.py (a 64-cell spherical-polar MeshBlock through snapy's own repair; rerun at the pin))

#### 8.2 Sedimentation

- **Derivation:** terminal-velocity donor fluxes of the condensates; the settling buffer fsed1
- **Code:** `src/sedimentation/sed_hydro.cpp:40` `SedHydroImpl::forward`
- **Tests:** `test_sedimentation_guards.release` (bitwise (equal x2); allclose(rtol=1.e-6, atol=1.e-6); EXPECT_NEAR i); `test_sedimentation_cubed_seam.release` (compared < 1.e-12, 1.e-7, 5.e-3)
- **Figures/tables:** `src/snapy_report/ch10/figures/fig_sed_donor.py` (deck: none (diagram))

#### 8.3 The flux positivity limiter and its census

- **Derivation:** theta per cell so that u + dt div F >= 0; borrow between species; census_ counts theta < 1 (kPosHits), theta < 0.9 with the withheld mass above the round-off of the gas in the cell (kPosSevere, #256) and the smallest theta (kPosMin); read by the budget diagnostics
- **Code:** `src/hydro/flux_positivity.cpp:42` `flux_positivity_theta`; `src/hydro/flux_positivity.cpp:207` `PositivityLimiter::census_`
- **Tests:** `test_flux_positivity_python` (compared < 1e-12); `test_flux_positivity_cubedsphere_python` (constants DRIFT_TOL=1e-13)
- **Figures/tables:** `src/snapy_report/ch08/figures/fig_theta_stencil.py` (deck: none (diagram)); `src/snapy_report/ch08/figures/fig_borrow.py` (deck: none (diagram))

#### 8.4 The settling carry on a prescribed face (#315)

- **Derivation:** the carry's advected part F - fsed1 and settling part fsed1 carry their own donors' energy; on a prescribed face their sum is share |fsed1| (h_upper - h_lower), so the post marks the direction
- **Code:** `src/hydro/flux_positivity.cpp:102` `flux_positivity_carry_`
- **Tests:** `test_flux_positivity_carry.release` (bitwise (equal x4); EXPECT_NEAR 0.; constants tol=1.e-12)
- **Figures/tables:** none

#### 8.5 EOS floors and the limiter marks

- **Derivation:** density and pressure floors; the limiter marks are the redo's limiter cause
- **Code:** `src/eos/equation_of_state.cpp:305` `EquationOfStateImpl::apply_conserved_limiter_`
- **Tests:** `test_check_redo_saturation_python` (bitwise (equal x2)); `test_species_face_clamp.release` (bitwise (equal x2))
- **Figures/tables:** none

#### 8.6 The EfStatus slots

- **Derivation:** none (record): one block's EOS, floor and positivity status in named device slots (marks, vapor failures, heat-capacity latch, positivity meters); check_redo reads its causes in one copy, the budget reads the meters
- **Code:** `src/eos/ef_status.hpp:20` `EfStatus`; `src/mesh/stage_executor.cpp:712` `StageExecutor::save_list`. At the pin, `StageExecutor::save_list` saves the block's EfStatus slots (`src/mesh/stage_executor.cpp:726-731`) across capture warm-ups, so a warm-up leaves no count; conversion order 4's fallback count `conversion_trips` is not an EfStatus slot but a hydro module buffer (`src/hydro/hydro.cpp:271-273`), which save_list adds by a name lookup of the hydro's buffers (`src/mesh/stage_executor.cpp:732-735`; 3c1c0c20, landed as 4c023985); the graph = eager meter test is test_graph_positivity_meters_match_eager (`tests/test_graph_positivity_meters_match_eager.py`, registered at `tests/CMakeLists.txt:519`)
- **Tests:** `test_flux_positivity_python` (compared < 1e-12); `test_cycle_diagnostics.release` (EXPECT_NEAR 1.e-12, 1.e-9, 1.e-9 * flux, 1.e-9 * ke_total); `test_fp32_compute_python` (bitwise (equal x2); compared < 2.0e-5); `test_graph_positivity_meters_match_eager_python` (bitwise (equal x1))
- **Figures/tables:** none

### Chapter 9. Time integration: RK stages, dt and step acceptance

#### 9.1 RK3 stages and the combine

- **Derivation:** SSP-RK3 weights (w0, w1, w2) from pyharp; u^{s+1} = w0 u0 + w1 u1 + w2 dt L(u1)
- **Code:** `src/mesh/meshblock_advance.cpp:314` `MeshBlockImpl::stage_combine`
- **Tests:** `test_mesh_executor.release` (bitwise (equal x2))
- **Figures/tables:** `src/snapy_report/ch07/figures/fig_rk_stages.py` (deck: none (diagram))

#### 9.2 The time step

- **Derivation:** acoustic CFL bound per direction, then the sources' bounds; the global minimum over blocks
- **Code:** `src/hydro/hydro.cpp:332` `HydroImpl::max_time_step`; `src/mesh/meshblock_advance.cpp:65` `MeshBlockImpl::global_time_step`
- **Tests:** `test_implicit_cfl.release` (no numeric tolerance in the source (pass/fail checks only)); `test_shear_cfl_python` (no numeric tolerance in the source (pass/fail checks only)); `test_implicit_advection_cfl_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch07/figures/fig_cfl_bounds.py` (deck: none (analytic))

#### 9.3 Step acceptance and the redo decision

- **Derivation:** none (control flow): redo causes stacked on the device, one host read, MAX over ranks; floor, clamp, limiter, NaN, saturation, solve
- **Code:** `src/mesh/meshblock_redo.cpp:90` `MeshBlockImpl::local_redo_flag_tensor`; `src/mesh/meshblock_redo.cpp:168` `MeshBlockImpl::decide_retry`; `src/mesh/meshblock_redo.cpp:32` `MeshBlockImpl::apply_redo`
- **Tests:** `test_check_redo_floor_python` (bitwise (equal x11)); `test_check_redo_parallel.release` (bitwise (equal x3))
- **Figures/tables:** `src/snapy_report/ch07/figures/fig_redo_flow.py` (deck: none (diagram))

#### 9.4 The implicit tracer transfer (S4 padded shifts)

- **Derivation:** tracer face transfer P = M r_donor capped by the held amount; S4 forms r_below and P_above by one pad each
- **Code:** `src/mesh/meshblock_advance.cpp:150` `MeshBlockImpl::stage_advance`
- **Tests:** `test_dry_carry_zero_base_python` (allclose(rtol=0., atol=1.e-12); constants rtol=0); `test_tracer_ratio_python` (compared < 1e-14, 1e-12)
- **Figures/tables:** none

### Chapter 10. The flux register, monitor amounts and conservation budgets

#### 10.1 Posts

- **Derivation:** none (bookkeeping): add, scale and overwrite of a direction's total and its kept parts; plain row adds
- **Code:** `src/hydro/flux_register.cpp:545` `FluxRegister::add`; `src/hydro/flux_register.cpp:574` `FluxRegister::scale`; `src/hydro/flux_register.cpp:587` `FluxRegister::overwrite_`
- **Tests:** `test_flux_register.release` (bitwise (equal x28); allclose(rtol=0., atol=0.))
- **Figures/tables:** **new** `src/snapy_report/ch10/figures/fig_register_order.py` (deck: none (diagram))

#### 10.2 Prescribed faces and their reapplication

- **Derivation:** boundary.prescribed books the removed wall flux; reapplied after the implicit correction or when the written mark is set (the #315 carry: see 8.4)
- **Code:** `src/hydro/flux_register.cpp:771` `FluxRegister::apply_prescribed`; `src/hydro/hydro_forward.cpp:299` `HydroImpl::forward`
- **Tests:** `test_prescribed_after_sources.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** none

#### 10.3 The wall batch

- **Derivation:** the wall rates of a stage gathered in push order, one dense row reduction per stage, shared by the hydro and scalar registers
- **Code:** `src/hydro/flux_register.cpp:380` `WallBatch::flush`; `src/hydro/flux_register.cpp:365` `WallBatch::push`; `src/hydro/flux_register.cpp:684` `FluxRegister::apply_wall_planes_`; `src/mesh/meshblock.cpp:437` `MeshBlockImpl::register_flux_storage`. At the pin, the wall batch and the count masks sit in fixed device storage, registered with the state store in `MeshBlockImpl::register_flux_storage` (`src/mesh/meshblock.cpp:469-474`); `stage_advance` begins the batch each stage (`src/mesh/meshblock_advance.cpp:161`) and the wall slots follow the stage's push order; `apply_wall_planes_` builds the wall weights' scalar operands in the wide dtype on the device
- **Tests:** `test_graph_monitor_amounts.release` (bitwise (equal x7)); `test_flux_register.release` (bitwise (equal x28); allclose(rtol=0., atol=0.))
- **Figures/tables:** none

#### 10.4 Amounts and the stage fold

- **Derivation:** A = sum_s c_s (w2_s dt a_s + p_s), c_s = prod_{t>s} w1_t over the stages ended this step; an open stage's rates wait for its w2 dt, its posts count at once; one device fold per read (FP64; FP32 on MPS)
- **Code:** `src/hydro/flux_register.cpp:254` `RegisterMonitor::fold_`
- **Tests:** `test_graph_monitor_amounts.release` (bitwise (equal x7)); `test_conservation_budget.release` (bitwise (equal x1))
- **Figures/tables:** none

#### 10.5 Monitor counts

- **Derivation:** count masks batched per stage and summed by one integer cumsum (no atomics)
- **Code:** `src/hydro/flux_register.cpp:75` `RegisterMonitor::count`; `src/hydro/flux_register.cpp:139` `RegisterMonitor::flush_counts`. At the pin, the count masks have fixed device storage (`RegisterMonitor::count_storage`, `src/hydro/flux_register.hpp:90`)
- **Tests:** `test_fp32_face_floor_counts_python` (no numeric tolerance in the source (pass/fail checks only)); `test_cycle_diagnostics.release` (EXPECT_NEAR 1.e-12, 1.e-9, 1.e-9 * flux, 1.e-9 * ke_total)
- **Figures/tables:** none

#### 10.6 Rollback and redo, through replay

- **Derivation:** a rejected attempt zeroes the device slots and counters and keeps the host stage bookkeeping, because a replayed redo does not rerun begin_stage/end_stage (pointer from 11)
- **Code:** `src/hydro/flux_register.cpp:514` `FluxRegister::rollback`; `src/hydro/flux_register.cpp:65` `RegisterMonitor::rollback`
- **Tests:** `test_graph_monitor_amounts.release` (bitwise (equal x7))
- **Figures/tables:** none

#### 10.7 Budgets and the budget series

- **Derivation:** conservation budgets read off the register after the step is accepted; the NetCDF budget series
- **Code:** `src/diagnostics/budget.cpp:170` `BudgetConsumer::observe`; `src/output/budget_series.hpp:19` `BudgetSeries`
- **Tests:** `test_budget.release` (bitwise (equal x5)); `test_conservation_budget.release` (bitwise (equal x1))
- **Figures/tables:** `src/snapy_report/ch13/figures/fig_cycle_line.py` (deck: test_cycle_diagnostics' deck); **new** `src/snapy_report/ch10/figures/fig_v3_meters.py` (deck: test_flux_positivity's deck; draws the V3 meters: EfStatus kPosHits, kPosSevere, kPosMin, kLimCut, kLimFlux and the monitor counts per step (the old fig_limiter_meters.py draws the pre-V3 meters at 6fbb3975 and is retired))

### Chapter 11. Execution: the stage executor and CUDA graphs

#### 11.1 The execution model: segments and the schedule

- **Derivation:** none (control flow): a stage is a list of ops grouped into segments (compute, host exchange, eager); the executor runs them in order
- **Code:** `src/mesh/segment.cpp:56` `StageSchedule::regroup`; `src/mesh/stage_executor.cpp:808` `StageExecutor::run_segments`
- **Tests:** `test_stage_schedule.release` (no numeric tolerance in the source (pass/fail checks only)); `test_stage_segments.release` (bitwise (equal x1)); `test_mesh_executor.release` (bitwise (equal x2)). Test_mesh_executor sets `execution: eager` (the eager executor path against the pre-executor sequence)
- **Figures/tables:** **new** `src/snapy_report/ch11/figures/fig_segments.py` (deck: none (diagram))

#### 11.2 Capture, warm-up, the save list and replay

- **Derivation:** none (control flow): three eager warm-ups on a side stream, restore the save list, capture once, replay (the device dt the replay reads: 11.5)
- **Code:** `src/mesh/stage_executor.cpp:864` `StageExecutor::capture`; `src/mesh/stage_executor.cpp:712` `StageExecutor::save_list`; `src/mesh/stage_executor.cpp:1021` `StageExecutor::replay`. At the pin, the save list carries the EfStatus slots and, by a name lookup outside EfStatus, the hydro buffer `conversion_trips` (8.6)
- **Tests:** `test_stage_executor_capture_python` (bitwise (equal x5)); `test_graph_pool_recapture_python` (bitwise (equal x1)); `test_state_store.release` (bitwise (equal x5))
- **Figures/tables:** none

#### 11.3 The refusal list and the allow-lists

- **Derivation:** none (rules): the named reasons a block runs eager; EOS ideal-gas/ideal-moist and Riemann lmars/hllc only
- **Code:** `src/mesh/stage_executor.cpp:579` `StageExecutor::config_reasons`; `src/mesh/stage_executor.cpp:545` `StageExecutor::capture_reasons`. At the pin, the allow-list is at `src/mesh/stage_executor.cpp:600-601`, the split-panel refusal (a cubed-sphere panel split into several blocks, #317) at `:623-628` and the forcing lift at `:629-634`
- **Tests:** `test_execution_capability_python` (no numeric tolerance in the source (pass/fail checks only)); `test_graph_capture_lifts_python` (bitwise (equal x1)); `test_graph_outflow_refused_python` (bitwise (equal x2)); `test_graph_user_forcing_python` (bitwise (equal x3))
- **Figures/tables:** **new** `src/snapy_report/ch11/figures/tab_refusals.py` (deck: none (table from config_reasons))

#### 11.4 The default and strict

- **Derivation:** none (rules): integration.execution (auto, eager, cuda-graph) and execution-strict; written against the pin, which carries the default (STRUCTURE 4.4)
- **Code:** `src/mesh/meshblock_options.cpp:17` `MeshBlockOptionsImpl::from_yaml`; `src/mesh/stage_executor.cpp:107` `StageExecutor::mode`. At the pin, `integration.execution` takes auto (the default, also with no key), eager or cuda-graph (`src/mesh/meshblock.hpp:98`, checked in `src/mesh/meshblock_options.cpp:58-59`); `StageExecutor::mode` (`src/mesh/stage_executor.cpp:107-115`) resolves auto to cuda-graph on a CUDA block and eager otherwise, and an explicit value is honoured; non-strict, each eager reason is logged once per block
- **Tests:** `test_execution_capability_python` (no numeric tolerance in the source (pass/fail checks only)); `test_default_execution_python` (bitwise (equal x2))
- **Figures/tables:** none

#### 11.5 Graph = eager bitwise, graph-safe code and the device dt

- **Derivation:** rules: no host reads, no host-to-device copies inside a captured segment, fixed shapes, host state replay does not refresh; the one home of the device dt: the executor fills a per-block 0-d FP64 dt before each replay, and StepDt forms w2 dt, dt_corr and x/dt from it bitwise like the host dt (x * (1/dt) for torch's host-scalar division)
- **Code:** `src/utils/step_dt.cpp:14` `StepDt::value`; `src/utils/step_dt.cpp:25` `StepDt::divide`; `src/utils/step_dt.cpp:36` `StepDt::scaled`; `src/mesh/stage_executor.cpp:283` `StageExecutor::device_dt`; `src/mesh/state_store.hpp:24` `StateStore`. At the pin, the stretched x1 reference rows are built on the grid's device (`src/recon/x1_reference.hpp:62-66`), so no host-to-device copy runs inside a captured segment (#316)
- **Tests:** `test_sk3_capture_moist_python` (bitwise (equal x4)); `test_graph_implicit_dt_python` (bitwise (equal x3)); `test_graph_chemistry_slot_python` (bitwise (equal x3)); `test_graph_stretched_x1_python` (bitwise (torch.equal of every block's state after every step, tests/test_graph_stretched_x1.py:85); its rtol=1.e-13 (:51-52) is the tolerance of the balance_column solve that builds the initial state, not a comparison tolerance)
- **Figures/tables:** none

#### 11.6 The S4 cost catalogue and the measured gap

- **Derivation:** none (catalogue): pointers to 4.3, 6.3, 9.4, 10.3 and 10.5; the measured eager-graph gap and the bound as ruled are cited from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan), and no number is repeated here; method in 16.3
- **Code:** none in `src/` (method or record section)
- **Tests:** none (no test covers this section at the pin)
- **Figures/tables:** table cited from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan)

#### 11.7 Eager ops inside a captured stage: the user forcing and the chemistry segment

- **Derivation:** none (rules): an op that cannot be captured runs as its own eager segment with the host dt, between captured segments; a registered chemistry step is such an op (eager FP64 kinetics), except in a Mesh block, which defers chemistry to the Mesh's global phase and keeps the op in the graph
- **Code:** `src/mesh/stage_executor.cpp:149` `StageExecutor::build_schedule`; `src/mesh/stage_executor.cpp:130` `StageExecutor::chemistry_slot`
- **Tests:** `test_graph_chemistry_slot_python` (bitwise (equal x3)); `test_graph_mesh_chemistry_python` (bitwise (equal x1)); `test_graph_user_forcing_python` (bitwise (equal x3)); `test_sk4_chemistry_source_python` (bitwise (equal x2))
- **Figures/tables:** none

### Chapter 12. Precision: the FP32 policy

#### 12.1 The precision policy

- **Derivation:** one row per quantity: storage, compute and accumulation dtype; FP64 masters for geometry and references; the wide dtype is FP32 on MPS (no FP64 there), the one home of MPS precision
- **Code:** `src/mesh/precision_policy.hpp:31` `PrecisionPolicy`
- **Tests:** `test_precision_policy.release` (bitwise (equal x1)); `test_fp32_policy_python` (bitwise (equal x4); compared < 2.0e-8, 2.0e-5)
- **Figures/tables:** **new** `src/snapy_report/ch12/figures/tab_policy_rows.py` (deck: none (table from precision_policy.hpp))

#### 12.2 FP32 storage, typed constants and the horizontal sweep

- **Derivation:** FP32 state with FP64-accumulated sums; constants cast once per dtype
- **Code:** `src/utils/tensor_cast_cache.hpp:17` `TensorCastCache`
- **Tests:** `test_fp32_typed_constants.release` (bitwise (equal x2)); `test_fp32_horizontal_sweep_python` (constants FIELD_TOL=2.0e-5, VELOCITY_TOL=5.0e-5); `test_fp32_compute_python` (bitwise (equal x2); compared < 2.0e-5)
- **Figures/tables:** none

#### 12.3 Anomaly storage and perturbation flux

- **Derivation:** FP32 needs the anomaly u - u_ref and the perturbation flux so the hydrostatic part cancels in FP64
- **Code:** `src/eos/ideal_moist.cpp:79` `IdealMoistImpl::perturbations`
- **Tests:** `test_anomaly_storage_python` (bitwise (equal x7); constants tol=1.0e-10); `test_perturbation_flux_python` (bitwise (equal x1); compared < 2.e-13, 2.e-10, 2.e-9); `test_perturbation_flux_split_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** none

#### 12.4 The FP32 implicit solve (pointer to 6.3)

- **Derivation:** block-row scaling before the FP32 LU; FP64 retry of failed columns only
- **Code:** `src/hydro/hydro.cpp:420` `HydroImpl::implicit_solve_float`
- **Tests:** `test_fp32_vic_pivot_python` (no numeric tolerance in the source (pass/fail checks only)); `test_fp32_implicit_python` (bitwise (equal x1); constants rtol=1.0e-13; compared < 2.0e-6); `test_fp32_restart_python` (bitwise (equal x8); compared < 2.0e-5)
- **Figures/tables:** none

### Chapter 13. Mesh, layout, exchange and multi-block runs

#### 13.1 Layouts and ranks

- **Derivation:** none (mapping): slab, cubed and cubed-sphere layouts; block -> rank -> panel
- **Code:** `src/layout/cubed_sphere_layout.cpp:514` `CubedSphereLayoutImpl::neighbor_rank`; `src/layout/cubed_sphere_layout.cpp:504` `CubedSphereLayoutImpl::loc_of`
- **Tests:** `test_process_group.release` (no numeric tolerance in the source (pass/fail checks only)); `test_local_horizontal_cells_python` (compared < 1.0e-12)
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_layouts_morton.py` (deck: none (diagram))

#### 13.2 The ghost exchange

- **Derivation:** none (control flow): pack, same-process rendezvous or process-group send/recv, unpack; corner refresh
- **Code:** `src/mesh/meshblock_exchange.cpp:124` `MeshBlockImpl::exchange_ghost_zones`; `src/mesh/mesh.cpp:446` `MeshImpl::exchange_ghost_zones`
- **Tests:** `test_exchange.release` (isclose(rtol=1e-05 (default), atol=1e-08 (default)) x2); `test_mesh_exchange_python` (no numeric tolerance in the source (pass/fail checks only)); `test_cubed_sphere_exchange.release` (bitwise (equal x9)); `test_periodic_corners_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_exchange_stencil.py` (deck: none (diagram))

#### 13.3 Exchange cuts and multi-block capture (B3)

- **Derivation:** none (control flow): an in-stage exchange is a cut; the blocks capture together after align_captures
- **Code:** `src/mesh/stage_executor.cpp:360` `exchange_cut`; `src/mesh/stage_executor.cpp:497` `StageExecutor::align_captures`
- **Tests:** `test_mesh_capture.release` (bitwise (equal x2)); `test_mesh_capture_moist_python` (bitwise (equal x1)); `test_cubed_sphere_capture_python` (bitwise (equal x1))
- **Figures/tables:** **new** `src/snapy_report/ch13/figures/fig_exchange_cut.py` (deck: none (diagram))

#### 13.4 Mesh-agree

- **Derivation:** none (rule): a Mesh whose blocks disagree (one refuses, or modes differ) runs eager as a whole
- **Code:** `src/mesh/stage_executor.cpp:529` `StageExecutor::agree`. At the pin, the blocks' resolved modes (`StageExecutor::mode`) are compared, not the deck key (`src/mesh/stage_executor.cpp:557-560`)
- **Tests:** `test_mesh_disagree.release` (bitwise (equal x1))
- **Figures/tables:** none

#### 13.5 The x1 seam split

- **Derivation:** none (control flow): x1 relays between blocks split in x1 (pz > 1); refused for capture and for the implicit solve
- **Code:** `src/layout/exchange.cpp:123` `Exchange::x1_seam_peers`
- **Tests:** `test_x1_seam_split.release` (no numeric tolerance in the source (pass/fail checks only)); `test_x1_seam_split_mp.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch03/figures/fig_x1seams_relay.py` (deck: none (diagram))

### Chapter 14. Boundary conditions and immersed solids

#### 14.1 Boundary kinds

- **Derivation:** reflecting, outflow/radiating, periodic and custom ghost fills
- **Code:** `src/mesh/meshblock_exchange.cpp:220` `MeshBlockImpl::apply_boundaries`; `src/mesh/meshblock_options.cpp:169` `MeshBlockOptionsImpl::is_wall_boundary`
- **Tests:** `test_radiating_boundary.release` (bitwise (equal x11); allclose(rtol=1e-05 (default), atol=1e-08 (default)) x6; allclose(rtol=1.e-5, atol=1.e-6) x4; allclose(rtol=1.e-12, atol=1.e-9) x2; allclose(rtol=1.e-12, atol=1.e-12) x5; allclose(rtol=1.e-10, atol=1.e-10)); `test_outflow_stale_ghosts_python` (constants REST_TOL=1.0e-3)
- **Figures/tables:** `src/snapy_report/ch11/figures/fig_reflecting_stencil.py` (deck: none (diagram)); `src/snapy_report/ch11/figures/fig_outflow_stencil.py` (deck: none (diagram))

#### 14.2 Which faces are prescribed, and with what

- **Derivation:** wall faces carry zero mass flux and the pressure term; how the register books them is 10.2
- **Code:** `src/layout/seam_register.cpp:215` `supply_prescribed`
- **Tests:** `test_wb_wall_corner.release` (no numeric tolerance in the source (pass/fail checks only)); `test_wall4_sphere_python` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** none

#### 14.3 Immersed solids

- **Derivation:** solid mask; fill of solid cells; the implicit solve on stratified solids
- **Code:** `src/bc/internal_boundary.cpp:66` `InternalBoundaryImpl::forward`; `src/bc/rectify_solid.cpp:109` `rectify_solid`
- **Tests:** `test_implicit_stratified_solid_python` (bitwise (equal x1); compared < 1e-12, 1.e-12); `test_rectify.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch11/figures/fig_solids_stencil.py` (deck: none (diagram))

### Chapter 15. Configuration, output and restart

#### 15.1 Options schema and YAML keys

- **Derivation:** none (schema): generated option tables, defaults and key checks
- **Code:** `src/mesh/meshblock_options.cpp:17` `MeshBlockOptionsImpl::from_yaml`; `src/hydro/hydro_options.cpp:19` `HydroOptionsImpl::from_yaml`. At the pin, `integration.execution` defaults to auto (`src/mesh/meshblock.hpp:98`, values checked in `src/mesh/meshblock_options.cpp:58-59`)
- **Tests:** `test_options_table.release` (no numeric tolerance in the source (pass/fail checks only)); `test_options_defaults.release` (bitwise (equal x1)); `test_yaml_keys.release` (no numeric tolerance in the source (pass/fail checks only))
- **Figures/tables:** `src/snapy_report/ch12/figures/fig_yaml_flow.py` (deck: none (diagram))

#### 15.2 Output writers

- **Derivation:** none (I/O): gridded writer core, NetCDF and the combine step
- **Code:** `src/output/writer_core.cpp:200` `GriddedWriter::write_output_file`
- **Tests:** `test_netcdf_axis_attrs` (no numeric tolerance in the source (pass/fail checks only)); `test_netcdf_cubed_lonlat_python` (bitwise (equal x2)); `test_combine_parts.release` (no numeric tolerance in the source (pass/fail checks only)); `test_user_output.release` (bitwise (equal x1); allclose(rtol=1.e-12, atol=1.e-12) x2; allclose(rtol=1e-05 (default), atol=1e-08 (default)) x2; EXPECT_NEAR 1.e-10, 1.e-12, 1.e-6)
- **Figures/tables:** none

#### 15.3 Restart

- **Derivation:** none (I/O): restart bundles, metadata checks and the resume schedule
- **Code:** `src/output/restart.cpp:40` `RestartOutput::write_output_file`; `src/output/restart.cpp:321` `check_restart_metadata`
- **Tests:** `test_restart_bundle.release` (bitwise (equal x1)); `test_restart_resume_schedule` (no numeric tolerance in the source (pass/fail checks only)); `test_fp32_restart_python` (bitwise (equal x8); compared < 2.0e-5)
- **Figures/tables:** none

### Chapter 16. Verification and reproducibility

#### 16.1 The bitwise gates

- **Derivation:** none (method): the face-deck harness (harness348) and the dump set (dumpset349) compared byte for byte between commits, CPU with CPU and CUDA with CUDA; both are tools on the closing-plan branch, pinned by path at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan): `docs/v3/closing/tools/harness348/h348.py` (and its `decks/`), `docs/v3/closing/tools/dumpset349/dumpset349.py` with `compare.py`, run by `docs/v3/closing/tools/gates.sh`, with no ctest of their own (test_straka is a reference L2 comparison, not a bitwise gate)
- **Code:** none in `src/` (method or record section)
- **Tests:** none (no test covers this section at the pin)
- **Figures/tables:** **new** `src/snapy_report/ch16/figures/tab_gates.py` (deck: the dump-set and harness decks (closing-plan tools))

#### 16.2 The capture sweep

- **Derivation:** none (method): every shipped test run under the cuda-graph default; table of captured / refused (reason) / failed
- **Code:** none in `src/` (method or record section)
- **Tests:** `test_stage_executor_capture_python` (bitwise (equal x5))
- **Figures/tables:** **new** `src/snapy_report/ch16/figures/tab_capture_sweep.py` (deck: the sweep's committed record (open slot))

#### 16.3 Launch counting and the S4 table

- **Derivation:** none (method): profiler trace; kernel, memcpy and memset events over N steps, taken with `docs/v3/closing/tools/speed/bench.py` (`--profile`, `SPEED_PROF_TRACE`) at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan); the table and its numbers are cited from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan), and no number is repeated here
- **Code:** none in `src/` (method or record section)
- **Tests:** none (no test covers this section at the pin)
- **Figures/tables:** table cited from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b702c074eae3aa05e12759357e2ee9f4b51` (v3/closing-dryrun-plan)

#### 16.4 Regression provenance

- **Derivation:** none (record): each regression test with the issue and the RED/GREEN commits it came from
- **Code:** none in `src/` (method or record section)
- **Tests:** `test_flux_positivity_carry.release` (bitwise (equal x4); EXPECT_NEAR 0.; constants tol=1.e-12); `test_graph_monitor_amounts.release` (bitwise (equal x7)); `test_seam_rotation_cache_python` (bitwise (equal x3)); `test_mesh_disagree.release` (bitwise (equal x1)); `test_graph_stretched_x1_python` (bitwise (torch.equal of every block's state after every step, tests/test_graph_stretched_x1.py:85); its rtol=1.e-13 (:51-52) is the tolerance of the balance_column solve that builds the initial state, not a comparison tolerance); `test_flux_register.release` (bitwise (equal x28); allclose(rtol=0., atol=0.)); `test_default_execution_python` (bitwise (equal x2)); `test_graph_positivity_meters_match_eager_python` (bitwise (equal x1)). Provenance on `v3/final`: test_graph_stretched_x1 #316 (RED 213189c4, fix 069769ba); test_graph_monitor_amounts `GraphSaveList.user_forcing_graph_books_wall_rates_like_eager` (RED 8019073b, fix a78dc50f then fa734fff); test_flux_register `FluxRegister.wall_prescription_books_in_float32_on_the_device` (added with 9766b2d7); test_default_execution (RED 53c46723, fix 02e4ef05); test_graph_positivity_meters_match_eager (`tests/test_graph_positivity_meters_match_eager.py`, ctest at `tests/CMakeLists.txt:519`; RED e276ece3 and 897ef787, fix 66614c32 (EfStatus slots in the save list) then 4c023985 = 3c1c0c20 (`conversion_trips` by a name lookup, `src/mesh/stage_executor.cpp:732-735`))
- **Figures/tables:** **new** `src/snapy_report/ch16/figures/tab_regressions.py` (deck: none (table))

#### 16.5 CPU, CUDA and MPS reproducibility

- **Derivation:** none (method): CPU and CUDA are compared bitwise; MPS is held to err = max|mps - cpu| / max|cpu| <= max(1e-4, 3 x the CPU FP32-vs-FP64 error) per arm, the reconstruction arms exactly 0
- **Code:** none in `src/` (method or record section)
- **Tests:** `test_mps_matches_cpu_python` (err = max|mps - cpu| / max|cpu| <= max(TOL=1e-4, 3 x the CPU FP32-vs-FP64 error) per arm; the reconstruction arms exactly 0 (tests/test_mps_matches_cpu.py:4-9, :27)); `test_weno5_cuda_line.release` (bitwise (equal x1))
- **Figures/tables:** `src/snapy_report/ch15/figures/fig_evidence.py` (deck: none (diagram))

### Chapter 17. Known limits (pointer rows only)

| limit | home | anchor at the pin |
|---|---|---|
| Multi-rank capture is refused (P1) | 11.3, 13.4 | `src/mesh/stage_executor.cpp:579` `StageExecutor::config_reasons` |
| EOS and Riemann solvers outside the allow-lists run eager | 11.3 | `src/mesh/stage_executor.cpp:579` `StageExecutor::config_reasons` |
| An x1 split (pz > 1) is refused for capture and for the implicit solve | 13.5, 6.2 | `src/hydro/hydro.cpp:436` `HydroImpl::_apply_implicit_correction` |
| FP32 on the cubed sphere needs anomaly storage, which is Cartesian only | 12.3 | `src/eos/ideal_moist.cpp:79` `IdealMoistImpl::perturbations` |
| A registered chemistry step runs as an eager segment between captured ones (a Mesh block keeps it in the graph) | 11.7 | `src/mesh/stage_executor.cpp:130` `StageExecutor::chemistry_slot` |
| MPS has no FP64: the wide dtype is FP32 there | 12.1 | `src/mesh/precision_policy.hpp:31` `PrecisionPolicy` |
| MPS runs eager only, and the moist column stops before saturation (no kintera MPS kernel) | 12.1, 16.5 | `src/mesh/stage_executor.cpp:536` `StageExecutor::vars_reasons` |
| MPS hard refusal: anomaly storage cannot move to MPS (its reference is FP64, which MPS does not have); `Fp64StoreImpl::to(Device)` stops with an error, it does not fall back | 12.3, 12.1 | `src/hydro/fp64_store.hpp:23` `Fp64StoreImpl` |
| MPS hard refusal: a stretched x1 grid (geometry/x1f or cells/x1rat) is not supported on MPS (#280); the hydro stops with an error, it does not fall back | 3.1, 16.5 | `src/hydro/hydro.cpp:302` `HydroImpl::_check_stretched_x1` |
| A cubed-sphere panel split into several blocks (nb2 or nb3 > 1) is refused for capture with its reason and runs eager (#317): its seams inside the panel are not cut into segments | 11.3, 13.3 | `src/mesh/stage_executor.cpp:579` `StageExecutor::config_reasons` |

### Appendices

A-F and H-J are indexes, tables or generated references with no sections of their own; they are built from the chapters (C test index from the Tests lines above, D and I from the options schema, J from the tree at the pin). G (the V3 lane map) carries `docs/v3/DESIGN.rst` on happysky19/snapy `v3-design/DESIGN` `449636b3e1963b5fc7d44407cb7d6db437e8b016`, Table 1 (lines 36-80) and Table 7 (lines 209-275), mapped to these chapters. E (external components) cites kintera, pyharp, pydisort and commux by their versions at the pin (no `code_pin.yml` entry; STYLE.md 3.1). In I, `integration.execution` defaults to `auto` (`src/mesh/meshblock.hpp:98`, values auto, eager and cuda-graph checked in `src/mesh/meshblock_options.cpp:58-59`).

## 7. Figures and tables: how the repo makes them

This restates the repo's own practice (STYLE.md sections 6 and 10.4-10.5, `tools/render_figures.py`); it adds no rule.

- **Where a figure lives.** A figure is a `{python}` cell in the section's `_<scheme>.qmd` (`#| label: fig-chNN-...`,
  `fig-cap`, `fig-alt`, `echo: false`) that calls `make_fig()` of a module
  `src/snapy_report/chNN/figures/fig_<scheme>_<what>.py`; the module imports only numpy, matplotlib and the shared
  `src/snapy_report/figstyle.py` (STYLE.md 6.1, 10.4). Fonts, sizes and palette come from `figstyle`, never set per
  figure (STYLE.md 6.2).
- **How it is drawn.** `python3 tools/render_figures.py --chapters chNN` draws a chapter's figure cells the way the
  render runs them, to `build/figures/<chapter>/<label>.svg|pdf|png`; `--manifest` rewrites `FIGURES.md` (figure ->
  module -> data -> the script that wrote the data) and `--check` is the render gate's comparison. A cell that cannot
  be drawn fails the draw.
- **Data figures.** A data figure reads committed data under `src/snapy_report/chNN/data/`, written by a committed
  script under `src/snapy_report/chNN/checks/` or `evidence/`, or printed by a ctest; the manifest records which. The
  inputs named in section 6 are those producers; where a producer ports code that changed or was deleted by the pin,
  section 6 marks it **new** or "re-port at the pin".
- **Captions and provenance.** One sentence on what is shown, then what to read from it (STYLE.md 10.4); the pin a
  section was written against is in its PROVENANCE block (STYLE.md 3.2), and a run's data file carries its
  provenance field (the command, the pin and, for a kintera-dependent run, the kintera version). Measured S4 numbers
  are cited from `docs/v3/closing/s4_eager_gap.md` at cshsgy/snapy `be267b70`, not re-plotted.
- **Tables.** Pipe tables as STYLE.md 10.5 sets them; a table a script generates follows the same path as a figure.
