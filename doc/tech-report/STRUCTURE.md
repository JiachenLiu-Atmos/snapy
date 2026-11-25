# snapy Technical Report: structure proposal for the V3 update

> Status: **proposal for comment**, chapter level only. No section outline and no writer briefs yet; those
> follow in `OUTLINE.md` once this structure is approved. Binding style: `STYLE.md` (every scheme section keeps
> the six layers: Summary, Derivation, Numerical method, Code, Tests, Limits).
>
> Code pin for this proposal: snapy `v3/final` = `70f6e15e44c70f76c1dc180007366a3e7a8d1a18` (UCzhangxi/snapy).
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
| capture rules and allow-lists (EOS ideal-gas/ideal-moist, Riemann lmars/hllc), the refusal list with its named reasons | 11 | `StageExecutor::config_reasons` in `src/mesh/stage_executor.cpp` (lines 558-616 at `70f6e15e`) |
| capture sweep (which shipped tests capture, and the table of results) | 16 | the sweep's committed record, cited when it lands |
| S4 cost catalogue: one short list of the eager-path folds, each pointing to its home (4, 6, 9, 10), plus the counting method's home (16) | 11 | pointers only; the code is described in the home chapters |
| the execution default (cuda-graph non-strict on CUDA, eager on CPU, explicit eager honoured) | 11 | see hypothesis H1 |
| graph = eager bitwise rule; why eager is launch-bound and graph faster | 11 | launches per step and timings as measured, cited from their committed source (see 4.2); no H100 headline |
| launch-counting procedure (profiler trace; kernel, memcpy and memset events over N steps) | 16 | `docs/v3/closing/tools/speed/bench.py` on the closing-plan branch (`--profile`, `SPEED_PROF_TRACE`) |
| graph-safe coding (no host reads, no host-to-device copies inside a captured segment, fixed shapes, device dt, host state replay does not refresh) | 11 | `src/utils/step_dt.*`, `src/mesh/state_store.hpp`; the copy rule's example at `70f6e15e` is `src/recon/x1_reference.hpp:140` (rows built on the host at `:62` copied to the device in `evaluate`), see 4.3 |
| mesh-agree (a Mesh whose blocks disagree runs eager) | 13 | `StageExecutor::agree` in `src/mesh/stage_executor.cpp` |
| exchange cuts and multi-block capture (B3) | 13 | `src/mesh/exchange_cut.hpp`, `src/layout/exchange.*` |
| flux register and monitor amounts: posts, prescribed faces, wall batch, stage fold, rollback/redo | 10 | `FluxRegister`, `RegisterMonitor`, `WallBatch` in `src/hydro/flux_register.hpp` |
| monitor counts (face floors, `implicit.fp64_columns`, events) and their rollback | 10 | `RegisterMonitor::count`, `flush_counts`, `rollback` |
| rollback through graph replay (the redo replays the captured stages; the host stage bookkeeping survives `FluxRegister::rollback`) | 10, pointer from 11 | `FluxRegister::rollback` in `src/hydro/flux_register.cpp`; `StageExecutor::replay` in `src/mesh/stage_executor.cpp` |
| prescribed-face reapplication after the implicit correction | 10 | `FluxRegister::apply_prescribed`; the call in `HydroImpl::forward` (`src/hydro/hydro_forward.cpp`) |
| the settling carry on a prescribed face (#315) | 8, pointer from 10 | `flux_positivity_carry_` in `src/hydro/flux_positivity.cpp` |
| regression provenance (which issue and RED/GREEN commits each regression test pins) | 16 | the tests' own headers and the issues they cite |
| seam rotation cache | 3 | `SeamRotationCache` in `src/layout/seam_register.hpp` |
| positivity census (EfStatus `kPosHits`, `kPosSevere`, `kPosMin`, `kScalarPosHits`) | 8 | `PositivityLimiter::census_` in `src/hydro/flux_positivity.cpp` |
| S4 status truth table | 6 | `ImplicitHydroImpl::status` in `src/implicit/implicit_hydro.cpp` |
| S4 padded tracer-transfer shifts (`r_below`, `P_above`) | 9 | `MeshBlockImpl::stage_advance` in `src/mesh/meshblock_advance.cpp` |
| S4 floor and event count posts | 4 | `src/recon/reconstruct.*`, `src/riemann/riemann.*`, `src/hydro/sweep.cpp` |
| FP32 precision policy | 12 | `PrecisionPolicy` in `src/mesh/precision_policy.hpp` |
| redo decision and step acceptance | 9 | `src/mesh/meshblock_redo.cpp` |

Every row was found at the pin by `git grep` on `70f6e15e` (file and symbol exist). Their behaviour is not
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

The S4 table (main eager, the default, explicit eager; ms/step and launches/step; two GPUs, one of them shared,
with that note) and the bound as ruled (the default CUDA path at least as fast as main eager on every S4 cell;
CPU eager within 5%; the CUDA eager gap reported) are committed in `docs/v3/closing` on the closing-plan branch
(`v3/closing-dryrun-plan`, UCzhangxi/snapy), not in this report. Chapter 11 cites that file for the numbers and
the bound, chapter 16 cites it for the procedure's results, and neither writes a number of its own. The
counting procedure itself (profiler trace; kernel, memcpy and memset events over N steps) is described in 16.
Until that file is committed, 11 and 16 carry the citation as an open slot, not a number.

### 4.3 Writing around a fix that has not landed

The capture crash on stretched x1 grids (an H2D copy inside a captured segment, `src/recon/x1_reference.hpp:140`
at `70f6e15e`) has a fix in progress whose mechanism is not settled at this pin (keep the rows on the device, or
refuse the case). Chapter 11 states the rule (no host-to-device copy inside a captured segment) and cites the
mechanism of the pinned commit, whichever it is; if a refusal remains, it is one more entry of the refusal list
in 11 and a pointer row in 17. Neither chapter is written against an unlanded fix.

### 4.4 The pin

One pin per package (STYLE.md 3.1, `code_pin.yml`): the whole report moves to the commit that lands the
execution default, by `tools/code_anchors.py bump`; chapter 11 is not pinned separately. This structure stays
pinned at `70f6e15e` until that commit is on `v3/final`. At `939f7928` (a fast-forward) only
`src/hydro/flux_register.cpp` changes; line numbers cited here in that file move by up to 10 after line ~204
(e.g. `:459-465` is `:469-475`, `:502` is `:512`), and those up to line ~204 do not.

## 5. Hypotheses (not checked in code)

- **H1** (confirmed in review). The S4 default (cuda-graph, non-strict, on CUDA; eager on CPU; an explicit
  `eager` honoured) is a ruling, not yet code at the pin: at `70f6e15e` `integration.execution` defaults to
  `eager` and `execution_strict` to false (`src/mesh/meshblock.hpp:96-97`). Chapter 11 is written against the
  commit that lands the default, to which the whole report is re-pinned (section 4.4), and cites the bound
  as ruled from `docs/v3/closing` (section 4.2; a ruling, not checked here).
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
