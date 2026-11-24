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
> cuts, mesh-agree), S4 (eager-path launch reduction, the execution default), seams (panel-seam register and
> rotation cache), capture (the capture rules and allow-lists).

## 1. Principle

The pre-V3 book follows the physics of one step and scatters the machinery that runs it (stages, retries,
budgets, GPU) over chapters 7, 13 and 14. V3 makes that machinery explicit and shared: every stage runs through
one stage executor (`src/mesh/stage_executor.*`, `segment.*`), every face flux goes through one flux register
with a monitor (`src/hydro/flux_register.*`), state lives in fixed storage (`src/mesh/state_store.hpp`), dt is a
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
| 4 | Reconstruction, Riemann solvers and the hydro sweep | kept (old 4.A and 4.B) | C, R (sweep) | C owns reconstruction and solvers; the sweep is where they meet the register |
| 5 | Hydrostatic reference and well-balanced treatment | kept (old 5, seam parts moved to 3) | C | unchanged scope, lane C |
| 6 | Gravity work and the vertical implicit correction | merged (old 6 + old 7.B) | I | lane I owns both; the VIC energy row is the face gravity work |
| 7 | Rotation, axial momentum, diffusion and forcing | kept (old 9, Coriolis and axial momentum gathered) | A | lane A's scope as one chapter |
| 8 | Moist physics, positivity, floors and limiters | merged (old 8 + old 10) | EF, K, S4 (census) | EF owns EOS, floors, positivity, tracers and sedimentation together |
| 9 | Time integration: RK stages, dt and step acceptance | split (from old 7.A) | X (integrator), I (FP64 retry flag) | the stage loop and redo decision, without the bookkeeping now in 10 |
| 10 | The flux register, monitor amounts and conservation budgets | merged (old 13 + new register material) | R, O (budget series), S4 | budgets are read off the register in V3; one home for fluxes, amounts and their rollback |
| 11 | Execution: the stage executor and CUDA graphs | new | X, B3, S4, capture | the central V3 addition; one home for the execution model and graph rules |
| 12 | Precision: the FP32 policy | new | X (policy), I, EF, C (their rows) | one policy table governs every FP32 decision; today it would be scattered |
| 13 | Mesh, layout, exchange and multi-block runs | split (from old 14.A) | S, X (Mesh), B3 | layout, exchange and the same-process rendezvous are the Mesh's own model |
| 14 | Boundary conditions and immersed solids | kept (old 11) | S | unchanged scope |
| 15 | Configuration, output and restart | merged (old 12 + old 14.B) | O | lane O owns the options schema, writer core and restart together |
| 16 | Verification and reproducibility | merged (old 15 + old 14.C) | all | the bitwise gates are the verification catalogue's own rules |
| 17 | Known limits | kept (old 16) | all | unchanged role |
| A | Notation | kept | - | |
| B | Derivation index | kept | - | |
| C | Test index | kept | - | |
| D | Environment-variable switches | kept | X, O | |
| E | External components: kintera, pyharp, pydisort, commux | kept, extended | K | gains the kintera V3 contract surface (K) |
| F | Topic index | kept | - | |
| G | V3 lane map and design-document index | merged (old G task map) | all | replaces the task map with the lane -> chapter -> design-document map |
| H | Glossary | kept | - | adds V3 terms (segment, cut, capture, census, fold) |
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
| capture rules and allow-lists (EOS ideal-gas/ideal-moist, Riemann lmars/hllc, named refusal reasons) | 11 | `StageExecutor::config_reasons` in `src/mesh/stage_executor.cpp` |
| the execution default (cuda-graph non-strict on CUDA, eager on CPU, explicit eager honoured) | 11 | see hypothesis H1 |
| graph = eager bitwise rule; why eager is launch-bound and graph faster | 11 | measured counts and timings to be cited from their posts, none invented |
| graph-safe coding (no host reads, fixed shapes, device dt, host state replay does not refresh) | 11 | `src/utils/step_dt.*`, `src/mesh/state_store.hpp` |
| mesh-agree (a Mesh whose blocks disagree runs eager) | 13 | `StageExecutor::agree` in `src/mesh/stage_executor.cpp` |
| exchange cuts and multi-block capture (B3) | 13 | `src/mesh/exchange_cut.hpp`, `src/layout/exchange.*` |
| flux register and monitor amounts: stage fold, rollback/redo of amounts | 10 | `FluxRegister`, `RegisterMonitor` in `src/hydro/flux_register.hpp` |
| seam rotation cache | 3 | `SeamRotationCache` in `src/layout/seam_register.hpp` |
| positivity census | 8 | `census_` in `src/hydro/flux_positivity.hpp` |
| FP32 precision policy | 12 | `PrecisionPolicy` in `src/mesh/precision_policy.hpp` |
| redo decision and step acceptance | 9 | `src/mesh/meshblock_redo.cpp` |

Every row was found at the pin by `git grep` on `70f6e15e` (file and symbol exist). Their behaviour is not
re-read here; that is the section outline's job.

## 5. Hypotheses (not checked in code)

- **H1.** The S4 default (cuda-graph, non-strict, on CUDA; eager on CPU; an explicit `eager` honoured) is a
  ruling, not yet code at the pin: at `70f6e15e` `integration.execution` defaults to `eager`
  (`src/mesh/meshblock.hpp:96`). Chapter 11 is written against the commit that lands the default.
- **H2.** Old 10's chemistry material splits cleanly into the chemistry segment (11) and the kintera contract
  (E); not checked against the old chapter's sections.
- **H3.** The census belongs with positivity (8) rather than the register (10): it records the limiter's own
  actions and is rolled back with the register (`src/mesh/meshblock_redo.cpp:56`, comment only read).
- **H4.** Precision warrants its own chapter (12): assumed from the single `PrecisionPolicy` class and the
  number of lanes with FP32 rows; the size of the material is not measured.
- **H5.** The lane owning each new chapter matches the lane scopes in DESIGN.rst Table 7; ownership of the
  closing work (B3, S4, seams, capture) after the design round is taken from the integration history, not a
  document.
