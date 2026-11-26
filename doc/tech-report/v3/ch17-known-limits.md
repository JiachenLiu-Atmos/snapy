# 17 Known limits: the V3 rows

<!-- PROVENANCE
pin: snapy@73c45a7c8c991209924e1137992352c335575062 (UCzhangxi/snapy v3/final at the code pin; code_pin.yml)
outline: doc/tech-report/STRUCTURE.md at UCzhangxi/snapy tech-report 790428d83e1ca0cc285dc05db2a8848e90c39a0c, chapter 17
sources: src/mesh/stage_executor.cpp, src/hydro/hydro.cpp, src/eos/ideal_moist.cpp, src/mesh/precision_policy.hpp, src/hydro/fp64_store.hpp and the tests named in the Tests layer, all read at the pin
form: pre-conversion draft (STYLE.md section 0): `path:lines@sha` code locations; section 10 (Quarto markup) does not bind it yet
status: draft
author: Xinyue Wang
reviewer: (checker from tools/chapter-owners.tsv: gitlinffff)
-->

> **Draft conventions** (as in the chapter 4 and 5 drafts). Every code location is `path:lines@73c45a7c`, the code
> pin of `code_pin.yml`. A bare `:N` refers to the file of the last full citation before it. Per the outline, this
> chapter holds pointer rows only: each row names its home chapter, repeats no measured number, and carries no
> regression history. A claim not read in the code at the pin is marked **unverified**.

This draft holds the limits that the V3 execution model and precision policy add to the register. The book
chapter `17-known-limits.qmd` keeps the existing register (165 items, by class). These ten rows are written for the
outline's chapter 17. Each row says what the code refuses or does not do, which chapter owns the explanation, and
where the refusal is written.

---

## 17.1 The V3 limits register (no switch: refusals and eager fallbacks)

### Summary

Ten limits of the V3 code, each one a refusal with a named message or an eager fallback. Four groups:

- **CUDA-graph capture.** It is refused for more than one process, for EOS and Riemann types outside its
  allow-lists, and for an $x_1$ split. A capture that runs out of memory stops rather than falling back.
- **The implicit solve.** It is refused for an $x_1$ split.
- **FP32.** It needs anomaly storage, which is Cartesian only.
- **MPS.** No FP64, eager only, and two hard refusals.

None of these produces a wrong result silently. Each either stops with a message or runs eagerly, which the
executor reports among its eager reasons.

::: {.callout-note title="At a glance"}
**Purpose.** One pointer row per V3 limit, with its home chapter and the line that enforces it.

**Data.** None: a register.

**Placement.** Setup (refusals), the stage schedule (eager fallbacks), or the first capture (out of memory).

**Activation.** Always: the rows describe code paths, not switches.

**Guarantee / limits.** Every row is a refusal or a reported eager fallback; none is silent.

**Evidence.** The Tests layer lists the test that exercises each row, or says that none was found.
:::

### Derivation

None. This is a register of code facts, and no row quotes a measured number.

### Numerical method

None. Where a row changes how a stage runs (eager instead of captured), the stage computes the same thing
either way. Chapter 11 states the graph = eager rule.

### Code

| limit | home | code at the pin |
|---|---|---|
| Multi-rank capture is refused: "more than one process: multi-rank capture is not done yet" | 11.3, 13.4 | `src/mesh/stage_executor.cpp:659-662@73c45a7c` (`StageExecutor::config_reasons`) |
| An EOS outside `ideal-gas`, `ideal-moist`, or a Riemann solver outside `lmars`, `hllc`, runs eager, with the reason named | 11.3 | `src/mesh/stage_executor.cpp:641-652@73c45a7c` |
| An $x_1$ split (`pz > 1`) is refused for capture, and any $x_1$ decomposition for the implicit solve ("[Hydro] implicit scheme requires nb1 = 1") | 13.5, 6.2 | `src/mesh/stage_executor.cpp:663-666@73c45a7c`; `src/hydro/hydro.cpp:442-446@73c45a7c` (`HydroImpl::_apply_implicit_correction`) |
| FP32 on the cubed sphere is refused: FP32 needs `perturbation-flux`, the perturbations need an anomaly background, and anomaly storage is Cartesian only | 12.3 | `src/hydro/hydro.cpp:38-39@73c45a7c`; `src/eos/ideal_moist.cpp:81-82@73c45a7c` (`IdealMoistImpl::perturbations`); `src/hydro/hydro.cpp:144-148@73c45a7c` |
| A registered chemistry step on a bare block is its own eager slot between captured segments; a Mesh's block defers it to the Mesh's global phase, which runs it eagerly after the stages | 11.7 | `src/mesh/stage_executor.cpp:184-190@73c45a7c` (`StageExecutor::chemistry_slot`) |
| MPS has no FP64: the wide dtype is FP32 there | 12.1 | `src/mesh/precision_policy.hpp:52-54@73c45a7c` (`PrecisionPolicy::wide`) |
| MPS runs eager only, because a state not on a CUDA device is an eager reason. The moist column stops before saturation there (no kintera MPS kernel): **unverified** in this draft, since it depends on kintera, which is not read here | 12.1, 16.5 | `src/mesh/stage_executor.cpp:581-588@73c45a7c` (`StageExecutor::vars_reasons`) |
| MPS hard refusal: anomaly storage cannot move to MPS (its reference is FP64); `Fp64StoreImpl::to` stops with an error and does not fall back | 12.3, 12.1 | `src/hydro/fp64_store.hpp:23-49@73c45a7c` (`Fp64StoreImpl`, the check at `:49-50`) |
| MPS hard refusal: a stretched $x_1$ (`geometry/x1f` or `cells/x1rat`) stops the hydro with an error | 3.1, 16.5 | `src/hydro/hydro.cpp:303-308@73c45a7c` (`HydroImpl::_check_stretched_x1`) |
| A capture that runs out of device memory stops with an error naming `execution: eager`; there is no automatic eager fallback | 11.2, 13.2 | `src/mesh/stage_executor.cpp:860-866@73c45a7c` (`StageExecutor::run_segments`) |

### Tests

| test (ctest name) | what it asserts | tolerance | row |
|---|---|---|---|
| `test_graph_capture_lifts_python` | a deck with `roe` runs eager with the reason "riemann solver is roe; capture allows lmars, hllc" | string match | EOS and Riemann allow-lists |
| `test_x1_seam_split_vic_refuses_split.release` (`X1SeamSplit.vic_refuses_x1_split`) | an implicit scheme on an $x_1$ split is refused | the refusal is raised | $x_1$ split, implicit solve |
| `test_graph_mesh_chemistry_python` | a Mesh with chemistry captures its blocks' stages and runs the chemistry in the global phase, eagerly | as in its source | chemistry slot |
| `test_execution_capability_python` | a state not on CUDA lists "not on a CUDA device" among its eager reasons | string match | MPS eager |
| `test_precision_policy.release` | the policy built for an MPS device holds FP32 in its FP64 rows | as in its source | MPS wide dtype |
| `test_mesh_block_job_failure_python` | a capture out of device memory raises one `OutOfMemoryError` naming `execution: eager` | string match | out of memory |
| `test_mps_matches_cpu_python` | MPS paths agree with CPU at FP32; exits 125 (skip) without MPS | as in its source | MPS rows |

No test at the pin was found that asserts the multi-rank capture refusal, the FP32 cubed-sphere chain, or the
`Fp64StoreImpl` MPS refusal. The MPS rows run only on an MPS device. **Run evidence at the pin: missing
evidence** (no run is recorded in this draft).

### Limits

1. The draft does not replace the book chapter's register. The two are to be merged at the `.qmd` conversion.
2. The kintera clause of the MPS-eager row is unverified.
3. The tolerance column says "as in its source" where the cited assertion was not re-read line by line for this
   draft.
