# Style guide for the snapy Technical Report

## 0. Audience

The target reader of the whole book is a first-year PhD student from another group, with a general physics
and numerics background. They have no snapy experience and no atmospheric-model code behind them.

If that reader cannot follow a section, they cannot use our code. Every section must let them understand
what the piece does, why it is there, and how to use it.

Writers test each section against that reader. If a step needs knowledge they would not have, write the step
out or point to where it is explained.

"Would that reader follow this?" is the first question of every review.

---

This file is binding for every chapter author. The worked example is
`chapters/06-gravity-energy/D_face_work_pe.md`; it is a **pre-conversion draft**, so sections 1-9 bind it in
full but section 10 (Quarto markup) does not. A pre-conversion draft may hand-number its equations
(`\tag{N.n}`) and its headings and may write code locations as `path:lines@sha`; it may not deviate from any
other rule. Converting a draft to `book/chapters/NN-slug/_<scheme>.qmd` is a mechanical step with exactly four
parts: every `\tag{N.n}` becomes a `{#eq-chNN-<scheme>-<what>}` label and every `(N.n)` in prose becomes
`([-@eq-chNN-<scheme>-<what>])`; every hand-typed heading number is deleted; every snapy `path:lines@sha`
becomes the `{{< code <path> <symbol> >}}` shortcode of section 3.1, resolved at the pin, and a `path:lines@sha`
in a package with no code pin becomes the link of section 10.6; every Unicode symbol becomes math. The author does the conversion and reruns the review
checklist of section 9 afterwards. Where this file and a converted chapter disagree, this file wins and the
chapter is a bug to report.

The binding example is the converted `.qmd`, not the draft: `book/chapters/06-gravity-energy/_dwork.qmd` is the
first real scheme file and the one every other author copies.

Contents: 1 Scope and voice, 2 the six-layer scheme template, 3 citations, 4 numbers and evidence, 5 equations and
notation, 6 figures, 7 executable checks, 8 directory layout, 9 review checklist, 10 Quarto markup.

---

## 1. Scope and voice

- Describe snapy only: its equations, its discretization, its code, its tests. No other model's code, no cross-code
  comparisons, no machine names, cluster paths, user names or personal paths.
- Write for a reader who knows finite-volume methods and wants to learn snapy well enough to change it. Every
  statement is either derived on the page, cited to code at the pinned sha, or cited to a test with its tolerance.
- Plain, direct prose. Present tense for what the code does ("the solver books the work"), past tense only for
  history ("#296 moved the term into the operator"). No "it can be shown", "obviously", "trivially", "it is easy to
  see". If a step is short, write it; if it is long, write it anyway.
- Name a scheme by what it does first and by its switch second: "the corrected-PE face work
  (`SNAP_GRAVITY_WORK_RADIAL_EXACT`)". Project-internal labels (leg D, option F, #NNN) are given once, in the
  Summary layer, as aliases, so a reader can match them to commit messages and the derivation notes.
- Units: SI unless the section states the nondimensionalisation. Give units of every dimensional symbol on first
  use in a section.
- Source format: Quarto (`.qmd`, Pandoc markdown with LaTeX math), rendered to an HTML site and a PDF from the same
  files. Section 10 gives the markup rules; they are binding. Wrap at about 120 characters.
- Physics is written as connected prose. Bullets are for procedures, lists of switches and checklists only.

### 1.1 Readability

These rules help the reader follow each section and use the code. Readability is checked in every review.

- Each section opens with a plain-language paragraph explaining what the piece does and why, before any symbol.
  The "At a glance" box and any local-symbols note come after this first prose paragraph (section 10.7).
- Write derivations out in full, with every intermediate step on its own numbered or aligned line.
  Never write "it follows that" or "one can show" without the steps. State each assumption where it is used.
- Define every symbol at first use in the section, with units.
- Keep "At a glance" box fields short. Put long evidence (test names, check commands, versions) in the section's
  Evidence/Verification subsection, and keep one line in the box pointing there.
- Keep sentences under about 30 words, with one idea per paragraph.
- Chapter-PR reviews block only on content: wrong claims, skipped derivation steps, undefined or clashing
  symbols, and unclear prose. Cite the paragraph for a `changes` verdict. Box order and box length do not
  block chapter-PR reviews; section 10.7 assigns their mechanical check and readability pass.

The exemplar is the chapter 9 forcing-loop section, `book/chapters/09-diffusion-forcing/_framework.qmd`. Every
chapter copies its five choices:

- A plain opening paragraph before a box of at most six one-line fields.
- Local symbols listed after the box, none clashing with NOTATION.md.
- Every "why" as numbered steps tied to code facts, with round-off given as a derived bound.
- A how-to-use block: the YAML, what is refused and why, and a code excerpt mapped to the equations.
- Tests in the section 10.5 table plus `[Cn]` prose, with the run provenance in a footnote.

## 2. The six-layer scheme template

Every scheme is one file `book/chapters/NN-slug/_<scheme>.qmd`, included into its chapter file (section 10.1), with
exactly these six layers, in this order, with these headings. Do not merge, reorder or rename them. A layer with nothing to say says so in one line
and why ("No known limits beyond those of the base scheme in §5.2.").

```markdown
## <Scheme name, what it does> (`<SWITCH>` or YAML key) {#sec-chNN-<scheme>}

<!-- PROVENANCE
pin: snapy@<sha> (<branch>)
sources: <files>
status: draft | reviewed | approved
author: <name>
reviewer: <name>
-->

### Summary
Open with a plain-language paragraph: what it does and why it is there (the error or failure it removes).

::: {.callout-note title="At a glance"}
**Purpose.** What it does (one line); the one equation that defines it (a reference to the labelled equation).

**Data.** Inputs and outputs, units, mass basis, where they live.

**Placement.** Stage and order in the step.

**Activation.** Switch or key, how it is set, default. Requires / implies / turns off.

**Guarantee / limits.** Order of accuracy, invariant, domain.

**Evidence.** One line pointing to the Tests layer.
:::

Each field label starts its own paragraph: a blank line before every `**Label.**` line after the first
(`python3 tools/check_glance_boxes.py` is the gate; `python3 tools/fix_glance_boxes.py` inserts the blank lines).

Any local-symbols note follows the opening paragraph too. Use at most three prose paragraphs in all, giving
the error order and when the scheme is on. Bullets only for the list of switch couplings. Aliases (PR numbers,
leg names, option letters a reader will meet in the code and history) in the last sentence.

### Derivation
Continuous equation -> integral form over a cell -> discrete form, every step written. Each numbered equation that
the code implements is tagged with the code site that implements it. Ends with a statement of what is exact and
what is truncated, with orders. Points to its executable check (section 7 of this guide).

### Numerical method
The discrete method as the code runs it: stencil (with a figure), where each quantity lives (cell average, cell
centroid point value, face average, face point value), ghosts and how they are filled, walls, block seams, solids,
which RK stage and which part of the step it runs in, explicit vs implicit, order of accuracy (interior, wall
cells, seams) with the evidence for each order.

### Code
A table: | what | code link | symbol | switch | (snapy code cited by the shortcode of section 3.1; the link of
section 10.6 only for a package with no code pin), then a short walk-through in step order. Every
switch read, every setup check, every warning a user can hit is listed.

### Tests
A table: | test (ctest name) | what it asserts | tolerance | where the tolerance comes from |. Then the evidence:
numbers with their sha, deck and run (section 4 of this guide). Say which assertions are covered on CPU only, which
also on CUDA, which on more than one rank.

### Limits
Known limits, failure modes, untested combinations, open issues, each with its evidence or "not measured".
```

**One narrow exception: a section that documents code which is not reachable.** Some of the tree is present but
cannot run — a directory that the build does not compile, a module the parser accepts but never installs, a
mechanism reachable only through an option the code refuses. A section whose whole subject is *that the thing
does not run* has no scheme to describe, so the six layers would be six empty headings. Such a section may be a
short piece of prose with no layer headings, and it must:

1. **say so in its own title** — "not built", "accepted by the parser, never installed", or the like;
2. **prove it from the code at the pin**, citing the specific place that makes it unreachable (the build file
   that omits the directory, the branch that refuses the type), not merely asserting it;
3. **say where the behaviour a reader expected actually lives**, or that it lives nowhere;
4. **stay short**. It is a signpost, not a chapter.

This is an exception for *unreachable code*, not for unfinished writing. A scheme that runs takes all six
layers however thin they are, and "the Tests layer is pending" is a to-do (section 11.1), never grounds for
dropping the headings. `book/chapters/09-diffusion-forcing/_turbulence.qmd` and `_plume.qmd` are the worked
examples.

Rules for the layers:

- **Summary** fits on one screen: the "At a glance" box and at most three paragraphs. It states the default
  explicitly, even when the default is "off".
- **Derivation** has no gaps. When the derivation exists in `sources/` or `docs/derivations/`, re-write it here in
  the report's notation (do not paste), check every step against the code, and cite the source. When no derivation
  exists, re-derive it from the code at the pinned sha and mark the outline entry "re-derive". Never summarise a
  derivation from memory.
- **Numerical method** contains at least one figure (section 6).
- **Code** citations must resolve at the pinned sha (section 3).
- **Tests** never quotes a number without its sha, deck and run; "missing evidence" is an acceptable entry, a bare
  number is not.
- **Limits** includes the switch combinations that are not tested (cross-reference the switch coverage
  matrix, `@sec-ch12-matrix`).

## 3. Citations

### 3.1 Code
A snapy code citation is the shortcode, and nothing else:

```markdown
{{< code src/hydro/hydro_forward.cpp HydroImpl::_flux_covariance >}}
```

It renders as the file's basename and the symbol, linked to that symbol's lines at the pin. A citation names a
**path and a symbol**, never a line number: line numbers move under the text and a reader cannot tell a stale
range from a fresh one. `python3 tools/code_anchors.py check` resolves every citation at the pin and fails on
any that does not resolve; it is the render's pre-render step and a CI step.

- The symbol is a function, method, class, struct, member or variable. C++ names keep their class and drop their
  namespace (`HydroImpl::forward`), gtest tests are `Suite.Name`, Python symbols are `Class.method`. A file as a
  whole is `{{< code <path> >}}`. When the gate reports a symbol as ambiguous it lists the overloads; add the
  `overload=N` it gives.
- Every citation still names its anchor in the prose around it: the identifier the sentence uses before the
  citation, and the "symbol" column of the Code table. A citation the surrounding sentence does not name is a
  review failure even when it resolves.
- Hand-written line numbers are forbidden in a `.qmd`, in both the link form and the `path:lines@sha` notation.
  The STYLE 10.6 link survives in exactly one place: a repository with no code pin. Everything in a repository
  that has a pin is cited by symbol.
- **The enclosing definition.** Where the thing a sentence names is not a citable symbol at the pin - a local
  variable, a lambda, a label inside a function, a macro, or a passage of top-level code - cite the enclosing
  definition instead, and keep the named thing in backticks in the prose so the reader still has the anchor:
  `sqrtdl` ({{< code src/riemann/roe_impl.h roe_impl >}}). A citation of this kind is registered in
  `reviews/code-anchors-unmapped.tsv` with the symbol actually cited and the reason the named anchor is not one,
  so the coarser citation is a recorded decision rather than an accident. Where no definition encloses the lines
  either, cite the file as a whole, `{{< code <path> >}}`, and register that.
- Tests are cited by file and ctest name: `tests/test_gravity_work_radial_exact.py` (ctest
  `test_gravity_work_radial_exact_python`).

#### The pin

**One pin per package, stated once, in one file.** For snapy that file is `code_pin.yml`, and it is the single
source of truth:

```yaml
repo: chengcli/snapy
sha: <the 40-character sha, written only here>
```

This guide does not restate that sha as the pin, and neither does any other file. Read it from `code_pin.yml`;
move it only with `python3 tools/code_anchors.py bump <full-sha>`, which refuses to write a new pin until every
citation resolves there and lists the symbols whose body changed so their claims can be re-read.

**One named exception, and only this one.** The change-history table immediately below writes shas out, including
the current pin. That table is **provenance, not a citation**: it records what three files said at one past
commit, which is evidence that cannot be stated without the shas. It is never a source to read the pin from, it
is not updated when the pin moves, and nothing else in this guide or in any `.qmd` may restate a sha on the same
excuse.

**So where may a sha be written at all?** Exactly four places, and nowhere else: `code_pin.yml`, which is the
pin; a PROVENANCE block, which records what a section was written against; a `[source: ...]`, `[sha: ...]` or
`[^label]:` provenance footnote; and the change-history table named above. A URL for a package with no code
pin carries its sha inside the link, which is the link's own form (section 10.6), not prose. **A sha in body
prose, outside those four, is a defect** — including in this guide, which holds itself to the same list.

A sha written anywhere else is a copy, and copies go stale. They had. Read at tech-report
`f38085bb02c2043b7002bfeb85a47601f52cc2d7`, three files stated a snapy pin, and the two prose copies agreed
with each other and disagreed with the one the tools use —

| where, at that sha | what it said |
|---|---|
| `code_pin.yml:3-4` | `repo: chengcli/snapy`, `sha: 68f4e066974ede87e4379285c4fba895b970b7f1` — what every `{{< code >}}` actually resolves at |
| `STYLE.md:137-140` (this section, before this revision) | a four-row pin table: snapy base `aea71ed852effb09e6aa155dd26349f1210ef556`, snapy round `e894700ff7aee30b52882e5202b16461413780b0`, kintera `c55b13b2204997d2d09e04498558ab9495d8ee77`, pyharp `4721715855e937c1e8b218e964c0655f46e56e29`; no pydisort or commux row |
| `book/index.qmd:9-10` | "Code is cited at the snapy main pin", then `e894700ff7aee30b52882e5202b16461413780b0`, described as "reproduced" from this section |

- `index.qmd` names the pinned repository and links to `code_pin.yml`; it does not carry a sha.
- A PROVENANCE block records the pin the section was written against (section 3.2). Those blocks are the one
  place a sha is written out per file, and they are the editor's to move with the pin.
- **Open slots.** `code_pin.yml` holds one `repo:`/`sha:` pair and `tools/code_anchors.py` reads only those two
  fields, so exactly one package can be pinned there today. **snapy has an entry; kintera, pyharp, pydisort and
  commux have none**, so they have no gate either. These are open slots, not shas waiting to be copied in: where
  a package has no entry, this guide says so rather than inventing one.

  | package | entry in `code_pin.yml` | pin, until it has one | cited in `book/**/*.qmd` |
  |---|---|---|---|
  | snapy (`chengcli/snapy`) | **yes** | read it from `code_pin.yml` | 582 links in 100 files |
  | kintera (`chengcli/kintera`) | **no** | `c55b13b2204997d2d09e04498558ab9495d8ee77`, stated here only | 41 links in 7 files |
  | pyharp (`chengcli/pyharp`) | **no** | `4721715855e937c1e8b218e964c0655f46e56e29`, stated here only | 16 links in 6 files |
  | pydisort (`zoeyzyhu/pydisort`) | **no** | **none recorded anywhere** | 0 |
  | commux (`zoeyzyhu/commux`) | **no** | **none recorded anywhere** | 0 |

  kintera and pyharp are cited without a gate, so a human checks them; pydisort and commux are allowed to be
  cited but are not cited yet, and the first citation to either needs a pin decided first. Extending
  `code_pin.yml` to a mapping of package to pin, and `code_anchors.py` with it, closes all four slots and is the
  concrete follow-up.
- The repositories the report may cite, each from one owner only: snapy (`chengcli/snapy`), kintera
  (`chengcli/kintera`), pyharp (`chengcli/pyharp`), pydisort (`zoeyzyhu/pydisort`), commux (`zoeyzyhu/commux`).
  Any other repository, or a fork off that list, is a review failure. A sha from a fork resolves through the
  parent on GitHub; cite the repository the commit lives on.

#### When a cited symbol changes at the pin

**PROVISIONAL — pending the editor's ruling. Do not treat the numbers below as settled practice.**

Moving the pin can rename a symbol, move it to another file, split it in two, or delete it. `code_anchors.py
bump` lists every cited symbol whose body changed, and refuses the move while any citation fails to resolve.
When that happens:

1. **Rewrite the citation to the symbol the code now has, and keep the pin.** The pin does not move backwards
   for one citation, and the citation is not deleted to make a gate pass. Dropping either hides the change
   instead of recording it.
2. **Re-read the claim beside it.** A symbol whose body changed is the usual reason a sentence has quietly gone
   false; the point of the list is the claim, not the link.
3. **Where the symbol is genuinely gone** — the behaviour was removed, not renamed — the claim goes with it. Cite
   what the code does now, or delete the sentence. A citation to a symbol that no longer exists is never correct.
4. **Flag every one of these in the PR body**, one line each: the old symbol, the new symbol, and whether the
   claim changed. A reviewer reads that list first.

**Legacy chapters.** `tools/code-anchors-legacy.txt` lists the chapters whose citations predate the pin: there a
citation that does not resolve is reported by `code_anchors.py check`, not failed, and renders as plain text; every
other chapter is checked in full. A chapter PR deletes its own line and must then pass `check` in full (a listed
chapter with nothing unresolved fails `check` until its line goes). The report is complete only when the list is
empty; `check` prints the count left per chapter and per owner. The contract generators (`tools/gen_*_contract.py`,
`gen_scheme_native.py`), the reader-reference Riemann block and their tests skip a target in a listed chapter the
same way, naming the list in the skip reason; deleting the line runs them in full again.

### 3.2 Sources, provenance, and no history in the text

The report describes the code **as it is at the pin**, as one coherent text. A reader opens a chapter to learn
what snapy does today, not how it got here. So the body text carries no change history:

- no "since `chengcli/snapy#307`", "before `#303`", "after `snapy@ae61e39`";
- no "main after ...", and no per-section note saying which sha a passage "was checked at";
- no PR or issue number in prose at all.

Write the behaviour plainly — "the limiter also runs when the covariance rows are on" — not the story of the
commit that introduced it. Two consequences worth stating, because both have been got wrong:

- A sentence that reads "at the pin X was true; since `#NNN` it is Y" is not merely stale style. Once `#NNN` is
  *in* the pin, that sentence is **false**, because it describes the pin's own content as something later.
- Deleting history does not mean deleting evidence. A measurement keeps its numbers and its provenance
  (section 4); only the narrative framing goes.

**Where provenance does belong**

- The **PROVENANCE block** at the top of each scheme file: the pin the section was written against, its sources,
  and its status. This is the one place per file that records where the text came from.
- A **`[source: ...]` or `[sha: ...]` footnote** attached to a specific number or claim, **or a markdown
  footnote definition, `[^label]: ...`**, carrying the same thing. Both forms are provenance and both are
  legal; the markdown form is the one to use when the note is long enough to interrupt the sentence, e.g.
  `[^ch07-cfl-src]: Source and credit: the implicit-advective and shear bounds are those of ...`. A PR number
  may appear in either — it is a credit and a trail, not narration — alongside the deck and the run.
- `ISSUES.md` and the review record for anything still open.

Measured over `book/chapters/**/*.qmd` at tech-report `f38085bb02c2043b7002bfeb85a47601f52cc2d7`, excluding
PROVENANCE blocks, the text still carries **103** "since/before/after <PR or sha>" in 23 files, **58**
"main after/at ..." in 15 files, **2** per-section "links at the pin <sha>" notes in 2 files, and **207** PR
numbers in prose in 49 files. Those are the editor's coherence pass; an author clears them in whatever
paragraph they touch and does not sweep the rest of the book in a content PR.

- Derivation notes: `sources/<file> §n` for a copy in this directory, or `docs/derivations/<file>@<full-sha>
  §n` for one in the code tree. The path, the sha and the section are repeated in full at every citation; no
  abbreviation to `§n` alone, and no second form for the same file within a chapter. A script is cited by path
  and sha with a function or a printed label, never a "§": `docs/derivations/optionF_replica.py@<full-sha>`,
  output label `[C6]`. A note or script at a sha that is not one of the pins of section 3.1 is cited only in
  the Limits or Tests layer, with the reason it is off-pin and what it would take to re-run at the pin.
- A derivation note that is still being edited is cited at the pin like any other file, but a statement *about*
  the note — that it has an error, a gap or a placeholder — is a statement about a moving target. Such a
  statement is written in this form: "at `<file>@<full-sha>` §n, ... ; check the note's current head before
  publication". The editor re-reads every such statement against the note's branch tip as the last step before
  each release render, and the author either removes the statement or re-pins it. The same applies to any
  sentence that says a defect was "flagged to its author".
- Pull requests and issues, **in a PROVENANCE block, a `[source: ...]` footnote or a markdown `[^label]:`
  footnote definition only**, never in prose: `chengcli/snapy#NNN` or `UCzhangxi/snapy#NNN`, always with the
  owner/repo.
- Commit: `snapy@<full-sha>`, the full 40 characters, with its subject line in quotes when the subject carries
  the meaning. A short sha appears nowhere in a rendered file: it is not stable as the repository grows and
  `git fetch` does not accept it. Where a short form is wanted for reading, write the full sha and let the link
  text carry the short form. Two exceptions, and no others:
  - **a short sha may appear only inside an example that is explicitly labelled as the old or forbidden form**,
    so that a rule can show what it converts away from;
  - planning files are exempt (section 10.2): OUTLINE.md and the sources keep the short-sha notation of
    section 3.1, and it is converted when the text moves into a `.qmd` — to the shortcode for snapy, to the
    link of section 10.6 for a package with no code pin. They are never swept for short shas, and a short sha
    in one of them is not a defect.

## 4. Numbers and evidence

Every number that is a measurement (an error, a drift, a rate, a timing, a growth) carries a provenance tag:

> max per-step $|\Delta(E+P)|/|E+P| = 3.9\times10^{-16}$ [sha: the code pin; deck:
> `tests/test_gravity_work_radial_exact.py` case `sph_vic`; run: `ctest -R test_gravity_work_radial_exact_python`,
> CPU, double, 1 rank; Python 3.12.3, numpy 2.4.6].

- **sha**: the code that produced it.
- **deck**: the input (YAML file, or the test/script and case that builds the input).
- **run**: the **exact command**, then the device, precision and ranks, then the **versions that can move the
  digits** — interpreter, and the numerical libraries the result goes through (`numpy`, `sympy`, `torch`, BLAS).
  "numpy, double" is not a run record: it names no command and no version, so nobody can tell a wrong number
  from a differently-rounded one.

**A quoted number must be reproducible by the reader from what the page says.** It therefore comes from a
committed check or a recorded run, and the record carries the command and the versions. Two failure modes this
rule exists to stop:

- *A number with no command.* The reader cannot re-run it, so it cannot be checked, only believed.
- *A number with a command but no versions.* It can be re-run and will disagree, and nobody can say whether the
  code changed or the host did. Three checks in this report are in exactly that state: `cellface_check` (ch06),
  `fixvapor_check` and `kinetics_check` (ch10) print a round-off magnitude whose committed value this project's
  own machines no longer reproduce (`1.3e-17` against a committed `5.0e-17`; `1.79e-15` against `2.20e-15`;
  `1.6e-15` against `2.1e-15`), and their provenance says only "numpy, double". Every claim still passes its
  threshold, so nothing is wrong with the code — the records are simply too thin to prove it.

  Where a reported quantity is pure round-off noise, say so and compare it with its threshold ("below
  $10^{-13}$ relative") rather than printing digits that no two machines agree on. A byte-compared `.out` file
  must not contain a number the text cannot promise.
- A number copied from a source note keeps the note's sha; if the note has none, the number is marked
  **missing evidence** and is either re-run at the pinned sha or removed (ISSUES.md items 2, 3, 5).
- A number from an executable check in this report cites the check script and its output file:
  [check `src/snapy_report/ch06/checks/dwork_check.py`, output `src/snapy_report/ch06/checks/dwork_check.out`].
- Tolerances are numbers with a reason: "1e-14 relative: 20 steps of round-off on a sum of 32 terms of size 1e2 with
  double precision", not "small".
- Orders of accuracy are fitted slopes from at least three resolutions, with the resolutions.

## 5. Equations and notation

- All symbols are defined in `NOTATION.md`, one meaning each. A chapter may introduce a local symbol only if
  NOTATION.md has no symbol for the concept. It then does three things: defines the symbol at first use, lists
  it in a "Local symbols" note at the top of the scheme file, and opens an ISSUES.md item naming the symbol and
  the scheme. When the editor adds the symbol to NOTATION.md, the editor closes the item and **deletes the entry
  from the chapter's local-symbol note**; a note that still lists a symbol NOTATION.md defines is a review
  failure (section 9, item 2). A chapter never redefines a symbol NOTATION.md already has, even to narrow it.
- Indices: cell $i$ (x1), $j$ (x2), $k$ (x3); faces at half integers $i\pm\tfrac12$. Code index names (`is`, `ie`,
  `il`, `iu`) appear only in the Code layer, mapped to math indices once.
- Cell averages carry an overbar only where a point value of the same quantity also appears: $\bar\rho_i$ against
  $\rho(x_{1,i})$. In a section that uses only cell averages, say so and drop the bar.
- Scripts are written in this order and no other: quantity, then the upright kind-label as a superscript, then
  the space index as a subscript, then the time level or RK stage as a trailing superscript in parentheses:
  $W^{\mathrm{D}}_i$, $F^{\mathrm{ref}}_{i+1/2}$, $\rho_i^{n}$, $\mathbf U^{(s)}$, $\Delta\mathbf U^{(0)}_{i}$,
  $\mathsf A^{(E,\rho)}_{i}$. Time levels are $n$ (step) and $(s)$ (RK stage), never both at once; a half time
  level is $n+\tfrac12$. Matrix block labels $(E,\rho)$ (row, column) are a superscript in parentheses. A bare
  subscript $0$ is a cell index only; a background or reference state is subscript $\mathrm{ref}$ (NOTATION.md
  §6), and a domain end is named ($r_{\mathrm{in}}$, $z_{\mathrm{bot}}$).
- Superscript $(0)$ is reserved for the explicit RK increment, $\Delta\mathbf U^{(0)}$, and means nothing else.
  A quantity at the thermodynamic reference state carries a star, not a zero: $u^{\star}_n$, $s^{\star}_n$, to
  match the $T_\star, p_\star$ of NOTATION.md §2a. Never write $u^{(0)}_n$ for a reference energy; the two
  would otherwise collide in chapter 6, which uses both.
- Face quantities: subscript $f$ or $i\pm\tfrac12$. Operators are only those of NOTATION.md §4, written exactly
  as that table writes them: $\Delta_i[q]$, $\langle q\rangle_{V_i}$, $\nabla_{\!1}\!\cdot G$, $s_i[q]$,
  $\operatorname{cov}_i(a,b)$, $[q]_{\mathrm{walls}}$. A chapter that needs another operator defines it at first
  use and the editor adds it to §4; it does not invent a short form for an operator the table already has.
- Equations that are referenced or that the code implements carry a Quarto label and are numbered by Quarto
  (section 10.3); never type an equation number by hand. Each such equation is tagged in the text with its code site.
- Orders: write $O(h^2)$ with $h$ the local x1 cell width, or $\Delta x_1$ when the section is about x1 specifically.
  "Second order" means the error of the quantity being discussed, which is named.
- Signs: gravity $g_1$ = `grav1` is negative when it points to decreasing $x_1$. Potentials are $\phi = -g_1x_1$
  (increasing upward). Never write $g$ for $|g_1|$ without defining it.
- Code identifiers in prose are in backticks. Math symbols are never in backticks.
- Upright sub- and superscripts use `\mathrm{}`: `W^{\mathrm{D}}`, `p_{\mathrm{sat}}`. Never `\rm`, `\bf`, `\it`, `\cal`:
  the KOMA class of the PDF removes them and the PDF build fails.
- Bold Greek with `\boldsymbol{}`, never `\mathbf{}` (which drops Greek). Bold Latin vectors with `\mathbf{}`.
- No math macros: write every symbol out (`\overline{\rho}_i`, `\mathrm{d}`, `\partial`). The HTML uses MathML
  (as the house books do), which has no macro layer.
- Differential and operator symbols are upright: `\mathrm{d}` in `\mathrm{d}V`, `\mathrm{d}x_1`,
  `\frac{\mathrm{d}}{\mathrm{d}t}`; `\partial` for partial derivatives; `\mathrm{D}/\mathrm{D}t` for a material
  derivative. Italic $d$ is never a differential; it is free for a symbol. A prime denotes a derivative with
  respect to $x_1$ only where the section says so in words at first use, and never in a chapter that also uses
  the perturbation notation of NOTATION.md §6 (see NOTATION.md §10).
- Approximate values: the number goes inside the math, `$\sim\!0.1$`, never `$\sim$0.1`.
- Relations and arrows in prose are math too: `$\le$`, `$\to$`, `$\times$`; never the Unicode characters ≤, →, ×.

## 6. Figures

Every scheme gets at least one cartoon (stencil, fluxes, faces vs cells, ghosts, walls, seams) and, where the scheme
has a measured property, one data figure. All figures are produced by committed scripts; no hand-drawn or
screenshot images.

### 6.1 Layout
```
src/snapy_report/chNN/figures/fig_<scheme>_<what>.py  # one module per figure: make_fig() -> matplotlib Figure
src/snapy_report/chNN/data/                            # the committed data a data figure reads
src/snapy_report/figstyle.py                           # shared style: palette, fonts, sizes, helpers
FIGURES.md                                             # the manifest: figure -> code -> data (tools/render_figures.py)
```
- A figure is drawn by a function in the `snapy_report` package and placed in the chapter by a Quarto code cell
  (section 10.4), so the figure, its caption and its label live in the chapter and its provenance in the package.
  The function imports only numpy, matplotlib and `figstyle`. A data figure reads its data from a committed output
  file of an executable check or a run (`src/snapy_report/chNN/data/`), never from a live run.
- The function's docstring states what the figure shows, its sha, and the data file it reads.
- `quarto render` re-draws every figure (with `freeze: auto`, only changed chapters re-run); a figure that differs
  after a clean re-render is a bug.

### 6.2 Look
- Size: full width 6.2 in (the PDF text width), half width 3.05 in; height as needed. Vector output (SVG for HTML, PDF for
  the PDF; section 10.4); no raster line drawings.
- Fonts: whatever `snapy_report.figstyle` sets, which a figure never restates for itself — **serif text with
  STIX, not sans**: `font.family: serif`, `font.serif: ["STIXGeneral", "DejaVu Serif"]` (DejaVu Serif is the
  fallback, not the choice) and `mathtext.fontset: stix`, at 9 pt for text and labels and 8 pt for ticks,
  annotations and legends (`src/snapy_report/figstyle.py:33-43`, with the same values as package defaults in
  `src/snapy_report/__init__.py:11-16`). Fonts embed as TrueType, `pdf.fonttype: 42`.
  Do not depend on a TeX installation: this is matplotlib's own mathtext, not LaTeX.

  **Two chapter modules do not follow this yet, and a reader should not be told otherwise.** Measured in the
  tree as this was written, `grep -rnE 'font\.family|mathtext\.fontset' src/snapy_report/`:

  | module | what it sets | diverges in |
  |---|---|---|
  | `src/snapy_report/figstyle.py:33-35`, `__init__.py:13-15` | `serif` / `STIXGeneral` / `stix` | — the rule above |
  | `src/snapy_report/ch05/_style.py:39,41` | `DejaVu Sans` / `dejavusans` | font family and mathtext |
  | `src/snapy_report/ch09/_style.py:37` | `DejaVu Sans` (mathtext not set) | font family |

  These two are **named exceptions of record, not permission**: no new chapter module may add a third stack,
  and a chapter that is rewritten moves onto `figstyle`. `ch05/_style.py:32-33` also sets its widths to
  `SINGLE = 3.4` and `DOUBLE = 7.0` while its own docstring calls them "the two widths of STYLE 6.2"; section
  6.2 gives 3.05 in and 6.2 in, so that docstring is wrong whichever widths are eventually chosen.
- Palette: Okabe-Ito, colour-blind safe, used in this order and with these meanings across the report where
  applicable:

  | name | hex | default use |
  |---|---|---|
  | black | `#000000` | grid, walls, axes |
  | orange | `#E69F00` | face quantities, fluxes |
  | sky blue | `#56B4E9` | cell quantities |
  | bluish green | `#009E73` | the scheme being described (its added term) |
  | yellow | `#F0E442` | highlights (use sparingly; poor on white) |
  | blue | `#0072B2` | reference / exact solution |
  | vermillion | `#D55E00` | errors, defects, the term removed |
  | reddish purple | `#CC79A7` | ghosts, other blocks |
  | light blue (fill) | `#EAF5FC` | cell fill in cartoons |
  | light reddish purple (fill) | `#F7E6EF` | ghost-cell fill in cartoons |

  Never encode meaning by colour alone: pair colour with marker shape, line style or a label.
- Ghost cells are hatched (`//`), solids are cross-hatched (`xx`), walls are thick black lines, block seams are
  dashed reddish purple lines.
- Axes are labelled with quantity and unit ("$x_1$ [m]", "relative error [-]"). Log axes for convergence; a
  reference slope triangle or line labelled with its order.
- Cartoons have no axes frame; they label every cell index and face index that the text refers to.
- Shared conventions, drawn only through helpers in `figstyle` (never re-drawn by hand): cell box; cell value =
  sky-blue circle; face value = orange triangle; wall = thick black line; ghost cell = hatched box with the light
  reddish-purple fill `#F7E6EF`; cell fill, where needed, the light blue `#EAF5FC`. Stencil cells are indexed $l$.
- At most 4 panels per figure; a cartoon and a data plot are separate figures. Panels that compare share the y range
  and the y label.
- Text in a figure is never smaller than 8 pt at print size; labels never overlap data or other labels.
- The caption, not the body text, defines every marker, line style and colour.
- Check every figure once in greyscale: it must still read.

## 7. Executable checks

Every derivation has an executable check committed next to it: the identities and closed forms of the Derivation
layer are verified symbolically (sympy) or numerically (numpy) from the discrete formulas.

### 7.1 Layout
```
src/snapy_report/chNN/checks/<scheme>_check.py   # the check
src/snapy_report/chNN/checks/<scheme>_check.out  # its committed output (stdout), regenerated by running the script
src/tests/test_checks.py                          # pytest runs every check; a failing check fails the build
```
- Runs with `python3 <scheme>_check.py` from any directory, needs only numpy and sympy, finishes in under a minute
  on a laptop, and exits non-zero if any assertion fails. It prints every number the chapter quotes from it, each on
  a line with a stable label, so the chapter can cite "check line `[C3]`".
- Structure: a header docstring (what is checked, the sha whose formulas it mirrors, how to run), then one function
  per numbered claim `check_C1_...`, `check_C2_...`, each printing `[Cn] PASS/FAIL <label>: <numbers>` and returning
  a bool, then `main()` that runs all and exits with the failure count.
- A check that mirrors code (a numpy port of a C++ function) says so, cites the code lines it ports, and is written
  line for line against them so a reviewer can diff the two by eye.
- Tolerances in checks follow section 4: each is stated with its reason.
- A check does not import snapy. Checks that need snapy are tests in `tests/` and are cited in the Tests layer.

## 8. Directory layout

```
doc/tech-report/
    README.md  BRIEF_*.md  ISSUES.md  STATUS.md
    STYLE.md  NOTATION.md  OUTLINE.md
    sources/                      # inputs, read-only
    book/                         # the Quarto project
        _quarto.yml               # book config, chapter list, math macros (HTML and PDF), crossref settings
        _freeze/                  # committed: frozen cell outputs (freeze: auto)
        index.qmd  references.qmd  references.bib  apj.csl
        chapters/NN-slug.qmd      # one file per chapter; schemes are its level-2 sections
        figs/                     # static images only (none expected)
    src/                          # python: package snapy_report (figures, checks), tests/ (pytest)
    reviews/                      # review notes, one file per review round
    tools/                        # scripts: citation checker, pin mover
```

## 9. Checklists and the review record

Two companion files hold the operational lists, so this guide can stay the rules and they can stay runnable:

- **`AUTHOR-CHECKLIST.md`** — the gates an author runs before opening a chapter PR, with the exact command and
  what each must say; what belongs in the PR body; what the text may not contain; and the questions that go to
  the editor instead of being settled in a chapter PR.
- **`REVIEW-RECORD.md`** — the template a reviewer fills: head, tree and base shas with the command and the UTC
  time each was read, the gate table compared against the base, the render run, what was read and what was not,
  the findings, and the verdict.

A review is of one head sha. If the head moves, the record is void and the review is redone.

The substance an author is checking for:

1. Six layers present, in order, with the headings of section 2; the book renders to HTML and PDF with no
   unresolved cross-reference and no warning from the chapter (section 10.9).
2. Every symbol in NOTATION.md or defined locally and flagged to the editor.
3. Every snapy citation is the shortcode of section 3.1, resolves at the pin, and the sentence around it names
   its symbol. No hand-written line numbers.
4. Every number carries its command and the versions behind it, or is marked missing evidence (section 4).
5. No change history in the body text (section 3.2).
6. Every figure re-renders from its script with no diff; colour-blind palette; labelled axes and units.
7. Every derivation's check runs, passes, and its `.out` is committed and current.
8. No machine names, cluster paths, personal paths; no other model's code.
9. Switch, default and couplings in the Summary agree with the switch coverage matrix,
   `@sec-ch12-matrix`.

## 10. Quarto markup (binding)

The report is a Quarto book (Quarto 1.10), the same toolchain as the project owner's other books. Content is Pandoc
markdown in `.qmd` files. The HTML site (MathJax) and the PDF (LuaLaTeX, KOMA `scrbook`) are both built from the same
files, so every construct below must work in both. When in doubt, render both and look.

### 10.1 Files and structure
- One file per chapter, `book/chapters/NN-slug.qmd`, listed in `book/_quarto.yml` under `book: chapters:` (grouped in
  `part:` entries as in OUTLINE.md). Only the editor edits `_quarto.yml` and the chapter files.
- The chapter file holds only: the chapter heading with its label (`# Gravity and energy {#sec-ch06}`), the chapter
  opening (below), and one include per scheme, in reading order: `{{< include 06-gravity-energy/_dwork.qmd >}}`.
  Each scheme lives in its own file `book/chapters/NN-slug/_<scheme>.qmd` (the leading underscore stops Quarto from
  rendering it alone), owned by one author. Includes are inlined before cross-references resolve, so labels work
  across files. Two authors never edit the same file.
- The chapter opening, written by the chapter's lead author: one paragraph on what the chapter covers; a dependency
  map (a figure: schemes as nodes, "requires / implies / turns off" as edges, from chapter 12's matrix); and a table
  with one row per scheme: scheme, switch, default, order, main test.
- Chapter identifiers are fixed now from OUTLINE.md and never change: `ch01` ... `ch17`; split chapters use a letter,
  `ch04a`, `ch04b`, `ch07a`, `ch07b`, `ch14a` ... `ch14c`; appendices `appa`, `appb`, ... (Quarto numbers appendices
  A, B, ...).
  Scheme numbers inside a chapter are fixed from OUTLINE.md at the same time as the chapter identifiers and do
  not change either; a cross-reference to a scheme uses its label (`@sec-ch12-matrix`), never its number.
- Each scheme is a level-2 section with a label, `## ... {#sec-ch06-dwork}`. Its six layers are level-3 sections with
  exactly the headings of section 2. Deeper structure inside a layer uses level 4 (`####`) at most. Quarto numbers all
  sections (`number-sections: true`); never type a section number.
- Label syntax for every cross-reference target: `<kind>-ch<NN>-<scheme>[-<what>]`, lower case, hyphens only, unique
  in the book. Kinds: `sec`, `eq`, `fig`, `tbl`, and `prp`/`lem` for stated identities (section 10.7). Examples: `sec-ch06-dwork`, `eq-ch06-dwork-pe`,
  `fig-ch06-dwork-stencil`, `tbl-ch06-dwork-code`.
- References in prose use Quarto's syntax only: `@sec-ch05-wbref`, `@eq-ch06-dwork-pe`, `@fig-...`, `@tbl-...`. Write
  "as shown in @fig-ch06-dwork-stencil", not "Figure 3" or "the figure above". A bare equation number is always
  written in parentheses, `([-@eq-ch06-dwork-pe])`; a range is `([-@eq-a])–([-@eq-b])`.
- The table of contents lists chapters and schemes only (`toc-depth: 2`; the printed PDF also lists sections,
  section 10.10). Layers and their subsections are numbered
  and may be referenced (`@sec-...` on a `###` or `####` heading that the text cites). An `{.unnumbered}` heading is
  linked as `[text](#sec-id)`, never with `@sec`.

### Reader standards

The following reader standards govern content and navigation. They supersede any earlier chapters-only printed TOC rule.
Implementation is staged: the registry, rendered glossary and N1/N4 gates precede the N2 generator; the N9
audit and warning lint follow in a separate change. These requirements do not claim that those gates exist yet.

- G0 Rendered-book evidence: reader-facing gates run on the rendered book (PDF, `.toc` and `.aux`), not on source
  files. A source preflight may diagnose an error early, but cannot establish a reader-facing pass; the checks
  must verify the destination, definition or note that the reader can actually reach in the rendered output.
- N1 Navigation: printed TOC to section depth; one concept registry, `tools/reader-glossary.json`, records each
  concept's defining-section id. Every registry concept is reachable from the TOC or the generated topic index.
  Index entries are kept in data, not chapter text. Every concept in a grouped or roll-up section gets its own
  topic-index entry resolving to its registry defining-section id, not the group section. Add a label if the
  concept section has none. The planned N1 rendered gate checks these destinations. Plain-word titles remain a
  review-checklist item. A rendered glossary appendix is generated from the same registry (N4), not maintained
  as a second list.
- N2 Task map: the "How do I ...?" appendix is task-indexed and section-anchored. Generate its rows from the N6
  blocks, DO cards and the curated reader-task list (seeded from both question banks and every held-out
  NO-ENTRY). Each row supplies the anchor of the section that answers the task, never a chapter or group
  roll-up, or an explicit `not provided: <reason>`. Add a section label where needed; the N2 generator and an
  `.aux` anchor check verify the answering-section destination. Preserve status = answered / partly documented /
  undocumented / unsupported by the code / outside scope; a pointer is not by itself evidence that the task is answered.
- N3 Render hygiene: the rendered text has no editorial markers (ISSUES.md, NOTATION.md, STYLE.md, "flagged to the
  editor"), no unresolved or raw @sec- references.
- N4 Glossary: acronyms, YAML keys, symbols and named concepts use the N1 registry as their single source.
  Generate chapter-first-use notes and the rendered glossary appendix from it; check symbol meanings against
  appendix A. Each entry points to its defining section. Where the book has no definition, say "not defined
  here" explicitly rather than inventing one, and keep the concept reachable under N1. Ratchet the count of
  those entries so it cannot rise: the registry PR sets the baseline from the number marked "not defined here"
  at the commit that lands the registry. Before that baseline exists, honestly undefined named concepts may be
  added; later lowerings are recorded only as verified reductions. The registry, appendix and rendered N1/N4
  checks implement this together in their assigned lane.
- N5 Consistency: defaults and install lines generated from the code at the pin; the rest (equations, assumptions,
  units, API behaviour) is a review-checklist item.
- N6 Extension contracts: one "How to add one" block per extension point (forcing, boundary callback, species,
  scheme, collective): signature and shapes incl. ghosts generated from source, registration, a minimal CI test,
  dtype/device/mutation/ownership rules. Exemplar: the forcing (ch09)
  ["How to add one" block](book/chapters/09-diffusion-forcing/_howto.qmd), generated by
  `tools/gen_forcing_contract.py`: step-0 setup, Python recipe, linked contract, runtime record and N6 gate.
- N7 Provenance: every quoted result carries a full sha that resolves and the command; performance adds timing
  scope and repetitions.
- N8 Runnable cards: CPU DO cards run in tools/tests (seconds); GPU/multi-rank cards in a validation tier with the
  evidence date. Troubleshooting only for crash / NaN / restart, as symptom -> diagnostic -> action.
- N9 Limitation notes: replace free prose such as "cannot show", "see elsewhere" or "beyond scope" with a
  structured note: `answered at <anchor>` or `not provided: <reason>`. The first resolves to the section that
  supplies the answer; the second states what is absent and why, without implying an answer exists elsewhere.
  A warning-level lint flags unlinked prose of this kind. The audit pass and lint are a later implementation;
  the lint warns rather than failing the build.
- Scope test for content: in if it states a fact about this code at the pin (convention, contract, assumption);
  out if it is a tutorial or a science study (then a "not provided" row). In: a conventions table (coordinates,
  units, signs, no pressure coordinate), the output contract, the H2-He assumptions snapy uses, the initial-state
  contract. Out: bug-report procedure, cross-model benchmarks, spin-up, regridding, damping sensitivity.

N3 checks run with `python3 -m pytest tools/tests/test_reader_hygiene.py` and
`python3 tools/check_reader_hygiene.py --book book`. After rendering, run the same tool with
`--html book/_book` and `--pdf book/_book/<book-name>.pdf`. The rendered checks reject raw section references
including those inside displayed code; the source preflight permits references Quarto will resolve in prose.
The existing render gate checks resolution. Editorial comments that do not render remain permitted.

### 10.2 Inline text
- Inline code (identifiers, switches, YAML keys, file names): backticks. Math symbols: `$...$`, never backticks.
- OUTLINE.md and the sources are planning files and are not rendered. They may use Unicode symbols, and they
  may use the short-sha citation notation `path:lines@<short-sha>` of section 3.1. Text copied from them into a
  `.qmd` is converted on the way: Unicode to math mode, and every short sha either to the shortcode of
  section 3.1 when it is snapy, or to the full-sha link of section 10.6 when its package has no code pin. Because the conversion happens at that point and not before, neither a Unicode symbol nor a short sha in
  a planning file is a defect, and these files are never swept for either. NOTATION.md is different: it is rendered,
  as Appendix A. The editor keeps it in the markup of this section (ASCII only, every symbol in `$...$`, pipe
  tables within the column limits of section 10.5) and converts it to `book/chapters/appa-notation.qmd` with the
  label `{#sec-appa}`; chapters reference it as `@sec-appa`. Authors edit NOTATION.md, never the generated
  `.qmd`.
- Units, chemical formulas, relations and arrows always in math mode: `$\mathrm{m\,s^{-2}}$`, `$\mathrm{H_2O}$`,
  `$\le$`, `$\to$`. Never Unicode superscripts, subscripts or symbols (`s⁻¹`, `H₂O`, ≤, →, ×): the PDF can drop them
  silently.
- No raw HTML and no raw LaTeX in prose. The only exception is a `{=latex}` block approved by the editor.
- Emphasis: `*italic*` for a defined term at first use, `**bold**` only in Summary bullets and table headers.
- Code locations appear only as the shortcode of section 3.1, or — for a package with no code pin — as a link
  with the short text of section 10.6. A bare path is never written out: long paths do not wrap in the PDF
  (tested: a full path overflowed the line by 180 pt and, in a table, printed over the next column).

### 10.3 Equations
- Display math: `$$ ... $$` on its own lines, followed by a label when the equation is referenced or implemented by
  the code:
  ```markdown
  $$
  P = \mathrm{PE}_d - g_1 \sum_i V_i \sigma_i^2 s_i[\rho]
  $$ {#eq-ch06-dwork-pe}
  ```
  Quarto numbers it per chapter, as "(6.4)" in the PDF and "Equation 6.4" in references. Unlabelled display math is
  allowed for intermediate steps that are never referenced.
- Multi-line derivations use `\begin{aligned} ... \end{aligned}` inside one `$$` block, with one label for the
  block (tested: one number, in HTML and PDF). Never use `align`, `equation` or `eqnarray` environments: they break
  the cross-references. Use `&` to align at `=`, and `\\` to break lines.
- `\qquad` has exactly one legal use: indenting a continuation line inside `aligned`, where one formula was
  broken at a relation or a `+` because it was wider than the text block. It is never used to put two formulas
  on one line, and never used to separate a formula from a condition on it. Two formulas, two definitions, or a
  formula and its side condition are separate `aligned` rows, aligned at `=` with `&`; a side condition that is
  not an equation goes in the prose sentence that introduces the block. Several definitions or updates are one
  row each. A row that still overflows is split at a relation or a `+`, with `\qquad` on the continuation line.
- An in-place update is written as an assignment with `\leftarrow`, never with a composed `-=` or `+=` glyph and
  never with `\mathrel`: `\mathsf A_i^{(E,\rho)} \leftarrow \mathsf A_i^{(E,\rho)} - g_1\sigma_i^2
  \tilde{\mathsf S}_{ii}/\Delta t_c`. Each update is its own `aligned` row. Where several entries are updated by
  the same expression, write one row per entry; do not collapse them with `\quad`. The code's `+=` and `-=`
  appear only in the Code layer, in backticks.
- Number only an equation that is referenced somewhere in the book or implemented by the code. Textbook
  background — the conservation laws in their usual form, standard vector identities, the definition of a
  standard operator — is unlabelled, even when the surrounding prose points at it; point at it in words ("the
  continuity equation above"). If a later chapter needs to reference a background equation, the editor adds the
  label then; do not pre-label. Before review, check that every label you wrote is used: an unreferenced label
  is a defect, not a spare. In the worked example, (6.4.1) and (6.4.2) are background and lose their numbers.
- A derivation step that needs a sentence of justification gets the sentence in prose between two display
  blocks, never inside the math. `\text{}` is allowed for exactly two things: a short qualifier appended to a
  result after `\quad` ("per stage", "in every cell"), at most six words; and a named placeholder for a group of
  terms that is defined in the prose and not written out ("Jacobian terms"). Everything else — a reason, a
  condition, a reference — is prose. Upright words that are part of a symbol are `\mathrm{}`, not `\text{}`
  (section 5).
- No macros (section 5): the HTML is rendered with `html-math-method: mathml`, as in the house books, so pages work
  offline, and MathML has no macro layer. Write every symbol out exactly as NOTATION.md gives it.
- Matrices and stencils: `\begin{pmatrix}` or `\begin{bmatrix}`. Stencils with more than 5 points go in a figure.
- Do not use `\boxed`, `\colorbox`, `\fbox` or any other framing construct in math: they are amsmath/LaTeX
  macros and the MathML path for the HTML has no macro layer. The defining equation of a scheme is marked by
  being the one the "At a glance" box references, not by a frame. The same applies to `\tag`, `\label`,
  `\nonumber`, `\notag` and `\ref`: numbering and referencing are Quarto's (section 10.3), never TeX's.
  **Status: pending CI.** Whether `\boxed` survives the MathML path has not been tested — there is no Quarto
  on the authors' machines and tech-report has no CI yet. The rule above is the safe default until the first
  CI render settles it; if CI shows `\boxed` renders in both HTML and PDF, the editor may relax this line.
- Every equation that the code implements is followed, in the sentence after the display block, by its code
  citation: "@eq-ch06-dwork-pe is computed in `HydroImpl::forward`
  ({{< code src/hydro/hydro_forward.cpp HydroImpl::forward >}})". For a package with no code pin, the link of
  section 10.6 stands in its place. The citation never precedes the equation and never sits in a heading or a
  bold lead-in; a lead-in names the quantity only.

### 10.4 Figures
- Every figure is a Python code cell, in the scheme's `_<scheme>.qmd`, that calls a function of the `snapy_report`
  package (section 6.1):
  ````markdown
  ```{python}
  #| label: fig-ch06-dwork-stencil
  #| fig-cap: "The corrected-PE face work on a three-cell stencil next to the lower wall. ..."
  #| fig-alt: "Cartoon of cells i-1, i, i+1 with faces, ghost cells hatched, the wall as a thick line."
  #| echo: false
  from snapy_report.ch06.figures.fig_dwork_stencil import make_fig
  make_fig()
  ```
  ````
  `echo: false` for every figure cell. The book sets `freeze: auto`, and `_freeze/` is committed, so a render does not
  re-run a chapter whose cells did not change.
- Captions: one sentence saying what is shown, then one or two sentences saying what to read from it, written so the
  figure can be understood without the text. Define every symbol, colour and line style in the caption or the legend.
  The source data and sha go in the function's docstring, not in the caption.
- `fig-alt` is required (accessibility, and it is what a reviewer reads first).
- Size: set by `figstyle` (3.05 in single, 6.2 in double width; section 6.2). Do not use `fig-width` or `out-width` in
  cells. Formats: SVG for HTML and PDF for the PDF (set in `_quarto.yml`), so lines and text stay sharp. Never use a
  raster image for a line drawing.
- Panels: one function makes the whole multi-panel figure with panel letters (a), (b) in the top left of each panel;
  the caption refers to them. No Quarto sub-figure layouts.
- A static image (none expected) goes in `book/figs/` with its provenance in the caption, and is included as
  `![caption](figs/x.pdf){#fig-... fig-alt="..."}`.

### 10.5 Tables
- Pipe tables with a caption and label on the line after the table: `: Code map of the corrected-PE face work.
  {#tbl-ch06-dwork-code}`.
- At most 5 columns and short cells: the PDF column is about 6.3 in wide. Put long text in prose after the table.
  Code in tables is the shortcode of section 3.1; a link, where its package has no code pin, uses the short
  link text of section 10.6.
- Numbers in tables use the same significant digits within a column; the evidence tag (section 4) goes in a footnote
  or the caption.
- The Code and Tests layers each use one table, as in section 2; the column headers there are fixed.

### 10.6 Code citations and links
- snapy code is cited by symbol, `{{< code <path> <symbol> >}}`, resolved at the pin of `code_pin.yml`
  (section 3.1; README, "Citing code"). That is the only form for snapy.
- **New and rewritten text may cite snapy only at the pin, and only through the shortcode.** A snapy citation
  at any other sha is a review failure, whatever form it takes. There is no "full sha is fine" rule for snapy:
  the pin is the only snapy sha the book may resolve a citation at.
- **The declared debt.** The book still carries snapy line-number links at two older shas, from before the
  shortcode conversion. Measured at this head over `book/**/*.qmd`:

  ```bash
  grep -rhoE 'https://github\.com/chengcli/snapy/blob/(e894700ff7aee30b52882e5202b16461413780b0|9ad0936779ae001dba65c9eafccaa7c058686c97)/[^)]*' book --include='*.qmd' | wc -l   # 563
  grep -rlE  'https://github\.com/chengcli/snapy/blob/(e894700ff7aee30b52882e5202b16461413780b0|9ad0936779ae001dba65c9eafccaa7c058686c97)/'       book --include='*.qmd' | wc -l   # 99
  ```

  **The count is a measurement, not a constant written here.** It falls as chapters are rewritten, so re-run
  the commands rather than quoting this paragraph. The reading as this section was written was **560 links
  across 97 `.qmd` files on tech-report `81e6984bb70c2a59aa68c5dd69c3dc7aaaa4d6e5`**; a branch that predates a
  conversion will read higher and one that follows it lower.

  `reviews/code-anchors-unmapped.tsv` is the **register**: one row per link, and it is the authority on which
  links are declared. A link that is not in the register is not covered by this clause.

  The rule is a **ratchet: the count may fall and must never rise.** A PR that adds a snapy link at a non-pin
  sha is rejected however many already exist; a PR that converts some to the shortcode lowers the count and
  removes their rows from the register. Converting the remaining files is a **separate migration** and is
  deliberately not done inside a style change.
- The link below is the form for a repository with **no code pin** (kintera, pyharp, pydisort, commux —
  section 3.1, "Open slots"), and for the registered passages above — and nowhere else. Its text is
  `basename:lines` and its target is the file at a full sha on GitHub:

  ```markdown
  [`kintera thermo_y.cpp:275-284`](https://github.com/chengcli/kintera/blob/<full-sha>/src/thermo/thermo_y.cpp#L275-L284)
  ```

  The full path and the sha are in the URL, not in the link text. Where two files share a basename, add the last
  directory (`hydro/hydro.cpp:120-140`). A Code table holds this link form only for a package with no code
  pin; for snapy it holds the shortcode. Its "symbol" column holds the anchor of section 3.1.

  **How the pin moves, now that snapy is cited by symbol.** There is no URL-rewriting script for snapy: a
  shortcode carries no sha, so moving the pin is editing one line of `code_pin.yml`, and
  `python3 tools/code_anchors.py bump <full-sha>` does it — it refuses the move while any citation fails to
  resolve at the new sha and lists the cited symbols whose body changed, so their claims can be re-read
  (section 11.2). The links in this bullet are the exception that still carries a sha in the text: a package
  with no code pin moves when a human changes it, and `tools/check_citations.py` checks those links (file,
  lines and anchor at the sha). It has no knowledge of `code_pin.yml` and resolves every sha by clone or
  fetch, so its count depends on whether it can reach those repositories.
- **Repetition is the point, in either form**: a reader opens the book at one page, so the second and tenth
  citation to the same file are written out in full like the first. For snapy that means the shortcode again,
  never a bare "as above"; for a package with no code pin it means a full link with the text `basename:lines`,
  never an abbreviation to a bare `:lines`. Where a walk-through cites the same file many times, the numbered
  list of section 10.7 carries one citation per step.
- Code excerpts: only when the text discusses the lines, at most 15 lines, in a fenced block with its language
  (`cpp`, `python`, `yaml`) and **the citation in the sentence before it** — the shortcode for snapy, a link
  for a package with no code pin. Never put line numbers inside the block: for snapy the citation carries no
  line numbers at all, because the shortcode resolves a symbol rather than a range, and the reader follows the
  citation to the lines; for a package with no code pin the link's own range carries them.
- Pull requests and issues, **in a PROVENANCE block or a source footnote only** and never in prose
  (section 3.2): `[chengcli/snapy#296](https://github.com/chengcli/snapy/issues/296)`.

### 10.7 Callouts and boxes
- The Summary layer opens with plain prose. One `::: {.callout-note title="At a glance"}` box follows the first
  prose paragraph: what the scheme does, the switch, the default and the couplings. Any local-symbols note
  also follows the first prose paragraph. This order applies to every section (section 1.1).
- Order is enforced mechanically by the `tools/fix_glance_order.py` gate, being added separately. The rule
  applies before the script lands; box order does not block chapter-PR reviews.
- Keep the box to at most 6 short lines. Longer boxes go on a report-only list for the owner to trim in the
  readability pass; box length is not a gate failure and does not block chapter-PR reviews.
- The Limits layer may use one `::: {.callout-warning title="Limits"}` box for the limits a user can hit by
  configuration.
- Four callout kinds, and no others. Each is a Quarto callout with a short title; the PDF theme (section 10.10)
  gives each kind its own colour, a light tint and a left bar, so a reader can scan for them:
  - `::: {.callout-note title="..."}`: definition or derivation, and the Summary's "At a glance" box. Titles:
    "At a glance", "Definition", "Derivation".
  - `::: {.callout-tip title="Code"}`: code anchor, the citations a passage rests on — the shortcode for snapy,
    a `basename:lines` link for a package with no code pin — with the symbol
    and the switch.
  - `::: {.callout-important title="Check"}` (or `"Test"`): a check or test, with its name and tolerance.
  - `::: {.callout-warning title="Limits"}` (or `"Open issue"`): limits and open issues.
  Never use `collapse`: the PDF prints everything, so text that only works folded is wrong in the PDF.
- An exact identity that the tests check (e.g. "E+P is conserved per step") is stated as a proposition,
  `::: {#prp-ch06-dwork-ep}` ... `:::`, so the Tests layer can cite it; a supporting result as `{#lem-...}`.
- Footnotes (`[^n]`) carry evidence tags and asides that would break the flow.
- Step order (an algorithm, the Code walk-through) is a numbered list, one step per item, each with its code link.
  No LaTeX algorithm packages.

### 10.8 Bibliography
- Papers: BibTeX in `book/references.bib`, cited as `[@key]` or `@key`. Keys follow `<firstauthor><year><word>`. Only
  the editor merges new entries. Every entry has a DOI or a URL.
- snapy's own history (PRs, issues, commits) is linked as in section 10.6 and, per section 3.2, appears only
  in a PROVENANCE block or a source footnote — never in prose, and never in the bibliography.

### 10.9 Rendering and the render gate
- `quarto render doc/tech-report/book --to html` and `--to pdf`. Python for the cells: an environment with numpy,
  matplotlib and the `snapy_report` package installed (`pip install -e doc/tech-report/src`).
- The render gate is the house books' `render_gate.py` (BOOKCRAFT section 6, G1), run on a PDF render made with
  `-M latex-clean:false` so the LaTeX log is kept. It catches unresolved cross-references (`?@...`), duplicate labels,
  references into fenced divs, dropped glyphs (missing characters), unrendered math in table cells, a table caption
  orphaned after a code fence, old font commands and unbalanced quotes. Overfull boxes over 10 pt in the chapter's
  text also fail. Python errors fail.
- `freeze: auto` re-runs a chapter only when its `.qmd` changes. After changing any `snapy_report` function, render
  that chapter file alone (`quarto render book/chapters/NN-slug.qmd`), which ignores the freeze. The release render
  is made with `_freeze/` removed.
- The author looks at every page of their chapter in the PDF before asking for review: equation lines inside the
  margin, tables inside the text width, figures legible at print size.
- `_book/` is build output and is not committed; `_freeze/` is committed.

### 10.10 Page layout and theme
- The look of the PDF is set once and chapters never set colours, fonts, page breaks or float placement:
  - `book/_quarto.yml`: KOMA `scrreprt`, A4, one-sided, chapters open on any page, `DIV=12`, 10 pt, compact
    headings, no dot after numbers; printed TOC at depth 2; figures placed `htbp` and drawn at the text width,
    6.2 in (`fig-width`), never shrunk with `out-width`.
  - `book/_tex/theme.tex`: one accent colour plus greys; title page and chapter openers; the four callout kinds
    of section 10.7; long tables in `\footnotesize` with a shaded header; light code blocks; links in the accent
    colour; numbered PDF bookmarks to depth 3, opened at the chapter level; float fractions.
  - `book/_filters/codemap-widths.lua`: fixed column widths for the Code-layer tables of section 2.
  - `snapy_report.figstyle` (and the package defaults in `snapy_report/__init__.py`): 9 pt serif text, STIX
    mathtext, fonts embedded as TrueType (`pdf.fonttype` 42), widths 3.05 and 6.2 in.
- The render gate reports, by file and line, every `?@` left in the rendered output and every hand-typed
  "Chapter N" in a `.qmd`; a chapter is named by `@sec-chNN`, or in words if it is not written yet. Both checks
  run in report mode until the editor switches them to fail (`REPORT_ONLY` in `tools/render_gate.py`).

## 11. House rules for the rewrite

Six rules for the chapter rewrite. Xi proposed them, the report thread adopted them, and the two conditions
Jiheng attached are written into rule 1. A clause nobody has ruled on is marked **PROVISIONAL** here, in the
same sense as the changed-symbol clause of section 3.1: follow it, and expect it to be replaced.

### 11.1 Evidence

**Every published number comes from a check output produced at the `code_pin.yml` pin, cited inline, and every
claim names the test that holds it up.** A number with no check behind it is marked **missing evidence**
(section 4) and is either produced or removed — it is not left to stand on the author's word.

Two conditions on how that is enforced:

- **(a) The freeze-anchor gate ratchets from a recorded baseline; it never fails from zero.** The citation
  check and `src/tests` are already red at the tip, so a gate that demanded zero would fail every PR on the
  day it landed and teach everyone to ignore it. The baseline is recorded once, and a PR fails only when it
  makes a count *worse*. See `AUTHOR-CHECKLIST.md`: compare against the base, never against zero.
- **(b) The record carries the versions that move the digits, alongside the command** (section 4). Command
  without versions is not a run record.

**Round-off caveat — this is our own evidence, not a hypothetical.** On this project's own machines
`cellface_check` (ch06), `fixvapor_check` and `kinetics_check` (ch10) print a round-off magnitude that drifts
**with no code change at all**: `1.3e-17` against a committed `5.0e-17`, `1.79e-15` against `2.20e-15`,
`1.6e-15` against `2.1e-15`, every claim still inside its threshold. Therefore:

> **Where a reported quantity is pure round-off noise, the number the text quotes is held against its stated
> tolerance, not against the digits of one host's run.**

**What the shipped harness does, and why this rule yields to it.** The harness at the pin *does* byte-compare:
`src/tests/test_ch06.py:51` is `assert out == script.with_suffix(".out").read_text()`, and
`src/tests/test_ch02_checks.py:13` and `src/tests/test_ch10_checks.py:13` are
`assert result.stdout == script.with_suffix('.out').read_text()`. (`src/tests/test_checks.py:12` is different:
it only asserts the exit status.) That is the code, so it decides:

- **Byte comparison is legal where the harness itself does it.** A committed `.out` is a regression fixture,
  and the three test modules above are the sanctioned place for it. Nobody is asked to weaken them, and a
  failing one is a real signal that something moved.
- **A byte comparison is not evidence for a quoted number.** When such a test fails only in a round-off digit,
  the finding is about the fixture and the host, not the claim: check the number against its tolerance
  (section 4), and if it holds, the fixture is stale, not the physics. The three checks above are in exactly
  that state.
- **Do not hand-write a byte comparison of rendered output.** Prose, tables, figures and PDF output are never
  compared byte for byte by a rule in this guide; that was the original intent of this clause and it stands.
  A figure is checked by re-rendering from its script (section 6), not by diffing bytes.

Report a round-off quantity against its threshold ("below $10^{-13}$ relative") rather than to digits no two
machines agree on, whatever the fixture stores.

**The pending Tests layers are an explicit to-do under this rule, not a silent gap.** Measured at this head,
**37** scheme files have a Tests layer containing "pending" and **30** contain "missing evidence". Each is a
claim without a named test, and each belongs to the chapter's owner to produce or to retract. They are counted
here so that they cannot be mistaken for finished sections.

### 11.2 Re-pin: what has to be re-read

When the pin moves, the re-read list is the **union** of two sets:

1. **claims whose check output changed at the new pin** — the primary set, because a changed number is a
   changed claim; and
2. **the changed-symbol diff** (`python3 tools/code_anchors.py bump <sha>` lists cited symbols whose body
   changed) **for claims that have no test yet**.

The second is **the floor, not a replacement for the first**. A chapter with no checks still has to be re-read
through the symbol diff; a chapter with checks is re-read through both. Taking only the symbol diff on a
chapter that has checks is the failure this rule exists to prevent.

### 11.3 Review record

Two requirements beyond the template in `REVIEW-RECORD.md`:

- **(a) A fresh-context reviewer.** The chapter is read against the code at the pin by someone who did not
  write it and is not carrying the author's working context. A reviewer who has been in the drafting
  conversation cannot test whether the text stands on its own.
- **(b) A blind restatement of sampled claims.** The reviewer restates **[N: to be set]** sampled claims from
  the chapter in their own words, without the author's framing, and the author confirms or corrects each.
  **The sample size N has not been set. It is left blank deliberately — Xi or the editor sets it; nobody
  should pick a number locally.**

There is no separate "recheck after fixes" rule, because one already follows from sign-off: **any push resets
every sign-off on the PR**, so a fix pushed after a review invalidates that review and the record is redone
against the new head (`REVIEW-RECORD.md`: a review is of one head sha).

### 11.4 Ownership lock

**One owner per chapter; one lock; the integrator never edits chapter text.** The integrator merges, resolves
nothing in prose, and hands a conflict back to the owner.

- **Stated exception.** Generated Appendix B and C output is regenerated by the **generator owner**, not by
  chapter owners. A chapter owner never hand-edits a generated appendix; the generator is run and its output
  committed (section 7).

**The gap, and the path — PROVISIONAL.** A per-chapter lock with no route for a contradiction *between*
chapters freezes the contradiction instead of resolving it: each owner is locked out of the other's text, so
neither can fix it and the book ships disagreeing with itself. The path, pending Xi's ruling:

> **A contradiction between chapters goes to the editor. It is not resolved locally under either lock.**
> Neither owner edits the other's chapter, and neither quietly changes their own text to match — the editor
> rules, and both chapters are then changed to the ruling.

The live example, read at this head:

| where | what it says |
|---|---|
| `book/chapters/12-configuration/_environment.qmd:12` | "Default: **four off; radial exact on**" |
| `book/chapters/appd-switches/_switches.qmd:14` | "**Defaults.** Four numerics switches off" |
| `book/chapters/04-spatial-discretization/_fluxcov.qmd:14-15` | `SNAP_FLUX_COVARIANCE` is "**on by default**" |

`appd-switches/_switches.qmd` also disagrees with itself: line 14 counts the covariance among the switches
that are off, while line 34 says "Two default to on: `SNAP_FLUX_COVARIANCE`" and the table at line 51 gives it
as `on ("1")`. Three files, two chapters and one appendix, and no single owner who may fix all of them — which
is the gap, stated concretely.

**Recorded precedent.** The ch06/ch08 re-read pass reached ch08 text that had been merged from another owner's
work; the question of keeping or reverting it was put to that owner, who ruled keep, no revert. Under this
lock that is an **ask-first**, and it is recorded here as the shape the rule expects: the owner of the text is
asked, not overruled.

### 11.5 Census

**The book map is the census of the text.** Chapters are written against it, not against what an author
remembers or re-derives locally. Map questions — a notation conflict, a symbol with two meanings, which
chapter is a mechanism's home, a definition sitting in the wrong place — **go to the editor** and are not
settled inside a chapter PR (`AUTHOR-CHECKLIST.md` 1c).

### 11.6 Burn-down

**The rewrite owners are the burn-down owners.** The counts a chapter carries — history lines, off-pin shas,
pending Tests layers, missing evidence — are burnt down by whoever owns that chapter's rewrite, not by a
separate clean-up pass.

**PROVISIONAL, pending the ratchet gate landing.** The counts are to come from the ratchet gate rather than
from anyone's hand count. As that gate stands in review it reports **per chapter**, not per owner, and it is
not merged, so there is no per-owner table yet. Under 11.4 one owner holds one chapter, so the per-chapter
counts map onto owners through the lock table; writing that mapping out as a printed per-owner table is the
remaining step, and this clause is provisional until the gate lands and prints one.
