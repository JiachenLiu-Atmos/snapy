# snapy Technical Report (work in progress)

This branch (`tech-report` on UCzhangxi/snapy) holds a technical description of snapy, in the style of the WRF, MPAS and CESM technical notes. It is never merged into snapy and never opened as a pull request.

- `sources/`: inputs to draw from. These are existing notes, derivations, PR bodies and issue threads, cleaned for this use. They are inputs, not the report.
- `BRIEF_C0.md`: the brief for the architect/editor. `ISSUES.md`: known gaps in the sources. `STATUS.md`: progress log.
- `STYLE.md`: the binding style guide, covering the six-layer template, citations, evidence, figures and checks.
- `AUTHOR-CHECKLIST.md`: the gates to run before opening a chapter PR, with the command for each and what it must say, and what belongs in the PR body. `REVIEW-RECORD.md`: the template a reviewer fills (head/tree/base shas with read times, the gate table against the base, the render run, findings, verdict).
- **Setting up.** `bash setup.sh`, then `source .report-env/activate`: one command that installs everything this report needs at the pinned versions, into one prefix, with no package manager and no `sudo`. `SETUP.md` says what it installs and why each version; `tools/env-pins.json` pins Quarto, TinyTeX, Universal Ctags, NetCDF and HDF5 with the sha256 of every download; `tools/requirements.txt` pins the Python environment. `bash setup.sh --with-native` also builds pydisort, pyharp, kintera and snapy at their pins, which the chapters' `ctest` and do-card commands need. `python3 tools/check_env.py` checks that the committed pins are consistent and installable.
- `code_pin.yml`: the one commit every `{{< code >}}` citation resolves at. It is the single source of truth for the snapy pin; no other file restates that sha.
- `NOTATION.md`: every symbol, with one meaning each.
- `OUTLINE.md`: the plan, chapter by chapter and scheme by scheme.
- **Where things live.** One home per kind of resource; `FIGURES.md` maps every figure to its code and data.

  | what | home | written by | read by |
  |---|---|---|---|
  | chapter text | `book/chapters/NN-slug.qmd` and its sections `book/chapters/NN-slug/_<scheme>.qmd` | authors | Quarto |
  | figure code | `src/snapy_report/chNN/figures/fig_<name>.py`: one module per figure, `make_fig()` returns it | authors | the figure's `{python}` cell in its section file |
  | chapter helpers | `src/snapy_report/chNN/*.py` (`flow.py`, `formulas.py`, `_draw.py`, ...); the one figure style `src/snapy_report/figstyle.py` | authors | figure modules and checks |
  | executable checks | `src/snapy_report/chNN/checks/<name>_check.py` with its committed output `<name>_check.out` | authors | `src/tests` (byte comparison), Appendix B (`tools/gen_derivation_index.py`) |
  | break arms of a check | the same `<name>_check.py`: `--list-breaks` names them, `--break <arm>` runs one and is expected to exit nonzero, and the green run executes every arm and fails if one of them passes (chapter 15: `invariants_zero_check.py`, `invariants_budget_check.py`) | authors | the check's own last claim; the committed `evidence/logs/break_arms.log` |
  | figure data | `src/snapy_report/chNN/data/` | the check or run script `FIGURES.md` names next to the file, or a snapy run named in the file's `provenance` | figure modules |
  | run records | `src/snapy_report/chNN/evidence/` (chapter 10: `run_evidence.py` and `logs/`; chapter 15: `aam_budget_run.py`, `port_validation.py` and `logs/`, which need a build at the pin in the active Python and, for the GPU arm, a CUDA-enabled one) | the run script | the chapter's Tests layer |
  | generated appendices | `book/chapters/17-appendices/_app*.qmd` | the pre-render generators in `book/_quarto.yml`, from their inputs in `tools/` (`*.yaml`, `*.json`) | the appendix chapter |
  | execution cache | `book/_freeze/` (committed for chapter 5 only) | Quarto (`freeze: auto`) | Quarto |
  | gates and their baselines | `tools/*.py`, `tools/*-baseline.json`; their tests `tools/tests/`, fixtures `tools/tests/fixtures/` | authors; a baseline by its gate's `--update` | CI, `AUTHOR-CHECKLIST.md` |
  | review records | `reviews/`: `review-record.tsv`, anchor maps, pin-move notes, and the one-off scripts of past reviews (kept as records, run by nothing) | reviewers | the ratchet gate (`review-record.tsv`), reviewers |
  | inputs | `sources/` | | authors |
  | drawn figures | `build/figures/<chapter>/<label>.svg\|pdf\|png` (gitignored) | `tools/render_figures.py` | you |

  **Figures.** `python3 tools/render_figures.py` draws every figure cell as the render does, to `build/figures/`
  (`--chapters ch05,ch06` for some chapters; `build/figures/index.json` holds each SVG's sha256, so two draws
  compare byte for byte). To change a figure's data, run the writer `FIGURES.md` names (a check rewrites its data
  file when run; `src/snapy_report/ch06/checks/d_face_work_pe_check.py` only with `--write-data`), then draw it.
  After adding a figure, or changing the modules or data files it uses, rewrite the manifest with
  `python3 tools/render_figures.py --manifest` in the same commit; `render_gate.py` checks it statically (gate
  `figure manifest`) and `python3 -m pytest tools/tests` tests the tool.
- `tools/check_citations.py`: checks that every code citation (STYLE 10.6 links and `path:line@sha`) resolves at its sha, that the cited lines exist and, in a `.qmd`, contain the link's anchor symbol, and that the repository is one STYLE 3.1 allows; `python3 -m pytest tools/tests` tests it.
- **Citing code.** Cite snapy code by file and symbol, never by line number:
  `{{< code src/hydro/hydro_forward.cpp HydroImpl::_flux_covariance >}}` renders as the link
  `hydro_forward.cpp` `HydroImpl::_flux_covariance` to the symbol's lines at the one pinned commit,
  `code_pin.yml`. The symbol is a function, method, class, struct, member or variable; C++ names keep their
  classes and drop their namespaces, gtest tests are `Suite.Name`, Python symbols are `Class.method`, and a file
  alone is `{{< code <path> >}}`. If the gate reports a symbol as ambiguous (overloads), add the `overload=N` it
  lists. `python3 tools/code_anchors.py check` resolves every citation at the pin (Universal Ctags 6.2.1:
  `brew install universal-ctags` or set `CTAGS=<binary>`; a clone holding the pin in `SNAPY_REPO`, else it is
  fetched into `~/.cache/snapy-report`); it is the render's pre-render and a CI step, and either fails on a
  citation that does not resolve. To move the pin: `python3 tools/code_anchors.py bump <full-sha>` lists the
  cited symbols whose body changed (re-read those, and fix the claims they carry), and writes the new sha into
  `code_pin.yml` only when every citation resolves there; `diff <old> <new>` prints the same list for any two
  commits. Links that could not be converted to symbols (lines across several symbols, top-level code,
  kintera/pyharp, generated appendices) remain STYLE 10.6 links, listed in `reviews/code-anchors-unmapped.tsv`.
- `tools/check_figure_data.py`: the figure-data check. Draws every figure the way `render_figures.py` does and compares the numbers it plots with `tools/figure-data-baseline.json`: a fingerprint of every plotted series, and which committed reference series the figure was found to be drawing (and how). It answers the question an SVG hash cannot — two SVGs can differ byte for byte and plot the same curve, and a figure whose data drifted can keep the same byte size. `--update` rewrites the baseline, deliberately as a separate run; `--unverifiable` lists the 129 of 145 figures that read no committed data, which this cannot speak for; `--unmatched` lists reference series the figure does not draw. `python3 -m pytest tools/tests` tests it; SETUP.md explains the tolerance.
- `tools/check_figstyle.py`: the figure-style gate (STYLE 6). Figure code under `src/snapy_report/` may style only through `snapy_report.figstyle`: it counts, per chapter and without importing anything, imports of other style modules, rcParams writes and inline font settings (`fontsize=8`, `family=...`), and compares the counts with `tools/figstyle-baseline.json`. Ratchet: a chapter above its baseline fails, and so does one below it until `python3 tools/check_figstyle.py --update` lowers the baseline in the same commit (it never raises one); `--list` prints every site. `render_gate.py` runs it as its `figstyle` source gate; `python3 -m pytest tools/tests` tests it.
- **Ratchet gate.** `tools/ratchet_gate.py` counts four things per chapter (and for the preface,
  `book/index.qmd`) and compares them with the committed baseline `tools/ratchet-baseline.json`; the
  `tech-report-ratchet` workflow runs it with its tests on every pull request and push. **history**: change history
  in body text (PR or issue numbers, "main after", "main at", "squash-merged", "the previous pin", "pull request",
  "is now", "was introduced in" and the other phrases of `HISTORY_PATTERNS`; words that also describe physics, such
  as "no longer" or "used to", count only in their history forms). Footnotes, HTML comments (PROVENANCE), shortcodes
  and fenced code are not body text, so a source or credit footnote may carry a PR number. **offpin**: a sha of
  snapy, kintera, pyharp, pydisort or commux that is not that package's one pin (`code_pin.yml` for snapy,
  `tools/package-pins.json` for the others), anywhere in the file: body text, footnotes, link targets, code comments
  and fenced code, and the plain fields of the PROVENANCE header (`<package>@<sha>`, `<path>@<sha>`, a GitHub
  blob/tree/commit URL, or any other full sha that is no pin); a shortcode carries no sha. **status**: a file whose
  PROVENANCE says `reviewed` or `approved` without a valid entry in `reviews/review-record.tsv`. **xref**: a
  reference that resolves nowhere (render_gate's own resolution) and every `sec-` label no other chapter cites.
  **The ratchet.** A rise fails; a fall must lower the baseline in the same pull request
  (`python3 tools/ratchet_gate.py update`). The baseline records the pin it was measured at and may rise only with a
  move of that pin: `update` refuses a higher count unless `code_pin.yml` has moved since the baseline was written,
  and `check --base-ref <target>` (what CI runs) fails when the committed baseline is higher than the target
  branch's without a pin change between them. `python3 tools/ratchet_gate.py counts -v` lists every instance.
  **Owners.** `tools/chapter-owners.tsv` names one rewrite owner per chapter: it is the lock (one owner per chapter;
  `check` fails on none or two) and the source of the per-owner burn-down table that `counts` and `check` print. Its
  third column names the chapter's checker (`-` for none), who may not be the owner.
  **Review record.** One tab-separated line per event, newest last: `file kind reviewer author pin sha256 date
  note`. A `review` says the reviewer read the file at this content and this pin and stands behind its status. A
  `restamp` is the same reviewer confirming that review after the file or the pin changed without changing the
  verdict, with a one-line note on what changed; anyone else, or a change that alters the verdict, needs a new
  `review`. `pin` is the full sha of `code_pin.yml`; `sha256` is the file without its PROVENANCE comment, so setting
  the status does not change it. Write entries with
  `python3 tools/ratchet_gate.py record review <file> --reviewer R --author A` and
  `python3 tools/ratchet_gate.py record restamp <file> --reviewer R --note "what changed"`, which fill in the hash,
  the pin and the date and refuse a re-stamp by anyone but the reviewer. A reviewed file passes when its newest entry
  has the current hash and pin and a reviewer who is not its author.
- `tools/render_gate.py`: renders `book/` to HTML and PDF and gates the render (STYLE 10.9). Outcomes: pass (exit 0), pending (exit 2: a `snapy_report` function a chapter calls is not written yet, or there is no book yet) and fail (exit 1). Nothing is released while anything is pending; `--release` makes pending a failure.
  References into a chapter that is not written yet (a valid chapter id of STYLE 10.1 whose `{#sec-<id>}` is defined nowhere) are listed in their own gate, by target id, as expected failures: they still fail, apart from the real dangling references. So are references into a scheme of a written chapter whose `_<scheme>.qmd` does not exist yet.
- `tools/check_do_cards.py` and `tools/do-cards.yaml`: the runnable cards (N8). A card is a Code layer's "**A minimal input**" paragraph; `do-cards.yaml` holds one record per card (section anchor, deck, class, trim, measured or estimated time, expected outcome) and the dated `validation_log` for the GPU and multi-rank cards. The check fails when the sources and the data file disagree on the card set, when a runnable card names no deck or its deck does not resolve, when a `needs-build-option` card does not name a CMake `option()` of the repository, and on a stale named exception; `render_gate.py` runs it as its `do cards` source gate. `tools/tests/test_do_cards_run.py` runs the CPU cards trimmed against a code-pin build, opt-in: `SNAP_PIN_BUILD=<build dir> python3 -m pytest -rs tools/tests/test_do_cards_run.py` (`-rs` prints each skip's reason). A `needs-build-option` card runs there only when the build's `CMakeCache.txt` sets its option, and is skipped with the reason otherwise.
- `tools/gen_test_index.py`: generates Appendix C, the test index (`book/chapters/17-appendices/_appc-test-index.qmd`), from `tools/ctest-index.json`, the tests `ctest` lists for snapy at the pin (Eigen's own tests, which the build fetches, are counted, not listed). Both files are committed; CI runs `python3 tools/gen_test_index.py --check`, which fails if the fragment is not what the fixture gives. When the pin moves, refresh both from a clean checkout at the new pin (configuring is enough, no build):
  ```bash
  git clone https://github.com/chengcli/snapy snapy-pin && git -C snapy-pin checkout <pin>
  cmake -S snapy-pin -B snapy-pin/build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON
  python3 tools/gen_test_index.py --refresh snapy-pin/build
  ```
- `tools/gen_deck_keys.py`: generates Appendix I, the deck key reference (`book/chapters/17-appendices/_appi-deck-keys.qmd`),
  from `tools/deck-keys.json`, the keys every `check_keys` list of snapy's parsers accepts at the pin, with the
  default each parser falls back to and the line that decides it; each key links to the section that names it, or
  says "no section". The fragment is rewritten before every render; `python3 tools/gen_deck_keys.py --check` fails
  when it is stale. When the pin moves, refresh the JSON with `python3 tools/gen_deck_keys.py --refresh`.
- **Generated files and rebasing.** Appendix A (`tools/gen_notation_index.py`) and Appendix B
  (`tools/gen_derivation_index.py`) are committed only in pieces that one chapter owns: Appendix B is one file per
  chapter (`book/chapters/17-appendices/_appb-chNN.qmd`) under a parent that holds no counts, and Appendix A
  depends on `NOTATION.md` alone. Every value that several chapters decide (the totals, the per-chapter counts of
  the summary, the symbol counts, the "first used" column) is a `{{< generated ... >}}` shortcode, filled in at
  render from `book/_generated/*.json`, which the generators write in the pre-render step and which is not
  committed. The one-line statements that Appendix B gives for equations are one file per chapter too:
  `tools/derivation-statements/chNN.json` holds the labels `eq-chNN-...` and no other (the generator refuses a
  label in another chapter's file or given twice), so a new statement goes in its own chapter's file. So two pull
  requests that edit different chapters touch different files or different lines, and git merges them
  (`tools/tests/test_chapter_prs_merge.py`). After a rebase or a merge of the target branch, never hand-merge a
  generated file: take either side, run `python3 tools/gen_derivation_index.py`, `python3
  tools/gen_notation_index.py` and, if your chapter's counts changed, `python3 tools/ratchet_gate.py update`, and
  commit what they change; `--check` on each generator and `ratchet_gate.py check` then pass. A change to
  `NOTATION.md` itself still rewrites `_appa-notation.qmd`, so two pull requests that both edit `NOTATION.md` can
  still conflict there.
- `book/_quarto.yml`, `book/index.qmd`: the Quarto book (STYLE 8, 10): KOMA `scrbook` PDF through LuaLaTeX, MathML in the HTML (no math macros), `freeze: auto`, `toc-depth: 2`. It includes `book/_tex/inline-code-breaks.tex`, which lets inline code break after `_`, `/`, `:` (after `::`) and `@` in the PDF, shows only the chapter title in running heads and lets long hex strings (shas) break inside; in the HTML it wraps code blocks (`code-overflow: wrap`) and inline code (`book/report.css`: `code { overflow-wrap: anywhere; }`).
- `tools/book_chapters.py`: writes the chapter list of `book/_quarto.yml` from OUTLINE.md's chapter ids and the chapter files that exist (`NN-slug.qmd`, `appX-slug.qmd`), in OUTLINE order; the render gate fails when the list is out of date. Run `python3 tools/book_chapters.py` after adding a chapter file.
- Planned: `reviews/`, `build/render_figures.sh`, the test-index generator, the PDF/HTML build.
