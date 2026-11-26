# Setting up a machine for this report

```bash
cd doc/tech-report
bash setup.sh                     # the report: checks, figures, every gate, the HTML and the PDF
source .report-env/activate
```

That is the whole thing. Add `--with-native` to also build pydisort, pyharp, kintera and snapy at
their pins, which is what the `ctest` and do-card commands in the chapters need.

Nothing is installed system-wide: no package manager, no `sudo`, no write outside the prefix and the
repository. Every download is checked against the sha256 committed in `tools/env-pins.json` before it
is unpacked; a mismatch stops the run rather than installing the file.

## What it installs, and where each version comes from

| | version | pinned in | why that version |
|---|---|---|---|
| Python | 3.12–3.13 | `tools/env-pins.json` | `src/pyproject.toml` requires >=3.12, so `snapy_report` does not install below it; torch 2.10.0 publishes no wheel below 3.11 or for 3.14; CI uses 3.12 |
| numpy, sympy, matplotlib | 2.4.6, 1.14.0, 3.11.2 | `tools/requirements.txt` | the versions the report's committed numbers were produced with |
| torch | 2.10.0 | `tools/requirements.txt` | what the cards and data provenance record throughout |
| Quarto | 1.10.19 | `tools/env-pins.json` | `QUARTO_VERSION` of `.github/workflows/tech-report.yml` |
| TinyTeX | v2026.10, scheme 1 | `tools/env-pins.json` | the workflow asks `quarto-actions/setup` for `tinytex: true` |
| Universal Ctags | 6.2.1 | `tools/env-pins.json` | `CTAGS_VERSION` of the workflow and of `tools/code_anchors.py:50` |
| NetCDF | 4.9.3, built | `tools/env-pins.json` | `-DNETCDF=ON` in the snapy, pyharp and kintera builds |
| HDF5 | 1.14.6, built | `tools/env-pins.json` | netcdf-c is a thin layer over it, so both come from the pin or neither does |
| pydisort | 1.8.13 | `tools/env-pins.json` | the report pins no pydisort; `FindDisort.cmake` needs it importable |
| pyharp | `47217158` | `tools/package-pins.json` | the report's pin |
| kintera | `0429d795` (v2.6.13) | `tools/env-pins.json` | see "The kintera revision" below |
| snapy | `73c45a7c` | `code_pin.yml` | the report's pin |

`tools/requirements-linux-x86_64.lock.txt` is the exact Python closure verified on linux-x86_64;
install it instead of `tools/requirements.txt` when you need that environment byte for byte.

## Why `activate` sets so many cache variables

Because a read-only `$HOME` is normal on a shared or containerised host, and almost everything here
defaults its cache under `$HOME`. Left alone, LuaTeX stops before the first page with

```
luaotfload | load : FATAL ERROR
luaotfload | load :   × "luaotfload-init.lua:301: system : no writeable cache path, quiting".
```

and no PDF is produced. `tools/code_anchors.py:51`, `tools/check_provenance.py:126` and
`tools/check_citations.py:407` default to `~/.cache/...` the same way, and matplotlib and Jupyter do
too. `activate` points `TEXMFVAR`, `TEXMFCACHE`, `TEXMFCONFIG`, `TEXMFHOME`, `SNAPY_REPORT_CACHE`,
`XDG_*`, `MPLCONFIGDIR`, `IPYTHONDIR`, `JUPYTER_*`, `PIP_CACHE_DIR` and `TMPDIR` inside the prefix, so
none of them ever looks at `$HOME`. `IPYTHONDIR` matters more than it looks: every figure cell starts
an IPython kernel, and at the default each one warns twice that `~/.ipython` is not writable, which is
112 lines of warning in a full render and fails `render_gate.py`'s `stdout` gate.

It also sets `SNAPY_REPO` to a clone of `chengcli/snapy` at `code_pin.yml`, which `setup.sh` makes.
With it, `tools/code_anchors.py` resolves every `{{< code >}}` without touching the network; without
it, the tool fetches into `SNAPY_REPORT_CACHE` on every run.

## The kintera revision

`tools/package-pins.json` (the commit the chapters cite) and `tools/env-pins.json` (the commit
`setup.sh` installs) name one kintera revision: **v2.6.13, `0429d795f79a0665afeca565e5a05c87ab4ba1e2`**
(the tag `v2.6.13` of chengcli/kintera). It is the kintera snapy at `code_pin.yml` builds against:
its CI installs `kintera==2.6.13` (`.github/workflows/ci.yml:86` and `:162` at the pin), and
`CMakeLists.txt:120` requires `find_package(Kintera 2.6.13 REQUIRED)`.

The chapters' existing kintera links still carry the sha they were written at, `c55b13b2` (v2.6.2);
`tools/ratchet_gate.py` counts them as `offpin` until each chapter's rewrite re-points them.
`python3 tools/kintera_pin_migration.py` compares every such citation at both revisions, byte for byte
(a clone holding both: `KINTERA_REPO`, or `bash setup.sh --with-native`, which leaves one in
`.report-env/src/kintera`):

```
194 citation(s) found
  identical, the sha can simply be swapped :  29
  same code at new line numbers            :  52
  BLOCKED, the cited code is gone          : 113
```

The blocked ones are in chapters 02, 09, 10 and 16 and in appendices A and E; the tool prints each with
its file, line and cited range. Re-pointing them is chapter work: the claim beside each has to be
re-read against v2.6.13.

## The pins are installed, not just written down

The default route **builds** the pinned NetCDF and the pinned HDF5 and then checks that `nc-config`
reports exactly the pinned version. It used to prefer whatever the machine already had and only build
when it found nothing usable, which meant the documented reproduction silently linked against a
different library on every host — 4.8.1 here against a 4.9.3 pin. A pin nobody installs is not a pin.

`bash setup.sh --allow-host-netcdf` is the deliberate opt-out. It still refuses anything below the
`minimum` in `tools/env-pins.json`, and it says in as many words that the result is not the
documented route.

## The TeX pin fails loudly or not at all

Every step that pins TeX is fatal: pointing `tlmgr` at the dated snapshot, `tlmgr`'s self-update from
that same snapshot, and the package install. After them `setup.sh` reads back the repository actually
in effect and compares it with the pinned one, so a silent fall-back to a default mirror cannot pass,
and a package the snapshot cannot serve stops setup and names itself.

That matters because carrying on is precisely what broke the PDF: a repository step that reported a
failure and continued let the render pull koma-script 3.49.3 from a rolling mirror.

`setup.sh` also records `tlmgr list --only-installed` in `.report-env/tex-packages.lock`, so "the
render fetched nothing" is checkable rather than assumed:

```bash
diff .report-env/tex-packages.lock <(tlmgr list --only-installed | sort)
```

Any difference is a package that arrived at render time. That check earned its keep immediately: a
full render pulled in `colortbl`, `fontawesome5`, `needspace` and `tikzfill` on its own. They came
from the pinned snapshot, so that render was still reproducible — but a package the setup step did
not install is a package whose version nobody chose, so all four are now named in
`tools/env-pins.json` and the lock is expected to be unchanged across a render.

## The native route

```bash
bash setup.sh --with-native
source .report-env/activate
```

adds pydisort, pyharp, kintera, `pinc` and snapy, each at its pin, and leaves the build in
`.report-env/snapy-pin/build` with `SNAP_PIN_BUILD` pointing at it. That is what the chapters'
`ctest` lines and the do-cards need.

Three things it has to get right, each of which silently did not work before:

- **CMake before pip.** Neither kintera's nor snapy's `setup.py` runs CMake. They link the Python
  extension against whatever is in `<src>/build/lib`, so with no CMake build there is nothing to
  link — and `pip install` still exits 0. The import is what fails, with
  `undefined symbol: _ZTIN7kintera17EquilibriumTPImplE`. Every package is therefore built, then
  installed, then **imported** to prove the wheel works.
- **`pinc`.** snapy defaults `PNETCDF=ON` on Linux (`CMakeLists.txt:14-18`) and
  `cmake/parameters.cmake:39` then refuses to configure without the Python package `pinc`. The
  report declares it nowhere, including in the configure line `README.md` prints for refreshing the
  test index. It is pinned in `tools/env-pins.json` at 0.2.0.
- **`SNAPY_TEST_PYTHONPATH`.** Without it CMake warns that the `*_python` ctests "will import the
  installed snapy, not this tree", and on a clean machine they all fail with `ModuleNotFoundError`.
  `setup.sh` points it at the environment it just installed snapy into.

`tools/requirements-build.txt` pins the four build-time requirements pyharp and kintera declare.
They are installed with `--no-build-isolation` so that they build against the pinned torch rather
than one pip fetches into a throwaway environment, which only works if those requirements are
genuinely present.

### Two documented commands this does not fix

They are the report's, not the environment's, and they are listed here so nobody mistakes them for
setup failures:

- `ctest --output-on-failure -E '^test_shallow_xy_decomp$'`, printed by eight chapter-15 section
  files, is not scoped to snapy. It sweeps in the Eigen suite the build fetches and reports **821
  failures out of 1054** — every one of them Eigen's own or the reference BLAS's, none of them
  snapy's. `README.md` already notes that Eigen's tests "are counted, not listed", so this wants a
  selector. **Chapter 15 is an excluded chapter** (see below); the command was run because it
  exercises snapy's code rather than the prose, and it yields no verdict on that chapter.

  `-R '^test_'` added to the same build gives **127 of 127 passed, exit 0**. That selector is mine,
  not the report's: it is a manually verified scoping of the documented command, offered as evidence
  that snapy's own tests pass, not as a committed fix.
- `run: ctest -R`, which names no test. Run verbatim it is `CMake Error: Invalid value used with
  -R`, exit 1. Four sites, at `ce725f672df68740d4c6cffc3d99a9bd74656eee`:

  | file:line | chapter |
  |---|---|
  | `book/chapters/04-spatial-discretization.qmd:85` | 04 — **excluded** |
  | `book/chapters/04-spatial-discretization/_fluxcov.qmd:334` | 04 — **excluded** |
  | `book/chapters/09-time-integration.qmd:49` | 07 |
  | `book/chapters/10-moist-coupling.qmd:60` | 10 |

### Authorship exclusions

Seven chapters are excluded from this audit as owner-authored: **03, 04, 06, 08, 12, 15, 16**.
Broad tooling unavoidably crosses them — the whole-book render, the figure manifest, the all-check
tooling run and the chapter-15 `ctest` line above all traverse excluded chapters. **Those executions
carry no verdict on an excluded chapter.** They are recorded here only so the leakage is visible.
`tools/chapter-owners.tsv` is a rewrite-owner lock, not a complete authorship register.

## Checking the numbers behind the figures

```bash
python3 tools/check_figure_data.py                 # every figure
python3 tools/check_figure_data.py --chapters ch05 # some chapters
python3 tools/check_figure_data.py --unmatched --unverifiable   # the full picture
```

Two SVGs can differ byte for byte and show the same curve, and a figure whose data drifted can keep
the same byte size. Hashing the SVG answers neither question, so this one asks about the numbers. It
draws each figure exactly as `tools/render_figures.py` does, with the same audit hook recording
which files the draw opened, and compares against `tools/figure-data-baseline.json`:

- **every series the figure plots**, by stored values. This catches a changed number wherever it
  came from — the data file, the figure module or a library.
- **which committed reference series the figure was found to be drawing, on which artist.** This
  catches a figure that quietly stops reading its committed data, or starts reading it somewhere
  else.

Neither half can be replaced by comparing a figure with its own data file: a figure that *reads*
that file moves with it, so perturbing the file makes the figure draw the perturbed value and the
two still agree. Only a committed reference stays put.

### Links and candidates

Only an **identity** reading counts as a link: this exact series, in this order, at this length, on
a named artist. Everything else — reversed, a slice, negated, absolute value, a unit factor — is
recorded as a **candidate** and is not provenance. A column can match a reversed x axis by
arithmetic coincidence while the figure never draws it, and that is what used to be counted. The
artist is stored with the link, so a reference moving from one curve to another on the same figure
is a change rather than a silent pass.

### Acceptance

Acceptance is the advertised `--rtol 1e-9` / `--atol 1e-12`, applied to the stored values. It is
deliberately **not** digest equality: quantising and comparing hashes puts a hard edge inside the
tolerance band, so `1.0000000004999` and `1.0000000005001` are within `rtol` of each other yet round
to different tenth digits. The digest is kept only as a fast path. Values are stored rounded to
twelve significant figures, three orders tighter than the default `rtol`, so storage never decides
an acceptance the tolerance would not.

Nonfinite values are kept in position. Filtering them per axis loses the pairing between x and y:
with `y = [1, NaN, 2, 3]` and `y = [1, 2, NaN, 3]` the surviving values are the same list, so
compacting them hides a moved point. A NaN matches only a NaN.

### It fails rather than passing quietly

A missing baseline, a figure in the baseline that was not drawn, a figure or numerical series that
appeared without being baselined, an unknown chapter selection and a selection that discovers
nothing are all **failures**. Missing coverage is not a pass; new coverage is read and baselined
deliberately with `--update`, not absorbed.

### What it still cannot do

- Of 145 figures, **16 read committed data**; 129 compute everything in the figure module, so there
  is no committed reference for them. `--unverifiable` lists them; they are never reported as
  passing. Their numbers are still stored, so drift in them is caught.
- Of 127 reference series, **80 are found in their figure** (66 identity links, 14 candidates) and
  47 are not. Those 47 are parameters, grid sizes and columns the figure does not draw, reported by
  `--unmatched` as information. A same-length plotted series is not evidence of intent.
- **The baseline is drift detection, not proof of historical equivalence.** It records this run
  against the current working tree: no source revision, no independently reviewed per-series oracle.
  Observing a file open proves access, not that its values produced a particular curve.

## The gate

```bash
python3 tools/check_env.py              # the committed pins are consistent and installable
python3 tools/check_env.py --installed  # also: what is importable clears snapy's floors
```

It fails when a pin is not a full sha, when an artifact has no url or no sha256, when a Python
requirement is not pinned with `==`, or when the kintera revision `setup.sh` installs is below the
floor the snapy pin declares. It is cheap, needs no network and no build, and exists because the
v2.6.2/2.6.3 contradiction was invisible to every gate the report already had: both files were
internally consistent and nothing compared the pair.

## What was and was not verified

Verified on linux-x86_64 (RHEL 9, glibc 2.34, GCC 11.5.0, Python 3.11.13, CPU only) from a fresh
clone with an empty prefix and empty caches:

| | |
|---|---|
| `bash setup.sh` | exit 0 |
<!-- generated by tools/setup_notes.py --write; do not edit by hand -->
| `bash setup.sh --with-native` | exit 0; pydisort 1.8.13, pyharp 2.7.1, kintera 2.6.3, pinc 0.2.0, snapy 2.13.4, torch 2.10.0+cu128, netCDF 4.9.3 |
| `python3 tools/code_anchors.py check` | 3973 citations, 768 distinct, 0 unresolved |
<!-- end generated -->
| `python3 tools/check_env.py` | 5 passed, 1 reported, 0 failed |
| `python3 -m pytest tools/tests -q` | see the branch's own run |
| the 77 `chNN/checks/*_check.py` | 77 run, exit 0 |
| `ctest -R '^test_' -E '^test_shallow_xy_decomp$'` | 127 of 127 passed, exit 0 |
| `ctest -R 'shallow_xy_decomp\|restart' --output-on-failure` | 6 of 6 passed, exit 0 |
| do-cards (`SNAP_PIN_BUILD=… pytest -rs tools/tests/test_do_cards_run.py`) | 60 passed, 0 failed |
| `python3 tools/render_gate.py book/` | HTML and PDF produced |
| `diff .report-env/tex-packages.lock <(tlmgr list --only-installed)` | unchanged: the render and the tests fetched no TeX package |

Not exercised, and not claimed:

- **The macOS and linux-aarch64 branches.** Their artifacts are in `tools/env-pins.json` with digests
  taken from the files those URLs serve, but the install path has only been run on linux-x86_64.
- **GPU.** This is a CPU machine. The 2 GPU do-cards, the chapter-14 GPU and bpp items and every
  CUDA ctest are untouched and remain GPU-only exclusions.
- **Multi-rank.** The 7 multi-rank cards need 2 ranks with `pd-run`/`pd-combine`/`pd-inspect`; they
  are not part of the CPU do-card runner.
- **The Eigen suite** that `ctest` fetches. It is third-party, it is not scoped out by the chapter-15
  command, and its 821 failures are not snapy's.
