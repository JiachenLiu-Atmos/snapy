# Lane EF gates: G-EF-1, G-EF-4 and the output-path gate

The gate tools used for V3 wave 1/2 steps R-2, R-3, EF-3 and EF-4 (EF.rst rev
3.8, section 5). They are gate tooling only, not part of a step. Each one runs
the same cases on two builds and compares the results byte for byte. The build
is chosen by `PYTHONPATH`, which points at that build's installed `snapy`.
Run each build from outside the source trees, so that `import snapy` picks up
the installed package.

These are the scripts as they ran. The only changes are path arguments in
place of the author's local layout (`--tree`, `--d349`, `--h348`) and a
`--device` flag, which calls `block.to(device)` and moves the initial state to
that device. The CPU verdicts below were reproduced after that change.

Dependencies (published commits):

- `--d349`: `tools/dumpset349` at b59762901489958603a3d8e22ae9d873185e65d8
  (Linfeng Li, the #349 dump set).
- `--h348`: `tools/harness348` at UCzhangxi/snapy
  94e4601e1a575cd64d580ae809a2f9325e988c61 (the #348 harness).
- `--tree`: any snapy checkout. Only deck files are read from it
  (`tests/test_flux_positivity_carry.yaml`, `examples/*.yaml`,
  `tests/test_forcing_3d.yaml`).

## G-EF-1: the register posts after step 4.C (`gef1.py`)

The script runs one stage (stage 0, the same input on both builds) and
compares the hydro register totals `hydro.F1/F2/F3` after step 4.C. It checks
them row group by row group (IDN, IVX..IVZ, IPR, species), together with the
positivity meters (`positivity_hits/severe/min`, `lim_cut`, `lim_flux`).

Arms (8):

- the moist carry deck `tests/test_flux_positivity_carry.yaml`, set up as its
  C++ test does it:
  - limiter on, advected with lmars and with hllc;
  - settling (const-vsed cloud -2);
  - mixed advected and settling (cloud -1, sheared column);
  - limiter off with settling (P1 alone);
- #349's 2D moist settling deck (`dumpset349.py` `deck()`/`initial_w()`/
  `fixed_dt()`), with the limiter on and off;
- `jupiter_1d`: G-1's `jupiter_evap_precip_1d` deck, with three settling
  particles, from its initial restart. The restart is in `data/`, written by
  the base build (R-3 206e342f) in a G-1 run. dt is `max_time_step`.
  The repository's `.gitignore` ignores `data/`, so these two files are
  tracked with `git add -f`; edit them with that in mind.

```sh
PYTHONPATH=<base-pkg> python gef1.py --tree <snapy> --d349 <dumpset349> --device cuda --out runs/base
PYTHONPATH=<step-pkg> python gef1.py --tree <snapy> --d349 <dumpset349> --device cuda --out runs/step
python gef1.py --compare runs/base runs/step
```

Pass criterion (EF.rst G-EF-1): bitwise in every row group and meter. The one
exception is EF-3 against R-3, where the declared settling rounding gives
exactly two differences on CPU:

- `carry_mixed` IVX..IVZ: 1/30 entries, 2.458e-15 relative;
- `jupiter_1d` IPR: 7/134 entries, 2.681e-16 relative (one ulp).

The compare prints `RESULT n row groups or meters differ` and exits 1 when
n > 0, so read the listed differences against this.

## G-EF-4: the shared PositivityLimiter on scalars (`gef4.py`)

#349's moist settling column with one passive scalar. Each arm runs 6 rk3
steps at `fixed_dt` and saves `scalar_s` and `hydro_u` after every step, plus
the census. Arms (7):

- `lim`: eos limiter;
- `lim_nocov`: eos limiter, flux covariance off;
- `lim_upper`, `lim_upper_nocov`: binding `upper-bound: 0.0012`;
- `cov_only`: covariance rows alone, limiter off;
- `hole`: an empty hole in the ratio;
- `weno5ss`: the ss reconstruction.

```sh
PYTHONPATH=<base-pkg> python gef4.py --d349 <dumpset349> --device cuda --out runs/base
PYTHONPATH=<step-pkg> python gef4.py --d349 <dumpset349> --device cuda --out runs/step
python gef4.py --compare runs/base runs/step
```

Pass criterion: `identical 7/7`, exit 0. The limiter fires in `lim_upper*`
(2,610 hits) and `hole` (1,778 hits).

## Output-path gate (`output_path.py`, lane O's O5 gate)

Netcdf block files written through `NetcdfOutput` after one rk step. The
output options are every variable group, double precision, ghost zones,
slices and sums on each axis, combinations, a user flux and a 2D user field,
and the single-block combine. Decks:

- `examples/straka_single.yaml`, `examples/bryan.yaml`,
  `examples/jupiter_evap_precip_1d.yaml` and `tests/test_forcing_3d.yaml`;
- two VIC decks from `h348.config` (gnomonic face, Cartesian cell).

That gives 68 netcdf files plus a log.

```sh
PYTHONPATH=<base-pkg> python output_path.py --tree <snapy> --h348 <harness348> --device cuda --out runs/base
PYTHONPATH=<step-pkg> python output_path.py --tree <snapy> --h348 <harness348> --device cuda --out runs/step
python output_path.py --compare runs/base runs/step
```

Pass criterion: `files 69 vs 69 (68 netcdf); identical 69`, exit 0. For EF-3
against R-3 the declared settling rounding leaves `identical 62`: the 7
`jup1d/*` files differ, and nothing else may.

## Not run here

None of these has been run on CUDA by the author. `--device cuda` only moves
the block and the initial state, as `dumpset349.py` does.
