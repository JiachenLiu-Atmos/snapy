# Lane EF gates: G-EF-1, G-EF-4, G-EF-9 and the output-path gate

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

## G-EF-9: the single sedimentation kernel and the base species energy (`gef9.py`)

The gate for EF-6 (one sedimentation kernel) and EF-7 (one base
`species_energy`). For each deck it saves:

- `"W->E"` on the initial state;
- the x1 register total `hydro.F1` after the first stage;
- `hydro_u` after each of 3 rk steps.

The compare reports, per row group (IDN, IVX..IVZ, IPR, species; `W->E`
whole), the bitwise count and the maximum relative difference. dt is
`max_time_step`.

Decks (6):

- `gravity_sed_im`: `tests/test_gravity_sedimentation.yaml` (from `--tree`;
  ideal-moist, cloud const-vsed -2), with `const-gravity` grav1 -1 added and
  a uniform state (rho 1, p 1e5, vapour 0.05, cloud 0.02, at rest);
- `stokes_sed_im`: `tests/test_stokes_sedimentation.yaml` (from `--tree`;
  Stokes settling), with grav1 -10 and the same state;
- `uranus_mm`: G-1's short `uranus` deck (moist-mixture, nlim 3, const-vsed
  -200), from its initial restart, with the particle row seeded at 1e-5 (the
  initial state holds none);
- `uranus_mm_stokes`: the same deck with Stokes settling of 100 um drops
  (const-vsed set to 0, radius 1e-4 m, in the script). This is the old
  moist-mixture tensor Stokes path;
- `earth_crm_im` and `jupiter_1d_im`: G-1's `earth_crm` and
  `jupiter_evap_precip_1d` decks (ideal-moist; jupiter has three
  particles), from their initial restarts.

The G-1 decks and initial restarts are in `data/`. They were written by the
base build (EF-4 c61412df) in a G-1 run; the jupiter files are byte-identical
to the earlier ones. They are tracked with `git add -f` (`.gitignore` has
`data/`). sha256:

- `uranus.00000.restart` `977b650abf44199325fa5c8d7dee8d196856cd5ba3579b97e4b259b016a25d65`;
- `earth_crm.00000.restart` `74315cb2d80e22b52a3eaa13dfcbc5183bc53535b47f3a19b00f76802898f7cd`.

```sh
PYTHONPATH=<base-pkg> python gef9.py --tree <snapy> --device cuda --out runs/base
PYTHONPATH=<step-pkg> python gef9.py --tree <snapy> --device cuda --out runs/step
python gef9.py --compare runs/base runs/step
```

**Pass criterion.** The tolerance is 1e-12 relative on the stage-1 flux and
on `"W->E"`, set from the EF-6 measurement as EF.rst asks. The compare
prints `worst stage-1 flux or W->E relative difference … PASS|FAIL` and exits
1 on FAIL. The `u0..u2` lines are the state after each step, reported but
not gated.

**CPU reference, EF-6r d6f75964 against EF-4 c61412df** (identical to
EF-6 d4a9536e): worst 3.402e-13, PASS.

- Bitwise: `gravity_sed_im`, `stokes_sed_im` and `earth_crm_im`.
- `uranus_mm`: equal in value. Its F1 differs in 1049/65508 momentum
  entries only in the sign of exact zeros.
- `uranus_mm_stokes` F1: IVX..IVZ 1628/65508 at 3.40e-13, IPR 65/21836 at
  1.26e-14, species 14006/65508 at 6.50e-16.
- `jupiter_1d_im` F1: IPR 3/134 at 3.14e-16.

EF-7 against EF-12: worst 1.641e-13 (uranus IPR), with `W->E` at most
6.4e-16.

## Not run here

None of these has been run on CUDA by the author. `--device cuda` only moves
the block and the initial state, as `dumpset349.py` does.
