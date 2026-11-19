# Lane A forcing-deck bitwise gate (A-ph1)

A gate tool for lane A phase 1 (A.rst G1, "existing tests against the
reference, bitwise"). It is not part of the step. The #348 harness and the
#349 dump set run no lane-A source (their decks use const-gravity only), so
this tool switches every lane-A forcing on in some arm and compares
`hydro_u`, saved after each of 6 steps, byte for byte between two builds.

The column, its perturbed hydrostatic initial state and the fixed dt are the
#348 harness's (UCzhangxi/snapy `94e4601e`, `tools/harness348/h348.py`:
`config()`, `initial_w()` and the constants, copied unchanged).

## Arms (18)

| arm | grid | implicit-scheme | forcings |
|---|---|---|---|
| `cart_every_s0`, `cart_every_s9` | cartesian 45 x 8 | 0, 9 | Coriolis xyz (Ω1, Ω2, Ω3), top and bottom sponge, relax-bot-velo, relax-bot-temp (at-face), body-heat, top-cool (depth), bot-heat (thickness) |
| `stretched_every_s0`, `stretched_every_s9` | cartesian, x1rat 1.05 | 0, 9 | the same, top-cool by thickness |
| `moist_every_fp64`, `moist_every_fp32` | `tests/test_anomaly_storage.yaml` (moist, 2D) | 0 | relax-bot-comp, relax-bot-temp, top sponge, Coriolis; fp32 = `precision: float32`, anomaly, perturbation-flux, HLLC |
| `cart_123_2d`, `cart_123_3d` | cartesian, nx3 1 and 4 | 0 | Coriolis type 123; top-cool (thickness), bot-heat (depth) in 2D |
| `cart_diffusion`, `cart_diffusion_dyn` | cartesian | 0 | diffusion, kinematic and dynamic (+ Coriolis) |
| `sph_coriolis_s0`, `sph_coriolis_s9` | spherical-polar | 0, 9 | Coriolis (before and after the implicit solve, #355), top sponge |
| `gno_coriolis_s0`, `gno_coriolis_s9` | gnomonic-equiangle | 0, 9 | Coriolis, relax-bot-velo |
| `gno_coriolis_trad_s0`, `gno_coriolis_trad_s9` | gnomonic-equiangle | 0, 9 | Coriolis, traditional |
| `gno_aam_transport` | gnomonic-equiangle | 0 | Coriolis, conserve-axial-momentum: transport |
| `gno_aam_full_closure` | gnomonic-equiangle, interp_order 4 | 0 | Coriolis, full + closure |

## Usage

Build the base and the step with the same flags, and install each build's
Python package into its own directory. Run each build from outside the
source trees, so that `import snapy` picks up the installed build:

```sh
cd /some/scratch/dir
PYTHONPATH=<base-pkg> python <step-checkout>/tools/gate_a_forcing/gate_a_forcing.py --device cpu --out runs/base
PYTHONPATH=<step-pkg> python <step-checkout>/tools/gate_a_forcing/gate_a_forcing.py --device cpu --out runs/step
python <step-checkout>/tools/gate_a_forcing/gate_a_forcing.py --compare runs/base runs/step
```

- `--device cpu|cuda` (default cpu) is the device the block runs on. The
  dumps are always copied to the CPU. Compare CPU with CPU and CUDA with CUDA,
  never across devices.
- `--moist-deck` defaults to `tests/test_anomaly_storage.yaml` of the
  checkout that holds the script. Run both builds with the same script and the
  same card.
- Run the base twice and compare it with itself first (run-to-run
  determinism).
- `OMP_NUM_THREADS=1` and, on CPU, `CUDA_VISIBLE_DEVICES=` were set for the
  numbers below.

Each run writes `<arm>.pt` (a `[6, nvar, nc3, nc2, nc1]` tensor) per arm, and
`<arm>.err` for an arm that raised. It prints a sha256 prefix per arm.

## Pass

`--compare` exits 0 only if all 18 arms ran on both sides (a `.pt`, no
`.err`) and every pair is bitwise identical (shape, dtype and bytes).
Otherwise it exits 1 and lists the arms that differ or failed. The per-arm
`maxabs` is printed for information.

## Expected CPU result (A-ph1)

Base UCzhangxi/snapy v3/wave1 `6e9c7d5fa10873ee9d25711456452b7158ac5e74` vs
step `31206ad41c01a3f9868e8c75282add4dcc795dd3`, macOS arm64, AppleClang 21,
torch 2.10.0, Release, CUDA/UCX off:

- base vs step: `identical 18/18`, exit 0;
- base vs base: 18/18;
- the 18 dumps are pairwise distinct and finite.

## Negative control (a build patch, not a switch of this tool)

In `src/hydro/hydro_forward.cpp`, give the lane-A sources dt one ulp up:

```cpp
StageState stage{w, temp, std::nextafter(dt, 2. * dt),
                 mass_w ? mw.wc1 : torch::Tensor()};
```

On CPU this gave `identical 2/18` against the step, exit 1. The two arms
that stay equal are `moist_every_fp32` (the FP32 cast absorbs the ulp) and
`sph_coriolis_s0`.
