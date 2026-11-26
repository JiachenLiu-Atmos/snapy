# 5 Hydrostatic reference and well-balanced treatment

<!-- PROVENANCE
pin: snapy@73c45a7c8c991209924e1137992352c335575062 (UCzhangxi/snapy v3/final, head of chengcli/snapy#377)
outline: doc/tech-report/STRUCTURE.md at tech-report-v3-design aca6a84bc06f29ce18a5d8a314c9e435b20a829d, chapter 5 (5.1-5.3)
sources: src/recon/x1_reference.*, src/hydro/hydro_ref_x1_impl.h, src/recon/well_balanced_x1.cpp, src/hydro/balance_column.*, src/recon/x1_rows.cpp, src/hydro/hydro.cpp, src/hydro/hydro_forward.cpp, src/forcing/const_gravity.cpp and the tests named in each Tests layer, all read at the pin
form: pre-conversion draft (STYLE.md section 0): hand-numbered equations, `path:lines@sha` code locations; section 10 (Quarto markup) does not bind it yet
status: draft
author: Xinyue Wang
reviewer: (checker from tools/chapter-owners.tsv)
-->

> **Draft conventions** (as in the chapter 4 draft). Every code location is `path:lines@73c45a7c`. The short sha
> names the pin in the PROVENANCE block. A bare `:N` refers to the file of the last full citation before it.
> Equations are numbered (5.n) by hand. Every tolerance is read from the test source at the pin. No number in
> this draft comes from a run; a number quoted from a code comment or a test docstring says so and is marked
> **unverified**. Units: SI; $g>0$ is the downward gravity magnitude, $g = -g_1$.

A resting, stratified atmosphere is the state a weather or climate model spends most of its time near. In such
a column, the pressure force and gravity are each large and nearly cancel. If the discrete pressure force and the
discrete gravity source do not cancel exactly, the difference is a spurious force. It drives spurious vertical
motion that can dwarf the physical signal. A scheme that keeps a discrete hydrostatic column exactly at rest is
called *well-balanced*.

snapy makes its $x_1$ (vertical) sweep well-balanced in three pieces. Section 5.1 builds a discrete hydrostatic
reference from the column's own density: face pressures whose differences equal the gravity source exactly, and
cell and face densities that go with them. Section 5.2 reconstructs only the departure from that reference and
adds the reference back at the faces, so the solver sees the balanced face pressures whenever the departure is
zero. Section 5.3 gives the tool that prepares an initial column that is at rest under this scheme
(`balance_column`), and the higher-order face density the reference uses with WENO5/cp5.

Chapter 4 describes the plain sweep this path replaces. Chapter 6 holds the gravity work, and chapter 12 the FP32
rows of the split path.

---

## 5.1 The $x_1$ hydrostatic reference (`dynamics/wb-wall-clamp`; engaged by gravity)

### Summary

The reference is a set of pressures and densities computed per column from the current cell densities. It has
face pressures $p^{\mathrm{ref}}_{i\pm1/2}$ obtained by integrating $\partial_1 p = -\rho g$ exactly over each cell.
It has a cell pressure $p_{\mathrm{ref},i}$, the cell average of a smooth profile through those faces. It has a
cell density $\rho_{\mathrm{ref},i}$ and a face density $\rho_{\mathrm{sf},i-1/2}$ at the same temperature as the
column. The reference is rebuilt every stage, so it follows the state. The well-balanced sweep (5.2) subtracts it
before the reconstruction.

::: {.callout-note title="At a glance"}
**Purpose.** Discrete hydrostatic face pressures, eq. (5.1), with cell and face reference values, eqs. (5.2)-(5.6).

**Data.** In: cell primitives $\rho_i, p_i$ (with ghosts), $h_i$, $g$. Out: `psf_lo` (face pressure below cell
$i$), `pref`, `dref` (cell), `dsf_lr` (face density per side) [Pa, kg m$^{-3}$].

**Placement.** Every RK stage, step (2) of `HydroImpl::forward`, before the $x_1$ reconstruction.

**Activation.** On whenever $g_1\ne0$, the state has a pressure row and the EOS is not shallow water.
`dynamics/wb-wall-clamp` (default `true`) keeps the stencils off the wall ghosts.

**Guarantee / limits.** $p^{\mathrm{ref}}_{i-1/2}-p^{\mathrm{ref}}_{i+1/2} = g\rho_ih_i$ up to the rounding of one
addition. Uniform cell pressure: a six-face rule, exact for quintics.

**Evidence.** `test_hydro_ref_x1.release`, `test_hydro_ref_x1_rf_wall.release`, `test_stretched_x1_ref.release`,
`test_wb_ref_wall.release` (Tests layer).
:::

**Local symbols** (NOTATION.md §6 has $\rho_{\mathrm{ref}}, p_{\mathrm{ref}}$ for the cell values and
$\rho_{\mathrm{sf}}, p_{\mathrm{sf}}$ for face values, and $\chi = \rho/p$; these are used as defined there):
$p^{\mathrm{ref}}_{i\pm1/2}$ the reference face pressures, written with the face index in the subscript and
`ref` as a superscript only to keep the face index readable (to be settled with the editor: NOTATION's
$p_{\mathrm{sf}}$ is the same quantity); $p_{\mathrm{a}}$ the top anchor; $\chi^{\mathrm s}_i$ the $\chi$ the density
reference uses (NOTATION §6 lists $\chi^s_i$); $w^{(6)}_m$ the six-face weights.

The reference is engaged for every gravity run with a pressure row, except shallow water
(`src/hydro/hydro_forward.cpp:479-480@73c45a7c`). The wall clamp is on by default
(`src/options/hydro.yaml:25@73c45a7c`). Turning it off makes the reference read the ghost cells a boundary
function wrote, which are not a hydrostatic continuation of the column (5.1 Tests, the clamp control).[^ch05-aliases]

[^ch05-aliases]: Aliases in the code: "wb", "WB" (well-balanced), "the kernel" (`hydro_ref_x1_impl`), "wb_ref4"
(the key-4 face density rows), "split path" (`perturbation-flux`), "lane C". Derivation notes the code points to:
`docs/derivations/lane-c-notes.md` and `docs/derivations/wb-ref4.md` (not re-read for this draft); report
sources `doc/tech-report/sources/deriv__wb-ref4.md`, `gw__NEXTPR_spec_wbref_exact.md`. Credit and trail:
chengcli/snapy#289. [source: comments at `src/recon/x1_reference.cpp:29@73c45a7c`,
`src/hydro/balance_column.hpp:14-17@73c45a7c`]

### Derivation

**Face pressures.** Integrate the hydrostatic equation $\partial_1 p = -\rho g$ over cell $i$, from
$x_{1,i-1/2}$ to $x_{1,i+1/2}$:

$$
p(x_{1,i-1/2}) - p(x_{1,i+1/2}) = g\int_{x_{1,i-1/2}}^{x_{1,i+1/2}}\rho\,\mathrm dx_1 = g\,\bar\rho_i\,h_i .
$$

No approximation enters: the cell average is exactly what the integral needs. Given one face value, the
recurrence fixes every other one:

$$
p^{\mathrm{ref}}_{i-1/2} = p^{\mathrm{ref}}_{i+1/2} + g\,\bar\rho_i\,h_i . \tag{5.1}
$$

The code scans (5.1) downward from the top owned cell `iu` to cell 0, and upward from `iu`+1 into the upper
ghosts, $p^{\mathrm{ref}}_{i+1/2} = p^{\mathrm{ref}}_{i-1/2} - g\bar\rho_ih_i$
(`hydro_ref_x1_pressure_scan_impl`, `src/hydro/hydro_ref_x1_impl.h:32-57@73c45a7c`). Each value is clamped below at
the smallest positive normal number of the type (`:43-44`, `:53-54`).

**The anchor.** On a column that one block owns, the top face of cell `iu` is anchored by an isothermal half-cell
step from the cell value. Assume $p/\rho$ is constant over the upper half of the cell and equal to the cell's
$p_{iu}/\rho_{iu}$. Then $p(x) = p_{iu}\exp(-g(x-x_{1,iu})\rho_{iu}/p_{iu})$, and at $x - x_{1,iu} = h_{iu}/2$:

$$
p_{\mathrm a} = p_{iu}\,\exp\!\Big(-\frac{g\,h_{iu}}{2\,p_{iu}/\rho_{iu}}\Big) \tag{5.2}
$$

(`hydro_ref_x1_scan_impl`, `src/hydro/hydro_ref_x1_impl.h:59-80@73c45a7c`). This treats the cell value as the
point value at the midpoint, an $O(h^2)$ approximation. On a column split into several $x_1$ blocks, the top block
anchors itself by (5.2). Every lower block takes the face value the block above computed at its own bottom face,
relayed down the column (`X1Reference::evaluate`, `src/recon/x1_reference.cpp:86-94,148-157@73c45a7c`). The
reference is then one continuous scan of the whole column. The anchor only adds a constant offset to the scanned
faces of a column. (5.1) holds whatever it is, and the anchor drops out of the balance (5.2 Derivation).

**Cell pressure.** The reconstruction needs a cell-average reference pressure. Without one it would subtract a
face value from a cell average, and the difference would not vanish at rest. On a uniform grid, take the quintic
$P(x)$ through the six face values $x_{1,i-5/2}..x_{1,i+5/2}$ and average it over cell $i$:

$$
p_{\mathrm{ref},i} = \sum_{m=0}^{5} w^{(6)}_m\, p^{\mathrm{ref}}_{i-5/2+m},\qquad
w^{(6)} = \Big(\tfrac{11}{1440}, -\tfrac{31}{480}, \tfrac{401}{720}, \tfrac{401}{720}, -\tfrac{31}{480}, \tfrac{11}{1440}\Big). \tag{5.3}
$$

These are the constants at `src/hydro/hydro_ref_x1_impl.h:179-180@73c45a7c`. Integrating the six Lagrange basis
polynomials over the cell reproduces them exactly (sympy 1.14.0, Python 3.11.13; script in Limits). The rule is
exact for polynomials of degree $\le5$. On $p = e^{-x_1/H}$ its error is
$\tfrac{191}{60480}(h/H)^6\,p$ to leading order (same script). Next to a clamped wall, the window is shifted inside
the owned cells, using faces $x_{1,i_l-1/2}..x_{1,i_l+9/2}$, and averaged over the first or second cell. That gives
the two one-sided rows

$$
\begin{aligned}
w^{(6,0)} &= \big(\tfrac{95}{288}, \tfrac{1427}{1440}, -\tfrac{133}{240}, \tfrac{241}{720}, -\tfrac{173}{1440}, \tfrac{3}{160}\big),\\
w^{(6,1)} &= \big(-\tfrac{3}{160}, \tfrac{637}{1440}, \tfrac{511}{720}, -\tfrac{43}{240}, \tfrac{77}{1440}, -\tfrac{11}{1440}\big),
\end{aligned} \tag{5.4}
$$

mirrored at the top wall (`:196-210`); both rows were checked symbolically. In every case the rule's value is
accepted only if it lies between the cell's two face pressures. Otherwise the cell takes the face mean
$\tfrac12(p^{\mathrm{ref}}_{i-1/2}+p^{\mathrm{ref}}_{i+1/2})$ (`:176`, `:190`). On a declared stretched grid the same
average is taken with per-cell weights. These come from three-point Gauss-Legendre quadrature of the quintic
Lagrange interpolant on the actual faces (`six_face_rows`, `src/recon/x1_rows.cpp:340-374@73c45a7c`). That
quadrature is exact for degree 5, so it is exact for the quintic. On an unequal grid that is not declared
stretched, the legacy branch uses the isothermal cell mean

$$
p_{\mathrm{ref},i} = \frac{p^{\mathrm{ref}}_{i-1/2}-p^{\mathrm{ref}}_{i+1/2}}{\ln\big(p^{\mathrm{ref}}_{i-1/2}/p^{\mathrm{ref}}_{i+1/2}\big)}
 = \frac{g\bar\rho_ih_i}{\ln\big(p^{\mathrm{ref}}_{i-1/2}/p^{\mathrm{ref}}_{i+1/2}\big)}, \tag{5.5}
$$

which is the exact cell mean of an exponential through the two faces (checked symbolically). It falls back to the
face mean when the face ratio is within $10^{-6}$ of 1 (`src/hydro/hydro_ref_x1_impl.h:231-236@73c45a7c`). The
operator picks the uniform branch when $\max h - \min h < 10^{-10}\,\overline{h}$
(`src/recon/x1_reference.hpp:55-60@73c45a7c`).

**Density reference.** The reference density is formed at the column's own temperature. With $\chi = \rho/p$,

$$
\rho_{\mathrm{ref},i} = p_{\mathrm{ref},i}\,\chi^{\mathrm s}_i,\qquad
\rho_{\mathrm{sf},i-1/2} = p^{\mathrm{ref}}_{i-1/2}\,\frac{h_i\chi^{\mathrm s}_{i-1} + h_{i-1}\chi^{\mathrm s}_i}{h_{i-1}+h_i} \tag{5.6}
$$

(`src/hydro/hydro_ref_x1_impl.h:238-255@73c45a7c`; on a non-stretched grid the face weight is $\tfrac12$,
`:250-252`). The fraction in (5.6) is the linear interpolation of $\chi^{\mathrm s}$ from the two cell midpoints to the
face. In the interior $\chi^{\mathrm s}_i = \rho_i/p_i$, the cell's own value. Within two cells of a clamped wall,
$\chi^{\mathrm s}$ is the binomial average $(1,4,6,4,1)/16$ over values continued past the wall
(`hydro_ref_x1_rop_smooth`, `:103-136`). The continuation is linear where $\chi$ rises toward the wall and
ln-linear where it falls, measured in cell-centre steps on a stretched grid (`:82-101`). So at rest
($p_i = p_{\mathrm{ref},i}$), $\rho_{\mathrm{ref},i} = \rho_i$ in the interior.

**Per-side face density with WENO5/cp5.** When the $x_1$ reconstruction is `weno5` or `cp5` and the block has
at least 6 $x_1$ cells, each side's face density is replaced by the reconstruction's linear cp5 row applied to
$\rho_{\mathrm{ref}}$ (chapter 4, eq. (4.2)), kept only where it lies between the two adjacent cell values. At a
reflecting wall face both sides take their mean (`X1Reference::evaluate`, `src/recon/x1_reference.cpp:173-241@73c45a7c`).
Section 5.3 gives the fourth-order face rows of conversion order 4.

**Exact and truncated.** (5.1) is exact given $\bar\rho_i$ and the anchor. (5.3)/(5.4) are sixth-order quadratures of
a quintic through exact face values. (5.5) is exact for an isothermal cell. (5.2) is $O(h^2)$, but it only shifts a
column by a constant. The density reference (5.6) is second order at the faces.

### Numerical method

**Order of evaluation** (`X1Reference::evaluate`, `src/recon/x1_reference.cpp:75-248@73c45a7c`, comment at
`src/recon/x1_reference.hpp:233-234@73c45a7c`):

1. take the anchor from the block above, if the column is split;
2. run the kernel: scan (5.1), then the cell values (5.3)-(5.6) per cell;
3. form the key-4 resolution flag (5.3);
4. pass this block's bottom face pressure to the block below;
5. exchange the ghost rows of $p_{\mathrm{ref}}, \rho_{\mathrm{ref}}$ with the $x_1$ neighbours, so that the departure
   $w'$ (5.2) is the same on both sides of a seam;
6. apply the key-4 face rows;
7. form the per-side face density.

The operator is cached per grid. It is rebuilt when `dx1f` changes identity, version, dtype or device, or any of
the wall flags change (`HydroReferenceX1Operator::matches`, `src/recon/x1_reference.hpp:71-78@73c45a7c`). The
stretched rows are built once, on the grid's device, so evaluating the reference inside a captured CUDA-graph
segment copies nothing from the host (`:61-68`).

**Wall clamp.** With `wb-wall-clamp: true` and a physical $x_1$ wall, the six-face window of the two cells next to
the wall is moved inside the owned cells (5.4). $\chi$ is continued past the wall rather than read from the ghosts
(`src/hydro/hydro_ref_x1_impl.h:165-173,239-245@73c45a7c`). A block with fewer than 5 owned cells, or with
`iu` $<4$, cannot fit the one-sided row and keeps the face mean (`:167-173`).

**Split path (`perturbation-flux`).** The kernel is evaluated once on the packed pair (FP64 background, finite
increment). Its nonlinear decisions (anchor, floors, guards) read the total, while the arithmetic keeps the
increment separately (`HydroReferenceX1Operator::evaluate_split`, `src/recon/x1_reference.hpp:108-137@73c45a7c`;
`src/recon/x1_reference.cpp:104-125@73c45a7c`). It is FP64 on CPU or CUDA only (`src/recon/x1_reference.hpp:117-118@73c45a7c`).
In FP32 runs the block also carries an FP64 face gauge, relayed with the anchor (`src/recon/x1_reference.cpp:122-125,150-154@73c45a7c`).

**Figure.** `src/snapy_report/ch05/figures/fig_refkernel_scan.py` (diagram of the scan; the outline's figure). Its
agreement with the pin is not checked in this draft (Limits).

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| entry from the hydro | `src/hydro/hydro.cpp:526-544@73c45a7c` | `HydroImpl::_hydro_ref_x1` | `wb-wall-clamp`, `conversion-order`, the $x_1$ `reconstruct` type |
| engage condition and call | `src/hydro/hydro_forward.cpp:479-480,514@73c45a7c` | `wb_x1` in `HydroImpl::forward` | gravity |
| evaluation, anchor relay, ghost rows, face density | `src/recon/x1_reference.cpp:75-248@73c45a7c` | `X1Reference::evaluate` | |
| grid branch, caches | `src/recon/x1_reference.hpp:41-164@73c45a7c` | `HydroReferenceX1Operator` | |
| scan, anchor | `src/hydro/hydro_ref_x1_impl.h:32-80@73c45a7c` | `hydro_ref_x1_pressure_scan_impl`, `hydro_ref_x1_scan_impl` | |
| cell and face references | `src/hydro/hydro_ref_x1_impl.h:82-270@73c45a7c` | `hydro_ref_x1_cell_impl`, `hydro_ref_x1_rop_smooth`, `hydro_ref_x1_wall_rop` | |
| stretched six-face rows | `src/recon/x1_rows.cpp:340-374@73c45a7c` | `six_face_rows` | `cells/x1rat`, `geometry/x1f` |
| grid kinds | `src/hydro/hydro_dispatch.hpp:10-12@73c45a7c` | `kX1RefNonUniform`, `kX1RefUniform`, `kX1RefStretched` | |

Walk-through. `HydroImpl::_hydro_ref_x1` fills an `X1ReferenceConfig`: the physical-wall flags, the clamp, the
reflecting walls, the conversion order and the $x_1$ interpolant. It passes $g = -g_1$ as a downward magnitude
(`src/hydro/hydro.cpp:528-543@73c45a7c`). Checks a user can hit: "hydro pressure scan: density and grid shapes
disagree" and the dtype check (`src/recon/x1_reference.hpp:87-93@73c45a7c`); "split hydro reference requires FP64
on CPU or CUDA" (`:117-118`).

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_hydro_ref_x1.release` `HydroRefX1Dispatch.cpu_matches_tensor_reference` | the CPU kernel equals a torch re-implementation, FP32 and FP64, uniform and not, with and without anchor | `allclose(rtol=atol=2.e-5)` FP32, `2.e-12` FP64 (through `expect_close`) | `tests/test_hydro_ref_x1.cpp:184-207@73c45a7c` |
| ... `wall_clamp_keeps_the_interior_free_of_the_ghosts` and its control `without_the_clamp_the_ghosts_do_reach_the_interior` | perturbing the ghosts leaves interior `pref`, `dsf`, `dref` unchanged with the clamp, and changes them without it | `torch::equal` (bitwise) | `tests/test_hydro_ref_x1.cpp:216-247@73c45a7c` |
| ... `cuda_matches_cpu` | CUDA equals CPU | `allclose(2.e-12, 2.e-12)` | `tests/test_hydro_ref_x1.cpp:249-275@73c45a7c` |
| ... `cuda_matches_cpu_on_the_stretched_rows` | CUDA equals CPU on the stretched rows | at most `kMaxUlp` = 32 ulp | `tests/test_hydro_ref_x1.cpp:331-403@73c45a7c` |
| ... `the_wall_reference_does_not_depend_on_the_blocks_x1_size` | a 4-cell block with the relayed anchor reproduces the bottom 4 cells of a 12-cell block | relative `1.e-14` | `tests/test_hydro_ref_x1.cpp:470-506@73c45a7c`; the comment gives 3.3e-3 relative as the cost of a blanket fallback (**unverified** here) |
| `test_hydro_ref_x1_rf_wall.release` | the face $\chi$ of (5.6) is the linear interpolation to the face position (stretched), the midpoint on uniform faces; wall continuation by distance | `EXPECT_NEAR` relative `1.e-12`; `1.e-14` on step counts; separation `> 1.e-6` | `tests/test_hydro_ref_x1_rf_wall.cpp:83-168@73c45a7c` |
| `test_stretched_x1_ref.release` `iv_rows_on_uniform_faces_are_the_uniform_weights` | `six_face_rows` on uniform faces reproduces (5.3)/(5.4) | $\le 4\epsilon_{\mathrm m}$ (FP64) | `tests/test_stretched_x1_ref.cpp:434-458@73c45a7c` |
| ... `iv_adiabat_cells_within_2x_uniform` | stretched-row cell error within twice that of a uniform patch of the same width | ratio $\le 2$ | `tests/test_stretched_x1_ref.cpp:407-432@73c45a7c` |
| ... other cases (`viscous_*`, `heat_layer_*`, `covariance_slopes_*`, refusals) | stretched-grid neighbours of this chapter | `EXPECT_NEAR` 1.e-12; `EXPECT_LT` 1.e-16, 1.e-12 | `tests/test_stretched_x1_ref.cpp:140-311@73c45a7c` |
| `test_wb_ref_wall.release` `faces_next_to_the_walls_are_second_order` (and `_cuda`) | the face density reference and the solver's face densities one and two faces from each wall converge at order $\ge1.7$ (nz 32$\to$64); the solver's face density error at nz 64 is $\le$ `kWallRho` | 1.7; `kWallRho` = 3.e-5 | `tests/test_wb_ref_wall.cpp:226-277@73c45a7c` |
| ... `order_table_cuda_matches_cpu` | every face error, CUDA against CPU | $\le$ 1.e-13 | `tests/test_wb_ref_wall.cpp:281-297@73c45a7c` |

`test_hydro_ref_x1` and `test_wb_ref_wall` run on CPU, with CUDA cases that skip without CUDA. `test_stretched_x1_ref`
also runs as `test_stretched_x1_ref_x1_centroid.release` with `SNAP_X1_CENTROID_EXACT` set
(`tests/CMakeLists.txt:57-62@73c45a7c`). One rank. The `test_wb_ref_wall` comment quotes measured wall errors
(2.30e-5, 3.1e-7, 1.39e-3, `tests/test_wb_ref_wall.cpp:229-236@73c45a7c`); they are **unverified** here, and the
asserted bounds above are what the chapter relies on. **Run evidence at the pin: missing evidence.**

### Limits

1. The anchor (5.2) is an isothermal half-cell step from the top cell value. It is $O(h^2)$ and shifts the column by a
   constant (harmless for the balance, 5.2), but the reference's absolute level at the top is approximate.
2. Three operators exist for the cell pressure (uniform, stretched, legacy non-uniform). The code says they are
   "different operators, not accuracy settings" (`src/recon/x1_reference.hpp:37-40@73c45a7c`). An unequal grid that
   is not declared stretched takes the legacy isothermal branch (5.5) at run time.
3. Without the wall clamp the reference reads boundary ghosts (Tests control). `balance_column` refuses to run
   without it (5.3).
4. The figure scripts named here were drawn against earlier pins. Their agreement with 73c45a7c is not checked.
5. The sympy script for (5.3)-(5.5) lives outside the report tree and is not a committed check. Local-symbol
   decisions ($p^{\mathrm{ref}}_{i\pm1/2}$ against NOTATION's $p_{\mathrm{sf}}$; $p_{\mathrm a}$; $w^{(6)}$) are open with the editor.
6. Curved $x_1$ (spherical-polar, cubed sphere): the scan (5.1) is the plain-measure integral. Its rest balance on
   curved grids depends on the geometric source and on `SNAP_X1_CENTROID_EXACT` (chapter 4, section 4.4; chapter 3).
   It is not derived here.

---

## 5.2 Well-balanced face states (`WellBalancedX1`; engaged by gravity)

### Summary

The $x_1$ sweep with gravity reconstructs the departure of $p$ and $\rho$ from the reference, not $p$ and $\rho$
themselves. It then adds the reference back at the faces. For a column at rest that equals its reference, the
departure is zero, both sides of every face see the same reference face pressure, and the Riemann solver returns
a momentum flux whose divergence cancels the gravity source exactly.

::: {.callout-note title="At a glance"}
**Purpose.** Face states $p^{\mathrm{L,R}} = \mathcal W[p-p_{\mathrm{ref}}] + p^{\mathrm{ref}}$ and
$\rho^{\mathrm{L,R}} = \mathcal W[\rho-\rho_{\mathrm{ref}}] + \rho_{\mathrm{sf}}$, eqs. (5.7)-(5.9).

**Data.** In: `hydro_w` (changed and restored in place) and the reference of 5.1. Out: $x_1$ face states
`(2, nvar, ...)`.

**Placement.** Every RK stage, step (2), instead of the plain $x_1$ reconstruction of chapter 4.

**Activation.** Same as 5.1. Reflecting (non-outflow) physical walls get even-parity ghosts.

**Guarantee / limits.** A column with $p_i - p_{\mathrm{ref},i}$ constant along the column and $v_1 = 0$ has zero
$x_1$ momentum tendency, to round-off (Cartesian).

**Evidence.** `test_hydrostatic.release`, `test_wb_ref_wall.release`, `test_face_floor.release` (Tests layer).
:::

**Local symbols:** $w'$ the departure of a row from its reference; $\mathcal W[\cdot]$ the reconstruction operator
(NOTATION §3).

### Derivation

**Face states.** Let $p'_i = p_i - p_{\mathrm{ref},i}$ and $\rho'_i = \rho_i - \rho_{\mathrm{ref},i}$. The sweep reconstructs
$p'$, $\rho'$ and the other rows with the chapter 4 operator $\mathcal W$, with the floors off, because a departure
may legitimately be negative. It then sets

$$
p^{\mathrm L}_{i-1/2} = \mathcal W^{\mathrm L}[p']_{i-1/2} + p^{\mathrm{ref}}_{i-1/2},\qquad
p^{\mathrm R}_{i-1/2} = \mathcal W^{\mathrm R}[p']_{i-1/2} + p^{\mathrm{ref}}_{i-1/2}, \tag{5.7}
$$
$$
\rho^{K}_{i-1/2} = \mathcal W^{K}[\rho']_{i-1/2} + \rho^{K}_{\mathrm{sf},i-1/2},\quad K\in\{\mathrm L,\mathrm R\} \tag{5.8}
$$

(`WellBalancedX1::forward`, `src/recon/well_balanced_x1.cpp:31-39,94-97,106-107,132-133@73c45a7c`). Both sides of a face
take the same reference face pressure, `psf_lo` of the cell above the face. The density takes the per-side face
density of 5.1. If a restored face pressure is not positive, that side takes the reference face pressure alone. If a
restored density is not positive, the left side takes the density of the cell below and the right side the density of
the cell above (`:114-143`).

**Balance.** Take a Cartesian column at rest, $v_1 = 0$, whose departure is a constant, $p'_i = c$ for every $i$.
Every scheme of chapter 4 reproduces a constant (eq. (4.1)), so $\mathcal W^{\mathrm{L,R}}[p'] = c$ and

$$
p^{\mathrm L}_{i-1/2} = p^{\mathrm R}_{i-1/2} = p^{\mathrm{ref}}_{i-1/2} + c .
$$

With equal pressures and zero velocity on both sides, HLLC returns momentum flux $p^{\mathrm c} = p$ and mass flux 0
(chapter 4, eqs. (4.10)-(4.11) with $\lambda^{\mathrm c} = 0$). LMARS returns $p_f = p$ and $v_{n,f} = 0$ (eq. (4.12)).
Neither depends on the face densities. The $x_1$ momentum tendency of cell $i$ is the flux difference over the cell
plus the gravity source, $\rho_ig_1 = -g\rho_i$ (`ConstGravityImpl::forward_split`,
`src/forcing/const_gravity.cpp:52-58@73c45a7c`, with `non-hydrostatic` 1):

$$
\begin{aligned}
\dot m_{1,i} &= -\frac{p^{\mathrm{ref}}_{i+1/2}+c - \big(p^{\mathrm{ref}}_{i-1/2}+c\big)}{h_i} - g\rho_i
 = \frac{p^{\mathrm{ref}}_{i-1/2}-p^{\mathrm{ref}}_{i+1/2}}{h_i} - g\rho_i\\
 &\overset{(5.1)}{=} \frac{g\rho_ih_i}{h_i} - g\rho_i = 0 .
\end{aligned} \tag{5.9}
$$

The constant $c$ and the anchor cancel in the difference, and the density in (5.1) is the same cell density the
source uses. The only error is the rounding of the scan's subtraction and of the division, so the balance holds to
round-off, not bit for bit. The code comment states the same identity, "psf_lo(i) - psf_hi(i) = g*rho(i)*dx1f(i)
identically" (`src/hydro/hydro_forward.cpp:469-477@73c45a7c`). With `non-hydrostatic` $\alpha_{\mathrm{nh}}<1$, the
hydrostatic split adds $(1-\alpha_{\mathrm{nh}})$ times the face-pressure difference
(`src/hydro/hydro_forward.cpp:565-572,1126-1132@73c45a7c`). The residual (5.9) is then scaled by $\alpha_{\mathrm{nh}}$
and is still zero.

**Exact and truncated.** (5.9) is exact for the stated state in Cartesian geometry. Away from rest the face states
are those of $\mathcal W$ applied to the departure, so they carry chapter 4's order on $p'$ and $\rho'$, plus the
reference's own face density error (5.1, 5.3).

### Numerical method

**Steps** (`src/recon/well_balanced_x1.cpp:16-152@73c45a7c`):

1. Save $p$ and $\rho$, and replace them in `wx1` by the departures (`:31-39`). On the split path, $p'$ is formed from the
   background and increment parts separately (`:34-35`). In FP32 runs the face gauge is also subtracted (`:38`).
2. At a physical, non-outflow $x_1$ wall, fill the ghosts of $p'$ and $\rho'$ with even parity,
   $q'_{i_s-m} = q'_{i_s+m-1}$ (`:41-55`). The wall flags are set in `HydroImpl::forward` as physical and not outflow
   (`src/hydro/hydro_forward.cpp:532-536@73c45a7c`). With conversion order 4 the wall ghosts come from the polynomial
   fit of chapter 4 instead (`src/recon/well_balanced_x1.cpp:56-57@73c45a7c`).
3. With `dynamics/x1-mass-covariance`, correct the cell velocity by $-\tfrac{h^2}{12}\partial_1\rho\,\partial_1v_1/\rho$, with
   one-sided slopes at the walls and odd-parity velocity ghosts (`:60-86`).
4. Reconstruct with `floor = false` and post the floor masks (`:94-97`).
5. Restore $p$, $\rho$, $v_1$ in `wx1` (`:99-101`).
6. Add the reference back at the faces, (5.7)-(5.8), with the positivity fallbacks. Each fallback is counted under
   `face_floor.p` or `face_floor.rho`, on faces `is..iu+1` (`:106-143`).
7. With conversion order 4, the wall face mirrors its owned side, with $v_1$ reversed (`:144-150`).

**Seams.** Across an $x_1$ block seam the departure is exchanged through the ghost rows of
$p_{\mathrm{ref}}, \rho_{\mathrm{ref}}$ (5.1, step 5), so both blocks reconstruct the same $w'$.

**Figure.** `src/snapy_report/ch05/figures/fig_wbx1_ghosts.py` (diagram of the parity ghosts; the outline's figure).
Not re-checked at the pin.

**Stage, order.** Explicit, every RK stage. The order is that of $\mathcal W$ on the departure (chapter 4). The
measured wall-face order is in Tests.

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| the WB sweep | `src/recon/well_balanced_x1.cpp:16-152@73c45a7c` | `WellBalancedX1::forward` | gravity |
| its inputs and flags | `src/recon/reconstruct.hpp:186-209@73c45a7c` | `WellBalancedX1` | `x1-mass-covariance`, `conversion-order` |
| set-up and call | `src/hydro/hydro_forward.cpp:503-546@73c45a7c` | `HydroImpl::forward` step (2) | `perturbation-flux` |
| gravity source | `src/forcing/const_gravity.cpp:47-71@73c45a7c` | `ConstGravityImpl::forward_split` | `forcing/const-gravity/grav1`, `non-hydrostatic` |
| hydrostatic split | `src/hydro/hydro_forward.cpp:565-572,1126-1132@73c45a7c` | `rho_grav` in `HydroImpl::forward` | `non-hydrostatic` |

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_hydrostatic.release` `HydrostaticAtmosphere.cubed_sphere_remains_at_rest` | six-panel gnomonic-equiangle mesh, $24\times8\times8$ per panel, WENO5, LMARS, RK3, $g_1=-1$, reflecting $x_1$ walls, an adiabatic column ($\gamma = 1.4$) initialised from point values: after 100 steps, $\max\lvert v_1\rvert$ | `EXPECT_LT` 1.e-8 | `tests/test_hydrostatic.cpp:100-166@73c45a7c` |
| `test_wb_ref_wall.release` | the face densities next to the walls (5.1 Tests) | 1.7; 3.e-5; 1.e-13 | `tests/test_wb_ref_wall.cpp:229-297@73c45a7c` |
| `test_face_floor.release` `face_floor_uses_adjacent_density`, `face_floor_fires_on_an_unresolved_column`, `face_floor_counter_sees_the_fallback` | a dipped column triggers the density fallback of step 6: dipped-face mass flux $\lvert F\rvert<10^{-9}$; flux at an unresolved column $1.28524\times10^{-8}$; `face_floor.rho` $\ge 1$ | `EXPECT_LT` 1.e-9; relative 1.e-5; `EXPECT_GE` 1 | `tests/test_face_floor.cpp:68-101,150-162@73c45a7c` |

`test_hydrostatic` initialises the column with point values of a continuous adiabat
(`tests/test_hydrostatic.cpp:100-123@73c45a7c`), not with a discretely balanced state. So it tests that the scheme
stays near rest, to $10^{-8}$, not the exact identity (5.9). No test at the pin asserts (5.9) bitwise for a state built
to equal its reference. CPU, FP64, one process. **Run evidence at the pin: missing evidence.**

### Limits

1. The balance (5.9) is derived for Cartesian $x_1$. On curved grids it relies on the geometric pressure source (see 5.1,
   Limits item 6). `test_hydrostatic` is the cubed-sphere evidence, to $10^{-8}$.
2. An outflow $x_1$ boundary gets no parity ghosts (`src/hydro/hydro_forward.cpp:532-536@73c45a7c`).
3. The positivity fallbacks change the face state silently. They are counted (chapter 4, section 4.3) but not refused.
4. The figure script is not re-checked at the pin.

---

## 5.3 Balance column and the fourth-order reference (`snapy.balance_column`; `dynamics/conversion-order: 4`)

### Summary

A column that satisfies the continuous hydrostatic equation is not at rest under the discrete scheme. Its
departure $p_i - p_{\mathrm{ref},i}$ is not constant, so (5.9) does not apply, and the column starts moving.
`balance_column` projects a column onto the scheme's own discrete balance at fixed temperature. It returns a state
whose departure is constant to a relative tolerance. That state can be used as an initial condition. The same
section holds the face density rows that make the reference fourth order: the cp5 rows of 5.1, and, with
conversion order 4, the four-cell face rows applied after the seam exchange.

::: {.callout-note title="At a glance"}
**Purpose.** A column with $p_i - p_{\mathrm{ref},i}(\rho) = c$, eq. (5.10); fourth-order face density, eq. (5.12).

**Data.** In: a ghost-free column `w` $(n_{\mathrm{var}}, n_3, n_2, n_1)$, `dx1f`, $g>0$. Out: `(w, residual, sweeps)`.

**Placement.** Initialisation, called by the user (Python `snapy.balance_column`); not part of a step.

**Activation.** Defaults `wall_clamp=True`, `rtol=1e-10`, `max_iter=120`, `conversion_order=2`.

**Guarantee / limits.** Residual $<$ `rtol` on return, or it throws. $p/\rho$ is unchanged per cell. At least 5 cells.

**Evidence.** `test_balance_column.release`, `test_balance_column_order4_python`, `test_wb_ref4_order_python` (Tests layer).
:::

**Local symbols:** $\theta_i = p_i/\rho_i$, the fixed cell temperature times the gas constant (the code's `rt`; a
local name, because NOTATION's $\theta$ is colatitude, to be renamed at conversion); $c$ the column's gauge;
$\varepsilon_{\mathrm{bc}}$ the residual.

### Derivation

**The fixed point.** 5.2 showed that a resting column with a constant departure is balanced. Hold the temperature
fixed per cell, $\theta_i = p_i/\rho_i$. Then $p_{\mathrm{ref}}$ is a function of the densities through (5.1)-(5.5), and
the balance condition is

$$
p_i - p_{\mathrm{ref},i}\big(\boldsymbol\rho\big) = c\quad\text{for all } i,\qquad \rho_i = p_i/\theta_i . \tag{5.10}
$$

The gauge $c$ is free, one constant per column. The code takes it at the top cell, $c = p_{n-1} - p_{\mathrm{ref},n-1}$,
so the top pressure does not change. It iterates

$$
p_i^{(k+1)} = p_{\mathrm{ref},i}\big(\boldsymbol\rho^{(k)}\big) + c^{(k)},\qquad \rho_i^{(k+1)} = p_i^{(k+1)}/\theta_i , \tag{5.11}
$$

stopping when the residual
$\varepsilon_{\mathrm{bc}} = \max_i\lvert p_i - p_{\mathrm{ref},i} - c\rvert/(\rho_i g h_i)$ is below `rtol`
(`balance_column`, `src/hydro/balance_column.cpp:105-143@73c45a7c`; the residual at `:126-129`, the update at `:130-131`).
The residual is measured before the update, so a returned column is described by the residual it is returned with, and
an already balanced column returns after zero updates (`:120-121`). The denominator $\rho_igh_i$ is the cell's own
pressure drop, so $\varepsilon_{\mathrm{bc}}$ is the imbalance as a fraction of the gravity force on the cell. Convergence of
(5.11) is not proved here: the code throws if the pressure goes non-positive or `max_iter` is exceeded (`:134-142`).

**Fourth-order face density.** Chapter 4, eq. (4.1), with four cells gives the face value of the cubic whose cell
averages are $\rho_{\mathrm{ref}}$. On a uniform grid in the interior,

$$
\rho_{\mathrm{sf},i-1/2} = \tfrac{1}{12}\big(-\rho_{\mathrm{ref},i-2} + 7\rho_{\mathrm{ref},i-1} + 7\rho_{\mathrm{ref},i} - \rho_{\mathrm{ref},i+1}\big) \tag{5.12}
$$

(`face_density_rows`, `src/recon/x1_rows.cpp:291-338@73c45a7c`, comment at `:310-311`). Substituting $\bar q = 1, x, x^2, x^3$
into (5.12) reproduces the face values (sympy: weights $(-1,7,7,-1)/12$), so it is exact for cubics: one degree higher than the face rule of chapter 4
with 4 cells. With conversion order 4 these rows replace $\rho_{\mathrm{sf}}$ on faces `is..iu+1`, after the seam exchange.
The window is kept inside the owned cells at a clamped wall (`:303-327`). A face value is kept only if it lies between
the two adjacent cell values, and only where neither neighbouring cell is flagged
(`wb_ref4_faces`, `src/recon/x1_reference.cpp:42-54@73c45a7c`). A cell is flagged, and the flag dilated by two cells,
where $\lvert\ln(p^{\mathrm{ref}}_{i-1/2}/p^{\mathrm{ref}}_{i+1/2})\rvert > 0.5$, that is, fewer than two cells per pressure scale
height (`kMaxLogDrop`, `:21-22`; `wb_ref4_flag`, `:31-40`). At key 2 with WENO5/cp5 the face density is the cp5 row of 5.1.
`test_wb_ref4_order_python` measures that configuration.

**Conversion order 4.** With `conversion_order=4`, `balance_column` balances the column the solver sees after its
fourth-order conversion (chapter 4, eq. (4.8)). Each pass converts the padded column, balances the converted primitives
at key 2, and moves the input by the difference, until a pass needs no sweep (`src/hydro/balance_column.cpp:58-104@73c45a7c`).

**Exact and truncated.** (5.10) is the exact discrete balance of 5.2's scheme, within the stopping tolerance. (5.12) is
fourth order at a face on a uniform grid.

### Numerical method

`balance_column` builds the reference operator with both walls physical and `iu` $= n_1-1$, no ghosts. It uses the
stretched-row branch for unequal widths (`src/hydro/balance_column.cpp:111-115@73c45a7c`), and evaluates it once per
sweep (`:125`). $\theta_i$ is taken once from the input (`:109`). Velocities and every other row are left as given.
The Python binding exposes the same defaults (`python/csrc/pyhydro.cpp:46-50@73c45a7c`).

Refusals (each a `TORCH_CHECK`, `src/hydro/balance_column.cpp:22-57@73c45a7c`): conversion order not 2 or 4; $g\le0$;
`w` not 4-D or without a pressure row; `dx1f` shape, dtype or device mismatch; `wall_clamp` false; under
`SNAP_X1_CENTROID_EXACT`, a geometry other than `cartesian`; fewer than 5 cells; a non-positive $p$ or $\rho$. With key 4:
no EOS, or the EOS of a block with $n_2$ or $n_3 \ne 1$ (`:58-66`).

**Figures.** `src/snapy_report/ch05/figures/fig_balcol_passes.py` (data `data/bryan_balance_ic.json`, recorded at an earlier
pin per the outline, "rerun at the pin") and `fig_wbref4_order.py` (data `data/wb_ref4_order.json`, recorded at an earlier
pin). Neither rerun is done in this draft.

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| the projection | `src/hydro/balance_column.cpp:18-144@73c45a7c` | `balance_column` | `wall_clamp`, `rtol`, `max_iter`, `stretched`, `conversion_order` |
| defaults | `src/hydro/balance_column.hpp:18-22@73c45a7c`, `python/csrc/pyhydro.cpp:46-50@73c45a7c` | `balance_column` | |
| key-4 face rows | `src/recon/x1_rows.cpp:291-338@73c45a7c` | `face_density_rows` | `conversion-order: 4` |
| flag and faces | `src/recon/x1_reference.cpp:17-54,136-146,171@73c45a7c` | `wb_ref4_flag`, `wb_ref4_faces`, `X1Reference::evaluate` | |

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_balance_column.release` `a_ghost_free_column_reproduces_a_blocks_own_reference` | the ghost-free operator equals a block's own reference | `torch::equal` (bitwise) | `tests/test_balance_column.cpp:149-185@73c45a7c` |
| ... `without_the_clamp_the_two_references_disagree` | the control | `EXPECT_FALSE(torch::equal)` | `tests/test_balance_column.cpp:194-208@73c45a7c` |
| ... `a_marched_column_comes_out_at_rest` (and `_cuda`) | a marched column with an initial residual $>10^{-4}$ comes out below `rtol` | `rtol` = 1.e-10 | `tests/test_balance_column.cpp:210-236@73c45a7c` |
| ... `the_temperature_and_every_other_channel_stay_put` | $p/\rho$ unchanged; velocities bitwise | relative 1.e-14; `torch::equal` | `tests/test_balance_column.cpp:260-277@73c45a7c` |
| ... `a_balanced_column_is_a_fixed_point` | a second call does 0 sweeps and returns the same column | `EXPECT_EQ` 0; `torch::equal` | `tests/test_balance_column.cpp:279-286@73c45a7c` |
| ... `a_moist_column_with_one_condensable_comes_out_at_rest` | a moist column balances; temperature and species stay put | 1.e-10; 1.e-14; 1.e-12 | `tests/test_balance_column.cpp:446-494@73c45a7c` |
| ... `a_moist_column_balanced_at_order_4_stays_at_rest_in_key_4` | order-4 balance stays at rest under key 4 (explicit, 100 steps, Courant 0.8; implicit scheme 9, 20 steps, Courant 10): $\max\lvert v_1\rvert/c_s$ and $\max\lvert\Delta\rho\rvert/\rho$; order-2 control moves | $\le$ 1.e-12; control $>$ 1.e-8 | `tests/test_balance_column.cpp:500-551@73c45a7c` |
| `test_stretched_x1_ref.release` `keyed_column_balances_to_1e13` | stretched column, 128 cells | residual $<$ 1.e-13 | `tests/test_stretched_x1_ref.cpp:332-339@73c45a7c` |
| `test_balance_column_order4_python` | key-4 balance from Python, 64 cells, `x1rat` 1 and 1.05: a second call takes 0 passes and changes $\le10^{-12}$; residual after one stage $<10^{-12}$; at rest under key 4 | 1.e-12; 1.e-12; $\max\lvert v_1\rvert/c<$ 1.e-13 | `tests/test_balance_column_order4.py:42-59@73c45a7c` |
| `test_wb_ref4_order_python` (and `_cuda`) | neutral polytrope, one RK3 step, WENO5 + LMARS + face gravity work, key 2: observed order of $\lvert N^2_{\mathrm{eff}}\rvert$ at the last doubling (nz 64$\to$128), 1 and 3 e-folds | `ORDER_ON` = 2.75 | `tests/test_wb_ref4_order.py:19-21,40,160-166@73c45a7c`; the docstring's measured 4.01 and 3.59 (`:18`) are **unverified** |

`test_balance_column` has CUDA variants that skip without CUDA. It also runs as `test_balance_column_x1_centroid.release`
(`tests/CMakeLists.txt:172-176@73c45a7c`). `test_wb_ref4_order_cuda_python` is registered (`tests/CMakeLists.txt:621-623@73c45a7c`).
One rank. **Run evidence at the pin: missing evidence.**

### Limits

1. Convergence of (5.11) is not proved. The function throws on divergence or after `max_iter` sweeps.
2. `balance_column` uses the stretched-row operator for unequal widths. A run on an unequal grid that is not declared
   stretched uses the legacy branch (5.5) instead. A column balanced by the function is then not at rest in that run,
   because the two are "different operators" (`src/recon/x1_reference.hpp:37-40@73c45a7c`; the run-time constructor
   defaults to `kX1RefNonUniform`, `:43-45`). No test at the pin covers that combination. Declare such grids stretched.
3. Under `SNAP_X1_CENTROID_EXACT` only a Cartesian column is balanced (`src/hydro/balance_column.cpp:42-47@73c45a7c`).
4. The fourth-order face rows are skipped on cells with fewer than two cells per pressure scale height, and on blocks
   whose clamped walls leave fewer than 4 owned cells (`src/recon/x1_rows.cpp:305-307@73c45a7c`).
5. The outline's two data figures need reruns at the pin. They are not done.

---

### Checker notes (remove at conversion)

- Outline anchors, all resolving at 73c45a7c: `src/hydro/hydro.cpp:526@73c45a7c` (`HydroImpl::_hydro_ref_x1`),
  `src/recon/x1_reference.hpp:41@73c45a7c` (`HydroReferenceX1Operator`), `src/recon/well_balanced_x1.cpp:16@73c45a7c`
  (`WellBalancedX1::forward`), `src/hydro/balance_column.cpp:18@73c45a7c` (`balance_column`).
- Symbolic checks (sympy 1.14.0, Python 3.11.13, `python3 -I`, outside the report tree): (5.3) and (5.4) weights equal the
  code's constants and sum to 1; the (5.3) error on $e^{-x/H}$ is $\tfrac{191}{60480}(h/H)^6$; (5.2) and (5.5) are the
  isothermal half-cell step and cell mean. All passed. (5.12) is chapter 4's (4.1) with four cells.
