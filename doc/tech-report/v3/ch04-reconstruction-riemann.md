# 4 Reconstruction, Riemann solvers and the hydro sweep

<!-- PROVENANCE
pin: snapy@73c45a7c8c991209924e1137992352c335575062 (UCzhangxi/snapy v3/final, head of chengcli/snapy#377)
outline: doc/tech-report/STRUCTURE.md at tech-report-v3-design aca6a84bc06f29ce18a5d8a314c9e435b20a829d, chapter 4 (4.1-4.4)
sources: src/recon/*, src/riemann/*, src/hydro/sweep.*, src/hydro/hydro_forward.cpp, src/options/hydro.yaml and the tests named in each Tests layer, all read at the pin
form: pre-conversion draft (STYLE.md section 0): hand-numbered equations, `path:lines@sha` code locations; section 10 (Quarto markup) does not bind it yet
status: draft
author: Xinyue Wang
reviewer: (checker: Tianhao)
-->

> **Draft conventions.** Every code location in this file is written `path:lines@73c45a7c`. The short form names
> the pin in the PROVENANCE block above (snapy `v3/final`, full sha there). It is converted to the
> `{{< code >}}` shortcode when this draft becomes `book/chapters/04-.../_<scheme>.qmd` files. Equations are
> numbered (4.n) by hand for the same reason. Every tolerance is read from the test source at the pin. A run
> number with no recorded command and versions is marked **missing evidence**. No number in this draft
> comes from a run.

This chapter follows one face flux from the cell averages to the flux register. A hydro stage first turns the
stored cell averages into primitive values. It then reconstructs two values at every face, one from each side
(section 4.1). It hands each pair to an approximate Riemann solver, which returns one flux per face
(section 4.2). The sweep around these two steps also floors the face states, counts every face a floor or a
solver fallback changed, and posts the flux to the block's flux register (section 4.3). On the $x_2$ and $x_3$
faces the sweep finally adds a second-order correction, because a face-average flux is not the flux of
face-averaged states (section 4.4).

The well-balanced $x_1$ path, which reconstructs perturbations about a hydrostatic reference, is chapter 5's.
This chapter describes the plain path that every direction takes without gravity, and the $x_2$/$x_3$ path
that every run takes. How the register books a posted flux, and how its counters are rolled back, is
chapter 10's.

---

## 4.1 Reconstruction: PLM, WENO5 and the conversion order (`dynamics/reconstruct/{vertical,horizontal}/type`, `dynamics/conversion-order`)

### Summary

A finite-volume cell stores averages. A flux needs values at a face. Reconstruction builds the two one-sided
face values of every primitive variable from the cell averages around the face. snapy reconstructs the
primitives $\mathbf W$ (density, velocity, pressure, mass fractions), not the conserved $\mathbf U$. The
primitives are computed per cell from the conserved averages, so by default they are second-order accurate
as averages. `dynamics/conversion-order: 4` makes the $x_1$ primitives fourth-order averages. It does so by
converting the conserved averages to point values, calling the equation of state there, and converting back.

::: {.callout-note title="At a glance"}
**Purpose.** Face states $q^{\mathrm L}_f, q^{\mathrm R}_f$ of every primitive row, eqs. (4.1)-(4.6); optional
fourth-order $x_1$ primitive averages, eq. (4.8).

**Data.** In: primitive cell averages `hydro_w`, $(n_{\mathrm{var}}, n_3, n_2, n_1)$, ghosts filled. Out:
`(2, n_var, n_3, n_2, n_1)`, side 0 = $q^{\mathrm L}$, side 1 = $q^{\mathrm R}$.

**Placement.** Every RK stage, inside `HydroImpl::forward`: $x_1$ in step (2), $x_2$ and $x_3$ in steps (3.A),
(3.B). The conversion runs in step (1), right after the equation of state.

**Activation.** `type` per direction group (`vertical` = $x_1$, `horizontal` = $x_2$, $x_3$): `dc` (default),
`plm`, `cp3`, `weno3`, `cp5`, `weno5`; `ppm` is accepted and throws at the first call. `scale` (default
false), `shock` (default false). `conversion-order` 2 (default) or 4.

**Guarantee / limits.** cp5/WENO5: fifth-order face values on a uniform grid where the data are smooth;
PLM: second order, TVD slope; conversion order 4: fourth-order $x_1$ primitive averages. Floors keep
$\rho^{\mathrm{L,R}} \ge$ the density floor and $p^{\mathrm{L,R}} \ge$ the pressure floor when the EOS limiter
is on.

**Evidence.** `test_plm.release`, `test_weno.release`, `test_conversion_order_python` (Tests layer).
:::

**Local symbols** (not in NOTATION.md; to be added there; see Limits, item 6): $q^{[r]}$ the face value of WENO
candidate stencil $r$; $\beta_r$ WENO smoothness
indicator of candidate $r$; $\mathsf d_r$ linear weight; $\mathsf a_r$ nonlinear weight; $\epsilon_{\mathrm w}$ the WENO
weight epsilon; $\mathsf L$ the $x_1$ conversion operator of eq. (4.7), with its weights $k^\pm_i$ and $\mathsf c_i$;
$\Delta q_{i\pm1/2}$ the one-sided differences of PLM; $\delta q_i$ the PLM slope.

The default is first order (`dc`: each side of a face takes the value of its own cell, `InterpImpl::left`/`right`,
`src/recon/reconstruct.cpp:151-167@73c45a7c`), and the default conversion is
second order. The reconstructed group pairs are fixed. With
`weno5`, density and the mass fractions take WENO5, and velocity and pressure take its linear cp5 partner.
With `weno3`, the pair is cp3. Every other scheme is used for all rows. `shock: true` uses one scheme for
every row.[^ch04-aliases]

[^ch04-aliases]: Aliases a reader meets in the code: "lane C" (reconstruction/Riemann design lane), "C5"
(FaceFloor and solver events), "C9" (typed constants), "C10" (`shock` default and floors on the shock path),
"route A" (per-face stretched $x_1$ rows), key 4 (`conversion-order: 4`), chengcli/snapy#339 (conversion
order 4). [source: code comments at `src/recon/reconstruct.hpp:127-128@73c45a7c`,
`src/recon/reconstruct.cpp:109-116@73c45a7c`, `tests/test_conversion_order.py:2@73c45a7c`]

### Derivation

**Face value from cell averages.** Let $\bar q_{i+l}$ be the averages of a smooth $q(x_1)$ over cells
$i+l$ of width $h$, $l\in\mathcal N$. Let $Q(x)=\int^x q$ be a primitive. Then $h\bar q_{i+l} = Q(x_{i+l+1/2}) -
Q(x_{i+l-1/2})$, so the averages fix $Q$ at the $|\mathcal N|+1$ faces of the stencil up to a constant. Interpolate
$Q$ by the polynomial $P$ of degree $|\mathcal N|$ through those faces. Then $P'$ is a polynomial of degree
$|\mathcal N|-1$ whose cell averages are the $\bar q_{i+l}$, and

$$
q(x_{i-1/2}) \approx P'(x_{i-1/2}) = \sum_{l\in\mathcal N} c_l\,\bar q_{i+l}, \qquad
c_l = h_{i+l}\sum_{j>l} \ell_j'(x_{i-1/2}), \tag{4.1}
$$

where $\ell_j$ is the Lagrange basis on the faces and $h_{i+l}$ is the cell width. The error is
$O(h^{|\mathcal N|})$. The second form, valid on any face spacing, is exactly what `face_value_row` evaluates
(`src/recon/x1_rows.cpp:16-40@73c45a7c`).

On a uniform grid, (4.1) gives the coefficients in the tables of `InterpImpl::reset`
(`src/recon/reconstruct.cpp:49-93@73c45a7c`). For the left face $x_{i-1/2}$ of cell $i$:

$$
\begin{aligned}
\text{cp3:}\quad q^{\mathrm R}_{i-1/2} &= \tfrac13\bar q_{i-1} + \tfrac56\bar q_i - \tfrac16\bar q_{i+1}, \\
\text{cp5:}\quad q^{\mathrm R}_{i-1/2} &= -\tfrac1{20}\bar q_{i-2} + \tfrac9{20}\bar q_{i-1} + \tfrac{47}{60}\bar q_i
  - \tfrac{13}{60}\bar q_{i+1} + \tfrac1{30}\bar q_{i+2}.
\end{aligned} \tag{4.2}
$$

The right-face table is the left one reversed (`cm.flip`, `src/recon/reconstruct.cpp:54,68@73c45a7c`). Both
rows of (4.2) were evaluated symbolically from (4.1) with sympy 1.14.0 (Python 3.11.13) and equal the code's
tables exactly. The script is not yet a committed check (Limits, item 5).

**WENO5.** Apply (4.1) to the three three-cell stencils $\{i-2,i-1,i\}$, $\{i-1,i,i+1\}$, $\{i,i+1,i+2\}$. This
gives the three candidate values at $x_{i-1/2}$:

$$
q^{[0]} = -\tfrac16\bar q_{i-2} + \tfrac56\bar q_{i-1} + \tfrac13\bar q_i,\quad
q^{[1]} = \tfrac13\bar q_{i-1} + \tfrac56\bar q_i - \tfrac16\bar q_{i+1},\quad
q^{[2]} = \tfrac{11}6\bar q_i - \tfrac76\bar q_{i+1} + \tfrac13\bar q_{i+2}. \tag{4.3}
$$

These are rows 0-2 of the WENO5 table (`src/recon/reconstruct.cpp:74-76@73c45a7c`). The linear weights
$\mathsf d = (0.3, 0.6, 0.1)$ (`:84`) combine them into the cp5 value of (4.2): $0.3q^{[0]}+0.6q^{[1]}+0.1q^{[2]}$ has the
coefficients $(-\tfrac1{20},\tfrac9{20},\tfrac{47}{60},-\tfrac{13}{60},\tfrac1{30})$. This was checked
symbolically in the same run. The smoothness indicators are the Jiang-Shu forms, from rows 3-8 of the table
(`:77-82`):

$$
\begin{aligned}
\beta_0 &= \tfrac{13}{12}(\bar q_{i-2}-2\bar q_{i-1}+\bar q_i)^2 + \tfrac14(\bar q_{i-2}-4\bar q_{i-1}+3\bar q_i)^2,\\
\beta_1 &= \tfrac{13}{12}(\bar q_{i-1}-2\bar q_i+\bar q_{i+1})^2 + \tfrac14(-\bar q_{i-1}+\bar q_{i+1})^2,\\
\beta_2 &= \tfrac{13}{12}(\bar q_i-2\bar q_{i+1}+\bar q_{i+2})^2 + \tfrac14(3\bar q_i-4\bar q_{i+1}+\bar q_{i+2})^2,
\end{aligned} \tag{4.4}
$$

and the face value is

$$
q^{\mathrm R}_{i-1/2} = \frac{\sum_r \mathsf a_r q^{[r]}}{\sum_r\mathsf a_r},\qquad
\mathsf a_r = \frac{\mathsf d_r}{(\beta_r+\epsilon_{\mathrm w})^2},\qquad \epsilon_{\mathrm w} = 10^{-6}. \tag{4.5}
$$

This is `weno5_point` (`src/recon/recon_point.h:114-147@73c45a7c`), with $\epsilon_{\mathrm w}$ from `weno_eps`
(`:64-67`). With `scale: true`, `weno_scale` (`:71-83`) first divides the five values by their mean
magnitude $\frac15\sum|\bar q|$, and multiplies the result back. The weights then do not depend on the units of
$q$. A stencil whose values are all zero returns 0. Where the $\beta_r$ are equal (smooth data), $\mathsf a_r/\sum\alpha
\to \mathsf d_r$ and (4.5) reduces to cp5. WENO3 is the same construction on the two two-cell candidates
$\tfrac12(\bar q_{i-1}+\bar q_i)$ and $\tfrac32\bar q_i-\tfrac12\bar q_{i+1}$ with $\mathsf d=(\tfrac23,\tfrac13)$ and
$\beta_r$ the squared one-sided differences (`src/recon/reconstruct.cpp:56-63@73c45a7c`,
`src/recon/recon_point.h:87-107@73c45a7c`).

**PLM.** With $\Delta q_{i-1/2} = \bar q_i-\bar q_{i-1}$ and $\Delta q_{i+1/2} = \bar q_{i+1}-\bar q_i$, the slope is
the harmonic mean when the two differences have the same sign, and zero otherwise:

$$
\delta q_i = \begin{cases}\dfrac{2\,\Delta q_{i-1/2}\Delta q_{i+1/2}}{\Delta q_{i-1/2}+\Delta q_{i+1/2}} &
\Delta q_{i-1/2}\Delta q_{i+1/2}>0\\[4pt] 0 & \text{otherwise,}\end{cases}\qquad
q^{\mathrm R}_{i-1/2} = \bar q_i - \tfrac12\delta q_i,\quad q^{\mathrm L}_{i+1/2} = \bar q_i + \tfrac12\delta q_i. \tag{4.6}
$$

This is `InterpImpl::forward` (`src/recon/reconstruct.cpp:132-145@73c45a7c`). The test on the product, not on a
denominator with an added epsilon, keeps a round-off-sized stencil finite. The code comment records that an
epsilon gives $0/0$ near $10^{-38}$ (`:138`). The PLM face value is second order where $q$ is smooth and
monotone, and first order at an extremum, where the slope is set to zero.

**Fourth-order $x_1$ conversion.** The equation of state maps a point value $\mathbf U(x)$ to $\mathbf W(x)$. Applying it
to an average gives $\mathbf W(\bar{\mathbf U}_i) = \bar{\mathbf W}_i + O(h^2)$, because $\mathbf W$ is nonlinear. Let
$q$ be the quadratic whose averages over cells $i-1, i, i+1$ are $\bar a_{i-1}, \bar a_i, \bar a_{i+1}$. Define

$$
\begin{aligned}
(\mathsf L\bar a)_i &= \frac{h_i^2}{24}\,q''(x_{1,i})
 = k^+_i(\bar a_{i+1}-\bar a_i) - k^-_i(\bar a_i-\bar a_{i-1}),\\
k^+_i &= \frac{\mathsf c_i}{x_{1,i+3/2}-x_{1,i-1/2}},\qquad
k^-_i = \frac{\mathsf c_i}{x_{1,i+1/2}-x_{1,i-3/2}},\qquad
\mathsf c_i = \frac{h_i^2}{4\,(x_{1,i+3/2}-x_{1,i-3/2})}
\end{aligned} \tag{4.7}
$$

(`conv4_rows`, `src/recon/x1_rows.cpp:250-260@73c45a7c`: the four faces `f(0..3)` of cell $i$ are
$x_{1,i-3/2}..x_{1,i+3/2}$, `c` is $\mathsf c_i$, and `{kp, km}` are $k^+_i, k^-_i$). On a uniform grid $k^\pm = \tfrac1{24}$ and
$(\mathsf L\bar a)_i = \tfrac1{24}(\bar a_{i+1}-2\bar a_i+\bar a_{i-1})$. The non-uniform identity
$k^+(\bar a_{i+1}-\bar a_i)-k^-(\bar a_i-\bar a_{i-1}) = \tfrac{h_i^2}{24}q''$ was checked symbolically on arbitrary
faces. The point value at the centroid and the average are then related by

$$
q(x_{1,i}) = \bar q_i - \frac{h_i^2}{24}q''(x_{1,i}) + O(h^4)
 = \bar q_i - (\mathsf L\bar q)_i + O(h^4).
$$

(For $q=x^4$ on a uniform cell the remainder is $h^4/80$; checked symbolically.) Conversion order 4 therefore
computes

$$
\bar{\mathbf W}_i = \mathbf W\big(\bar{\mathbf U}_i - (\mathsf L\bar{\mathbf U})_i\big) + \big(\mathsf L\,\mathbf W(\bar{\mathbf U})\big)_i + O(h^4).
\tag{4.8}
$$

The first term is the primitive point value. The second adds back $\tfrac{h^2}{24}w''$. That term is formed from the
second-order primitives $\mathbf W(\bar{\mathbf U})$, which is enough because it carries a factor $h^2$
(`Conversion::forward`, `src/recon/conversion.cpp:63-102@73c45a7c`; the step is labelled in
`src/hydro/hydro_forward.cpp:366-367@73c45a7c`). The conversion acts on the total averages, background included
(`:376-380`). The derivation assumes plain (uniform-measure) cell averages. On spherical-polar and
gnomonic-equiangle grids the stored $x_1$ averages are $r^2\,\mathrm dr$ means, while `Conversion` is built from the
plain faces `x1f` (`src/hydro/hydro_forward.cpp:371-375@73c45a7c`, `src/recon/conversion.cpp:24@73c45a7c`). The
derivation above does not cover that case; the order there rests on the `radial` arm of
`test_conversion_order_python` (fitted order $\ge 3.8$ on both grids, `tests/test_conversion_order.py:189-207@73c45a7c`).

**What is exact and what is truncated.** (4.1)-(4.3) reproduce polynomials of degree $\le|\mathcal N|-1$
exactly. cp5 and WENO5 in the linear limit are fifth order at a face on a uniform grid. Conversion order 4 is
fourth order in the $x_1$ primitive averages. $x_2$ and $x_3$ stay at second order
(`src/options/hydro.yaml:26-32@73c45a7c`). The flux built from these face values is a point flux at the face.
Section 4.4 treats the gap between that and a face-average flux. The executable checks for these
derivations are listed in Limits, item 5.

### Numerical method

**Where each quantity lives.** Input: cell averages of the primitives in `hydro_w`, with $n_g$ ghost cells per
side filled by the exchange and the boundary conditions. Output: a tensor `(2, n_var, ...)` whose side 0
(`ILT`) is $q^{\mathrm L}$ and side 1 (`IRT`) is $q^{\mathrm R}$. Face index $i$ stores face $x_{1,i-1/2}$
(`src/snap.h:44-45@73c45a7c` for `ILT = 0`, `IRT = 1`).

**Stencil and index ranges.** `ReconstructImpl::forward` sets $n_g^{\mathrm{rec}} = \lfloor s/2\rfloor + 1$ from the
stencil width $s$ of the scheme (`src/recon/reconstruct.cpp:256@73c45a7c`). $s$ is 1 for dc, 3 for PLM, cp3 and
WENO3, and 5 for cp5 and WENO5 (`SchemeTraits`, `src/recon/recon_point.h:25-39@73c45a7c`). The computed cells are
`il` $= n_g^{\mathrm{rec}}$ to `iu` $= n - n_g^{\mathrm{rec}}$ (`src/recon/reconstruct.cpp:257-258@73c45a7c`). `_apply_inplace`
(`src/recon/reconstruct.cpp:217-229@73c45a7c`) writes the left-face values of cells `il-1..iu` into side
`IRT` and the right-face values of the same cells into side `ILT`, shifted by one face. It then copies the end
faces into the unused faces on both sides ("dummy regions", `:225-228`). The stencil kernel steps one cell per
output along the swept dimension. Output $m$ reads inputs $m..m+s-1$ (`x1_uniform_iterator`,
`src/recon/x1_rows.hpp:50-61@73c45a7c`).

**Figure.** `src/snapy_report/ch04/figures/fig_recon_stencil.py` draws where `_apply_inplace` writes and which
rows take which interpolant. Its docstring cites an earlier snapy sha. Re-checking it against the pin is open
(Limits, item 4). A data figure of the measured face order (outline: **new** `recon_order_run.py` +
`fig_recon_order.py`, a 1-D smooth profile through `ReconstructImpl` at the pin) is not yet written. The
existing `fig_recon_order.py` reads the output of `recon_check.py`, a port of a scalar reference and not a
run of `ReconstructImpl`.

**Groups and floors.** Without `shock`, density goes through `pinterp1`, velocity and pressure through `pinterp2`,
and the mass fractions through `pinterp1` (`src/recon/reconstruct.cpp:313-330@73c45a7c`; the pairing is
`velocity_pressure_scheme`, `:28-32`). After each group, and only when the parent Hydro's EOS has
`limiter: true` and the caller did not pass `floor = false`, the face density and pressure are clamped from
below to the EOS floors (`:287-301`). The mass fractions are clamped at 0 when the limiter is on (`:280-285`).
`clamp_min` keeps $-0$, so a face that is already non-negative is unchanged bit for bit (`:278-279`). Each
floor that runs leaves a mask of the faces it changed in `floored`; section 4.3 posts those masks. With `shock`
the whole state goes through `pinterp1`, then the same three floors run (`:303-311`). Without a parent Hydro
(a bare module) the result is clamped at 0 and no mask is left (`:262-266`).

**Stretched and radial $x_1$.** On a stretched $x_1$ (`cells/x1rat` or `geometry/x1f`), cp5 and WENO5 use one row
of (4.1) per face, built on the host in double from the actual faces (`weno5_x1_coeffs`, `cp5_x1_coeffs`,
`src/recon/x1_rows.cpp:103-176@73c45a7c`). Any other $x_1$ scheme is refused with the message at
`src/recon/reconstruct.cpp:109-116@73c45a7c`. On spherical-polar and gnomonic-equiangle grids with `weno5` or
`cp5`, the rows are built on the volume coordinate $v = x_1^3/3$ instead of $x_1$, but only when `nx1` $> 1$ and, on
spherical-polar, only when `SNAP_X1_CENTROID_EXACT` is off; with the switch on, the reconstruction reads plain
means and keeps the plain $x_1$ rows (`src/coord/coordinate.cpp:465-471@73c45a7c`; section 4.4). At a physical wall the ghost
widths are mirrored in $v$ (`CoordinateImpl::x1_row_faces`, `src/coord/coordinate.cpp:462-482@73c45a7c`). The
rows are installed by `HydroImpl::reset` (`src/hydro/hydro.cpp:50-56@73c45a7c`).

**Conversion order 4: ghosts, walls and fallback.** The operator (4.7) is centred everywhere. It reads one
ghost cell per side, which the exchange or the boundary condition filled (`Conversion::lap`,
`src/recon/conversion.cpp:51-61@73c45a7c`). At a reflecting $x_1$ wall, the ghosts the operator reads are first
refilled from the four owned cells next to the wall by a polynomial fit (`wall4_rows`,
`src/recon/x1_rows.cpp:262-289@73c45a7c`; `fill_walls`, `src/recon/conversion.cpp:33-49@73c45a7c`). The wall
momentum gets odd parity. The guarded rows are `IDN` and `IPR` (for $\bar{\mathbf U}$ the latter is the total
energy): a ghost of those rows whose value, reference included where there is one, is not positive keeps its
mirror value and counts as a trip (`src/recon/conversion.cpp:29-30,44-46,77@73c45a7c`). Fewer than four $x_1$ cells keep the mirrored ghosts, with a one-time warning
(`src/hydro/hydro_forward.cpp:341-353@73c45a7c`). Cell by cell, the fourth-order value is kept only if
both checks of `conv4_admissible` pass (`src/recon/conversion.cpp:10-18@73c45a7c`): on
$\bar{\mathbf U}-\mathsf L\bar{\mathbf U}$, finite, density above the density floor and mass fractions
non-negative with the limiter on, with no pressure test (`prim = false`, `:82`); and on the converted $\mathbf W$,
the same plus pressure above the pressure floor (`prim = true`, `:98`). The floors are zero with the limiter off. With the limiter on, a cell whose point state
the EOS limiter changed is also rejected. A rejected cell keeps $\mathbf W(\bar{\mathbf U})$, and its count is
added to the EfStatus slot `kConvTrips` (`:98-101`). With the limiter on, the limiter marks and vapor failures
raised by the point-value EOS call are restored to their values before the call (`:87-97`); with the limiter
off they are left as the call set them. After the conversion, every primitive ghost is refilled from the converted
owned cells: block exchange, $x_1$ seam rows, then the boundary functions
(`src/hydro/hydro_forward.cpp:380-401@73c45a7c`).

**Order and stage.** All of this runs in every RK stage, explicitly. Interior order is given in Derivation. At
walls, conversion order 4 uses the one-sided fit above. The `wall` arm of `test_conversion_order_python` asserts
that the error of the mass tendency in the three cells next to each wall falls by at least $2^{2.5}$ from
$n_1 = 64$ to $128$: the observed order between two resolutions, not a fitted one
(`tests/test_conversion_order.py:252-266@73c45a7c`). The wall order itself is not measured in this draft. Across an
$x_1$ block seam the stencil reads exchanged ghosts, so a seam is not a boundary for the reconstruction.

**Precision and devices.** The stencil schemes run one CPU loop and one CUDA line kernel. Both call the same
`recon_point` (`src/recon/recon_point.h:152-165@73c45a7c`). MPS runs the CPU loop on the host
(`src/recon/recon_dispatch.cpp:114-116@73c45a7c`). The coefficient tables are built in FP64 and cast to the state's
dtype through a cache (`TensorCastCache`, `src/recon/reconstruct.cpp:184-186@73c45a7c`). WENO5's linear
weights are stored in the table, so an FP32 run reads them in FP32 (`:72-73`). Chapter 12 holds the FP32 rows.

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| reconstruct options and YAML keys | `src/recon/reconstruct.hpp:29-43,103-132@73c45a7c`, `src/recon/reconstruct.cpp:197-213@73c45a7c` | `InterpOptionsImpl`, `ReconstructOptionsImpl::from_node` | `dynamics/reconstruct/{vertical,horizontal}/{type,scale,shock}` |
| the two modules, $x_1$ and $x_2$/$x_3$ | `src/hydro/hydro.cpp:50-63@73c45a7c`, `src/hydro/hydro_options.cpp:28-31@73c45a7c` | `HydroImpl::reset` (`precon1`, `precon23`) | `vertical`, `horizontal` |
| scheme name to enum | `src/recon/reconstruct.cpp:17-32@73c45a7c` | `parse_scheme`, `velocity_pressure_scheme` | `type` |
| coefficient tables | `src/recon/reconstruct.cpp:49-93@73c45a7c` | `InterpImpl::reset` | |
| PLM, dc and dispatch | `src/recon/reconstruct.cpp:119-187@73c45a7c` | `InterpImpl::forward`, `left`, `right`, `apply` | |
| one-face kernels | `src/recon/recon_point.h:44-165@73c45a7c` | `poly_point`, `weno3_point`, `weno5_point`, `recon_point` | `scale` |
| CPU loop, MPS on host | `src/recon/recon_dispatch.cpp:32-116@73c45a7c` | `recon_line_cpu`, `call_recon_cpu` | |
| groups, floors, masks | `src/recon/reconstruct.cpp:217-333@73c45a7c` | `_apply_inplace`, `ReconstructImpl::forward`, `floored` | `shock`, EOS `limiter` |
| stretched/radial $x_1$ rows | `src/recon/x1_rows.cpp:16-176@73c45a7c`, `src/coord/coordinate.cpp:462-482@73c45a7c` | `face_value_row`, `ptype_row`, `window_weno5`, `window_cp5`, `x1_row_faces` | `cells/x1rat`, `geometry/x1f` |
| conversion order 4 | `src/recon/conversion.cpp:10-102@73c45a7c`, `src/recon/x1_rows.cpp:250-289@73c45a7c`, `src/hydro/hydro_forward.cpp:341-401@73c45a7c` | `Conversion::forward`, `lap`, `fill_walls`, `conv4_admissible`, `conv4_rows`, `wall4_rows` | `dynamics/conversion-order` |
| key range check | `src/hydro/hydro.cpp:30-33@73c45a7c` | `HydroImpl::reset` | |

Walk-through, in step order. (i) `HydroOptions` reads `dynamics/reconstruct/vertical` and `.../horizontal`.
Each section accepts only `shock`, `type` and `scale`, and a missing key keeps its default. Any other key is
refused by `check_keys` (`src/recon/reconstruct.cpp:199,208@73c45a7c`). (ii) `HydroImpl::reset` builds
`precon1` and `precon23` and installs the $x_1$ rows. It refuses a `conversion-order` other than 2 or 4 with
"[Hydro] dynamics/conversion-order must be 2 or 4" (`src/hydro/hydro.cpp:30-33@73c45a7c`). With key 4 it binds
the `conversion_trips` buffer to the EfStatus slot (`:271-274`). (iii) Step (1) of `HydroImpl::forward` calls
the EOS and, with key 4, the conversion. (iv) Steps (2), (3.A) and (3.B) call `ReconstructImpl::forward`
through the sweep (section 4.3).

Errors and warnings a user can hit: "Interp: unknown type ..." (`src/recon/reconstruct.cpp:25@73c45a7c`);
"PPM interpolation is not implemented yet." (`:125-127`); the stretched-$x_1$ refusal for schemes other than cp5
and WENO5 (`:109-116`); "il > iu" when a block has fewer cells than the stencil needs (`:260`);
"dynamics/conversion-order 4 needs nc1 >= 4 cells ..." (`src/recon/x1_rows.cpp:252-255@73c45a7c`);
"[x1_coeffs] face vector must be strictly increasing" and the other face checks
(`src/recon/x1_rows.cpp:133-149@73c45a7c`); the one-time warning "fewer than 4 x1 cells: wall ghosts stay
mirrored" (`src/hydro/hydro_forward.cpp:350-351@73c45a7c`).

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_plm.release` `DeviceTest.interp_plm` | the scalar reference on (1, 2, 3) gives the left face 1.5 | `EXPECT_NEAR` 1.E-10 | `tests/test_plm.cpp:24-31@73c45a7c`; exact in binary arithmetic, so the bound is slack |
| `test_plm.release` `interp_plm_torch1..3` | the tensor PLM equals the scalar reference `interp_plm` (`tests/interp_simple.hpp:57-63@73c45a7c`) on random data, 1-3 variables | `EXPECT_NEAR` 2.E-6 (FP32); 1.E-6 (FP64, torch1), 2.E-6 (FP64, torch2/3) | `tests/test_plm.cpp:33-173@73c45a7c`; the source gives no reason |
| `test_plm.release` `interp_plm_round_off_slopes_stay_finite` | round-off-sized stencils give finite face values, equal to the reference in FP64 | `ASSERT_TRUE(isfinite)`; `EXPECT_EQ` (bitwise, FP64) | `tests/test_plm.cpp:175-198@73c45a7c` |
| `test_plm.release` `reconstruct_preserves_a_constant_field` | dc, plm and weno5 reproduce a constant 2 on all three dims, output shape `(2, ...)` | `torch::allclose` defaults rtol=1e-05, atol=1e-08 | `tests/test_plm.cpp:200-216@73c45a7c`; libtorch defaults |
| `test_weno.release` `interp_cp5*`, `interp_weno3*`, `interp_weno5*` | the module equals the scalar references of `tests/interp_simple.hpp` | `EXPECT_NEAR` 1.E-6; 2.E-6 in `interp_weno5m_torch2` (FP32 branch) | `tests/test_weno.cpp:92-636@73c45a7c` |
| `test_weno.release` `interp_weno3_smooth_limit` | WENO3 on $1+\{1,2,4\}\cdot10^{-7}$ agrees with cp3 to 1.E-6. Every convex combination of the two candidates is within $\tfrac13\cdot10^{-7}$ of cp3 there, so the test does not show the weights reaching their linear limit | `EXPECT_NEAR` 1.E-6 | `tests/test_weno.cpp:226-245@73c45a7c` |
| `test_weno.release` `interp_weno3a`, `interp_weno5b` | scalar references on linear data: 1.5 and 2.5000000000000004 | `EXPECT_NEAR` 1.E-10 | `tests/test_weno.cpp:217-256@73c45a7c` |
| `test_weno.release` cast-cache tests | the FP64 tables follow buffer edits, storage changes and dtype | `torch::equal` (bitwise) | `tests/test_weno.cpp:21-90@73c45a7c` |
| `test_conversion_order_python` arm `order` | acoustic pulse, $n_1$ = 64, 128, 256 against 1024, uniform and stretched $x_1$: fitted order $\ge 3.8$; tracer ratio within $10^{-13}$ | 3.8; `TOL` = 1.e-13 | `tests/test_conversion_order.py:43,171-186@73c45a7c` |
| `test_conversion_order_python` arm `radial` | the conversion alone on spherical-polar and gnomonic blocks: order $\ge 3.8$; a rewritten `x1f` gives the converted $\mathbf W$ of a fresh block | 3.8; `torch.equal` (bitwise) | `tests/test_conversion_order.py:189-225@73c45a7c` |
| `test_conversion_order_python` arms `mass`, `floor`, `slab`, `wall`, `gravity`, `theta` | mass: $|\Delta M|/M \le 10^{-13}$ (wall, seam jump), and the outflow run's $\Delta M/M$ within $10^{-10}$ (relative) of the pinned $-1.687639303786939\times10^{-2}$; floor: bitwise insensitivity; slab: $|\Delta M|/M \le 10^{-13}$, seam ghosts bitwise, $\sum y^2$ within $10^{-10}$ of its pinned value; wall: two-resolution order $\ge 2.5$; gravity: fitted order $\ge 3.8$; theta: $\theta'$ stays within its initial range widened by 2 K on each side | `TOL` = 1.e-13; 1.e-10 (twice); bitwise; 2.5; 3.8; 2 K | `tests/test_conversion_order.py:2-31,252-266,280-306,313,330-337@73c45a7c` |

The `DeviceTest` suites run on CPU FP32 and FP64 and on CUDA FP32 and FP64. MPS is commented out, and CUDA
skips when unavailable (`tests/device_testing.hpp:44-50@73c45a7c`). `test_conversion_order_cuda_python` runs the
same script with `--device cuda` (`tests/CMakeLists.txt:606-607@73c45a7c`). All of these run on one rank. The
`slab` and `mass` arms use two $x_1$ blocks in one process. **Run evidence at the pin: missing evidence.** This
draft records no run of these tests (command, device and versions) at the pin. The fitted orders the
`order` arm prints are therefore not quoted.

### Limits

1. `ppm` is accepted by `parse_scheme` and throws at the first `forward` (`src/recon/reconstruct.cpp:125-127@73c45a7c`).
2. Stretched $x_1$ is supported only with cp5 and WENO5, and only in $x_1$ (`src/recon/x1_rows.hpp:78-89@73c45a7c`).
   $x_2$ and $x_3$ are uniform-table only.
3. Conversion order 4 is $x_1$ only. It refuses CUDA-graph capture when an outflow boundary is set, because
   the hydro step refills its primitive ghosts with it (`StageExecutor::config_reasons`,
   `src/mesh/stage_executor.cpp:675-681@73c45a7c`). At walls the test asserts only a two-resolution order (64 to 128) of at least 2.5, not a fitted
   order (Tests), and
   with fewer than four $x_1$ cells it keeps mirrored ghosts.
4. The figure scripts cited above were drawn against earlier snapy shas. Their agreement with the pin is not
   checked in this draft. The outline's **new** order figure (`recon_order_run.py`) is not written.
5. The derivation checks (4.1)-(4.8) were run once with sympy 1.14.0 / Python 3.11.13 outside the report tree.
   They are not yet a committed `src/snapy_report/ch04/checks/*_check.py` with a `.out` (STYLE 7).
6. Local symbols ($q^{[r]}, \beta_r, \mathsf d_r, \mathsf a_r, \epsilon_{\mathrm w}, \mathsf L, k^\pm_i, \mathsf c_i, \Delta q_{i\pm1/2}, \delta q_i$) need an
   ISSUES.md item and a NOTATION.md decision. $\mathsf L$ was chosen because $\mathcal L$ is the RK right-hand side;
   the sans-serif $\mathsf a_r, \mathsf c_i, \mathsf d_r$ avoid NOTATION's $\alpha$ (never bare), $\kappa$ (thermal
   diffusivity) and $d$ (always carries its quantity).
7. No test at the pin resolves WENO's smooth limit: `interp_weno3_smooth_limit` is satisfied by any convex
   combination of the candidates (Tests).
8. Untested combinations: `scale: true` with FP32 WENO5 on CUDA has no dedicated test found at the pin; the
   switch-matrix cross-reference (chapter 15) is not filled in this draft.

---

## 4.2 Riemann solvers (`dynamics/riemann-solver/type`)

### Summary

A Riemann solver takes the two face states $\mathbf W^{\mathrm L}_f$, $\mathbf W^{\mathrm R}_f$ and returns one flux
$\mathbf F_f$. It adds just enough upwinding to keep the scheme stable. snapy solves in the face-local orthonormal
frame. It rotates the velocity so that the first component is normal to the face, solves a one-dimensional
problem, and rotates the flux back. Six types are accepted: `hllc` (default), `lmars`, `roe`, `shallow-roe`,
`upwind` and `plume-roe`. Only `hllc` and `lmars` are on the CUDA-graph capture allow-list (chapter 11).

::: {.callout-note title="At a glance"}
**Purpose.** $\mathbf F_f(\mathbf W^{\mathrm L}_f, \mathbf W^{\mathrm R}_f)$, eqs. (4.9)-(4.12).

**Data.** In: face states `(n_var, n_3, n_2, n_1)` each, the EOS for internal energy, $\gamma$ and $c_s$. Out:
the flux (written into the register's slot), optionally the face pressure, optionally the central mass flux
`riemann.central`, and per-face event masks.

**Placement.** Every RK stage, after the reconstruction of each direction (section 4.3).

**Activation.** `dynamics/riemann-solver/type`, default `hllc`; `dir` (`omni`, or `xy`/`yz` for
`shallow-roe`).

**Guarantee / limits.** All: consistent. HLLC: three waves, contact pressure clamped non-negative.
LMARS: one acoustic face pressure and velocity, upwinded on the face velocity. Roe: falls back to local Lax-Friedrichs where its
star densities are negative.

**Evidence.** `test_riemann.release`, `test_roe_cubed_sphere_rest_python` (Tests layer).
:::

**Local symbols:** $\lambda^-,\lambda^+$ the HLLC outer wave speeds; $b^-, b^+$ the same clipped at zero;
$\lambda^{\mathrm c}$ the contact speed; $p^{\mathrm c}$ the contact pressure; $v_n$ the face-normal and $v_t$ a
tangential velocity in the face frame; $\mathcal Z_f$ the face impedance $\bar\rho\bar c$ of LMARS (NOTATION's $Z$ is the
background impedance); $p_f, v_{n,f}$ the LMARS face pressure and velocity; $\hat h_t$ the total specific enthalpy;
$J_K, Q_K, \Gamma_K$ the HLLC intermediates (renamed from $m, t, q$, which NOTATION reserves for momentum, time and a generic quantity); $p_{\mathrm{bg}}$ the face pressure background of the perturbation entry.

Every solver carries the dry-air mass flux $\rho_{\mathrm d}v_n = \rho(1-\sum_n y_n)v_n$ in the density row and
$\rho y_n v_n$ in species row $n$. The face pressure output is kept as a total for the implicit and energy
consumers (`src/hydro/sweep.cpp:134-136@73c45a7c`) and feeds the $x_1$ pressure source
(`src/hydro/hydro_forward.cpp:701@73c45a7c`). The central mass flux $\tfrac12(\mathbf F(\mathbf W^{\mathrm L})+\mathbf F(\mathbf W^{\mathrm R}))$ of the
mass rows feeds the axial-momentum closure of chapter 7.

### Derivation

**Frame.** For a face normal to $x_a$, `to_face_frame_` rotates both states to the face-local orthonormal frame
(`prim2local{3,2,1}_` for $x_1, x_2, x_3$). The kernel then reads the normal velocity from row `IPR - dim` and
the tangential ones in cyclic order (`face_axes`, `src/riemann/riemann_kernels.h:35-40@73c45a7c`). The flux is
rotated back by `flux2global*_` (`src/riemann/riemann.cpp:16-51,292,343@73c45a7c`). Below, $v_n$ is the normal
velocity and $v_t$ either tangential velocity in that frame, $K\in\{\mathrm L,\mathrm R\}$ labels a side, and
$c_s^K$, $\gamma_K$ are that side's sound speed and ratio of specific heats from the EOS.

**HLLC.** HLLC solves the one-dimensional problem with three waves, $\lambda^-\le\lambda^{\mathrm c}\le\lambda^+$
(Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, ch. 10). The outer speeds come from the
primitive-variable pressure estimate (the code comment cites Toro 10.5.2):

$$
p_{\mathrm{mid}} = \tfrac12\big(p^{\mathrm L}+p^{\mathrm R}\big) + \tfrac12\big(v_n^{\mathrm L}-v_n^{\mathrm R}\big)\,\bar\rho\,\bar c,
\qquad \bar\rho = \tfrac12(\rho^{\mathrm L}+\rho^{\mathrm R}),\quad \bar c = \tfrac12(c_s^{\mathrm L}+c_s^{\mathrm R}),
$$
$$
\Gamma_K = \begin{cases}1 & p_{\mathrm{mid}}\le p^K\\ \sqrt{1+\frac{\gamma_K+1}{2\gamma_K}\big(\frac{p_{\mathrm{mid}}}{p^K}-1\big)} & \text{otherwise,}\end{cases}
\qquad \lambda^- = v_n^{\mathrm L}-c_s^{\mathrm L}\Gamma_{\mathrm L},\quad \lambda^+ = v_n^{\mathrm R}+c_s^{\mathrm R}\Gamma_{\mathrm R} \tag{4.9}
$$

(`src/riemann/riemann_kernels.h:79-102@73c45a7c`). Write $J_{\mathrm L} = \rho^{\mathrm L}(v_n^{\mathrm L}-\lambda^-)$,
$J_{\mathrm R} = \rho^{\mathrm R}(\lambda^+-v_n^{\mathrm R})$ and $Q_K = p^K + \rho^Kv_n^K(v_n^K-\lambda_K)$, with
$\lambda_{\mathrm L}=\lambda^-$, $\lambda_{\mathrm R}=\lambda^+$. The normal-momentum jump condition across the left wave,
with pressure $p^{\mathrm c}$ and normal velocity $\lambda^{\mathrm c}$ in the star region, is

$$
p^{\mathrm c} - p^{\mathrm L} = \rho^{\mathrm L}(\lambda^- - v_n^{\mathrm L})(\lambda^{\mathrm c}-v_n^{\mathrm L})
\quad\Longleftrightarrow\quad p^{\mathrm c} = Q_{\mathrm L} - J_{\mathrm L}\lambda^{\mathrm c},
$$

and across the right wave $p^{\mathrm c} = Q_{\mathrm R} + J_{\mathrm R}\lambda^{\mathrm c}$. Equating the two and solving
first for $\lambda^{\mathrm c}$, then for $p^{\mathrm c}$:

$$
\lambda^{\mathrm c} = \frac{Q_{\mathrm L}-Q_{\mathrm R}}{J_{\mathrm L}+J_{\mathrm R}},\qquad
p^{\mathrm c} = Q_{\mathrm L} - J_{\mathrm L}\lambda^{\mathrm c} = \frac{J_{\mathrm L}Q_{\mathrm R}+J_{\mathrm R}Q_{\mathrm L}}{J_{\mathrm L}+J_{\mathrm R}}. \tag{4.10}
$$

These are the code's `am` and `cp` (`src/riemann/riemann_kernels.h:120-129@73c45a7c`). Their equality with Toro's
$S^*$ and $p^*$ (from either side) was checked symbolically (sympy 1.14.0). The code clamps $p^{\mathrm c}$ at 0 (at
$-p_{\mathrm{bg}}$ on the perturbation entry, below) and writes it as the face pressure (`:130-132`).

The flux is computed with the outer speeds clipped, $b^+ = \lambda^+$ if $\lambda^+>0$ else $10^{-10}$, and
$b^- = \lambda^-$ if $\lambda^-<0$ else $-10^{-10}$ (`riemann_tiny`, `:51-53`; `:104-105`). Define per side
$\mathbf f_K = \mathbf F(\mathbf W^K) - b_K\mathbf U^K$ with $b_{\mathrm L} = b^-$, $b_{\mathrm R} = b^+$. Its rows are
$\rho_{\mathrm d}(v_n-b)$, $\rho v_n(v_n-b)+p$, $\rho v_t(v_n-b)$, $E(v_n-b)+pv_n$ and $\rho y_n(v_n-b)$ (`:145-163`,
`:189-193`). With $\mathbf D = (0, p^{\mathrm c}, 0, 0, p^{\mathrm c}\lambda^{\mathrm c}, 0, \dots)$ (normal-momentum
and energy rows),

$$
\mathbf F_f = \begin{cases}
\dfrac{\lambda^{\mathrm c}}{\lambda^{\mathrm c}-b^-}\,\mathbf f_{\mathrm L} + \dfrac{-b^-}{\lambda^{\mathrm c}-b^-}\,\mathbf D & \lambda^{\mathrm c}\ge0,\\[8pt]
\dfrac{-\lambda^{\mathrm c}}{b^+-\lambda^{\mathrm c}}\,\mathbf f_{\mathrm R} + \dfrac{b^+}{b^+-\lambda^{\mathrm c}}\,\mathbf D & \lambda^{\mathrm c}<0
\end{cases} \tag{4.11}
$$

(`:167-187`). To see that this is HLLC, take $\lambda^{\mathrm c}\ge0$, so the face lies in the left star region.
Toro's star state satisfies

$$
(b^--\lambda^{\mathrm c})\,\mathbf U^{*\mathrm L} = (b^--v_n^{\mathrm L})\,\mathbf U^{\mathrm L}
 + \big(0,\; p^{\mathrm c}-p^{\mathrm L},\; 0,\; 0,\; p^{\mathrm c}\lambda^{\mathrm c}-p^{\mathrm L}v_n^{\mathrm L}\big),
$$

and the star flux is $\mathbf F^{*\mathrm L} = \mathbf F(\mathbf W^{\mathrm L}) + b^-(\mathbf U^{*\mathrm L}-\mathbf U^{\mathrm L})$.
Substituting $\mathbf U^{*\mathrm L}$ and collecting terms, the $p^{\mathrm L}$ terms cancel against the pressure
parts of $\mathbf F(\mathbf W^{\mathrm L})$ and what is left is the first line of (4.11). The right branch is the mirror
image. Both branches were checked symbolically against Toro's $\mathbf F^{*\mathrm L}$ and $\mathbf F^{*\mathrm R}$. In
the subsonic case ($\lambda^-<0<\lambda^+$) the clipped and unclipped speeds coincide. In the supersonic case
$\lambda^-\ge0$, $b^- = -10^{-10}$, and the left branch differs from $\mathbf F(\mathbf W^{\mathrm L})$ only by terms
proportional to $10^{-10}$.

**HLLC perturbation entry.** On the perturbation-flux path the pressure rows are anomalies about a face background
$p_{\mathrm{bg}}$. The wave speeds and the energy transport use totals $p_{\mathrm{bg}}+p$, while the momentum pressure and
the clamp use the anomaly (`:61-66`, `:85-86`, `:130-131`, `:186-187`).

**LMARS.** LMARS takes one face pressure $p_f$ and one face velocity $v_{n,f}$ from the linear acoustic Riemann
problem with impedance $\mathcal Z_f=\bar\rho\bar c$, where $\bar c = \sqrt{\tfrac12\bar\gamma(p^{\mathrm L}+p^{\mathrm R})/\bar\rho}$ and
$\bar\gamma = \tfrac12(\gamma_{\mathrm L}+\gamma_{\mathrm R})$. The characteristic variable $p+\mathcal Z_fv_n$ carries the left state
into the face, and $p-\mathcal Z_fv_n$ the right state. Solving $p^{\mathrm L}+\mathcal Z_fv_n^{\mathrm L} = p_f+\mathcal Z_fv_{n,f}$ and
$p^{\mathrm R}-\mathcal Z_fv_n^{\mathrm R} = p_f-\mathcal Z_fv_{n,f}$ gives

$$
p_f = \tfrac12(p^{\mathrm L}+p^{\mathrm R}) + \tfrac12 \mathcal Z_f(v_n^{\mathrm L}-v_n^{\mathrm R}),\qquad
v_{n,f} = \tfrac12(v_n^{\mathrm L}+v_n^{\mathrm R}) + \frac{p^{\mathrm L}-p^{\mathrm R}}{2\mathcal Z_f} \tag{4.12}
$$

(add and subtract the two conditions; checked symbolically). The flux is upwinded on the sign of $v_{n,f}$:
$\mathbf F_f = v_{n,f}\,(\rho_{\mathrm d}, \rho v_n, \rho v_t, \rho\hat h_t, \rho y_n)^{K} + (0, p_f, 0, 0, 0)$, with
$K=\mathrm L$ if $v_{n,f}>0$ and $\mathrm R$ otherwise, and $\hat h_t = e + \tfrac12|\mathbf v|^2 + p/\rho$
(`lmars_impl`, `src/riemann/riemann_kernels.h:197-255@73c45a7c`). The face pressure output is $p_f$ (`:221`).

**Roe, shallow-roe, upwind, plume-roe** (one paragraph, as the outline asks). `roe` is the Roe linearisation with
$\sqrt\rho$-weighted averages. With an ideal-moist EOS it also reads the vapor rows' molar-mass and heat-capacity
ratios and the energy offsets (`src/riemann/riemann.cpp:262-280@73c45a7c`). It checks the intermediate dry and
species densities of the Roe fan. Where one is negative, the face falls back to a local Lax-Friedrichs flux,
$\mathbf F_f = \tfrac12(\mathbf F^{\mathrm L}+\mathbf F^{\mathrm R}) - a(\mathbf U^{\mathrm R}-\mathbf U^{\mathrm L})$ with
$a = \tfrac12\max(|v_n^{\mathrm L}|+c_s^{\mathrm L}, |v_n^{\mathrm R}|+c_s^{\mathrm R})$, and its `llf` flag is set
(`roe_impl`, `src/riemann/riemann_kernels.h:263-444@73c45a7c`; the checks at `:363-392`, the fallback at
`:429-443`). `shallow-roe` is the Roe solver of the shallow-water equations in the $x$-$y$ or $y$-$z$ plane
(`:447-495`). `upwind` returns $\mathbf F = v\,\mathbf W^{\mathrm L}$ if $v>0$ and $v\,\mathbf W^{\mathrm R}$ otherwise, for a
face velocity $v$ the caller passes. `plume-roe` is a Lax-Friedrichs flux of a four-variable plume model whose dissipation differences the primitive
rows (`src/riemann/riemann.cpp:151-153@73c45a7c`). It refuses every `dim` but 1 (`src/riemann/riemann.cpp:176-177@73c45a7c`),
and with `DIM1 = 3, DIM2 = 2, DIM3 = 1` (`src/hydro/hydro_forward.cpp:301@73c45a7c`) that is the $x_3$ face, not $x_1$.
The last two are torch formulas (`src/riemann/riemann.cpp:129-186@73c45a7c`).

**Central mass flux.** When asked, the kernel also writes the mass rows of $\tfrac12(\mathbf F(\mathbf W^{\mathrm L}) +
\mathbf F(\mathbf W^{\mathrm R}))$: $\tfrac12(\rho_{\mathrm d}^{\mathrm L}v_n^{\mathrm L} + \rho_{\mathrm d}^{\mathrm R}v_n^{\mathrm R})$
in the density row and $\tfrac12(\rho^{\mathrm L}y_n^{\mathrm L}v_n^{\mathrm L} + \rho^{\mathrm R}y_n^{\mathrm R}v_n^{\mathrm R})$
in the species rows (`riemann_central`, `src/riemann/riemann_kernels.h:516-535@73c45a7c`).

**Exact and truncated.** Setting $\mathbf W^{\mathrm L} = \mathbf W^{\mathrm R}$ in (4.9)-(4.12) gives
$\mathbf F_f = \mathbf F(\mathbf W)$ for HLLC and LMARS, so both are consistent. For HLLC: $p_{\mathrm{mid}} = p$,
$\Gamma_K = 1$, $\lambda^{\mathrm c} = v_n$, $p^{\mathrm c} = p$, and (4.11) collapses to $\mathbf F(\mathbf W)$. For LMARS:
$p_f = p$, $v_{n,f} = v_n$. The dissipation is first order in the jump $\mathbf W^{\mathrm R}-\mathbf W^{\mathrm L}$, and
that jump is itself as small as the reconstruction error. The scheme's order on smooth data therefore comes from
(4.1). Properties beyond these (contact and shear resolution of HLLC, the low-Mach behaviour of LMARS) are
the solvers' textbook properties. They are not re-derived or tested in this draft.

### Numerical method

**Where it runs.** `RiemannSolverImpl::forward` dispatches `upwind` and `plume-roe` to torch formulas, and every
other kind to `solve` (`src/riemann/riemann.cpp:156-190@73c45a7c`). `solve` first gathers the EOS quantities per
side, in each kind's order of EOS calls (`:229-289`):

- `hllc`: energy density `W->I`, then $\gamma$ and $c_s$ from `gamma_sound_speed`, into buffers `elr`, `glr`,
  `clr`. On the perturbation entry the EOS sees totals (`:232-255`).
- `lmars`: specific internal energy `W->I`$/\rho$ and $\gamma$ (`:256-261`).
- `roe`: energies, $\gamma$, $c_s$ and, for ideal-moist, the vapor-row constants (`:262-280`).

It then rotates to the face frame and builds one tensor iterator over the faces. The iterator holds the flux, the
face pressure, the central rows and the LLF flag as outputs, and the two states and the EOS quantities as inputs.
It calls `call_riemann` once (`:292-326`). One function per kind, `riemann_face<Kind>`, is the loop body of the
CPU loop and of the CUDA kernel (`src/riemann/riemann_kernels.h:539-562@73c45a7c`). MPS runs the CPU loop on the
host (comment at `:18-21`).

**Events.** After the solve, `solve` records per-face event masks on the interior faces (`interior_faces`,
`src/riemann/riemann.cpp:208-217@73c45a7c`). Row 0 marks HLLC faces whose contact pressure was clamped
($p^{\mathrm c}\le0$, or $p^{\mathrm c}\le -p_{\mathrm{bg}}$ on the perturbation entry). Row 1 marks Roe faces that fell back to LLF
(`:333-341`). The sweep posts them (section 4.3).

**Figure.** `src/snapy_report/ch04/figures/fig_riemann_face.py` (cartoon: the face-local frame, the two states,
the face pressure). Its docstring cites the LMARS code at an earlier sha. Re-checking it against the pin is
open (Limits, item 4).

**Stage, explicit, order.** Every RK stage, explicit. Consistency and first-order upwinding as stated in
Derivation. Walls, seams and solids enter only through the face states (chapters 14, 13, 3).

**Precision.** HLLC keeps all arithmetic in the dispatched type, and a `static_assert` enforces it
(`src/riemann/riemann_kernels.h:94-97@73c45a7c`). In FP32 it forms $v_n-\lambda^\mp$ directly as $\pm c_s^K\Gamma_K$ instead
of subtracting large velocities (`:109-118`, `:136-143`). The FP64 branch keeps the subtraction (`:114-118`).

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| options and YAML | `src/riemann/riemann.hpp:55-65@73c45a7c`, `src/riemann/riemann.cpp:53-75@73c45a7c` | `RiemannSolverOptionsImpl::from_yaml` | `dynamics/riemann-solver/{type,dir}` |
| type to kind | `src/riemann/riemann.cpp:77-85@73c45a7c` | `parse_riemann_kind` | `type` |
| buffers and setup checks | `src/riemann/riemann.cpp:98-126@73c45a7c` | `RiemannSolverImpl::reset` | `dir` |
| entry points | `src/riemann/riemann.cpp:156-204@73c45a7c` | `RiemannSolverImpl::forward`, `forward_perturbations` | |
| face thermo, iterator, events | `src/riemann/riemann.cpp:219-346@73c45a7c` | `RiemannSolverImpl::solve` | |
| face frame | `src/riemann/riemann.cpp:16-51@73c45a7c`, `src/riemann/riemann_kernels.h:35-47@73c45a7c` | `to_face_frame_`, `to_global_frame_`, `face_axes`, `shallow_axes` | |
| HLLC | `src/riemann/riemann_kernels.h:56-194@73c45a7c` | `hllc_impl` | `hllc` |
| LMARS | `src/riemann/riemann_kernels.h:197-255@73c45a7c` | `lmars_impl` | `lmars` |
| Roe, shallow Roe | `src/riemann/riemann_kernels.h:263-495@73c45a7c` | `roe_impl`, `shallow_roe_impl` | `roe`, `shallow-roe` |
| central mass rows, face body | `src/riemann/riemann_kernels.h:516-562@73c45a7c` | `riemann_central`, `riemann_face` | |
| dry fraction | `src/eos/ideal_moist_impl.h:11-18@73c45a7c` | `dry_fraction` | |

Walk-through. (i) `from_yaml` accepts only `type` and `dir` under `dynamics/riemann-solver`. A missing block or key
keeps `hllc` and `omni` (`src/riemann/riemann.cpp:64-75@73c45a7c`, defaults at
`src/riemann/riemann.hpp:61,64@73c45a7c`). (ii) `create` parses the type, so an unknown one fails at
construction: "RiemannSolver: unknown type ..." (`src/riemann/riemann.cpp:84,353@73c45a7c`). (iii) `reset`
allocates the `elr`, `clr`, `glr` buffers for `hllc` and `lmars`. Its check that `shallow-roe` uses `dir = yz` on
gnomonic-equiangle compares the type with `"gnomonic_equiangle"` (`:118`), while the coordinate type is spelled
`gnomonic-equiangle` (`src/coord/coordinate.cpp:468,714@73c45a7c`), so it never fires; the only `dir` check that
runs is `solve`'s `xy`-or-`yz` test (`src/riemann/riemann.cpp:284-287@73c45a7c`). (iv) Per sweep, `solve` as described above.

Errors a user can hit: "riemann.central is not defined for Riemann solver type ..." (`:160-163`); "Face-pressure
output is not implemented for ..." for `upwind`, `plume-roe` and `shallow-roe` (`:165-168`, `:174-175`,
`:282-283`); "PlumeRoeSolver only supports dim = 1" (`:176-177`); "Riemann solver perturbations need hllc"
(`:196-197`); "ShallowRoeSolver takes options->dir() = 'xy' or 'yz'" (`:284-287`).

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_riemann.release` `RiemannSolverOptions.from_yaml_reads_its_section` | `type`/`dir` are read from the deck | `EXPECT_EQ` | `tests/test_riemann.cpp:151-167@73c45a7c` |
| `test_riemann.release` `RiemannSolver.{hllc,lmars,roe}_writes_face_pressure_output`, `roe_..._ideal_moist` | the face pressure equals the formula recomputed in torch, e.g. HLLC's clamped $p^{\mathrm c}$ of (4.10) | `torch::allclose(..., 1.e-10, 1.e-10)` | `tests/test_riemann.cpp:169-375@73c45a7c` (rtol, atol positional) |
| `test_riemann.release` `HLLCPerturbation.all_directions_and_contact_branches_match_total_flux` | perturbation entry with $p_{\mathrm{bg}} = 10^7$ equals the total flux, 3 dims $\times$ 5 velocities | $2\cdot10^{-12}\lvert F\rvert + 2\cdot10^{-6}$ per row; face pressure 1.e-7 | `tests/test_riemann.cpp:377-409@73c45a7c` |
| `test_riemann.release` `HLLCPerturbation.float_pressure_difference_survives_large_background` | FP32, $p_{\mathrm{bg}}=10^7$, $\pm0.01$ Pa anomalies: flux signs right; a negative anomaly is kept | `EXPECT_NEAR` 1.e-8f; `EXPECT_EQ` | `tests/test_riemann.cpp:411-431@73c45a7c` |
| `test_riemann.release` `HLLCFloat.shocks_rarefactions_and_boosts_match_double` | FP32 HLLC against FP64 on Sod, strong shock, expansion, boosts up to $10^8$ | $3\cdot10^{-4}\max(1,\lvert F\rvert)$ | `tests/test_riemann.cpp:433-481@73c45a7c` |
| `test_riemann.release` `DeviceTest.hllc_pressure_split_dispatch_and_on_host` | the dispatched kernel and the `OnHost` CPU path agree; species rows are the mixture flux times their mass fraction | on CPU `torch::equal` (bitwise); on CUDA `allclose(2.e-5, 1.e-6)` flux and `allclose(2.e-5, 1.e-7)` face pressure; species `allclose(2.e-6, 1.e-12)` | `tests/test_riemann.cpp:483-549@73c45a7c` |
| `test_riemann.release` `DeviceTest.riemann_central_matches_the_two_extra_solves` | `riemann.central` mass rows equal $\tfrac12(\mathbf F(\mathbf W^{\mathrm L},\mathbf W^{\mathrm L})+\mathbf F(\mathbf W^{\mathrm R},\mathbf W^{\mathrm R}))$; non-mass rows 0 | relative max error 1.e-14 (FP64), 2.e-6 (FP32) | `tests/test_riemann.cpp:551-601@73c45a7c` |
| `test_riemann.release` `DeviceTest.hllc_perturbation_central_matches_total_central` | the central rows on the HLLC perturbation path equal those on the total path; non-mass rows 0; FP64 mass-row flux difference between the paths | `EXPECT_EQ` 0 (non-mass rows); relative 1.e-14 (FP64) / 2.e-6 (FP32) on both central comparisons; relative 1.e-9 (FP64 flux) | `tests/test_riemann.cpp:603-691@73c45a7c` (the bounds at `:651,679-680,686-688`) |
| `test_riemann.release` `DeviceTest.g_r8_riemann_central_on_the_aam_decks` | FP64 only: the kernel's `riemann.central` against $\tfrac12(\mathbf F(\mathbf W^{\mathrm L},\mathbf W^{\mathrm L})+\mathbf F(\mathbf W^{\mathrm R},\mathbf W^{\mathrm R}))$ on every $x_2$/$x_3$ face of two decks (lmars/weno5 3D arm; shallow-roe W2 C16), normalised by $\max\rho(\lvert v_n\rvert+c_s)$ | `EXPECT_LE` 1.e-13 | `tests/test_riemann.cpp:693-700,843-850@73c45a7c` |
| `test_riemann.release` `DeviceTest.riemann_events_count_clamped_and_llf_faces` | $v_1=\mp1$ (left/right): no events; $v_1=\mp2000$: HLLC clamps every interior face, Roe flags $>0$ and $\le$ interior faces | `EXPECT_EQ`/`GT`/`LE` (counts) | `tests/test_riemann.cpp:856-901@73c45a7c` |
| `test_roe_cubed_sphere_rest_python` | a resting six-panel atmosphere stays at rest with `roe` over 2 cycles | $\max\lvert v_{2,3}\rvert < 10^{-9}$ m s$^{-1}$, no redo | `tests/test_roe_cubed_sphere_rest.py:36@73c45a7c`; the docstring says that without the face-local projection the corners spin up to about 2 m s$^{-1}$ (`:4-5`), a statement of the test's author, not a run of this draft |

Devices: the `DeviceTest` suites run on CPU and CUDA in FP32 and FP64 (`tests/device_testing.hpp:44-50@73c45a7c`).
The plain `TEST`s (perturbation, FP32-versus-FP64) call `hllc_impl` on the host. One rank throughout. **Run
evidence at the pin: missing evidence** (no recorded run in this draft).

### Limits

1. Only `hllc` and `lmars` capture as CUDA graphs. `roe`, `shallow-roe` and `upwind` are refused with a named
   reason (`src/mesh/stage_executor.cpp:645-652@73c45a7c`). Chapter 11 owns the rule.
2. The perturbation entry is HLLC only (`src/riemann/riemann.cpp:196-197@73c45a7c`).
3. The contact-pressure clamp and Roe's LLF fallback change the flux silently. They are counted (section 4.3)
   but not refused.
4. The figure's agreement with the pin is not checked in this draft.
5. Local symbols ($\lambda^\pm, b^\pm, \lambda^{\mathrm c}, p^{\mathrm c}, v_n, v_t, \mathcal Z_f, p_f, v_{n,f}, \hat h_t, J_K, Q_K, \Gamma_K,
   p_{\mathrm{bg}}$) need an ISSUES.md item.
6. No convergence or shock-tube test of the full scheme is cited here. The outline lists none for 4.2.
   Chapter 16 holds the verification catalogue.
7. The gnomonic `dir = yz` check for `shallow-roe` never fires (a string mismatch, Code walk-through (iii)), so a
   gnomonic `shallow-roe` deck with `dir: xy` is not refused.

---

## 4.3 The sweep and its register posts

### Summary

The sweep is the control flow that joins sections 4.1 and 4.2 for one direction. It reconstructs the face states,
counts the faces the floors changed, solves, counts the solver's events, and posts the flux to the block's flux
register. On the FP32 path it also builds the pressure gauge and floors the total face states itself. There is no
physics to derive. What matters is the order, the counts and the one post per solve.

::: {.callout-note title="At a glance"}
**Purpose.** Reconstruct, then floor, then solve, then post, per direction; one reduced count per floor or event that ran.

**Data.** In: `hydro_w`, optional split-path background. Out: flux in `freg.slot(d, "riemann")`; masks into
`RegisterMonitor` counters `face_floor.{rho,p,species}`, `riemann.{pstar_clamp,llf}`.

**Placement.** `HydroImpl::forward` steps (2) $x_1$, (3.A)/(3.B) and (4.A)/(4.B) $x_2$/$x_3$; counts summed once
per stage at `flush_counts`.

**Activation.** Always on. The counts exist whenever their floor or event ran.

**Guarantee / limits.** A floor count is the number of face states changed, each side of a face counted;
a solver-event count is the number of faces. Exact (integer sums).

**Evidence.** `test_face_floor.release`, `test_fp32_face_floor_counts_python`, `test_riemann.release` (events).
:::

The $x_1$ sweep with gravity takes the well-balanced path of chapter 5. This section describes the plain $x_1$ path
and the $x_2$/$x_3$ path.

### Derivation

None: this is control flow. The one quantitative statement is the count. Each floor that runs posts a boolean
mask over both sides of the faces it computed (the side axis of the result, `src/recon/reconstruct.cpp:280-300@73c45a7c`,
`src/recon/reconstruct.hpp:168-175@73c45a7c`), `true` where it changed a face state. The monitor sums the masks as
integers (`RegisterMonitor::flush_counts`, `src/hydro/flux_register.cpp:139-156@73c45a7c`). Integer addition is
exact, so a floor counter equals the number of face states changed in the stage, up to two per face; this is what
`test_face_floor.release` asserts, summing over both sides (`tests/test_face_floor.cpp:135-146@73c45a7c`). The
solver events are one mask per interior face (`src/riemann/riemann.cpp:334-341@73c45a7c`), so those counters count
faces.

### Numerical method

**Order of operations, FP64 path** (`reconstruct_faces`, `src/hydro/sweep.cpp:37-49@73c45a7c`;
`riemann_solve`, `:105-118`):

1. `recon->forward(w, dim)` with `precon1` for $x_1$ and `precon23` for $x_2$/$x_3$. The floors run inside it
   (section 4.1).
2. `post_face_floors` posts one mask per floor that ran, under `face_floor.rho`, `face_floor.p` or
   `face_floor.species` (`:23-29`).
3. For $x_2$/$x_3$: the immersed-solid step, then the covariance hook (section 4.4) on the unrotated states
   (`solve_faces`, `:139-157`).
4. Without `perturbation-flux`, `priemann->forward` into the register's slot `slot(d, "riemann")`, optionally
   with `riemann.central` (`:114-118`, `:153-155`); with `perturbation-flux` on, in FP64 as in FP32, the solve is
   `forward_perturbations` on the split states (`:120-133`; Split-path solve, below).
5. `post_events`, the lambda in `riemann_solve` (`:111-113`), calls `post_riemann_events` (`:31-35`). It posts
   the solver's masks under `riemann.pstar_clamp` and `riemann.llf`.
6. `reg.post_slot(d, "riemann")` (`:156`). For $x_1$ the same post is at
   `src/hydro/hydro_forward.cpp:584-586@73c45a7c`.

In `HydroImpl::forward`, $x_1$ runs in step (2) (`src/hydro/hydro_forward.cpp:454@73c45a7c`). On the plain path
(no well-balanced reference, `wb_x1` false at `:479-480`), it reconstructs at `:552-553`. $x_2$ and $x_3$ are
reconstructed in steps (3.A) and (3.B) (`:706-740`) and solved in (4.A) and (4.B) (`:759-834`). On a cubed
sphere, the $x_2$/$x_3$ face states are exchanged across panels between those steps (`:722-727`, `:734-739`).

**FP32 split path** (`src/hydro/sweep.cpp:50-102@73c45a7c`). The state is reconstructed in FP32 with the pressure
row replaced by $(p_{\mathrm{bg}} - p_{\mathrm{gauge}}) + p'$, where $p_{\mathrm{bg}}$ is the split path's cell pressure
background (`pressure0`), $p'$ its increment (`pressure_delta`), and $p_{\mathrm{gauge}}$ the midrange
$\tfrac12(\min+\max)$ of $p_{\mathrm{bg}}$ along the sweep. The code comment gives the reason for the midrange: on a
column that is uniform along the sweep it equals the column value exactly in every block, while a rounded mean
would depend on the decomposition (`:50-57`). The reconstruction's own density and pressure floors are off (`floor = false`) on this
path. The species floor stays. The sweep then floors the total face density and pressure itself and posts the
counts with `count_below`:

- on an active horizontal sweep without covariance: the floor tests the total face pressure, the reconstructed
  value plus $p_{\mathrm{gauge}}$, and the static face gauge is subtracted after the clamp (`:72-88`);
- otherwise after casting the face states back to FP64 and adding $p_{\mathrm{gauge}}$ (`:90-101`).

**Split-path solve.** If no background was set (horizontal or gravity-free sweeps), `riemann_solve` subtracts the
static face gauge, the mean $p_{\mathrm{bg}}$ of the two cells beside each face (`static_face_gauge`, `:16-21`), and calls
`forward_perturbations`. It adds the background back to the face pressure output, so every consumer sees totals
(`:120-136`).

**How the counts are reduced.** `RegisterMonitor::count` copies each mask into a per-stage buffer slot
(`src/hydro/flux_register.cpp:75-92@73c45a7c`). `flush_counts` sums all slots with one cumulative sum over the
buffer. It gathers each counter's segment ends, adds once into the counters, and zeros the buffer. No
`index_add_` is involved, which would scatter with FP64 atomics on CUDA (`:139-156`). A floor that did not run
posts nothing. A slot no poster wrote this stage counts zero.

**Figure.** `src/snapy_report/ch04/figures/fig_stage_pipeline.py` (cartoon of one stage inside `HydroImpl::forward`,
drawn against an earlier sha). It does not yet show the count posts. Both re-checking it and adding them are
open (Limits).

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| face floors to monitor | `src/hydro/sweep.cpp:23-29@73c45a7c` | `post_face_floors` | EOS `limiter` |
| solver events to monitor | `src/hydro/sweep.cpp:31-35@73c45a7c` | `post_riemann_events` | |
| reconstruct + FP32 gauge/floors | `src/hydro/sweep.cpp:37-103@73c45a7c` | `reconstruct_faces` | `dynamics/precision`, `perturbation-flux` |
| solve + events (`post_events` lambda) | `src/hydro/sweep.cpp:105-137@73c45a7c` | `riemann_solve` | |
| $x_2$/$x_3$ solve and post | `src/hydro/sweep.cpp:139-157@73c45a7c` | `solve_faces` | `disable-flux-x2`, `disable-flux-x3` |
| static face gauge | `src/hydro/sweep.cpp:16-21@73c45a7c` | `static_face_gauge` | |
| counts | `src/hydro/flux_register.cpp:75-156@73c45a7c` | `RegisterMonitor::count`, `flush_counts` | |
| the masks | `src/recon/reconstruct.hpp:168-175@73c45a7c`, `src/riemann/riemann.hpp:105-111@73c45a7c` | `ReconstructImpl::floored`, `RiemannSolverImpl::events` | |

`disable-flux-x<n>` skips the solve and the post of that direction, but not the immersed-solid step
(`src/hydro/sweep.cpp:148@73c45a7c`; defaults false, `src/options/hydro.yaml:22-24@73c45a7c`).

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_face_floor.release` `hydro.floor_counts_equal_the_faces_changed` (and `_cuda`) | with a negative density in one cell and a negative pressure in another, `face_floor.rho` and `face_floor.p` equal the number of faces the floors changed; `face_floor.species` is 0 | `EXPECT_EQ` (exact counts) | `tests/test_face_floor.cpp:104-148@73c45a7c` |
| `test_face_floor.release` `hydro.shock_path_floors_density_and_pressure` (and `_cuda`), `shock_defaults_to_false` | the shock path floors too; the default is false | `EXPECT_EQ` | `tests/test_face_floor.cpp:200-217@73c45a7c` |
| `test_face_floor.release` `hydro.face_floor_counter_sees_the_fallback` | the well-balanced fallback shows in `face_floor.rho` | `EXPECT_GE` 1 | `tests/test_face_floor.cpp:150-162@73c45a7c` (the path is chapter 5's) |
| `test_face_floor.release` `hydro.face_floor_fires_on_an_unresolved_column` | dipped-face mass flux $1.28524\times10^{-8}$ | relative $10^{-5}$ (`1.e-5 * 1.28524e-8`) | `tests/test_face_floor.cpp:91-101@73c45a7c`; a pinned value of the well-balanced $x_1$ path, cited here only because the outline lists it (chapter 5 owns it). The `_cuda` variant compares CUDA with CPU to relative 1.e-5 (`:181-194`) |
| `test_face_floor.release` `hydro.face_floor_uses_adjacent_density` | a face away from the dip carries mass flux 4.07888 | `EXPECT_NEAR` 1.e-3 | `tests/test_face_floor.cpp:68-84@73c45a7c` (chapter 5's path) |
| `test_fp32_face_floor_counts_python` | FP32 split path: 0 counts at the deck's pressure floor $10^{-10}$ Pa; more than 0 `face_floor.p` at $5\times10^4$ Pa | none numeric (pass/fail counts) | `tests/test_fp32_face_floor_counts.py:67-90@73c45a7c` |
| `test_riemann.release` `riemann_events_count_clamped_and_llf_faces` | see section 4.2 | counts | `tests/test_riemann.cpp:856-901@73c45a7c` |

The `test_face_floor` suite also runs as `test_face_floor_x1_centroid.release` with `SNAP_X1_CENTROID_EXACT=1`
(`tests/CMakeLists.txt:172-176@73c45a7c`). The `_cuda` cases skip without CUDA. One rank. **Run evidence at the pin:
missing evidence.**

### Limits

1. Counts are per stage and summed into the monitor's counters. Their rollback on a redo is chapter 10's.
2. The FP32 path's sweep-level floors count with `count_below` on the same computed faces as the reconstruction,
   `il..iu+1` (`src/hydro/sweep.cpp:64-69@73c45a7c`). No test asserts that this count equals the number of faces
   changed. `test_fp32_face_floor_counts_python` checks zero against nonzero only.
3. `fig_stage_pipeline.py` predates the pin and the count posts.

---

## 4.4 Face-average corrections at $O(h^2)$ (`dynamics/flux-covariance`)

### Summary

A finite-volume flux through an $x_2$ or $x_3$ face must be the face average of the point flux. The Riemann
solver evaluates the flux from face states that are themselves averages over the face's $x_1$ extent, so it
returns the flux of the averages. The two differ at $O(h^2)$, and in a stratified atmosphere the difference is not
small. It acts like a spurious stratification of the horizontal energy flux. `dynamics/flux-covariance` (default
**on**) adds the missing term per row to the $x_2$/$x_3$ flux. On curved grids it also adds the shift between the
centroid of the cell measure and that of the face measure.

::: {.callout-note title="At a glance"}
**Purpose.** $\Delta\mathbf F_f$ added to the $x_2$/$x_3$ face flux, eqs. (4.13)-(4.16).

**Data.** In: the $x_2$/$x_3$ face states before the solver's frame rotation; the grid's face moment and centroid
shift. Out: correction rows (energy, species, dry mass; momentum on curved grids only).

**Placement.** Inside `solve_faces`, as the `before_solve` hook; added to the flux after the solve, steps (4.A)/(4.B).

**Activation.** `dynamics/flux-covariance: true` (default). Off restores the plain face product. It must be the same in every block.

**Guarantee / limits.** At rest every row vanishes except the curved-grid normal-momentum pressure part, which the geometric source cancels. Removes the $O(h^2)$ covariance of the $x_1$ profile. The other in-face direction is neglected.

**Evidence.** `test_flux_covariance_rows_python`, `test_horizontal_flux_covariance_python`, `test_x1_centroid_rest_python`.
:::

**Local symbols:** $\varsigma^2_{i}$ the second central $x_1$ moment of an $x_2$/$x_3$ face's own area measure over
cell $i$ (NOTATION.md's $\sigma_i^2$ is the cell's own-measure variance; the two agree in Cartesian); $\delta^{\mathrm f}_i$
the $x_1$ offset between the centroid of the cell measure and that of the face measure (NOTATION.md's $\delta_i$ is
a different offset); $v_n$ the face-normal velocity, as in section 4.2; $\tilde x$ the $x_1$ offset from the face
measure's centroid (NOTATION's $\xi$ is a panel coordinate) and $q_c$ a field's value there (expansion only); $D_1[q]_i = (q_{i+1}-q_{i-1})/(x_{1,i+1}-x_{1,i-1})$ the
centred $x_1$ difference the code uses.[^ch04-cov-src]

[^ch04-cov-src]: Derivation sources, cited, not pasted: `doc/tech-report/sources/deriv__289-covariance-x3-curved.md`,
`doc/tech-report/sources/study__289-allrows_derivation.md`; the code points to
`docs/derivations/289-covariance-x3-curved.md` (`src/hydro/hydro_forward.cpp:139@73c45a7c`). Not re-read against
this draft; see Limits, item 4. Credit and trail: chengcli/snapy#289.

### Derivation

**Covariance of a density-weighted product.** Inside one cell, write each field linearly in $\tilde x$, the $x_1$ offset
from the face measure's centroid: $\rho = \rho_c+\tilde x\,\partial_1\rho$, $v_n = v_{n,c}+\tilde x\,\partial_1v_n$,
$q = q_c+\tilde x\,\partial_1q$. Assume, for this paragraph, that the cell values are averages over the same measure
(exact in Cartesian; on curved grids the centroid term (4.16) carries the leading difference). Let the measure be
symmetric to this order, $\langle\tilde x\rangle = 0$, $\langle\tilde x^2\rangle = \varsigma^2$, $\langle\tilde x^3\rangle=0$. The
stored cell values are density-weighted: $\bar\rho = \langle\rho\rangle$, $\tilde v_n = \langle\rho v_n\rangle/\bar\rho$,
$\tilde q = \langle\rho q\rangle/\bar\rho$. Then

$$
\begin{aligned}
\bar\rho &= \rho_c, \\
\tilde v_n &= v_{n,c} + \varsigma^2\,\partial_1\rho\,\partial_1v_n/\rho_c, \qquad
\tilde q = q_c + \varsigma^2\,\partial_1\rho\,\partial_1q/\rho_c,\\
\langle\rho v_nq\rangle &= \rho_cv_{n,c}q_c + \varsigma^2\big(\rho_c\,\partial_1v_n\,\partial_1q + v_{n,c}\,\partial_1\rho\,\partial_1q
 + q_c\,\partial_1\rho\,\partial_1v_n\big),\\
\bar\rho\,\tilde v_n\tilde q &= \rho_cv_{n,c}q_c + \varsigma^2\big(q_c\,\partial_1\rho\,\partial_1v_n + v_{n,c}\,\partial_1\rho\,\partial_1q\big)
 + O(\varsigma^4),
\end{aligned}
$$

so, subtracting the last two lines,

$$
\langle\rho v_nq\rangle - \bar\rho\,\tilde v_n\tilde q = \varsigma^2\rho_c\,\partial_1v_n\,\partial_1q + O(h^4). \tag{4.13}
$$

(checked symbolically). No term contains $\partial_1\rho$, and the total mass flux $\langle\rho v_n\rangle = \bar\rho\tilde v_n$
has no gap at all. This is the code's statement that "no pair contains rho'" (`src/hydro/hydro_forward.cpp:147-162@73c45a7c`).
Applied row by row, with $\partial_1$ replaced by $D_1$:

$$
\begin{aligned}
\Delta F^{(y_n)} &= \varsigma^2\rho\,D_1[v_n]\,D_1[y_n], \qquad
\Delta F^{(\rho_{\mathrm d})} = -\textstyle\sum_n \Delta F^{(y_n)},\\
\Delta F^{(E)} &= \varsigma^2\rho\,D_1[\hat h]\,D_1[v_n],\qquad \hat h = (\rho e + p)/\rho,
\end{aligned} \tag{4.14}
$$

with $\rho, v_n, \hat h$ evaluated from the face-average state $\tfrac12(\mathbf W^{\mathrm L}+\mathbf W^{\mathrm R})$. The
dry row carries minus the tracer sum because the total mass flux has no gap. The energy row uses the solver's own
specific enthalpy, `W->I` plus $p$, over $\rho$, with every species' energy offset (`:100-104`, `:175`). The code
leaves out the velocity-squared parts of the momentum and energy gaps, calling them "the scheme's ordinary
second-order truncation". It also leaves out a pressure Hessian term, which it estimates at "~1e-3 of the energy
term" (`:160-162`). That estimate is the code comment's and is not re-derived here. The covariance along the other
in-face direction is dropped, because only $x_1$ carries an $O(1)$ background gradient (`:80-82`).

**Face moment.** In Cartesian the face weight along $x_1$ is uniform, so $\varsigma^2 = h^2/12$. On spherical-polar
and cubed-sphere grids an $x_2$/$x_3$ face has area density $\propto r$. With $r_\pm$ the cell's faces,
$h = r_+-r_-$ and $\bar r = \tfrac12(r_-+r_+)$:

$$
r_c = \frac{3\bar r^2 + h^2/4}{3\bar r},\qquad \varsigma^2 = \frac{h^2}{12}\Big(1-\frac{h^2}{12\bar r^2}\Big)
\tag{4.15}
$$

(`src/coord/coordinate.hpp:294-307@73c45a7c`; both checked symbolically from $\int r\cdot r\,\mathrm dr/\int r\,\mathrm dr$
and $\int r^2\cdot r\,\mathrm dr/\int r\,\mathrm dr - r_c^2$).

**Centroid shift.** A cell stores the average over the cell measure ($r^2\mathrm dr$ on these grids), whose
centroid is $r_v$. The face flux needs the average over the face measure ($r\,\mathrm dr$), whose centroid is $r_c$.
Evaluating the flux at the wrong centroid misses $-(r_v-r_c)\partial_1 F$:

$$
\Delta\mathbf F^{\mathrm{ctr}} = -\delta^{\mathrm f}\,D_1[\mathbf F],\qquad
\delta^{\mathrm f} = r_v - r_c = \frac{h^2(12\bar r^2-h^2)}{12\bar r(12\bar r^2+h^2)} \tag{4.16}
$$

(`src/coord/coordinate.hpp:340-364@73c45a7c`; the closed form checked symbolically). It applies to every row, the
momentum rows included, with $\mathbf F$ the plain point flux of the face-average state (`ctr`,
`src/hydro/hydro_forward.cpp:173@73c45a7c`). The normal-momentum pressure part, $-\delta^{\mathrm f}D_1p$, uses the
one-sided difference at the first and last interior cell (`d1_pressure`, `:36-46`). The geometric source uses the
same difference, so that rest stays exact (`:221-224`). In Cartesian $\delta^{\mathrm f}\equiv0$ and none of this is
computed (`:140-143`, `:181-196`).

**What is exact and what is truncated.** Every term of (4.14), and every row of (4.16) but one, carries a factor
$v_n$ or $D_1[v_n]$, and both are exactly zero at rest. The exception is the normal-momentum pressure part
$-\delta^{\mathrm f}D_1p$, which has no velocity factor and is nonzero at rest on a curved grid
(`src/hydro/hydro_forward.cpp:221-224@73c45a7c`). A state at rest stays at rest because the same difference is
subtracted from the cell pressure the geometric source reads (`src/hydro/hydro_forward.cpp:946-959@73c45a7c`); this
draft does not claim that cancellation is bitwise. The code
comment (`src/hydro/hydro_forward.cpp:84-88@73c45a7c`) adds that the covariance part is also zero for an isothermal state but the centroid part is not, so that
second exact zero holds in Cartesian only (`:86-88`); for the energy row this follows from (4.14) when $\hat h$
depends on $T$ alone, and it is not re-derived here for the tracer rows. The correction is added to the face flux, so it telescopes in the $x_2$/$x_3$ sums like the flux
itself and conserves every row. The remaining error is $O(h^4)$ in the $x_1$ profile plus the neglected
velocity-squared and other-direction terms above.

### Numerical method

**Where and when.** In steps (4.A)/(4.B), the sweep lambda installs the covariance as the `before_solve` hook of
`solve_faces` (`src/hydro/hydro_forward.cpp:801-830@73c45a7c`). The hook first makes the $x_1$ seam rows
single-valued (`x1_seam_rows`), then calls `_flux_covariance` on the face states before the solver rotates them.
After the solve, `add_cov` adds the result (`:782-795`). In Cartesian only the interior $x_1$ cells of the energy,
species and dry rows are written, stacked as (energy, species..., dry). On curved grids the full flux-shaped
correction is added. On $x_3$ faces the moment is taken along $x_1$ too, because $x_1$ is the stratified direction.

**Stencil.** $D_1$ is centred over $x_{1,i\pm1}$ and needs both neighbours. The check `n1 < 3`
(`src/hydro/hydro_forward.cpp:92-96@73c45a7c`) reads the tensor extent, ghosts included (`nc1`,
`src/coord/coordinate.hpp:62@73c45a7c`), so only an `nx1 = 1` block skips the correction. Shallow water gets none
(`src/hydro/hydro_forward.cpp:98@73c45a7c`). The ghost rows next to a cubed-sphere panel edge are not read by the
pressure difference (`src/hydro/hydro_forward.cpp:30-35@73c45a7c`).

**Caching.** `face_moment2_x1()` is grid-only. It is cached per block, keyed on the `x1f` tensor, its version,
device and dtype (`:117-132`).

**Passive scalars.** When scalars are present, the same face states are kept for `scalar_flux_covariance`, so the
scalar rows get the corrections of hydro tracers with the same mass fraction (`:814-820`;
`src/hydro/hydro.hpp:152-163@73c45a7c`).

**Figure.** `src/snapy_report/ch04/figures/fig_x1cov_order.py` reads `data/x1cov_orders.json` from
`checks/covariance_check.py`. That check ports `_flux_covariance` and the $x_1$ mass-flux covariance at two earlier
shas, and the outline asks for a re-port at the pin. Neither the re-port nor a re-draw is done in this draft
(Limits, item 3).

**The $x_1$ face counterpart** (`SNAP_X1_CENTROID_EXACT`, environment, off by default). On spherical-polar grids it
converts every $x_1$ reconstruction input from $r^2$ means to plain means first, and uses an $r^2$-consistent $x_1$
pressure source (`src/coord/x1_centroid.hpp:8-21@73c45a7c`; read once per process, `src/coord/x1_centroid.cpp:68-78@73c45a7c`;
applied at `src/hydro/hydro_forward.cpp:484-501@73c45a7c`). It is cited here because the outline's Tests line for
4.4 includes `test_x1_centroid_rest_python`. Its derivation belongs with the $x_1$ geometry (chapter 3) and
reference (chapter 5).

### Code

| what | code location | symbol | switch |
|---|---|---|---|
| the correction | `src/hydro/hydro_forward.cpp:58-231@73c45a7c` | `HydroImpl::_flux_covariance` | `dynamics/flux-covariance` |
| one-sided pressure difference | `src/hydro/hydro_forward.cpp:36-46@73c45a7c` | `d1_pressure` | |
| centroid shift fetch | `src/hydro/hydro_forward.cpp:49-56@73c45a7c` | `centroid_shift_x1` | |
| hook and add | `src/hydro/hydro_forward.cpp:782-830@73c45a7c` | `add_cov`, `sweep` lambda in `HydroImpl::forward` | |
| face moment, centroid shift | `src/coord/coordinate.hpp:285-364@73c45a7c` | `CoordinateImpl::face_moment2_x1`, `radial_face_moment2_`, `face_centroid_shift_x1`, `face_centroid_shift_zero`, `radial_face_centroid_shift_` | geometry type |
| switch and default | `src/options/hydro.yaml:43-51@73c45a7c`, `src/hydro/hydro.hpp:197@73c45a7c` | `HydroImpl::flux_covariance` | default `true` |
| $x_1$ counterpart | `src/coord/x1_centroid.hpp:8-21@73c45a7c` | `x1_centroid_exact_enabled` | `SNAP_X1_CENTROID_EXACT` |

The option's schema text says every block of a run must make the same choice, or shared faces stop being
single-valued (`src/options/hydro.yaml:47-51@73c45a7c`). No code check enforcing that rule was found at the pin
(Limits, item 2). The $x_1$ mass-flux covariance `dynamics/x1-mass-covariance` (default false,
`src/options/hydro.yaml:52-59@73c45a7c`) is the $x_1$-face sibling. It enters through `WellBalancedX1::covariance`
(`src/recon/reconstruct.hpp:200-201@73c45a7c`) and is described with the well-balanced path in chapter 5.

### Tests

| test (ctest name) | what it asserts | tolerance | where the tolerance comes from |
|---|---|---|---|
| `test_flux_covariance_rows_python` arm `rest` | balanced isothermal shell, spherical-polar and six-panel gnomonic, term on: $\max\lvert v\rvert/c_s$ after 20 steps | `REST_TOL` = 1e-9 | `tests/test_flux_covariance_rows.py:34,37,295-299@73c45a7c` |
| ... arm `uniform` | six-panel moist shell, solid-body wind, 20 steps: uniform vapor stays uniform | `UNIFORM_TOL` = 1e-13 | `:35,300-304` |
| ... arm `offset` | vapor energy offset $u_0/R$ = 0 versus 2000: max relative primitive difference, term on | `OFFSET_TOL` = 1e-10 | `:36,305-313` |
| ... arm `cartesian` | dry Cartesian one-step $\varepsilon_{\mathrm{eff}}n_1^2$ at $n_1$ = 16, 32, 64 against recorded 0.009686, 0.007529, 0.006987 | `CART_TOL` = 2e-3 absolute | `:38-43,314-320`; the recorded values are constants in the test source, not a run of this draft |
| `test_horizontal_flux_covariance_python` | $\varepsilon_{\mathrm{eff}}n_1^2$ off in $[-0.32,-0.20]$; on $\lvert\cdot\rvert<0.04$ ($n_1$ = 16, 32); key unset equals `true` bitwise; `false` differs; $E+P$ drift over 50 steps | `BASE` = (-0.32, -0.20); `FIXED` = 0.04; `torch.equal`; `EP_TOL` = 1.e-13 | `tests/test_horizontal_flux_covariance.py:35-39,174-200@73c45a7c`; the docstring gives the defect estimate $\varepsilon_{\mathrm{spur}}\approx-(\pi^2/12)\beta h^2 = -0.235/n_1^2$ (`:7-8`), a statement of the test, not re-derived here |
| `test_x1_centroid_rest_python` | `SNAP_X1_CENTROID_EXACT`: spherical-polar column of $r^2$ means at $r_0$ = 5 and 1000, $n_1$ = 32, explicit and implicit, full and split pressure: force imbalance $f$ | on: $f <$ `TOL_ON` = 1.0e-10; off: $f >$ `TOL_OFF` = 1.0e-8 at $r_0 = 5$ | `tests/test_x1_centroid_rest.py:15-21,40-41@73c45a7c` |

Devices: these scripts take `--device`. `test_x1_centroid_rest_cuda_python` is registered for CUDA
(`tests/CMakeLists.txt:624-626@73c45a7c`). `tests/CMakeLists.txt` at the pin registers no CUDA run of the two covariance scripts
(`grep -n flux_covariance tests/CMakeLists.txt` lists only CPU registrations for them, `:475-476`), so they are
CPU-only in ctest. One rank (the six panels run in one process). **Run evidence at the pin: missing evidence.**

### Limits

1. The covariance along the other in-face coordinate, and the velocity-squared parts, are neglected by design
   (Derivation). Their size in a GCM configuration is not measured.
2. "Every block must make the same choice": no runtime check enforcing this was found at the pin. A Mesh
   built from one deck satisfies it.
3. `checks/covariance_check.py` ports the code at earlier shas. The outline asks for a re-port at the pin, which is
   not done. `fig_x1cov_order.py` therefore draws pre-pin data.
4. The derivation notes cited in the footnote were not re-read for this draft. The derivation above was re-derived
   from the code and checked symbolically for (4.13), (4.15) and (4.16).
5. Under `dynamics/precision: float32` the FP32 sweep keeps the total face states when the covariance is on
   (`src/hydro/sweep.cpp:70-74@73c45a7c`). The FP32 rows are chapter 12's.

---

### Checker notes (remove at conversion)

- Every `path:lines@73c45a7c` above was read in a `git archive` export of snapy `73c45a7c8c991209924e1137992352c335575062`.
  The outline's chapter-4 anchors all resolve to the named symbols at that sha:
  `src/recon/reconstruct.cpp:252,119@73c45a7c` (`ReconstructImpl::forward`, `InterpImpl::forward`),
  `src/riemann/riemann.cpp:156,219@73c45a7c` (`RiemannSolverImpl::forward`, `solve`),
  `src/hydro/sweep.cpp:37,23,105,31@73c45a7c` (`reconstruct_faces`, `post_face_floors`, `riemann_solve`,
  `post_riemann_events`), `src/hydro/hydro_forward.cpp:58@73c45a7c` (`HydroImpl::_flux_covariance`).
- Symbolic checks (not committed to the report tree): sympy 1.14.0, Python 3.11.13, run as `python3 -I <script>`.
  (4.2) cp3/cp5 rows; (4.3) WENO5 candidates and $0.3q^{[0]}+0.6q^{[1]}+0.1q^{[2]} = $ cp5; (4.7) uniform $k^\pm=1/24$ and the
  non-uniform identity; the $h^4/80$ remainder; (4.10)-(4.11) against Toro's $S^*$, $p^*$, $\mathbf F^{*\mathrm L}$, $\mathbf F^{*\mathrm R}$; (4.12);
  (4.13); (4.15); (4.16). All passed.
