# #289: $r^2$-exact x1 cell-to-face maps on spherical-polar grids (`SNAP_X1_CENTROID_EXACT`)

This file derives the switch `SNAP_X1_CENTROID_EXACT` (off unless set, read once per process; it acts on
spherical-polar blocks only) and states the oracle it is tested against. It was written as an extension of
the fourth-order well-balanced reference of `wb-ref4.md` to the $r^2$ cell measure; that face density now runs
only with `dynamics/conversion-order: 4` (the switch `SNAP_WB_REF4` is gone). The exact identities used
below are checked by `docs/derivations/verify_x1_centroid.py` (sympy, exact rational arithmetic). The code is
`src/coord/x1_centroid.cpp`, called from `HydroImpl::forward` (`src/hydro/hydro_forward.cpp`, the x1 block),
`HydroImpl::_hydro_ref_x1` (`src/hydro/hydro.cpp`) and `SphericalPolarImpl::forward`
(`src/coord/spherical_polar.cpp`). The test is `tests/test_x1_centroid_rest.py`.

## 0. Notation

- $r \equiv x_1$, gravity $g$ toward $-r$. Cell $i$ spans $[r_{i-1/2}, r_{i+1/2}]$, width $h_i$, mid-radius
  $\bar r_i$. Face areas and the radial volume, per unit solid angle: $A_{i\pm1/2} = r_{i\pm1/2}^2$,
  $V_i = (r_{i+1/2}^3 - r_{i-1/2}^3)/3$, as `face_area1()` and `cell_volume()` form them.
- A finite-volume cell holds the **$r^2$ mean** $\langle q\rangle_i = V_i^{-1}\int r^2 q\,dr$. The **plain mean**
  is $\overline{q}_i = h_i^{-1}\int q\,dr$.
- $x = (r - \bar r_i)/h_i$ is the local coordinate of cell $i$.

## 1. The defect: the stored value is not the plain mean

Expanding $q$ about $\bar r_i$,

$$
\langle q\rangle_i = \overline{q}_i + \delta_i\, q'(\bar r_i) + O(h^4/\bar r^2), \qquad
\delta_i = \frac{\int r^2 (r-\bar r_i)\,dr}{\int r^2\,dr} = \frac{h_i^2}{6\bar r_i} + O(h^4/\bar r^3),
$$

so the stored value sits at the centroid $x_{1v} = \bar r_i + \delta_i$, not at the mid-radius. Every x1
cell-to-face map in the solver is a uniform-measure formula: the WENO5 reconstruction, the hydrostatic scan
$p_{i-1/2} = p_{i+1/2} + g h_i \rho_i$, the cell pressure reference (the cell average of the quintic through
six faces) and Leg W's density reference. Fed $r^2$ means they are off by $\delta_i q' = O(h^2/r)$:

- (i) the face values of the reconstruction carry $O(h^2/r)$;
- (ii) the scan step is $g h\langle\rho\rangle$ where hydrostatics needs $g\int\rho\,dr = g h\overline\rho$, so the
  face pressures drift by $O(h^2/r)$ per cell against the stored cell pressures $\langle p\rangle$; at rest the
  perturbation $\langle p\rangle - p_{\rm ref}$ is then $O(h^2/r)$ instead of zero, and its reconstructed face
  value is a spurious force;
- (iii) the pressure source. The momentum equation in the cell is
  $-\frac{1}{V}\int r^2 \partial_r p\,dr = -\frac{1}{V}[A p] + \frac{1}{V}\int 2 r p\,dr$. The base source
  (with the face pressures in the flux) is $(A_{i+1/2}p_{i+1/2} - A_{i-1/2}p_{i-1/2})/V_i - (p_{i+1/2} -
  p_{i-1/2})/h_i$, so the net pressure force is the plain difference $-(p_{i+1/2} - p_{i-1/2})/h_i$. It balances
  gravity $g\langle\rho\rangle_i$ only with the scan of (ii); once the scan is right ($g h\overline\rho$), the
  force has to be the $r^2$ mean of $-\partial_r p$, which the plain difference is not.

On a hydrostatic column of $r^2$ means the base therefore leaves an $O(h^2/r)$ radial force, and the
horizontal-energy covariance harness sees a $1/R$ term of magnitude about $0.022/R$ in $\varepsilon_{\rm eff} n_z^2$ (`#289`, leg (d); section 6).

## 2. (i)+(ii): convert every x1 input to plain means

Before the x1 block, every primitive row is replaced, for the x1 maps only, by its plain mean

$$
\overline{q}_i = \sum_{k=0}^{4} c_{i,k}\,\langle q\rangle_{s_i + k},
$$

with weights fixed by exactness for the monomials $x^n$, $n = 0..4$ (in cell $i$'s own coordinate):

$$
\sum_{k} c_{i,k}\,\langle x^n\rangle_{s_i+k} = \overline{x^n}_i = \left(1,\ 0,\ \tfrac{1}{12},\ 0,\ \tfrac{1}{80}\right)_n .
$$

- The window $s_i = \max(\mathrm{lo}, \min(\mathrm{hi}-4, i-2))$ is centred, and kept inside the owned cells at
  a physical x1 wall ($\mathrm{lo} = i_s$, $\mathrm{hi} = i_u$), inside the whole array at a seam. The
  outermost ghosts' windows are off-centre there, so after the conversion the seam ghost rows are replaced
  by the x1 neighbour's own plain means (`HydroImpl::_x1_ghost_rows`), which is what one block holds.
- The $5\times5$ moment matrix $\langle x^n\rangle_{s+k}$ is integrated by four-point Gauss-Legendre, exact
  because $x^n r^2$ has degree $\le 6$, and solved once per block (dense, partial pivoting).
- Exactness to degree 4 makes the conversion error $O(h^5 q^{(5)})$ times an $O(h/r)$ measure factor, i.e.
  $O(h^6/r)$; it is not exact at degree 5 (both checked in `verify_x1_centroid.py`, interior and both wall
  windows, uniform and stretched grids; the weights sum to 1).
- Ghost cells past a clamped wall get the mirrored correction of the owned cells,
  $\overline{q}_{i_s-1-m} = \langle q\rangle_{i_s-1-m} \pm (\overline{q}_{i_s+m} - \langle q\rangle_{i_s+m})$,
  odd for the normal velocity at a reflecting wall, so a mirrored state stays mirrored.
- The primitives themselves are not changed: the conversion is a view for the x1 maps. After it, the scan step
  is $g h\overline\rho = \int \rho\,dr + O(h^7/r)$, which is (ii), and the reconstruction sees the plain means
  its weights assume, which is (i).

## 3. The reference: Leg W on plain means is $r^2$-exact

On plain means the Leg W reference (`wb-ref4.md`: the quartic-primitive face
value and the range and resolution guards) is exactly the
Cartesian algorithm, whose $O(\Delta z^4)$ face-density error is derived there. That face density now runs only
with `dynamics/conversion-order: 4`. The switch implied `SNAP_WB_REF4` until that switch was removed: on its own
it no longer turns the `wb-ref4.md` face density on (its face density is then the x1 reconstruction's own
linear rows on the plain-mean reference), and it keeps only its nghost >= 3 requirement (ctest
`test_x1_seam_split_nghost_check_x1_centroid`). A Cartesian column
balanced outside the solver is at the solver's fixed point (ctest `test_balance_column_x1_centroid`,
`test_face_floor_x1_centroid`).
`balance_column` has no $r^2$-to-plain-mean conversion, so under the switch it balances only a column declared
`geometry='cartesian'` and refuses any other: a spherical-polar column's rest state is the solver's own reference. With the `wb-ref4.md`
face density on (conversion-order 4; the numbers below were measured when the switch implied it), the face
density is fourth order in the interior. At the three faces
next to a wall the error is $O(h^3)$, Leg W's own property, the same in Cartesian: the anchor's $O(h^2)$
constant $c$ enters $\rho' = \rho - p_{\rm ref} F[\rho/p]$ as $-c\,F[\rho/p]$, and the even-parity ghosts of
$\rho'$ turn its slope into an $O(h c)$ face value. The base clamped binomial is $O(h)$ there; in a Python replica of the x1
pipeline the wall face-density error goes from $2.7\times10^{-3}$ (base) to
$5.7\times10^{-8}$ ($n_z = 64$) and $7.2\times10^{-9}$ ($n_z = 128$).

## 4. (iii): the pressure source as the $r$-moment of a quintic

With $\tilde p$ the quintic through the six face pressures nearest cell $i$ (window
$s = \max(\mathrm{lo}, \min(\mathrm{hi} - 5, i - 2))$ over faces; at a physical wall $\mathrm{lo} = i_s$ or
$\mathrm{hi} = i_u + 1$, so the window is one-sided and still contains both faces of cell $i$; at a seam
$\mathrm{lo} = i_s - 2$ or $\mathrm{hi} = i_u + 3$, two faces past the seam, whose pressures the x1 neighbour
supplies, so the window is the centred one of one block), the x1 pressure source is

$$
S_i = \frac{2}{V_i}\int_{r_{i-1/2}}^{r_{i+1/2}} r\,\tilde p\,dr = \sum_{j=0}^{5} w_{i,j}\, p_{s+j}, \qquad
w_{i,j} = \frac{2}{V_i}\int r L_j(r)\,dr ,
$$

$L_j$ the Lagrange basis on the six faces, the integral by four-point Gauss-Legendre (exact: degree 6).
Integration by parts, with $\tilde p$ passing through both faces of the cell,

$$
-\frac{A_{i+1/2} p_{i+1/2} - A_{i-1/2} p_{i-1/2}}{V_i} + S_i = -\frac{1}{V_i}\int r^2\,\tilde p'\,dr ,
$$

so the net pressure force is the $r^2$ mean of $-\partial_r\tilde p$, the finite-volume form of the momentum
equation. A constant $p$ gives $S_i V_i = A_{i+1/2} - A_{i-1/2}$ exactly (with $V_i$ in the same cube form as
`cell_volume()`), so a constant pressure exerts no force against the face-area flux. Both identities are
checked exactly in `verify_x1_centroid.py`.

With `non-hydrostatic` $< 1$ the gravity term is replaced, in part, by the hydrostatic correction, the pressure
gradient of the cell's own reconstructed face states $p^L_{i+1/2}$, $p^R_{i-1/2}$. Without the switch it is the
plain difference $(p^L_{i+1/2} - p^R_{i-1/2})/\Delta x_1$, the same operator as the pressure force, so the two
cancel at rest whatever the column. With the switch the correction takes the operator of the pressure force:

$$
\frac{A_{i+1/2}\,p^L_{i+1/2} - A_{i-1/2}\,p^R_{i-1/2}}{V_i} - S_i ,
$$

$S_i$ from the Riemann face pressures as in the source, so at rest ($p^L = p^R = p^*$) it again cancels the
pressure force exactly, and away from rest the two differ only by $p^* - p^{L,R}$, as without the switch.

## 5. The rest balance and the oracle

Take a hydrostatic column, $p' = -g\rho$, initialised as the cell holds it: $r^2$ means of $\rho$ and $p$. With
the switch:

1. the conversion gives $\overline\rho_i$ to $O(h^6/r)$;
2. the scan gives faces $p_{i-1/2} = p(r_{i-1/2}) + c + O(h^6/r)$ (smooth in $i$), $c$ the anchor constant;
3. the reference cell pressure matches the cell's plain mean to the reference's own (Cartesian) order, so
   at rest the reconstructed perturbation vanishes to that order and the Riemann pressure is the face
   pressure; for the quadratic $p$ of the test below it vanishes to round-off;
4. the net force in cell $i$ is $-\frac{1}{V}\int r^2(\tilde p' + g\rho)\,dr$: the constant $c$ drops out
   (section 4) and $\tilde p' + g\rho = O(h^5/r)$.

For a density of degree $\le 4$ every step is exact and the column is balanced exactly, whatever $c$, and it
is not balanced without the conversion (`verify_x1_centroid.py`, check 4). For a smooth non-polynomial column
the curvature part of the residual is $O(h^6/r)$; what is left is the truncation of the fourth-order reference
itself, the same as in Cartesian geometry. In a Python replica of the x1 pipeline, where the reference is the
exact quintic, the maximum radial force is $3\times10^{-14}$ to $8\times10^{-14}$ at $r/H = 5, 20, 160, 1000$
($n_z = 64, 128$), against the base's $1.5\times10^{-6}$ ($r/H = 5$) to $1.0\times10^{-11}$ ($r/H = 1000$). In
the code (isothermal column $e^{-(r - r_0)/H}$, $n_z = 32$, one RK3 step, $\max|\rho v_1|/(\Delta t\,g\rho)$,
interior / wall cells) the switch takes $r_0/H = 5$ from $2.4\times10^{-5}$ / $2.5\times10^{-5}$ to
$1.0\times10^{-9}$ / $1.8\times10^{-8}$, which is the level of the base at $r_0/H = 1000$
($1.4\times10^{-9}$ / $1.4\times10^{-8}$), where curvature no longer matters: that floor, larger at the walls
(the reference's one-sided wall closure), does not depend on $r_0$.

The oracle (`tests/test_x1_centroid_rest.py`) therefore uses a column the design is exact for: a linear
density $\rho = 1 - 0.3\,(r - r_0)$ (quadratic $p$), cells initialised with their $r^2$ means, at $r_0/H = 5$
and $1000$, $n_z = 32$, explicit and vertically implicit, one RK3 step; asserted: switch on,
$\max|\rho v_1|/(\Delta t\,g\rho) < 10^{-10}$ in the interior and wall cells; switch off, $> 10^{-8}$ at
$r_0/H = 5$ (an ignored or misspelled switch fails). Measured: on, $1\times10^{-14}$ to $3\times10^{-14}$
everywhere; off, $2.4\times10^{-5}$ ($r_0/H = 5$) and $6.5\times10^{-10}$ ($r_0/H = 1000$).

## 6. What is left of the $1/R$ term

In the covariance harness (`#289` leg (d), spherical column, divergence-free seed, cell work, `#293` $x_3$
correction in full) the $1/R$ content $R\,[\varepsilon n_z^2(R) - \varepsilon n_z^2(\infty)]$ goes from
$-0.009 \ldots -0.027$ (base, $R/H = 5 \ldots 1000$, $n_z = 64, 128$) to $+0.0004 \ldots +0.0007$; it tends to
$+0.00045$, an $O(h^2/R)$ term. It is not the centroid and not the reference: it is the reconstruction of
the Favre velocity $\langle\rho w\rangle/\langle\rho\rangle$, a ratio of $r^2$ means and not itself a mean, as if
it were a cell mean of $w$ (a Cartesian $O(h^2)$ error, seen through the $2/r$ of the divergence). Feeding the
exact plain mean of $w$ instead takes it to $-0.00001$ at $n_z = 256$ (same replica). The
same holds for any x1 input that is a ratio of means ($T = p/\rho$, $\rho/p$ inside $F$); the conversion makes
the means plain, not the ratios exact.

The code reproduces the replica (one step, $R/H = 5, 20, 160, 1000$): off $-0.0011, -0.0170, -0.0226,
-0.0236$ and on $+0.0005, +0.0006, +0.0005, -0.0000$ at $n_z = 64$; off $-0.0030, -0.0205, -0.0274, -0.0340$ and
on $+0.0004, +0.0004, -0.0004, -0.0060$ at $n_z = 128$, where $R/H = 1000$ is noise-limited ($R$ times a
$\Delta\varepsilon$ of a few $10^{-10}$). These are column means with unit weight per cell, the harness's
metric. Weighted by cell volume instead, the on arm reads $+0.018 \ldots +0.020$ and the off arm
$-0.004 \ldots -0.012$, in the replica as in the code: the volume weight carries its own $1/R$ ($r^2 \propto
1 + 2x/R$) across the column's error profile, so it does not measure the scheme's $1/R$ term.

## 7. Scope

- **Seams.** The windows are one-sided only at physical x1 walls (`x1_neighbors()` is $-1$ there, and on a
  periodic or unsplit x1 column). At a seam the ghost plain means ($n_g$ cells) and the two ghost face
  pressures each side come from the x1 neighbour, local or remote, so a column split into x1 blocks keeps the
  one-block state to round-off (`tests/test_x1_seam_split.cpp`: an isothermal column with a density bump on
  the seam, 20 steps, full and hydrostatic-split pressure). Every block makes the same choice because the
  switch is read once per process.
- **Cubed sphere.** Not implemented: the gnomonic grids store $x_{1v}$ at the mid-radius, and their x1 maps
  would need the same conversion with the panel's own radial measure. The switch does nothing there.
- **Cartesian.** No conversion and the base source: a Cartesian block runs as with the switch off, apart from
  the nghost >= 3 check.
- **Off.** With the switch off every line of the base runs as before.
- **Small blocks.** With fewer than five cells to fit (the plain means) or six faces (the pressure source) the
  stencils are not usable and the base maps stay. Such a cache holds no weights, and its device/dtype check
  (`stale()`) does not read them (#298; before, the second RK stage threw at nx1 2–4;
  `tests/test_stencil_cache_small_block.py`).
