# The default well-balanced x1 reference at a physical wall

This is the default reference (conversion-order 2, without the face density of `wb-ref4.md`), with
`dynamics/wb-wall-clamp` on (the default). With the
repeated wall cell, the face density reference has an $O(\Delta z)$ error at the first three faces next to each
wall, against $O(\Delta z^2)$ elsewhere. Continuing $\rho/p$ past the wall, linearly or ln-linearly, restores the
interior order there and leaves every other face bit for bit. The code is `src/hydro/hydro_ref_x1_impl.h`
(`hydro_ref_x1_rop_smooth`, `hydro_ref_x1_cell_impl`), with the tensor (MPS) path in
`src/hydro/hydro_dispatch.cpp`. The test is `tests/test_wb_ref_wall.cpp`.

## 1. What the kernel builds

Per column, with $r_i = \bar\rho_i/\bar p_i$ and the binomial $B = (1, 4, 6, 4, 1)/16$:

$$
r^s_i = \sum_{m=-2}^{2} B_m\, r_{i+m},\qquad
\rho_{{\rm ref},i} = p_{{\rm ref},i}\, r^s_i,\qquad
\rho_{{\rm sf},f} = p_{{\rm sf},f}\,\tfrac12\big(r^s_{f-1} + r^s_f\big),
$$

where face $f$ is the lower face of cell $f$. The solver's face density is $\rho_{\rm sf} + \mathcal W[\bar\rho -
\rho_{\rm ref}]$, where $\mathcal W$ is WENO5 and the perturbation's ghosts at a wall are filled with even parity
(`hydro_forward.cpp`). At 37dce4e:
- the clamp is at `hydro_ref_x1_impl.h:74`: an index past the wall is replaced by the wall cell;
- `jlo`/`jhi` are set at `:159-160`;
- $r^s$, $\rho_{\rm ref}$ and $\rho_{\rm sf}$ are formed at `:161-167`;
- the even-parity fill is at `hydro_forward.cpp:276-292`.

## 2. Interior order

For smooth $r$, $B$ has unit sum, zero first moment and second moment 1 (in cells), so $r^s_i = r_i +
\tfrac{\Delta z^2}{2} r''_i + O(\Delta z^4)$. The mean of two neighbouring cells adds $\tfrac{\Delta z^2}{8}r''$.
With $p_{{\rm sf},f} = p(z_f)$ from the scan, and $r_i = R_i + O(\Delta z^2)$ for a ratio of averages,

$$
\rho_{{\rm sf},f} = \rho(z_f) + O(\Delta z^2)\qquad\text{(interior faces)}.
$$

## 3. The repeated wall cell is first order

Near the bottom wall, write $r_j = r_0 + j\,a + O(\Delta z^2)$ with $a = r'\Delta z = O(\Delta z)$, and index the
cells from the wall, $j = 0, 1, \dots$. Repeating $r_0$ for every $j < 0$ gives:

- wall cell: $r^s_0 = (11 r_0 + 4 r_1 + r_2)/16 = r_0 + \tfrac{3}{8}a$;
- next cell: $r^s_1 = (5 r_0 + 6 r_1 + 4 r_2 + r_3)/16 = r_1 + \tfrac{1}{16}a$;
- first ghost (it enters the wall face): $r^s_{-1} = r_0 + \tfrac{1}{16}a$, where the true value is $r_0 - a$.

Against the true face values $r(z_f)$, the face references are off by:

| face (cells from the wall) | error of $\tfrac12(r^s_{f-1}+r^s_f)$ |
|---|---|
| 0 (the wall face) | $\tfrac{23}{32}a$ |
| 1 | $\tfrac{7}{32}a$ |
| 2 | $\tfrac{1}{32}a$ |
| $\ge 3$ | $O(\Delta z^2)$ |

At the top wall the signs flip. Relative to $r$ the error is a fixed multiple of $(r'/r)\Delta z$, so it is
**first order**. It vanishes when $r' = 0$, so an isothermal column has no wall error.

Check against the test column (polytrope $T = 1 - \beta z$, $\beta = 0.5$, one pressure scale height, nz 64,
$\Delta z = 0.01230$; $r'/r = \beta/T$). The prediction at face 1 is $\tfrac{7}{32}\cdot0.5\cdot0.0123 = 1.35\times10^{-3}$,
and `test_wb_ref_wall` measures $1.39\times10^{-3}$. At face 2 the prediction is $1.9\times10^{-4}$ plus the
interior $O(\Delta z^2)$ part; it measures $2.5\times10^{-4}$.

**The even-parity ghosts.** $\rho' = \bar\rho - \rho_{\rm ref}$ inherits the wall bias,
$-\tfrac38 a\,p$ in the wall cell and $-\tfrac1{16}a\,p$ in the next. The even mirror carries that $O(\Delta z)$
feature into the ghosts, so $\mathcal W[\rho']$ does not cancel it, and the solver's face densities $\rho_{L,R}$ at
faces 1 and 2 are also first order (measured: $\rho_L$ at face 1 is $-5.2\times10^{-4}$, then $-2.5\times10^{-4}$ at nz 32, 64).
The mirror is not the source. Applied to a $\rho'$ that is $O(\Delta z^2)$ and smooth near the wall, it costs
only $O(\Delta z^2)$. The source is the bias in $\rho_{\rm ref}$.

## 4. The closure: continue $\rho/p$ past the wall, linearly or ln-linearly

Past a clamped wall, take

$$
r_{-k} = \begin{cases} r_0 + k\,(r_0 - r_1), & r_1 \le r_0,\\ r_0\,(r_0/r_1)^k, & r_1 > r_0,\end{cases}
\qquad k = 1, 2, 3
$$

(and the mirror image at the top), in place of $r_0$ (`hydro_ref_x1_wall_rop`). The first branch is the linear
continuation of $r$ and the second that of $\ln r$. The two differ by $O(\Delta z^2)$, and $B$ reproduces a linear
profile exactly, so the $O(a)$ terms above cancel. What remains is the continuation's own curvature error,
$\tfrac{k(k+1)}{2}\Delta z^2$ times $r''$ or $(\ln r)''\,r$, which is $O(\Delta z^2)$. The wall cells, and faces
0–2, then have the interior's order. For constant $r$ the first branch gives $r_0$ exactly, so an isothermal
column is unchanged bit for bit.

**Why two branches.** Each branch is unbounded on one side:
- linear in $r$ goes to zero when $r$ rises away from the wall. With $d = (r_1 - r_0)/r_0$, $r_{-3} = r_0(1 - 3d)$
  vanishes at $d = 1/3$ and is negative past it;
- ln-linear goes to infinity when $r$ falls away from the wall. A nearly empty neighbour, $r_1 \ll r_0$, gives
  $r_0(r_0/r_1)^k$.

Each case takes the branch that stays closer to $r_0$, so for $r_1 \ge 0$ the continuation lies in
$[\,r_0(r_0/r_1)^k,\; r_0(1+k)\,]$. In `test_face_floor`'s unresolved column, the cell next to the top wall is
dipped to $10^{-4}$, and the dipped-face mass flux is
- $3.0\times10^{-8}$ with the clamp,
- $2.83\times10^{-8}$ with either the linear form or this closure,
- $3.36\times10^{-7}$ with the ln-linear form everywhere.

Guards (the clamp stays):
- the wall side must own at least two cells, so only owned cells are read and the clamp's property tests in
  `test_hydro_ref_x1` hold;
- a value that is not positive falls back to $r_0$.

Measured with `test_wb_ref_wall`, $\beta = 0.5$, dsf at face 1 (the bottom wall, where $r$ rises away from the
wall, so the ln-linear branch applies; the top wall takes the linear branch):

| nz | clamp only | linear in $r$ everywhere | this closure |
|---|---|---|---|
| 16 | $6.16\times10^{-3}$ | $6.48\times10^{-4}$ | $8.04\times10^{-4}$ |
| 32 | $2.88\times10^{-3}$ | $1.57\times10^{-4}$ | $1.95\times10^{-4}$ |
| 64 | $1.39\times10^{-3}$ | $3.87\times10^{-5}$ | $4.82\times10^{-5}$ |
| 128 | $6.84\times10^{-4}$ | $9.78\times10^{-6}$ | $1.22\times10^{-5}$ |

The observed order goes from about 1.05 to about 2.0 (1.81–2.12 over faces 1–2 at both walls, for dsf,
$\rho_L$ and $\rho_R$, nz 32 → 64). The solver's $\rho_L$ at face 1 goes from $-2.54\times10^{-4}$ (nz 64) to
$7.5\times10^{-6}$. Faces 1–2 at both walls stay at or below the largest interior error ($1.45\times10^{-4}$ at
nz 64).

**What changes.** At each clamped physical wall: $\rho_{\rm sf}$ at faces 0, 1, 2 and $\rho_{\rm ref}$ at cells 0, 1,
plus their ghost rows. $p_{\rm sf}$, $p_{\rm ref}$ and every other face and cell are bit for bit as before
(compared at nz 16–128). The rest balance does not involve $\rho_{\rm sf}$ or $\rho_{\rm ref}$ (§1 of
`wb-ref4.md`), so `balance_column` and the discrete rest state are unchanged.

**Why not the `wb_ref4.cpp` closure.**
- The face density of `wb_ref4.cpp` (run by `dynamics/conversion-order: 4`; also by the switch `SNAP_WB_REF4`
  until that switch was removed) is a post-pass that replaces $\rho_{\rm sf}$ with a fourth-order face value on
  *every* face, so applying it would change every interior face. It also needs four owned cells.
- A cubic (or quadratic) continuation inside $B$ would also restore second order, with weights of 10–20 against
  $1+k$; it was not chosen.

## 5. The straka CFL 1.6 run

`test_straka_redo` runs `examples/straka_single.yaml` with CFL 1.6 and tlim 60 s, and fails if one step needs more
than 5 redos. CFL 1.6 is past the scheme's stability limit, so this is a robustness test, and it is marginal: most
redos are triggered by non-finite states. Runs reaching 60 s, with nx1 64, one thread, and
$\Delta T = -15(1 + k\epsilon)$:

| reference | test deck | $\epsilon = 10^{-12}$, $k = 0..9$ | $\epsilon = 10^{-4}$, $k = 0..19$ |
|---|---|---|---|
| clamp only | pass | 10/10 | 20/20 |
| linear in $r$ everywhere | fail (31.67 s) | 0/10 | 18/20 |
| ln-linear everywhere | pass | 9/10 | 19/20 |
| this closure | pass | 10/10 | 18/20 |

With $\epsilon = 10^{-12}$ the runs of this closure stay on one trajectory, ending within $10^{-4}$ in kinetic
energy, so that column says little beyond the test deck itself. The $\epsilon = 10^{-4}$ column does not separate
the four references. With `SNAP_WB_REF4` on (when that switch existed), the $10^{-12}$ ensemble gave 8/10 on the clamp-only reference and
9/10 on the linear one.

Resolution scan with the test deck (nx2 = 4 nx1):
- the clamp-only reference fails at nx1 128 (34.38 s, in the interior);
- the linear one fails at nx1 64;
- this closure reaches 60 s at nx1 32, 64 and 128, with at most 3 consecutive redos.

**The linear form's failure.** The references at $t = 0$ are not odd for any closure. In the straka column the top
cell's $\rho_{\rm ref}$ is within 0.002% of the cell average with the continuation, against 0.15% with the clamp,
and the positivity fallback does not fire. The failure develops after about 28 s:
- on the failing run, the linear $r_{+3}/r_0$ at the top wall falls to 0.35 at 30.9 s and to $-0.15$ at 31.67 s;
- on the clamp-only run, which survives, the same quantity would reach $-4.9$ at 32.1 s.

That the near-zero continuation starves the top cells is a hypothesis; it was not isolated.

**Not covered.** Non-uniform x1: the continuation is in index, as the binomial is. The MPS tensor path is changed
to the same rule but was not run here, since there is no MPS device.
