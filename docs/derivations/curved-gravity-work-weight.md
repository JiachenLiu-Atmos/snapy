# Face-form gravity work on a radial grid: exact weights versus discrete E+PE conservation

Scope: the x1 face-form gravity work (`gravity-work: face`) on a spherical-polar grid, explicit path
(`src/hydro/hydro_forward.cpp`, the `face_gravity_work` block) and implicit path (`src/implicit/implicit_hydro.cpp`,
`work_lo`/`work_hi`, which carry the same weights times 1/2 for the face average of the two cell momenta).
Base: 8cea3ae (the pre-rebase head of #293, merged to main as aea71ed). Every closed form below is checked by `curved_gravity_work_weight.py` (sympy + numpy;
`python docs/derivations/curved_gravity_work_weight.py` prints every number quoted; "replica" below).

**Result.** The exact $r^2$-measure weights remove both $O(h^2/\bar r)$ error terms, but they do **not** conserve
discrete E+PE: the per-face defect is $h^3/3$ (relative $h^2/(3R^2)$), the same order as the terms they remove.
No two-point weight choice that keeps the face form's conservation of E+PE$_d$ removes both terms; one or the other, not both.

**Option F (§7).** Conserving a corrected discrete PE, $P = \mathrm{PE}_d - g_1\sum_iV_i\sigma_i^2 s_i[\rho]$ (exact
to $O(h^4)$), instead of $\mathrm{PE}_d$ resolves the conflict: the work $W = W_{\rm face} + g_1\sigma^2 s[\dot\rho]$ is
$O(h^4)$-exact and conserves E+$P$ to round-off by construction. It changes Cartesian too and replaces (not adds to)
the cp3/cp5/weno5 curvature flux. §8 is the Cartesian case written to be ported on its own: there F removes the
$O(h)$ wall-cell error that face+H leaves, and leaves the interior unchanged to $O(h^4)$. §9 compares F with the $\bar r$ potential form; §10 lists
every place that computes a PE. F is always on with `gravity-work: face` (§11.2).

## 1. Setup

Per steradian (the polar and azimuthal factors are common to $A$ and $V$ and cancel):
$$A(r) = r^2,\qquad V_i = \tfrac13\left(r_{i+1/2}^3 - r_{i-1/2}^3\right),\qquad
r_{c,i} = \frac{\int_i r^3\,dr}{\int_i r^2\,dr} = \frac34\,\frac{r_{i+1/2}^4 - r_{i-1/2}^4}{r_{i+1/2}^3 - r_{i-1/2}^3},$$
where $r_c$ is `x1v` (`radial_centers` in `src/coord/spherical_polar.cpp`), the volume centroid (on a
gnomonic-equiangle grid `x1v` is the face midpoint instead; §12 adds the one term that changes). Gravity is
$g_1 = $ `grav1` $< 0$ and the potential is $\phi = -g_1 r$, so $\langle\phi\rangle_{V_i} = \phi(r_{c,i})$ exactly.
$F_f$ is the x1 mass-flux density at face $f$ and $G = A F = r^2 F$ the mass flow per steradian. Write
$r_\pm$ for a cell's faces, $h = r_+ - r_-$, $\bar r = (r_+ + r_-)/2$, $\delta = r_c - \bar r$.

Mass: $V_i\,\dot\rho_i = -(A_+F_+ - A_-F_-)$. Discrete potential energy: $\mathrm{PE} = \sum_i \rho_i\,\phi(r_{c,i})\,V_i$
(the exact $\int\rho\phi\,dV$ for piecewise-constant $\rho$).

Exact gravity work per unit volume: $g_1\langle F\rangle_V = \frac{g_1}{V}\int_{r_-}^{r_+} r^2F\,dr$.

## 2. The face-form booking

The code books, per unit volume and time,
$$W_i = \frac{1}{V_i}\Big[\phi_{c,i}\,(A_+F_+ - A_-F_-) - (A_+\phi_+F_+ - A_-\phi_-F_-)\Big]
      = \frac{g_1}{V_i}\Big[(r_+ - r_c)\,r_+^2F_+ + (r_c - r_-)\,r_-^2F_-\Big]. \tag{1}$$
The second form follows from $\phi_c - \phi_\pm = g_1(r_\pm - r_c)$.

**Leading error.** Expand $F = F_0 + F_1 s + F_2 s^2/2$, $s = r - \bar r$, and integrate exactly (replica §1):
$$\frac{W - g_1\langle F\rangle_V}{g_1} = \underbrace{\frac{h^2}{12}F''}_{\text{Cartesian}}
  + \underbrace{\frac{h^2}{6\bar r}F' - \frac{h^2}{6\bar r^2}F}_{\text{curvature}} + O(h^4). \tag{2}$$
The curvature pair is the term the radial ablation found (with its sign: face minus exact). The Cartesian term is the trapezoid
error present on every grid.

Equivalent form in $G = r^2F$: $\;W - g_1\langle F\rangle_V = \frac{g_1}{V}\big[-\delta h\,G' + \frac{h^3}{12}G''\big] + O(h^5)$,
with $\delta = h^2/(6\bar r) + O(h^4)$. So (1) is **exact for $G$ = const**, a steady divergence-free radial mass
flow ($F \propto r^{-2}$): $W = g_1 G h / V = g_1\langle F\rangle_V$ for any $h$.

## 3. Exact replacement weights

A two-point booking $W = g_1(a F_+ + b F_-)$ that is exact for $F \in \{1, r\}$ on the $r^2$ measure needs
$a + b = 1$ and $a r_+ + b r_- = \langle r\rangle_V = r_c$. The solution is unique:
$$a = \frac{r_c - r_-}{h},\qquad b = \frac{r_+ - r_c}{h}. \tag{3}$$
Note that (3) puts the weight $(r_c - r_-)$ on $F_+$, where (1) puts $(r_+ - r_c)r_+^2/V$ on $F_+$: the roles
of the two offsets are swapped. Its error (replica §1, $F$ expanded to $s^4$; the odd orders vanish):
$$\frac{W_{\rm exact} - g_1\langle F\rangle_V}{g_1} = \frac{h^2}{12}F''
  + h^4\Big(-\frac{F''}{360\,\bar r^2} + \frac{F'''}{360\,\bar r} + \frac{F''''}{480}\Big) + O(h^6).\tag{4}$$
Both curvature terms of (2) are gone; the Cartesian term stays. Cartesian limit: $r_c = \bar r$, so
$a = b = 1/2$, and (1) gives $(h/2)\cdot A/V = 1/2$ as well, so (3) reduces to the old weights there.

## 4. Discrete E+PE conservation

**Lemma (what conservation requires).** Let every cell book a two-point work $W_i V_i = g_1(\alpha_i F_{i+1/2} +
\beta_i F_{i-1/2})$. With closed walls ($F = 0$ on the two wall faces) the E+PE change per unit time is
$$\frac{d}{dt}(E + \mathrm{PE}) = \sum_i W_iV_i + \sum_i \phi_{c,i}\,V_i\dot\rho_i
 = g_1\sum_{f\ \rm interior} F_f\Big[\underbrace{\alpha_i + \beta_{i+1}}_{S_f} - A_f\,(r_{c,i+1} - r_{c,i})\Big],$$
using $\sum_i\phi_{c,i}V_i\dot\rho_i = \sum_f A_fF_f(\phi_{c,i+1} - \phi_{c,i})$ (cell $i$ below face $f$, cell $i+1$ above),
and $\phi_{c,i+1} - \phi_{c,i} = -g_1(r_{c,i+1} - r_{c,i})$. The interior $F_f$ are independent, so E+PE is
conserved for every flux field **iff** $S_f = A_f(r_{c,i+1} - r_{c,i})$ on every interior face. The other energy
fluxes telescope to the walls and drop out.

Equivalently: conserving two-point bookings are exactly the face form with an arbitrary face potential,
$W_iV_i = \phi_{c,i}(A_+F_+ - A_-F_-) - (A_+\phi(s_+)F_+ - A_-\phi(s_-)F_-)$, where $s_f$ (the split point of
the face's total weight between the two cells) is free. The code uses $s_f = r_f$.

**Face form (1):** $S_f = A_f(r_f - r_{c,i}) + A_f(r_{c,i+1} - r_f) = A_f(r_{c,i+1} - r_{c,i})$. Conserved
identically, for any grid (replica: `face : 0`).

**Exact weights (3):** $S_f = \frac{V_i}{h_i}(r_{c,i} - r_{i-1/2}) + \frac{V_{i+1}}{h_{i+1}}(r_{i+3/2} - r_{c,i+1})$.
For uniform $h$ and the face at $r_f = R$ the exact closed form is
$$S_f - A_f(r_{c,i+1} - r_{c,i}) = \frac{h^3\,(3R^4 - R^2h^2 + h^4/6)}{9R^4 - 3R^2h^2 + h^4}
 = \frac{h^3}{3} + O(h^5/R^2) \neq 0, \tag{5}$$
relative to the required value: $h^2/(3R^2) + h^4/(18R^4)$. Hence the E+PE change per unit time is
$$\frac{d}{dt}(E+\mathrm{PE}) = g_1\,\frac{h^3}{3}\sum_f F_f + O(h^5) = g_1\,\frac{h^2}{3}\int F\,dr + O(h^4)
\quad\text{(per steradian)},$$
nonzero whenever the column-integrated vertical mass flux density is nonzero. **The exact weights do not keep the
face form's conservation.** The defect is $O(h^2/R^2)$ of the work, the same order as the curvature terms in (2)
that they remove.

**No conserving two-point booking removes both curvature terms.** Take the conserving family with
$s_f = r_f + \lambda h^2/r_f$ (any smooth $O(h^2)$ shift has this leading form locally). Replica §1:
$$\frac{W_\lambda - g_1\langle F\rangle_V}{g_1} = \frac{h^2}{12}F'' + \Big(\lambda + \frac16\Big)\frac{h^2}{\bar r}F'
 + \Big(\lambda - \frac16\Big)\frac{h^2}{\bar r^2}F + O(h^4).$$
$\lambda = 1/6$ removes the $F$ term and doubles the $F'$ term; $\lambda = -1/6$ removes the $F'$ term and doubles
the $F$ term; no $\lambda$ removes both. For a general shift $s_f = r_f + h^2\varepsilon(r_f)$ the two conditions are
$\varepsilon = -1/(6r)$ and $\varepsilon' + 2\varepsilon/r = +1/(6r^2)$, which contradict each other
($-1/(6r)$ gives $-1/(6r^2)$). Only $\lambda = 0$ (the current code) is exact for divergence-free flow ($G$ = const),
because only a constant shift keeps $s_+ - s_- = h$.

**Wider stencils (leading order, a remark not a proof).** Any booking that equals the exact cell average up to the
common Cartesian term books the exact total work, and
$\sum_i g_1\!\int_i G\,dr + \sum_i\phi_{c,i}\dot M_i = -g_1\sum_i\int_i (r - r_{c,i})\,G'\,dr$, whose curvature part is
$+g_1\frac{h^2}{6}\int G'/r\,dr = g_1\frac{h^2}{6}\int F\,dr$ for closed walls. This is a column integral, not a sum of
local face differences, so no local telescoping correction of the *weights* cancels it (the potential is held at
$\mathrm{PE}_d$ here; §7 lifts that). The Cartesian part, $(h^2/12)[G']_{\rm walls}$, is a pure wall term, so it is no real conflict:
a wall closure plus a modified discrete PE removes it with exact conservation (§§7–8, the corrected-PE work that
`gravity-work: face` books, and an independent four-point booking that is $O(h^4)$ in every cell including the
walls). Only the face form with $\mathrm{PE}_d$ pays the trapezoid error to conserve.

## 5. Numbers (replica §2, §3)

One step, closed column, random interior $F_f$, $g_1 = -10$, column $3H$, $\rho = 1$:

| $R/H$ | $h/R$ | face: $\Delta(E{+}PE)/\sum\lvert WV\rvert$ | exact (3) | cons $\lambda=\pm1/6$ | exact, $\Delta(E{+}PE)/(E{+}PE)$ |
|---|---|---|---|---|---|
| 5 | 9.4e-3 | 3.9e-15 | **-1.3e-6** | 3.9e-15 | -4.1e-8 |
| 5 | 2.3e-3 | -9.8e-16 | **-5.7e-8** | -1.0e-15 | -2.5e-9 |
| 1000 | 4.7e-5 | -6.5e-13 | **+1.2e-10** | -6.5e-13 | +2.8e-14 |
| 1 | 9.4e-2 | 3.1e-15 | **+2.0e-4** | 3.2e-15 | +2.2e-5 |

The face form and both $\lambda$ variants stay at the floor of these sums ($\le 4\times10^{-15}$ for $R \le 5H$; $6.5\times10^{-13}$
at $R = 1000H$, where the $r^2$-weighted terms span a wide range: a cancellation floor, not shown to be round-off); the exact weights fail a 1e-14 relative gate
at every $R$ tested, by both measures (at $R = 1000H$: 2.8e-14 of $E{+}PE$, 1.2e-10 of the step's work).

Convergence against the exact $r^2$-measure cell average, $F = e^{-(r-r_0)/H}\sin(\pi(r-r_0)/4H)$:
- $h$-ladder at $R = 5H$, $\max|e - \tfrac{h^2}{12}F''|$: face 1.18e-3 → 6.27e-6 over nz 16 → 256, slope 1.75 → 1.97
  ($h^2$); exact 2.42e-5 → 3.86e-10, slope 3.96 → 4.00 ($h^4$).
- $R$-ladder at $h/H = 1/16$: face $\times R$ = 4.74e-4, 4.77e-4, 4.60e-4 at $R/H$ = 5, 50, 500 ($h^2/R$); exact at the
  $h^4$ floor 4e-8.
So the exact weights pass oracle (e) and fail oracle (c).

## 6. Two-point options (superseded by option F, §7)

| option | curvature terms in (2) | E+PE per step | divergence-free flow exact |
|---|---|---|---|
| A. current face form ($\lambda = 0$) | both present | round-off | yes |
| B. exact weights (3) | both removed | defect $g_1(h^2/3)\int F\,dr$, rel $h^2/(3R^2)$ | no ($O(h^2/R^2)$) |
| C. conserving, $\lambda = +1/6$ | $F$ term removed, $F'$ doubled | round-off | no |
| D. conserving, $\lambda = -1/6$ | $F'$ term removed, $F$ doubled | round-off | no |
| E. B plus a global E+PE fixer (as gravity-work-fixer does for cell) | both removed | round-off after the fix | no |

Hypothesis (unverified): the ablation's sign surprise (shift +1.05 / +1.19 against the predicted -1.07 / -1.23)
comes from option B's defect, which is the same order as the term removed. The check that settles it: rerun the
ablation arm with option D (conserving, $F'$ term only) and with C; if each moves the excess by its own share of
the predicted term, the defect explains the rest.

## 7. Option F: a corrected discrete PE and the work it implies

The lemma of §4 is about two-point bookings measured against $\mathrm{PE}_d = \sum_i V_i\rho_i\phi(r_{c,i})$. That
functional is itself only $O(h^2)$-accurate: an exact work conserves the exact PE, not $\mathrm{PE}_d$. Option F
changes the functional, not the weights. Checked by `optionF_replica.py` (numpy;
`python docs/derivations/optionF_replica.py` prints every number quoted; "F-replica" below; entries at the
round-off or cancellation floor, $\lesssim 10^{-12}$, differ in their leading digits between numpy builds).

**The functional.** Expand $\rho$ about the centroid inside cell $i$, $\rho = \rho(r_c) + \rho'(r - r_c) +
\tfrac12\rho''(r - r_c)^2 + \dots$. With $\phi = -g_1 r$,
$$\int_i\rho\phi\,dV = -g_1\Big[r_c M_i + \int_i\rho\,(r - r_c)\,dV\Big]
 = V_i\big[\rho_i\phi(r_c) - g_1\,\sigma_i^2\,\rho'(r_c)\big] + O(h^4 V),$$
where $M_i = \rho_iV_i$ (exact), $\sigma_i^2 = \langle(r - r_c)^2\rangle_{V_i}$ (the $r^2$-measure variance; the
third central moment is $O(h^4/R)$). Replace $\rho'$ by $s_i[\rho]$, the slope at $r_{c,i}$ of the quadratic through
the cell values at the centroids (interior: cells $i-1, i, i+1$; the two cells at each closed wall: the one-sided
stencil $0,1,2$ or $n-3,n-2,n-1$). The cell values sit $O(h^2)$ off the point values by a smooth amount, so
$s_i = \rho'(r_c) + O(h^2)$, and
$$P[\rho] = \sum_i V_i\big[\rho_i\,\phi(r_{c,i}) - g_1\,\sigma_i^2\,s_i[\rho]\big] = \int\rho\phi\,dV + O(h^4).\tag{6}$$
On a uniform centroid spacing this is a $\kappa$ form ($\kappa_i \propto \sigma_i^2/(r_{c,i+1} - r_{c,i-1})$);
the quadratic stencil keeps it $O(h^4)$ on the non-uniform centroid spacing as well. Per steradian, cancellation-free:
$\sigma^2 = (\bar r^2h^3/12 + h^5/80)/V - \delta^2$, $\delta = r_c - \bar r = \bar r h^3/(6V)$.

**The work.** Define $W_iV_i := -\dot P_i - (A_+\phi_+F_+ - A_-\phi_-F_-)$ with $\dot\rho$ from the discrete continuity
$V\dot\rho = -(A_+F_+ - A_-F_-)$. Since $P$ is linear in $\rho$,
$$W_i = W_{{\rm face},i} + g_1\,\sigma_i^2\,s_i[\dot\rho]. \tag{7}$$
The first term is (1); the second couples faces $i-3/2 \dots i+3/2$ (wall cells: the first four faces).

**Conservation (exact, by construction).** $\sum_iW_iV_i + \dot P = -\sum_i\Delta_i(A\phi F) = 0$ for closed walls,
for every flux field and every grid. E+P is conserved to round-off; E+PE$_d$ is not (it changes by
$+g_1\sum_iV_i\sigma_i^2s_i[\dot\rho]$, an $O(h^2)$ amount, which is the point: $\mathrm{PE}_d$ is the wrong target).
With F, `gravity_work_defect()` and every E+PE$_d$ check therefore measure the wrong invariant; check E+P.

**Accuracy.** The exact cell budget is $g_1V_i\langle F\rangle = -\dot{\mathrm{PE}}_i - \Delta_i(A\phi F)$, so
$W_iV_i - g_1V_i\langle F\rangle = \frac{d}{dt}(\mathrm{PE}_i - P_i) = O(h^4V)$ by (6). Both curvature terms of (2)
and the Cartesian term go: to leading order $\sigma^2s[\dot\rho] = -\tfrac{h^2}{12}(F'' + 2F'/r - 2F/r^2)$, which is
minus (2).

**Properties.**
1. Exact to $O(h^4)$, both $h^2/R$ terms and the $h^2F''/12$ term removed, interior and wall cells (table below).
2. Cartesian changes. There $\sigma^2 = h^2/12$ and (7) removes the trapezoid term $h^2F''/12$ that the face form
   keeps; E+P is conserved and E+PE$_d$ is not. Change at nz 64: $1.6\times10^{-3}$ of $\max|W|$ (interior),
   $1.9\times10^{-3}$ (wall cells). F applies on Cartesian too, under the same switch; the Cartesian case is
   written out on its own in §8.
3. Closed walls need nothing beyond the one-sided slope stencil: conservation does not use the wall fluxes' value
   (they are zero), and the wall cells converge at $h^{4.4-4.6}$. At an internal x1 block seam the
   slope is the centred one a single block takes (#303): the explicit work exchanges one row of the stage's
   $\Delta\rho$ with the x1 neighbour (`_x1_ghost_rows`; a ghost $\Delta\rho$ cannot be computed locally, because it
   needs the mass flux on the ghost cell's far face, which no block computes), and the logged `pe=` reads the
   ghost $\rho$, which the mesh exchanges after every stage. The work and $P$ are then those of one block, and
   a split column gives the one-block state: measured (`tests/test_x1_seam_split.cpp`, isothermal spherical
   column with a seam density bump, $t$ fixed), the 1-vs-2-block gap is 0 bit for bit at $n_z$ = 32, 64, 128
   (before: $8.0\times10^{-7}$, $6.3\times10^{-8}$, $2.3\times10^{-9}$), and the split E+P drift is
   $\le 4.4\times10^{-15}$. An implicit scheme refuses any x1 split and a periodic x1
   column: it requires both x1 faces of a block to be physical (`hydro.cpp:413-416`), and a block next to an x1
   seam or on a periodic x1 column has its x1 boundary functions nulled (`meshblock.cpp:142-148`). So a VIC
   block always holds the whole column between two walls, and the per-block slope of the VIC row and of its
   post-solve term (`implicit_hydro.cpp`) is the one-block slope; they never meet a seam
   (`test_x1_seam_split_vic_refuses_split` pins the refusal). **Limit: periodic x1.** The explicit work finds
   its seams through `x1_neighbors()`, which is $\{-1, -1\}$ on a periodic x1 column (`hydro.cpp:263`), so the
   internal seams of a periodic x1 column keep the one-sided slope and the split differs there from one block. A
   gravitating periodic x1 column has no single-valued potential ($-g_1x_1$ jumps at the periodic face), so the
   corrected PE is not defined there in any case. In 2-D/3-D, x2/x3 fluxes book no gravity work and move mass between columns at
   the same radius; on spherical-polar $V = \Omega(\theta,\varphi)V_r(r)$ and $\sigma^2$, $s$ depend on $r$ only, so
   $\sum_{\rm columns}P$ depends only on each shell's total mass, which they conserve: E+P stays exact (§8.6 for Cartesian).
4. Numbers below.

**Interaction with the cp3/cp5/weno5 curvature flux.** `hydro_forward.cpp` already subtracts
$g_1\,\Delta(AH)/V$, $H_f = (r_{c,i} - r_{c,i-1})(m_i - m_{i-1})/12$, zero at physical boundaries, for these
reconstructions. It is a divergence (keeps E+PE$_d$) and removes $\tfrac{h^2}{12}(r^{-2}(r^2F')')
= \tfrac{h^2}{12}(F'' + 2F'/r)$. So, on curved grids, face+H leaves $-\tfrac{h^2}{6\bar r^2}F$ in the interior,
and, because $H = 0$ at the walls, its wall cells are only first order ($h^1$, Cartesian and spherical alike).
F already removes the $F''$ term, so **F must replace H, not add to it**: F+H is back to $O(h^2)$ (double count).

**Numbers** (F-replica §1, §2, §5; $g_1 = -10$; closed walls).

| one step, random interior $F_f$ | face: $\Delta(E{+}P)/\sum\lvert WV\rvert$ | face: $\Delta(E{+}PE_d)$ | F: $\Delta(E{+}P)$ | F: $\Delta(E{+}PE_d)$ | F: $\Delta(E{+}P)/(E{+}P)$ |
|---|---|---|---|---|---|
| sph $R = 5H$ | 8.0e-3 | -3.0e-15 | **-1.3e-15** | -7.3e-3 | -2.0e-16 |
| sph $R = 1000H$ | 3.5e-3 | 5.3e-13 | **-5.7e-14** | -3.2e-3 | -4.6e-17 |
| sph $R = H$ | -2.1e-2 | 9.0e-16 | **1.3e-15** | 1.9e-2 | 1.2e-15 |
| Cartesian | 5.8e-3 | -3.5e-15 | **3.7e-15** | -5.2e-3 | 5.9e-16 |

Oracle (c) measured against $P$: F passes at round-off ($\le 1.2\times10^{-15}$ of E+P); the face form does not
conserve $P$ (it conserves $\mathrm{PE}_d$).

| $\max\lvert W - g_1\langle F\rangle\rvert/\lvert g_1\rvert$, $R = 5H$, nz 16 → 256 | interior | slope | wall cells | slope |
|---|---|---|---|---|
| face (1) | 3.2e-3 → 2.5e-5 | 1.4 → 1.95 | 5.8e-3 → 2.5e-5 | 1.9 → 2.0 |
| F (7) | 3.5e-5 → 5.4e-10 | 3.99 → 4.00 | 5.4e-5 → 1.8e-10 | 4.6 → 4.4 |
| face + H (cp3/cp5/weno5 today), nz 32 → 256 | 2.0e-5 → 3.3e-7 | 2.0 | 8.0e-3 → 1.0e-3 | 1.0 |
| F + H (double count) | 1.2e-3 → 2.5e-5 | 1.9 | 6.5e-3 → 9.9e-4 | 0.96 |
| Cartesian F | 6.9e-5 → 1.2e-9 | 4.00 | 1.3e-4 → 1.9e-9 | 4.0 |

$R$-ladder at nz 64 ($h = H/16$), F interior | wall: 1.4e-7 | 9.0e-8, 2.8e-7 | 4.8e-7, 2.9e-7 | 5.0e-7, 3.0e-7 | 5.0e-7
at $R/H$ = 5, 50, 500, 5000 (flat: no $1/R$ term left), against face $\times R$ = 1.8e-3, 2.1e-2, 0.21, 2.1 (the
face form's interior error at this $h$ is dominated by the $h^2F''/12$ term, which does not fall with $R$).

**Settling check (can a column-integrated diagnostic tell the options apart?).** Same profile,
closed walls, column sums normalised by $\sum\lvert g_1\langle F\rangle\rvert V$ (F-replica §4):

| $R/H$, nz | A face | B exact | C ($\lambda = +1/6$) | D ($\lambda = -1/6$) | F |
|---|---|---|---|---|---|
| 5, 32: work error | +1.420e-3 | +1.292e-3 | +1.420e-3 | +1.420e-3 | -7.3e-6 |
| 5, 32: PE$_d$ defect | 1e-16 | **-1.29e-4** | 3e-16 | 1e-16 | -1.43e-3 |
| 1000, 64: work error | +5.249e-4 | +5.249e-4 | +5.249e-4 | +5.249e-4 | -8.0e-7 |
| 1000, 64: PE$_d$ defect | 3e-14 | -1.3e-9 | 3e-14 | 3e-14 | -5.26e-4 |

A, C and D book the same column total (they differ only in how it is split between cells), so a column-integrated
diagnostic cannot separate them; only B differs, and by exactly its E+PE$_d$ defect. At $R = 5H$, nz 32 that defect is
9% of the column work error. So if the ablation's $(d)$ arm used the exact weights, the shift it saw includes B's
non-conservation; a C or D arm would show the column total of A. The real ablation's $\varepsilon$ harness was not
rebuilt here. F is the only option whose column total is right ($-7\times10^{-6}$ against $+1.4\times10^{-3}$): its
PE$_d$ "defect" is the $O(h^2)$ error of PE$_d$ itself, equal and opposite to A's work error.

## 8. Option F on a Cartesian $x_1$ grid, self-contained ($R \to \infty$)

This section can be read and ported without §§1–7. Every identity is checked in `optionF_replica.py` (§1, §6, §7) and in `tests/test_gravity_work_radial_exact.py` (closed columns in the code).

**8.1 Grid and notation.** One column, $x_1 = z$, cells $i = 0,\dots,n-1$ with faces $z_{i-1/2} < z_{i+1/2}$,
width $h_i = z_{i+1/2} - z_{i-1/2}$, centre $z_i = (z_{i-1/2} + z_{i+1/2})/2$ (the volume centroid, `x1v`). All
quantities are per unit horizontal area. $\rho_i$ is the cell-average total density (dry plus every condensate
and tracer that carries mass, the same sum the code's x1 mass flux uses). $F_{i+1/2}$ is the x1 mass flux through
face $i+1/2$ as the Riemann solver returns it; closed walls mean $F_{-1/2} = F_{n-1/2} = 0$. Gravity is
$g_1 = $ `grav1` $< 0$, the potential $\phi(z) = -g_1 z$. $E$ is the total energy density (the code's `IPR` row).

**8.2 Discrete continuity.** The x1 part of the mass update is
$$h_i\,\dot\rho_i = -(F_{i+1/2} - F_{i-1/2}). \tag{8.1}$$
In the code $\dot\rho_i\,dt$ is the stage's x1 density increment, `-dt * vertical_mass_div`.

**8.3 What the code books today (face form).** `hydro_forward.cpp` books, per cell,
$$W^{\rm face}_i h_i = g_1\big[(z_{i+1/2} - z_i)F_{i+1/2} + (z_i - z_{i-1/2})F_{i-1/2}\big],$$
which is the same as
$$W^{\rm face}_i h_i = -h_i\dot\rho_i\,\phi(z_i) - \big(\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2}\big),
\qquad \phi_{i\pm1/2} = \phi(z_{i\pm1/2}). \tag{8.2}$$
(Insert (8.1) and $\phi = -g_1z$: the right side is $-g_1z_i(F_+ - F_-) + g_1(z_+F_+ - z_-F_-) =
g_1[(z_+ - z_i)F_+ + (z_i - z_-)F_-]$. The code's form is `dt*(phi_cell*div - div(phi_face*F))` with
`div` $= (F_+ - F_-)/h_i$, the same expression divided by $h_i$.)
Summing (8.2) over the column, the bracket telescopes to $\phi F$ at the two walls, which is zero, so
$\sum_iW^{\rm face}_ih_i + \tfrac{d}{dt}\mathrm{PE}_d = 0$ with $\mathrm{PE}_d = \sum_ih_i\rho_i\phi(z_i)$: the face form
conserves $E + \mathrm{PE}_d$.
*Its error.* The exact cell work is $g_1\langle F\rangle_i = \tfrac{g_1}{h_i}\int_i F\,dz$. On a uniform grid
$W^{\rm face}_i = g_1(F_{i+1/2} + F_{i-1/2})/2$, the trapezoid rule, so
$W^{\rm face}_i - g_1\langle F\rangle_i = g_1\tfrac{h^2}{12}F''(z_i) + O(h^4)$ in every cell, walls included.
*With the cp3/cp5/weno5 curvature flux H* (also in `hydro_forward.cpp`):
$W^{\rm face+H}_i = W^{\rm face}_i - g_1(H_{i+1/2} - H_{i-1/2})/h_i$, $H_{i+1/2} = \tfrac{z_{i+1} - z_i}{12}(m_{i+1} - m_i)$,
$m = \rho v_1$ at the cell centres, $H = 0$ on the walls. In the interior this subtracts $g_1\tfrac{h^2}{12}F''$ and
the result is $O(h^4)$. In a wall cell only one H face acts: $H_{1/2}/h \approx \tfrac{h}{12}F'(z_{\rm wall})$ is left,
an $O(h)$ error, $\approx g_1\tfrac{h}{12}F'(z_{\rm wall})$ (F-replica §6: $1.65\times10^{-2}$ at nz 16 for $F' = \pi/4$,
$h = 1/4$, where $h F'/12 = 1.64\times10^{-2}$).

**8.4 The exact potential energy of a cell, and its discrete stand-in $P$.** Let $\rho(z)$ be smooth and expand about
$z_i$: $\rho = \rho(z_i) + \rho'(z - z_i) + \tfrac12\rho''(z - z_i)^2 + \tfrac16\rho'''(z - z_i)^3 + \dots$ Then, with
$\int_i(z - z_i)\,dz = 0$, $\int_i(z - z_i)^2dz = h^3/12$, $\int_i(z - z_i)^4dz = h^5/80$,
$$\int_i\rho\phi\,dz = -g_1\Big[z_i\int_i\rho\,dz + \int_i\rho\,(z - z_i)\,dz\Big]
= h_i\Big[\rho_i\phi(z_i) - g_1\tfrac{h_i^2}{12}\rho'(z_i)\Big] - g_1\tfrac{h_i^5}{480}\rho'''(z_i) + \dots \tag{8.3}$$
so the exact cell PE is $h_i[\rho_i\phi(z_i) - g_1\sigma_i^2\rho'(z_i)] + O(h^5)$ with $\sigma_i^2 = h_i^2/12$, the variance
of $z$ over the cell. $\mathrm{PE}_d$ drops the $\sigma^2\rho'$ term and is therefore only $O(h^2)$ accurate per unit
length; that is the whole reason the face form, which conserves $\mathrm{PE}_d$ exactly, keeps an $O(h^2)$ work error.
Replace $\rho'(z_i)$ by a slope built from the cell averages,
$$s_i[\rho] = \text{derivative at } z_i \text{ of the quadratic through } (z_k,\rho_k),\ k \in S_i,\qquad
S_i = \{i-1,i,i+1\}\ (0 < i < n-1),\ S_0 = \{0,1,2\},\ S_{n-1} = \{n-3,n-2,n-1\}. \tag{8.4}$$
On a uniform grid: $s_i = (\rho_{i+1} - \rho_{i-1})/(2h)$ inside, $s_0 = (-3\rho_0 + 4\rho_1 - \rho_2)/(2h)$,
$s_{n-1} = (3\rho_{n-1} - 4\rho_{n-2} + \rho_{n-3})/(2h)$; the non-uniform weights are in `centroid_slope`
(`src/hydro/gravity_work_radial.hpp`). Cell averages differ from point values by $\tfrac{h^2}{24}\rho''$, a smooth
amount, so $s_i = \rho'(z_i) + O(h^2)$ (interior and wall alike), and $\sigma^2 s_i = \sigma^2\rho' + O(h^4)$. Define
$$P[\rho] = \sum_i h_i\Big[\rho_i\,\phi(z_i) - g_1\,\sigma_i^2\,s_i[\rho]\Big]
 = \int\rho\phi\,dz + O(h^4). \tag{8.5}$$
$P$ is linear in $\rho$ and needs no data outside the column (no ghost cells): that is the whole wall closure.

**8.5 The work that conserves $E + P$.** Define the booked work by the same recipe as (8.2), with $P$ in place of
$\mathrm{PE}_d$: $W_ih_i := -\dot P_i - (\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2})$, where $P_i$ is the $i$-th
summand of (8.5). Because $P_i$ is linear in $\rho$, $\dot P_i = h_i[\dot\rho_i\phi(z_i) - g_1\sigma_i^2s_i[\dot\rho]]$,
and comparing with (8.2),
$$W_i = W^{\rm face}_i + g_1\,\sigma_i^2\,s_i[\dot\rho],\qquad \dot\rho \text{ from (8.1)}. \tag{8.6}$$
This is all F adds to the face work (`corrected_pe_work`; explicit: in `hydro_forward.cpp` on `-dt*vertical_mass_div`;
VIC: in `implicit_hydro.cpp` on the solve's own density change `du - du0`, rows IDN and the condensates, partly inside
the operator, §10). It changes
only the energy row: mass and momentum are untouched, so a state at rest ($F \equiv 0$, $\dot\rho = 0$) gets exactly
zero from it.
*Uniform-grid closed forms* (substitute (8.1) into (8.4); F-replica §7 checks them against the matrix form to
$8\times10^{-16}$):
$$\frac{W_i}{g_1} = \frac{F_{i+1/2} + F_{i-1/2}}{2} - \frac{F_{i+3/2} - F_{i+1/2} - F_{i-1/2} + F_{i-3/2}}{24}
\quad(0 < i < n-1),$$
$$\frac{W_0}{g_1} = \frac{19F_{1/2} - 5F_{3/2} + F_{5/2}}{24},\qquad
\frac{W_{n-1}}{g_1} = \frac{19F_{n-3/2} - 5F_{n-5/2} + F_{n-7/2}}{24}.$$
(Cell 1 and cell $n-2$ use the interior form with the wall flux $F_{\mp1/2} = 0$.)
Hand check of the wall form: for $F = z, z^2, z^3$ on the cell $[0,h]$ with $F(0) = 0$ it gives $h/2$, $h^2/3$, $h^3/4$,
the exact averages; for $z^4$ it gives $5h^4/6$ against $h^4/5$, so the wall cell is exact through cubics, error
$O(h^4)$. Interior: expanding about $z_i$, the trapezoid is $F + \tfrac{h^2}{8}F''$, the 4-face difference is
$2h^2F''$, so $W_i/g_1 = F + \tfrac{h^2}{24}F'' + O(h^4) = \langle F\rangle_i + O(h^4)$.

**8.6 Why $E + P$ is conserved (exactly, not to truncation error).** Sum (8.6)·$h_i$ over the column using the
definition in 8.5: $\sum_iW_ih_i + \dot P = -\sum_i(\phi_{i+1/2}F_{i+1/2} - \phi_{i-1/2}F_{i-1/2}) =
-(\phi_{n-1/2}F_{n-1/2} - \phi_{-1/2}F_{-1/2}) = 0$. Nothing else is used: not the stencil weights, not the grid
spacing, not the size of $F$. The energy fluxes other than gravity work telescope as before. So in exact arithmetic
$E + P$ is constant per stage; every RK stage is $u \leftarrow a u_0 + b u + c\,dt\,\dot u$ and $E + P$ is linear in $u$,
so it is constant per step too, to round-off. $E + \mathrm{PE}_d$ is not conserved any more: it changes by
$+g_1\sum_ih_i\sigma_i^2s_i[\Delta\rho]$ per stage, an $O(h^2)$ amount that is the error of $\mathrm{PE}_d$ itself.
*Horizontal directions.* In 2-D/3-D, x2/x3 fluxes move mass between columns at the same level $i$ and book no
gravity work. $\sigma_i^2$, $h_i$ and the stencil (8.4) are the same in every column and $s$ is linear, so
$\sum_{\rm columns}P = \sum_ih_i[\phi(z_i)\bar M_i - g_1\sigma_i^2s_i[\bar M]]$ with $\bar M_i$ the level's total mass,
which x2/x3 fluxes conserve (periodic or closed lateral boundaries): $E + P$ stays exact.
*Several blocks in x1.* Each block uses its own one-sided stencil at its x1 ends, so $P = \sum_{\rm blocks}P_b$; at a
shared face both blocks use the same $F$ and $\phi$, the $\phi F$ terms cancel between them, and global $E + P$ is exact.
*Rest.* The added term is proportional to $\dot\rho$; at discrete hydrostatic rest it is zero to round-off and the
momentum balance is not touched.

**8.7 Accuracy.** The exact cell budget is $g_1h_i\langle F\rangle_i = -\tfrac{d}{dt}\mathrm{PE}^{\rm exact}_i -
(\phi F)\big|_{i-1/2}^{i+1/2}$; subtract the definition of $W_i$: $W_ih_i - g_1h_i\langle F\rangle_i =
\tfrac{d}{dt}(\mathrm{PE}^{\rm exact}_i - P_i) = O(h^5)$ by (8.5), i.e. $O(h^4)$ per unit length in every cell, wall
cells included. Measured (F-replica §6, $|W - g_1\langle F\rangle|/|g_1|$, profile $F = e^{-z}\sin(\pi z/4)$ on $[0,4]$,
closed walls, nz 16 → 128):

| Cartesian | first cell (wall) | slope | last cell (wall) | slope | interior | slope |
|---|---|---|---|---|---|---|
| face + H (today, cp3/cp5/weno5) | 1.65e-2 → 2.05e-3 | 1.00 | 3.04e-4 → 3.75e-5 | 1.00 | 3.7e-5 → 1.0e-8 | 4.00 |
| F (8.6) | 1.29e-4 → 3.08e-8 | 4.02 | 1.46e-6 → 5.42e-10 | 3.96 | 6.9e-5 → 1.85e-8 | 4.00 |
| face (plm, no H) | — | 2 | — | 2 | — | 2 |

(The last cell is small because $F$ has decayed by $e^{-4}$ there.) The change F makes against today's face+H on a
Cartesian weno5 deck (F-replica §7, $\max|\Delta W|/\max|W|$): interior $1.2\times10^{-4} \to 3.2\times10^{-8}$
(slope 4.00, so the interior is unchanged to $O(h^4)$), first wall cell $6.2\times10^{-2} \to 7.7\times10^{-3}$ and
last wall cell $1.2\times10^{-3} \to 1.4\times10^{-4}$ (slope 1: the change is the removal of face+H's $O(h)$ wall error).
Against plm decks (no H) the change is the $O(h^2)$ trapezoid term in every cell.

**8.8 Porting checklist.** (i) The sum of mass rows that the x1 mass flux carries, as $\dot\rho$. (ii) $\sigma^2 = h^2/12$
per cell. (iii) The 3-point slope (8.4), one-sided at both x1 ends of every block. (iv) Add $g_1\sigma^2s[\Delta\rho]$ to
the energy row after the face work, where $\Delta\rho$ is that stage's x1 density change (implicit solvers: the solved
change, and the term inside the implicit operator, §10). (v) Turn H off (F already removes the $F''$ term; F + H is $O(h^2)$ again). (vi) Test: closed column, any
flow, $\Delta(E + P)$ per step at round-off with $P$ from (8.5).

## 9. F against the $\bar r$ potential form

A smaller alternative keeps the face form and the curvature flux but moves the potential: $\phi_c = -g_1\bar r$,
$\bar r = (r_{i+1/2} + r_{i-1/2})/2$, with the cp3/cp5/weno5 curvature flux applied to $r^2m$,
$A_fH_f = \tfrac{1}{12}(r_{c,i} - r_{c,i-1})(\bar r_i^2m_i - \bar r_{i-1}^2m_{i-1})$, zero at the walls. It is a
divergence plus a face form, so it conserves $E + P_{\bar r}$, $P_{\bar r} = \sum_iV_i\rho_i\phi(\bar r_i)$. Checked in
F-replica §8; spherical, closed walls, the profile of §5, $R = 5H$ unless noted.

| | $\bar r$ form | F |
|---|---|---|
| conserved functional, one step, random $F_f$ and $m$, / $\sum\lvert WV\rvert$ | $E + P_{\bar r}$: 5e-15 ($R = H, 5H$), $\lesssim$ 2e-12 ($R = 1000H$) | $E + P$: 6e-15, $\lesssim$ 2e-12 |
| interior max cell error, nz 64 / 256 | 1.1e-7 / 5.4e-10 ($h^{3.9}$) | 1.4e-7 / 5.4e-10 ($h^{4.0}$) |
| first (wall) cell, nz 64 / 256 | **4.0e-3 / 1.0e-3 ($h^{1.0}$)** | 9.0e-8 / 1.8e-10 ($h^{4.4}$) |
| last (wall) cell, nz 64 / 256 | **7.6e-5 / 1.9e-5 ($h^{1.0}$)** | 4.5e-9 / 2.0e-11 |
| column-summed work error, nz 64, $R = 5H$ / $1000H$ | **+3.39e-4 / +5.25e-4** | -4.7e-7 / -8.0e-7 |
| same, nz 256 | +2.1e-5 / +3.3e-5 ($h^2$) | -1.9e-9 / -3.4e-9 ($h^4$) |
| conserved PE against $\int\rho\phi\,dV$, nz 64 / 256 | $P_{\bar r}$: 3.5e-5 / 2.2e-6 ($h^2$) | $P$: 3.0e-8 / 1.3e-10 ($h^4$) |

($R = 1000H$ conservation is the cancellation floor of §5, the same for both forms. For reference, $\mathrm{PE}_d$ is
5.3e-5 / 3.3e-6 off at nz 64 / 256.)

*Why the $\bar r$ form's wall cells stay $O(h)$.* Its $H$ is zero at the walls, as today's is, so its wall cells are
face+H's: the first cell is 1.02e-3 for both at nz 256 (F-replica §5, §8b). Its column total equals the face form's at
$R = 1000H$ (+5.249e-4 for both, §7 settling table) and is close to it at $R = 5H$ (+3.39e-4 against +3.55e-4). The
interior error is $O(h^4)$, the cell-by-cell and column errors are not.

*Why the two forms cannot be equivalent.* A conservative booking conserves $E$ plus its own functional, so its column
work error is the time derivative of that functional's error. $P_{\bar r} - \int\rho\phi\,dV$ is a pure wall term,
$g_1\tfrac{h^2}{12}[r^2\rho]$ (top minus bottom) $+ O(h^4)$ (F-replica §8e: ratio 0.99998 at nz 64, 1.00000 at nz 256),
so the $\bar r$ form's $O(h^2)$ column error sits in its wall cells. Closing $H$ at the walls would break conservation
unless the PE changes as well, and the PE that makes the walls $O(h^4)$ is $P$ of (6): F is the $\bar r$ idea plus the
wall closure. F's $P$ is not of the form $\sum_iV_i\rho_i\phi_{c,i}$, so the §4 lemma (which bounds those) does not
apply to it.

*Scope.* The $\bar r$ form is spherical only and leaves Cartesian unchanged, so it cannot remove the Cartesian $O(h)$
wall error of face+H (§8.7). Adding its $H$ on top of F would remove the $F''$ term twice, as F + H does (§7). F is the
form implemented.

## 10. Every place that computes a PE

F conserves $E + P$, not $E + \mathrm{PE}_d$, so every site that computes or checks a PE has to agree with it.
Searched: `src/`, `python/`, `tests/`.

| site | PE used | with F |
|---|---|---|
| explicit face work, `hydro_forward.cpp` | $P$ | the booked work (7); the reference |
| implicit face work: the matrix's work rows and the projection/clamp work in `implicit_hydro.cpp` | $\mathrm{PE}_d$ increments | consistent: the energy row books $g_1\sigma^2\tilde s[\delta\rho_{\rm raw} - \Delta\rho_0]$ (below) and the post-solve term the rest, so the total is $g_1\sigma^2s[\Delta\rho]$ with $\Delta\rho$ the solved change `du - du0`, which includes the redistributed and projected mass; $\Delta(E + P)$ per step $\le 4\times10^{-16}$ on the implicit cases of the test |
| gravity-work fixer (`gwfix_stage` in `hydro_forward.cpp`, the implicit `epe` lambda) and its `fixgrav=` printout | $\mathrm{PE}_d$ | consistent by construction: the fixer runs only with `gravity-work: cell` (`hydro.cpp`), F acts only with `gravity-work: face` (`radial_exact_work()`); they never meet |
| cycle diagnostics `pe=` (`print_cycle_diagnostics`, `meshblock.cpp`) | $P$ | logs $P$ via `corrected_pe_work` with the same per-block one-sided slope stencil, so the logged `ie=` + `pe=` is the conserved $E + P$ (it logged $\mathrm{PE}_d$ before; the test's check 5 failed on that at $\sim2\times10^{-5}$ relative and passes at $\le 3\times10^{-14}$) |
| netCDF outputs | none | no PE field is written |
| energy oracles of the face-work tests in `test_forcing.cpp` | $P$ | each check adds the corrected-PE work the stage books (`corrected_pe_work` on its density change, x1 flux only where the deck is 3-D) to $E + \mathrm{PE}_d$, so it checks $E + P$; the cell-work tests are unchanged |
| `test_gravity_work_fixer`, `test_horizontal_flux_covariance`, `test_flux_covariance_rows` | $\mathrm{PE}_d$ (cell work) / recorded values | the fixer arms use `gravity-work: cell`; the covariance tests' recorded values are those of the face work with F; the reference oracle is `tests/test_gravity_work_radial_exact.py` ($E + P$ per step, and the logged `ie=` + `pe=` against it) |
| face-work energy oracles of `test_implicit_face_work_operator` and `test_implicit_stratified_solid` | $P$ | they measure $E + P$ on the final state ($P$ with the centroid term of §12 on a gnomonic-equiangle block) (#296) |

**The work inside the implicit operator (#296).** Booked only after the solve, the implicit part of (7) is explicit,
and it is not small where it matters: for a grid-scale density change the slope term is not $O(h^2)$ below the face
work (by the closed forms of §8.5, a quarter of it for a face-flux wave of wavelength $3h$). A discretely balanced 11.3H rest column (`test_implicit_gravity_tall_column.py`,
implicit-scheme 9) then grew from round-off: max $w$ $1.3\times10^{-7}$ m/s at step 13 for vertical acoustic Courant
197 and step 9 for 250, Cartesian and spherical-polar alike, against $\le 4.9\times10^{-9}$ with the switch off.
Dropping only the post-solve term made every rung pass at the switch-off level; dropping only the explicit term did not.
So the energy row now carries the term. With $\delta\rho$ the solve's total-mass unknown and $\Delta\rho_0$ the explicit
density change it starts from (`du0`), the row gains $-g_1\sigma_i^2\tilde s_i[\delta\rho]/\Delta t$ and its right side
$-g_1\sigma_i^2\tilde s_i[\Delta\rho_0]/\Delta t$, so it books $g_1\sigma^2\tilde s[\delta\rho - \Delta\rho_0]$, the work of
the mass the solve moves. $\tilde s$ is $s$ of (8.4) where the matrix can hold it: the interior 3-point stencil is
tridiagonal as it is, and at each end the one-sided slope's third weight is lumped onto the neighbour (it still
annihilates a constant). After the solve the term $g_1\sigma^2(s[\Delta\rho] - \tilde s[\delta\rho_{\rm raw} -
\Delta\rho_0])$ is added, $\Delta\rho$ the solved change after redistribution and clamps, so the booked total and the
E + P identity are what they were: only how much of the work the solve sees changes. Measured on a CPU build:
every rung of the tall column passes in both geometries with the switch on, max $w$ $4.69$–$4.89\times10^{-9}$, at or
below its switch-off rung, settled to $\le 7.4\times10^{-13}$ at step 40; $\Delta(E + P)$ per step
$\le 3.9\times10^{-16}$ of $E + P$ there, on the implicit cases of `test_gravity_work_radial_exact.py`, and on its
column under partial VIC (implicit-scheme 1); switch-off states and switch-on explicit states are bitwise unchanged.
The same defect failed `test_implicit_face_work_operator` with the switch on (rest column at Courant 197: max $w$
0.37 m/s; non-finite at steps 16-22 at Courant 197 and 657, full and partial VIC) and the face-work tall columns of
`test_implicit_stratified_solid` (nz 120/140/150 rolled back after 13/11/9 of 300 steps); with the term in the
operator they run to the end (rest columns $\le 4.9\times10^{-9}$ m/s at Courant 197 and 657; the stratified columns
$\le 4.4\times10^{-6}$ m/s, the level cell work reaches on them). Their remaining switch-on
"failures" were $E + \mathrm{PE}_d$ oracles, which measure the wrong invariant here: on the same runs $E + P$ closes,
e.g. the spherical implicit energy check gives $E + \mathrm{PE}_d = -0.166$ but $E + P = 3.5\times10^{-14}$
(scale 2119), and the moving tall columns drift by $\le 4.1\times10^{-15}$ in $E + P$ against up to
$3.9\times10^{-9}$ in $E + \mathrm{PE}_d$.

**The wall rows in the operator and the after-solve formula.** The lumping above left the two one-sided wall rows
of $s$ partly explicit: each reaches one cell beyond the block-tridiagonal band (cell 2 for the bottom row, cell
$n-3$ for the top row), and only the post-solve term booked that part. A rest column then still grew at large
vertical acoustic Courant numbers, from a mode in a wall cell. The operator now holds $s$ itself,
$K = g_1\,\mathrm{diag}(\sigma^2)\,s$ (plus $g_1\,\mathrm{diag}(r_c - x_{1v})$ on a gnomonic-equiangle block, §12):
its tridiagonal part goes into the blocks, and the two wall entries form a rank-2 update $UV^\top$ of the
block-tridiagonal matrix $T$ ($U$ the energy rows of cells 0 and $n-1$, $V$ the mass columns of cells 2 and $n-3$).
The solve uses the Woodbury identity on one factorisation of $T$ per column: the same factors solve $Ty = r$ and
$TZ = U$, and $x = y - Z(I + V^\top Z)^{-1}V^\top y$ (a $2\times2$ system per column). With $\Delta\rho_0$ the explicit
density change (`du0`), $\delta\rho_{\rm raw}$ the solve's total-mass change and $\Delta\rho$ the change after
redistribution, projection and clamps (`du - du0` plus $\Delta\rho_0$), the work booked in a step is
$$
W = \underbrace{K[\Delta\rho_0]}_{\text{explicit stage}}
  + \underbrace{K[\delta\rho_{\rm raw} - \Delta\rho_0]}_{\text{energy row of the solve}}
  + \underbrace{K[\Delta\rho - \Delta\rho_0] - K[\delta\rho_{\rm raw} - \Delta\rho_0]}_{\text{after the solve}}
  = K[\Delta\rho],
$$
so $E + P$ closes exactly, as before; the after-solve term is now non-zero only where redistribution, projection or a
clamp changes the solved density ($\Delta\rho \ne \delta\rho_{\rm raw}$), in every cell including the walls. On the
tall column of `test_implicit_gravity_tall_column.py` (nz 23 and 45, Cartesian and spherical-polar, horizontally
uniform perturbations, one-step finite-difference Jacobian) the largest growth factor stays $\le 1 + 2.5\times10^{-6}$
up to $N\Delta t = 600$, where the lumped rows had a wall mode from $N\Delta t \approx 119$ (nz 23) and $149$ (nz 45);
the Woodbury solve matches a dense solve of the full matrix to $\le 4.7\times10^{-12}$, and its cost is one more
pair of back-substitutions per column.

## 11. The default reference at the walls, F on by default with `gravity-work: face`, the remaining error, and the x1 mass-flux covariance

### 11.1 The wall closure of the default x1 reference

With `SNAP_WB_REF4` off, the default well-balanced reference repeated the wall cell's $r = \rho/p$ in the ghosts
past a clamped physical wall, which makes the face density reference first order at the three faces next to each
wall. It now continues $r$ past the wall (`wb-ref-wall.md` §4): linearly, $r_{-k} = r_0 + k(r_0 - r_1)$, where $r$
rises towards the wall ($r_1 \le r_0$), and ln-linearly, $r_{-k} = r_0(r_0/r_1)^k$, where it falls ($r_1 > r_0$).
Neither form alone is bounded:
- linear in $r$ everywhere reaches zero, $r_{-3} = r_0(1 - 3d)$ with $d = (r_1 - r_0)/r_0$, and goes negative past
  $d = 1/3$; `test_straka_redo` (CFL 1.6, nx1 64) then fails at 31.67 s with $p < 0$ below the top wall;
- ln-linear everywhere grows without bound next to a nearly empty cell: in `test_face_floor`'s unresolved column it
  gives a dipped-face mass flux of $3.36\times10^{-7}$, against $2.83\times10^{-8}$ for the linear form.

Taking per column the branch that stays closer to $r_0$ keeps the continuation in $[r_0(r_0/r_1)^k, r_0(1+k)]$. The
observed order at faces 1-2 of both walls is 1.81-2.12 (nz 32 to 64), and every interior face and cell is bit for
bit as before. The tests that pinned values of the old wall reference were re-pinned with it.

### 11.2 Why F is on by default with `gravity-work: face`

`gravity-work: face` is the default x1 gravity-work form (const_gravity.cpp:28, forcing.hpp:61); `cell` (with its global fixer) must now be set explicitly.

With `gravity-work: face` on a Cartesian, spherical-polar or gnomonic-equiangle grid and $g_1 \ne 0$, F is always on;
there is no switch to turn it off. The reasons for making it the default are that the plain face work it replaces
is first order in the two x1 wall cells (§8) while F is $O(h^4)$ there and in the interior, that F conserves the
exact discrete energy $E + P$ to round-off (§7), and that with F on the cp3/cp5/weno5 curvature flux has nothing
left to correct (F already removes the $F''$ term; F + H is $O(h^2)$ again, §7), so that flux and the wall-cell
variant `face-wallc` are removed. The first-order wall error of plain face work shows:
- in the dry Cartesian onset box with `dynamics/flux-covariance` and `SNAP_WB_REF4` on (cell-average initial state),
  the one-step $\varepsilon_{\rm eff}\,n_z^2$ at $n_z$ 16, 32, 64 is $+0.0306$, $+0.0158$, $+0.0080$ without F and
  $-0.00092$, $-0.00020$, $-0.00004$ with F;
- on a coarse polytrope the kinetic energy runs away to $3.1\times10^{-4}$ without F, $5.4\times10^{-5}$ with F, and
  $8.5\times10^{-5}$ with `gravity-work: cell`.

On a gnomonic-equiangle (cubed-sphere) grid F carries one more term, because `x1v` is not the $r^2$ centroid
there (§12). Other grids (cylindrical) have no form for F: there the setup warns once and the plain face work is
kept. `gravity-work: cell` is unchanged bit for bit. The energy oracles of §10 measure $E + P$.

### 11.3 The remaining error

With the wall closure and F, the convective onset (linear growth) rate keeps a second-order truncation error
that face and cell work share, so it does not separate the two forms.

The deck is the T1L box, 1 H tall with superadiabatic excess $\varepsilon = 10^{-3}$, started from the point-value
initial state; the entry is the relative error of the fitted onset growth rate against the linear eigenvalue rate,
so $\Delta z = H/n_z$. The runs used 37dce4e plus the x1 covariance code that this branch commits as e04b783 (the
same hunks). All of them have `SNAP_WB_REF4` on, where the default-reference wall closure of §11.1 is a bitwise
no-op, and all are explicit, where f256ab5 (the implicit path) changes nothing. So the numbers hold at this head,
but they do not exercise the default-reference wall closure; its own order test is `test_hydro_ref_x1`
(`wb-ref-wall.md`). The switches are named per column: cov = `dynamics/flux-covariance`, W = `SNAP_WB_REF4`, F,
and x1 = `dynamics/x1-mass-covariance` (cov and x1 were the environment switches `SNAP_FLUX_COVARIANCE` and
`SNAP_X1_MASS_COVARIANCE` when these runs were made).
The runs are jobs 644220 (nz 16, and every column without x1), 644132 (nz 32 with x1) and 644133 (nz 64 with x1):

| $n_z$ | face cov+W+F+x1 | face cov+W+F | cell cov+W+x1 | cell cov+W |
|---|---|---|---|---|
| 16 | $+1.4423\times10^{-2}$ | $+1.4421\times10^{-2}$ | $+1.1004\times10^{-2}$ | $-1.662\times10^{-3}$ |
| 32 | $+3.8163\times10^{-3}$ | $+3.8155\times10^{-3}$ | $+3.5997\times10^{-3}$ | $+1.239\times10^{-3}$ |
| 64 | $+9.6574\times10^{-4}$ | $+9.6548\times10^{-4}$ | $+9.5205\times10^{-4}$ | $+4.144\times10^{-4}$ |

With x1 on, face and cell share this error (§11.4), so the cell cov+W+x1 column stands for it. Its least-squares fit
to the three points is $4.518\,\Delta z^2 - 27.22\,\Delta z^3$ ($\Delta z$ in H), with residuals
$-1.5\times10^{-6}$, $+1.8\times10^{-5}$, $-4.7\times10^{-5}$ at nz 16, 32, 64. The $\Delta z^2$ term carries
the error: at nz 64 it is 1.16 times the measured value. With three points, $c_3$ also absorbs the higher orders.
The observed orders are 1.61 and 1.92 for cell cov+W+x1, and 1.92 and 1.98 for face cov+W+F+x1.
So a second-order truncation error remains with either form. nz 128 is not in this table.

### 11.4 The x1 mass-flux covariance (`dynamics/x1-mass-covariance`, default off)

The x1 reconstruction takes the cell velocity $w_c = \overline{m_1}/\bar\rho$ as if it were the cell average
$\bar w$. Expanding the product over a cell of width $\Delta z$,
$$
\overline{\rho w} = \bar\rho\,\bar w + \frac{\Delta z^2}{12}\,\rho_z w_z + O(\Delta z^4),
\qquad
w_c = \bar w + \frac{\Delta z^2}{12}\,\frac{\rho_z w_z}{\bar\rho} + O(\Delta z^4),
$$
so in a stratified column the face mass flux carries an extra $\frac{\Delta z^2}{12}\rho_z w_z$, the x1 analogue of
the horizontal term that `dynamics/flux-covariance` removes. With the key on, the cell velocity handed to the
reconstruction is $w_c - \frac{\Delta z^2}{12}\rho_1 w_1/\bar\rho$, with $\rho_1$ and $w_1$ centred differences in
$x_1$, and the full cell velocity is restored after the reconstruction. Two details at a reflecting wall:
- the even ghost density is not the stratified column, so $\rho_1$ in the first and last interior cells is the
  one-sided second-order difference $\pm(-3\rho_0 + 4\rho_{\pm1} - \rho_{\pm2})/(2h)$ over interior cells, upper
  signs at the lower wall and $h$ the cell-centre spacing;
- the ghost velocities are refilled as the odd mirror of the corrected interior, so the wall face still carries
  zero mass flux.

At rest $w_1 = 0$ and the correction vanishes, so rest balance is unchanged; with the switch off nothing changes.
The correction uses the total density, so it removes the covariance from the total mass flux only. Each species
flux is that face mass flux times the species mass fraction reconstructed from the cell values (LMARS:
$\bar u\,\rho\,q$ from the upwind side), so where $q$ varies along $x_1$ the split between species keeps an
$O(\Delta z^2)$ error, which this switch does not remove.

The onset test uses the deck, build and runs of §11.3: T1L 1 H with $\varepsilon = 10^{-3}$ from the point-value
initial state, on the build of §11.3 with the switch, in jobs 644220, 644132 and 644133. These are the measured values:
- cell cov+W+x1 is $+1.1004\times10^{-2}$, $+3.5997\times10^{-3}$ and $+9.5205\times10^{-4}$ at nz 16, 32, 64.
  Without x1 (job 644220) the values are $-1.662\times10^{-3}$, $+1.239\times10^{-3}$ and $+4.144\times10^{-4}$. The
  values predicted from the face-flux error, $+2.63$ to $+4.03\times10^{-3}$ at nz 32 and $+0.80$ to
  $+1.11\times10^{-3}$ at nz 64, contain the measured ones; there is no prediction at nz 16.
- face cov+W+F+x1 is $+1.4423\times10^{-2}$, $+3.8163\times10^{-3}$ and $+9.6574\times10^{-4}$. The switch moves face
  cov+W+F by only $+2.1\times10^{-6}$, $+7.5\times10^{-7}$ and $+2.7\times10^{-7}$, because F's work already carries
  this covariance.
- face minus cell is $+3.418\times10^{-3}$, $+2.165\times10^{-4}$ and $+1.369\times10^{-5}$. That falls by 15.79
  and 15.81 per doubling of $n_z$, an order of 3.98, so the two forms agree through $\Delta z^3$. Without x1 the gap
  is $+1.608\times10^{-2}$, $+2.577\times10^{-3}$ and $+5.511\times10^{-4}$, falling by only 6.2 and 4.7.

## 12. Option F on a gnomonic-equiangle (cubed-sphere) grid: the centroid term (#300)

Base: e894700. Every closed form and number in §§12.1–12.7 is checked or printed by
`cubed_sphere_centroid_term.py` (sympy + numpy; `python docs/derivations/cubed_sphere_centroid_term.py`, "CS-replica"
below). The code numbers in §12.9 come from the ctests named there.

### 12.1 Geometry

On a gnomonic-equiangle block $x_1$ is the radius. `face_area1` is $\Omega_{kj}\,r^2$ and `cell_volume` is
$\Omega_{kj}\,(r_+^3 - r_-^3)/3$ (`src/coord/gnomonic_equiangle.cpp`), with the same exact gnomonic solid angle
$\Omega_{kj}$ in both, constant along $x_1$. So, per steradian, $A$, $V$, $\sigma^2$ and $r_c$ are those of §1, and
$\Omega_{kj}$ cancels per column as it does on spherical-polar. Gravity is along $x_1$ (`const-gravity` adds
`grav1` to the $x_1$ momentum), and $x_1$ is metrically orthogonal to the panel coordinates ($g_{11} = 1$, only
$g_{23} \ne 0$), so $\phi = -g_1 r$.

The one difference from §1: `x1v` is the face midpoint, $x_{1v} = \bar r = (r_+ + r_-)/2$
(`gnomonic_equiangle.cpp`, `x1v = 0.5 (x1f_i + x1f_{i+1})`), not the $r^2$ centroid. With $h = r_+ - r_-$ and
$V_r = (r_+^3 - r_-^3)/3 = \bar r^2h + h^3/12$, the centroid sits at
$$
r_c = \bar r + \delta,\qquad
\delta = \frac{\bar r\,h^3}{6\,V_r} = \frac{h^2}{6\bar r} - \frac{h^4}{72\,\bar r^3} + O(h^6/\bar r^5)
\tag{12.1}
$$
(CS-replica §1). On spherical-polar $\delta$ is zero by construction (`x1v` $= r_c$); on the cubed sphere it is
$O(h^2/R)$.

### 12.2 The exact cell potential energy when `x1v` is the midpoint

The expansion of §7 is about the centroid and does not depend on where `x1v` is:
$$
\int_i\rho\phi\,dV = -g_1\Big[r_c\,M_i + \int_i\rho\,(r - r_c)\,dV\Big]
 = V_i\big[\rho_i\,\phi(r_{c,i}) - g_1\,\sigma_i^2\,\rho'(r_{c,i})\big] + O(h^4V),
$$
with $\sigma_i^2 = \langle(r - r_c)^2\rangle_{V_i}$ (the $r^2$-measure variance about $r_c$; `x1_variance` with the
radial measure computes exactly this, whatever `x1v` is). Now write the potential at `x1v`:
$\phi(r_c) = \phi(x_{1v} + \delta) = \phi(x_{1v}) - g_1\delta$, exactly, because $\phi$ is linear. So
$$
\int_i\rho\phi\,dV = V_i\big[\rho_i\,\phi(x_{1v,i}) - g_1\,\delta_i\,\rho_i - g_1\,\sigma_i^2\,\rho'(r_{c,i})\big] + O(h^4V).
\tag{12.2}
$$
CS-replica §1 checks (12.2) with sympy for a quartic $\rho = \sum_k c_k(r - \bar r)^k$: the residual is
$g_1h^4(4c_2 - 3\bar r c_3)/(240\bar r)$ per unit volume, $O(h^4)$. Without the $\delta$ term the residual is
$-g_1c_0h^2/(6\bar r)$, i.e. $-g_1\delta\rho$ to leading order, $O(h^2/R)$.

### 12.3 The slope on the `x1v` nodes

The code takes $s_i[\rho]$, the slope at $x_{1v,i}$ of the quadratic through $(x_{1v,k}, \rho_k)$
(`centroid_slope`). The cell values are the $r^2$ means, which sit at $r_c$ up to a smooth $O(h^2)$ amount. If every
node were shifted by the same $\delta$, the quadratic through the shifted nodes would be the translate of the
quadratic through the centroids, and its slope at $x_{1v,i}$ would equal the centroid quadratic's slope at
$r_{c,i}$ exactly. $\delta$ varies across the 3-cell stencil by $O(h^3/R^2)$, so the two slopes differ by a relative
$O(h^2/R^2)$, and $\sigma^2 s$ by $O(h^4/R^2)$ absolute: below the $O(h^4)$ of (12.2). Measured (CS-replica §5,
$\rho = e^{-(r - r_0)}$, relative to $\max|\sigma^2 s|$): $3.9\times10^{-4} \to 6.5\times10^{-6}$ at $R = 5H$ and
$2.9\times10^{-6} \to 4.5\times10^{-8}$ at $R = 60H$ over $n_z$ 16 → 128, order 2.0 relative to a term that is
itself $O(h^2)$, with a coefficient falling as $1/R^2$. So $s$ stays on the `x1v` nodes, as on the other grids.

### 12.4 The functional

From (12.2) and §12.3,
$$
P[\rho] = \sum_i V_i\big[\rho_i\,\phi(x_{1v,i}) - g_1\,\delta_i\,\rho_i - g_1\,\sigma_i^2\,s_i[\rho]\big]
 = \int\rho\phi\,dV + O(h^4).
\tag{12.3}
$$
It is §7's (6) with one more diagonal, linear term. On spherical-polar and Cartesian grids $\delta \equiv 0$ and (12.3)
is (6) itself.

### 12.5 The work

Define the booked work as in §7 and §8.5, $W_iV_i := -\dot P_i - (A_+\phi_+F_+ - A_-\phi_-F_-)$, with $\dot\rho$ from
$V\dot\rho = -(A_+F_+ - A_-F_-)$. $P_i$ is linear in $\rho$, so
$$
\dot P_i = V_i\big[\dot\rho_i\,\phi(x_{1v,i}) - g_1\,\delta_i\,\dot\rho_i - g_1\,\sigma_i^2\,s_i[\dot\rho]\big],
\tag{12.4}
$$
and since the face form with $\phi_c = \phi(x_{1v})$ is $W_{\rm face}V = -V\dot\rho\,\phi(x_{1v}) - \Delta(A\phi F)$ (§2
with $r_c \to x_{1v}$; the code's `phi_cell = -grav1 * x1v`),
$$
\boxed{\;W_i = W_{{\rm face},i} + g_1\,\sigma_i^2\,s_i[\dot\rho] + g_1\,\delta_i\,\dot\rho_i\;}
\tag{12.5}
$$
The new term is diagonal: it uses only the cell's own density change.

### 12.5a The change of $P$ equals the booked work, step by step

Per RK stage, with $\Delta\rho_i$ the stage's x1 density change (the code's `-dt * vertical_mass_div`):

1. **The functional, in three parts.** From (12.3),
   $P_i[\rho] = \underbrace{V_i\rho_i\phi(x_{1v,i})}_{P^d_i} \;\underbrace{-\,g_1V_i\delta_i\rho_i}_{P^\delta_i}\;
   \underbrace{-\,g_1V_i\sigma_i^2s_i[\rho]}_{P^\sigma_i}$.
2. **Its change is exact, not a truncation.** $P_i$ is linear in $\rho$ ($\phi$, $\delta$, $\sigma^2$, $V$ depend on the grid
   only, and $s$ is a fixed linear stencil), so
   $\Delta P_i = V_i\big[\Delta\rho_i\,\phi(x_{1v,i}) - g_1\delta_i\,\Delta\rho_i - g_1\sigma_i^2\,s_i[\Delta\rho]\big]$.
3. **Continuity.** $V_i\Delta\rho_i = -\Delta t\,\big[(AF)_{i+1/2} - (AF)_{i-1/2}\big] \equiv -\Delta t\,\Delta_i(AF)$.
4. **The face part that the code books.** `face_gravity_work = dt * (phi_cell * div - div(phi_face * F))` with
   `div` $= \Delta_i(AF)/V_i = -\Delta\rho_i/\Delta t$ and `phi_cell` $= \phi(x_{1v})$, so
   $\Delta t\,W^{\rm face}_iV_i = -V_i\Delta\rho_i\,\phi(x_{1v,i}) - \Delta t\,\Delta_i(A\phi F)$.
5. **The corrected part that the code books.** `corrected_pe_work(drho = Δρ, ..., radial_mid)` returns
   $g_1\sigma_i^2s_i[\Delta\rho] + g_1\delta_i\Delta\rho_i$ per unit volume, so
   $\Delta t\,W_iV_i = \Delta t\,W^{\rm face}_iV_i + g_1V_i\sigma_i^2s_i[\Delta\rho] + g_1V_i\delta_i\Delta\rho_i$.
6. **Add 2 and 5, term by term.**
   $\Delta t\,W_iV_i + \Delta P_i = -\Delta t\,\Delta_i(A\phi F)
   + \underbrace{\big[-V\Delta\rho\,\phi(x_{1v}) + V\Delta\rho\,\phi(x_{1v})\big]_i}_{P^d\text{ vs face form}}
   + \underbrace{\big[g_1V\delta\Delta\rho - g_1V\delta\Delta\rho\big]_i}_{P^\delta\text{ vs centroid term}}
   + \underbrace{\big[g_1V\sigma^2s[\Delta\rho] - g_1V\sigma^2s[\Delta\rho]\big]_i}_{P^\sigma\text{ vs slope term}}
   = -\Delta t\,\Delta_i(A\phi F).$
   The centroid term of the work cancels exactly the change of $P^\delta$; with it in the work but not in $P$ (or the
   reverse) the bracket would leave $\pm g_1V_i\delta_i\Delta\rho_i$, an $O(h^2/R)$ defect per cell.
7. **Sum over the column.** $\sum_i\Delta_i(A\phi F)$ telescopes to the two x1 wall faces, where $F = 0$, so the
   booked gravity work and the change of $P$ cancel: $\Delta E_{\rm grav} + \Delta P = 0$ per stage, hence per step.

**The implicit (VIC) path.** The explicit part books step 5 for its own change $\Delta\rho_0$. The energy row books
$g_1(\sigma^2\tilde s + \delta)[\delta\rho_{\rm raw} - \Delta\rho_0]$ ($\tilde s$ the tridiagonal stencil; $\delta$ is
diagonal, so it is not lumped). The post-solve term books $g_1(\sigma^2s + \delta)[\Delta\rho_{\rm moved}]$ minus that
row's $g_1(\sigma^2\tilde s + \delta)[\delta\rho_{\rm raw} - \Delta\rho_0]$, with $\Delta\rho_{\rm moved}$ = `du - du0`.
The sum is $g_1(\sigma^2s + \delta)[\Delta\rho_0 + \Delta\rho_{\rm moved}]$, step 5 for the stage's total change, and the
face part is the face form of the total flux, so steps 6–7 hold unchanged.

**Where the term sits.** `hydro_forward.cpp` line numbers are those of the current tree; the other sites keep the
numbers of the tree this section was written against (the second number, where given, was that branch's head).

| site | booked work | $P$ |
|---|---|---|
| shared helper | `src/hydro/gravity_work_radial.hpp:100` (head `:101`): `work += grav1 * x1_centroid_offset(...) * drho` (`radial_mid` only); $\delta$ itself at `:76-84` (head `:77-85`) | the same helper, called on $\rho$ instead of $\Delta\rho$ |
| explicit face work | `src/hydro/hydro_forward.cpp:918-919`: `corrected_pe_work(drho, ..., measure)` with `drho = -dt * vertical_mass_div` (`:910`), or `corrected_pe_work_seam` at `:929-930` across an x1 seam; face part `phi_cell = -grav1 * x1v` at `:886` | |
| implicit energy row | `src/implicit/implicit_hydro.cpp:315`: `rx_mid += grav1 * x1_centroid_offset(...)`; into the matrix at `:333`, the right side at `:339` | |
| #296 post-solve remainder | `src/implicit/implicit_hydro.cpp:466-467`: `corrected_pe_work(moved, ..., x1_measure(type))`, minus `rx_tri(raw - rx_mass0)` at `:473`, added at `:476-477` | |
| logged `pe=` | | `src/mesh/meshblock.cpp:1047` ($P^d$, $\phi(x_{1v})$) and `:1056-1058`: `pe -= corrected_pe_work_seam(rho, ..., x1_measure(type), below >= 0, above >= 0)` $= P^d + P^\sigma + P^\delta$ |

(The logged `pe=` is computed in `print_cycle_diagnostics`, `src/mesh/meshblock.cpp`; `src/hydro/hydro.hpp:172-180`
declares `radial_exact_work()`, not the diagnostic.)

### 12.6 Conservation

Sum (12.5)$\cdot V_i$ over a closed column with the definition of 12.5:
$\sum_iW_iV_i + \dot P = -\sum_i\Delta_i(A\phi F) = -(A\phi F)\big|_{\rm walls} = 0$. As in §8.6 nothing else is used,
so $E + P$ is conserved per stage and per RK step to round-off, for any $\delta$. The same holds with $\delta$ dropped
from both $P$ and $W$ (the gate lifted with no other change conserves its own functional): CS-replica §2, one
step with random interior fluxes, $R/H$ = 5, 60, 1000:

| $R/H$ | with $\delta$, against (12.3) | without $\delta$, against $P$ without $\delta$ | with $\delta$, against $P$ without $\delta$ |
|---|---|---|---|
| 5 | $-3.9\times10^{-15}$ | $-7.5\times10^{-15}$ | $+5.8\times10^{-7}$ |
| 60 | $+5.1\times10^{-14}$ | $+2.3\times10^{-14}$ | $-2.2\times10^{-8}$ |
| 1000 | $-3.5\times10^{-13}$ | $-6.3\times10^{-13}$ | $-2.6\times10^{-11}$ |

(entries are $(\Delta E + \Delta P)/\sum|WV|$; the $R = 1000H$ floor is the cancellation floor of §5). So
conservation cannot tell the two apart; accuracy can.

*Lateral directions and panel seams.* $\sigma^2$, $\delta$, $s$ and $V_r$ depend on $r$ only and $V = \Omega_{kj}V_r$,
so $\sum_{\rm columns}P = \sum_iV_{r,i}[\phi(x_{1v,i})\bar M_i - g_1\delta_i\bar M_i - g_1\sigma_i^2s_i[\bar M]]$, with
$\bar M_i = \sum_{kj}\Omega_{kj}\rho_{kji}$ the shell's mass per unit $V_r$. The x2/x3 fluxes book no gravity work
and move mass at fixed radius, so $E + P$ stays exact as long as they conserve each shell's mass. They do on
the six panels, whose seam fluxes are single-valued (#222): with lateral flow across the seams $E + P$ closes to
$1.9\times10^{-15}$ per step (§12.10). A lone face block with reflecting x2/x3 walls does not: under lateral flow
it loses mass at up to $\sim6\times10^{-11}$ per step, with F or without it (#301, present at e894700), so the
single-block cases of `test_gravity_work_radial_exact.py` use radial flow only and the lateral directions are tested
on six panels. $x_1$ is never split on the cubed sphere (the layout requires one block in $x_1$), so the
one-sided slope sits only at the two physical walls and the seam limitation of §7 does not arise.

### 12.7 Accuracy

The exact cell budget is $g_1V_i\langle F\rangle_i = -\tfrac{d}{dt}\int_i\rho\phi\,dV - \Delta_i(A\phi F)$; subtract the
definition of $W_i$:
$$
W_iV_i - g_1V_i\langle F\rangle_i = \frac{d}{dt}\Big(\int_i\rho\phi\,dV - P_i\Big) = O(h^4V)
\tag{12.6}
$$
by (12.3), in every cell, wall cells included. Without the $\delta$ term ("gate lifted only") the same identity
gives an error of $-g_1\delta\dot\rho = g_1\frac{h^2}{6\bar r}\big(F' + \frac{2F}{r}\big) + O(h^4)$
(using $\dot\rho = -r^{-2}(r^2F)'$): second order with a $1/R$ coefficient, in every cell. The plain face form on
this grid, $\phi_c = \phi(\bar r)$, is (2) shifted by the same amount: its error is
$\frac{h^2}{12}F'' + \frac{h^2}{3r}F' + \frac{h^2}{6r^2}F$; the cp3/cp5/weno5 curvature flux H removes
$\frac{h^2}{12}(F'' + 2F'/r)$ in the interior, which leaves $\frac{h^2}{6r}F' + \frac{h^2}{6r^2}F$ there, and
H $= 0$ at the walls leaves the wall cells first order (§7). Measured against the exact $r^2$ cell average
(CS-replica §3; $F = e^{-(r - r_0)/H}\sin(\pi(r - r_0)/4H)$, closed walls, $\max|W - g_1\langle F\rangle|/|g_1|$,
"wall" = the two cells at each wall):

| $R = 60H$ (the six-panel shell of #300) | $n_z$ 16 | 32 | 64 | 128 | 256 | observed orders |
|---|---|---|---|---|---|---|
| face + H (today, weno5), wall | 1.64e-2 | 8.16e-3 | 4.08e-3 | 2.04e-3 | 1.02e-3 | 1.00, 1.00, 1.00, 1.00 |
| face + H, interior | 5.62e-5 | 1.89e-5 | 6.25e-6 | 1.82e-6 | 4.93e-7 | 1.57, 1.60, 1.78, 1.89 |
| gate lifted only, wall | 1.27e-4 | 2.73e-5 | 7.53e-6 | 2.04e-6 | 5.23e-7 | 2.22, 1.86, 1.89, 1.96 |
| gate lifted only, interior | 8.80e-5 | 2.12e-5 | 6.40e-6 | 1.83e-6 | 4.93e-7 | 2.06, 1.73, 1.80, 1.89 |
| (12.5), wall | 1.25e-4 | 7.82e-6 | 4.80e-7 | 2.95e-8 | 1.83e-9 | 4.00, 4.03, 4.02, 4.01 |
| (12.5), interior | 6.58e-5 | 4.48e-6 | 2.81e-7 | 1.76e-8 | 1.10e-9 | 3.88, 4.00, 4.00, 4.00 |

At $R = 5H$ the same ordering holds: face + H wall order 0.92–0.99, gate lifted only 1.77–1.98 (wall) and
0.93–1.89 (interior), (12.5) 4.35–4.57 (wall) and 3.99–4.00 (interior). R-ladder at $n_z$ 64 (CS-replica §4): the
gate-lifted interior error times $R$ is $3.81\times10^{-4}$, $3.81\times10^{-4}$, $5.13\times10^{-4}$ at
$R/H$ = 5, 50, 500 (the $h^2/R$ term, until the $h^4$ floor takes over), while (12.5) stays at
$1.3$–$3.0\times10^{-7}$ (interior) and $1.0$–$5.0\times10^{-7}$ (wall) for $R/H$ = 5 to 5000.

### 12.8 The implicit energy row (#296)

§10's coupling puts $-g_1\sigma^2\tilde s[\delta\rho]/\Delta t$ in the energy row's total-mass column and books the
remainder after the solve. (12.5) adds $g_1\delta_i\dot\rho_i$, which is diagonal, so the matrix holds it exactly (no
lumping): the diagonal weight `rx_mid` gains $g_1\delta_i$, and the row then books
$g_1(\sigma^2\tilde s + \delta)[\delta\rho - \Delta\rho_0]$. The post-solve term calls `corrected_pe_work`, which
carries the $\delta$ term, minus the same tridiagonal operator, so the total is
$g_1(\sigma^2 s + \delta)[\Delta\rho]$ with $\Delta\rho$ the solved change, and $E + P$ closes over the explicit and
implicit parts as before.

### 12.9 The code

- `src/hydro/gravity_work_radial.hpp`: `X1Measure` (`plain` Cartesian, `radial` spherical-polar, `radial_mid`
  gnomonic-equiangle, `none` otherwise) and `x1_measure(type)`, which replace the four `type == "spherical-polar"`
  tests; `x1_centroid_offset(x1f, x1v)`, $\delta$ written about the midpoint, $\bar r h^3/(6V_r) + (\bar r - x_{1v})$,
  so it does not cancel at large $r$; `corrected_pe_work(..., X1Measure)` adds $g_1\delta\,\Delta\rho$ on `radial_mid`
  only, so Cartesian and spherical-polar evaluate exactly the expressions they did before.
- `src/hydro/hydro.cpp`: `radial_exact_work()` admits every grid with a measure; the warn-once fires only where
  there is none (cylindrical).
- The explicit face work (`hydro_forward.cpp`), the post-solve implicit term (`implicit_hydro.cpp`) and the logged
  `pe=` (`meshblock.cpp`) call `corrected_pe_work` with `x1_measure(type)`; the implicit row adds $g_1\delta$ to
  `rx_mid` (§12.8).

### 12.10 Measured in the code (CPU)

The deck is `tests/test_cubed_sphere_cell_volume.yaml` (six panels, shell $6.0$–$6.4\times10^6$ m, weno5, rk3,
reflecting x1 walls, `gravity-work: face`, switch on), with $8\times8$ columns per panel. Every number below compares
a CPU build of this branch with one of e894700.

**Work error against the exact cell average, one RK stage** from radial flow $u_1 = 0.05c_s\sin(\pi z/L)$. The
mass flow per face is recovered from the stage's own density change, $(AF)_{i+1/2} = (AF)_{i-1/2} - V_i\Delta\rho_i/\Delta t$
(closed walls; it agrees with the code's x1 mass flux to $2.4\times10^{-14}$). The reference is $g_1\int_i AF\,dr/V_i$, with $AF$ the quintic
through the six nearest faces ($O(h^6)$). Explicit: the booked work is read from the code's energy change,
$W = \Delta E/\Delta t + {\rm div}_1(F_{1,E})$. VIC: the energy row cannot be separated from the solve's other terms,
so $W$ is the per-cell budget of §12.5a evaluated on the solved $\Delta\rho$ (e894700: $\mathrm{PE}_d$ plus the curvature
flux H), at the explicit $\Delta t$. Error = $\max|W - W_{\rm exact}|/\max|W_{\rm exact}|$:

| | $n_z$ 16 | 32 | 64 | 128 | observed orders |
|---|---|---|---|---|---|
| explicit, e894700, wall cells | 6.18e-2 | 3.09e-2 | 1.54e-2 | 7.72e-3 | 1.00, 1.00, 1.00 |
| explicit, e894700, interior | 1.70e-4 | 6.92e-5 | 2.35e-5 | 6.88e-6 | 1.30, 1.56, 1.77 |
| explicit, this branch, wall cells | 4.90e-4 | 3.00e-5 | 1.82e-6 | 1.12e-7 | 4.03, 4.04, 4.03 |
| explicit, this branch, interior | 2.47e-4 | 1.69e-5 | 1.06e-6 | 6.63e-8 | 3.87, 3.99, 4.00 |
| VIC, e894700, wall cells | 5.78e-2 | 2.96e-2 | 1.51e-2 | 7.63e-3 | 0.96, 0.97, 0.98 |
| VIC, e894700, interior | 7.98e-4 | 2.69e-4 | 7.29e-5 | 1.88e-5 | 1.57, 1.89, 1.95 |
| VIC, this branch, wall cells | 5.62e-4 | 2.17e-4 | 5.81e-5 | 1.47e-5 | 1.37, 1.90, 1.99 |
| VIC, this branch, interior | 1.60e-4 | 4.54e-5 | 1.42e-5 | 3.73e-6 | 1.82, 1.67, 1.93 |

The explicit rows are the booked work itself: first order at the walls before, fourth order everywhere after. The
VIC rows are second order after the change. This is not the booking. The solve's effective mass flux is not smooth
at the grid scale: its fifth difference, relative to $\max|AF|$, falls at order 1.3–2.0 from $n_z$ 32 to 128 on both
builds (explicit: order 5.0), so no booking can integrate it to fourth order. The booking is the formula of
§12.5a on that flux. Within the VIC rows, the change still removes the first-order wall error.

**E + P from the code's own log** (`Mesh.print_cycle_info` after every step: `energy=` + `pe=`), six panels, ten
steps, $n_z$ 16, lateral flow $u_1(1 + 0.5\sin 4x_2)$ so x2/x3 fluxes cross the panel seams, both x1 walls closed:
max per-step $|\Delta(E + P)|/|E + P|$ = $1.9\times10^{-15}$ (explicit) and $3.1\times10^{-15}$ (VIC); the log prints
15 significant digits. With F's work but $P$ without $\delta$ the same runs drift by $1.3\times10^{-10}$
(explicit) and $2.6\times10^{-10}$ (VIC) per step (recomputed $P$). E+P alone does not measure the order: e894700
closes its own $E + \mathrm{PE}_d$ to $2.1\times10^{-15}$ and $2.5\times10^{-15}$ on the same runs.
