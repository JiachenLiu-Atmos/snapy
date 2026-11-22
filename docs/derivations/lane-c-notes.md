# Lane C reference notes

Documentation moved out of the lane C sources (V3 C.ceiling.trim) so the code
stays within the lane's line ceiling. Each section names the declaration or
block it documents; the text is the former source comment.

## balance_column (src/hydro/balance_column.hpp)

Project a ghost-free x1 column onto the well-balanced scheme's own
discrete hydrostatic balance, at fixed temperature.

At rest the well-balanced x1 reconstruction reconstructs p' = p - pref(p),
where pref is the reference it rebuilds from the state on every call. A
column marched through the hydrostatic ODE satisfies the CONTINUUM balance
to O(dz^2), not this discrete one, so p' is not constant and its variation
is a standing force on an atmosphere that is supposed to be at rest. This
iterates

    p_i <- pref_i(p) + C,     rho_i <- p_i / (p_i/rho_i)_0

to the fixed point p' == C. C is a free gauge (any constant p' is balanced)
and is read per column at the TOP cell, the lowest-pressure one, so the
column moves least where it is thinnest; anchoring it at the bottom
diverges on a deep column. p/rho is held fixed per cell, which for
`ideal-gas` and `ideal-moist` at fixed mass fractions holds the TEMPERATURE
fixed exactly -- only p and rho move, together (conversion_order 2).

\param w        (nvar, nc3, nc2, nx1) primitive state, nvar > IPR. The last
                dimension is the WHOLE column with NO ghost cells: every
                value the reference reads is then one this caller owns.
                Channels other than IDN and IPR are carried through
                untouched, and `w` itself is not modified.
\param dx1f     (nx1,) cell widths, same dtype and device as `w`. Whether
                the grid counts as uniform is decided here by the SAME test
                the solver applies to its own grid, and it selects between
                two different reference operators -- so a caller whose
                ghost-free dx1f would classify differently from the block's
                padded one would be balancing against the wrong operator.
\param grav     downward gravity MAGNITUDE (> 0), as the kernel takes it.
\param wall_clamp  must be true, and must match `dynamics/wb-wall-clamp` in
                the config that will run the result. With the clamp off the
                reference's two cells at each wall are quadratured from
                faces BELOW the wall -- ghost faces, which a ghost-free
                column does not have -- so the fixed point found here would
                not be the one the solver later enforces.
\param rtol     bound on the residual below, dimensionless. Its floor is
                about eps (p/rho) / (g dz_min), ~10x that at order 4: pass
                a larger rtol when the thinnest cells are millimetres.
\param max_iter sweeps before giving up; at conversion_order 4 it bounds
                both the outer passes and each pass's inner sweeps.

\return (balanced w, residual, sweeps). The residual is
        max |p' - C| / (rho g dz), which bounds what the leftover imbalance
        can still do: it is the acceleration the column can still feel, in
        units of g. A dimensionless bound is the only portable one -- 1e-9
        Pa is below double precision on a column whose bottom cell is
        1.2e7 Pa. The residual is measured BEFORE the update that is
        skipped on convergence, so it describes the state that is returned.
        `sweeps` counts updates applied, so a column already at the fixed
        point comes back with 0.

\throws if the sweeps do not reach `rtol`: a column that did not converge is
        not balanced, and starting a run from it is the bug this exists to
        remove.

Both x1 ends are treated as PHYSICAL boundaries, which is what a column
bounded by walls is. The caller owns four conditions this cannot check:

 * the x1 boundaries really are physical, not periodic;
 * every block the column is later split into has at least five x1 cells --
   below that the reference falls back to a two-point mean at the wall, and
   the balance found here is not the one that block will enforce;
 * the `uniform` classification below, which is made ONCE for the whole
   column, is the same one every block will make for itself. The solver
   classifies per block, so a grid that is uniform over part of the x1 range
   and stretched over the rest can give a block inside the uniform part the
   six-face rows while this took the log-mean branch for the column -- and
   then the fixed point found here belongs to an operator no block applies.
   The two branches are different operators, not different accuracies.
 * a moist caller re-applies saturation after the projection and iterates
   the two while saturation still changes the state (examples/bryan.cpp).

And one consequence rather than an obligation: the fixed point is p' == C
for a FREE constant C, so nothing ties the result to the input. The column
moves
-- 2.5e-4 of p on a Neptune-like column, 1.2e-2 on a 300 km giant-planet
one -- and no surface pressure, column mass or anchor level survives that.
An IC builder that pinned something must re-pin it against the BALANCED
column, which means running the projection inside its own fixed-point loop.

What transfers from this ghost-free column to a real block is the CELL
PRESSURE reference `pref`, and it transfers bitwise -- that is the property
the whole arrangement rests on, and `tests/test_balance_column.cpp` gates
it. The face density `dsf` is the one reference output that does NOT, at
the bottom-most cell only, because its one-sided fallback keys on the
absolute index i > 0 rather than on i > il. `dsf` plays no part in the
fixed point, so the balance is unaffected; nothing else here depends on it.

`stretched` is the grid key's classification (#280): pass
`coord->options->stretched_x1()` of the grid the column comes from. A
stretched x1 is non-uniform in every block whatever its widths, and its
cell pressure is the six-face degree-5 Lagrange cell average with per-cell
rows (x1_rows' six_face_rows); a column of unequal widths takes them
whatever the flag.

Under SNAP_X1_CENTROID_EXACT a spherical-polar block's reference works on
plain means converted from its r^2 means, which this column does not model:
the caller then passes geometry "cartesian", and anything else (unknown, or
spherical-polar, whose rest state is the solver's own) is refused.

`conversion_order` is the run's dynamics/conversion-order. At 4 the solver
applies the reference to w4 = W(u - L u) + L W(u), not to W(u), so where W
is not linear in u (mass fractions varying in x1) a column balanced at 2 is
not at rest under key 4. Order 4 balances w4 instead: each pass projects
w4 at order 2 and moves w by the same dp and drho, until w4 needs no sweep;
`eos` (the run's) gives W -> U -> W, and the return counts passes. The x1
ends are reflecting walls, as above. The temperature (p/rho) is held per
pass, not over the call: w's p/rho drifts at second order in dz. The
conversion is the solver's own (Conversion): cells where conv4_admissible()
fails keep W(u), as in the solver (its limiter-repair fallback is not
modelled). The eos limiter flag is cleared
for the call and restored on return and on error. `eos` must come from a
block with nx2 = nx3 = 1.

## wb_ref4 (src/recon/x1_reference.cpp, formerly src/hydro/wb_ref4.{hpp,cpp})

#289, dynamics/conversion-order 4: fourth-order x1 reference FACE DENSITY
for the well-balanced reconstruction -- the face value of the kernel's own
cell density reference, which is what makes the face offset of
docs/derivations/wb-ref4.md section 2 vanish. The cell reference and the
face pressures (the hydrostatic scan) are never changed; a guarded face
keeps the kernel's value. The rows are x1_rows' face_density_rows.

`wb_ref4_flag`: the resolution flag (nc3, nc2, nc1): cells (dilated by two) where the
column has fewer than two cells per pressure scale height and the kernel's
face value stays. Needs no seam exchange, the scan pressures being local.

`wb_ref4_faces`: face part, after the x1 seam exchange: dsf on the owned faces is..iu+1 is
the fourth-order face value of dref itself. Needs is >= 1 (a ghost cell).

## X1Reference::evaluate (src/recon/x1_reference.cpp)

### The anchor relay

Global x1 reference across a vertical (nb1>1) decomposition: make the
hydrostatic reference continuous over the x1 process seams. The block
owning x1-outer anchors at the true domain top; every block below receives
the running seam-face pressure from the block above and passes on that
value plus its own interior hydrostatic drop (= its bottom-face pressure).
A serial top->bottom scan along the x1 block column, whether the blocks
sit on other processes or on other threads of this one (#254). nb1 == 1
(no x1 neighbor) => an empty anchor tells the backend to compute the
local top anchor.

### wb_ref4 in X1Reference::evaluate

dynamics/conversion-order 4 (#289): the only fourth-order part is the
face density below, the face value of the kernel's own cell dref -- that
is what makes the O(dz^2) face offset vanish, whatever dref's own accuracy
(wb-ref4.md section 2), and replacing dref broke the theta bound of a
density current at a wall. psf is never changed, so the rest balance is
the kernel's. Only the resolution flag is wanted here; the face part
follows the seam exchange, so a column split along x1 gets one block's
faces.

### The ghost-row exchange

Ghost-row reference exchange across x1 seams: overwrite this block's
ghost rows of (pref, dref) with the NEIGHBOR'S interior rows for the same
physical cells. The block-local ghost quadrature (w6e edge rows, and the
[lo,hi] guard whose accept/fallback decision is stencil-dependent) does
NOT reproduce the owner's interior values -- near a physical wall the
mismatch reaches ~1e2 Pa (measured), which enters
w' = w - ref as a spurious seam perturbation every step and makes the
dynamics nb1-dependent. After this exchange the perturbation field seen
by the reconstruction is identical on both sides of every seam, so the
seam-face states agree and the single-valued seam flux average becomes a
no-op. nb1=1: no seams, bit-unchanged. A same-process neighbor goes
through a board, a remote one through the process group (#254).

### wb_ref4 faces

conversion-order 4: the face density is the fourth-order face value of
the (exchanged) cell density reference, so a split column gets the same
faces as one block.

### Per-side face density

The face density on each side of a face is the x1 reconstruction's own
optimal-linear row applied to dref (cp5 = the weno5 candidates combined
with their linear weights), so rho_f = R_lin(rho) + N(rho') whatever
dref is. Read after the seam exchange; a face whose value leaves
[min, max] of its two cells' dref keeps the face density above, which is
also what any other reconstruction gets on both sides.

### Reflecting walls

A reflecting x1 wall face is a mirror: one face density on both sides,
the mean of the two rows. Two values there give a Riemann solver whose
wave speeds read the face density (hllc, roe) a nonzero contact speed,
which moves mass through the wall.

### Reflecting walls at conversion-order 4

conversion-order 4: a side whose cp5 row reads a ghost keeps wb_ref4's
face density (both sides on the two wall faces, the reading side on
the third), all walls in one gather and one index_put on (side, face)
rows built once per layout. Both sides then hold dsf at the wall face,
where the mean below is that value (0.5 * (x + x) = x), so it is
skipped.

## src/hydro/hydro_ref_x1_impl.h

### hydro_ref_x1_wall_rop

rho/p k cells past a wall, from the wall cell r0 and its neighbour r1:
linear, r0 + k (r0 - r1), where rho/p rises towards the wall (r1 <= r0),
and ln-linear, r0 (r0 / r1)^k, where it falls (r1 > r0). Each is the
branch that stays closer to r0, so the continuation lies within
[r0 (r0 / r1)^k, r0 (1 + k)] for r1 >= 0: positive, and bounded above when
r1 is nearly empty. A non-positive value keeps the wall cell.

### hydro_ref_x1_rop_smooth

rho/p for the cell density reference. In the interior it is the cell's
own rho/p: an index-space binomial preserves functions linear in index,
not in x, so at a width change it leaves an O(dz) kink in rho - dref that
no face density can undo (the face density is now the x1 reconstruction's
own linear rows applied to dref, hydro.cpp). A cell whose 5-point stencil
reaches past a clamped wall keeps the 5-point binomial below, unchanged.
Past a clamped wall (ext_lo/ext_hi: the wall side owns at least two cells)
the stencil continues rho/p from the two cells next to the wall instead of
repeating the wall cell (hydro_ref_x1_wall_rop). Either continuation is
linear in index to O(dz^2), so the wall cells keep the interior's O(dz^2)
bias; a repeated wall cell is exact only for constant rho/p and leaves an
O(dz) error at the first faces (docs/derivations/wb-ref-wall.md). Only
owned cells are read.

### six-face rows near a wall (thin)

the six-face stencil never crosses a physical wall
A one-sided row spans SIX faces, so on a block with fewer than five x1
cells it reaches one face past the owned range -- at the OPPOSITE end from
the wall it is protecting. That is a seam face, i.e. the neighbour's own
data, and reading it is right; it is only unreadable when that end is a
physical wall too, which means the whole column is under five cells.

### six-face rows near a wall (fits)

and the row has to FIT THE ARRAY. The inner row reads faces il..il+5, the
outer iu-4..iu+1, and with nc1 == il + iu + 1 both fit exactly when
iu >= 4. Below that the outer row's start goes negative -- and
hydro_ref_x1_face saturates above nc1 but has no lower guard, so it would
read before this column. nx1 >= 5 always satisfies it; a block thin enough
to fail it has fewer cells than it has ghosts.

## src/riemann/riemann.hpp

### RiemannSolverImpl::create

Create and register a RiemannSolver module

This function registers the created module as a submodule
of the given parent module `p`.

\param[in] opts      options for creating the `RiemannSolver` module
\param[in] p         parent module for registering the created module
\param[in] name      name for the created module
\return created      `RiemannSolver` module

### RiemannSolverImpl::forward

Solve the Riemann problem

upwind: vel_or_flux is the face velocity or mass flux and the upwind
flux is returned. plume-roe (dim 1 only) returns its Lax-Friedrichs
flux of the plume equations. The other kinds write the flux into
vel_or_flux and return it.

central, if defined (hllc, lmars, roe, shallow-roe): the kernel writes
riemann.central there, the mass rows (IDN and the species) of
1/2 (F(w_L) + F(w_R)) from the face states it solved; other rows are
left as they are.

### RiemannSolverImpl::forward_perturbations

hllc only. Pressure rows are perturbations about one shared face
background. Returns perturbation normal-momentum flux and contact
pressure; energy flux includes the total pressure. Other primitive rows
retain their normal meanings. This is separate from the total-state
forward contract.

central, if defined: riemann.central as in forward. Its mass rows read
only the density, normal velocity and species rows, which are totals
here too, so it is the total-state central.

### call_riemann (riemann.hpp)

One stub for the face-function kinds (hllc, lmars, roe, shallow-roe). The
iterator's operands are the flux, the face pressure (not shallow-roe), wl,
wr, then e and gamma (hllc, lmars, roe), c (hllc, roe) and the face
background p0 (hllc's perturbation entry). inv_mu_ratio_m1, cv_ratio_m1 and
u0 are roe's ideal-moist tables, undefined otherwise; they are arguments so
that OnHost copies them for MPS.

## src/recon/reconstruct.hpp

### ReconstructImpl::create

Create and register a Reconstruct module

This function registers the created module as a submodule
of the given parent module `p`.

\param[in] opts      options for creating the `Reconstruct` module
\param[in] p         parent module for registering the created module
\param[in] name      name for the created module
\return              created `Reconstruct` module

## src/riemann/riemann.cpp

### plume_flux

calculate flux for plume equations

    w = [R, W, B, V]

    F1 = (1 - 1/e) * R^2 * W
    F2 = 1/2 * R^2 * W * W - 1/2 * R^3 * V - 1/8 * R^2 * V^2
    F3 = 1/2 * R^2 * B * W
    F4 = 1/4 * R^3 * V * W + 1/2 * R^4 * W

## src/recon/well_balanced_x1.cpp

### positivity fallback

Positivity fallback = the adjacent cell's density, not dsf: a
reference that overestimates density aloft (a bottom-anchored
isentrope did, by orders of magnitude) turns every floor event at a
reflecting top wall into a large spurious wall impedance.
Shift by one along x1 with EDGE REPLICATION, not torch::roll: roll is
circular, so at index 0 it would substitute the density from the TOP
of the column. That word is not consumed today (physical faces run
il..iu+1 with il = nghost), but a wraparound inside a positivity
fallback becomes live the moment a caller changes the range.

## src/recon/x1_rows.hpp

### wall4_rows

conversion order 4 at a reflecting x1 wall, (2, 4, ng, 4): E[side][c] maps
the four owned averages next to the wall (narrow order) to the ghosts.
c = 0: the cubic through them. Parity ghosts, the mirrored cell plus twice
the opposite-parity part of a fit: c = 1 odd v + v''(0) z^2 (least squares
on the four cells), c = 2 even rho' - rho'_z(0) z - rho'_zzz(0) z^3 / 6
(the cubic), c = 3 even p' - p'_zzz(0) z^3 / 6 (a0 + a2 z^2 + a3 z^3
through the three wall cells)

## src/recon/x1_reference.hpp

### HydroReferenceX1Operator::scan_pressure

Linear pressure scan with an explicit upper face anchor at cell iu.
rho is mixture density (or its increment), [nc3,nc2,nc1]; anchor has
shape [nc3,nc2,1]. Signed values and zero are retained without floors.
This does not linearize pref/dref, wall continuation, or nonlinear guards.
Caller owns anchor relay and any background cache. Output dtype/device
follow rho; geometry and anchor are converted before scanning. Returns
{lower, upper} face pressure for each cell, including ghosts.

## src/recon/recon_point.h

### recon_point.h

One definition of each reconstruction at one face, shared by the CPU line
loop (recon_dispatch.cpp) and the CUDA line kernel (recon_line.cuh). src
points at the first of the stencil's inputs, which are stride apart; c holds
the scheme's coefficient rows. With scale, the stencil is divided by its
mean absolute value before the nonlinear weights and multiplied back after;
a stencil of zeros gives 0. Every constant is typed T (V3 C9), so a float
instantiation computes in FP32 throughout; FP64 is unchanged.

## src/recon/recon_dispatch.cu

### recon_line_cuda

One launcher for every stencil scheme: a line that fits one CUDA block (1024
threads, the block limit) keeps the one-block kernel, split into launches of
at most `per` variables when their shared memory does not fit; a longer line
is walked in tiles (recon_line_tiled). Uniform: the table goes to shared
memory. Nu: the per-face rows stay in global memory, indexed by face, so
each part reads the same rows. Every output is the same point function
either way.

## src/riemann/riemann_kernels.h

### riemann_central

riemann.central's mass rows at one face (C.rst 3.6, section 4): 1/2 (F(w_L)
+ F(w_R)) of the dry row and of each species row, from the face-frame
states the solve used; u is the face-normal velocity, rd = 1 - sum q.
Shallow-roe has the one row 1/2 (h_L u_L + h_R u_R). The other rows are
not written.

### riemann_face

one face of kind K. p holds the iterator's operands at the face, in the
order of call_riemann (riemann.hpp): the outputs (flux, the face
pressure except for shallow-roe, riemann.central if C, roe's LLF flag if
E), then wl, wr, e, gamma and c, (2, ...) with the right side one output
stride after the left, and p0. C and E are template parameters so that
the face functions compile as before when neither is requested.

## src/recon/conversion.hpp

### conv4_admissible

dynamics/conversion-order 4's per-cell fallback test: true where `a` is
admissible -- finite, rho above the density floor and, with `limiter` on,
no species below zero (u - L u, prim false), and p above the pressure
floor too (its primitives, prim true). The floors are zero with `limiter`
off.

### Conversion

Cell averages to primitives along x1 at conversion order 4, the one stage
of the solver (HydroImpl::forward) and of balance_column, so that the key-4
rest state balance_column finds is the solver's own fixed point:
w = W(u - L u) + L W(u), fourth order where W(u) is second, and W(u) in a
cell where u - L u or its primitives are not admissible. At a reflecting
x1 wall the ghosts come from the owned cells (wall4_rows).

## src/recon/recon_line.cuh

### recon_line_tiled

One CUDA block cannot hold a line longer than 1024 threads. Walk that
line in tiles of blockDim.x outputs. A tile loads blockDim.x + Stencil - 1
inputs (the overhang) and calls the same shared-memory weights as the
one-thread-per-cell kernel. Nu: coeff is a per-face x1 table, one row of
NCoeff per output, read from global memory.

## _apply_inplace (src/recon/reconstruct.cpp)

```
|<---- nghost --->|<--- interior -->|<---- nghost --->|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
                     ^           ^
                     |           |
                     il          iu
```
