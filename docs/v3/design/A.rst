=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 1 (due Wed 11-18 18:00 EST); revision 2 due Fri 11-20 18:00
:Reviewers (round 2): Jiheng (lane S), Xi (lane X)
:Pin: snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410`` (``9e5de2d FP32 step5
      validation (#368)``)
:Skeleton: ``docs/v3/skeleton/A/``, four headers, declarations only, 298
           lines (``wc -l docs/v3/skeleton/A/*``)

All ``path:line`` references below are at the pin
``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410`` unless another sha is given.

.. contents::
   :local:
   :depth: 1


1. Scope, sources and what I could not read
===========================================

This lane covers the twelve modules declared in ``src/forcing/forcing.hpp``.
That covers gravity's interface, but not ``const_gravity.cpp`` itself, which is
outside the baseline. It also covers the AAM machinery spread over hydro,
coordinate and mesh.

The doc answers the eight design questions of the lane brief (sections 3 to
10) and the two decide-and-justify items. The conserve-axial-momentum default
and kill list are in section 4.4. When diffusion enters the register is in
section 9.4. Section 11 gives the line-count plan and decision log, and section
12 lists the cross-lane questions.

Read for this revision:

* ``src/forcing/*`` (all 15 files), ``src/coord/gnomonic_equiangle.{hpp,cpp}``
  (AAM parts), ``src/coord/coordinate.hpp:370-392``,
  ``src/hydro/hydro_forward.cpp:988-1009, 1088-1145, 1288-1411, 1624-1696``,
  ``src/hydro/hydro.hpp:89-133, 219-236``, ``src/hydro/hydro.cpp:27-76``,
  ``src/hydro/register_forcing_modules.cpp``,
  ``src/mesh/meshblock.cpp:983-1034``, ``src/mesh/mesh.cpp:363-382``,
  ``src/riemann/hllc_impl.h:100-163``, ``V3_ROADMAP.md:258-279`` (G5).
* PR bodies, via ``gh pr view``:

  * #355: merged, ``46710ac570668a699cb2fc5ad17a580cd4156caa``.
  * #357: merged, ``2f5f51b2466704ef1390d087b74926b546627c31``.
  * #363: merged, ``dcf5a5a12e644b67232adb718bb6166b2ef0c0ae``.
  * #365: merged, ``3fe398a917646851ed42db3f964a3fc2751de2fe``.

  ``gh`` returned zero comments and zero reviews for all four.

* #314 is an issue, not a PR (``gh pr view 314`` answered "Could not resolve
  to a PullRequest"). I read it with ``gh issue view 314``. It is open and
  holds the AAM drift report, the budget split and the sponge ruling-out.

Not read, and therefore not used:

* The Slack threads named in the brief: the #357 review thread
  1794565311.770069 and the side study 1794706777.579419. Workers cannot read
  Slack. Everything said here about the side study's local-torque scheme comes
  from the one-line description in the lane brief, and is marked as such.
* The kickoff brief's flux-register requirements (part 2/3), including
  "requirement 7", are known to me only from the lane brief's wording.
  Section 3.1 states the register model this doc assumes. If the real
  requirements differ, the register-facing parts of sections 3, 5 and 9 change.
* No code was built or run for this revision. Every cost or behaviour claim
  for V3 that does not carry a PR measurement is a **hypothesis**.


2. Baseline and inventory at the pin
====================================

2.1 Measured baseline
---------------------

Command (at the pin)::

    wc -l $(ls src/forcing/* | grep -v const_gravity.cpp)

==================================  =====
file                                lines
==================================  =====
body_heat.cpp                          57
bot_heat.cpp                           88
bot_sponge_lyr.cpp                     78
coriolis.cpp                          232
diffusion.cpp                         745
diffusion_cubed_sphere.cpp            376
diffusion_radial.cpp                  103
forcing.hpp                           674
plume_forcing.cpp                      43
relax_bot_comp.cpp                    121
relax_bot_temp.cpp                    108
relax_bot_velo.cpp                     79
top_cool.cpp                           88
top_sponge_lyr.cpp                     78
**total**                           **2870**
==================================  =====

This matches the brief's 2,870.

Of ``forcing.hpp``, 326 lines are the twelve option structs and 270 are the
module classes. Command: ``awk`` over ``struct .*OptionsImpl`` ... ``using``
and ``class .*Impl`` ... ``TORCH_MODULE``. There are 48 ``ADD_ARG`` options in
total.

2.2 Lane-A code outside ``src/forcing`` (not in the baseline)
-------------------------------------------------------------

Counted with ``sed -n 'a,bp' FILE | wc -l`` for each range:

================================================  =======  =======================================
range                                               lines  what
================================================  =======  =======================================
hydro_forward.cpp:988-1003, 1108, 1141                 18  full: second Riemann eval (central)
hydro_forward.cpp:1288-1298                            11  transport: AAM flux form call
hydro_forward.cpp:1316-1363, 1390-1391                 50  Coriolis gate, x1 mass-w buffers, F_d
hydro_forward.cpp:1624-1680                            57  step 8: x1 mass-flux w, metric source
hydro.hpp:89-133, 219-236, 462-466                     68  AAM options (docs), accessors, state
hydro.cpp:27-37, 55-76                                 33  mode predicates and checks
coordinate.hpp:370-392                                 23  four virtual AAM / x1-source hooks
gnomonic_equiangle.hpp:22-35, 84-122                   53  AAM buffers and declarations
gnomonic_equiangle.cpp:18-36, 169-221                  72  rotation field k, face averages
gnomonic_equiangle.cpp:485-604, 664-719               176  rate_to_rows, x1 source, flux form, rate
meshblock.cpp:983-990, 1009-1034                       34  closure reduction and increment
mesh.cpp:364, 367-382                                  17  closure over local blocks
**total**                                         **612**
================================================  =======  =======================================

2.3 What the code does today (the facts the design rests on)
------------------------------------------------------------

* **Forcings are a flat list of ``AnyModule`` objects** built in
  ``register_forcing_modules.cpp:6-86``. They are called in order on a shared
  ``du`` (``hydro_forward.cpp:1378-1385``), with the signature
  ``forward(du, w, temp, dt)``. Nothing records what each one added.
  Issue #314 had to rebuild the sponge torque offline from output frames. Its
  third comment says "A stage-integrated sponge torque would need an in-model
  diagnostic".
* **Coriolis has two classes.** ``Coriolis123`` (``coriolis.cpp:40-67``)
  applies the same Ω in every coordinate system. ``CoriolisXYZ``
  (``coriolis.cpp:69-231``) rotates Ω into the local basis, for Cartesian,
  cylindrical, spherical-polar and gnomonic grids. On Cartesian grids the two
  agree (``coriolis.cpp:91-94`` against ``:53-63``), except for the 2D guard at
  ``:61``. No deck or test in the repository sets ``type: 123``. Command:
  ``grep -rn "'123'\|\"123\""``, which matched only ``coriolis.cpp``.
* **#355 (x1 mass-flux w)** gets the Coriolis rows driven by ρw from the x1
  face mass transfers. It does this by calling the Coriolis forward twice per
  stage on hand-built "primitive" tensors:

  * ``(ρ, 0, v2, v3)`` before the implicit solve (``hydro_forward.cpp:1351-1352,
    1384``);
  * ``(1, m1, 0, 0)`` after it (``:1650-1655``).

  The metric term ``-m1 v/r`` moved into a coordinate hook,
  ``x1_mass_source_`` (``gnomonic_equiangle.cpp:664-674``). #355 measured the
  w-free rows going through the scheme-9 solve: without that, w was off by up
  to 1.27 m/s after one step.
* **transport (#357)** rewrites the K component of the finished divergence
  from the same face fluxes (``hydro_forward.cpp:1290-1298``,
  ``gnomonic_equiangle.cpp:551-604``). This costs no flux evaluation.
* **full (#363)** recomputes a central flux. It makes two extra Riemann calls
  per x2/x3 direction (``hydro_forward.cpp:997-1003, 1108, 1141``), stores
  ``F - F_c`` (``:1354-1363``), and lets ``CoriolisXYZ`` add the flux-form
  torque (``coriolis.cpp:176-213``). It refuses FP32 and perturbation-flux
  (``hydro_forward.cpp:993-995``). #363 measured a cost of 1.14-1.30 times
  ``off`` at C16 and 1.32-1.48 at C32.
* **closure (#365)** has ``CoriolisXYZ`` compute R and I
  (``coriolis.cpp:189-209``). ``MeshBlock``/``Mesh`` then make one allreduce
  per stage and apply a solid-body increment (``meshblock.cpp:983-990,
  1014-1034``, ``mesh.cpp:367-382``). The path calls ``.item<double>()``
  twice (``meshblock.cpp:1018``), a host sync. #365 measured a cost of
  1.04-1.09 times ``full``, and a drift of 1.0e-15 to 1.5e-15 against
  4e-10 to 1e-7 for ``full`` alone.
* **Diffusion keeps its own fluxes and divergence.**

  * Cartesian: ``du -= dt * divergence(...)`` (``diffusion.cpp:671-673``).
  * Cubed sphere: Cartesian-component traction fluxes, its own seam flux
    exchange (``share_fluxes``, ``diffusion_cubed_sphere.cpp:103-176``, 74
    lines), a divergence, then a Cartesian-to-covariant projection
    (``:368-373``).
  * A separate Strang-split implicit radial conduction
    (``diffusion_radial.cpp``), driven by ``MeshBlock`` (``meshblock.cpp:715,
    905, 1480``).

  Viscous heating is already in flux form, as the energy-row term ``v · τ``
  (``diffusion.cpp:625-627``, ``diffusion_cubed_sphere.cpp:345``).
* **Decks.** No deck in the repository sets ``conserve-axial-momentum`` or
  ``conserve-axial-momentum-closure``. Only
  ``tests/test_axial_momentum_flux_form.{py,yaml}`` does. Command:
  ``grep -rln "conserve-axial-momentum\|conserve_axial_momentum"``. Decks
  outside the repository, such as the K2-18b GCM runs, cannot be checked from
  here.


3. Q1: AAM as a consumer of the register
========================================

3.1 Register model assumed (to be checked against kickoff 2/3)
---------------------------------------------------------------

I assume one face-flux register per block and stage. Contributors (Riemann,
#289 covariance, diffusion, ...) add face fluxes per direction under their own
identity. The register forms the divergence once, keeps each contributor's part
readable until the end of the stage, and makes shared faces single-valued
across block and panel seams. Fixers are registered separately and declare what
they change (requirement 7, as worded in the brief). If the kickoff
requirements say otherwise, see the cross-lane questions R-1 and X-1.

3.2 What AAM needs registered, and from whom
--------------------------------------------

AAM reads fluxes and never evaluates them. Every item below is a read of
something another lane already produces.

=====  ===========================================  ============  =====================================
id     quantity                                     from          used by
=====  ===========================================  ============  =====================================
A-r1   total face flux F, x1/x2/x3, all rows        R (+ any)     transport: K-component in flux form
A-r2   central mass rows F_c on x2/x3 faces:        R             exact: torque of F_d = F - F_c;
       1/2 (ρ_L u_L + ρ_R u_R), dry + species                     closure residual R
A-r3   explicit x1 mass transfer A_1 F_1 (mass)     R             x1 mass-flux w (#355): m1
A-r4   implicit x1 mass transfer per unit time      C / I         m1 (``implicit_mass_correction``)
A-r5   implicit momentum correction (tangential)    C / I         ``axial_x1_source_`` (face radius)
=====  ===========================================  ============  =====================================

A-r2 is the only new item. It is what removes the **second flux
evaluation**.

* Today ``full`` gets ``F_c`` as ``1/2 (F(w_L, w_L) + F(w_R, w_R))`` from two
  extra Riemann calls (``hydro_forward.cpp:997-1003``).
* For a consistent solver ``F(w, w)`` is the physical flux, so the mass rows of
  ``F_c`` are ``1/2 Σ_s (ρ_s u_n)_L + (ρ_s u_n)_R``. This is one tensor
  expression on the face states the solver already holds.
* AAM only uses mass rows (``axial_coriolis_rate``,
  ``gnomonic_equiangle.cpp:690-695``). So lane R emits ``n_mass = 1 + ny`` rows
  per x2/x3 face, in the same face frame as F. That is about ``(1+ny)/(5+ny)``
  of a flux tensor, and no Riemann evaluation.
* HLLC builds ``ρ u`` per side only in a moving frame (``vxl = u - b_m``,
  ``hllc_impl.h:104-120``). The emission must therefore use the face states,
  not ``fl``/``fr``.

**Hypothesis H1:** in FP64 this F_c equals today's construction to round-off
for HLLC and LMARS. It is untested. Gate: a one-stage comparison of ``F_d`` on
the #363 test deck, with tolerance 1e-14 relative.

**Hypothesis H2:** ``exact`` then costs about what ``transport`` costs plus the
rate evaluation. That would remove most of the 1.14-1.48 times overhead that
#363 measured. It is unmeasured.

The #289 covariance correction is added to F after the solve
(``hydro_forward.cpp:1110``), so today it counts as "dissipative" in
``F - F_c``. V3 keeps that meaning: F_d is the total registered mass flux minus
the central part, whoever contributed it.

3.3 Order in the stage
----------------------

1. Reconstruct and solve. Register ``F`` (A-r1) and ``F_c`` (A-r2), x2/x3.
2. The register forms the divergence. **transport/exact:** the AAM consumer
   rewrites the K component of the divergence from A-r1. This is today's
   ``conserve_axial_momentum_``, unchanged in arithmetic.
3. Sources, pre-implicit phase (section 8). **exact:** the AAM consumer adds
   the flux-form torque of ``F_d`` along K. **exact:** it also records the
   closure partial sums (section 5).
4. Implicit solve (lane I). Register A-r4 and A-r5.
5. Sources, post-implicit phase: Coriolis and metric rows driven by m1, from
   A-r3 and A-r4 (section 6). **transport/exact:** the AAM consumer applies
   ``axial_x1_source_``.
6. End of stage: the closure fixer (section 5).

The consumer holds no flux state between stages. Today's ``_aam_mass_flux``
side channel and its "clear after the first Coriolis call" rule go away
(``hydro_forward.cpp:1390-1391``, ``hydro.hpp:227-236``).


4. Q2: which modes survive
==========================

4.1 Evidence
------------

=========  =========================  ==================================  =============================
mode       AAM drift (measured)       cost (measured)                     restrictions
=========  =========================  ==================================  =============================
off        reference                  1                                   none
transport  3D jet 50-290x below off;  no extra flux eval                  gnomonic only; worse than off
           W2 1.1-1.7x; K2-18b C36                                        at Ω/(u/a) = 14 (#357,
           day 10: +7.1e23, off                                           ``hydro.hpp:101-102``)
           +1.50e25
full       3D arms 217-589x; W2       1.14-1.30x off (C16),               FP64 only, no perturbation-
           29.8x; not exact           1.32-1.48x (C32)                    flux, ``interp_order: 4``;
                                                                          C16 KE creep for k >= 2
closure    1.0e-15 to 1.5e-15         1.04-1.09x full                     needs full; one allreduce per
           (relative)                                                     stage; host sync
=========  =========================  ==================================  =============================

Sources: the #357, #363 and #365 bodies, and the ``hydro.hpp:89-133``
comments. ``full`` was "not yet run in a GCM configuration" (#363).

4.2 Decision: three modes, ``off | transport | exact``
------------------------------------------------------

* ``off``: unchanged.
* ``transport``: unchanged arithmetic.
* ``exact``: today's ``full`` plus ``closure``, as one mode. The torque of F_d
  reduces the residual locally, and the closure fixer removes what is left, to
  round-off.

``full`` without the closure is removed, and so is the separate boolean key
``conserve-axial-momentum-closure``.

Why:

* ``full`` alone is neither cheap (1.14-1.48x) nor exact (4e-10 to 1e-7).
  #365 shows that the closure on top costs only 4-9% more and makes it exact.
* The closure has no meaning without ``full`` (``hydro.cpp:67-69``).
* Two keys that are only valid in one combination are one mode.

What each removal changes:

* **full without closure**: anyone who wants a run with no global reduction
  and a flux-form torque loses that combination. It survives only as a test
  configuration of the consumer (a C++ test flag), not as a deck value. The
  test arms (f)-(h) and (k) of ``tests/test_axial_momentum_flux_form.py``
  move to ``exact`` or to the test flag. Arm (l)'s "closure off must fail"
  check keeps working through the test flag.
* **the closure key**: absorbed. ``exact`` always runs the fixer.

4.3 The side study's local-torque scheme
----------------------------------------

The brief describes it as "the torque of the dissipative mass flux
``F_d = F - 0.5 (F_L + F_R)`` and the local zonal torque on x2/x3 faces". I
have not read it (section 1).

The first half is what ``axial_coriolis_rate`` already computes from
``M - M_central`` (``gnomonic_equiangle.cpp:676-719``). So I read the scheme as
a face-local rule that would make the residual R vanish without a global
reduction. This reading is a **hypothesis**.

Decision: the torque is a pluggable rule inside the AAM consumer
(``FaceTorqueRule`` in the skeleton). The #363 rate is rule 1.

* If the side-study rule passes the gates below **without** the closure,
  ``exact`` switches to it and the fixer is dropped. That also removes the
  graph boundary (section 5).
* If it does not, it does not replace ``full``'s torque.

Gates, all on the existing test file:

* arm (l) drift <= 1e-12 with no reduction;
* arms (f)-(h), (k) at least as good as rule 1;
* noisy-jet KE ratio no worse than ``full``'s 1.005-1.012 at C32;
* cost <= rule 1.

4.4 Decide: default and kill list
---------------------------------

**Default: ``off``, unchanged.** Revisit after G5.

Why:

* The first migration PRs are judged by bitwise equality against the pin, and
  ``transport`` and ``exact`` are not bitwise ``off``.
* ``transport`` is measured to be *worse* than ``off`` at
  Ω/(u/a) = 14 (``hydro.hpp:101-102``, #357). That is about the Earth's regime
  (a hypothesis from Ω = 7.3e-5 s^-1, u = 20 m/s, a = 6.4e6 m, which gives a
  ratio of about 23). So it cannot be a safe universal default.
* ``exact`` needs FP64, no perturbation-flux and ``interp_order: 4``. A
  default cannot impose those on every deck.

What changes instead: the AAM budget line is **always reported** by the
monitor on gnomonic grids (section 8), whatever the mode, so the drift is
visible. Proposed follow-up after G5: make ``exact`` the default for gnomonic
grids with rotation once its FP32 and perturbation-flux gates pass (section 7).

**Kill list**

==========================================  ===================================  ==========================
removed                                     replacement                          decks affected
==========================================  ===================================  ==========================
``conserve-axial-momentum: full``           ``exact`` (refused with a message    in repo: test only.
                                            naming ``exact``; no silent remap,   **FOR CHENG**: any
                                            since ``exact`` adds the closure)    out-of-repo deck with full
``conserve-axial-momentum-closure: true``   ``conserve-axial-momentum: exact``   in repo: test only.
                                                                                 **FOR CHENG** (as above)
``coriolis/type: 123`` (``Coriolis123``)    ``type: xyz`` (identical on          none found in repo.
                                            Cartesian grids)                     **FOR CHENG** if a 2D
                                                                                 deck relies on ``:61``
``conserve-axial-momentum`` on a non-       refused instead of the silent        none in repo
gnomonic grid (today a no-op plus a
warning, ``hydro.cpp:70-76``)
==========================================  ===================================  ==========================

I cannot see the K2-18b and other GCM decks outside the repository. Every
**FOR CHENG** above means: if such a deck uses the value, Cheng decides between
migrating it and keeping an alias.


5. Q3: the closure as a registered fixer and a graph boundary
=============================================================

5.1 What it declares to the monitor
-----------------------------------

``AxialClosureFixer`` registers with:

* ``kind = numerical_fixer``. It is not a physical torque. The monitor books it
  apart from the sponge and drag, so that "physical torques are kept"
  (``hydro.hpp:131-132``) can be checked.
* ``rows = {IVY, IVZ}``. Mass, species and the energy row are untouched
  (``meshblock.cpp:1028-1033``). Since the energy row is not changed, the
  kinetic energy the increment adds or removes is absorbed by internal energy.
  The fixer declares ``moves_ke_to_ie = true``, and the monitor reports that
  amount. #365: "the KE change equals the change of L²/2I to about 3 digits".
* Its AAM change per stage is declared exactly as ``-w2 dt R``. The monitor
  checks the booked value against the measured one. A mismatch above 1e-12
  relative is an error.
* ``reduction = {R, I}``, two FP64 scalars per block. ``scope = global``,
  ``cadence = every stage``.

5.2 Where the reduction sits
----------------------------

Today: after the stage's saturation adjustment and before the boundaries
(``meshblock.cpp:983-990``). Mesh with several local blocks: sum, then one
allreduce, then apply and apply boundaries again (``mesh.cpp:367-382``).

V3 keeps that position, the **last momentum change of the stage**. R must see
every numerical torque of the stage, and the boundaries must see the
corrected momentum. Two changes:

* **No host sync.** R and I stay device tensors. The increment
  ``-w2 dt R / I · ρ s²`` along K is computed on the device, with
  ``I > 0`` and finiteness handled by ``where``. This removes the two
  ``.item<double>()`` calls (``meshblock.cpp:1018``).
* **One collective per stage for all fixers.** The fixer registry stacks the
  partial sums of every fixer due at that stage into one FP64 vector and makes
  one allreduce. Today's gravity-work fixer (``meshblock.cpp:997-1004``, last
  stage) and this closure (every stage) would share the last-stage collective.

The collective is a **graph boundary**: the stage graph splits there unless
lane X makes a device collective capturable (X-1, X-2). With rule 2 of
section 4.3, ``exact`` would have no reduction and no boundary.

R's definition is unchanged: ``Σ V (K · F_cell - rate(F_c))``. Since
``F_c`` is registered (A-r2), R is computed once. Today it calls
``axial_coriolis_rate`` twice and subtracts (``coriolis.cpp:199-203``).


6. Q4: one Coriolis and metric source (#355)
============================================

6.1 One definition
------------------

``force = -2 Ω × m`` in the local orthonormal frame, lowered to covariant rows.
Here:

* ``m = (m1, ρ v2, ρ v3)``;
* ``m1`` is the momentum consistent with the x1 mass transfers (A-r3 + A-r4),
  ``m1 = [Φ_lo (r² - r_lo²) + Φ_hi (r_hi² - r²)] / (2 r V)``
  (``hydro_forward.cpp:1626-1653``);
* the metric pair ``-m1 v_h / r`` (horizontal) is part of the same source.
  It is moved back from ``x1_mass_source_`` (``gnomonic_equiangle.cpp:664-674``).
  ``+ρ v_h² / r`` stays in the coordinate divergence, as #355 left it.

The source has two phases, declared to the stage graph:

* ``pre_implicit``: the w-free rows, using ``m1 = 0``. They must pass through
  the scheme-9 solve (#355 commit 6: 1.27 m/s otherwise).
* ``post_implicit``: the m1-driven rows only.

This replaces the hand-built ``wc1``/``wc2`` primitive tensors and the
``coriolis()`` type test (``hydro_forward.cpp:1323-1353, 1650-1655``) with
named inputs.

6.2 Per grid
------------

==================  ==============================  ===========================================
grid                Ω in local basis (lane S)       m1 source
==================  ==============================  ===========================================
Cartesian f-plane   constant (``coriolis.cpp:91``)  cell ρw (gate unchanged: not in
                                                    ``mass_w``, ``hydro_forward.cpp:1319-1322``)
spherical-polar     (θ, φ) rotation (``:101-109``)  x1 transfers when Coriolis is on (as today)
cubed sphere        contra -> spherical transform   x1 transfers (nc1 > 1, not shallow water);
                    (``:110-143``)                  cell ρw otherwise (#355 commit 2)
cylindrical         (``:95-100``)                   cell ρw (as today)
==================  ==============================  ===========================================

The table keeps the #355 gate exactly, so migration is bitwise.

**Hypothesis H3:** m1 from x1 transfers would also help Cartesian f-plane
energy consistency. Out of scope; only a measured follow-up would change the
gate.

The per-grid Ω rotation (``coriolis.cpp:85-151``, 67 lines) moves to the
geometry (section 10) as ``omega_local()``. Coriolis keeps one formula.
``traditional`` stays: it zeros the horizontal Ω components, and with it the
m1 rows.

6.3 Lanes I and S
-----------------

* Lane I must give the post-implicit phase a hook after its solve, and register
  A-r4 and A-r5 before that hook (I-1).
* Lane S provides ``omega_local`` and the m1 split weights ``lo``/``hi``.
  These are geometry today, built in ``hydro_forward.cpp:1331-1349``
  (S-1).


7. Q5: what lifts the FP32 and perturbation-flux restriction on ``full``
========================================================================

Today, ``hydro_forward.cpp:993-995`` refuses ``full`` with FP32 or
perturbation-flux. The PR says why: "its central flux does not see the
perturbation states" (#363). G5 asks for a "validated formulation"
(``V3_ROADMAP.md:268-270``). Three things together lift it. None of them alone
is enough.

1. **A consistent F_c.** A-r2 is emitted by the Riemann stage from the
   states it actually solved, whether perturbation or total. ``F - F_c`` is
   then a difference of two fluxes of the same representation by
   construction. The mass rows contain no pressure, so the reference-pressure
   gauge (``hllc_impl.h:96, 155``) does not enter. Whether face densities are
   totals under perturbation-flux is question R-1.
2. **FP64 accumulation.** The torque rate and the closure sums R and I are
   accumulated in FP64, in the same way as the forcing work buffer
   (``hydro_forward.cpp:1365-1375``). They are then rounded once into
   ``du``.

   **Hypothesis H4:** in FP32, ``F_d`` is a difference of nearly equal numbers
   at smooth faces, so its relative error is large where it is small. The
   torque should not care, because it is multiplied by
   ``s_f² - s_c² = O(h)``. This needs measuring.
3. **A G5 eager budget gate.** Rerun ``tests/test_axial_momentum_flux_form.py``
   arms (f)-(h), (k) and (l) under FP32 with perturbation-flux, compared with
   FP64. ``exact`` must hold drift <= 1e-12 relative with the closure. The
   closure works on totals, so FP32 storage of momentum bounds what it can
   remove per stage (this is a hypothesis). The gate gives the number.
   ``interp_order: 4`` ghosts under anomaly storage are G5 work in lane S.

The refusal is lifted only by the PR that passes item 3. Until then it stays a
reset-time check in the consumer, not in ``Hydro::forward``.


8. Q6: one source-term interface for the twelve modules
=======================================================

8.1 Interface
-------------

Every module is a ``SourceTerm`` (skeleton ``source_term.hpp``) and declares:

* ``phase``: ``pre_implicit`` (default), ``post_implicit``, or ``split``
  (operator-split around the stage, as the radial conduction is).
* ``rows``: a bit set over {mass, species, momentum, energy}.
* ``kind``: ``physical`` (sponge, drag, heating, relaxation), ``numerical``
  (AAM torque) or ``fixer``.
* ``where``: the interior, or a physical-boundary slab (``bottom``/``top``,
  depth or thickness). The base class does the ``is_physical_boundary`` test
  that six modules repeat today.
* ``add(du, StageState const&)``: adds its increment. ``StageState`` carries
  ``w``, ``temp``, ``dt``, the geometry and the registered fluxes. It replaces
  the ``(du, w, temp, dt)`` signature and the ``phydro`` back pointers.

The **monitor computes the budget itself**, from the increment.

* When budgets are on, the registry gives each source its own scratch ``dS``
  and then adds it to ``du``. When they are off, the source writes straight
  into ``du``, with no extra cost.
* The monitor forms ``Σ V dS[mass rows]``, ``Σ V dS[momentum]``,
  ``Σ V dS[IPR]`` and the AAM ``Σ V r K · dS[IVY, IVZ]``. The last uses the
  covariant rows with contravariant K, as in
  ``gnomonic_equiangle.cpp:520``.
* No module writes budget code.
* A source must not touch rows it did not declare (checked in debug builds).

Example: the K2-18b top-sponge torque. ``TopSpongeLyr`` declares
``rows = momentum`` and ``kind = physical``, and the monitor prints its
stage-integrated AAM. That is the number issue #314 had to estimate offline. In
#357's table, the removed ``full`` changed AAM by +6.7e23 at day 10, which
"equals the top sponge torque". With the interface, that comparison is one
line of the monitor output.

8.2 The twelve modules in V3
----------------------------

=======================  ===================================  =====================================
today                    V3                                   change
=======================  ===================================  =====================================
ConstGravity             SourceTerm (interface only)          implementation outside this lane (Q K-1)
Coriolis123/CoriolisXYZ  ``Coriolis`` (one class)             section 6; type 123 removed
Diffusion                ``DiffusionFlux`` + split radial     section 9
BodyHeat                 ``BodyHeat``                         base class does masks and options
TopCool / BotHeat        ``LayerHeat{side}``                  two 88-line files -> one
TopSpongeLyr /           ``Sponge{side}``                     two 78-line files -> one
BotSpongeLyr
RelaxBotTemp / Velo /    three classes on a shared            slab selection, boundary test and
Comp                     ``BottomSlab`` base                  covariant lowering in the base
PlumeForcing             ``PlumeForcing``                     unchanged arithmetic
=======================  ===================================  =====================================

The deck keys stay (``top-sponge-lyr``, ``bot-heat``, and so on). Only the
classes merge, so no deck changes. Python bindings follow the classes; whether
the two sponge option types stay distinct in Python is question O-2.


9. Q7: diffusion as a register contributor
==========================================

9.1 Target
----------

``DiffusionFlux`` is a register contributor. Per face direction it adds:

* the viscous momentum flux ``-μ τ · n``;
* the energy-row flux ``-k ∇T · n - v_f · τ · n``. The second term is the
  viscous heating, already in flux form today
  (``diffusion.cpp:625-627, 657-665``; ``diffusion_cubed_sphere.cpp:345``).

It no longer calls ``divergence`` and no longer touches ``du``.

Consequences:

1. Diffusive fluxes enter the AAM consumer's flux form like any other flux, so
   the **viscous torque telescopes** with ``transport``/``exact``.
   **Hypothesis H5:** today the cubed-sphere viscous term conserves Cartesian
   linear momentum per face, but its AAM is only as good as the post-divergence
   projection (``diffusion_cubed_sphere.cpp:369-372``).
2. The register's seam reconciliation replaces the flux-averaging half of
   ``share_fluxes``. The interior-penalty half needs the neighbour's traces
   (``trace``, ``normals``), so it stays a diffusion-owned trace exchange.
3. The cubed-sphere traction is computed in Cartesian components
   (``diffusion_cubed_sphere.cpp:297, 344``). To register it, it must be
   projected onto the hydro flux's face basis (the basis of
   ``flux2global2_``/``3_``). That needs a transform from lane S (S-3).
4. The implicit radial conduction stays an operator-split ``SourceTerm``
   (``phase = split``). Its wall heat flux is reported to the monitor as an
   energy flux through the boundary.
5. The time-step bound (``max_time_step``, ``diffusion.cpp:677-743``) is
   unchanged and is declared to the driver as a constraint.

9.2 Cubed-sphere variant
------------------------

It stays a separate face-flux evaluator (4-point stencils, reciprocal basis,
wall traction), because its operator is genuinely different. What it loses is
its own divergence, half of ``share_fluxes``, and the final projection. Both
evaluators share the coefficient, profile and wall-temperature code.

9.3 Profiles
------------

The per-cell profile tensors (``nu_scale_x1``, ``kappa_scale_x1``) are used by
C++ tests (``tests/test_diffusion_cubed_sphere.cpp:463, 598-599``) and the
Python binding (``python/csrc/pyforcing.cpp:64-65``), so they stay. The
128 lines that detect a later change (``diffusion.cpp:221-261, 429-516``)
become one call to a generic "frozen after build" check from the options
schema (O-1).

9.4 Decide: register in the first migration PRs, or later?
-----------------------------------------------------------

**Later.** Diffusion enters the register in phase 3 of the migration (section
11), after the register lands (phase 1) and after AAM consumes it (phase 2).
In phases 1 and 2 diffusion is a ``SourceTerm`` (``rows = momentum, energy``,
``kind = physical``) that keeps its own divergence, so its budget is declared
from the start.

Why:

* Phases 1 and 2 are gated on bitwise equality with the pin. Registering
  diffusion cannot be bitwise. On Cartesian grids, summing fluxes before one
  divergence changes the order of additions. On the cubed sphere, the face-
  basis projection and seam averaging change the arithmetic.
* Diffusion has its own validation suite (``docs/notes/cubed_sphere_*``,
  ``tests/test_diffusion_cubed_sphere.cpp``), which needs a tolerance-based
  acceptance of its own.
* Mixing the two would make a register defect and a diffusion defect
  indistinguishable in one PR.

The cost of waiting: until phase 3, the viscous torque is outside AAM's flux
form. ``exact``'s closure would remove it as a numerical residual, which is
wrong, since viscosity is physical. So phase 3 must land before ``exact`` is
used with nonzero viscosity. Until then, ``exact`` with ``nu_iso > 0`` is
refused at reset.


10. Q8: where the AAM geometry lives
====================================

**Decision: in the coordinate, as a ``RotationGeometry`` part owned by lane S.
The AAM physics (consumer, torque rule, closure) lives in
``src/forcing/axial_momentum.cpp``, owned by lane A.**

* Geometry: ``k`` at centres and exact face averages, ``k·k``, ``cos² lat`` at
  faces, ``rate_to_rows`` (the projection along K, including the on-axis cell
  rule), and the m1 split weights. Today these are
  ``gnomonic_equiangle.{hpp,cpp}`` buffers and ``coordinate.hpp:370-392``
  hooks. Add ``omega_local`` (section 6) and keep the 7 AAM buffers.
  Implement it for spherical-polar too, where ``K = r sin θ φ̂``. Cartesian
  returns "no axis".
* Physics leaves the coordinate: ``axial_coriolis_rate``
  (``gnomonic_equiangle.cpp:676-719``) and the closure arithmetic are AAM
  physics, not geometry. The four virtual AAM hooks on ``Coordinate`` become
  one accessor, ``rotation_geometry()``, which returns null where there is no
  axis. That is −23+3 lines in ``coordinate.hpp``.
* Why: lane S owns metric exactness, and the face averages of k are metric
  identities: the lateral pressure fluxes sum to zero
  (``gnomonic_equiangle.cpp:181-182``). The torque rule is the part that
  changes (section 4.3), and it should change without touching geometry.


11. Line-count plan, migration order and decision log
=====================================================

11.1 ``src/forcing`` (the baseline, 2,870 lines)
------------------------------------------------

Every number in the V3 column is an **estimate**. The measured sub-counts it
rests on come from ``sed -n 'a,bp' FILE | wc -l`` at the pin:

* ``Coriolis123`` 28 lines (``coriolis.cpp:40-67``); full/closure glue 38
  (``:176-213``); Ω set-up 67 (``:85-151``);
* diffusion ``from_yaml`` 91 (``diffusion.cpp:271-361``); profile checking 128
  (``:221-261, 429-495, 497-516``);
* ``share_fluxes`` 74 (``diffusion_cubed_sphere.cpp:103-176``);
* ``forcing.hpp`` option structs 326, module classes 270.

=================================  ======  ======  =======================================================
file                               now     V3      basis
=================================  ======  ======  =======================================================
forcing.hpp                         674     260    options 326 -> 150 via schema report/clone/create (O-1);
                                                   classes 270 -> 70 (one base, merged modules);
                                                   includes, macros, comments 78 -> 40
source_term.hpp (new)                 0     110    base, registry, budget declaration
coriolis.cpp                        232     120    −28 (123), −38 (glue to AAM), −50 (Ω to geometry),
                                                   +4 (two phases)
axial_momentum.cpp (new)              0     170    moved in: torque, R/I, closure apply, consumer order
diffusion.cpp                       745     480    −70 options, −110 profile checks, −40 region helpers,
                                                   −45 divergence/dispatch
diffusion_cubed_sphere.cpp          376     320    −35 flux averaging, −15 divergence/projection,
                                                   −6 boilerplate (phase 3)
diffusion_radial.cpp                103     103    unchanged
top_cool + bot_heat                 176      70    one ``LayerHeat{side}``
top_sponge + bot_sponge             156      50    one ``Sponge{side}``
relax_bot_temp/velo/comp            308     235    shared ``BottomSlab`` base
body_heat + plume_forcing           100      70    base does ctor, reset, options
**total**                          2870    1988
=================================  ======  ======  =======================================================

Arithmetic:

* 260 + 110 + 120 + 170 + 480 + 320 + 103 + 70 + 50 + 235 + 70 = **1988**.
* The reduction is 2870 − 1988 = **882 lines, 30.7 %** (882 / 2870 = 0.307).

Sensitivity:

* **If lane O gives no schema-generated report/clone** (O-1 declined):
  forcing.hpp is +110, which gives 2098 (−26.9 %).
* **Before phase 3**: diffusion_cubed_sphere.cpp stays at 376 and diffusion.cpp
  at 525, which gives 2089 (−27.2 %).
* **Both**: 2199 (−23.4 %).

Every case stays above the 20 % target. The 170 lines moved *into*
``axial_momentum.cpp`` are counted against the lane. Without them, the pure
forcing reduction is 2870 − 1818 = 1052 (−36.7 %).

11.2 Lane-A code outside ``src/forcing`` (612 lines, section 2.2)
-----------------------------------------------------------------

This is an estimate, reported separately so that moving lines into
``src/forcing`` does not hide them.

* hydro_forward 136 -> 25: two phase hooks, the consumer call, and the
  register reads.
* hydro.hpp/.cpp 101 -> 45: the option doc keeps the measured rows.
* coordinate.hpp 23 -> 3.
* gnomonic AAM 301 -> 240: ``axial_coriolis_rate`` moves out; spherical-polar
  K is added in a separate file of about 40 lines.
* mesh/meshblock 51 -> 10: generic fixer registry, in lane X's or O's code.

That gives 612 -> 323 + 40 = 363. Lane total: 2870 + 612 = 3482 -> 1988 + 363
= 2351 (−32.5 %).

11.3 Migration order
--------------------

Each phase is one PR, or a small stack, with a RED/GREEN test.

1. **Phase 1** (bitwise): ``SourceTerm`` base and registry; all twelve modules
   ported. The merges are sponge, layer heat, the ``BottomSlab`` base and one
   Coriolis. The monitor gets the budget. Gate: bitwise against the pin on
   every existing test.
2. **Phase 2** (bitwise for off/transport; round-off for exact): the AAM
   consumer reads the register, F_c is emitted by lane R, closure becomes a
   fixer, ``exact`` replaces full+closure. Gates: H1 (1e-14), and arm (l) <=
   1e-12. Cost is measured against today's ``full`` + closure.
3. **Phase 3** (tolerance): diffusion fluxes into the register. Gates:
   ``test_diffusion_cubed_sphere``, the convection-onset notes, and an AAM arm
   with viscosity under ``exact``.
4. **Phase 4** (G5): FP32 and perturbation-flux gate for ``exact`` (section
   7); then decide on the default.

11.4 Decision log (choices the brief left open)
-----------------------------------------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c mass rows emitted from face states; no second     removes 2 Riemann calls per x2/x3
    flux evaluation                                       direction (3.2)
D2  modes ``off | transport | exact``; full and closure   full alone is neither cheap nor exact (4.2)
    key removed; type 123 removed
D3  default stays ``off``; AAM budget always reported     bitwise migration; transport worse than off
                                                          at high Ω/(u/a) (4.4)
D4  closure = fixer at the last momentum change; device   keeps #365 semantics, drops host sync,
    R/I; one collective for all fixers                    shares the gravity fixer's collective (5)
D5  one Coriolis ``-2Ω × m`` with pre/post-implicit       one formula; #355 phase split kept (6)
    phases; Ω and m1 weights from geometry
D6  FP32 lifted only by consistent F_c + FP64 accum +     a code change alone does not establish
    G5 gate                                               the budget (7)
D7  monitor derives budgets from each source's            no budget code in modules; sponge torque
    increment; modules merge by class, keys stay          in-model (8)
D8  diffusion into the register in phase 3, not phase 1   bitwise gates of phases 1 and 2 (9.4)
D9  ``exact`` with ``nu_iso > 0`` refused until phase 3   closure would remove a physical torque (9.4)
D10 AAM geometry in coordinate (lane S); AAM physics in   geometry is stable, the torque rule is not
    ``src/forcing/axial_momentum.cpp``                    (10)
D11 side-study torque is a pluggable rule with gates      not read; decided by measurement (4.3)
=== ====================================================  ==========================================


12. Cross-lane questions
========================

These go to the kickoff thread. They are not resolved here.

* **R-1** (lane R): Can the Riemann stage emit, per x2/x3 face, the central mass
  rows ``1/2 Σ_s (ρ_s u_n)_L + (ρ_s u_n)_R`` (dry + species, ``1 + ny`` rows)
  into the register, in the frame of the registered flux, for HLLC, LMARS and
  Roe, in FP64 and FP32 and under perturbation-flux? Under perturbation-flux,
  are the face densities totals or anomalies?
* **R-2** (lane R / register owner): Does the register keep each contributor's
  part (Riemann, #289 covariance, diffusion) readable until the end of the
  stage? AAM needs "total minus central"; it does not need the parts
  separately, but diffusion's budget does.
* **C-1 / I-1** (lanes C, I): Who registers the implicit x1 mass transfer
  (``implicit_mass_correction()[IVZ] / dt``, ``hydro_forward.cpp:1645-1646``)
  and the tangential momentum correction used by ``axial_x1_source_``
  (``picorr->correction()``, ``:1668-1669``)? Can the stage graph give a
  post-implicit source phase after them? #355 requires the w-free Coriolis rows
  *before* the solve.
* **X-1** (lane X): Is a device-side allreduce of a small FP64 vector
  capturable inside a stage graph (NCCL or commux/UCX), or is it always a
  segment boundary? Where does lane X want the end-of-stage fixer point in the
  graph?
* **X-2** (lane X): May all fixers due at a stage (AAM closure every stage,
  gravity-work fixer at the last stage) share one collective through a fixer
  registry? Who owns that registry: X, or the register owner?
* **S-1** (lane S): Will S own ``RotationGeometry`` (k fields, face averages,
  ``cos² lat`` at faces, ``rate_to_rows``, ``omega_local``, the m1 split
  weights) for gnomonic and spherical-polar grids?
* **S-2** (lane S): Will the register store x2/x3 fluxes in each face's own
  basis, as ``flux2global2_``/``3_`` do today? The AAM face averages
  ``aam_k_f2``/``aam_k_f3`` are in that basis.
* **S-3** (lanes S, R): Who provides the Cartesian-to-face-basis transform that
  diffusion needs to register its cubed-sphere traction? Does the register
  reconcile shared-face fluxes across panel seams, which would replace the
  averaging half of ``share_fluxes``?
* **O-1** (lane O): Can the options schema generate ``create``/``clone``/
  ``report``/``from_yaml``, and freeze tensor options after the block is
  built? The line plan assumes yes (−110 lines if not).
* **O-2** (lane O): Should merged classes (``Sponge{side}``,
  ``LayerHeat{side}``) keep the separate Python option types for API
  stability?
* **K-1** (kickoff): ``ConstGravity`` is one of the twelve modules in
  ``forcing.hpp``, but ``const_gravity.cpp`` is outside this lane's baseline.
  Which lane migrates gravity, its face/cell gravity work and the gravity-work
  fixer? Lane A only provides the ``SourceTerm`` interface it would implement.
* **K-2** (kickoff): Please confirm or correct the register model of section
  3.1 against the flux-register requirements (kickoff 2/3, including
  requirement 7), which I could not read.
