=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 2 (design and contract only, CHANGE 5). Written against lane R's
           header; answers Xi's N1-N18 (appendix A) and Zoey's review
           (appendix B); folds in lane S's S-1..S-3.
:Reviewers (round 2): Xi (lane X), Zoey
:Pins: snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``; kintera ``3f5ff938``
       (section 8); lane R header ``happysky19/snapy@543ff84f4adb4b79c3f43fa7c2f0a5a6eb54fa48``
       ``docs/v3/skeleton/R/flux_register.hpp`` (cited "R:line"); lane X
       design ``77c65d5aac1f96ff6c34a45ef48faeb996c2128e`` (cited "X.rst:line")
:Previous: rev 1 ``2c9589c6``; rev 2 part 1 ``afd08eaa``
:Skeleton: ``docs/v3/skeleton/A/``, four headers in namespace ``snap``,
           declarations only, 294 lines (``wc -l docs/v3/skeleton/A/*``);
           index in section 12

All snapy ``path:line`` references are at the pin ``9e5de2d4`` unless another
sha is given. Nothing in this revision was built or run.

.. contents::
   :local:
   :depth: 1


1. Scope and pins
=================

1.1 Scope
---------

Lane A covers:

* the twelve modules declared in ``src/forcing/forcing.hpp``. That covers
  gravity's interface, but not ``const_gravity.cpp`` itself (K-1);
* the AAM physics in hydro and coord;
* the source-term interface;
* diffusion as a register contributor.

The rotation data itself is lane S's (S-1, section 3.8).

The sections follow the round's common order. Appendix A answers Xi's N1-N18,
and appendix B answers Zoey's review.

1.2 Pins and sources
--------------------

* snapy ``9e5de2d4``: all ``src/``, ``tests/`` and ``python/`` references.
* Lane R header at ``543ff84f``: the register contract. Its names and
  signatures replace every placeholder name of rev 2 part 1.
* Lane X design at ``77c65d5a``: the global phase, segments and line rows.
* kintera ``3f5ff938`` (section 8). pyharp ``6bc01d53`` and pydisort
  ``9e7f0075`` are not used by lane A.
* Lane I's answer (``vic.*`` names, the 18 forcing lines) and lane S's
  answers S-1..S-3, both from their threads.
* PR #370 (merged into main at ``730b35007d16389c64116e4bf9d2fe93707e29e9``,
  after the pin): the cubed-sphere sync of the positivity carry (appendix B).
* PR bodies of #355, #357, #363 and #365; issue #314.

Not read: the side-study Slack thread on the local-torque scheme, and Zoey's
formal Z1-Z8 text (appendix B answers the points of her report as summarised
in the lane A thread).

1.3 The ten register requirements (brief 2/3)
---------------------------------------------

1. One owner of every conservative face flux, one documented frame and area
   weighting.
2. Named increments in a declared order; total and parts reportable.
3. Stage fluxes accumulated with the integrator's stage weights into the step
   transfer.
4. Flux divergence and every flux-derived term read the register and never
   rebuild a flux.
5. A shared face has two entries and one reconciliation rule;
   single-valuedness is a gate; rotation is one function owned by coord.
6. One device-side monitor; no host reads inside a segment.
7. Every fixer and floor registers what it added.
8. A stated dtype under FP32, and fixed storage a CUDA graph can capture.
9. No added full-field copy per stage; memory and cost stated.
10. The register is rebuilt, not written to restart.

1.4 Register names lane A uses (lane R header, 543ff84f)
--------------------------------------------------------

=========================================  ==========  ====================================================
name or call                               R:line      what lane A does with it
=========================================  ==========  ====================================================
``RegisterRequest::keep``                  56-60       asks for ``riemann.central@2``, ``@3`` (``exact``)
                                                       and ``vic.momentum_t@1`` (R-4)
``riemann.central``                        9-12        one summed mass row (dry + species), from the
                                                       solver's own face states, totals under
                                                       perturbation flux, valid to the next
                                                       ``begin_stage``
``FluxRegister::total(d)``                 20, 89      the reconciled, limited and carried total, read in
                                                       the divergence phase (flux form) and in
                                                       ``post_implicit`` (m1)
``FluxRegister::kept(d, part)``            90          ``riemann.central``, ``vic.momentum_t``
``FluxRegister::add(dim, name, rows, x)``  74-75       ``diffusion`` posts (phase 3)
``FluxRegister::post_split(...)``          85-87       split radial conduction, weight 0.5 per half step
``FluxRegister::cell_amount(name, vdu)``   83          per-source amounts, the closure, floors
``RegisterPhase``                          43-51       sources in ``pre_implicit``; m1 rows in
                                                       ``post_implicit``
``class Fixer``                            106-116     base of ``AxialClosureFixer``; registry is lane X's
``RegisterMonitor``                        96-104      R owns the monitor; lane A's AAM column is R-3
=========================================  ==========  ====================================================

The posting order lane A relies on is R:25-28: x1 riemann < sedimentation <
x1 seam replace < x2/x3 riemann and covariance < positivity < divergence <
forcings and gravity work < implicit < post_implicit < end_stage.

Lane I's names: ``vic.mass_total`` (kept, R:49), and the posts
``vic.mass_dry``, ``vic.mass_species`` and ``vic.momentum_t[2]``.

1.5 Decisions recorded
----------------------

* **Settled by Cheng:**

  * ``full`` becomes ``exact``;
  * the closure key is removed;
  * the old ``full`` is refused with a message naming ``exact``;
  * the closure is a registered fixer inside ``exact``.
* **Settled with lane S** (lane S thread): coord owns the rotation data;
  registered fluxes stay in the panel frame; diffusion keeps its own exchange
  until phase 3.
* **FOR CHENG** (nothing waits on them; section 10.3):

  * abort versus redo for the radial check;
  * Coriolis ``type: 123``;
  * the AAM key on grids with no axis;
  * a non-z rotation axis.
* **Default:** ``off``.


2. Concepts and current sites
=============================

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

Of ``forcing.hpp``, 326 lines are the twelve option structs: 47 ``ADD_ARG``,
20 ``//!`` comments and 259 boilerplate. Another 270 lines are the module
classes.

2.2 AAM code outside ``src/forcing``, and who books it
------------------------------------------------------

Counted with ``sed -n 'a,bp' FILE | wc -l`` for each range. The bookings
follow the round's ownership rules: ``hydro.hpp`` is R's, ``src/mesh`` is X's,
and coord, including the rotation data, is S's (S-1).

================================================  =======  =======  =======================================
range                                             lines    booked   what
================================================  =======  =======  =======================================
hydro_forward.cpp:988-1003, 1108, 1141                 18  A        full: second Riemann eval (central)
hydro_forward.cpp:1288-1298                            11  A        transport: AAM flux form call
hydro_forward.cpp:1316-1363, 1390-1391                 50  A        Coriolis gate, x1 mass-w buffers, F_d
hydro_forward.cpp:1624-1680                            57  A        step 8: x1 mass-flux w, metric source
hydro.cpp:27-37, 55-76                                 33  A        mode predicates and checks
gnomonic_equiangle.hpp:84-108, 117-122                 31  A        decls: x1 sources, flux form, rate
gnomonic_equiangle.cpp:526-604                         79  A        ``axial_x1_source_``, flux form
gnomonic_equiangle.cpp:664-719                         56  A        ``x1_mass_source_``, rate
**lane A subtotal**                               **335**
gnomonic_equiangle.hpp:22-35, 109-116                  22  S        AAM buffers, rate_to_rows decl
gnomonic_equiangle.cpp:18-36, 169-221, 485-525        113  S        rotation field k, face averages,
                                                                    rate_to_rows
coordinate.hpp:370-392                                 23  S        four virtual AAM / x1-source hooks
hydro.hpp:89-133, 219-236, 462-466                     68  R        AAM options (docs), accessors, state
meshblock.cpp:983-990, 1009-1034                       34  X        closure reduction and increment
mesh.cpp:364, 367-382                                  17  X        closure over local blocks
================================================  =======  =======  =======================================

Rev 2 part 1 counted 493 for A. The 135 gnomonic rotation-data lines and the
23 ``coordinate.hpp`` lines now go to S (S-1). That leaves
493 − 135 − 23 = 335, which is the sum of A's ranges:
18 + 11 + 50 + 57 + 33 + 31 + 79 + 56 = 335.

2.3 What the code does today
----------------------------

* **Forcings are a flat list of** ``AnyModule`` **objects**
  (``register_forcing_modules.cpp:6-86``) called in order on a shared ``du``
  (``hydro_forward.cpp:1378-1385``).
* **Coriolis has two classes.** ``Coriolis123`` (``coriolis.cpp:40-67``) and
  ``CoriolisXYZ`` (``:69-231``). They agree only in 3D: ``123`` adds the IVZ
  row only when ``w.size(1) > 1`` (``:61-63``).
* **#355:** φ is a mass rate (``hydro_forward.cpp:1637-1646``), and ``lo``
  and ``hi`` hold ``1/(2rV)`` (``:1340-1342``), so
  ``m1 = φ_lo·lo + φ_hi·hi`` (``:1652-1653``).
* **The positivity limiter** scales the species flux rows
  (``hydro_forward.cpp:1165-1190``) and keeps the withheld mass's energy and
  momentum in the donor cell (the carry). #370 makes that carry consistent
  across cubed-sphere seams; at the pin, the AAM row of
  ``test_conservation_budget`` with sedimentation is RED (#370 body).
* **full (#363)** rebuilds a central flux from the post-terrain states
  (``hydro_forward.cpp:997-1003, 1089-1090, 1108``). The solver reads the
  same tensor (``:1109``).
* **closure (#365)** has host copies (``coriolis.cpp:204-207``;
  ``meshblock.cpp:1011, 1018``; ``mesh.cpp:371``).
* **Diffusion** keeps its own fluxes, seam rule and divergence. The radial
  conduction is split, and its check aborts (``diffusion_radial.cpp:97-100``).
* **Floors:** the user-forcing species clip (``meshblock.cpp:879-884``).
* **Restart** writes every defined ``Variables`` entry
  (``restart.cpp:31-37``).


3. Target design
================

3.1 AAM as a consumer of the register (Q1)
------------------------------------------

AAM reads the register and never evaluates a flux (requirement 4):

=====  ===============================================  ===========  ======================================
id     read                                             phase        used by
=====  ===============================================  ===========  ======================================
A-r1   ``reg.total(d)``, d = 1, 2, 3: reconciled,       divergence   transport and exact: K-component in
       after the positivity scale and carry                          flux form; torque of F_d
       (R:16-19, 25-28)
A-r2   ``reg.kept(d, "riemann.central")``, d = 2,       divergence,  exact: F_d = total − central; closure
       3: one summed mass row, never scaled             pre_impl.    residual R
       (R:18-19), valid to the next ``begin_stage``
A-r3   ``reg.total(1)`` mass rows (dry + species) in    post_impl.   m1 (#355): explicit plus the
       ``post_implicit``, which holds the                            ``vic.mass_dry``/``vic.mass_species``
       ``vic.*`` posts (R:49; R-4)                                   posts
A-r5   ``reg.kept(1, "vic.momentum_t")`` (R-4)          post_impl.   ``x1_source``; drops the prefix sum
                                                                     (``gnomonic_equiangle.cpp:541-543``)
=====  ===============================================  ===========  ======================================

**F_d is the limited mass flux minus the pre-limiter central flux** (appendix
B, item 1). The total is scaled by the positivity limiter (R:76-78) and
gains its carry (R:16). The kept central part is never scaled (R:18-19). So
``F_d = total(d) mass rows − kept central`` contains the covariance, the
withheld species mass and everything else that is not central. This is what
today's ``_flux2 − cen2`` holds (``hydro_forward.cpp:1359``, after the limiter
at ``:1165-1190``).

**The flux form reads the momentum rows after the limiter's carry.** Lane A's
K-rewrite runs on ``du`` in the divergence phase, which comes after
positivity (R:25-28, 46-47). This is correct only with #370's seam-consistent
carry; hence the gate reference in 5.3.

Lane A therefore needs from lane R exactly the R:25-28 order:

* ``riemann`` with ``keep`` before covariance and positivity;
* positivity's scale and carry before divergence;
* ``vic.*`` posts before ``post_implicit``.

Face states (placeholder): the header builds ``riemann.central`` "from the
solver's own face states" (R:10-11). Rev 1 asked for the post-terrain states.
A separate check is comparing the two. At the pin, lane A's own read finds one
tensor (``hydro_forward.cpp:1089-1090``, used by both ``:1108`` and
``:1109``). The result is recorded in R-1.

3.2 Modes (Q2), settled
-----------------------

**Modes: off | transport | exact (settled by Cheng).**

* The old ``full`` is refused with a message naming ``exact``.
* The closure key is refused by ``check_keys`` (``hydro_options.cpp:56``).
* ``full`` without the closure survives only as a test hook on
  ``AxialMomentum`` (arm (l), ``tests/test_axial_momentum_flux_form.py:332,
  565``).

Evidence, from the PR bodies (not rerun):

* ``transport`` is 50-290 times below ``off`` in 3D, but worse at
  Ω/(u/a) = 14;
* ``full`` is 217-589 times below ``off``, not exact, and costs 1.14-1.48
  times ``off``;
* the closure gives 1e-15 drift, and costs 1.04-1.09 times ``full``.

**Default: off.** Phases 1 and 2 are bitwise gated, and ``transport`` is worse
than ``off`` at high rotation. The AAM budget is always reported.

**Side-study torque:** a pluggable ``FaceTorqueRule``, with #363's rate as
rule 1.

3.3 The closure as a registered fixer (Q3)
------------------------------------------

``AxialClosureFixer : Fixer`` (R:106-116). The fixer registry is lane X's
(R:106).

* ``name()`` returns ``"aam.closure"``, and ``rows()`` returns {IVY, IVZ}.
* ``partial_sums()`` returns {R, I} as FP64 device tensors, with
  R = ``Σ V (K · F_cell − rate(kept central))``. It is computed once.
* ``apply(u, global_sums, dt, stage_weight, reg)``:

  * ``dt`` is the host double (N14);
  * it adds ``−w2 dt R / I · ρ s²`` along K, ``where``-masked;
  * it posts ``reg.cell_amount("aam.closure", V du)`` (R:83, 115).
* Lane X reduces the fixed 7-vector ``{R, I, D, m, wall, mwall, redo}``
  through the host and never captures it (X.rst:98, 187; N5, N14). The fixer
  point is after the saturation adjustment and before the boundaries.

3.4 One Coriolis and metric source (Q4)
---------------------------------------

``force = −2 Ω × m``, with ``m = (m1, ρ v2, ρ v3)``, plus the metric pair
``−m1 v_h / r``. The source is registered in two phases:

* ``RegisterPhase::pre_implicit`` (m1 = 0; through the scheme-9 solve);
* ``RegisterPhase::post_implicit`` (the m1 rows, R:50).

``m1 = φ_lo·lo + φ_hi·hi``, where φ is the face area times the mass rows of
``reg.total(1)`` read in ``post_implicit`` (A-r3). This already contains the
``vic.mass_dry`` and ``vic.mass_species`` posts, so lane A does not add
``vic.mass_total`` (no double count; R-4 confirms).

``omega_local`` and the m1 weights ``lo``/``hi`` are lane A's (S-1). They
are built at reset from coord's rotation data and the x1 geometry.

3.5 FP32 restriction on exact (Q5)
----------------------------------

See 6.2.

3.6 One source-term interface (Q6)
----------------------------------

Every module is a ``SourceTerm`` and declares:

* ``phase``, a ``RegisterPhase``: ``pre_implicit`` or ``post_implicit``;
* ``split``;
* ``rows``;
* ``kind``;
* ``region``;
* ``capturable`` (N14).

``SourceRegistry::runs(phase)`` returns maximal runs of equal
``capturable``, in declaration order, so sources are never reordered.

=======================  ===================================  =====================================
today                    V3                                   change
=======================  ===================================  =====================================
ConstGravity             SourceTerm (interface only)          migration owner: K-1
Coriolis123/CoriolisXYZ  ``Coriolis`` (one class)             3.4; type 123 FOR CHENG
Diffusion                ``DiffusionFlux`` + split radial     3.7
BodyHeat                 ``BodyHeat``                         base class does masks and options
TopCool / BotHeat        ``LayerHeat{side}``                  two 88-line files -> one
TopSpongeLyr /           ``Sponge{side}``                     two 78-line files -> one
BotSpongeLyr
RelaxBotTemp / Velo /    three classes on ``BottomSlab``      ``RelaxBotComp`` is
Comp                                                          ``capturable = false`` (section 8)
PlumeForcing             ``PlumeForcing``                     unchanged arithmetic
=======================  ===================================  =====================================

**User stage forcings stay in lane X's block compute segment** (X.rst:97;
N16). Their species clip is a floor registered by X.

3.7 Diffusion as a register contributor (Q7), from phase 3
----------------------------------------------------------

**Phase 3:** ``DiffusionFlux`` posts ``diffusion`` with ``reg.add`` (R:74-75):
the viscous momentum flux and the energy flux ``−k ∇T · n − v_f · τ · n``.
Its part is an FP64 monitor amount only (R:12).

* The cubed-sphere traction is rotated into the panel frame by coord's rotation
  (S-2).
* Diffusion keeps its own exchange (``share_fluxes``, then the 7-channel
  penalty traces) until phase 3 (S-3 accepted). From phase 3, the flux
  averaging is R's ``replace`` with S's transport (R:79-80), and the penalty
  traces keep diffusion's own exchange.
* **Radial conduction:** each half step posts its x1 heat flux with
  ``reg.post_split(1, "diffusion.radial", flux, 0.5, dt)`` (R:85-87). Its
  failure is a device flag (N3).

**Diffusion enters the register in phase 3, not phase 1.** Phases 1 and 2 are
bitwise gated; diffusion has its own tolerance suite (G8); and one PR should
not mix a register defect with a diffusion defect. Until then, ``exact`` with
``nu_iso > 0`` is refused.

3.8 Where the AAM geometry lives (Q8), settled with lane S
----------------------------------------------------------

**S-1 accepted.** Lane S's coord owns the rotation and the rotation data:

* the axis;
* ``k`` and its exact face averages;
* ``cos² lat`` at faces;
* the rate-to-rows projection;
* ``has_on_axis_cell``, set at reset, which replaces the host read at
  ``gnomonic_equiangle.cpp:506``.

Lane A reads them through ``RotationGeometry`` (skeleton) and owns the AAM
physics: the flux form, the x1 sources, the Coriolis rate, ``omega_local``,
the m1 weights, the torque rules and the closure. These move into
``src/forcing/axial_momentum.cpp`` and ``coriolis.cpp``.

**A non-z axis is out of V3 scope.** The axis is a declared parameter; only z
is supported, and others are refused (FOR CHENG, 10.3).


4. Flux-register contract
=========================

4.1 Lane A against requirements 1-5 and 10
------------------------------------------

**Requirement 1.** From phase 3 lane A owns one post, ``diffusion``. It is a
flux density per unit area and time at the step dt, in the register's
components (R:5-8). The area weighting is the register's divergence. Registered
fluxes stay in the panel frame, and S rotates at seams (S-2).
``A F · K_f`` is derived, never posted.

**Requirement 2.**

* Lane A posts ``diffusion`` (phase 3) and ``post_split`` entries, and keeps
  ``riemann.central`` and ``vic.momentum_t`` (R:56-60).
* The order is R:25-28 (3.1).
* Lane A's parts are FP64 monitor amounts (R:12).

**Requirement 3.**

* R's ``end_stage`` weights the stage total with
  ``c_s = w2_s Π_{t>s} w1_t`` (R:13-15).
* The closure applies ``stage_weight`` inside ``Fixer::apply``.
* The radial conduction posts through ``post_split`` with weight 0.5 per half
  step (R:85-87).

**Requirement 4.**

* The flux form and F_d read ``total`` and ``kept``.
* m1 reads ``total(1)`` in ``post_implicit``.
* ``x1_source`` reads ``kept(1, "vic.momentum_t")``.
* The two extra Riemann calls (``hydro_forward.cpp:997-1003, 1108, 1141``)
  and the ``_flux1`` times area rebuild (``:1637-1646``) go.
* Diffusion keeps its own divergence until phase 3.

**Requirement 5.**

* AAM reads the reconciled total.
* ``K_f`` comes from coord's rotation data in the panel frame (S-2).
* **Gate G5 has a precondition:** R's ``replace`` (R:79-80) and S's seam
  rotation and transport must have landed. Until then G5 is not run.
* Diffusion keeps its own exchange until phase 3 (S-3).

**Requirement 10.** Nothing lane A adds is restart-written:

* lane A adds no ``Variables`` entry (``restart.cpp:31-37``);
* R rebuilds the register (R:23-24);
* lane A's buffers (``omega_local``, the m1 weights, ``scratch_``, ``aam_``)
  are rebuilt at reset.

4.2 Compliance table
--------------------

====  ========  ==================================================================
req   section   status for lane A
====  ========  ==================================================================
1     4.1       complies; panel frame (S-2); diffusion owner from phase 3
2     4.1       complies with R:9-12, 25-28, 56-60; R-4 (vic.momentum_t keep)
3     4.1       complies: R:13-15, Fixer::apply, post_split (R:85-87)
4     4.1       complies from phase 2; diffusion from phase 3
5     4.1       complies from phase 3; G5 after R's replace and S's seam rotation
6     5.1, 7.1  complies after the 7.1 host reads go; monitor is R's; AAM
                column R-3
7     5.2       complies: ``AxialClosureFixer : Fixer`` posts ``cell_amount``;
                species clip registered by X
8     6, 7.1    complies; kept-central dtype in FP32 is R-5
9     7.2       memory stated from shapes; cost is a measurement plan
10    4.1       complies: nothing restart-written (R:23-24)
====  ========  ==================================================================


5. Invariants and gates
=======================

5.1 Monitor (requirement 6)
---------------------------

The monitor is lane R's (R:21-22, 96-104).

* On monitor steps each lane-A source writes into the registry's one FP64
  ``scratch_``, zeroed per source.
* The registry posts ``reg.cell_amount(name, V·scratch)`` (R:83), then adds the
  scratch once to ``du`` (FP64 run) or ``_forcing_work`` (FP32 run).
* Bitwise: every lane-A source writes each element at most once and none reads
  ``du``. The exception outside lane A is ``const_gravity.cpp:52, 58, 63``.
* R's amounts are per conserved row (R:99). The AAM share ``Σ V r K · dS``
  needs coord's rotation data, so lane A keeps ``aam_[nsrc]`` (FP64 device)
  until R answers R-3.
* Budgets on and budgets off are two graph topologies.

5.2 Fixers and floors (requirement 7)
-------------------------------------

=============================  =======  =============  ==========  ========================================
fixer / floor                  kind     rows           registers   declares
=============================  =======  =============  ==========  ========================================
AAM closure (``exact``)        fixer    IVY, IVZ       A (Fixer)   ``cell_amount("aam.closure", V du)``;
                                                                   AAM change ``−w2 dt R`` (G6)
user-forcing species clip      floor    ICY.., IDN     X           species mass withheld per stage
side-study torque (rule 2)     numer.   IVY, IVZ       A           its AAM rate; no reduction
=============================  =======  =============  ==========  ========================================

5.3 Gates (specified, not run)
------------------------------

**Reference.** Bitwise and tolerance gates compare against chengcli main at or
after ``730b35007d16389c64116e4bf9d2fe93707e29e9`` (#370), not against the
pin. At the pin, the AAM row of ``test_conservation_budget`` with sedimentation
is RED (#370 body). The flux form reads after the carry, so an AAM gate on a
sedimentation deck needs #370's seam-consistent carry. File and line citations
in this document stay at the pin.

====  ==============================================  ===============================================  =========  =====
id    gate                                            case                                             kind       phase
====  ==============================================  ===============================================  =========  =====
G1    existing tests against the reference            full ctest list, CPU, FP64 and FP32 storage,     bitwise    1
                                                      budgets on and off
G2    kept ``riemann.central`` vs today's central:    AAM test (``test_axial_momentum_flux_form``)     tolerance  2
      e_∞ = max_f abs(ΔF_c) / max_f ρ(abs(u_n) + c)   W2 C16 and 3D arm decks, one stage, FP64, all
      ≤ 1e-13; L2 analogue reported                   x2/x3 faces
G3    ``exact`` AAM drift ≤ 1e-12 relative            arm (l): 3D at Ω = 5e-6 and 7e-5; W2 at 1 Ω and  tolerance  2
                                                      4 Ω
G4    ``off`` and ``transport`` against the           arms (a)-(c), (f)-(h); on sedimentation decks    bitwise    2
      reference                                       see the EF-3 rule below                          (rule)
G5    seam single-valuedness of ``A F · K_f``         W2 C16, one stage, every panel-seam face; after  1e-14 rel  2
                                                      R's replace and S's seam rotation land
G6    closure: declared vs measured AAM change        arm (l) decks                                    1e-12 rel  2
G7    sponge / drag torque booked = AAM change        ``relax-bot-velo`` drag arm and a top-sponge     1e-15 of   2
                                                      arm, FP64, budgets on                            L
G7s   AAM with sedimentation                          ``test_conservation_budget`` row ``d_sed``       C n eps,   2
                                                      (#370) with ``transport`` and ``exact``          C = 4
G8    diffusion suite                                 ``test_diffusion_cubed_sphere``,                 own tol.   3
                                                      ``test_diffusion_x1_scale``, convection-onset
                                                      notes
G9    viscous torque booked as physical               new arm: ``exact`` with ``nu_iso > 0``, W2 C16   1e-12 rel  3
G10   FP32 + perturbation-flux AAM drift vs FP64      arms (f)-(h), (k), (l) in FP32 with              drift      4
                                                      perturbation-flux                                ≤ 1e-12
====  ==============================================  ===============================================  =========  =====

**EF-3 rule for G4** (appendix B, item 3). G4 is bitwise against a reference
that contains every change merged before lane A's phase-2 PR. If lane EF's
rounding change for sedimentation (EF-3) is in the reference, G4 stays bitwise
on sedimentation decks. If EF-3 lands after A's phase 2, or in the same merge
window, G4 on sedimentation decks becomes the ``C n eps`` (C = 4) tolerance of
``test_conservation_budget`` (#370's convention), and stays bitwise on decks
without sedimentation. The deck list is fixed in the phase-2 PR.

5.4 Hypotheses and how each is decided
--------------------------------------

===  ===========================================================  ==============================================
id   hypothesis                                                   decided by
===  ===========================================================  ==============================================
H1   kept ``riemann.central`` equals today's central mass flux    gate G2 (tolerance, FP64)
     in FP64
H2   ``exact`` costs about ``transport`` plus one rate; the       measurement plan M1 (7.2), after approval
     closure keeps one host round trip per stage
H3   the error of today's ``F(w, w)`` scales as eps·ρ·c (N4)      reported with G2; no separate gate
H4   in FP32 the face-state central part and ``F(w, w)``          gate G10 (tolerance)
     differ by about TINY_NUMBER·ρ, harmlessly
H5   diffusion in the register makes the viscous torque           gate G9 (tolerance)
     telescope
H6   the per-source scratch is bitwise against the reference      gate G1 (bitwise), budgets on and off
H7   removing the per-stage host reads (7.1) changes no bits      gates G1, G4 (bitwise)
H8   NCCL or commux/UCX capture of the fixer collective           not needed: X never captures it (N14)
H9   on MPS (no FP64; ``meshblock.cpp:1057``) {R, I} can be       unverified; no MPS gate
     reduced on the host in FP64
H10  #370's carry makes G3 and G7s pass on sedimentation decks    gate G7s (tolerance)
===  ===========================================================  ==============================================


6. FP32
=======

6.1 Stated dtypes (requirement 8)
---------------------------------

==========================================  ================  =========================================
quantity                                    FP64 run          FP32 run (``precision: float32``)
==========================================  ================  =========================================
source increments                           FP64, into du     FP64 ``_forcing_work``, rounded once
                                                              (``hydro_forward.cpp:1365-1375``)
per-source ``scratch_`` (monitor steps)     FP64              not used: amounts from ``_forcing_work``
``aam_[nsrc]``                              FP64              FP64
kept ``riemann.central``                    FP64              unstated in R's header (R-5); lane A
                                                              forms ``F − F_c`` and the rate in FP64
R's step accumulator                        FP64              FP64 (R:13-14)
AAM flux form, torque rate, R and I         FP64              FP64; summed in FP64 before reduction
``diffusion`` part                          FP64 amounts      FP64 amounts (R:12)
rotation data                               FP64              lane S's master and cast views
==========================================  ================  =========================================

6.2 What lifts the restriction on exact (Q5)
--------------------------------------------

Three things lift today's refusal (``hydro_forward.cpp:993-995``) together:

1. a consistent central part, which the header gives as totals under
   perturbation flux (R:11);
2. FP64 accumulation of the rate and of R and I;
3. gate G10.

Until G10 passes, ``exact`` refuses FP32 and perturbation-flux at reset.


7. CUDA graph
=============

7.1 Capturable storage and host reads (requirements 6 and 8)
------------------------------------------------------------

* The lane-A buffers have fixed shapes and are allocated at reset:
  ``omega_local``, the m1 weights, ``scratch_``, ``aam_`` and ``{R, I}``.
* The kept parts live in R's storage (R:64-66).
* ``dt`` is a host double, read at step start (X.rst:93, 180; R:113-115).
* Sources with ``capturable = false`` run in eager segments.

======================================================  ===============================================
today                                                   V3
======================================================  ===============================================
``coriolis.cpp:204-207`` {R, I} to CPU per stage        ``Fixer::partial_sums()`` on device
``meshblock.cpp:1011``, ``mesh.cpp:371`` CPU zeros      device zeros (X's code)
``meshblock.cpp:1018`` two ``.item()``                  device scale in ``Fixer::apply``
``gnomonic_equiangle.cpp:506`` on-axis ``.item()``      ``has_on_axis_cell`` set at reset (S)
``diffusion.cpp:507`` ``torch::equal`` per call         frozen options, checked at build (lane O)
``diffusion_radial.cpp:97-100`` aborting check          device flag on the redo slot; action stays
                                                        abort (FOR CHENG: redo)
``diffusion.cpp:677-743`` ``max_time_step``             stays, in the driver's dt step
======================================================  ===============================================

7.2 Memory (stated) and cost (measurement plan, requirement 9)
--------------------------------------------------------------

Memory is stated from tensor shapes; nothing was measured.

============================================  =================================  ==============================
item                                          today (pin)                        V3
============================================  =================================  ==============================
``exact`` central flux, per stage             ``cen2``, ``cen3`` 2(5+ny) rows;   kept ``riemann.central``: 2
                                              transient 4(5+ny); stored          face rows (one per direction)
                                              ``F − F_c`` 2(5+ny)
``vic.momentum_t`` kept                       none (prefix sum in place)         2 x1-face rows (R-4)
per-source scratch                            none                               one FP64 field, 8·(5+ny) B per
                                                                                 cell, FP64 runs only
per-source amounts                            none                               R's monitor rows; ``aam_[nsrc]``
closure                                       2 scalars + ρ clone                2 scalars + ρ view
============================================  =================================  ==============================

No full-field copy is made per stage.

**Measurement plan (after approval, on the phase-2 and phase-3 PRs):**

* M1: time per step (minimum of 3 runs);
* M2: peak memory per step;
* M3: kernel launches per stage.

Each is reference against V3, for ``off``/``transport``/``exact``, with
budgets on and off. The decks are the AAM test arms, ``jupiter_gcm_dry`` C48
+ Coriolis on one GPU, and ``test_diffusion_cubed_sphere``. No GCM runs are
included.

7.3 Graph boundaries
--------------------

* The end-of-stage fixer point is lane A's only boundary. X reduces the
  7-vector through the host and never captures it.
* The pre- and post-implicit phases are segments around lane I's solve.
* ``capturable = false`` sources are eager segments.


8. Cross-repository interface changes and merge order
=====================================================

Lane A asks for **no interface change in any other repository**.

* **kintera (pin** ``3f5ff938`` **):** ``RelaxBotComp`` calls
  ``ThermoY::compute`` and ``ThermoX::compute`` (kintera
  ``src/thermo/thermo.hpp:162, 321``). The conversions it uses are "Y->X",
  "DY->V", "VT->U" (``thermo_y.cpp:198, 214, 242``) and "X->Y"
  (``thermo_x.cpp:141``); the call sites are ``relax_bot_comp.cpp:98-111``.

  * The ``compute`` bodies (``thermo_y.cpp:180-275``, ``thermo_x.cpp:139-203``)
    contain no ``.item()``.
  * Whether they can be captured is a hypothesis, so ``RelaxBotComp`` is
    ``capturable = false`` until lane K confirms (K-1).
* **pyharp, pydisort:** none. ``git grep`` finds neither in lane A's files.
* **commux/UCX:** none. The collective is lane X's.

Merge order (provider first), inside snapy:

1. lane O schema;
2. lane A phase 1;
3. lane R register (``543ff84f`` contract), lane I ``vic.*``, lane X global
   phase and segments, lane S rotation data and seam rotation;
4. lane A phase 2;
5. lane A phase 3;
6. lane A phase 4.

The reference for gates (5.3) is main with #370.


9. Line budget
==============

9.1 The agreed rows
-------------------

* **X** deletes ``meshblock.cpp:1009-1035`` and ``meshblock.hpp:223-228``
  (booked in X). By ``wc -l`` that is 27 + 6 = 33 lines; the round's figure is
  29.
* **A** drops its 51 mesh lines and adds ``AxialClosureFixer : Fixer`` in
  ``axial_momentum.cpp``.
* **A** counts the 7 lane-I lines in ``src/`` (``forcing.hpp:48, 56-58,
  60-62``); the 11 python lines are counted in 10.2.
* **S** books the 135 gnomonic rotation-data lines and ``coordinate.hpp``'s 23
  (S-1; recounted for this revision, 2.2).
* **O**'s schema is a precondition (N18).

9.2 Counted lines and the lane number (recounted after S-1)
-----------------------------------------------------------

S-1 moves ``omega_local`` and the AAM physics into lane A's files, so the
rows changed. The "now" column is recounted with ``wc -l`` and
``sed -n 'a,bp' | wc -l`` at the pin. The "after" column holds estimates.

=================================  ======  ======  =======================================================
file                               now     after   basis
=================================  ======  ======  =======================================================
forcing.hpp                         674     253    options 326 -> 150 via schema (O); classes 270 -> 70;
                                                   includes and comments 78 -> 40; −7 lane-I lines
source_term.hpp (new)                 0     110    base and registry
coriolis.cpp                        232     178    −28 (123), −38 (glue), +4 (two phases), +8 metric
                                                   ``−m1 v/r``; Ω set-up stays (``omega_local``, S-1)
axial_momentum.cpp (new)              0     225    consumer, torque rules, ``AxialClosureFixer`` (170) +
                                                   flux form (45) + x1 source from ``vic.momentum_t`` (10)
diffusion.cpp                       745     480    −70 options, −110 profile checks (schema), −40, −45
diffusion_cubed_sphere.cpp          376     320    phase 3: −35 averaging, −15 divergence, −6
diffusion_radial.cpp                103     103    ``post_split`` replaces nothing at the pin
top_cool + bot_heat                 176      70    ``LayerHeat{side}``
top_sponge + bot_sponge             156      50    ``Sponge{side}``
relax_bot_temp/velo/comp            308     235    ``BottomSlab`` base
body_heat + plume_forcing           100      70    base does ctor, reset, options
hydro_forward.cpp (AAM ranges)      136      25    reads register; two phase hooks
hydro.cpp (AAM ranges)               33      15    mode predicate; ``full`` refusal
gnomonic_equiangle.* (A physics)    166       0    moved into ``axial_momentum.cpp`` and ``coriolis.cpp``
**total**                          3205    2134
=================================  ======  ======  =======================================================

**Lane A: 3,205 → 2,134 lines (−1,071, −33.4 %).**

Arithmetic:

* now: 2870 + 136 + 33 + 166 = 3205. The gnomonic physics lines are
  ``.hpp:84-108, 117-122`` (31) and ``.cpp:526-604, 664-719`` (79 + 56).
* after: 253 + 110 + 178 + 225 + 480 + 320 + 103 + 70 + 50 + 235 + 70 + 25 +
  15 + 0 = 2134.
* cut: 1071 / 3205 = 0.334.

The ``src/forcing`` part alone goes 2870 → 2094 (−27.0 %), against −31.0 % in
part 1. The difference is the Ω set-up staying in lane A and the physics
moving in.

**Precondition (N18): lane O's options schema lands before lane A's phase 1.**
Without it, ``src/forcing`` would be 2,405 (−16.2 %) after phase 3 and 2,506
(−12.7 %) before it (+131 options, +70 parser, +110 profile checks; +101 for
phase 3). Margin deletions that need no schema: the ``type`` key (5 lines, if
Cheng removes type 123) and the legacy-key check ``diffusion.cpp:275-277``
(3 lines).


10. Migration
=============

10.1 Phases
-----------

1. **Phase 1** (G1): the ``SourceTerm`` base and registry, all twelve modules,
   the merges, one Coriolis, the per-source scratch and ``cell_amount``,
   ``capturable``. Needs lane O's schema.
2. **Phase 2** (G2-G7, G7s):

   * the AAM consumer reads ``total``/``kept``;
   * ``AxialClosureFixer : Fixer`` in X's registry;
   * ``exact`` replaces full + closure;
   * the host reads of 7.1 go;
   * lane A reads lane S's rotation data.

   Needs R, I, X, S, and #370 in the reference.
3. **Phase 3** (G8, G9): diffusion posts to the register; ``post_split``.
4. **Phase 4** (G10): FP32 and perturbation-flux for ``exact``.

10.2 Sites changed by the full → exact decision, by lane
--------------------------------------------------------

The sites, with each line counted once:

* **R:** 47 lines;
* **X:** 9;
* **A:** 152. Of these, 28 are in ``src``, and 124 are in python and tests.

Lane A also counts the 11 lane-I python lines (``pyforcing.cpp:29``,
``forcing.pyi:61-70``).

10.3 Deck and API changes
-------------------------

==========================================  ===================================  ==========================
change                                      replacement                          status
==========================================  ===================================  ==========================
``conserve-axial-momentum: full``           refused with a message naming        **settled** (Cheng)
                                            ``exact``
``conserve-axial-momentum-closure``         removed; ``exact`` includes the      **settled** (Cheng)
                                            closure; refused by ``check_keys``
``coriolis/type: 123``                      ``type: xyz``: identical in 3D       **FOR CHENG**
AAM key on a grid with no rotation axis     refused (today a no-op + warning)    **FOR CHENG**
non-z rotation axis                         refused; only z in V3 (S-1)          **FOR CHENG**
radial-conduction positivity failure        device flag; abort kept (default)    **FOR CHENG** (redo needs
                                            or redo                              cause bit 64 from X)
==========================================  ===================================  ==========================

10.4 Decision log
-----------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c is the kept ``riemann.central``                   no second flux evaluation (3.1; R:9-12)
D2  modes ``off | transport | exact``; full refused       settled by Cheng (3.2, 10.3)
D3  default ``off``; AAM budget always reported           bitwise migration (3.2)
D4  ``AxialClosureFixer : Fixer``; X's registry           requirement 7; R:106-116 (3.3)
D5  one Coriolis; m1 from ``total(1)`` in post_implicit   requirement 4; R:50 (3.4)
D6  FP32 lifted only by G10                               6.2
D7  per-source scratch + ``cell_amount``; ``aam_``        bitwise; R:83 (5.1)
D8  diffusion into the register in phase 3                bitwise gates of phases 1-2 (3.7)
D9  ``exact`` with ``nu_iso > 0`` refused until phase 3   closure would remove a physical torque
D10 rotation data in S's coord; physics in lane A         S-1 (3.8)
D11 user forcings stay in X's segment                     order and G1 (3.6; N16)
D12 ``capturable`` flag, ordered runs                     X's segment flag (3.6; N14)
D13 lane A writes nothing to restart                      requirement 10 (4.1)
D14 lane O schema before lane A phase 1                   20 % target (9.2; N18)
D15 gates against main with #370                          carry consistency (5.3; appendix B)
=== ====================================================  ==========================================


11. Open questions to other lanes
=================================

Answered since part 1:

* **R-2** by the header: ``post_split`` (R:85-87).
* **R-1, mostly:** one summed row, ``keep``, totals under perturbation flux,
  valid to the next ``begin_stage``, and parts as monitor amounts
  (R:9-12, 56-60).
* **Fixer base and registry:** ``class Fixer`` (R:106-116); the registry is
  lane X's.
* **S-1, S-2, S-3** (lane S thread).
* **I-1** (lane I thread).

Open:

* **R-1, Tianhao (placeholder).** The header builds ``riemann.central`` "from
  the solver's own face states" (R:10-11); rev 1 asked for the post-terrain
  states. A separate check compares them. At the pin, lane A's read finds one
  tensor (``hydro_forward.cpp:1089-1090, 1108-1109``). The final result and
  any conflict note go in the lane R thread.
* **R-3, Tianhao.** ``RegisterMonitor::amounts`` is per conserved row (R:99).
  Requirement 6 includes AAM, which needs ``r K · m`` from coord's rotation
  data. Does R's monitor add an AAM column (reading S's rotation data), or does
  lane A keep its ``aam_[nsrc]`` shares?
* **R-4, Tianhao.** Can lane A keep ``vic.momentum_t@1``? The x1 source needs
  the implicit tangential transfer alone, while ``total(1)`` mixes explicit and
  implicit. And does ``total(1)`` read in ``post_implicit`` hold the
  ``vic.mass_dry``/``vic.mass_species`` posts (R:49), so that lane A does not
  add ``vic.mass_total``?
* **R-5, Tianhao.** What is the dtype of a kept part under FP32? R:12 states
  FP64 only for monitor amounts.
* **X-1, Xi.** Can the radial flag ride the redo slot
  (``meshblock.cpp:1053-1056``)? Will X take cause bit 64 if Cheng picks redo?
* **K-1, chen sihe.** Does lane K migrate ``ConstGravity`` (its work and the
  gravity-work fixer)? Can kintera's four conversions (section 8) be captured?

FOR CHENG: see 10.3.


12. Skeleton index
==================

``wc -l docs/v3/skeleton/A/*`` gives 294 lines, with declarations only, in
namespace ``snap``. The headers include lane R's ``flux_register.hpp``. A
syntax check against R's header at ``543ff84f`` passes.

==========================  ==============================================================  ===============
file (lines)                declarations                                                    serves
==========================  ==============================================================  ===============
``source_term.hpp`` (92)    ``SourceKind``, ``SourceRows``, ``SourceRegion``,               3.6, 5.1, 7.1
                            ``SourceDeclaration`` (``RegisterPhase phase``, ``split``,
                            ``capturable``), ``StageState``, ``SourceTerm``,
                            ``SourceRegistry`` (``Run``, ``runs``, ``run``,
                            ``scratch_``, ``aam_``)
``axial_momentum.hpp``      ``AxialMode``, ``AxialFluxView`` (``total``, ``central``,       3.1-3.3, 5.2,
(81)                        ``momentum_t``), ``FaceTorqueRule``, ``AxialMomentum``          6.2, 7.3
                            (``request``), ``AxialClosureFixer : Fixer``
``coriolis_diffusion.hpp``  ``Coriolis`` (``omega_local_``, ``m1_lo_``, ``m1_hi_``),        3.4, 3.7, 3.6
(89)                        ``DiffusionCoefficients``, ``DiffusionFlux``,
                            ``DiffusionSource``, ``RadialConduction`` (``post_split``,
                            ``failed_``), ``Sponge``, ``LayerHeat``, ``BottomSlab``
``rotation_geometry.hpp``   ``RotationGeometry``, lane S's: ``axis``, k fields,             3.8
(32)                        ``rate_to_rows``, ``has_on_axis_cell``
==========================  ==============================================================  ===============


Appendix A: responses to Xi's notes N1-N18
==========================================

N14 withdraws N2 and supersedes N5 where they overlap.

* **N1 (accept).** Host copies are listed in 7.1, and {R, I} come from
  ``Fixer::partial_sums()`` on the device.
* **N2 (withdrawn by N14).** ``dt`` is a host double (R:113-115).
* **N3 (accept).** The radial check becomes a device flag; the action stays
  abort, and redo is FOR CHENG plus lane X (X-1).
* **N4 (accept).** G2 uses the normalised e_∞; the error scale is H3.
* **N5 (accept; superseded in part by N14).** The 7-vector and its placement
  stand; the collective goes through the host and is never captured. The
  registry is X's (R:106).
* **N6 (accept, changed by R's header).** R owns the monitor (R:21-22).
  Per-source amounts go to ``reg.cell_amount`` (R:83) instead of a lane-A
  ``budget_``. The AAM share is ``aam_`` until R-3 is answered.
* **N7 (accept).** The skeleton's references are remapped, and the m1 comment
  is corrected (no ``/ dt``).
* **N8 (accept).** The no-schema basis is stated in 9.2.
* **N9 (accept).** The kept central part comes from the solver's states (3.1,
  R-1 placeholder). F_d contains the covariance and the limiter scaling.
* **N10-N12 (agree).** No change.
* **N13 (accept).** The type-123 row says "identical in 3D". It stays FOR
  CHENG.
* **N14 (accept).** "No host sync" is removed; ``capturable`` and ordered runs
  are added.
* **N15 (replaced by the later agreement).** X deletes
  ``meshblock.cpp:1009-1035`` and ``meshblock.hpp:223-228``. A drops its 51
  mesh lines and adds ``AxialClosureFixer``.
* **N16 (accept).** User forcings stay in X's segment, and X counts
  ``meshblock.cpp:819-884``.
* **N17 (accept with change).** The per-source FP64 scratch keeps G1/G4
  bitwise, and ``cell_amount`` carries the amounts.
* **N18 (accept, corrected basis).** The schema is a precondition. With S-1,
  the no-schema numbers are worse (9.2).


Appendix B: Zoey's round-2 review
=================================

Her formal notes Z1-Z8 were not in my inputs. The points below come from her
report as summarised in the lane A thread, and will be renumbered to Z1-Z8
when the formal notes are posted.

1. **Must: the AAM flux form reads the momentum rows after the limiter's
   carry; F_d is the limited mass flux minus the pre-limiter central flux.**
   Accepted and made explicit in 3.1 and 4.1.

   * The flux form runs on ``du`` in the divergence phase, after positivity's
     scale and carry (R:16-19, 25-28).
   * ``F_d = total(d) mass rows − kept riemann.central``. The total is limited,
     and the kept part is never scaled (R:18-19).
   * The posting order lane A needs from R is R:25-28, unchanged.
2. **Must: use main with #370 as the gate reference, not the pin.** Accepted
   (5.3, D15). Gates compare against main at or after ``730b3500``; at the pin,
   the AAM row with sedimentation is RED. A new gate, G7s (``d_sed``, C n eps,
   C = 4), covers sedimentation with ``transport`` and ``exact``.
3. **Should: the transport bitwise gate vs EF-3 rounding on sedimentation
   decks.** Accepted with a rule (5.3). G4 is bitwise against a reference that
   contains EF-3 when EF-3 merged first. Otherwise G4 uses C n eps (C = 4) on
   sedimentation decks and stays bitwise elsewhere.
