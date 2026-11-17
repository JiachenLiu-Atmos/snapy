=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 4, part 1 (design and contract only, CHANGE 5): the radial rows
           (failure masking, X's retry contract, the diffusion_radial.cpp
           booking; 1.6). Rev 3 final (2b0ac426) is cited to lane R's frozen
           header 4c59162a. src/forcing 2,870 -> 2,087.
:Reviewers (round 2): Xi (lane X), Zoey
:Pins: snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``; kintera ``3f5ff938``
       (section 8); lane R frozen header ``happysky19/snapy@4c59162ae37080d0ab307b2823742b44f46da7c3``
       ``docs/v3/skeleton/R/flux_register.hpp`` (cited "R:line") and R.rst at
       the same sha (cited "R.rst:line"); lane EF ``zoeyzyhu/snapy@c21dd66e``
       (cited "EF.rst:line"); lane X
       design ``77c65d5aac1f96ff6c34a45ef48faeb996c2128e`` (cited "X.rst:line")
:Previous: rev 1 ``2c9589c6``; rev 2 ``afd08eaa``, ``b6a40897``; rev 3
           part 1 ``9ee83756``
:Skeleton: ``docs/v3/skeleton/A/``, four headers in namespace ``snap``,
           declarations only, 299 lines (``wc -l docs/v3/skeleton/A/*``);
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
* Lane R frozen header at ``4c59162a`` (R rev 2.2): the register contract.
  Every R citation in this revision is re-cited from ``543ff84f`` to
  ``4c59162a`` and checked against it (rev 3 final).
* Lane EF design at ``c21dd66e`` (EF rev 3.1): floors, the positivity posts P2
  and P3, ``EfStatus``, and its questions Q-A1/Q-A2 to lane A.
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

1.4 Register names lane A uses (lane R frozen header, 4c59162a)
---------------------------------------------------------------

=========================================  ==========  ====================================================
name or call                               R:line      what lane A does with it
=========================================  ==========  ====================================================
``RegisterRequest``                        32-39       ``keep`` (35): ``riemann.central@2``, ``@3``
                                                       (``exact``), ``vic.momentum_t@1``, kept per face;
                                                       ``derived`` (34): ``"aam"``; ``retry`` (37):
                                                       ``"diffusion.radial"``; ``step_field`` (38): {1}
``riemann.central``                        35          a kept part written in place by C's kernel
                                                       (R.rst:357-359), one summed mass row on x2/x3
                                                       (R.rst:165-168), totals (R.rst:958-961), never
                                                       scaled (R.rst:455-457)
``FluxRegister::total(d)``                 82          the reconciled, limited and carried total, read in
                                                       the divergence phase (flux form) and in
                                                       ``post_implicit`` (m1)
``FluxRegister::kept(d, part)``            83          ``riemann.central``, ``vic.momentum_t``
``FluxRegister::step_field(1)``            84          the stage-weighted step face transfer per unit
                                                       area; lane A owns its K-weighted x1-wall reduction
``FluxRegister::add(dim, name, rows, x)``  57-58       ``diffusion`` posts (phase 3)
``FluxRegister::post_split(...)``          80-81       split radial conduction, weight 0.5 per half step;
                                                       after ``begin_step()`` (6, 48)
``cell_amount(name, vol_du, w, kind)``     67-70       per-source amounts (``w2_s``, ``cell_source``); the
                                                       closure (1.0, ``fixer``)
``derived_amount(name, row, v, w, kind)``  72-74       every AAM amount, on the derived row ``"aam"``
``derived_total(row, value)``              75          the step-end AAM total ``L``
``flag_retry(cause, any)``                 78          the radial-conduction failure flag
``begin_stage(int, IntegratorWeight)``,    11, 49-50   ``w2`` of the stage for ``stage_weight``; there is
``weights()``                                          no ``StageWeights`` type any more
``RegisterPhase``                          14-23       sources in ``pre_implicit``; m1 rows in
                                                       ``post_implicit``; fixers in ``after_combine``
``AmountKind``                             25-31       ``cell_source``, ``fixer``, ``diagnostic``
``class Fixer``                            104-116     base of ``AxialClosureFixer``; registry and
                                                       collective are lane X's (104)
``RegisterMonitor``                        89-103      R owns the monitor; AAM is the derived row
                                                       ``aam`` (R.rst:538, 698-703)
=========================================  ==========  ====================================================

The posting order lane A relies on is the ``RegisterPhase`` order (R:14-22):
sweep (riemann, sedimentation, covariance) < reconcile < limit (EF's carry P3,
then scale P2) < divergence < pre_implicit < implicit < post_implicit <
after_combine. The step frame is ``begin_step`` < ``post_split`` < stages <
``after_combine`` (R:6).

Lane I's names: ``vic.mass_total`` (kept, R.rst:165-167), and the posts
``vic.mass_dry``, ``vic.mass_species`` and ``vic.momentum_t[2]``.

1.5 Decisions recorded
----------------------

* **Settled by Cheng:**

  * ``full`` becomes ``exact``;
  * the closure key is removed;
  * the old ``full`` is refused with a message naming ``exact``;
  * the closure is a registered fixer inside ``exact``.
* **Settled with lane S** (lane S thread): coord owns the rotation data;
  registered fluxes stay in the panel face basis; diffusion keeps its own exchange
  until phase 3.
* **FOR CHENG** (nothing waits on them; section 10.3):

  * abort versus redo for the radial check;
  * Coriolis ``type: 123``;
  * the AAM key on grids with no axis;
  * a non-z rotation axis.
* **Default:** ``off``.

1.6 Revision 4 change log
-------------------------

* **Part 1 (radial rows; this commit):**

  * the radial failure contract in 3.7: the temperature increment is masked
    per column on failure, and abort stays the default;
  * lane X's retry contract (X rev 3.3 ``0fe47c6f``, X.rst:146-157) for the
    retry proposal, in 10.3 and 11.4. The cause is bit 6, not "bit 64" as
    rev 3 had it;
  * lane A books the ``diffusion_radial.cpp`` change (9.2, RULING V3-1):
    ``src/forcing`` goes 2,086 -> 2,087.
* **Still to come in rev 4:** Jiheng's H1-H17 by number, and Zoey's Z1-Z3,
  Z7 and Z8 (appendix B).


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

2.2 AAM code outside ``src/forcing``: cross-references (RULING V3-1)
--------------------------------------------------------------------

Under RULING V3-1, a deletion is counted by the lane that owns the file, and
lane A's number is ``src/forcing`` only (section 9). The ranges below are
lane A's AAM code in other lanes' files. They are listed as cross-references
for those lanes' budgets and are **not** in lane A's count. Counted with
``sed -n 'a,bp' FILE | wc -l`` at the pin.

================================================  =======  =======  =======================================
range                                             lines    booked   what
================================================  =======  =======  =======================================
hydro_forward.cpp:988-1003, 1108, 1141                 18  R        full: second Riemann eval (central);
                                                                    :988 is a heading that stays
hydro_forward.cpp:1288-1298                            11  R        transport: AAM flux form call
hydro_forward.cpp:1316-1363, 1390-1391                 50  R        Coriolis gate, x1 mass-w buffers, F_d
hydro_forward.cpp:1624-1680                            57  R        step 8: x1 mass-flux w, metric source
hydro.cpp:27-37, 55-76                                 33  R        mode predicates and checks
gnomonic_equiangle.hpp:84-108, 117-122                 31  S        decls: x1 sources, flux form, rate
gnomonic_equiangle.cpp:526-604                         79  S        ``axial_x1_source_``, flux form
gnomonic_equiangle.cpp:664-719                         56  S        ``x1_mass_source_``, rate
gnomonic_equiangle.hpp:22-35, 109-116                  22  S        AAM buffers, rate_to_rows decl
gnomonic_equiangle.cpp:18-36, 169-221, 485-525        113  S        rotation field k, face averages,
                                                                    rate_to_rows
coordinate.hpp:370-392                                 23  S        four virtual AAM / x1-source hooks
hydro.hpp:89-133, 219-236, 462-466                     68  R        AAM options (docs), accessors, state
meshblock.cpp:983-990, 1009-1034                       34  X        closure reduction and increment
mesh.cpp:364, 367-382                                  17  X        closure over local blocks
================================================  =======  =======  =======================================

What each owner counts:

* **R** counts the hydro ranges: 136 lines in ``hydro_forward.cpp`` (of which
  ``:988``, the "(3.A) Calculate dimension 2 LR states" heading, stays) and 33
  in ``hydro.cpp``. R's "after" holds the lines lane A leaves there, an
  estimated 25 + 15 (two phase hooks, the mode predicate and the ``full``
  refusal).
* **S** counts all 301 ``gnomonic_equiangle.*`` AAM lines and
  ``coordinate.hpp``'s 23. Of the 301:

  * 135 are the rotation data;
  * 153 are AAM physics, which moves into ``src/forcing`` as new lane-A lines
    (9.2);
  * 13 are ``x1_mass_source_`` (``.cpp:664-674``, ``.hpp:84-85``), the metric
    source every cubed-sphere run uses. It stays in coord with S (J1).
* **X** counts the mesh lines.

This replaces rev 2's "A counts 335 outside ``src/forcing``".

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
       (R:14-18, 56-61)
A-r2   ``reg.kept(d, "riemann.central")``, d = 2,       divergence,  exact: F_d = total − central; closure
       3: one summed mass row, never scaled             pre_impl.    residual R
       (R.rst:455-457), read in the same stage
A-r3   ``reg.total(1)`` mass rows (dry + species) in    post_impl.   m1 (#355): explicit plus the
       ``post_implicit``, which holds the                            ``vic.mass_dry``/``vic.mass_species``
       ``vic.*`` posts (R:20; R-4 answered)                          posts
A-r5   ``reg.kept(1, "vic.momentum_t")``, per face      post_impl.   ``x1_source``; drops the prefix sum
                                                                     (``gnomonic_equiangle.cpp:541-543``)
=====  ===============================================  ===========  ======================================

**F_d is the limited mass flux minus the pre-limiter central flux** (appendix
B, item 1). The total is scaled by the positivity limiter (R:60-61) and
gains its carry (R:56-58). The kept central part is never scaled (R.rst:455-457). So
``F_d = total(d) mass rows − kept central`` contains the covariance, the
withheld species mass and everything else that is not central. This is what
today's ``_flux2 − cen2`` holds (``hydro_forward.cpp:1359``, after the limiter
at ``:1165-1190``).

**The flux form reads the momentum rows after the limiter's carry.** Lane A's
K-rewrite runs on ``du`` in the divergence phase, which comes after
positivity (R:17-18). This is correct only with #370's seam-consistent
carry; hence the gate reference in 5.3.

Lane A therefore needs from lane R exactly the R:14-22 order:

* ``riemann`` with ``keep`` before covariance and positivity;
* positivity's scale and carry before divergence;
* ``vic.*`` posts before ``post_implicit``.

Face states (placeholder): the header builds ``riemann.central`` "from the
solver's own face states" (R.rst:357-359; K4 at R.rst:809-810). Rev 1 asked for the post-terrain states.
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

``AxialClosureFixer : Fixer`` (R:104-116). The fixer registry and
collective are lane X's (R:104).

* ``name()`` returns ``"aam.closure"``, and ``rows()`` returns {IVY, IVZ}.
* ``partial_sums()`` returns {R, I} as FP64 device tensors, with
  R = ``Σ V (K · F_cell − rate(kept central))``. It is computed once.
* ``apply(u, global_sums, dt, stage_weight, reg)`` (R:113-115):

  * ``dt`` is the host double (N14);
  * it adds ``−w2 dt R / I · ρ s²`` along K, ``where``-masked;
  * it posts the momentum-row change with
    ``reg.cell_amount("aam.closure", V du, 1.0, AmountKind::fixer)``
    (R:67-70, 111-112);
  * it posts the AAM change with
    ``reg.derived_amount("aam.closure", "aam", −w2 dt R, 1.0, AmountKind::fixer)``
    (R:72-74; R.rst:698-701).
* **w2 (R-6, closed by the frozen header).** ``stage_weight`` is "w2 of the
  stage just combined" (R:111). That matches the pin, where the increment uses
  that stage's own weight, ``pintg->stages[stage].wght2()``
  (``meshblock.cpp:1027``). X's registry reads it from ``weights()`` after
  ``begin_stage(int, harp::IntegratorWeight const&)`` (R:49-50). There is no
  ``StageWeights`` type any more.
* **The collective carries {R, I} only: two FP64 numbers per stage.**

  * Rev 2's "7-vector" added the gravity-work fixer's five sums
    ``{D, Σ m V, wall mass, mwall, redo}`` (``meshblock.cpp:1058-1060``).
  * Cheng's ruling (b) removes that global fixer (lane I thread), and lane X
    deletes its sums (``meshblock.cpp:1036-1062``; X.rst:54). So nothing else
    shares the closure's collective.
  * The redo flag rides X's stacked retry read instead (7.1).
  * Lane X reduces {R, I} through the host and never captures it (X.rst:98,
    187; N14). The fixer point is after the saturation adjustment and before
    the boundaries.

3.4 One Coriolis and metric source (Q4)
---------------------------------------

``force = −2 Ω × m``, with ``m = (m1, ρ v2, ρ v3)``. The metric pair
``−m1 v_h / r`` stays in coord's ``x1_mass_source_``
(``gnomonic_equiangle.cpp:664-674``, declared at ``.hpp:84-85``; 13 lines),
which every cubed-sphere run uses. Lane S keeps it, and lane A's source calls
it with lane A's m1 (J1; this revision drops rev 3 part 1's "+8 metric lines in
``coriolis.cpp``"). The source is registered in two phases:

* ``RegisterPhase::pre_implicit`` (m1 = 0; through the scheme-9 solve);
* ``RegisterPhase::post_implicit`` (the m1 rows, R:21).

``m1 = φ_lo·lo + φ_hi·hi``, where φ is the face area times the mass rows of
``reg.total(1)`` read in ``post_implicit`` (A-r3). This already contains the
``vic.mass_dry`` and ``vic.mass_species`` posts, so lane A does not add
``vic.mass_total`` (no double count). Tianhao confirmed that ``total(1)``
includes the ``vic`` posts (R-4, answered).

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
N16). Their species clip (``meshblock.cpp:879-884``) is a floor **owned by
lane X**, because ``meshblock.cpp`` is X's file and X's item w rewrites the
code around it (X.rst:350). X registers it through R's ``cell_amount``
(R:69-70, kind floor).

3.7 Diffusion as a register contributor (Q7), from phase 3
----------------------------------------------------------

**Phase 3:** ``DiffusionFlux`` posts ``diffusion`` with ``reg.add`` (R:57-58):
the viscous momentum flux and the energy flux ``−k ∇T · n − v_f · τ · n``.
Its part is an FP64 monitor amount only (R.rst:452-454).

* The cubed-sphere traction is rotated into the panel face basis by coord's rotation
  (S-2).
* Diffusion keeps its own exchange (``share_fluxes``, then the 7-channel
  penalty traces) until phase 3 (S-3 accepted). From phase 3, the flux
  averaging is R's ``replace`` with S's transport (R:63-64), and the penalty
  traces keep diffusion's own exchange.
* **Radial conduction:** each half step posts its x1 heat flux with
  ``reg.post_split(1, "diffusion.radial", flux, 0.5, dt)`` (R:80-81). Its
  failure is a device flag (N3).
* **Radial failure contract (rev 4, Tianhao's request).** At the pin the
  check is a host ``TORCH_CHECK``: finite and positive ``next``
  (``diffusion_radial.cpp:97-100``). The increment
  ``u[IPR] += capacity · (next − temp)`` (``:101``) is applied only when the
  check passes. In V3 the check becomes device-side and the increment is
  masked:

  * ``ok_col = (isfinite(next) & next > 0).all(-1)`` per column ``(k, j)``.
    The SDIRK system couples a whole column (``:71-96``), so one bad cell
    makes the column's solution untrusted. Masking the whole column, not single
    cells, is lane A's choice.
  * ``next = where(ok_col, next, temp)``. So the increment at ``:101`` is zero
    on failed columns, and the energy row and ``u`` stay finite.
  * The ``post_split`` wall flux is formed from the masked ``next``, so the
    register and the state agree: a masked column posts zero.
  * ``failed_ |= !ok_col.all()`` is a 0-d device flag, posted with
    ``reg.flag_retry("diffusion.radial", failed_)`` (R:78). There is no host
    read inside the segment.
  * **Default action: abort** (unchanged from the pin's behaviour). X's
    step-end read sees the flag and ends the run. The masking keeps the state
    finite until that read.
  * **Proposal: retry** (FOR CHENG, 10.3). The flag takes X's slot 8, cause
    bit 6 (value 64). It is zeroed at step start and counts against
    ``max_redo`` like bits 0-4 (X.rst:146-147, 155-157 at ``0fe47c6f``). The
    redo restores ``split_initial_`` (``diffusion_radial.cpp:14-17``), so the
    masked state is discarded.

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
components (R:4-5). The area weighting is the register's divergence. Registered
fluxes stay in the panel face basis, and S rotates at seams (S-2).
``A F · K_f`` is derived, never posted.

**Requirement 2.**

* Lane A posts ``diffusion`` (phase 3) and ``post_split`` entries, and keeps
  ``riemann.central`` and ``vic.momentum_t`` (R:35).
* The order is R:14-22 (3.1).
* Lane A's parts are FP64 monitor amounts (R.rst:452-454).

**Requirement 3.**

* R's ``end_stage`` weights the stage total with
  ``c_s = w2_s Π_{t>s} w1_t`` (R:85; R.rst:371-376).
* The closure applies ``stage_weight`` inside ``Fixer::apply``.
* The radial conduction posts through ``post_split`` with weight 0.5 per half
  step (R:80-81).

**Requirement 4.**

* The flux form and F_d read ``total`` and ``kept``.
* m1 reads ``total(1)`` in ``post_implicit``.
* ``x1_source`` reads ``kept(1, "vic.momentum_t")``.
* The two extra Riemann calls (``hydro_forward.cpp:997-1003, 1108, 1141``)
  and the ``_flux1`` times area rebuild (``:1637-1646``) go.
* Diffusion keeps its own divergence until phase 3.

**Requirement 5.**

* AAM reads the reconciled total.
* ``K_f`` comes from coord's rotation data in the panel face basis (S-2).
* **Gate G5 has a precondition:** R's ``replace`` (R:63-64) and S's seam
  rotation and transport must have landed. Until then G5 is not run.
* Diffusion keeps its own exchange until phase 3 (S-3).

**Requirement 10.** Nothing lane A adds is restart-written:

* lane A adds no ``Variables`` entry (``restart.cpp:31-37``);
* R rebuilds the register (R:48, 86; R.rst:464-467);
* lane A's buffers (``omega_local``, the m1 weights, ``scratch_``)
  are rebuilt at reset.

4.2 Compliance table
--------------------

====  ========  ==================================================================
req   section   status for lane A
====  ========  ==================================================================
1     4.1       complies; panel face basis (S-2); diffusion owner from phase 3
2     4.1       complies with R:14-22, 32-39; vic.momentum_t@1 kept (R-4)
3     4.1       complies: R:85, post_split (R:80-81); stage_weight = w2 of
                the stage (R:111, R-6 closed)
4     4.1       complies from phase 2; diffusion from phase 3
5     4.1       complies from phase 3; G5 after R's replace and S's seam rotation
6     5.1, 7.1  complies after the 7.1 host reads go; AAM is the derived row
                (R:34, 72-75); radial flag via ``flag_retry`` (R:78)
7     5.2       complies: ``AxialClosureFixer : Fixer`` posts ``cell_amount``;
                species-clip floor owned and registered by X
8     6, 7.1    complies: parts kFlux, amounts FP64 (FP32 on MPS)
                (R.rst:452-454; R-5 closed)
9     7.2       memory stated from shapes; cost is a measurement plan
10    4.1       complies: nothing restart-written (R.rst:467)
====  ========  ==================================================================


5. Invariants and gates
=======================

5.1 Monitor (requirement 6)
---------------------------

The monitor is lane R's (R:87, 89-103).

* On monitor steps each lane-A source writes into the registry's one FP64
  ``scratch_``, zeroed per source.
* The registry posts ``reg.cell_amount(name, V·scratch, w2_s,
  AmountKind::cell_source)`` (R:67-70). ``w2_s`` is used because the scratch
  is this stage's ``du`` (R:68). It then adds the scratch once to ``du`` (FP64
  run) or ``_forcing_work`` (FP32 run).
* Bitwise: every lane-A source writes each element at most once and none reads
  ``du``. The exception outside lane A is ``const_gravity.cpp:52, 58, 63``.
* **AAM is a derived row (R:34, 72-75; R.rst:538, 698-703).** Lane A
  requests ``derived = {"aam"}``. AAM goes in through two calls:

  * each source's share: ``derived_amount(name, "aam", Σ V r K · scratch,
    w2_s, AmountKind::cell_source)``;
  * the closure's: ``derived_amount("aam.closure", "aam", ..., 1.0,
    AmountKind::fixer)`` (3.3).

  At step end, ``derived_total("aam", L)`` posts ``L = Σ V r K · m`` over
  fluid cells, after the last ``after_combine``. ``cell_amount`` carries only
  the conserved rows. Lane A keeps no AAM buffer of its own.
* **x1-wall AAM, "<row>.boundary.<dim>" convention (R.rst:542-546, 701-703).**
  ``step_field(1)`` (R:84) is the stage-weighted step face transfer per unit
  area, and already carries dt and the stage weights (R.rst:557-558). Lane A
  owns its K-weighted reduction over the physical x1 wall faces,
  ``Σ_wall A_f r_f K · F_step``, inflow positive.

  * It is posted once at step end as
    ``derived_amount("aam.boundary.1", "aam", value, 1.0,
    AmountKind::cell_source)``. Weight 1.0 is used because the value is
    already weighted.
  * The name ``aam.boundary.1`` replaces rev 3 part 1's ``aam.x1_wall``.
  * No ``aam.boundary.2`` or ``.3`` is posted: the cubed sphere has no
    physical x2/x3 boundary, and grids without an axis have no AAM row.
  * It requests ``step_field = {1}`` (R:38).
* Budgets on and budgets off are two graph topologies.

5.2 Fixers and floors (requirement 7)
-------------------------------------

=============================  =======  =============  ==========  ========================================
fixer / floor                  kind     rows           registers   declares
=============================  =======  =============  ==========  ========================================
AAM closure (``exact``)        fixer    IVY, IVZ       A (Fixer)   ``cell_amount(…, 1.0, fixer)``;
                                                                   ``derived_amount`` AAM ``−w2 dt R``
                                                                   (G6)
user-forcing species clip      floor    ICY.., IDN     X (owner)   species mass withheld per stage, via
(``meshblock.cpp:879-884``)                                        ``cell_amount(…, floor)`` (R:69-70)
side-study torque (rule 2)     numer.   IVY, IVZ       A           its AAM rate; no reduction
=============================  =======  =============  ==========  ========================================

**Lane EF's floors, repairs and positivity posts, and their AAM effect (Z4).**
The table cites EF rev 3.1 at ``c21dd66e`` and the pin. "AAM effect" is lane
A's reading at the pin.

====  ====================================  ===========  ===============  ==============================================
id    post (EF.rst:556-597)                 phase        kind             AAM effect (lane A)
====  ====================================  ===========  ===============  ==============================================
P2    ``scale(d, "positivity.withhold",     limit        flux_correction  mass rows only, so no momentum change. It is
      species, factor)``                                                  inside ``total(d)``, so it enters F_d (the
      (``flux_positivity.cpp:99-121``)                                    ``exact`` torque) and m1 on x1. No AAM leak,
                                                                          and no AAM amount.
P3    ``add(d, "positivity.carry",          limit        flux_correction  changes the momentum flux rows. The flux form
      {IPR}/{IVX..IVZ}, ...)``                                            reads ``total(d)`` after P3 (Q-A1), so the
      (``flux_positivity.cpp:124-167``)                                   carry telescopes, and AAM changes only through
                                                                          boundary faces. Needs #370's seam-consistent
                                                                          carry (G7s). No AAM amount.
F1    ``cell_amount("eos.floor", ...,       after\_      floor            density floor (``equation_of_state.cpp:244-
      1.0, floor)``                         combine                       250``) and energy floor (``:268-287``) leave
                                                                          the momentum rows alone, so AAM does not
                                                                          change. NaN zeroing (``:236-238``) does touch
                                                                          momentum, but it raises ``ef.nan`` and the
                                                                          step is redone. No AAM amount.
F1    ``eos.borrow`` (``:307-324``),        after\_      floor            species rows only (EF.rst:192-202: mass
      ``eos.fix_vapor`` (``:326-380``)      combine                       conserving), so AAM does not change. No AAM
                                                                          amount.
====  ====================================  ===========  ===============  ==============================================

So no EF floor, repair or positivity post needs a ``derived_amount`` on
``aam``. The monitor's AAM budget closes from lane A's own amounts, the
``aam.boundary.1`` transfer and the closure. That the floors leave momentum
unchanged is read from the pin code, not run (hypothesis H11).

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
H11  EF's floors and repairs leave the momentum rows unchanged    read from the pin (5.2); gate G7s
     (no AAM amount)                                              and G3 with floors active
H12  the kept ``vic.momentum_t`` top face is round-off when the   gate G3 with ``implicit-scheme: 9``
     solve conserves column tangential momentum (11.2 c)
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
kept ``riemann.central``,                   FP64              FP32 storage on CUDA, FP64 accumulation
``vic.momentum_t``                                            (R-5); MPS pending with R; lane A forms
                                                              ``F − F_c`` and the rate in FP64
EF's P2 factor / P3 carry on ``total(d)``   FP64              flux dtype (FP32) in ``total(d)`` (kFlux,
                                                              R.rst:452-453); their amounts FP64
                                                              (R.rst:452-454); lane A forms the flux form
                                                              and F_d from them in FP64 (Z6)
R's step field and amounts                  FP64              FP64; FP32 on MPS (R.rst:452-454)
AAM flux form, torque rate, R and I         FP64              FP64; summed in FP64 before reduction
``diffusion`` part                          FP64 amounts      FP64 amounts (R.rst:452-454)
rotation data                               FP64              lane S's master and cast views
==========================================  ================  =========================================

6.2 What lifts the restriction on exact (Q5)
--------------------------------------------

Three things lift today's refusal (``hydro_forward.cpp:993-995``) together:

1. a consistent central part, which the header gives as totals under
   perturbation flux (R.rst:958-961);
2. FP64 accumulation of the rate and of R and I;
3. gate G10.

Until G10 passes, ``exact`` refuses FP32 and perturbation-flux at reset.


7. CUDA graph
=============

7.1 Capturable storage and host reads (requirements 6 and 8)
------------------------------------------------------------

* The lane-A buffers have fixed shapes and are allocated at reset:
  ``omega_local``, the m1 weights, ``scratch_`` and ``{R, I}``.
* The kept parts live in R's storage (R:43-44).
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
``diffusion_radial.cpp:97-100`` aborting check          ``reg.flag_retry("diffusion.radial", any)``
                                                        (R:78), read in X's one step-end read
                                                        (R:102). Until R's PR R-1 moves the retry
                                                        flags, it rides EF's ``EfStatus`` tensor in
                                                        X's stacked read (EF.rst:319-342, 606-607;
                                                        ``meshblock.cpp:1559-1566``) (Z5). Action
                                                        stays abort (FOR CHENG: redo)
``specific_heat_cv`` -> ``species_cv_ref`` host read     EF caches the reference cv on the host (Q-A2):
per call (``moist_base.cpp:28``), from                  ``body_heat.cpp:49``,
``body_heat.cpp:49``, ``relax_bot_temp.cpp:69``,         ``relax_bot_temp.cpp:69``,
``diffusion.cpp:566``                                   ``diffusion.cpp:566``,
                                                        ``diffusion_cubed_sphere.cpp:200`` and
                                                        ``diffusion_radial.cpp:36`` lose the read
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
``vic.momentum_t`` kept                       none (prefix sum in place)         2 x1-face rows, per face (R-4)
per-source scratch                            none                               one FP64 field, 8·(5+ny) B per
                                                                                 cell, FP64 runs only
per-source amounts                            none                               R's monitor (rows and AAM)
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
  closure's {R, I} (two FP64 numbers) through the host and never captures it.
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
3. lane R register (``4c59162a`` frozen contract), lane I ``vic.*``, lane X global
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
* **RULING V3-1:** a deletion is counted by the lane that owns the file.
  Lane A's number is ``src/forcing`` only, new files included. There is one
  lane number.
* **A** drops its 51 mesh lines (X's) and adds ``AxialClosureFixer : Fixer``
  in ``axial_momentum.cpp``.
* **A** counts the 7 lane-I lines in ``forcing.hpp`` (``:48, 56-58, 60-62``).
  The 11 python lines are listed in 10.2 and are not in the number.
* **R** counts the hydro AAM ranges (136 + 33; ``hydro_forward.cpp:988`` is a
  heading that stays). R's "after" holds lane A's left-behind 25 + 15
  (2.2).
* **S** counts the 301 gnomonic AAM lines (166 physics + 135 rotation data)
  and ``coordinate.hpp``'s 23 (2.2).
* **O**'s schema is a precondition (N18).

9.2 Counted lines and the lane number
-------------------------------------

The "now" column is ``wc -l`` at the pin (2.1). The "after" column holds
estimates. Under S-1, ``omega_local`` and the AAM physics move into lane A's
``src/forcing`` files, as new lines.

=================================  ======  ======  =======================================================
file                               now     after   basis
=================================  ======  ======  =======================================================
forcing.hpp                         674     253    options 326 -> 150 via schema (O); classes 270 -> 70;
                                                   includes and comments 78 -> 40; −7 lane-I lines
source_term.hpp (new)                 0     110    base and registry
coriolis.cpp                        232     170    −28 (123), −38 (glue), +4 (two phases); Ω set-up
                                                   stays (``omega_local``, S-1); the metric source
                                                   stays in coord (J1)
axial_momentum.cpp (new)              0     225    consumer, torque rules, ``AxialClosureFixer`` (170) +
                                                   flux form (45) + x1 source from ``vic.momentum_t`` (10)
diffusion.cpp                       745     480    −70 options, −110 profile checks (schema), −40, −45
diffusion_cubed_sphere.cpp          376     320    phase 3: −35 averaging, −15 divergence, −6
diffusion_radial.cpp                103     104    −4 host check (``:97-100``); +3 masked increment and
                                                   flag (3.7); +2 ``post_split`` wall flux
top_cool + bot_heat                 176      70    ``LayerHeat{side}``
top_sponge + bot_sponge             156      50    ``Sponge{side}``
relax_bot_temp/velo/comp            308     235    ``BottomSlab`` base
body_heat + plume_forcing           100      70    base does ctor, reset, options
**total**                          2870    2087
=================================  ======  ======  =======================================================

**Lane A: 2,870 → 2,087 lines (−783, −27.3 %), ``src/forcing`` only, new
files included (RULING V3-1).**

Arithmetic:

* now: 674 + 232 + 745 + 376 + 103 + 176 + 156 + 308 + 100 = 2870
  (2.1, ``wc -l``);
* after: 253 + 110 + 170 + 225 + 480 + 320 + 104 + 70 + 50 + 235 + 70 = 2087;
* cut: 783 / 2870 = 0.273.

J1 drops rev 3 part 1's +8 ``coriolis.cpp`` lines: the metric source stays in
coord (3.4). That moves the number from 2,094 to 2,086. Rev 4 books the
``diffusion_radial.cpp`` change once, here (RULING V3-1; 3.7, 11.4): 103 ->
104, an estimate, so 2,086 -> 2,087.

Cross-references, not in lane A's number:

* R: hydro AAM ranges 136 + 33, with 25 + 15 remaining in R's "after";
* S: gnomonic 301 and ``coordinate.hpp`` 23;
* X: mesh 51.

Rev 2's lane total (3,205 → 2,134) is withdrawn.

**Precondition (N18): lane O's options schema lands before lane A's phase 1.**
Without it, ``src/forcing`` would be 2,398 (−16.4 %) after phase 3 and 2,499
(−12.9 %) before it (+131 options, +70 parser, +110 profile checks; +101 for
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
* **A:** 152. Of these, 124 are in python and tests. The 28 ``src`` lines are
  counted by file owner (RULING V3-1):

  * R: ``hydro.cpp:36, 59-69`` and ``hydro_forward.cpp:989-995, 1355``, 20
    lines;
  * S: ``gnomonic_equiangle.hpp:117``, 1 line;
  * A: ``coriolis.cpp:176-179, 184, 189, 209``, 7 lines.

The 11 lane-I python lines (``pyforcing.cpp:29``, ``forcing.pyi:61-70``) are
listed here and are not in lane A's ``src/forcing`` number.

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
radial-conduction positivity failure        device flag; abort kept (default)    **FOR CHENG** (redo uses
                                            or redo                              X's slot 8, cause bit 6)
==========================================  ===================================  ==========================

10.4 Decision log
-----------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c is the kept ``riemann.central``                   no second flux evaluation (3.1; R:35)
D2  modes ``off | transport | exact``; full refused       settled by Cheng (3.2, 10.3)
D3  default ``off``; AAM budget always reported           bitwise migration (3.2)
D4  ``AxialClosureFixer : Fixer``; X's registry           requirement 7; R:104-116 (3.3)
D5  one Coriolis; m1 from ``total(1)`` in post_implicit   requirement 4; R:21 (3.4)
D6  FP32 lifted only by G10                               6.2
D7  per-source scratch + ``cell_amount``                  bitwise; R:67-75 (5.1)
D8  diffusion into the register in phase 3                bitwise gates of phases 1-2 (3.7)
D9  ``exact`` with ``nu_iso > 0`` refused until phase 3   closure would remove a physical torque
D10 rotation data in S's coord; physics in lane A         S-1 (3.8)
D11 user forcings stay in X's segment                     order and G1 (3.6; N16)
D12 ``capturable`` flag, ordered runs                     X's segment flag (3.6; N14)
D13 lane A writes nothing to restart                      requirement 10 (4.1)
D14 lane O schema before lane A phase 1                   20 % target (9.2; N18)
D15 gates against main with #370                          carry consistency (5.3; appendix B)
D16 one lane number, ``src/forcing`` only                 RULING V3-1 (9.2)
D17 lane A owns the K-weighted x1-wall AAM reduction      R-4 answer (5.1)
D18 AAM on derived row ``aam``; wall transfer             R:34, 72-75; R.rst:542-546 (5.1, 11.2 b)
    ``aam.boundary.1`` at step end, weight 1.0
D19 metric source stays in coord; −8 in coriolis.cpp      J1 (3.4, 9.2)
=== ====================================================  ==========================================


11. Open questions to other lanes
=================================

11.1 Answered and closed
------------------------

* **R-1** is closed by the frozen header and R.rst (re-cited from
  ``543ff84f``):

  * ``riemann.central`` is a kept part written in place by C's kernel
    (R.rst:357-359), one summed mass row on x2/x3 (R.rst:165-168), in totals
    (R.rst:958-961);
  * it is never scaled (R.rst:455-457);
  * parts are kFlux, and amounts are FP64 (R.rst:452-454).

  The face states are answered in 11.2 (a).
* **R-2** is closed: ``post_split`` (R:80-81), after ``begin_step()``
  (R:6, 48).
* **R-3** is closed and superseded by the derived row: AAM goes on ``aam``
  through ``derived_amount``/``derived_total`` (R:34, 72-75; R.rst:538,
  698-703), and the conserved rows through ``cell_amount`` (5.1).
* **R-4** is closed: ``vic.momentum_t@1`` is kept per face (R.rst:165-167),
  and ``total(1)`` includes the ``vic`` posts (R.rst:521-523). Lane A owns the
  ``step_field(1)`` reduction at x1 walls (R:84; 5.1).
* **R-5** is closed. Kept parts are kFlux: FP32 storage under FP32 on CUDA,
  with FP64 accumulation. Amounts and step sums are FP64, or FP32 on MPS
  (R.rst:452-454). Whether the closure's {R, I} survive FP32 on MPS stays
  hypothesis H9 (no MPS gate).
* **R-6** is closed: ``stage_weight`` is the w2 of the stage just combined
  (R:111-112; R.rst:160-162). This matches pin ``meshblock.cpp:1027`` (3.3).
* **The fixer base and registry:** ``class Fixer`` (R:104-116); registry and
  collective are lane X's (R:104).
* **The radial reset order (my X-1 note in part 1)** is closed by R. X calls
  ``begin_step()`` at step start, before the radial ``post_split``, and
  ``begin_stage(0)`` zeroes nothing (R:6, 48; R.rst:31-34, 70-73). So the
  half step at ``meshblock.cpp:715`` is no longer erased by the stage-0 resets
  at ``:722-738``.
* **S-1..S-3** (lane S thread) and **I-1** (lane I thread).
* **K-1** was routed by the PM: the kintera half to Xi (lane K), and
  ``ConstGravity`` to Xinyue (lane I).

11.2 Lane R's questions to A (R.rst:1598-1601), answered
--------------------------------------------------------

* **(a) K4: post-solid reading. Confirmed.** At the pin, the central part and
  the solver take one tensor:

  * ``wlr2 = has_solid ? pmb->pib->forward(wtmp2, DIM2, solid) : wtmp2``
    (``hydro_forward.cpp:1089-1090``; x3 at ``:1123-1124``);
  * it is formed after the cubed-sphere L/R sync (``:1013-1048``);
  * both ``central(wlr2[ILT], wlr2[IRT], ...)`` (``:1108``) and
    ``solve(wlr2, ...)`` (``:1109``) read it.

  ``central`` passes clones into ``priemann->forward`` (``:997-1003``), so it
  sees the post-solid, unrotated states, and ``u_n`` is formed by the solver's
  own projection (the comment at ``:1092-1093``). The mass rows are frame-free
  (R.rst:450-451).
* **(b) The "<row>.boundary.<dim>" convention for aam. Confirmed.** Lane A
  posts the x1-wall AAM transfer as ``derived_amount("aam.boundary.1", "aam",
  value, 1.0, AmountKind::cell_source)``, inflow positive (R.rst:542-546).
  ``value`` is the K-weighted reduction ``Σ_wall A_f r_f K · step_field(1)``
  over physical x1 wall faces. It is posted once at step end with weight 1.0,
  because ``step_field`` already carries dt and the stage weights
  (R.rst:557-558). The name replaces ``aam.x1_wall``. There is no
  ``aam.boundary.2`` or ``.3``: the cubed sphere has no physical x2/x3
  boundary.
* **(c) Does ``vic.momentum_t.top`` enter the axial torque? No.** Lane A's x1
  source reads the kept per-face field ``kept(1, "vic.momentum_t")``. Its top
  face enters only as ``P_hi`` of the top cell, exactly as the pin's prefix
  sum does (``gnomonic_equiangle.cpp:541-546``: ``T``, ``Phi =
  -T.cumsum(-1)``, ``Plo``, ``target``). Lane A never reads the
  ``diagnostic`` amount ``vic.momentum_t.top`` (R:30). If the solve conserves
  column tangential momentum, the top value is round-off (hypothesis H12). If
  EF ever prescribes tangential momentum (R.rst:516-520), lane A's source
  changes and needs lane A's agreement.

11.3 Lane EF's questions to A (EF.rst:1857-1863), answered
----------------------------------------------------------

* **Q-A1: yes.** Lane A reads ``total(d)`` in phase ``divergence``, after
  ``limit`` (R:17-18), so the flux form includes P3's carry and P2's scale
  (3.1; Z4 table in 5.2).
* **Q-A2: no objection; one correction.** The forcings do not call
  ``species_cv_ref()`` directly. They call ``specific_heat_cv``
  (``body_heat.cpp:49``, ``relax_bot_temp.cpp:69``, ``diffusion.cpp:566``,
  plus ``diffusion_cubed_sphere.cpp:200`` and ``diffusion_radial.cpp:36``).
  That calls ``species_cv_ref`` (``ideal_gas.cpp:27``,
  ``ideal_moist.cpp:61``), which reads the host with ``.item()``
  (``moist_base.cpp:28``). A host cache removes that read from all five
  (7.1).

11.4 Open
---------

* **X-1, Xi.** Abort or redo for the radial failure is Cheng's call (FOR
  CHENG, 10.3). For redo, lane A takes X's contract (X rev 3.3 ``0fe47c6f``):

  * ``diffusion.radial`` is stack slot 8, mapped to cause bit 6 (value 64)
    (X.rst:146-147);
  * it is zeroed in ``kStepStart`` and counts against ``max_redo`` like bits
    0-4, terminating without a restore (X.rst:155-157);
  * lane A's side is the device flag and the masked increment (3.7), and the
    request ``retry = {"diffusion.radial"}`` (R:37).

  **Booking conflict, open for Xi.** X.rst:153-155 says that no lane claims
  ``src/forcing/diffusion_radial.cpp``, so X books its change outside the
  lane count. Under RULING V3-1 the file is lane A's (``src/forcing``), so
  lane A books it (9.2), and X should drop that sentence.

FOR CHENG: see 10.3.


12. Skeleton index
==================

``wc -l docs/v3/skeleton/A/*`` gives 299 lines, with declarations only, in
namespace ``snap``. The headers include lane R's ``flux_register.hpp``. A
syntax check against R's frozen header at ``4c59162a`` passes.

==========================  ==============================================================  ===============
file (lines)                declarations                                                    serves
==========================  ==============================================================  ===============
``source_term.hpp`` (90)    ``SourceKind``, ``SourceRows``, ``SourceRegion``,               3.6, 5.1, 7.1
                            ``SourceDeclaration`` (``RegisterPhase phase``, ``split``,
                            ``capturable``), ``StageState``, ``SourceTerm``,
                            ``SourceRegistry`` (``Run``, ``runs``, ``run``,
                            ``scratch_``)
``axial_momentum.hpp``      ``AxialMode``, ``AxialFluxView`` (``total``, ``central``,       3.1-3.3, 5.1,
(88)                        ``momentum_t``), ``FaceTorqueRule``, ``AxialMomentum``          5.2, 6.2, 7.3
                            (``request``, ``wall_torque``, ``aam_total``),
                            ``AxialClosureFixer : Fixer``
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
* **N3 (accept).** The radial check becomes a device flag that joins X's
  stacked redo flags (``meshblock.cpp:1559-1560``). The action stays abort;
  redo is FOR CHENG plus lane X (X-1, including the stage-0 reset order).
* **N4 (accept).** G2 uses the normalised e_∞; the error scale is H3.
* **N5 (accept; superseded in part by N14 and by the fixer removal).** The
  placement stands. The 7-vector does not: with the gravity-work fixer removed
  (Cheng's ruling (b); ``meshblock.cpp:1036-1062`` deleted by X), the
  collective carries {R, I} only. It goes through the host and is never
  captured. The registry is X's (R:104).
* **N6 (accept, changed by R's header).** R owns the monitor (R:87, 89-103).
  Per-source amounts go to ``reg.cell_amount`` (R:67-70) instead of a lane-A
  ``budget_``. AAM is the derived row ``aam`` (R:34, 72-75; R-3 closed).
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


Appendix B: Zoey's round-2 review (Z1-Z8)
==========================================

Rev 2 answered three points from Zoey's report by topic. This revision
answers by number the notes the PM carried into this thread (Z4, Z5, Z6). Her
formal text for Z1-Z3, Z7 and Z8 was not in my inputs, so those keep their
rev 2 topical answers, and their status is "to be confirmed against her text".

* **Z1-Z3 (status: answered by topic in rev 2; numbering to be confirmed).**
  The three rev 2 points stand:

  * the flux form reads ``total(d)`` after the limiter's carry, and F_d is the
    limited total minus the never-scaled kept central (3.1; R:17-18;
    R.rst:455-457);
  * gates compare against main with #370 (5.3, D15; new G7s);
  * the EF-3 rule for G4 (5.3).
* **Z4 (status: done).** §5.2 now lists EF's floors and repairs (F1:
  ``eos.floor``, ``eos.borrow``, ``eos.fix_vapor``) and the positivity posts
  P2 and P3, each with its AAM effect (EF.rst:556-597). §11.3 answers Q-A1
  (yes) and Q-A2 (no objection, with the ``specific_heat_cv`` correction).
* **Z5 (status: done).** The radial failure flag is
  ``reg.flag_retry("diffusion.radial", any)`` (R:78). Until R's PR R-1 moves
  the retry flags, it rides EF's ``EfStatus`` tensor in X's stacked read
  (EF.rst:319-342, 606-607; ``meshblock.cpp:1559-1566``) (7.1).
* **Z6 (status: done).** §6.1 gains the P2/P3 dtype row: flux dtype in
  ``total(d)``, FP64 amounts, and the flux form and F_d formed in FP64.
* **Z7, Z8 (status: text not available).** These will be answered by number
  when the formal notes are posted.
