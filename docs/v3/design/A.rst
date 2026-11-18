=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 4.1 (design and contract only, CHANGE 5): C9; part 1b (m1 cite,
           R-7/R-8, G5 at the end of ``limit``); part 2 (S-4..S-8 from lane
           S's fallback text, ``AxialRotationData``, hooks 16/4/3, C-1
           closed). src/forcing 2,870 -> 2,088 (1.6).
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
           declarations only, 281 lines (``wc -l docs/v3/skeleton/A/*``);
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
                                                       none in V3 (``diffusion.radial`` reserved,
                                                       4.3); ``step_field`` (38): {1}
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
``flag_retry(cause, any)``                 78          not called by lane A in V3 (radial abort kept,
                                                       4.3)
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
  * the closure is a registered fixer inside ``exact``;
  * **the radial-conduction failure keeps the abort for V3** (Cheng's
    decision, lane A thread 1794951687). A failed radial solve stops the run,
    as at the pin, and no behaviour changes. ``diffusion.radial`` stays a
    reserved retry cause (4.3).
* **Settled with lane S**, as S rev 1.1 records it (``d03e4888``,
  S.rst:65-74, 1002-1012):

  * coord owns ``PanelRotation`` and ``AxialRotationData``. Lane A reads them
    and owns the Coriolis rate, the closure, ``omega_local`` and the m1
    weights (S-1);
  * posts are in coord's ``flux2global<d>_`` frame (S-2; R:4-5);
  * diffusion keeps its own exchange until phase 3, then moves onto S's
    ``Exchange::seam_slab`` (S-3).

  Anything beyond that text is worded **lane A proposal, pending lane S**
  (3.8, 11.4).
* **FOR CHENG** (open proposals with their "what changes" rows; nothing waits
  on them; section 10.3):

  * Coriolis ``type: 123``;
  * the AAM key on grids with no axis;
  * refusing the old ``full`` with a message naming ``exact``;
  * a non-z rotation axis;
  * whether ``exact`` refuses the upwind and plume-roe Riemann solvers at
    ``reset()`` (C9; 3.2).
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
* **Part 2a (Cheng's decision):** the abort is kept and settled (1.5, 3.7,
  7.1, 10.3, 11.4). The masking and flag of part 1 are withdrawn from V3 and
  kept only as a note on the reserved future path (3.7). ``diffusion.radial``
  is a reserved cause in 4.3, and the retry path is out of the PR plan
  (10.1). ``diffusion_radial.cpp`` is recounted 103 -> 105 (9.2), so
  ``src/forcing`` goes 2,087 -> 2,088. "Refusing ``full``" moves back to the
  open proposals, as the PM lists it with Cheng.
* **Final part:**

  * Jiheng's H1-H17 are answered by number in appendix C;
  * S-1..S-3 are restated as S rev 1.1 wrote them (1.5, 3.8);
  * new lane A proposals pending S: S-4 to S-8; R-7, R-8 (since answered,
    11.4);
  * m1 moves into ``X1Momentum`` (3.4);
  * viscous momentum stays in the Cartesian channel until G9b (3.7);
  * G5 moves to phase 2 (4.1; placed at the end of ``limit`` in rev 4.1,
    D22); new gates G3b and G9b, G10's metric, and
    hypotheses H13-H15 (5.3, 5.4);
  * §7.3 gains the Z6 sentence and the radial split exchange;
  * the 10.3 rows gain H6's three cases and H12's ``interp_order`` refusal;
  * ``hydro.hpp``'s 68 lines go in the R cross-references (H17);
  * appendix B is updated for Zoey's statuses.

  ``src/forcing`` stays 2,870 -> 2,088.
* **Rev 4.1, part 1 (C9):**

  * section 3.2 states lane A's proposal that ``exact`` refuses
    ``dynamics/riemann-solver/type`` upwind and plume-roe at ``reset()``,
    worded to match lane C rev 6.2 (``bcac27c9``, C.rst:765-770,
    1103-1105);
  * a 10.3 row records it, and whether to refuse is in FOR CHENG (1.5);
  * the skeleton adds the check to ``AxialMomentum::check``'s contract
    comment;
  * no count changes: the check is one ``TORCH_CHECK`` inside the
    ``axial_momentum.cpp`` estimate of 9.2.
* **Rev 4.1, part 1b:**

  * cite fix: ``hydro_forward.cpp:1672-1680`` is the ungated gnomonic branch,
    which does not read m1; ``:1650-1657`` stays as m1's metric-source site.
    ``X1Momentum``'s instance is held by Hydro (R) and the class is lane
    A's (3.4, D23);
  * R-7 and R-8 are marked answered by Tianhao (3.4, 7.3, 11.4);
  * J9/G5: choice (b). G5 sits at the end of ``limit`` and waits on lane S's
    G-S-K7, because the carry rewrites the momentum rows that ``A F · K_f``
    is built from (4.1, 5.3, 7.3; D22). The phase stays 2;
  * no count change (2,870 -> 2,088).
* **Rev 4.1, part 2 (written from the lane S rev 2 fallback text (local commit ``cf5a22a9``, not pushed; data_in ``laneS-rev2-commit3-L3-fallback.md``); to be re-checked against
  S.rst when it is pushed):**

  * S-5 and S-6 are answered and folded into 3.8. S-4 is answered in skeleton
    only. S-7 and S-8 are pending S text (11.4);
  * the view is renamed to S's ``AxialRotationData`` everywhere, and the
    skeleton drops its own view type (3.8, 12, D24);
  * the ``coordinate.hpp`` hooks are restated as 16 / 4 / 3 against the pin
    (2.2);
  * C-1 is closed; K-1's kintera half stays open for Xi (11.1, 11.4);
  * no count change (2,870 -> 2,088).


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
coordinate.hpp:370-392                                 23  S        AAM / x1-source hooks: 16 + 4 + 3
hydro.hpp:89-133, 219-236, 462-466                     68  R        AAM options (docs), accessors, state
meshblock.cpp:983-990, 1009-1034                       34  X        closure reduction and increment
mesh.cpp:364, 367-382                                  17  X        closure over local blocks
================================================  =======  =======  =======================================

What each owner counts:

* **R** counts the hydro ranges: 136 lines in ``hydro_forward.cpp`` (of which
  ``:988``, the "(3.A) Calculate dimension 2 LR states" heading, stays), 33
  in ``hydro.cpp``, and 68 in ``hydro.hpp`` (``:89-133, 219-236, 462-466``,
  the AAM options, accessors and state; H17). R's "after" holds the lines lane A leaves there, an
  estimated 25 + 15 (two phase hooks, the mode predicate and the ``full``
  refusal).
* **S** counts all 301 ``gnomonic_equiangle.*`` AAM lines and
  ``coordinate.hpp``'s 23 lines. Of those 23, at the pin:

  * 16 are the deletable AAM hooks, ``:375-380, 382-385, 387-392``
    (6 + 4 + 6);
  * 4 stay: ``:370-373``, the ``x1_mass_source_`` hook (J1);
  * 3 are blank: ``:374, 381, 386``.

  S books the 16 in its section 9 (S's fallback text). Of the 301:

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
carry; hence the gate reference in 5.3, and G5's position at the end of
``limit`` (D22).

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

* Lane A proposes that the old ``full`` is refused with a message naming
  ``exact``. This is still open with Cheng (10.3).
* The closure key is refused by ``check_keys`` (``hydro_options.cpp:56``).
* ``full`` without the closure survives only as a test hook on
  ``AxialMomentum`` (arm (l), ``tests/test_axial_momentum_flux_form.py:332,
  565``).

**C9, lane A's proposal: ``exact`` refuses the upwind and plume-roe solvers
at ``reset()``.** Whether to refuse is a question for Cheng (FOR CHENG, 1.5,
10.3).

* **The check.** In ``AxialMomentum::check`` (skeleton), with mode ``exact``,
  ``dynamics/riemann-solver/type`` must be one of hllc, lmars, roe or
  shallow-roe. The type comes from the deck through ``riemann_solver.cpp:58,
  67`` and ``hydro_options.cpp:32``, is dispatched at
  ``riemann_solver.cpp:96-110`` ("upwind" at ``:102``, "plume-roe" at
  ``:106``), and defaults to hllc (``riemann_solver.hpp:33``).
* **The reason.** ``exact`` needs ``riemann.central``, the summed hydro mass
  row (3.1). Lane C defines it for hllc, lmars, roe and shallow-roe; upwind
  and plume-roe have no hydro mass rows, so ``riemann.central`` is undefined
  for them. C.rst:765-770 (section 3.6) says: "Lane C proposes that an option
  that needs it refuse at ``reset()`` with a message naming the solver. That
  check is lane A's (A2; Jiachen's C9)". Section 4 says the same
  (C.rst:1103-1105: "a mode that needs ``central`` should refuse them at
  ``reset()``; that check is lane A's (A2)"). ``transport`` reads only
  ``total(d)``, so it is not refused.
* **The message**, naming the solver as C asks: "dynamics/conserve-axial-
  momentum: exact needs riemann.central, which riemann-solver type '<type>'
  does not provide; use hllc, lmars, roe or shallow-roe".
* **Where it sits.** It sits with today's reset refusals of the same kind:
  ``full`` needs ``interp_order: 4`` (``hydro.cpp:63-66``, kept for ``exact``,
  H12), the closure key needs ``full`` (``:67-69``), and perturbation-flux
  needs HLLC (``:207-210``).
* **What changes:** nothing for any deck. Neither solver works as a hydro
  solver at the pin (C.rst:476-488), and no deck or test runs either in hydro.
  A misconfigured ``exact`` run would stop at ``reset()`` with that message,
  instead of failing in the solver or using an undefined central part.

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

**m1 is not private to Coriolis (H3).**

* Lane A builds m1 once per stage in ``post_implicit``, in a standalone
  ``X1Momentum`` (skeleton), from ``A_f total(1)`` (R ``hydro_stage.hpp:61-62``).
* It does so whenever #355's gate holds (``hydro_forward.cpp:1319-1322``: a
  gnomonic grid, or spherical-polar with Coriolis; ``nc1 > 1``; not shallow
  water), with or without Coriolis.
* Coord's metric source ``x1_mass_source_`` (S) takes m1 from the mass
  transfers inside #355's gate (``:1650-1657``) and needs it whether or not
  Coriolis is on, so it reads ``X1Momentum`` too, as does ``Coriolis``.
* **The ungated gnomonic branch does not read m1** (Tianhao's correction). At
  the pin it is ``hydro_forward.cpp:1672-1680``, the
  ``else if (ptype == "gnomonic-equiangle")`` branch outside #355's gate
  (``nc1 == 1`` or shallow water). It calls
  ``x1_mass_source_(src, w, w[IDN] * w[IVX], dt)`` at ``:1676`` with the cell
  ``ρw``, not ``X1Momentum``, and stays so. The PM's message gave these lines
  as ``:1674-1681``/``:1677``; the pin read gives ``:1672-1680``/``:1676``.
* **Who owns ``X1Momentum``.** Lane A owns the class (``src/forcing``).
  Hydro (R) constructs and holds the instance at ``reset()``, because it must
  exist without Coriolis, and R's ``post_implicit`` already books "A's m1
  from ``A_f total(1)``" (R ``hydro_stage.hpp:61-62``). Hydro passes it by
  reference to ``Coriolis`` and to S's metric source call.
* **Order (R-7, answered by Tianhao).** Hydro's ``post_implicit`` calls
  ``X1Momentum`` whenever the gate at ``hydro_forward.cpp:1319-1322`` holds,
  with or without Coriolis, then Coriolis, then S's metric source. That is the
  pin's order at ``:1631-1657``.

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

* The cubed-sphere traction is rotated into coord's ``flux2global<d>_`` frame by coord's rotation
  (S-2).
* Diffusion keeps its own four-round seam exchange (``share_fluxes``,
  ``diffusion_cubed_sphere.cpp:132-143``) until phase 3. In phase 3 it moves
  onto S's ``Exchange::seam_slab`` as a face-slab row (S-3 as S answered it,
  S.rst:1010-1012). This replaces rev 3's "R's replace for the averaging, own
  exchange for the traces". The exchange then carries the 7-channel traces;
  the averaged flux is posted to the register.
* **Viscous metric terms (H9; hypothesis H13, untested).** Hydro's metric
  source covers only pressure and ``ρvv`` (``gnomonic_equiangle.cpp:654-659``).
  Rotating the Cartesian traction into coord's ``flux2global<d>_`` frame and
  taking R's divergence may therefore lose the viscous metric terms that the
  Cartesian-channel divergence carries today
  (``diffusion_cubed_sphere.cpp:369-372``).

  * Lane A's choice: phase 3 posts only the **energy** flux, a scalar with no
    metric term.
  * The viscous **momentum** keeps its Cartesian-channel divergence as a
    ``DiffusionSource`` until gate G9b (5.3) shows the register route matches
    it.
  * This keeps phase 3 correct whichever way H13 turns out.
* **Radial conduction:** each half step posts its x1 heat flux with
  ``reg.post_split(1, "diffusion.radial", flux, 0.5, dt)`` (R:80-81).
* **Radial failure: abort, settled (Cheng's decision, 1.5).** The pin's host
  check stays: finite and positive ``next`` (``diffusion_radial.cpp:97-100``).
  It runs before the increment ``u[IPR] += capacity · (next − temp)``
  (``:101``). So a failed column's increment is never applied, and the run
  stops, as today.

  * **Tianhao's masking request, answered under abort.** The masking was
    there to keep ``u`` finite until a device flag was read. With the host
    check before ``:101``, no failed increment ever reaches ``u``, so V3 needs
    no mask and no device flag.
  * **The host read is not inside a stage segment.** The half steps run at
    the step's split points, before stage 0 (``meshblock.cpp:715``) and after
    the last stage's combine (``:903-905``). The register orders them
    ``begin_step`` < ``post_split`` < stages (R:6), and X treats the check as
    a segment end (X.rst:153 at ``0fe47c6f``). Requirement 6 is therefore not
    affected.
  * **Note on the reserved path only (not in the PR plan).** If Cheng ever
    moves to reject-and-retry, the host check must become device-side, and
    then the increment needs a mask. A device flag does not stop ``:101``,
    and a redo would otherwise restore from a state that already holds the
    failed increment. The design for that is part 1's:

    * a per-column ``ok`` mask, because SDIRK couples each column
      (``:71-96``);
    * ``next = where(ok, next, temp)``;
    * the wall flux from the masked ``next``;
    * ``reg.flag_retry("diffusion.radial", ...)`` on the reserved cause (4.3).

    I keep it as this note because it is the only part of the reserved path
    that is not an interface: 4.3 fixes the names, and this fixes the order
    "mask before increment".

**Diffusion enters the register in phase 3, not phase 1.** Phases 1 and 2 are
bitwise gated; diffusion has its own tolerance suite (G8); and one PR should
not mix a register defect with a diffusion defect. Until then, ``exact`` with
``nu_iso > 0`` is refused.

3.8 Where the AAM geometry lives (Q8), settled with lane S
----------------------------------------------------------

**S-1, as lane S answered it** (S.rst:1004-1008; ``coord_rotation.hpp:15-18,
48-55`` at ``d03e4888``). Coord owns:

* ``RotationAxis`` (a Cartesian unit vector, only z supported);
* ``AxialRotationData``: ``k_cell``, ``k_face2``/``k_face3``, ``cos2_lat`` and
  ``rate_to_rows_``.

Lane A reads S's ``AxialRotationData`` directly, through S's accessor
``CoordinateImpl::rotation_geometry() const``, which returns
``AxialRotationData const*`` (null without an axis; lane S rev 2 fallback text (local commit ``cf5a22a9``, not pushed; data_in ``laneS-rev2-commit3-L3-fallback.md``)). Lane A's
skeleton defines no view type of its own (rev 4.1 part 2 rename, D24). Lane A owns the AAM physics: the flux form,
the x1 sources, the Coriolis rate, ``omega_local``, the m1 weights, the
torque rules and the closure. These move into
``src/forcing/axial_momentum.cpp`` and ``coriolis.cpp``.

**Lane A's own side (Jiheng's H5, H7, H11):**

* ``rotation_geometry()`` is null on grids without an axis. That null, not
  the type string (``hydro.cpp:28-29``), keys lane A's refusal of the AAM key
  and whether lane A requests the derived row ``"aam"`` (R:34). Lane A
  requests it only when the geometry is non-null (H5).
* The Ω ∥ axis check reads ``RotationAxis::unit``, not ``omega2``/``omega3``
  (``hydro.cpp:32``) (H7).
* Geometry is read through that one accessor, never by ``dynamic_cast``
  (``coriolis.cpp:182-183``) (H11).

**Lane A proposals and lane S's answers** (11.4, S-4 to S-6; S's answers are
from the lane S rev 2 fallback text (local commit ``cf5a22a9``, not pushed; data_in ``laneS-rev2-commit3-L3-fallback.md``), to be re-checked against S.rst when it is pushed):

* **H4.** Lane A owns ``rate_to_rows``, including its on-axis conservation
  rule (``gnomonic_equiangle.cpp:499-514``), and S exposes only data plus an
  on-axis mask (``has_on_axis_cell`` today in the skeleton). S also checks
  ``nx2 >= 3`` at reset when an on-axis cell exists: with one cell per side
  the shares at ``:509-512`` fall off the slices. S's fallback skeleton lists
  ``on_axis`` in ``AxialRotationData`` and no longer lists ``rate_to_rows_``,
  which is consistent with this. S's sentence on ownership and on the
  ``nx2 >= 3`` check is pending S text.
* **H5 (answered by S's fallback skeleton).** ``AxialRotationData`` has
  ``K_f2``, ``K_f3`` and ``s2``. So lane A reads ``K_f``, with the radial
  factor, and ``s²`` from coord, instead of forming the radial factor itself
  (``gnomonic_equiangle.cpp:563-567``).
* **H7 (answered by S's fallback skeleton).** S declares
  ``centre_frame(CoordinateImpl const&)``. Lane A builds ``omega_local`` from
  it on every grid, replacing the per-grid set-up at ``coriolis.cpp:85-151``.

**A non-z axis is out of V3 scope.** The axis is a declared parameter; only z
is supported, and others are refused (FOR CHENG, 10.3).


4. Flux-register contract
=========================

4.1 Lane A against requirements 1-5 and 10
------------------------------------------

**Requirement 1.** From phase 3 lane A owns one post, ``diffusion``. It is a
flux density per unit area and time at the step dt, in the register's
components (R:4-5). The area weighting is the register's divergence. Registered
fluxes stay in coord's ``flux2global<d>_`` frame, and S rotates at seams (S-2).
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
* ``K_f`` comes from coord's rotation data in coord's ``flux2global<d>_`` frame (S-2).
* **G5 runs in phase 2, at the end of ``limit``, before ``divergence``
  (H10, J9; D22).**

  * Before the carry, x2/x3 single-valuedness comes from S's face-state sync
    before the solve (``hydro_forward.cpp:1004-1048``) and R's panel-seam
    ``replace`` with ``SeamRule::owner_copy`` (R ``hydro_stage.hpp:49-51``,
    ``reconcile``, R:16).
  * But lane A's ``A F · K_f`` is built from the post-carry momentum rows (3.1),
    and the carry rewrites the x2/x3 momentum rows on every face with each
    side's own donor-cell velocity (``flux_positivity.cpp:150-152``, called at
    ``hydro_forward.cpp:1234-1235`` before the scale at ``:1237``). At the pin
    those donor-velocity ghosts are interpolated, not synced (#370 adds the
    sync), so the carry does not keep seam single-valuedness exactly.
  * So G5 checks ``A F · K_f`` after ``limit`` (R:17), the state lane A
    reads, and waits on lane S's G-S-K7 there.
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
1     4.1       complies; ``flux2global<d>_`` frame (S-2); diffusion owner from phase 3
2     4.1       complies with R:14-22, 32-39; vic.momentum_t@1 kept (R-4)
3     4.1       complies: R:85, post_split (R:80-81); stage_weight = w2 of
                the stage (R:111, R-6 closed)
4     4.1       complies from phase 2; diffusion from phase 3
5     4.1       complies from phase 3; G5 at the end of limit, after G-S-K7 (D22)
6     5.1, 7.1  complies after the 7.1 host reads go; AAM is the derived row
                (R:34, 72-75); radial host check stays at a segment end (3.7)
7     5.2       complies: ``AxialClosureFixer : Fixer`` posts ``cell_amount``;
                species-clip floor owned and registered by X
8     6, 7.1    complies: parts kFlux, amounts FP64 (FP32 on MPS)
                (R.rst:452-454; R-5 closed)
9     7.2       memory stated from shapes; cost is a measurement plan
10    4.1       complies: nothing restart-written (R.rst:467)
====  ========  ==================================================================

4.3 Reserved retry cause ``diffusion.radial``
---------------------------------------------

The retry cause ``diffusion.radial`` stays in the contract as a **reserved**
slot. It is named in lane X's retry stack (X rev 3.3 ``0fe47c6f``):

* stack slot 8, mapped to cause bit 6 (value 64) (X.rst:146-147);
* zeroed in ``kStepStart`` and counted against ``max_redo`` like bits 0-4
  (X.rst:155-157).

Lane A does not write it in V3, because Cheng keeps the abort (1.5, 3.7). Lane
A's ``RegisterRequest::retry`` is empty in V3 (R:37), and lane A makes no
``flag_retry`` call (R:78).

A later move to reject-and-retry needs no interface change: lane A would add
``"diffusion.radial"`` to ``retry`` and call ``flag_retry``. The device mask in
the note in 3.7 would then become part of that PR, not of V3's.


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
G3b   block-count invariance (H14): AAM drift and     W2 C16, ``exact``, 200 steps; plan only          1e-12 rel  2
      {R, I} with nb = 1 vs nb = 4
G4    ``off`` and ``transport`` against the           arms (a)-(c), (f)-(h); on sedimentation decks    bitwise    2
      reference                                       see the EF-3 rule below                          (rule)
G5    seam single-valuedness of ``A F · K_f``         W2 C16, one stage, every panel-seam face, at     1e-14 rel  2
      (post-carry rows)                               the end of ``limit``, after lane S's G-S-K7
G6    closure: declared vs measured AAM change        arm (l) decks                                    1e-12 rel  2
G7    sponge / drag torque booked = AAM change        ``relax-bot-velo`` drag arm and a top-sponge     1e-15 of   2
                                                      arm, FP64, budgets on                            L
G7s   AAM with sedimentation                          ``test_conservation_budget`` row ``d_sed``       C n eps,   2
                                                      (#370) with ``transport`` and ``exact``          C = 4
G8    diffusion suite                                 ``test_diffusion_cubed_sphere``,                 own tol.   3
                                                      ``test_diffusion_x1_scale``, convection-onset
                                                      notes
G9    viscous torque booked as physical               new arm: ``exact`` with ``nu_iso > 0``, W2 C16   1e-12 rel  3
G9b   register route vs Cartesian-channel             solid-body rotation (zero strain) and uniform    1e-12 rel  3
      divergence of the viscous momentum (H9)         shear, C16, one stage; plan only
G10   FP32 + perturbation-flux AAM drift vs FP64      arms (f)-(h), (k), (l) in FP32 with              drift      4
                                                      perturbation-flux                                ≤ 1e-12
====  ==============================================  ===============================================  =========  =====

**G10's metric (H15, plan only).**
``max_t abs(L(t) − L(0) − Σ booked AAM amounts) / abs(L(0))``, with every term
in FP64 from R's monitor (R.rst:452-454). The seam AAM flux ``A F · K_f`` is
reconciled in FP64, and lane A reads only the FP64 ``K_f`` master, never the
compute-dtype cast views (6.1). Hypothesis H15: FP32 seam rotation rounds at
about 1e-7 per stage, which could put 1e-12 out of reach for ``transport``;
G10 then reports ``transport`` and holds only ``exact`` to 1e-12.

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
H13  the register route loses the viscous metric terms (H9)       gate G9b (plan only)
H14  diffusion and the {R, I} sum order are not invariant in nb   gate G3b (plan only)
     (H14)
H15  FP32 seam rotation (about 1e-7 per stage) puts 1e-12 out     gate G10 with its FP64 metric (plan only)
     of reach for ``transport`` (H15)
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
rotation data                               FP64              lane A reads the FP64 master only (H15)
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
``diffusion_radial.cpp:97-100`` aborting check          stays, as at the pin (Cheng's decision,
                                                        settled): the check runs before the increment
                                                        (``:101``), at the step's split points
                                                        (``meshblock.cpp:715, 903-905``), a segment
                                                        end (X.rst:153). ``diffusion.radial`` is a
                                                        reserved cause (4.3)
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
* **Order against EF's exchange (Z6).** The flux form, F_d and the closure
  residual R read ``total(d)`` in the ``divergence`` phase. So they run after
  EF's ``limit`` phase: total read, theta sync (an exchange), carry, scale
  (R:17-18; EF.rst:648). That includes #370's cross-panel syncs of theta, the
  species enthalpy and the donor velocity (``730b3500``
  ``hydro_forward.cpp:1230, 1254-1271``). Nothing lane A reads in a stage
  precedes that exchange. Gate G5 checks the seam scalar at the same point,
  the end of ``limit``, after lane S's G-S-K7 (D22).
* **Radial split exchange (H13; R-8 answered by Tianhao).** The first radial
  half step and its ``post_split`` (``meshblock.cpp:715``; R:6) come before
  ``apply_boundaries`` and ``exchange_ghost_zones`` (``:716-717``), which come
  before the ``u0`` save (``:721``). That is an exchange segment at step
  start, outside the stages. It belongs in S's exchange table (S-7).


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
diffusion_radial.cpp                103     105    host check kept (abort, 3.7); +2 ``post_split`` wall
                                                   flux (phase 3); no mask or flag in V3
top_cool + bot_heat                 176      70    ``LayerHeat{side}``
top_sponge + bot_sponge             156      50    ``Sponge{side}``
relax_bot_temp/velo/comp            308     235    ``BottomSlab`` base
body_heat + plume_forcing           100      70    base does ctor, reset, options
**total**                          2870    2088
=================================  ======  ======  =======================================================

**Lane A: 2,870 → 2,088 lines (−782, −27.2 %), ``src/forcing`` only, new
files included (RULING V3-1).**

Arithmetic:

* now: 674 + 232 + 745 + 376 + 103 + 176 + 156 + 308 + 100 = 2870
  (2.1, ``wc -l``);
* after: 253 + 110 + 170 + 225 + 480 + 320 + 105 + 70 + 50 + 235 + 70 = 2088;
* cut: 782 / 2870 = 0.272.

J1 drops rev 3 part 1's +8 ``coriolis.cpp`` lines: the metric source stays in
coord (3.4). That moves the number from 2,094 to 2,086.

**H9 sensitivity.** The ``diffusion_cubed_sphere.cpp`` row's −15 assumes gate
G9b passes. If it fails, the viscous momentum keeps its Cartesian-channel
divergence and projection (``:369-373``, 5 lines; 3.7), and the number would
be 2,093 (−27.1 %).

Rev 4 books the ``diffusion_radial.cpp`` change once, here (RULING V3-1;
3.7, 11.4). With the abort kept (part 2a), the host check stays and only the
two ``post_split`` lines are added: 103 -> 105, an estimate. That makes
2,086 -> 2,088. Part 1's 104 (−4 check, +3 mask and flag) is withdrawn.

Cross-references, not in lane A's number:

* R: hydro AAM ranges 136 + 33, with 25 + 15 remaining in R's "after"; and
  ``hydro.hpp:89-133, 219-236, 462-466`` (68; H17);
* S: gnomonic 301 and ``coordinate.hpp`` 23 (16 deleted, 4 kept, 3 blank);
* X: mesh 51.

Rev 2's lane total (3,205 → 2,134) is withdrawn.

**Precondition (N18): lane O's options schema lands before lane A's phase 1.**
Without it, ``src/forcing`` would be 2,399 (−16.4 %) after phase 3 and 2,500
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

**Not in the PR plan:** the radial retry path (``flag_retry`` for
``diffusion.radial``, the device mask and the flag). The abort stays as at
the pin (Cheng's decision; 3.7, 4.3). No phase touches
``diffusion_radial.cpp:97-101``; phase 3 adds only its ``post_split`` lines.

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
``conserve-axial-momentum: full``           refused with a message naming        **open proposal** (FOR
                                            ``exact``; what changes: a deck      CHENG)
                                            with ``full`` stops at reset
                                            instead of running ``exact``.
                                            ``exact`` keeps ``full``'s
                                            ``interp_order != 4`` refusal
                                            (``hydro.cpp:63-66``; H12)
``conserve-axial-momentum-closure``         removed; ``exact`` includes the      **settled** (Cheng)
                                            closure; refused by ``check_keys``
``coriolis/type: 123``                      ``type: xyz``: identical in 3D       **FOR CHENG**
AAM key on a grid with no rotation axis     refused, keyed on a null             **FOR CHENG**
                                            ``rotation_geometry()`` (H5). What
                                            changes, by case (H6;
                                            ``hydro.cpp:28-30, 70-76``): (1)
                                            no axis (Cartesian, cylindrical):
                                            silent no-op today, refused; (2)
                                            an axis but no AAM code
                                            (spherical-polar): silent no-op
                                            today, refused; (3) gnomonic with
                                            a non-axial Coriolis (``omega2``
                                            or ``omega3`` set): warning and
                                            no-op today, refused
non-z rotation axis                         refused; only z in V3 (S-1)          **FOR CHENG**
``exact`` with ``riemann-solver/type``       refused at ``reset()`` with a        **FOR CHENG** (C9,
upwind or plume-roe                         message naming the solver (3.2);     proposal)
                                            what changes: nothing for any deck
                                            (neither works as a hydro solver at
                                            the pin, C.rst:476-488)
radial-conduction positivity failure        abort kept, as at the pin; no        **settled** (Cheng,
                                            behaviour change; retry cause        1794951687)
                                            reserved (4.3)
==========================================  ===================================  ==========================

10.4 Decision log
-----------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c is the kept ``riemann.central``                   no second flux evaluation (3.1; R:35)
D2  modes ``off | transport | exact``; no closure key     settled by Cheng; ``full`` refusal open (10.3)
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
D20 radial abort kept; ``diffusion.radial`` reserved      Cheng's decision (1.5, 3.7, 4.3)
D21 ``exact`` refuses upwind and plume-roe (proposal)     no ``riemann.central``; C9 (3.2, 10.3)
D22 G5 at the end of ``limit``, waits on G-S-K7 (J9 b)    ``A F · K_f`` uses post-carry rows (4.1)
D23 ``X1Momentum``: class lane A, instance held by Hydro  exists without Coriolis; R's booking (3.4)
D24 view renamed: lane A reads S's ``AxialRotationData``  one type, S's; no duplicate view (3.8)
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
* **C-1** is closed. It was closed with I-1 in rev 2, and no C-1 item has been
  in this document since.
* **K-1** was routed by the PM: the kintera half to Xi (lane K), and
  ``ConstGravity`` to Xinyue (lane I). The ``ConstGravity`` half is not a
  lane C or K question. The kintera half is still open (11.4).

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

* **X-1, Xi (updated for Cheng's decision).** The radial failure keeps the
  abort for V3 (1.5). ``diffusion.radial`` stays named in X's retry stack as a
  reserved slot (slot 8, cause bit 6; X.rst:146-147, 155-157 at
  ``0fe47c6f``). Lane A neither requests nor writes it in V3 (4.3).
* **X.rst:153-155, for Xi.** That passage says V3 retries a radial failure
  ("a behaviour change") and that X books the ``diffusion_radial.cpp`` change
  outside the lane count. With the abort kept, there is no retry change to
  book. The only V3 change in ``diffusion_radial.cpp`` is lane A's two
  ``post_split`` lines (phase 3), booked in lane A under RULING V3-1 (9.2).
  Request: X drops the "V3 retries it" and booking sentences, and keeps slot
  8 as reserved.

**New in rev 4 (from Jiheng's H notes).** These are lane A proposals for lane
S or R to confirm. The proposed owner is in brackets.

**Status of S-4..S-8 (rev 4.1 part 2, from the lane S rev 2 fallback text (local commit ``cf5a22a9``, not pushed; data_in ``laneS-rev2-commit3-L3-fallback.md``)):**
S-5 and S-6 are answered; S-4 is answered in skeleton only (text pending); S-7
and S-8 are pending S text. The questions below keep their text. Owner is now
Jiheng (lane S).

* **K-1 (kintera half), Xi (lane K), open.** Are kintera's "Y->X", "DY->V",
  "VT->U" (``thermo_y.cpp:198, 214, 242`` at ``3f5ff938``) and "X->Y"
  (``thermo_x.cpp:141``) conversions, called by ``RelaxBotComp``
  (``relax_bot_comp.cpp:98-111``), capturable in a CUDA graph? Until lane K
  confirms, ``RelaxBotComp`` is ``capturable = false`` (section 8).
* **S-4, Xinyue (lane S): ``rate_to_rows`` and the on-axis rule (H4).**
  Lane A proposes to own ``rate_to_rows``, including the on-axis conservation
  rule (``gnomonic_equiangle.cpp:499-514``) [owner: A], while coord exposes the
  rotation data plus an on-axis mask and checks ``nx2 >= 3`` at reset when an
  on-axis cell exists (``:509-512``) [owner: S]. S rev 1.1 lists
  ``rate_to_rows_`` as coord's (``coord_rotation.hpp:52-54``). Do you agree to
  move it?
* **S-5 (answered: ``K_f2``, ``K_f3``, ``s2`` in S's ``AxialRotationData``).** Can coord provide
  ``K = s·k̂``, its exact face averages ``K_f`` with the radial factor, and
  ``s²``, next to ``k_cell``/``k_face2``/``k_face3``? Today lane A forms the
  radial factor (``gnomonic_equiangle.cpp:563-567``). [owner: S]
* **S-6 (answered: S's ``centre_frame(CoordinateImpl const&)``).**
  ``omega_local`` (lane A's, S-1) needs a never-null centre-frame call on every
  grid: Cartesian, cylindrical, spherical-polar and gnomonic. Today that is
  ``coriolis.cpp:85-151``, and ``PanelRotation`` covers the cubed sphere only.
  [owner: S]
* **S-7 (pending S text; the fallback omits S's 3.3 exchange schedule), Xinyue (lane S): exchange-table rows lane A depends on (H11,
  H13).** Three entries:

  * #370's cross-panel syncs of theta, the species enthalpy and the donor
    velocity (``730b3500`` ``hydro_forward.cpp:1230, 1254-1271``);
  * diffusion's own exchange, ``phyid 12`` (``diffusion_cubed_sphere.cpp:143``),
    until phase 3;
  * the radial split's ``apply_boundaries``/``exchange_ghost_zones`` at step
    start (``meshblock.cpp:716-717``).

  Can these go in S's exchange table? [owner: S]
* **S-8 (pending S text), Xinyue (lane S): the metric source reads A's m1 (H3).** Coord's
  ``x1_mass_source_`` reads lane A's ``X1Momentum`` m1, not a Coriolis
  private. [owners: A provides, S consumes]
* **R-7 (answered by Tianhao, lane R thread; no R sha change).** Yes:
  hydro's ``post_implicit`` calls ``X1Momentum`` whenever the gate at
  ``hydro_forward.cpp:1319-1322`` holds, with or without Coriolis, then
  Coriolis, then S's metric source (pin ``:1631-1657``). Hydro holds the
  instance (3.4).
* **R-8 (answered by Tianhao; no R sha change).** Yes: the radial
  ``post_split`` (``meshblock.cpp:715``) comes before the boundaries and
  exchange (``:716-717``), which come before the ``u0`` save (``:721``)
  (7.3).

FOR CHENG: see 10.3.


12. Skeleton index
==================

``wc -l docs/v3/skeleton/A/*`` gives 281 lines, with declarations only, in
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
``axial_momentum.hpp``      ``AxialMode``, ``AxialFluxView`` (``total``, ``central``,       3.1-3.3, 3.8,
(87)                        ``momentum_t``), ``FaceTorqueRule``, ``AxialMomentum``          5.1, 5.2, 6.2,
                            (``request``, ``check``, ``wall_torque``, ``aam_total``,        7.3
                            ``rate_to_rows``, H4), ``AxialClosureFixer : Fixer``
``coriolis_diffusion.hpp``  ``X1Momentum`` (H3), ``Coriolis`` (``omega_local_``),           3.4, 3.7, 3.6
(92)                        ``DiffusionCoefficients``, ``DiffusionFlux``,
                            ``DiffusionSource``, ``RadialConduction`` (``post_split``),
                            ``Sponge``, ``LayerHeat``, ``BottomSlab``
``rotation_geometry.hpp``   forward declaration of S's ``AxialRotationData`` and the        3.8
(12)                        members lane A reads; no lane-A view type (D24)
==========================  ==============================================================  ===============


Appendix A: responses to Xi's notes N1-N18
==========================================

N14 withdraws N2 and supersedes N5 where they overlap.

* **N1 (accept).** Host copies are listed in 7.1, and {R, I} come from
  ``Fixer::partial_sums()`` on the device.
* **N2 (withdrawn by N14).** ``dt`` is a host double (R:113-115).
* **N3 (accept, then superseded by Cheng's decision).** The abort is kept with
  the pin's host check at a segment end (3.7), so no device flag is written in
  V3. ``diffusion.radial`` is a reserved cause (4.3).
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

Statuses follow Zoey's recheck at ``76904ce8``, as posted in the lane A thread.
Her full numbered text was still to be posted, so Z1-Z3 keep their rev 2
topical answers.

* **Z1-Z3 (status: closed; lines moved).**
  The three rev 2 points stand:

  * the flux form reads ``total(d)`` after the limiter's carry, and F_d is the
    limited total minus the never-scaled kept central (3.1; R:17-18;
    R.rst:455-457);
  * gates compare against main with #370 (5.3, D15; new G7s);
  * the EF-3 rule for G4 (5.3).
* **Z4 (status: closed).** §5.2 now lists EF's floors and repairs (F1:
  ``eos.floor``, ``eos.borrow``, ``eos.fix_vapor``) and the positivity posts
  P2 and P3, each with its AAM effect (EF.rst:556-597). §11.3 answers Q-A1
  (yes) and Q-A2 (no objection, with the ``specific_heat_cv`` correction).
* **Z5 (status: moot).** ``diffusion.radial`` is a reserved cause (4.3), so
  no flag rides EF's ``EfStatus`` in V3. The radial failure keeps
  the abort, so no flag rides EF's ``EfStatus`` in V3. The reserved path
  would use ``flag_retry("diffusion.radial")`` (R:78) and, until R's PR R-1,
  EF's ``EfStatus`` in X's stacked read (EF.rst:319-342, 606-607) (4.3).
* **Z6 (status: done in rev 4 final).** §6.1 has the P2/P3 dtype row (flux
  dtype in ``total(d)``, FP64 amounts, the flux form and F_d formed in FP64).
  §7.3 now has the missing sentence: the flux form, F_d and R run in the
  ``divergence`` phase, after EF's theta sync (an exchange) and #370's
  cross-panel syncs.
* **Z7 (status: no change requested).** The count is now 2,870 -> 2,088
  (−27.2 %), and the 9.2 rows sum to it.
* **Z8 (status: no change requested).**


Appendix C: Jiheng's round-2 notes H1-H17
=========================================

Jiheng's notes cite lane A at ``9ee83756``. Each cited line was re-found at
``76904ce8`` before editing, and other sites were checked at the pin
``9e5de2d4`` and at lane R ``4c59162a``. The H1-H17 here are Jiheng's note
numbers; the hypothesis ids of 5.4 (H1-H15) are lane A's own and are a
separate list.

**Choices this revision made where the notes left them open, with the
reasons:**

* **H1 is declined, not accepted.** S rev 1.1 states S-1..S-3 in its own
  text. Calling them pending would misstate lane S. Only the parts beyond
  that text are marked pending.
* **H9: phase 3 registers only the energy flux.** A scalar flux has no metric
  term, so this is correct whichever way H13 resolves. The −15 in 9.2 assumes
  G9b passes, and the cost if it fails is stated (2,093).
* **H4: the skeleton already moves ``rate_to_rows`` into ``AxialMomentum``,
  marked pending S-4.** S's ``AxialRotationData`` keeps only the mask. If S
  declines, this reverts with no other change.
* **H14/H15: the new gates are G3b and G9b, plus G10's metric, all marked as
  plans only.** Nothing is run under CHANGE 5.
* **H3: the gate for building m1 stays #355's gate** (``hydro_forward.cpp:1319-1322``),
  so its bits are unchanged.

* **H1 (decline, with the reason).** Lane S did answer S-1..S-3. S rev 1.1
  (``d03e4888``, the current ``v3-design/S`` head, checked with ``ls-remote``)
  records them as settled at S.rst:65-74 and 1002-1012. So lane A keeps them
  settled, but restates them as S wrote them (1.5, 3.8). Everything beyond
  S's text is now worded "lane A proposal, pending lane S": H4, H5, H7, the
  exchange rows and m1 (3.8; 11.4 S-4 to S-8).
* **H2 (agree; already resolved).** J1 is unchanged: ``x1_mass_source_`` stays
  in coord, and S's count is 166 − 13 = 153 physics lines (2.2, 3.4).
* **H3 (accept).** m1 moves out of ``Coriolis`` into lane A's standalone
  ``X1Momentum``. It is built whenever #355's gate holds, and both Coriolis
  and coord's metric source read it (3.4; skeleton
  ``coriolis_diffusion.hpp``). The cross-lane parts are R-7 and S-8.
* **H4 (accept as a lane A proposal, pending S).** Lane A owns
  ``rate_to_rows`` and the on-axis rule; S exposes data plus the on-axis mask
  and checks ``nx2 >= 3`` (3.8; S-4). Skeleton: ``rate_to_rows`` moves to
  ``AxialMomentum``.
* **H5 (accept).** On lane A's side, a null ``rotation_geometry()`` keys the
  refusal and the ``"aam"`` request (3.8). The axis as a Cartesian unit vector
  matches S's ``RotationAxis::unit`` (``coord_rotation.hpp:15-18``). ``K``,
  ``K_f`` and ``s²`` from coord are question S-5. The AAM-key refusal stays a
  FOR CHENG proposal, keyed on the null geometry (10.3).
* **H6 (accept).** The 10.3 row now lists the three cases with what changes
  in each, citing ``hydro.cpp:28-30, 70-76``.
* **H7 (accept).** On lane A's side, the Ω ∥ axis check reads
  ``RotationAxis::unit`` (3.8). The never-null centre frame is question S-6.
* **H8 (accept).** "Panel face basis" becomes "coord's ``flux2global<d>_``
  frame" throughout (R:4-5; S.rst:118-122).
* **H9 (accept as a hypothesis, H13).** Phase 3 posts only diffusion's energy
  flux, and the viscous momentum keeps its Cartesian-channel divergence until
  gate G9b (plan only) shows the register route matches (3.7, 5.3).
* **H10 (accept).** G5 runs in phase 2 and cites R's ``owner_copy`` slot
  (R ``hydro_stage.hpp:49-51``, R:16). Rev 4.1 places it at the end of
  ``limit``, after lane S's G-S-K7, because ``A F · K_f`` is built from the
  post-carry rows (4.1, 5.3; D22).
* **H11 (accept).** Geometry is read through one accessor, not by
  ``dynamic_cast`` (3.8). The #370 syncs and ``phyid 12`` are question S-7.
* **H12 (accept).** ``exact`` keeps ``full``'s ``interp_order != 4`` refusal
  (``hydro.cpp:63-66``; skeleton ``AxialMomentum::check``). Refusing ``full``
  itself stays a FOR CHENG proposal (10.3).
* **H13 (accept).** The radial split exchange at ``meshblock.cpp:716-717`` is
  in 7.3. Cross-lane parts: S-7 (S's table) and R-8 (its order, answered:
  ``post_split`` at 715, then 716-717, then the ``u0`` save at 721).
* **H14 (accept as a hypothesis, H14).** New gate G3b, nb = 1 vs nb = 4 on W2
  C16, as a plan only (5.3).
* **H15 (accept as a hypothesis, H15).** G10's metric is defined in FP64, and
  lane A reads only the FP64 ``K_f`` (5.3, 6.1), as a plan only.
* **H16 (agree).** The ranges sum to R 136 + 33, S 166 + 135 = 301 and X 51,
  matching RULING V3-1.
* **H17 (accept).** ``hydro.hpp:89-133, 219-236, 462-466`` (68 lines, R's) is
  added to both cross-reference lists (2.2, 9.2). Lane A's number does not
  change: RULING V3-1 counts it in R.
