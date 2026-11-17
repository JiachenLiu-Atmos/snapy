=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 2, part 1 (design and contract only, CHANGE 5). Answers Xi's round-2
           notes N1-N18 (appendix). Register names marked † are provisional
           until lane R's header (part 2).
:Reviewers (round 2): Xi (lane X), Zoey
:Pins: snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410``; kintera ``3f5ff938``
       (section 8); lane X design read at ``77c65d5aac1f96ff6c34a45ef48faeb996c2128e``
:Previous: revision 1 ``2c9589c68c633796cfa5f749d003af7a2b57f38a``
:Skeleton: ``docs/v3/skeleton/A/``, four headers, declarations only, 299
           lines (``wc -l docs/v3/skeleton/A/*``); index in section 12

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
  gravity's interface, but not ``const_gravity.cpp`` itself (section 11,
  K-1);
* the AAM machinery in hydro, coordinate and mesh;
* the source-term interface;
* diffusion as a register contributor.

The sections follow the round's common order: 1 scope, 2 current sites, 3
target design, 4 register contract, 5 invariants and gates, 6 FP32, 7 CUDA
graph, 8 cross-repository changes, 9 line budget, 10 migration, 11 open
questions, 12 skeleton index. The responses to round-2 notes N1-N18 follow as
an appendix.

1.2 Pins
--------

* snapy ``9e5de2d4`` (all ``src/``, ``tests/``, ``python/`` references);
* kintera ``3f5ff938`` (section 8; the pin listed in X.rst:11);
* lane X design ``77c65d5a`` (X.rst, read for the global phase, segments and
  the line rows);
* pyharp ``6bc01d53`` and pydisort ``9e7f0075``, the X.rst:11 pins. Lane A
  uses neither (section 8).

1.3 The ten register requirements (brief 2/3)
---------------------------------------------

1. One owner of every conservative face flux, one documented frame and area
   weighting.
2. Named increments in a declared order; total and parts reportable.
3. Stage fluxes accumulated with the integrator's stage weights into the step
   transfer.
4. Flux divergence and every flux-derived term (gravity work, the ``rho w`` of
   Coriolis and metric sources, the axial torque) read the register and never
   rebuild a flux.
5. A shared face has two entries and one reconciliation rule;
   single-valuedness is a gate; rotation is one function owned by coord.
6. One device-side monitor for mass, species, E+PE, AAM and scalars; no host
   reads inside a segment.
7. Every fixer and floor registers what it added.
8. A stated dtype under FP32, and fixed storage a CUDA graph can capture.
9. No added full-field copy per stage; memory and cost stated.
10. The register is rebuilt, not written to restart.

1.4 Sources
-----------

Read:

* ``src/forcing/*``; the AAM parts of ``src/coord/gnomonic_equiangle.*`` and
  ``coordinate.hpp:370-392``; ``src/hydro/hydro_forward.cpp:988-1009,
  1088-1145, 1288-1411, 1624-1696``; ``src/hydro/hydro.{hpp,cpp}``;
  ``src/mesh/meshblock.cpp:819-1110, 1456-1480``; ``src/mesh/mesh.cpp:360-402``;
  ``src/output/restart.cpp:26-67``; ``src/riemann/hllc_impl.h:100-163``;
  ``V3_ROADMAP.md:258-279``.
* The PR bodies of #355, #357, #363 and #365, and issue #314.
* X.rst at ``77c65d5a``.
* The lane I answers in the lane I thread (``vic.*`` names, the 18 forcing
  lines).

Not read: the Slack side-study thread on the local-torque scheme. Everything
said about it here comes from the lane brief's one-line description.

1.5 Provisional names (marker †)
--------------------------------

Lane R's interface header (Tianhao, backup owner) is not yet available. Every
register name lane A did not get from another lane's written answer carries one
marker, †, and part 2 of this revision switches them to the header's names:
``riemann.central``†, ``riemann.dissipative``†, ``covariance``†,
``diffusion``†, the fixer base class ``Fixer``† and ``FluxContributor``†.

Lane I's names are final (lane I thread): ``vic.mass_total``,
``vic.mass_dry``, ``vic.mass_species`` and ``vic.momentum_t[2]``.

1.6 Decisions recorded
----------------------

* **Settled by Cheng:**

  * ``full`` becomes ``exact`` (full plus the closure);
  * the key ``conserve-axial-momentum-closure`` is removed;
  * the old ``full`` is refused with a message naming ``exact``;
  * inside ``exact``, the closure is a registered fixer.
* **FOR CHENG** (nothing waits on them; section 10.3):

  * abort versus redo for the radial-conduction check;
  * Coriolis ``type: 123``;
  * the AAM key on grids with no rotation axis.
* **Default:** ``off`` (needs no decision, per Cheng's PM).


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
20 ``//!`` comments and 259 boilerplate. Command: ``awk`` over
``struct .*OptionsImpl`` ... ``using .*Options =``. Another 270 lines are the
module classes.

2.2 AAM code outside ``src/forcing``, and who books it
------------------------------------------------------

Counted with ``sed -n 'a,bp' FILE | wc -l`` for each range. The booking
follows the round's ownership rules: ``hydro.hpp`` belongs to R, and
``src/mesh`` to X.

================================================  =======  =======  =======================================
range                                             lines    booked   what
================================================  =======  =======  =======================================
hydro_forward.cpp:988-1003, 1108, 1141                 18  A        full: second Riemann eval (central)
hydro_forward.cpp:1288-1298                            11  A        transport: AAM flux form call
hydro_forward.cpp:1316-1363, 1390-1391                 50  A        Coriolis gate, x1 mass-w buffers, F_d
hydro_forward.cpp:1624-1680                            57  A        step 8: x1 mass-flux w, metric source
hydro.cpp:27-37, 55-76                                 33  A        mode predicates and checks
coordinate.hpp:370-392                                 23  A        four virtual AAM / x1-source hooks
gnomonic_equiangle.hpp:22-35, 84-122                   53  A (S-1)  AAM buffers and declarations
gnomonic_equiangle.cpp:18-36, 169-221                  72  A (S-1)  rotation field k, face averages
gnomonic_equiangle.cpp:485-604, 664-719               176  A (S-1)  rate_to_rows, x1 source, flux form, rate
**lane A subtotal**                               **493**
hydro.hpp:89-133, 219-236, 462-466                     68  R        AAM options (docs), accessors, state
meshblock.cpp:983-990, 1009-1034                       34  X        closure reduction and increment
mesh.cpp:364, 367-382                                  17  X        closure over local blocks
================================================  =======  =======  =======================================

Revision 1 counted all 612 lines for A. Revision 2 counts 493: the 68 lines go
to R and the 51 to X. The geometry rows are A's until lane S accepts S-1.

2.3 What the code does today
----------------------------

* **Forcings are a flat list of** ``AnyModule`` **objects**
  (``register_forcing_modules.cpp:6-86``). They are called in order on a
  shared ``du`` (``hydro_forward.cpp:1378-1385``), and nothing records what
  each one added.
* **Coriolis has two classes.** ``Coriolis123`` (``coriolis.cpp:40-67``) and
  ``CoriolisXYZ`` (``:69-231``). On Cartesian grids they agree only in 3D:
  ``123`` adds the IVZ row only when ``w.size(1) > 1`` (``:61-63``), while
  ``xyz`` always does (``:226-228``). No deck sets ``type: 123``.
* **#355 (x1 mass-flux w)** calls the Coriolis forward twice per stage on
  hand-built tensors (``hydro_forward.cpp:1351-1352, 1384, 1650-1655``).

  * φ is a mass rate: the explicit flux times the area, plus the implicit
    transfer divided by dt (``:1637-1646``).
  * ``lo`` and ``hi`` already hold ``(r² − r_lo²)/(2rV)`` and
    ``(r_hi² − r²)/(2rV)`` (``:1340-1342``).
  * ``m1 = φ_lo·lo + φ_hi·hi`` (``:1652-1653``).
* **transport (#357)** rewrites the K component of the finished divergence
  (``hydro_forward.cpp:1290-1298``, ``gnomonic_equiangle.cpp:551-604``).
* **full (#363)** rebuilds a central flux with two extra Riemann calls per
  x2/x3 direction (``hydro_forward.cpp:997-1003, 1108, 1141``). The states it
  uses are the post-terrain ``wlr2`` (``:1089-1090``). ``F - F_c`` then
  contains the #289 covariance (``:1110``) and the positivity limiter's
  species scaling (``:1165-1190``).
* **closure (#365)** has these host reads:

  * {R, I} are copied to the host every stage (``coriolis.cpp:204-207``);
  * the sums start as CPU zeros (``meshblock.cpp:1011``, ``mesh.cpp:371``);
  * the apply reads two ``.item()`` (``meshblock.cpp:1018``).
* **Diffusion keeps its own fluxes, seam rule and divergence**
  (``diffusion.cpp:671-673``; ``diffusion_cubed_sphere.cpp:103-176,
  368-373``). It also has a Strang-split radial conduction
  (``diffusion_radial.cpp``) with an aborting host check (``:97-100``).
* **Floors:** the species clip of user stage forcings
  (``meshblock.cpp:879-884``).
* **Restart writes every defined entry of the block's** ``Variables``
  (``src/output/restart.cpp:31-37``).


3. Target design
================

3.1 AAM as a consumer of the register (Q1)
------------------------------------------

AAM reads fluxes and never evaluates them (requirement 4):

=====  ==============================================  ===========  =====================================
id     quantity                                        posted by    used by
=====  ==============================================  ===========  =====================================
A-r1   total reconciled face flux, x1/x2/x3            register     transport: K-component in flux form
A-r2   ``riemann.central``†: one summed mass row on    R            exact: torque of F_d = F - F_c;
       x2/x3 faces, 1/2 Σ_s (ρ_s u_n)_L +                           closure residual R
       (ρ_s u_n)_R, post-terrain states, valid to
       stage end
A-r3   explicit x1 mass transfer (area-weighted rate)  register     m1 (#355)
A-r4   ``vic.mass_total`` (``vic.mass_dry``,           I            m1; replaces
       ``vic.mass_species``)                                        ``implicit_mass_correction``
A-r5   ``vic.momentum_t[2]``, posted under scheme 9    I            ``axial_x1_source_``; replaces its own
       too                                                          prefix sum (``gnomonic_equiangle.cpp:
                                                                    541-543``)
=====  ==============================================  ===========  =====================================

``riemann.central``† is a named part of the Riemann increment. The rest of the
Riemann flux is ``riemann.dissipative``†. Details:

* It is one summed row, because AAM only uses the total mass flux
  (``gnomonic_equiangle.cpp:690-695``).
* It is formed from the same post-terrain face states as the solver, with the
  solver's face-normal ``u_n`` after its frame projection (N9).
* F_d = F − F_c therefore contains the covariance and limiter edits, as it does
  today.
* In FP32, HLLC replaces ``u − b_m`` with ``c·q`` or ``u + TINY_NUMBER``
  (``hllc_impl.h:102-105``). So G2 is an FP64 gate.

3.2 Modes (Q2), settled
-----------------------

**Modes: off | transport | exact (settled by Cheng).**

* ``exact`` is ``full`` plus the closure.
* The old ``full`` is refused with a message naming ``exact``, so no deck
  silently gains the closure's solid-body increment (``meshblock.cpp:1027-1033``).
* The removed closure key is refused by ``check_keys`` (``hydro_options.cpp:56``).
* ``full`` without the closure survives only as a test hook on the consumer
  (``AxialMomentum``, not a deck key). It keeps arm (l)'s "closure off must
  fail" check (``tests/test_axial_momentum_flux_form.py:332, 565``).

Evidence, from the PR bodies (not rerun):

* ``transport``: 50-290 times below ``off`` in 3D, but worse than ``off`` at
  Ω/(u/a) = 14 (``hydro.hpp:101-102``).
* ``full``: 217-589 times below ``off``, not exact, 1.14-1.48 times the cost
  of ``off``.
* closure: 1.0e-15 to 1.5e-15 drift, 1.04-1.09 times the cost of ``full``.

**Default: off** (decision moved here from rev 1 §8.1; no decision needed per
Cheng's PM). Reasons:

* phases 1 and 2 have bitwise gates;
* ``transport`` is measured worse than ``off`` at high rotation;
* ``exact`` needs FP64 until G10.

The AAM budget is always reported on grids with a ``RotationGeometry``.

**Side-study torque.** The torque is a pluggable ``FaceTorqueRule``, and #363's
rate is rule 1. The side study's rule replaces it only if it passes G3 with no
reduction, G4, the noisy-jet KE bound, and a cost measurement after approval.

3.3 The closure as a registered fixer (Q3)
------------------------------------------

``AxialClosureFixer`` is a registered fixer (requirement 7, section 5.2).

* It sits at the stage's last momentum change, after the saturation
  adjustment and before the boundaries (``meshblock.cpp:983-990`` today).
* R = ``Σ V (K · F_cell − rate(riemann.central†))``. It is computed once from
  A-r2; today it is computed twice (``coriolis.cpp:199-203``).
* {R, I} stay FP64 device tensors.
* Lane X's global phase reduces them **through the host and never captures
  the collective** (X.rst:98, 187; N14). They share one fixed 7-vector
  ``{R, I, D, m, wall, mwall, redo}`` with the gravity-work fixer
  (``meshblock.cpp:1058-1060``; N5).
* ``apply`` runs on the device. Its ``I > 0`` and finiteness checks are
  ``where`` masks.

3.4 One Coriolis and metric source (Q4)
---------------------------------------

``force = -2 Ω × m``, lowered to covariant rows, with:

* ``m = (m1, ρ v2, ρ v3)``;
* ``m1 = φ_lo·lo + φ_hi·hi``, where φ is the mass rate (A-r3 plus
  ``vic.mass_total``) and ``lo``/``hi`` hold ``1/(2rV)``.

The metric pair ``-m1 v_h / r`` is part of the same source. It has two phases:

* ``pre_implicit``: the w-free rows; they pass through the scheme-9 solve
  (#355: 1.27 m/s otherwise);
* ``post_implicit``: the m1 rows.

Ω in the local basis comes from ``RotationGeometry`` (3.8). The grid gate is
#355's, unchanged.

3.5 FP32 restriction on exact (Q5)
----------------------------------

See section 6.2.

3.6 One source-term interface (Q6)
----------------------------------

Every module is a ``SourceTerm`` and declares:

* ``phase``;
* ``rows``;
* ``kind`` (physical, numerical or fixer);
* ``region``;
* ``capturable`` (N14; false means lane X runs it in an eager segment, X.rst:76-77).

``SourceRegistry::runs(phase)`` returns maximal runs of equal
``capturable``, in declaration order, so sources are never reordered and phase
1 stays bitwise.

The twelve modules in V3:

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
N16). They run after ``Hydro::forward`` (``meshblock.cpp:871-877``). Moving
them into a lane-A phase would change their order and break G1. Their species
clip is a floor registered by X (5.2).

3.7 Diffusion as a register contributor (Q7), entering in phase 3
-----------------------------------------------------------------

From phase 3, ``DiffusionFlux`` posts ``diffusion``† per face direction:

* the viscous momentum flux ``-μ τ · n``;
* the energy flux ``-k ∇T · n − v_f · τ · n``, which includes the viscous
  heating.

The cubed-sphere traction (``diffusion_cubed_sphere.cpp:297, 344``) is rotated
by coord's one rotation function (S-3). The symmetric interior-penalty term
(``:164-173``) keeps a trace exchange. The radial conduction stays an
operator-split source, with a device failure flag (N3).

**Decision (moved here from rev 1 §8.2): diffusion enters the register in
phase 3, not phase 1.** Reasons:

* phases 1 and 2 are bitwise gated, and registering diffusion changes the order
  of additions;
* diffusion has its own tolerance suite (G8);
* a register defect and a diffusion defect should not land in one PR.

Until then, ``exact`` with ``nu_iso > 0`` is refused, because the closure
would otherwise remove a physical torque.

3.8 Where the AAM geometry lives (Q8)
-------------------------------------

The geometry moves into the coordinate, as ``RotationGeometry``, proposed for
lane S (S-1, now Xinyue). It holds:

* ``k`` and its exact face averages, ``k·k`` and ``cos² lat`` at faces;
* ``rate_to_rows``;
* ``omega_local``;
* the m1 weights;
* ``has_on_axis_cell``, set at reset. This replaces the host read at
  ``gnomonic_equiangle.cpp:506``.

The AAM physics (consumer, torque rules, closure fixer) goes in
``src/forcing/axial_momentum.cpp`` (lane A).


4. Flux-register contract
=========================

4.1 Lane A against requirements 1-5 and 10
------------------------------------------

**Requirement 1 (one owner, one frame, one area weighting).** Lane A owns one
flux, ``diffusion``†, from phase 3. It is posted per unit area in the
register's face frame (S-2), and the area weighting is the register's
divergence. ``A F · K_f`` is derived from the register and never posted.
``riemann.central``† is R's.

**Requirement 2 (named increments, declared order, parts reportable).**

* Order: ``riemann``† (parts ``central``†/``dissipative``†), then
  ``covariance``† (#289), then ``diffusion``†, then face fixers.
* ``diffusion``† has no mass rows, so its position does not affect the torque.
* Parts are reported as integrals. If the register must keep a part as a
  field, diffusion's costs 3(5+ny) face rows (R-1).

**Requirement 3 (stage weights).**

* Budgets and the boundary AAM flux use ``w2_s Π_{t>s} w1_t``
  (``hydro_forward.cpp:1683-1691``).
* The closure applies ``w2_s`` inside the stage (``meshblock.cpp:1027``).
* The radial conduction's two half-step wall fluxes need a split entry with
  explicit weight 1/2. That is on Tianhao's list (R-2).

**Requirement 4 (read, never rebuild).**

* ``transport`` reads A-r1; ``exact`` reads A-r1 and A-r2. The two extra
  Riemann calls (``hydro_forward.cpp:997-1003, 1108, 1141``) go.
* m1 reads A-r3 and ``vic.mass_total``. The ``_flux1`` times area rebuild
  (``:1637-1646``) goes.
* ``axial_x1_source_`` reads ``vic.momentum_t[2]`` and drops its prefix sum.
* No source term builds a face flux.
* Diffusion keeps its own divergence until phase 3 (3.7).

**Requirement 5 (reconciliation, rotation in coord).**

* AAM reads the reconciled flux.
* ``K_f`` is built with coord's rotation function.
* Gate G5: ``A F · K_f`` is single-valued on seam faces.
* In phase 3, diffusion drops the averaging half of ``share_fluxes``
  (``diffusion_cubed_sphere.cpp:159-163``).
* Questions to Xinyue (lane S): S-2 and S-3.

**Requirement 10 (rebuilt, not restart-written).** Restart writes every
defined ``Variables`` entry (``restart.cpp:31-37``), so lane A adds none. Its
buffers (``omega_local``, the m1 weights, ``scratch_``, ``budget_``) are
rebuilt at reset. ``{R, I}`` are stage-local.

4.2 Compliance table
--------------------

====  ========  ==================================================================
req   section   status for lane A
====  ========  ==================================================================
1     4.1       complies; diffusion owner from phase 3; frame per S-2
2     4.1       complies; ``riemann.central``† on lane R's list (R-1)
3     4.1       complies; split radial entry on lane R's list (R-2)
4     4.1       complies from phase 2 (AAM, closure, m1, vic.*); diffusion phase 3
5     4.1       complies from phase 3; gate G5; S-2, S-3 open
6     5.1, 7.1  complies after the host reads in 7.1 go; radial flag (N3, X)
7     5.2       complies: closure registered; species clip registered by X
8     6, 7.1    complies: dtypes in 6.1; fixed buffers and ``capturable`` in 7.1
9     7.2       memory stated from shapes; cost is a measurement plan (7.2)
10    4.1       complies: nothing in ``Variables``; everything rebuilt at reset
====  ========  ==================================================================


5. Invariants and gates
=======================

5.1 Monitor (requirement 6)
---------------------------

* *AAM:* ``L = Σ V r K · m``, always reported on grids that have a
  ``RotationGeometry``.
* *Per-source attribution (N6, N17):*

  * On monitor steps each lane-A source writes into the registry's one FP64
    ``scratch_``, zeroed per source.
  * Its six reductions (mass, m1, m2, m3, E, AAM) go into the device buffer
    ``budget_[nsrc, 6]``.
  * The scratch is then added once to ``du`` (FP64 run) or to
    ``_forcing_work`` (FP32 run).
  * Bitwise: every lane-A source writes each element at most once and none
    reads ``du`` (grep of every ``du`` write in ``src/forcing``). So 0 + x = x,
    and ``du + x`` matches today's ``du += x``.
  * Exception outside lane A: ``const_gravity.cpp:52, 58, 63`` adds to
    ``du[IPR]`` up to three times (K-1).
  * In an FP32 run the scratch is skipped. Shares are differences of
    reductions of ``_forcing_work``, which holds only sources
    (``hydro_forward.cpp:1365-1375``).
* *Graph topologies:* budgets on and budgets off are two captured topologies.
  The monitor cadence picks one.

5.2 Fixers and floors (requirement 7)
-------------------------------------

=============================  =======  =============  ==========  ========================================
fixer / floor                  kind     rows           registers   declares
=============================  =======  =============  ==========  ========================================
AAM closure (``exact``)        fixer    IVY, IVZ       A           AAM change ``-w2 dt R`` (checked by G6);
                                                                   KE moved to IE; {R, I} in the 7-vector
user-forcing species clip      floor    ICY.., IDN     X           species mass withheld per stage
(``meshblock.cpp:879-884``)
side-study torque (rule 2)     numer.   IVY, IVZ       A           its AAM rate; no reduction
=============================  =======  =============  ==========  ========================================

Not fixers:

* ``relax_bot_comp.cpp:104`` clamps the target, and the applied increment
  conserves mass (``:113-117``);
* the diffusion wall-coefficient fallbacks only shape a flux.

5.3 Gates (specified, not run)
------------------------------

Each gate states the case it runs on and whether it is bitwise or has a
tolerance. They run on the migration PRs after approval, not in this design
round.

====  ==============================================  ===============================================  =========  =====
id    gate                                            case                                             kind       phase
====  ==============================================  ===============================================  =========  =====
G1    existing tests against the pin                  full ctest list at the pin, CPU, FP64 and FP32   bitwise    1
                                                      storage, budgets on and off
G2    ``riemann.central``† vs today's central flux:   AAM test (``test_axial_momentum_flux_form``)     tolerance  2
      e_∞ = max_f abs(ΔF_c) / max_f ρ(abs(u_n) + c)   W2 C16 and 3D arm decks, one stage, FP64, all
      ≤ 1e-13; L2 analogue reported                   x2/x3 faces
G3    ``exact`` AAM drift ≤ 1e-12 relative            arm (l): 3D at Ω = 5e-6 and 7e-5; W2 at 1 Ω and  tolerance  2
                                                      4 Ω
G4    ``off`` and ``transport`` against the pin       arms (a)-(c), (f)-(h) with off and transport     bitwise    2
G5    seam single-valuedness of ``A F · K_f``         W2 C16, one stage, every panel-seam face         1e-14 rel  2
G6    closure: declared vs measured AAM change        arm (l) decks                                    1e-12 rel  2
G7    sponge / drag torque booked = AAM change        ``relax-bot-velo`` drag arm (as #365) and a      1e-15 of   2
                                                      top-sponge arm, FP64, budgets on                 L
G8    diffusion suite                                 ``test_diffusion_cubed_sphere``,                 their      3
                                                      ``test_diffusion_x1_scale``, convection-onset    own tol.
                                                      notes
G9    viscous torque booked as physical               new arm: ``exact`` with ``nu_iso > 0``, W2 C16   1e-12 rel  3
G10   FP32 + perturbation-flux AAM drift vs FP64      arms (f)-(h), (k), (l) in FP32 with              drift      4
                                                      perturbation-flux                                ≤ 1e-12
====  ==============================================  ===============================================  =========  =====

5.4 Hypotheses and how each becomes a gate or a measurement plan
----------------------------------------------------------------

Run-dependent claims are hypotheses until their gate or measurement runs. None
was run in this round.

===  ===========================================================  ==============================================
id   hypothesis                                                   decided by
===  ===========================================================  ==============================================
H1   ``riemann.central``† equals today's central mass flux in     gate G2 (tolerance, FP64)
     FP64
H2   ``exact`` costs about ``transport`` plus one rate; the       measurement plan M1 (7.2), after approval
     closure keeps one host round trip per stage
H3   the error of today's ``F(w, w)`` scales as eps·ρ·c, not      reported with G2 (N4); no separate gate
     eps·ρ·abs(u) (N4)
H4   in FP32 the face-state form of ``riemann.central``† and       gate G10 (tolerance)
     ``F(w, w)`` differ by about TINY_NUMBER·ρ, harmlessly
     after the ``O(h)`` factor
H5   diffusion in the register makes the viscous torque           gate G9 (tolerance)
     telescope
H6   the per-source scratch is bitwise against the pin            gate G1 (bitwise), budgets on and off
H7   removing the per-stage host reads (7.1) changes no bits      gates G1, G4 (bitwise)
H8   NCCL or commux/UCX capture of the fixer collective           not needed: X never captures it (N14);
                                                                  recorded for lane X only
H9   on MPS (no FP64; ``meshblock.cpp:1057``) the closure's       unverified; no MPS gate; ``exact`` on MPS
     {R, I} can be reduced on the host in FP64                    stays a hypothesis
H10  m1 from x1 transfers would help Cartesian f-plane energy     out of scope; no gate
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
per-source ``scratch_`` (monitor steps)     FP64              not used: shares come from
                                                              ``_forcing_work`` reductions
``budget_[nsrc, 6]``                        FP64              FP64
``riemann.central``† (one row)              FP64              flux dtype; ``F − F_c`` and the rate in
                                                              FP64 (H4)
AAM flux form, torque rate, R and I         FP64              FP64; summed in FP64 before reduction
                                                              (N1)
``diffusion``† part                         FP64              flux dtype, as the other parts
RotationGeometry buffers, m1 weights        FP64              FP64 master, compute-dtype view (lane S)
==========================================  ================  =========================================

6.2 What lifts the restriction on exact (Q5)
--------------------------------------------

Three things together lift today's refusal (``hydro_forward.cpp:993-995``;
G5 in ``V3_ROADMAP.md:268-270``):

1. a consistent ``riemann.central``†, formed from the states the solver used
   (3.1, R-1);
2. FP64 accumulation of the rate and of R and I;
3. gate G10.

Until G10 passes, ``exact`` refuses FP32 and perturbation-flux at reset.


7. CUDA graph
=============

7.1 Capturable storage and host reads (requirements 6 and 8)
------------------------------------------------------------

* Every lane-A buffer is allocated at reset with fixed shapes: ``omega_local``,
  the m1 weights, ``scratch_``, ``budget_`` and ``{R, I}``.
* ``dt`` stays a host ``double``. It is read once at step start, outside the
  segments (X.rst:93, 180), and a captured graph is keyed on dt (X.rst:82-83).
  This is N14, which withdraws N2.
* Sources with ``capturable = false`` run in eager segments.

Host reads lane A removes or moves out of segments:

======================================================  ===============================================
today                                                   V3
======================================================  ===============================================
``coriolis.cpp:204-207`` {R, I} to CPU per stage        0-d FP64 device tensors until X's global phase
``meshblock.cpp:1011``, ``mesh.cpp:371`` CPU zeros      device zeros (X's code)
``meshblock.cpp:1018`` two ``.item()``                  device scale with ``where(I > 0)``
``gnomonic_equiangle.cpp:506`` on-axis ``.item()``      ``has_on_axis_cell`` set at reset
``diffusion.cpp:507`` ``torch::equal`` per call         frozen options, checked at build (lane O)
``diffusion_radial.cpp:97-100`` aborting check          device flag on the redo slot; action stays
                                                        abort (FOR CHENG: redo)
``diffusion.cpp:677-743`` ``max_time_step``             stays, in the driver's dt step
======================================================  ===============================================

7.2 Memory (stated) and cost (measurement plan, requirement 9)
--------------------------------------------------------------

Memory is stated from tensor shapes; nothing was measured. One face row is
one value per face of one direction.

============================================  =================================  ==============================
item                                          today (pin)                        V3
============================================  =================================  ==============================
``exact`` central flux, per stage             ``cen2``, ``cen3`` 2(5+ny) rows;   ``riemann.central``†: 2 rows
                                              transient 4(5+ny); stored          (one per direction)
                                              ``F − F_c`` 2(5+ny)
per-source scratch                            none                               one FP64 field, 8·(5+ny) B per
                                                                                 cell, FP64 runs only,
                                                                                 allocated at reset
per-source budgets                            none                               ``[nsrc, 6]`` FP64
closure                                       2 scalars + ρ clone                2 scalars + ρ view
diffusion fluxes (phase 3)                    3 full flux tensors per call       posted into register storage
============================================  =================================  ==============================

For ny = 0, ``exact`` goes from 20 face rows per stage to 2. No full-field
copy is made per stage. The scratch is one fixed field (N17).

**Measurement plan M1-M3 (after approval, on the phase-2 and phase-3 PRs):**

* M1: time per step, minimum of 3 runs, pin against V3, for ``off``,
  ``transport`` and ``exact``, with budgets on and off. Decks:
  ``test_axial_momentum_flux_form`` W2 C16/C32 and the 3D arms;
  ``jupiter_gcm_dry`` C48 + Coriolis on one GPU;
  ``test_diffusion_cubed_sphere``.
* M2: peak memory per step on the same runs
  (``torch.cuda.reset_peak_memory_stats``, ``max_memory_allocated``).
* M3: kernel launches per stage (#355 counted 4772 at C45).

No GCM runs (K2-18b or others) are part of this plan.

7.3 Graph boundaries
--------------------

* The end-of-stage fixer point is lane A's only boundary. Lane X reduces the
  7-vector through the host and never captures it (X.rst:98, 187). Rule 2 of
  3.2 would remove the closure's share.
* The pre- and post-implicit phases are segments around lane I's solve.
* ``capturable = false`` sources (``RelaxBotComp``; any user forcing not
  declared capturable, X.rst:184) are eager segments.


8. Cross-repository interface changes and merge order
=====================================================

Lane A asks for **no interface change in any other repository**. The only
cross-repository code lane A touches is kintera's.

* **kintera (pin** ``3f5ff938`` **):** ``RelaxBotComp`` holds
  ``kintera::ThermoY`` and ``ThermoX`` (``forcing.hpp:13-14, 435-436``;
  ``relax_bot_comp.cpp:66-67``). It calls ``compute`` with "Y->X", "DY->V",
  "VT->U" and "X->Y" (``relax_bot_comp.cpp:98-111``).

  * Those branches are at kintera ``src/thermo/thermo_y.cpp:198, 214, 242``
    and ``src/thermo/thermo_x.cpp:141``. The ``compute`` bodies
    (``thermo_y.cpp:180-275``, ``thermo_x.cpp:139-203``) contain no
    ``.item()``. The host reads at ``thermo_y.cpp:385, 402, 480, 525`` and
    ``thermo_x.cpp:301, 343`` are on other paths.
  * Whether these four conversions are capturable is a hypothesis.
    ``RelaxBotComp`` is declared ``capturable = false`` until lane K (chen
    sihe) confirms (K-1).
  * No interface change is requested.
* **pyharp:** none. No lane-A file includes pyharp. ``git grep harp`` on
  ``src/forcing`` and the AAM files at the pin returns nothing.
* **pydisort:** none, for the same reason (``git grep disort`` returns nothing).
* **commux/UCX:** none. The fixer collective is lane X's.

Merge order. No provider-repository change is needed. Inside snapy, the order
(provider first) is:

1. lane O options schema;
2. lane A phase 1;
3. lane R register and header, lane I ``vic.*`` posts, lane X global phase and
   segments, lane S ``RotationGeometry``;
4. lane A phase 2;
5. lane A phase 3 (diffusion into the register);
6. lane A phase 4 (G10).


9. Line budget
==============

9.1 The agreed rows
-------------------

* **X** deletes ``meshblock.cpp:1009-1035`` and ``meshblock.hpp:223-228``,
  booked in X. By ``wc -l`` these are 27 + 6 = 33 lines (31 non-blank). The
  round's figure is 29. Either way they are not in lane A's number.
* **A** drops its 51 mesh lines (2.2; booked in X) and adds
  ``AxialClosureFixer`` in ``src/forcing/axial_momentum.cpp``. That replaces
  ``apply_axial_closure`` and is inside the file's 170-line estimate.
* **A** counts the 18 lines lane I's gravity-work decision frees. Only
  ``forcing.hpp:48, 56-58, 60-62`` (7 lines) is in ``src/``. ``pyforcing.cpp:29``
  and ``forcing.pyi:61-70`` (11 lines) are counted in section 10.2, outside
  the ``src/`` number.
* **O**'s options schema is a precondition (N18).

This replaces the N15 draft statement that A keeps ``meshblock.cpp:1014-1034``.

9.2 Counted lines and the lane number
-------------------------------------

Counted: ``src/forcing`` without ``const_gravity.cpp`` (2,870, section 2.1),
plus lane A's 493 lines outside it (2.2).

=================================  ======  ======  =======================================================
file                               now     after   basis
=================================  ======  ======  =======================================================
forcing.hpp                         674     253    options 326 -> 150 via schema (O); classes 270 -> 70;
                                                   includes and comments 78 -> 40; −7 lane-I gravity
                                                   lines (48, 56-58, 60-62)
source_term.hpp (new)                 0     110    base, registry, budget buffer
coriolis.cpp                        232     120    −28 (123), −38 (glue), −50 (Ω to geometry), +4
axial_momentum.cpp (new)              0     170    consumer, torque rules, ``AxialClosureFixer``
diffusion.cpp                       745     480    −70 options, −110 profile checks (schema), −40, −45
diffusion_cubed_sphere.cpp          376     320    phase 3: −35 averaging, −15 divergence, −6
diffusion_radial.cpp                103     103    unchanged
top_cool + bot_heat                 176      70    ``LayerHeat{side}``
top_sponge + bot_sponge             156      50    ``Sponge{side}``
relax_bot_temp/velo/comp            308     235    ``BottomSlab`` base
body_heat + plume_forcing           100      70    base does ctor, reset, options
hydro_forward.cpp (AAM ranges)      136      25    reads register; two phase hooks
hydro.cpp (AAM ranges)               33      15    mode predicate; ``full`` refusal
coordinate.hpp (AAM hooks)           23       3    one accessor
gnomonic_equiangle.* (AAM)          301     280    240 geometry + 40 spherical-polar K (S-1)
**total**                          3363    2304
=================================  ======  ======  =======================================================

**Lane A: 3,363 → 2,304 lines (−1,059, −31.5 %).**

Arithmetic:

* now: 2870 + 136 + 33 + 23 + 301 = 3363;
* after: 253 + 110 + 120 + 170 + 480 + 320 + 103 + 70 + 50 + 235 + 70 + 25 +
  15 + 3 + 280 = 2304;
* cut: 3363 − 2304 = 1059, and 1059 / 3363 = 0.315.

The ``src/forcing`` part alone goes 2870 → 1981 (−31.0 %).

**Precondition (N18): lane O's options schema lands before lane A's phase 1.**
Lane O is ours. Three rows rest on it: the options row (+131 without it,
after the two merges), diffusion's parser (+70) and diffusion's profile checks
(+110; the change refusal is tested, ``tests/test_diffusion_x1_scale.cpp:736``).
Without the schema, ``src/forcing`` would be 2,292 (−20.1 %) after phase 3 and
2,393 (−16.6 %) before it. The phase-3 dependence is the 101 lines of the two
diffusion rows.

Margin deletions that need no schema, not counted above:

* ``coriolis.cpp:28-30`` and ``forcing.hpp:107, 115``: the ``type`` key, 5
  lines, if Cheng removes type 123;
* ``diffusion.cpp:275-277``: the legacy-key refusal, 3 lines. ``check_keys``
  already rejects unknown keys (``check_keys.cpp:26``).


10. Migration
=============

10.1 Phases
-----------

1. **Phase 1** (G1): the ``SourceTerm`` base and registry, all twelve modules,
   the merges, one Coriolis, the per-source scratch and ``budget_``,
   ``capturable``. Needs lane O's schema.
2. **Phase 2** (G2-G7): the AAM consumer reads the register
   (``riemann.central``†, ``vic.*``); ``AxialClosureFixer`` in lane X's global
   phase; ``exact`` replaces full + closure; the host reads of 7.1 go.
3. **Phase 3** (G8, G9): diffusion posts to the register.
4. **Phase 4** (G10): FP32 and perturbation-flux for ``exact``; the default is
   reconsidered.

10.2 Sites changed by the full → exact decision, by lane
--------------------------------------------------------

From the ``git grep`` of every ``conserve.axial.momentum``,
``axial_coriolis_form``, ``axial_closure``, ``_aam_cl``,
``defer_axial_closure`` and ``"full"`` site at the pin. Each line is counted
once:

* **R:** 47 range lines (``hydro.hpp:53-55, 89, 106-126, 127-133, 224``;
  ``hydro_options.cpp:56, 68-79, 176``);
* **X:** 9 (``meshblock.cpp:983-984``, ``meshblock.hpp:223-225``,
  ``mesh.cpp:367-370``);
* **A:** 152. Of these, 28 are in ``src`` (``hydro.cpp:36, 59-69``;
  ``hydro_forward.cpp:989-995, 1355``; ``coriolis.cpp:176-179, 184, 189, 209``;
  ``gnomonic_equiangle.hpp:117``). The other 124 are in python and tests, plus
  the 11 lane-I python lines of 9.1.

10.3 Deck and API changes
-------------------------

==========================================  ===================================  ==========================
change                                      replacement                          status
==========================================  ===================================  ==========================
``conserve-axial-momentum: full``           refused with a message naming        **settled** (Cheng)
                                            ``exact``
``conserve-axial-momentum-closure``         removed; ``exact`` includes the      **settled** (Cheng)
                                            closure; refused by ``check_keys``
``coriolis/type: 123``                      ``type: xyz``: identical in 3D;      **FOR CHENG**
                                            with one x3 cell ``xyz`` adds the
                                            IVZ row ``123`` skips (N13)
AAM key on a grid with no rotation axis     refused (today a no-op + warning,    **FOR CHENG**
                                            ``hydro.cpp:70-76``)
radial-conduction positivity failure        device flag; abort kept (default)    **FOR CHENG** (redo needs
                                            or redo with a new cause bit         cause bit 64 from X)
==========================================  ===================================  ==========================

10.4 Decision log
-----------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c is ``riemann.central``†, one summed row           no second flux evaluation (3.1)
D2  modes ``off | transport | exact``; full refused       settled by Cheng (3.2, 10.3)
D3  default ``off``; AAM budget always reported           bitwise migration (3.2)
D4  closure = registered fixer; {R, I} on device;         requirements 6, 7; N5, N14 (3.3, 7.3)
    reduced via host in X's global phase (7-vector)
D5  one Coriolis ``-2Ω × m``; m1 from A-r3 + vic.*        requirement 4 (3.4)
D6  FP32 lifted only by G10                               6.2
D7  per-source FP64 scratch + ``budget_``; FP32 uses      bitwise and G7 (5.1; N6, N17)
    ``_forcing_work``
D8  diffusion into the register in phase 3                bitwise gates of phases 1-2 (3.7)
D9  ``exact`` with ``nu_iso > 0`` refused until phase 3   closure would remove a physical torque
D10 AAM geometry in coordinate (S-1); physics in lane A   rotation owned by coord (3.8)
D11 user forcings stay in X's segment                     order and G1 (3.6; N16)
D12 ``capturable`` flag, ordered runs                     X's segment flag (3.6; N14)
D13 lane A writes nothing to ``Variables``                requirement 10 (4.1)
D14 lane O schema before lane A phase 1                   20 % target (9.2; N18)
=== ====================================================  ==========================================


11. Open questions to other lanes
=================================

Closed since rev 1:

* **I-1:** answered by Xinyue. Lane I posts ``vic.mass_total``,
  ``vic.mass_dry``, ``vic.mass_species`` and ``vic.momentum_t[2]``, under
  scheme 9 too.
* **C-1:** closed with I-1.
* **X-1/X-2 (capture, collective):** answered by Xi (N5, N14). The collective
  is never captured, and it carries a 7-vector.

Open:

* **R-1, Tianhao (lane R backup owner; on his list).** ``riemann.central``†
  as one summed row from post-terrain states, valid to stage end. Are face
  densities totals or anomalies under perturbation-flux? Are parts kept as
  fields or as integrals?
* **R-2, Tianhao (on his list).** The split radial entry with weight 1/2 in
  the step transfer. Also the name of the fixer base class (``Fixer``†).
* **S-1, S-2, S-3, Xinyue (lane S).** Ownership of ``RotationGeometry``; the
  frame of registered x2/x3 fluxes; whether the rotation function takes
  Cartesian-component traction; whether the seam exchange can carry 7
  penalty-trace channels.
* **X-1, Xi.** Who owns the fixer registry (X or the register owner)? Can the
  radial flag ride the redo slot (``meshblock.cpp:1053-1056``), and does lane
  X take cause bit 64 if Cheng picks redo?
* **K-1, chen sihe (lane K).** Does lane K migrate ``ConstGravity`` (its work
  and the gravity-work fixer, requirements 4 and 7)? Are kintera's "Y->X",
  "DY->V", "VT->U" and "X->Y" conversions capturable (section 8)?

FOR CHENG: see 10.3.


12. Skeleton index
==================

``wc -l docs/v3/skeleton/A/*`` gives 299 lines, with declarations only.

==========================  ==============================================================  ===============
file (lines)                declarations                                                    serves
==========================  ==============================================================  ===============
``source_term.hpp`` (98)    ``SourcePhase``, ``SourceKind``, ``SourceRows``,                3.6, 5.1, 7.1
                            ``SourceRegion``, ``SourceDeclaration`` (with
                            ``capturable``), ``StageState``, ``SourceTerm``,
                            ``SourceBudget``, ``SourceRegistry`` (``Run``, ``runs``,
                            ``run``, ``scratch_``, ``budget_``); ``FluxRegister``†
``axial_momentum.hpp``      ``AxialMode``, ``AxialFluxView`` (``riemann.central``†,          3.1-3.3, 5.2,
(80)                        ``vic.*``), ``FaceTorqueRule``, ``AxialMomentum``,              6.2, 7.3
                            ``AxialClosureFixer``; ``Fixer``†
``coriolis_diffusion.hpp``  ``Coriolis``, ``DiffusionCoefficients``, ``DiffusionFlux``,      3.4, 3.7, 3.6
(86)                        ``DiffusionSource``, ``RadialConduction`` (``failed_``
                            flag), ``Sponge``, ``LayerHeat``, ``BottomSlab``;
                            ``FluxContributor``†
``rotation_geometry.hpp``   ``RotationGeometry`` (k fields, ``rate_to_rows``,               3.8, 3.4
(35)                        ``omega_local``, m1 weights, ``has_on_axis_cell``)
==========================  ==============================================================  ===============


Appendix: responses to round-2 notes N1-N18 (Xi)
================================================

N14 withdraws N2 and supersedes N5 where they overlap (dt on the host; the
collective is never captured). Every cited site was re-read at the pin.

* **N1 (accept).** The host copies at ``coriolis.cpp:204-207``,
  ``meshblock.cpp:1011``, ``mesh.cpp:371`` and ``meshblock.cpp:1061`` are
  added to the 7.1 table. {R, I} stay on the device until lane X's global
  phase.
* **N2 (withdrawn by N14).** ``dt`` stays a host ``double`` (7.1). No skeleton
  change.
* **N3 (accept).** The radial check becomes a device flag on the redo slot,
  and the action stays abort. Redo with cause bit 64 is FOR CHENG plus lane X
  (10.3, 11 X-1). Skeleton: ``RadialConduction::failed_``.
* **N4 (accept).** G2 uses the normalised e_∞ ≤ 1e-13 plus an L2 analogue
  (5.3). The eps·ρ·c scale is hypothesis H3.
* **N5 (accept, superseded in part by N14).** The fixed 7-vector and its
  placement stand (3.3). Capture is replaced by N14: the collective goes
  through the host and is never captured. Registry ownership stays open
  (X-1).
* **N6 (accept).** ``budget_[nsrc, 6]`` is an FP64 device buffer, and
  ``SourceRegistry::run`` takes ``bool budgets`` instead of a host vector.
  Budgets on and off are two topologies (5.1).
* **N7 (accept).** The skeleton's section references are remapped to this
  revision's numbering, and the stale question ids are replaced. The m1
  comment now says that φ is a rate and that ``lo``/``hi`` hold 1/(2rV), with
  no ``/ dt``.
* **N8 (accept).** The no-schema basis is stated (+131 options, +70 parser),
  plus the profile checks (+110) found under N18 (9.2).
* **N9 (accept).** ``riemann.central``† comes from post-terrain states and
  the solver's ``u_n``. F_d contains the covariance and the limiter scaling.
  The FP32 caveat is H4 (3.1, 6.1).
* **N10, N11, N12 (agree).** No change. The §9 numbers changed anyway with
  the booking (9.1).
* **N13 (accept).** The type-123 row reads "identical in 3D". With one x3 cell,
  ``xyz`` adds the IVZ row (2.3, 10.3). It stays FOR CHENG.
* **N14 (accept).** "No host sync" is removed from the skeleton; H2 is
  reworded; ``capturable`` and ordered runs are added (3.3, 3.6, 7.1, 7.3).
* **N15 (accept, replaced by the later agreement).** X deletes
  ``meshblock.cpp:1009-1035`` and ``meshblock.hpp:223-228``. A drops its 51
  mesh lines and adds ``AxialClosureFixer`` (9.1). A no longer keeps
  ``meshblock.cpp:1014-1034``.
* **N16 (accept).** User forcings stay in X's compute segment. X counts
  ``meshblock.cpp:819-884`` (66 lines) and registers the species clip (3.6,
  5.2).
* **N17 (accept with change).** A per-source FP64 scratch keeps G1 and G4
  bitwise. A single stage buffer would not, because (du + s1) + s2 ≠ du +
  (s1 + s2). The memory is 8·(5+ny) B per cell in FP64 runs and 0 in FP32 runs
  (5.1, 7.2).
* **N18 (accept, with corrected basis).** 45 of the 176 option lines go in the
  merges without a schema, but the 110 profile-check lines also need it. The
  schema is a precondition, and the no-schema numbers are stated (9.2).
