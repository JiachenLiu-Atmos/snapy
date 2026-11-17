=======================================================================
V3 design, lane A: rotation, axial angular momentum, sources, diffusion
=======================================================================

:Lane: A (rotation, axial angular momentum (AAM), source terms, diffusion)
:Owner: Jiachen Liu (alternate: Xi Zhang)
:Revision: 1 (due Wed 11-18 18:00 EST), checked against the ten flux-register
           requirements of brief 2/3; revision 2 due Fri 11-20 18:00
:Reviewers (round 2): Jiheng (lane S), Xi (lane X)
:Pin: snapy ``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410`` (``9e5de2d FP32 step5
      validation (#368)``)
:Skeleton: ``docs/v3/skeleton/A/``, four headers, declarations only, under
           300 lines (``wc -l docs/v3/skeleton/A/*``)

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

Where things are:

* Section 3: the eight design questions.
* Sections 4 to 7: the round's register contract, invariants and gates, FP32,
  and CUDA graph topics. These are checked against the ten flux-register
  requirements of brief 2/3, with a compliance table in section 4.2.
* Section 8: the two decide-and-justify items.
* Section 9: the line-count plan.
* Section 10: the migration order and decision log.
* Section 11: the open cross-lane questions.
* Section 12: the hypotheses and how they will be measured.

The ten requirements, as quoted from the kickoff (brief 2/3) in Cheng's
Claude's reply in the lane A thread:

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

Read for this revision:

* ``src/forcing/*`` (all 15 files), ``src/coord/gnomonic_equiangle.{hpp,cpp}``
  (AAM parts), ``src/coord/coordinate.hpp:370-392``,
  ``src/hydro/hydro_forward.cpp:988-1009, 1088-1145, 1288-1411, 1624-1696``,
  ``src/hydro/hydro.hpp:89-133, 219-236``, ``src/hydro/hydro.cpp:27-76``,
  ``src/hydro/register_forcing_modules.cpp``,
  ``src/mesh/meshblock.cpp:850-905, 983-1034``, ``src/mesh/mesh.cpp:363-382``,
  ``src/output/restart.cpp:26-67``, ``src/riemann/hllc_impl.h:100-163``,
  ``V3_ROADMAP.md:258-279`` (G5).
* PR bodies, via ``gh pr view``:

  * #355: merged, ``46710ac570668a699cb2fc5ad17a580cd4156caa``.
  * #357: merged, ``2f5f51b2466704ef1390d087b74926b546627c31``.
  * #363: merged, ``dcf5a5a12e644b67232adb718bb6166b2ef0c0ae``.
  * #365: merged, ``3fe398a917646851ed42db3f964a3fc2751de2fe``.

  ``gh`` returned zero comments and zero reviews for all four.
* #314 is an issue, not a PR (``gh pr view 314`` answered "Could not resolve
  to a PullRequest"). I read it with ``gh issue view 314``.

Not read, and therefore not used:

* The Slack threads named in the brief: the #357 review thread
  1794565311.770069 and the side study 1794706777.579419. Workers cannot read
  Slack. Everything said here about the side study's local-torque scheme comes
  from the one-line description in the lane brief, and is marked as such.
* The rest of the kickoff brief beyond the ten requirements above.
* No code was built or run for this revision. Every V3 cost or behaviour claim
  without a PR measurement is a **hypothesis**, numbered H1 to H7 in section
  12.

Numbering change from the first local draft (``e82a1a13``): the eight
design-question sections are now section 3. The register, invariants, FP32 and
graph material moved into sections 4 to 7, the decide-and-justify items into
section 8, and the cross-lane questions into section 11. The line-count numbers
are unchanged.


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
and ``class .*Impl`` ... ``TORCH_MODULE``.

2.2 Lane-A code outside ``src/forcing`` (not in the baseline)
-------------------------------------------------------------

Counted with ``sed -n 'a,bp' FILE | wc -l`` for each range:

================================================  =======  =======================================
range                                             lines    what
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

2.3 What the code does today
----------------------------

* **Forcings are a flat list of** ``AnyModule`` **objects** built in
  ``register_forcing_modules.cpp:6-86``. They are called in order on a shared
  ``du`` (``hydro_forward.cpp:1378-1385``), with the signature
  ``forward(du, w, temp, dt)``. Nothing records what each one added. Issue #314
  had to rebuild the sponge torque offline.
* **Coriolis has two classes.** ``Coriolis123`` (``coriolis.cpp:40-67``)
  applies the same Ω in every coordinate system. ``CoriolisXYZ``
  (``coriolis.cpp:69-231``) rotates Ω into the local basis, for Cartesian,
  cylindrical, spherical-polar and gnomonic grids. On Cartesian grids the two
  agree (``coriolis.cpp:91-94`` against ``:53-63``), except for the 2D guard at
  ``:61``. No deck or test sets ``type: 123``.
* **#355 (x1 mass-flux w)** builds m1 from the x1 face mass transfers. It does
  this by calling the Coriolis forward twice per stage on hand-built
  "primitive" tensors:

  * ``(ρ, 0, v2, v3)`` before the implicit solve
    (``hydro_forward.cpp:1351-1352, 1384``);
  * ``(1, m1, 0, 0)`` after it (``:1650-1655``).

  The x1 transfer is recomputed from ``_flux1`` times the face area there
  (``:1637-1646``). The metric term ``-m1 v/r`` lives in
  ``x1_mass_source_`` (``gnomonic_equiangle.cpp:664-674``).
* **transport (#357)** rewrites the K component of the finished divergence
  from the same face fluxes (``hydro_forward.cpp:1290-1298``,
  ``gnomonic_equiangle.cpp:551-604``).
* **full (#363)** rebuilds a central flux with two extra Riemann calls per
  x2/x3 direction (``hydro_forward.cpp:997-1003, 1108, 1141``). It stores
  ``F - F_c`` (``:1354-1363``), and ``CoriolisXYZ`` adds the torque
  (``coriolis.cpp:176-213``). It refuses FP32 and perturbation-flux
  (``hydro_forward.cpp:993-995``). Measured cost: 1.14-1.48 times ``off``.
* **closure (#365)** has ``CoriolisXYZ`` compute R and I
  (``coriolis.cpp:189-209``). There is one allreduce per stage and two
  ``.item<double>()`` host reads (``meshblock.cpp:983-990, 1014-1034``;
  ``mesh.cpp:367-382``).
* **Diffusion keeps its own fluxes, seam rule and divergence.**

  * Cartesian: ``du -= dt * divergence(...)`` (``diffusion.cpp:671-673``).
  * Cubed sphere: Cartesian-component traction, its own seam flux exchange
    (``share_fluxes``, ``diffusion_cubed_sphere.cpp:103-176``), a divergence,
    then a Cartesian-to-covariant projection (``:368-373``).
  * A Strang-split radial conduction (``diffusion_radial.cpp``).

  Viscous heating is already in flux form (``diffusion.cpp:625-627``,
  ``diffusion_cubed_sphere.cpp:345``).
* **Floors.** User stage forcings have a species clip that withholds part of
  the increment (``meshblock.cpp:860-884``). ``relax_bot_comp.cpp:104`` clamps
  the *target* dry fraction. The increment it then applies conserves total
  mass, since ``delta[IDN]`` is minus the species sum (``:113-117``).
* **Restart writes every defined entry of the block's** ``Variables``
  (``src/output/restart.cpp:31-37``).
* **Decks.** No deck in the repository sets ``conserve-axial-momentum`` or its
  closure key; only ``tests/test_axial_momentum_flux_form.{py,yaml}`` does.


3. Design answers (Q1 to Q8)
============================

3.1 Q1: AAM as a consumer of the register
-----------------------------------------

AAM reads fluxes and never evaluates them (requirement 4). What it reads:

=====  ===========================================  ============  =====================================
id     quantity                                     posted by     used by
=====  ===========================================  ============  =====================================
A-r1   total reconciled face flux, x1/x2/x3         register      transport: K-component in flux form
A-r2   part ``riemann.central`` of the x2/x3        R             exact: torque of F_d = F - F_c;
       increment: mass rows 1/2 (ρ_L u_L +                        closure residual R
       ρ_R u_R), dry + species
A-r3   explicit x1 mass transfer (area-weighted)    register      m1 (#355)
A-r4   implicit x1 mass transfer per unit time      I             m1 (``implicit_mass_correction``)
A-r5   implicit tangential momentum correction      I             ``axial_x1_source_`` (face radius)
=====  ===========================================  ============  =====================================

A-r2 is a named part of the Riemann increment (requirement 2), not a second
flux evaluation. The rest of the Riemann flux is the ``riemann.dissipative``
part, and the two sum to the posted Riemann flux.

* For a consistent solver, ``F(w, w)`` is the physical flux. So the mass rows of
  today's ``1/2 (F(w_L, w_L) + F(w_R, w_R))`` are
  ``1/2 Σ_s (ρ_s u_n)_L + (ρ_s u_n)_R``. This is one tensor expression on the
  face states the solver already holds.
* HLLC builds ``ρ u`` per side only in a moving frame (``hllc_impl.h:104-120``),
  so the part is formed from the face states.
* **H1:** this equals today's construction to round-off in FP64. Gate: 1e-14
  relative on the #363 deck.

F_d is the total registered mass flux minus the central part, whoever posted
the rest. Today the #289 covariance term (``hydro_forward.cpp:1110``) already
counts as dissipative. Phase 3 adds diffusion, but diffusion carries no mass
flux, so F_d is unchanged by it.

3.2 Q2: which modes survive
---------------------------

Evidence (from the #357, #363 and #365 bodies and ``hydro.hpp:89-133``):

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

**Decision: three modes, off | transport | exact.**

* ``exact`` is today's ``full`` plus the closure, as one mode.
* ``full`` alone and the separate ``conserve-axial-momentum-closure`` key are
  removed. They are refused with a message naming ``exact``.
* ``full`` alone is neither cheap (1.14-1.48x) nor exact (4e-10 to 1e-7). The
  closure costs 4-9% more and makes it exact.
* ``full`` without the closure survives only as a C++ test flag. That keeps
  arm (l)'s "closure off must fail" check.

**The side study's local-torque scheme.** The brief describes it as "the torque
of ``F_d = F - 0.5 (F_L + F_R)`` and the local zonal torque on x2/x3 faces". I
have not read it. The first half is what ``axial_coriolis_rate`` already
computes (``gnomonic_equiangle.cpp:676-719``).

So the torque is a pluggable rule (``FaceTorqueRule``), and the #363 rate is
rule 1. If the side-study rule passes the gates below without the closure,
``exact`` switches to it and drops the fixer and its collective. The gates:

* arm (l) drift <= 1e-12 with no reduction;
* arms (f)-(h) and (k) at least as good as rule 1;
* noisy-jet KE ratio no worse than ``full``'s 1.005-1.012 at C32;
* cost no higher than rule 1.

3.3 Q3: the closure as a registered fixer
-----------------------------------------

``AxialClosureFixer`` is a registered fixer (requirement 7; section 5.2 gives
what it declares). It sits at the stage's last momentum change, after the
saturation adjustment and before the boundaries, as today
(``meshblock.cpp:983-990``).

R is ``Σ V (K · F_cell - rate(riemann.central))``. It is computed once, from
A-r2. Today it calls ``axial_coriolis_rate`` twice and subtracts
(``coriolis.cpp:199-203``).

The reduction is device-side and shares one collective with the other fixers
due that stage (section 7.3).

3.4 Q4: one Coriolis and metric source (#355)
---------------------------------------------

``force = -2 Ω × m``, in the local orthonormal frame, lowered to covariant
rows. Here:

* ``m = (m1, ρ v2, ρ v3)``;
* ``m1 = [Φ_lo (r² - r_lo²) + Φ_hi (r_hi² - r²)] / (2 r V)``;
* Φ is read from the register: A-r3 plus A-r4. It is no longer recomputed from
  ``_flux1`` times the area (``hydro_forward.cpp:1637-1646``).

The metric pair ``-m1 v_h / r`` is part of the same source.

There are two phases:

* ``pre_implicit``: the w-free rows (m1 = 0). They must pass through the
  scheme-9 solve; #355 measured 1.27 m/s of w error otherwise.
* ``post_implicit``: the m1 rows only.

Ω in the local basis comes from the geometry (section 3.8). The grid gate is
#355's, unchanged:

* x1 transfers on cubed-sphere grids with nc1 > 1 (not shallow water), and on
  spherical-polar grids with Coriolis;
* cell ρw elsewhere, including Cartesian f-plane and cylindrical grids.

``traditional`` zeros the horizontal Ω components.

3.5 Q5: the FP32 restriction on full
------------------------------------

See section 6.

3.6 Q6: one source-term interface for the twelve modules
--------------------------------------------------------

Every module is a ``SourceTerm`` (skeleton ``source_term.hpp``). It declares:

* ``phase``: ``pre_implicit``, ``post_implicit`` or ``split``;
* ``rows``: a bit set over mass, species, momentum and energy;
* ``kind``: ``physical``, ``numerical`` or ``fixer``;
* ``region``: the interior, or a bottom or top slab.

The base class does the physical-boundary test that six modules repeat today.
A source adds its increment to ``du`` through ``add(du, StageState)``.

The monitor attributes budgets per source without a scratch copy (requirement
9; section 7.2). The sponge torque of the K2-18b runs becomes an in-model line,
which issue #314 had to estimate offline.

The twelve modules in V3:

=======================  ===================================  =====================================
today                    V3                                   change
=======================  ===================================  =====================================
ConstGravity             SourceTerm (interface only)          migration owner asked in section 11
Coriolis123/CoriolisXYZ  ``Coriolis`` (one class)             section 3.4; type 123 removed
Diffusion                ``DiffusionFlux`` + split radial     section 3.7
BodyHeat                 ``BodyHeat``                         base class does masks and options
TopCool / BotHeat        ``LayerHeat{side}``                  two 88-line files -> one
TopSpongeLyr /           ``Sponge{side}``                     two 78-line files -> one
BotSpongeLyr
RelaxBotTemp / Velo /    three classes on a shared            slab selection, boundary test and
Comp                     ``BottomSlab`` base                  covariant lowering in the base
PlumeForcing             ``PlumeForcing``                     unchanged arithmetic
=======================  ===================================  =====================================

Deck keys stay. User stage forcings (``meshblock.cpp:871-877``) are run as
sources by the same registry, and their species clip is a registered floor
(section 5.2).

3.7 Q7: diffusion as a register contributor
-------------------------------------------

From phase 3 (section 8.2), ``DiffusionFlux`` posts the increment
``diffusion`` per face direction:

* the viscous momentum flux ``-μ τ · n``;
* the energy flux ``-k ∇T · n - v_f · τ · n``. The second term is the viscous
  heating, already in flux form today.

It no longer calls ``divergence`` and no longer reconciles its own seams.

* The cubed-sphere traction, computed in Cartesian components
  (``diffusion_cubed_sphere.cpp:297, 344``), is rotated into the register's
  face frame by coord's one rotation function (requirement 5).
* The interior-penalty term needs the neighbour's traces. It is symmetric in
  the two sides (it uses the jump and averaged coefficients,
  ``:164-173``), so both entries of a shared face carry the same value, and the
  register's rule leaves it unchanged.
* The radial conduction stays an operator-split source.
* The time-step bound (``diffusion.cpp:677-743``) is evaluated by the driver,
  outside any graph segment.
* The per-cell profile tensors stay, because tests use them
  (``tests/test_diffusion_cubed_sphere.cpp:463, 598-599``). Their 128-line
  change check becomes the options schema's "frozen after build" (lane O,
  ours).

Expected effect, as **H5**: the viscous torque then telescopes under
``transport`` and ``exact``.

3.8 Q8: where the AAM geometry lives
------------------------------------

The geometry goes in the coordinate, as a ``RotationGeometry`` part owned by
lane S. It holds:

* ``k`` at centres and its exact face averages, ``k·k``, and ``cos² lat`` at
  faces;
* ``rate_to_rows``;
* ``omega_local``;
* the m1 split weights;
* a reset-time ``has_on_axis_cell`` flag, which replaces the per-call host
  read at ``gnomonic_equiangle.cpp:506``.

It covers gnomonic and spherical-polar grids. A Cartesian grid has none. The
four AAM hooks on ``Coordinate`` (``coordinate.hpp:370-392``) become one
accessor.

The AAM physics stays in lane A, in ``src/forcing/axial_momentum.cpp``: the
consumer, the torque rules and the closure. ``axial_coriolis_rate`` moves out
of the coordinate. Lane S owns metric exactness, and the face averages are
metric identities: the lateral pressure fluxes sum to zero
(``gnomonic_equiangle.cpp:181-182``). The torque rule is the part expected to
change.


4. Flux-register contract
=========================

4.1 Lane A against requirements 1-5 and 10
-------------------------------------------

**Requirement 1 (one owner, one frame, one area weighting).**

* *Posts:* in phase 3, ``DiffusionFlux`` is the only owner of the diffusive
  momentum and energy face fluxes. They are posted per unit face area, in the
  register's documented face frame (the frame of today's ``flux2global2_``/
  ``3_``, assumed in S-2 of section 11). The area weighting is the register's
  divergence, not a lane-A one.
* *Owns no other flux:* the AAM face flux ``A F · K_f`` is a derived quantity
  computed from A-r1, never posted. The central mass rows are a part of R's
  flux, owned by R.
* Phases 1 and 2 have a declared exception: diffusion still forms its own
  divergence (section 8.2).

**Requirement 2 (named increments, declared order, parts reportable).**

* Lane A's names: ``diffusion`` (phase 3), and the AAM consumer's derived,
  unposted quantity ``aam.flux_form``.
* Lane A reads ``riemann.central`` (A-r2).
* *Order:* ``riemann`` (with parts ``central`` and ``dissipative``), then
  ``covariance`` (#289), then ``diffusion``, then any face fixer. ``diffusion``
  carries no mass rows, so the AAM torque is independent of where it sits.
* *Needs from R:* ``riemann.central`` as a named part, kept until the
  post-implicit phase of the same stage (Linfeng, R-1).

**Requirement 3 (stage weights into the step transfer).**

* *Reads:* the monitor's per-source budgets and the boundary AAM flux are
  weighted like the register's step transfer. A stage weighs
  ``w2_s Π_{t>s} w1_t``, the weight today's gravity-work fixer uses
  (``hydro_forward.cpp:1683-1691``).
* *Corrects:* the closure applies its increment inside the stage with
  ``w2_s``, as #365 does (``meshblock.cpp:1027``). It declares its step AAM
  change with the same step weight.
* *Needs:* the radial conduction is operator-split, not a stage flux. Its two
  half-step wall heat fluxes must enter the step transfer as an entry with an
  explicit weight of 1/2 each (Linfeng, R-2).

**Requirement 4 (every flux-derived term reads the register; no rebuild).**

* *AAM torque:* ``transport`` reads A-r1. ``exact`` reads A-r1 and A-r2. Today's
  two extra Riemann calls (``hydro_forward.cpp:997-1003, 1108, 1141``) and the
  stored ``_flux - cen`` (``:1354-1363``) are removed.
* *Coriolis and metric* ``rho w``: m1 reads Φ from A-r3 (explicit) and A-r4
  (implicit, posted by lane I). Today's ``_flux1`` times area rebuild
  (``:1637-1646``) is removed.
* *Closure:* R reads A-r1 and A-r2. It does not call the rate twice.
* *Source terms:* none of the twelve reads or builds a face flux. Sponge,
  relaxation, heating and plume are cell-local. Coriolis takes only m1.
  Gravity work belongs to the gravity migration owner (section 11, Isaac).
* *Diffusion:* it computes its own flux; it does not rebuild anyone else's. It
  is compliant from phase 3, when its divergence goes to the register.

**Requirement 5 (two entries, one reconciliation rule, rotation in coord).**

* *Reads:* ``transport`` telescopes only if each shared face carries one value.
  AAM reads the reconciled flux and never averages it itself.
* *Rotation:* at a panel seam, the face average ``K_f`` must be expressed in
  the same frame as the reconciled flux. So ``K_f`` is built with coord's one
  rotation function, not lane-A code.
* *Gate:* ``A F · K_f`` is equal from both sides of every seam face to 1e-14
  relative. Today's arm (a) exactness (1.3e-15) is the bound.
* *Diffusion (phase 3):* the averaging half of ``share_fluxes``
  (``diffusion_cubed_sphere.cpp:159-163``) is dropped for the register's rule.
  The penalty term stays and is symmetric (section 3.7).
* *Needs from S:* whether coord's rotation takes Cartesian-component vectors,
  and whether the register's seam exchange can carry diffusion's penalty
  traces (Jiheng, S-3).

**Requirement 10 (rebuilt, not restart-written).**

Restart writes every defined entry of the block's ``Variables``
(``restart.cpp:31-37``). So lane A adds nothing to ``Variables``.

* The AAM consumer holds no state across stages: ``{R, I}`` and the density
  are stage-local.
* Coriolis buffers (``omega_local``, the m1 weights) and the geometry are
  rebuilt at reset.
* Diffusion's split state (``split_initial_``) is step-local.
* The monitor's running budgets restart from the state read at restart.

Today's diffusion seam exchange uses a local ``Variables``
(``diffusion_cubed_sphere.cpp:129``), not the block's. It is dropped in phase
3 anyway.

4.2 Compliance table (all ten requirements)
-------------------------------------------

====  ========  ==================================================================
req   section   status for lane A
====  ========  ==================================================================
1     4.1       complies; diffusion owner from phase 3; frame per S-2
2     4.1       complies; needs ``riemann.central`` as a named part (R-1)
3     4.1       complies; split radial wall flux needs a weighted entry (R-2)
4     4.1       complies from phase 2 (AAM, closure, m1); diffusion from phase 3
5     4.1       complies from phase 3; seam gate on ``A F · K_f``; needs S-3
6     5.1       complies after the host reads in 7.1 are removed; needs X-3
7     5.2       complies: closure and the user-forcing species clip register
8     6, 7.1    complies: dtypes in 6.1; fixed buffers in 7.1
9     7.2       stated, with the measurement plan; numbers measured in phases 2-3
10    4.1       complies: nothing in ``Variables``; everything rebuilt at reset
====  ========  ==================================================================


5. Invariants and gates
=======================

5.1 Monitor (requirement 6)
---------------------------

Lane A reports to the one device-side monitor. It runs no monitor of its own.

* *AAM:* ``L = Σ V r K · m`` (covariant rows, contravariant K, as
  ``gnomonic_equiangle.cpp:520``). It is always reported on grids that have a
  ``RotationGeometry``, whatever the mode.
* *Per-source attribution:* for each source, the monitor reduces ``du`` after
  the source runs. The source's share is the difference from the previous
  reduction. Mass, species, momentum, E and AAM are all linear in the
  increment, so this needs no scratch copy (section 7.2).
* *Fixers:* booked apart from physical sources (5.2).
* *Host reads:* none inside a segment. Section 7.1 lists the ones lane A
  removes.

5.2 Fixers and floors (requirement 7)
-------------------------------------

=============================  =======  =============  ===================================================
fixer / floor                  kind     rows           declares
=============================  =======  =============  ===================================================
AAM closure (``exact``)        fixer    IVY, IVZ       AAM change ``-w2 dt R`` (checked against the
                                                       measured change, 1e-12); KE moved to IE; reduction
                                                       {R, I}, FP64, every stage
user-forcing species clip      floor    ICY.., IDN     species mass withheld per stage (``du`` before
(``meshblock.cpp:879-884``)                            minus after the clip), and its dry-mass carry
side-study torque (rule 2)     numer.   IVY, IVZ       its AAM rate; no reduction
=============================  =======  =============  ===================================================

Not fixers, with the reason:

* ``relax_bot_comp.cpp:104`` clamps the target composition, and the increment
  it applies conserves total mass.
* The wall coefficient fallbacks (``diffusion.cpp:160``,
  ``diffusion_cubed_sphere.cpp:290``) shape a flux coefficient. A flux cannot
  break conservation.
* Gravity's work fixer belongs to the gravity migration owner (section 11).

5.3 Gates
---------

====  =============================================================  ===================  ===========
id    gate                                                           bound                phase
====  =============================================================  ===================  ===========
G1    all existing tests bitwise against the pin                     0 ULP                1
G2    ``riemann.central`` against today's central construction       1e-14 rel (H1)       2
G3    arm (l): ``exact`` AAM drift                                   <= 1e-12 rel         2
G4    off and transport bitwise against the pin                      0 ULP                2
G5a   seam single-valuedness of ``A F · K_f``                        1e-14 rel            2
G6    closure: declared vs measured AAM change                       1e-12 rel            2
G7    sponge / drag torque booked = AAM change (as #365's drag arm)  1e-15 of L           2
G8    ``test_diffusion_cubed_sphere`` and convection-onset notes     their tolerances     3
G9    ``exact`` with viscosity: viscous torque booked as physical    1e-12 rel            3
G10   FP32/perturbation-flux arms (section 6.2)                      drift <= 1e-12 rel   4
====  =============================================================  ===================  ===========


6. FP32
=======

6.1 Stated dtypes (requirement 8)
---------------------------------

==========================================  ================  =========================================
quantity                                    FP64 run          FP32 run (``precision: float32``)
==========================================  ================  =========================================
source increments                           FP64, into du     FP64 accumulator, rounded once into du
                                                              (today's ``_forcing_work``,
                                                              ``hydro_forward.cpp:1365-1375``)
``riemann.central`` (mass rows)             FP64              flux dtype (FP32); ``F - F_c`` and the
                                                              rate formed in FP64 (H4)
AAM flux form, torque rate, R and I         FP64              FP64
monitor reductions                          FP64              FP64
``diffusion`` part                          FP64              flux dtype (FP32), as the other parts
RotationGeometry buffers, m1 weights        FP64              FP64 master, compute-dtype view (lane S)
==========================================  ================  =========================================

6.2 What lifts the restriction on full (now exact), Q5
------------------------------------------------------

Today's refusal (``hydro_forward.cpp:993-995``) exists because "its central
flux does not see the perturbation states" (#363). G5 asks for a validated
formulation (``V3_ROADMAP.md:268-270``). Three things lift it together:

1. **A consistent F_c.** ``riemann.central`` is formed from the states the
   solver actually used. The mass rows carry no pressure, so the
   reference-pressure gauge (``hllc_impl.h:96, 155``) does not enter. Whether
   face densities are totals under perturbation-flux is asked in R-1.
2. **FP64 accumulation** of the rate and of R and I (6.1). **H4:** ``F_d`` is
   small at smooth faces and loses relative precision in FP32. The torque
   multiplies it by ``s_f² - s_c² = O(h)``, so the effect should be small.
   This needs measuring.
3. **Gate G10.** Rerun arms (f)-(h), (k) and (l) under FP32 with
   perturbation-flux, against FP64. ``exact`` must hold drift <= 1e-12. The
   panel ghosts with ``interp_order: 4`` under anomaly storage are lane S's G5
   work.

Until G10 passes, ``exact`` refuses FP32 and perturbation-flux at reset.


7. CUDA graph
=============

7.1 Capturable storage and no host reads (requirements 6 and 8)
---------------------------------------------------------------

Fixed storage:

* every lane-A buffer is allocated at reset with fixed shapes: ``omega_local``,
  the m1 weights and Φ (today's ``_mass_w`` cache), and the 2-vector
  ``{R, I}``;
* ``riemann.central`` lives in the register's storage;
* there is no shape- or data-dependent branch inside a stage.

Host reads and host branches lane A removes or moves out of segments:

==================================================  ===============================================
today                                               V3
==================================================  ===============================================
``meshblock.cpp:1018`` closure ``.item()`` (x2)     device-side scale with ``where(I > 0)``
``gnomonic_equiangle.cpp:506`` on-axis ``.item()``  ``has_on_axis_cell`` computed at reset
``diffusion.cpp:507`` ``torch::equal`` per call     frozen options, checked at build only
``diffusion_radial.cpp:97-100`` positivity check    device failure flag to the monitor (X-3)
``diffusion.cpp:677-743`` ``max_time_step``         stays, in the driver's dt step, outside
                                                    segments
==================================================  ===============================================

The radial conduction's column loop (``diffusion_radial.cpp:80-93``) has a
fixed trip count n, so it captures as a fixed launch sequence.

7.2 Memory and per-step cost (requirement 9)
--------------------------------------------

Counted in face-field rows of one direction (one row = one value per face).
``ny`` is the number of species.

============================================  =================================  ============================
item                                          today (pin)                        V3
============================================  =================================  ============================
``exact`` central flux, per stage             ``cen2``, ``cen3`` (2 (5+ny)),     ``riemann.central``: 2 (1+ny)
                                              two solver outputs per call        rows, inside the register
                                              (transient 4 (5+ny)), stored
                                              ``F - F_c`` (2 (5+ny))
diffusion fluxes (phase 3)                    3 full flux tensors per call       posted into register storage;
                                              (``diffusion.cpp:580``)            part reported as reductions
per-source budgets                            none                               none: reductions of ``du``,
                                                                                 6 FP64 per source
closure                                       2 scalars + ρ clone (1 field)      2 scalars + ρ view
============================================  =================================  ============================

For ny = 0, ``exact`` drops from 20 face-field rows kept in a stage (8 of them
transient) to 2. V3 adds no full-field copy per stage.

The diffusion part stays reportable as integrals, not as a stored field. If
the register must keep each part as a field, diffusion adds 3 (5+ny) rows.
That is asked in R-1.

Cost (expected, **H2**): ``exact`` drops from 1.14-1.48 times ``off`` (#363)
to about ``transport`` plus one rate evaluation. The closure's +4-9% (#365)
should fall once the host sync is gone. Per-source budgets add one fused
reduction over ``du`` per source and stage, only on monitor steps.

How it will be measured, in phases 2 and 3, the way #355, #363 and #365
measured:

* time per step, minimum of 3 runs, pin against V3, for ``off``,
  ``transport`` and ``exact``, with budgets on and off. Decks: the
  ``test_axial_momentum_flux_form`` W2 C16/C32 and 3D arms;
  ``jupiter_gcm_dry`` C48 + Coriolis on GPU; and the
  ``test_diffusion_cubed_sphere`` configuration;
* peak memory per step, from ``torch.cuda.reset_peak_memory_stats()`` and
  ``max_memory_allocated()`` around one step, on the same runs;
* kernel launches per stage (#355 counted 4772 at C45).

7.3 Graph boundaries
--------------------

* The fixers' collective at the end of the stage is the only lane-A boundary.
  All fixers due at that stage (the AAM closure every stage, gravity work at
  the last) share one FP64 vector and one allreduce.
* Rule 2 of section 3.2, if it passes, removes the boundary.
* The pre- and post-implicit phases are segments around lane I's solve, not
  extra boundaries.
* Whether a device collective is capturable is asked in X-1.


8. Decide and justify
=====================

8.1 Default for conserve-axial-momentum, and the kill list
----------------------------------------------------------

**Default: off**, unchanged. Cheng's Claude replied that this needs no
decision.

Why:

* phases 1 and 2 are gated bitwise (G1, G4);
* ``transport`` is measured worse than ``off`` at Ω/(u/a) = 14;
* ``exact`` needs FP64 and ``interp_order: 4`` until G10.

The AAM budget is always reported (5.1). The default is revisited after G10.

==========================================  ===================================  ==========================
removed                                     replacement                          decks affected
==========================================  ===================================  ==========================
``conserve-axial-momentum: full``           ``exact`` (refused with a message;   in repo: test only.
                                            no silent remap)                     **FOR CHENG** (recorded)
``conserve-axial-momentum-closure: true``   ``conserve-axial-momentum: exact``   in repo: test only.
                                                                                 **FOR CHENG** (recorded)
``coriolis/type: 123``                      ``type: xyz`` (identical on          none in repo.
                                            Cartesian grids)                     **FOR CHENG** (recorded)
AAM key on a grid with no rotation axis     refused (today a no-op + warning,    none in repo
                                            ``hydro.cpp:70-76``)
==========================================  ===================================  ==========================

8.2 Diffusion fluxes into the register: later (phase 3)
-------------------------------------------------------

Diffusion enters the register in phase 3, not in the first migration PRs.

Why:

* Phases 1 and 2 are bitwise. Registering diffusion changes the order of
  additions on Cartesian grids, and the projection and seam arithmetic on the
  cubed sphere.
* Diffusion has its own tolerance suite (G8).
* One PR should not mix a register defect with a diffusion defect.

Until phase 3, diffusion is a ``SourceTerm`` (momentum and energy rows,
physical) with its own divergence. That is the declared exception to
requirements 4 and 5 in 4.1. ``exact`` with ``nu_iso > 0`` is refused until
then, because the closure would remove the viscous torque as if it were
numerical.


9. Line-count plan
==================

9.1 ``src/forcing`` (the baseline, 2,870 lines)
-----------------------------------------------

The V3 numbers are estimates. The measured sub-counts behind them come from
``sed -n 'a,bp' FILE | wc -l`` at the pin:

* ``Coriolis123`` 28 lines (``coriolis.cpp:40-67``); full/closure glue 38
  (``:176-213``); Ω set-up 67 (``:85-151``);
* diffusion ``from_yaml`` 91 (``diffusion.cpp:271-361``); profile checking 128;
* ``share_fluxes`` 74;
* ``forcing.hpp`` option structs 326 and classes 270.

=================================  ======  ======  =======================================================
file                               now     V3      basis
=================================  ======  ======  =======================================================
forcing.hpp                         674     260    options 326 -> 150 via schema (lane O); classes
                                                   270 -> 70; includes, macros, comments 78 -> 40
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
* The reduction is 2870 − 1988 = **882 lines, 30.7 %**.

Sensitivity:

* without the lane-O schema: 2098 (−26.9 %);
* before phase 3: 2089 (−27.2 %);
* both: 2199 (−23.4 %);
* without the 170 lines moved into ``axial_momentum.cpp``: 1818 (−36.7 %).

9.2 Lane-A code outside ``src/forcing`` (612 lines, section 2.2)
----------------------------------------------------------------

* hydro_forward 136 -> 25;
* hydro.hpp/.cpp 101 -> 45;
* coordinate.hpp 23 -> 3;
* gnomonic AAM 301 -> 240, plus 40 for spherical-polar K;
* mesh/meshblock 51 -> 10.

That gives 612 -> 363. Lane total: 3482 -> 2351 (−32.5 %).


10. Migration order and decision log
====================================

10.1 Migration order
--------------------

1. **Phase 1** (bitwise, G1): the ``SourceTerm`` base and registry, all twelve
   modules ported, the merges, one Coriolis, and monitor attribution.
2. **Phase 2** (G2-G7): the AAM consumer reads the register, R posts
   ``riemann.central``, the closure becomes a fixer, ``exact`` replaces
   full + closure, m1 reads Φ, and the host reads are removed. Cost and memory
   are measured (7.2).
3. **Phase 3** (G8, G9): diffusion posts to the register, and the averaging
   half of ``share_fluxes`` goes.
4. **Phase 4** (G10): FP32 and perturbation-flux for ``exact``; then the
   default is reconsidered.

10.2 Decision log (choices the brief left open)
-----------------------------------------------

=== ====================================================  ==========================================
#   decision                                              reason (section)
=== ====================================================  ==========================================
D1  F_c is the named part ``riemann.central``; no         removes 2 Riemann calls per x2/x3
    second flux evaluation                                direction (3.1, 4.1)
D2  modes ``off | transport | exact``; full and closure   full alone is neither cheap nor exact
    key removed; type 123 removed                         (3.2)
D3  default stays ``off``; AAM budget always reported     bitwise migration (8.1)
D4  closure = fixer at the last momentum change; device   requirements 6, 7 (3.3, 5.2, 7.3)
    R/I; one collective for all fixers
D5  one Coriolis ``-2Ω × m``, pre/post-implicit; m1       requirement 4; #355 phase split (3.4)
    reads Φ from the register
D6  FP32 lifted by consistent F_c + FP64 accum + G10      section 6
D7  per-source budgets by reductions of ``du``, no        requirement 9 (5.1, 7.2)
    scratch copy
D8  diffusion into the register in phase 3                bitwise gates of phases 1-2 (8.2)
D9  ``exact`` with ``nu_iso > 0`` refused until phase 3   closure would remove a physical torque
D10 AAM geometry in coordinate (lane S); physics in       requirement 5: rotation owned by coord
    ``axial_momentum.cpp``                                (3.8)
D11 side-study torque is a pluggable rule with gates      not read; decided by measurement (3.2)
D12 user-forcing species clip is a registered floor       requirement 7 (5.2)
D13 lane A writes nothing to ``Variables``                requirement 10 (4.1)
=== ====================================================  ==========================================


11. Open questions (cross-lane)
===============================

These are for the kickoff thread, addressed to each lane owner. They are not
resolved here.

Changes from the first draft:

* Old R-2 ("are contributor parts readable") is covered by requirement 2. It is
  narrowed into R-1's last clause.
* Old S-3 ("does the register reconcile seams") is covered by requirement 5. It
  is narrowed to the two details below.
* Old K-2 ("confirm the register model") is answered by the ten requirements,
  and dropped.
* Old O-1 and O-2 are lane O's, which is ours, and are kept out of the posted
  list.

The questions:

* **R-1, Linfeng (R).** Can the Riemann increment carry a named part
  ``riemann.central``? It would hold the mass rows
  ``1/2 Σ_s (ρ_s u_n)_L + (ρ_s u_n)_R``, ``1 + ny`` rows on x2/x3 faces, in the
  registered face frame, for HLLC, LMARS and Roe, and kept until the
  post-implicit phase of the stage. Under perturbation-flux, are the face
  densities totals or anomalies? Are parts kept as fields, or reported only as
  integrals? Diffusion's part adds 3 (5+ny) face rows if it must be a field.
* **R-2, Linfeng (R).** Can the step transfer take an operator-split entry with
  an explicit weight (the radial conduction's two half-step wall heat fluxes,
  1/2 each)? Or should split operators report to the monitor only?
* **I-1, Xinyue (I), and C-1, Tianhao (C).** Which of you posts the implicit x1
  mass transfer (``implicit_mass_correction()[IVZ] / dt``,
  ``hydro_forward.cpp:1645-1646``)? And which posts the tangential momentum
  correction (``picorr->correction()``, ``:1668-1669``) as named register
  increments after the solve? Can the stage give lane A a post-implicit source
  phase after them, while the w-free Coriolis rows still enter the solve
  (#355, 1.27 m/s otherwise)?
* **S-1, Jiheng (S).** Will S own ``RotationGeometry`` (k, its face averages,
  ``cos² lat`` at faces, ``rate_to_rows``, ``omega_local``, the m1 weights,
  ``has_on_axis_cell``) for gnomonic and spherical-polar grids?
* **S-2, Jiheng (S).** Is the register's documented frame for x2/x3 fluxes each
  face's own basis, as in ``flux2global2_``/``3_`` today? ``aam_k_f2`` and
  ``aam_k_f3`` are in that basis.
* **S-3, Jiheng (S), narrowed by requirement 5.** Does coord's one rotation
  function accept Cartesian-component vectors (diffusion's traction), or only
  panel-local ones? Can the register's seam exchange carry diffusion's
  interior-penalty traces (7 channels per seam face), or should diffusion keep
  its own trace exchange?
* **X-1, Xi (X).** Is a device-side allreduce of a small FP64 vector
  capturable inside a stage graph (NCCL or commux/UCX), or always a segment
  boundary? Where do you want the end-of-stage fixer point?
* **X-2, Xi (X).** May all fixers due at a stage (AAM closure every stage,
  gravity-work fixer at the last) share one collective through one fixer
  registry? Who owns that registry?
* **X-3, Xi (X).** Can the radial conduction's positivity check
  (``diffusion_radial.cpp:97-100``, a host read today) become a device failure
  flag, with the same redo path (``meshblock.cpp:1480``)?
* **K-1, Isaac (K).** ``ConstGravity`` is one of the twelve modules in
  ``forcing.hpp`` but outside lane A's baseline. Does lane K migrate gravity,
  its work (requirement 4: it reads the register) and the gravity-work fixer
  (requirement 7), on lane A's ``SourceTerm`` interface? If not K, the kickoff
  should assign it.


12. Hypotheses and how they are measured
========================================

==  ==============================================================  =================================
id  hypothesis                                                      measured by
==  ==============================================================  =================================
H1  ``riemann.central`` equals today's central mass rows (FP64)     G2, phase 2
H2  ``exact`` costs about ``transport`` plus one rate; the          7.2 timing, phase 2
    closure is cheaper without the host sync
H3  m1 from x1 transfers would help Cartesian f-plane energy        out of scope; separate study
H4  FP32 ``F_d`` error is harmless after the ``O(h)`` factor        G10, phase 4
H5  diffusion in the register makes the viscous torque telescope    G9, phase 3
H6  per-source budgets from ``du`` reductions are accurate in FP64  compare with a scratch-copy
    despite the flux divergence already in ``du``                   budget in a test, phase 1
H7  removing the per-stage host reads (7.1) changes no bits         G1/G4 bitwise runs, phases 1-2
==  ==============================================================  =================================
