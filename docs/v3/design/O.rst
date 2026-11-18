:orphan:

V3 lane O: output, input, utilities, diagnostics, options schema
=================================================================

Revision 2, 2026-11-18. Design and contract only: nothing in this branch
was written as code, built, tested or run. Owner: Jiachen (alternate: chen sihe).
Reviewers: Tianhao (C) and Xi (X), who replaces chen sihe as second
reviewer (chen sihe did not reply in the lane K thread). Design only: this branch proposes no
implementation change. Declarations are in ``docs/v3/skeleton/O/`` (not
built). The line table is section 9, with the same rows in
``docs/v3/design/O_lines.csv``.

Every ``path:line`` below is at chengcli/snapy
``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410`` (main, "FP32 step5 validation
(#368)") unless another commit is named. Line counts are ``wc -l`` at that
commit. Numbers marked *estimate* are design targets, not measurements.
Claims marked **hypothesis** are untested.

.. contents::
   :local:
   :depth: 1

1. Scope and pins
-----------------

**Scope.** ``src/output``, ``src/input``, ``src/utils``,
``src/diagnostics``, ``src/math``, ``src/z.junk``, and the options schema:
one table that generates the option structs, deck keys, Python stubs and
documentation.

**Pins.** snapy ``9e5de2d4`` (above). This lane changes no provider source.
The only provider call on its path is ``kintera::save_tensors`` /
``load_tensors`` (``output/restart.cpp:8,123``;
``input/read_restart_file.cpp:213,230``). #367 was validated with kintera
2.6.10 (#367 body); I did not re-check that version here.

**Sources read.**

* Every file in the lane directories, including the two ``.cpp_`` files and
  ``z.junk``.
* The callers in ``src/mesh/meshblock.cpp``: output set-up 201-255,
  ``make_outputs`` 1207-1224, ``print_cycle_diagnostics`` 1226-1339 and
  ``_init_from_restart`` 1603-1830.
* ``src/mesh/mesh.cpp:441-463``, ``python/csrc/{snapy.cpp, pyoutput.cpp,
  pyoptions.hpp}`` and ``V3_ROADMAP.md``.
* The bodies of PRs #313, #321, #323, #350, #352, #356 and #367 (#321 and
  #323 have empty bodies).
* The kickoff's twelve sections and ten register requirements, as quoted in
  the lane thread by Cheng's agent (2026-11-18). I did not read the kickoff
  posts themselves.

**Inventory at 919936da.** I found no document named as Tianhao's inventory
of this lane. The nearest is his technical report on happysky19/snapy branch
``report/g5g6-919936da`` (head ``2634e2351b04eb04b810046eb431a19a98bb4cd5``),
whose anchors are at ``919936da``:

* chapter 13, ``13-diagnostics.qmd:19-20``: ``src/diagnostics`` is legacy and
  not compiled;
* chapter 14, ``14-parallelism-io/_netcdf.qmd``: output table O-1 to O-36;
* chapter 16: known limits.

Each claim used here was rechecked at the pin.

**Where the design answers 1-8 of the lane post are.**

* Q1 (legacy and reachable): section 2.1.
* Q2 (writer core): sections 2.2 and 3.1.
* Q3 (diagnostics and budget format): sections 3.2 and 4.
* Q4 (options table): sections 2.4 and 3.3.
* Q5 (restart): section 3.4.
* Q6 (utils): section 3.5.
* Q7 (FP32 and CUDA graphs): sections 6 and 7.
* Q8 (line table): section 9.
* The formats and parser decisions, with the FOR CHENG items: section 3.6.
  Open choices: section 3.7.

1.1 Change list
~~~~~~~~~~~~~~~

**Rev 5.1** (on ``4afdbf37``; text only).

* **L-2 answered yes** (section 11.1; Tianhao). The callback type is
  ``std::function<Variables(Variables const&, FluxRegister const&)>``. The
  const accessors ``total``/``kept``/``step_field`` (R:82-84 at
  ``4c59162a``) give step transfer per unit area.

  Pin citations hold: the declaration is at ``meshblock.hpp:125`` (X's
  file; today one argument), and the caller at
  ``load_user_output_data.cpp:14, 19`` (O's file). The skeleton does not
  declare the callback, so it is unchanged.
* **L-3 pending R** (section 11.1).

  * Panel seams are not physical boundaries.
  * ``boundary(dim)`` sums the wall, outflow and custom faces.
  * The immersed-solid faces are O's proposal, accepted by Jiheng and
    pending Tianhao.

  Section 4 (``boundary`` row), section 3.2 (budget ``boundary``) and the
  requirement 4 and 5 rows are aligned.
* Section 9 is unchanged.

**Rev 5 final** (on ``5a7b46fa``; pin ``9e5de2d4``; register header R rev 2.2
``4c59162a``).

* **Part 1, EF/F6.** ``src/output/output_utils.cpp:53-55`` (``eps``,
  ``tke``, ``mut``; verified at the pin) is deleted with ``src/turbulence``
  (Cheng's F6; EF rev 3.4 ``b7bece9c``). Lane O counts it as its own row.
  The cut does not move, because the rewrite estimate already excluded
  these dead names (section 9).
* **Part 1, RULING V3-2.** From R-1 the positivity census lives in R's
  monitor per accepted step. O's ``observe`` accumulates the lifetime sums
  after the accept (section 4).
* **Item 1, lane S ranges final.** Jiheng's final split, in the lane S
  thread, matches 9.3 exactly. Each range was re-verified with ``wc -l`` at
  the pin.

  * To O, 127 lines: ``coordinate.cpp:62-114`` (53), ``:125-149`` (25),
    ``:213-214`` (2); ``layout.cpp:249-272`` (24);
    ``internal_boundary.cpp:14-36`` (23).
  * Stays in S, 99 lines: ``coordinate.cpp:115-124`` (10), ``:150-178``
    (29), ``:179-212`` (34), ``:215-240`` (26).

  The wording that said the ranges were pending is removed. Under RULING V3-1, S counts the
  deletion and O counts only its +20 new lines. Section 9 is unchanged.
* **Item 2, C18.** O's replies to Tianhao's notes 8 and 9 (11.5) are
  re-cited to lane C rev 6.2,
  ``bcac27c97a9efc1de2cb2041e6b709f07eedcefe`` (happysky19/snapy
  ``v3-design/C``). The old counter names ``face_floor_count`` and
  ``pstar_clamp_count`` become C's five names, adopted by R
  (``C.rst:1249-1251`` at ``bcac27c9``): ``face_floor.rho``,
  ``face_floor.p``, ``face_floor.species``, ``riemann.pstar_clamp`` and
  ``riemann.llf``. No citation of C's older heads remains in ``docs/v3``.

  The register header stays at ``4c59162a``. R rev 2.3 ``2b4c5775`` changes
  only comments in ``flux_register.hpp``: same 117 lines, and the
  declarations with comments stripped are identical.
* **Item 3, C14: matches.** C rev 6.2 says the five hand-written option rows
  (``ReconstructOptionsImpl::type``, ``scale``, ``shock``;
  ``RiemannSolverOptionsImpl::type``, ``dir``) are rows of O's one
  generated table. Lane O owns the table and the generator, and lane C owns
  the keys' meaning and proposes the defaults (``C.rst:873-881`` at
  ``bcac27c9``). The rows stay in lane C's headers until lane C switches
  them to O's generator after O's PR O4 (``C.rst:26-27, 44-45,
  883-885``).

  That is O's own statement:

  * one generated table, O owns it (section 3.3; 9.3, "Options rows belong
    in the one generated table");
  * the ``ADD_ARG`` blocks belong to the lane that owns the header
    (section 3.3, "What the lane does not claim");
  * other lanes switch after O4 (section 8.2, item 4).

  No change to O, and no conflict. C notes that its own line estimate still
  counts the 5 generated lines (``C.rst:886-888``). That is C's count; O
  counts no lines of C's headers.

2. Concepts and current sites
-----------------------------

2.1 Baseline, reachability and callers (Q1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``wc -l`` at the pin reproduces the lane post's 9,996 over 71 files:

* ``src/output``: 4,487;
* ``src/input``: 2,041;
* ``src/utils``: 1,717;
* ``src/diagnostics``: 1,585 (``.cpp``/``.hpp`` only, which leaves out
  ``CMakeLists.txt`` 19 and ``radiative_flux.cpp_`` 108 and
  ``v1_moments.cpp_`` 73);
* ``src/math``: 166.

Deleted but not counted toward the target are 785 lines outside that
baseline: those three files and ``z.junk/load_athena_output_data.cpp_``
(585). ``src/interface/stride_iterator.hpp`` (140), whose only includers are
in ``src/diagnostics`` (``rg -l stride_iterator src``), is in
``src/interface``, which lane X books in full (section 9).

"Compiled" below means matched by ``file(GLOB src_files ...)`` at
``src/CMakeLists.txt:35-53`` or by the CUDA glob at
``src/CMakeLists.txt:122-136``. Neither glob names ``diagnostics/``,
``math/`` or ``z.junk/``.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - file(s)
     - compiled
     - callers at the pin (verdict)
   * - ``input/parameter_input.*`` (993)
     - yes
     - **None.** This is the Athena++ ``<block> name=value`` parser
       (``parameter_input.cpp:1-40``). Its header is included only at
       ``python/csrc/snapy.cpp:10`` and ``output/hdf5.cpp:6``, and neither
       file uses a symbol from it. No ``*.inp``/``athinput*`` deck exists in
       the tree. **Legacy.**
   * - ``input/io_wrapper.*`` (364)
     - yes
     - Only ``parameter_input.hpp:12`` includes it. **Legacy.**
   * - ``input/signal.*`` (187)
     - yes
     - **No includer.** It is an older copy of ``utils/signal_handler``
       (compare ``signal.hpp:29-60`` with ``signal_handler.hpp:20-64``).
       **Duplicate.**
   * - ``input/command_line.*`` (186)
     - yes
     - ``examples/run_hydro.cpp:23,212`` and
       ``examples/jupiter_evap_precip_1d.cpp:193,278``. **Examples only.**
   * - ``input/read_restart_file.*`` (257)
     - yes
     - ``meshblock.cpp:1609``, ``python/csrc/snapy.cpp:47``. **Live.**
   * - ``input/check_keys.*`` (54)
     - yes
     - 56 call sites in ``src``. **Live.**
   * - ``output/mppnccombine.cpp`` (1,014)
     - yes
     - One call, at ``combine_netcdf.cpp:73``. This is the GFDL tool
       (``mppnccombine.cpp:1-40``) with file-scope state
       (``mppnccombine.cpp:93``). **A tool compiled into the library.**
   * - ``output/hdf5.cpp`` (27), ``HDF5Output``/``FITSOutput``
       (``output_formats.hpp:42-69, 98-104``)
     - body not compiled
     - ``HDF5OUTPUT`` is never defined (``configure.h.in:12-16``), the
       ``hdf5.cpp:23`` signature does not match the header, and the factory
       branch is commented out (``meshblock.cpp:242-244``). ``FITSOutput``
       has no definition. **Stubs.**
   * - ``src/diagnostics/*``
     - **no**
     - Reached by no glob and no ``add_subdirectory``. It includes headers
       that are not in snapy (``diagnostics.hpp:8-13``). The live per-cell
       diagnostics are in ``output/load_diag_output_data.cpp:54-338``.
       **Legacy.**
   * - ``utils/pull_neighbors.*`` (249), ``utils_dispatch.*`` (131),
       ``cuda_utils.*`` (22)
     - yes
     - **Never called.** The only hits are two commented lines in
       ``equation_of_state.cpp:254,257``. ``cuda_utils.h`` is included at
       ``implicit_dispatch.cu:15``, but ``checkCudaError`` is never called.
   * - ``utils/flip_zero.{cpp,h}`` (350)
     - yes
     - Only ``tests/test_rectify.cpp:12``. The library uses
       ``bc/flip_zero_impl.h`` (``bc/bc_dispatch.cpp:15``).
   * - ``src/math/{lubksb,ludcmp}.h`` (166)
     - header only
     - ``implicit/forward_sweep_impl.h:14-15``. **One caller module.**

Commands behind the rows (run at the pin):

.. code-block:: text

   $ rg -n "ParameterInput|parameter_input\.hpp" src python tests examples \
        -g "!src/input/**" -g "!src/diagnostics/**" -g "!src/turbulence/**"
   python/csrc/snapy.cpp:10   src/output/hdf5.cpp:6
   src/output/output_formats.hpp:15,23,30,74   (inside /* */ comments)
   $ rg -n "io_wrapper|IOWrapper" src python tests examples -g "!src/input/**"   (no match)
   $ rg -n 'diagnostics/' src python tests examples CMakeLists.txt cmake \
        -g '!src/diagnostics/**'                                             (no match)
   $ rg -n "mppnccombine\(" src python tests examples
   src/output/combine_netcdf.cpp:17,73   src/output/mppnccombine.cpp:94
   $ rg -n "pull_neighbors|bdot_out|checkCudaError" src python tests -g "!src/utils/**"
   src/eos/equation_of_state.cpp:254,257  (both commented out)

2.2 Output path today (Q2)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Set-up.** ``meshblock.cpp:232-249`` picks ``restart``, ``netcdf`` or
  ``pnetcdf``. The slice checks run three times:
  ``output_type.cpp:90-98`` (slice against sum),
  ``meshblock_options.cpp:65-86`` (bounds and order) and
  ``meshblock.cpp:203-230`` (both).
* **Write.** ``make_outputs`` (``meshblock.cpp:1207-1224``) calls
  ``AccumulateStats`` every step (``:1210``) and writes when
  ``current_time >= next_time``.
* **#350's writer core** (186 lines; merge ``ffaffdd25f3b7d3d63204fde84cdcbb476f08517``) shares
  ``output_range``, ``make_output_dir``, ``output_block_location``,
  ``combine_on_root`` and ``glob_block_files``. It did not touch the format
  code.
* **What is left in the format code:**

  * Two copies of layout, name expansion and grid-kind packing:
    ``netcdf.cpp:287-318, 320-348, 490-562`` and
    ``pnetcdf.cpp:209-236, 350-415``.
  * A hand-managed ``OutputData`` list of ``AthenaArray`` copies
    (``output_type.hpp:93-105``; ``output_type.cpp:201-244``), with slice
    and sum written as nested loops (``output_type.cpp:415-557``).
  * ``strtok`` name splitting (``utils/vectorize.cpp:12-22``). The define
    pass splits on ``;`` (``netcdf.cpp:269,288``) and the write pass on
    ``,`` (``netcdf.cpp:486``, ``pnetcdf.cpp:352``).
  * Per-cell ``.item<double>()`` reads of the coordinates
    (``netcdf.cpp:433-481``, six sites; ``output_type.cpp:438``).

* **Combine.** ``combine_netcdf.cpp:64-75`` builds ``argv`` for
  ``mppnccombine``, which joins files that snapy wrote itself and that carry
  ``domain_decomposition`` (``netcdf.cpp:163-237``) and ``NumFilesInSet``
  (``:262``).

2.3 Monitor, fixers and budgets today (Q3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``print_cycle_diagnostics`` (``meshblock.cpp:1226-1339``; the line is
produced at ``:1232-1338``) does the following
every ``ncycle_out`` cycles:

* sums the conserved rows, KE and PE (``:1257-1283``);
* reads the meters with ``.cpu()`` (``:1288-1295``);
* reduces over the process group (``:1300-1313``);
* prints with ``.item<double>()`` (``:1319-1337``).

It writes no file. The fixer state lives in hydro:

* limiter (``lim_cut``, ``lim_flux``);
* positivity (``positivity_severe``, ``positivity_min``);
* gravity-work fix, as a pending/commit/drop accumulator;
* axial closure sums (``_aam_cl``).

All are in ``hydro.hpp:235-311``. The VIC clamp is
``picorr->clamp_residual()`` (``meshblock.cpp:1295``).

The convection examples each write their own CSV
(``examples/planar_convection.cpp:3``, ``spherical_convection.cpp:19``,
``cubed_sphere_convection.cpp:3``). The legacy ``HydroFlux`` recomputed
``rho*v1`` from cell centres (``diagnostics/hydro_flux.cpp:56-63``). Output
names ending in ``_flux`` get grid ``CCF`` (#321,
``output_utils.cpp:216-219``); such fields come only from user callbacks
(``load_user_output_data.cpp:19``), which compute their own fluxes.

2.4 Options and environment switches (Q4), with corrections to the brief
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**32/48 against 31/47.** The brief's "``hydro.hpp`` 32 option entries,
``forcing.hpp`` 48" are ``grep -c ADD_ARG`` counts. That pattern also
matches the ``#undef ADD_ARG`` line that ends each header:
``src/hydro/hydro.hpp:490`` and ``src/forcing/forcing.hpp:674``.
``grep -n ADD_ARG <file> | grep -v 'ADD_ARG('`` prints exactly those two
lines. Counting declarations with ``rg -o 'ADD_ARG\(' <file> | wc -l``
gives **31** and **47**.

Over ``src/`` there are 213 ``ADD_ARG(`` entries (``add_arg.h``, which
defines the macro, excluded) and 137 ``ADD_OPTION(`` bindings in
``python/csrc`` (``pyoptions.hpp`` excluded). These are counts only. I did
not match options to bindings by name.

**Each option is written in up to six places:**

* the ``ADD_ARG`` line with its default;
* ``report()``;
* the ``check_keys`` list;
* the ``from_yaml`` read, which carries its own default (#352 fixed three
  options where the two defaults differed);
* the pybind ``ADD_OPTION`` line;
* the ``.pyi`` stub and the docs.

**Environment switches.** ``rg -n 'getenv\(' src -g '!src/layout/**'``
finds:

* ``SNAP_FLUX_COVARIANCE`` and ``SNAP_X1_MASS_COVARIANCE``
  (``hydro.cpp:47,51``) are **only warned about** since #352
  (``hydro.cpp:46-54``).
* ``SNAP_ANOMALY`` overrides the deck (``hydro_options.cpp:84-95``).
* ``SNAP_FP32_STORAGE`` and ``SNAP_FP32_STORAGE_ROWS`` were missing from
  the brief. They are read once per process, in a function-local static
  (``fp32_storage.cpp:88-97``: mode at ``:92``, rows at ``:93``, through
  ``env_or`` with ``getenv`` at ``:25``). They are documented at
  ``fp32_storage.hpp:19-25``, applied at ``meshblock.cpp:621-642``, and
  validated with messages naming the variables (``fp32_storage.cpp:36, 57,
  79``).
* ``SNAP_X1_CENTROID_EXACT`` is read once per process at
  ``x1_centroid.cpp:68-79``, through ``get_env`` (``layout.hpp:39-42``).
  It appears 53 times in 21 files over the repository, ``docs/v3``
  excluded. A proposed table row is in section 3.3; the owning lane
  decides.

3. Target design
----------------

Skeleton: ``writer_core.hpp``, ``diagnostics.hpp``, ``restart.hpp``,
``options_table.yaml``, ``gen_options.pyi``, ``budget_series.cdl``.

**Data flow** (one accepted step; lane R and lane X boxes are other lanes'):

.. code-block:: text

   [captured segment(s), lane X]                 host, after global accept
   RK stages --F_s--> FluxRegister (R) ----.      .------------------------------.
   fixers  --amount-> Monitor (R) ---------+----> | Consumer::observe (device)   |
                                           |      |   BudgetConsumer, FluxProfile|
   state U, W ---------------------------- +----> | GriddedWriter::accumulate_   |
                                                  |   stats (device)             |
                                                  '--------------+---------------'
                                     output cadence only         v
                    .----------------------+---------------------------------.
                    | Consumer::flush: collective + host read -> BudgetSeries |
                    | GriddedWriter::write: Frame -> to_host -> Sink -> part  |
                    |   -> combine_on_root -> combine_parts                   |
                    | write_restart (stored set only; register NOT written)   |
                    '---------------------------------------------------------'

3.1 Writer core (Q2)
~~~~~~~~~~~~~~~~~~~~

**Types.**

* ``Field`` holds names, a ``GridKind``, units, the long name and one tensor.
* ``Frame`` holds the fields, the coordinates and the per-axis placement. It
  replaces ``OutputData`` and the list. Slicing and summing become one
  ``index_select`` or ``sum`` per field. ``to_host()`` makes one copy per
  field, the only host read on the gridded path.
* ``Sink`` is one per format: netcdf and pnetcdf, about 120-140 lines each
  (*estimate*).
* ``GriddedWriter`` replaces ``OutputType`` and the four format classes. It
  keeps the schedule, both #313 schedule keys and the Welford statistics
  (``output_type.cpp:285-381``).

**Combining.** ``combine_parts`` (new file ``src/output/combine_parts.cpp``,
about 200 lines, *estimate*; an addition row in section 9) reads only the
attributes snapy writes. It replaces ``mppnccombine.cpp`` after gate G-2
passes (sections 5 and 10.1). Removal approved by Cheng; settled.

**No AthenaArray.** The writer core holds ``torch::Tensor`` only. It drops
every ``AthenaArray`` use in lane O: ``output_type.hpp:22, 99``,
``output_utils.hpp:10, 47``, ``output_type.cpp:388, 399, 467, 521, 528,
535`` and the 14 ``CopyFromTensor``/``InitFromTensor`` sites in
``load_diag_output_data.cpp:120-332`` (``rg -c 'CopyFromTensor|InitFromTensor'
src/output/load_diag_output_data.cpp`` gives 14). It also drops the
``AthenaArray`` element reads in the two sinks:

* ``pdata->data(...)`` at ``netcdf.cpp:496, 507, 518, 525, 533, 541, 548,
  558`` and ``pnetcdf.cpp:362, 371, 380, 386, 393, 399, 404, 413``;
* ``get_num_variables(grid, pdata->data)`` at ``netcdf.cpp:271, 290, 488``
  and ``pnetcdf.cpp:197, 213, 354`` (the ``AthenaArray`` template at
  ``output_utils.hpp:47``).

(``load_hydro_output_data.cpp:98-133`` is inside a ``/* */`` comment.)
Lane X's deletion of ``src/interface`` depends on this (section 9).

**Ownership.** The writer borrows the block's variables for one write and
owns only host copies, its schedule and its statistics tensors.

**Moves out of src.** ``input/command_line.*`` goes to ``examples/``. No
combiner command-line wrapper is added (section 10.1).

3.2 Diagnostics as register consumers (Q3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Consumer`` has two methods:

* ``observe``: runs on the device every accepted step, after the accept,
  and reads the register's step transfer, the monitor and X's host ``dt``
  (section 7);
* ``flush``: makes the collective and the host read, at its own cadence
  only.

Two consumers replace ``src/diagnostics``:

* ``BudgetConsumer``. For each row it accumulates in FP64:

  * ``boundary``, the outward register transfer through the physical domain
    boundary: wall, outflow and custom faces. Cubed-sphere panel seams are
    not physical boundaries. Immersed-solid faces are O's proposal, pending
    R (L-3);
  * ``fixer[f]``, the monitor's amount for each fixer.

  At each record it writes ``residual = (total - total_prev) + boundary -
  sum fixer``.

  **No double counting.**

  * Amounts of ``AmountKind`` ``cell_source``, ``floor`` and ``fixer``
    enter the identity (R:25-31; R.rst:550-554).
  * A ``flux_correction`` amount is a face change already inside
    ``boundary``. It is reported as a part (requirement 2) and not added
    again.
  * Counters (limiter, positivity, face-floor and p* counts; R:99-100)
    are counts and never enter it.
  * The positivity census (RULING V3-2, section 4) is the per-step
    ``positivity.carry``/``positivity.withhold`` amounts, which are
    ``flux_correction`` (R.rst:620-621), plus the positivity counters and
    ``pos_min`` (R.rst:549, 582-588). It enters the lifetime sums and the
    cycle line, never the residual.

* ``HydroFluxProfile``: the horizontal mean of the registered x1 transfer,
  on grid ``--F``.

The per-cell diagnostics stay as field loaders. A ``FieldDiagnostic``
registry replaces the ``ContainVariable`` chains (11 lines,
``rg -c 'ContainVariable\(' src/output/load_diag_output_data.cpp``). The
cycle line becomes a printer over ``BudgetConsumer``'s totals. Its content
is unchanged.

**Budget time-series format** (``budget_series.cdl``):

* One file per run, ``<basename>.budget.nc``, NetCDF4, appended by the root
  rank. Every variable is ``NC_DOUBLE``.
* Dimensions: ``time`` (unlimited), ``row`` and ``fixer``.
* Rows: ``mass``, ``species/<n>``, ``energy`` (E+PE), ``aam`` and
  ``scalar/<n>``.
* Sources: every ``amount_names()`` entry whose ``kind`` is ``cell_source``,
  ``floor`` or ``fixer`` (for example A's ``aam_closure``, a ``fixer``).
  ``positivity.carry``, ``positivity.withhold`` (R.rst:620-621) and
  ``vic.mass_clamped`` are ``flux_correction``, and
  ``chem.elements`` and ``vic.momentum_t.top`` are ``diagnostic``; these are
  reported, not added (R.rst:550-554, 572-575). The gravity-work fixer has
  no poster after Cheng's rulings (a) and (b) (R.rst:572-573).
* Variables: ``total``, ``boundary``, ``fixer``, ``residual``, and the
  monitor counters.
* Requested with ``- {type: budget, dt: ...}``.

Reasons for NetCDF rather than CSV:

* every snapy reader already reads NetCDF;
* units travel with the data;
* adding a species or a fixer changes no reader;
* double precision, because ``NC_FLOAT`` hides relative changes below
  ``1e-7`` (``output_type.hpp:79-83``).

3.3 Options table (Q4)
~~~~~~~~~~~~~~~~~~~~~~

**Table format.** One YAML table per module (``options_table.yaml``). Each
row has the columns ``name``, ``key``, ``type``, ``default`` (the only
default), ``env``, ``python`` and ``doc``, and a module can add cross-option
``checks``.

**Generator** (``gen_options.pyi``). It emits:

* the C++ struct;
* ``from_yaml``, with the ``check_keys`` list;
* the pybind lines;
* the ``.pyi`` stubs, which are checked in and compared in CI;
* one ``rst`` page per module.

It runs at CMake configure time and in ``setup.py``. A one-time migration
seeds the tables from the pin's headers, and a test checks that the
generated members and defaults equal the pin's.

**Environment switches in the table:**

* ``SNAP_FLUX_COVARIANCE`` and ``SNAP_X1_MASS_COVARIANCE``: the warning
  stubs (``hydro.cpp:46-54``) are deleted.
* Execution and precision rows (Xi note 2): ``dynamics/precision`` exists at
  the pin (``hydro_options.cpp:55, 67``). ``integration/execution`` and
  ``integration/execution-strict`` are sample rows with **provisional**
  names, which are Cheng's call (``V3_ROADMAP.md:300-301``).
* Lane S's options parsing is **accepted** into the schema (rev 4, section
  9.3). The ``geometry``, ``distribute`` and ``boundary-condition/internal``
  options become generated rows. S keeps its grid and repartition logic.
* ``SNAP_X1_CENTROID_EXACT`` (read at ``x1_centroid.cpp:68-79``): a proposed
  row ``geometry/x1-centroid-exact`` with this ``env`` (note 14). The lane
  that owns ``src/coord`` decides whether to adopt it.
* ``SNAP_ANOMALY``: the row ``dynamics/anomaly`` with ``env: SNAP_ANOMALY``,
  the only permanent override. It is named in the user-facing message at
  ``meshblock.cpp:1782`` and used on 5 test lines. **FOR CHENG.**
* ``SNAP_FP32_STORAGE`` and ``SNAP_FP32_STORAGE_ROWS``: **removed, not
  tabled (settled: Cheng approved removing the FP32 storage emulation).**
  The switches get no row in the options table. ``src/utils/fp32_storage.*``
  (175 lines, counted in O) is deleted together with its application in
  lane X's files (28 lines, agreed with Xi; section 8.1).

**What the lane does not claim.** The ``ADD_ARG`` blocks, ``from_yaml``
bodies, ``python/csrc/py*.cpp`` and ``.pyi`` stubs that the generator
replaces belong to the lanes that own those files. They are not in this
line budget.

3.4 Restart (Q5)
~~~~~~~~~~~~~~~~

**Stored** (unchanged from the pin):

* ``hydro_u``, FP64 totals (``restart.cpp:44``);
* ``hydro_u_stored``, the anomaly bits in the run dtype (``:45``);
* ``hydro_ref``, ``hydro_ref_species`` and ``hydro_ref_x{1,2,3}f``
  (``:46-61``);
* ``hydro_restart_metadata`` (``:50-53``);
* ``hydro_w`` (copied with every other defined entry of ``vars``,
  ``restart.cpp:31-38``; kept stored, Xi note 1);
* ``scalar_s`` (``:30-34``);
* ``solid``, ``fill_solid_hydro_u`` and ``fill_solid_hydro_w``, when present
  (J16). ``solid`` is a ``vars`` entry set at initialization
  (``meshblock.cpp:514, 541-561``). ``restart.cpp:31-38`` writes it with
  every defined entry, and ``_init_from_restart`` restores it in its
  move-to-device loop (``meshblock.cpp:1733-1744``). It is then read by the
  solid fill (``:936-938``) and the time step (``:602-603``).
  ``fill_solid_hydro_w`` is read at ``hydro.cpp:495``;
* the boundary references, when present;
* ``last_time`` and ``last_cycle``;
* the output schedule (``:73-99``).

The stored set is the explicit list ``kRestartStoredKeys`` in
``restart.hpp`` (14 keys, with ``solid`` and ``fill_solid_hydro_w`` added
for J16). ``write_restart`` copies only those keys. Registers and
``StateStore`` tensors are never in the ``Variables`` it copies
(requirement 10).

New: ``BudgetCarry``, the pending FP64 sums since the last budget record.
It is written **only when a budget output is requested**. A file without a
carry, including every pin-era file, starts a new budget interval at the
first ``observe`` (Xi note 10). Decks without a budget output keep exactly
today's stored set, so this needs no FOR CHENG item.

**Load into fixed storage (Xi note 9).** The restart load ``copy_``\ s each
stored tensor into its ``StateStore`` buffer, cast to the dtype of that
buffer's ``PrecisionPolicy`` row (lane X). A restart is written only after
an accepted step, after ``begin_step()`` has zeroed the register (R:48;
R.rst:464-467). So the file needs no new stage or dtype field.

**Derived** (never written):

* ``scalar_r``: not written when ``scalar_s`` and ``hydro_u`` are present
  (``restart.cpp:30-34``), and rebuilt on load (``meshblock.cpp:1827-1829``,
  ``set_scalar_primitive``). The rev 2 text listed ``hydro_w`` here by
  mistake (Xi note 1);
* **the flux register and the monitor** (requirement 10);
* graph executables, captured workspaces and cast caches
  (``V3_ROADMAP.md:80``);
* coordinates, metrics and reconstruction coefficients.

**The #367 metadata is kept unchanged** (``restart_metadata.hpp:9-12``).
Its checks move from ``meshblock.cpp:1623-1653`` into
``check_restart_metadata``.

**Changes:**

* The writer, the bundle code and the reader merge into one module, with one
  bundle magic. Today it is defined twice: ``combine_restart.cpp:16`` and
  ``read_restart_file.cpp:23``.
* The reader throws an error instead of printing to ``std::cerr`` and
  returning an empty map (``read_restart_file.cpp:157-220``).

3.5 ``src/utils`` (Q6)
~~~~~~~~~~~~~~~~~~~~~~

Moved:

* ``fp64_store.hpp`` to ``src/hydro``; its only includer is
  ``hydro.hpp:19``.
* ``signal_handler.*`` to ``src/mesh``; its includers are ``mesh.cpp:28``
  and ``meshblock.cpp:24``.
* ``math/*.h`` to lane I, at ``src/math/`` (Xinyue accepted; section 9).

Deleted:

* ``pull_neighbors.*``, ``utils_dispatch.*`` and ``cuda_utils.*``: never
  called.
* ``vectorize.*``: ``Field.names`` replaces it.
* ``refine.*``: its only caller is super-resolution output
  (``output_type.cpp:143-187``), which no deck uses (it appears only at
  ``tests/test_user_output.cpp:1144``). **FOR CHENG**: it is bound in Python
  at ``pyoutput.cpp:43``.
* ``utils/flip_zero.*``: the int-only copy (``flip_zero.h:30-80``) of the
  templated ``bc/flip_zero_impl.h:24-202``. They have the same four
  functions, except that ``reconstruct_solution`` writes in place in the
  ``bc`` copy (``flip_zero_impl.h:202``). ``test_rectify`` is ported to
  ``T = int`` (gate G-6).

* ``fp32_storage.{cpp,hpp}`` (119 + 56): the FP32 storage emulation.
  **Settled: Cheng approved its removal.** Its callers in lane X's files are
  listed in section 8.1.

Kept: ``log.hpp``, ``loops.cuh``, ``on_host.hpp`` and
``tensor_cast_cache.hpp`` (11, 8, 3 and 3 includers).

3.6 Decisions and FOR CHENG
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**D1 Formats that remain.**

* ``netcdf``: requested by 18 of 25 ``examples/*.yaml`` and 4 files in
  ``tests/``.
* ``restart``.
* ``pnetcdf``, as a sink. **Settled: Cheng decided it stays.** It is used
  by ``examples/straka.yaml:57`` and ``shallow_splash.yaml:58``. No test
  runs it on several ranks (``tests/run_straka.cmake:28`` swaps it for
  netcdf), so that test is added with the sink (section 10, O5).
* ``budget`` (new).

Removed:

* ``hdf5`` and ``fits`` stubs. **Settled: removal approved by Cheng.**
* the built-in GFDL combiner ``mppnccombine.cpp``. **Settled: removal
  approved by Cheng.** It is replaced by the smaller ``combine_parts``
  (decision in section 10.1).
* the FP32 storage emulation (``SNAP_FP32_STORAGE``,
  ``SNAP_FP32_STORAGE_ROWS``, ``src/utils/fp32_storage.*``). **Settled:
  removal approved by Cheng.**
* the ``super-resolution`` option. **FOR CHENG (still open).** I proceed on
  this proposal; the row stays.

**D2 The legacy input parser goes.** ``parameter_input.*`` and
``io_wrapper.*`` (1,357 lines) have no caller and no deck uses their format.
If a workflow needs one, an Athena++-to-YAML converter belongs in
``tools/``.

**D3 ``src/diagnostics`` goes.** Register consumers replace it. Its todo
list becomes budget rows or ``FieldDiagnostic`` entries when someone needs
them, written against the register.

**Settled by Cheng (2026-11-18):** pnetcdf stays; the hdf5/fits stubs,
``mppnccombine.cpp`` and the FP32 storage emulation are removed.

**Still FOR CHENG** (I proceed on the proposal and keep the rows):

* super-resolution (removed in this design, with ``utils/refine.*``);
* the ``SNAP_ANOMALY`` override (kept as the only ``env`` row).

3.7 Open choices made, with reasons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O-1 Baseline excludes ``.cpp_`` and ``diagnostics/CMakeLists.txt``. This is
the only reading that gives 9,996. Those files are deleted but not counted.

O-2 Moves are not cuts. Moving files leaves the repository's size the same.

O-3 Budget in NetCDF, not CSV (section 3.2).

O-4 pnetcdf stays (settled by Cheng). It costs about 130 lines with the
core.

O-5 mppnccombine is removed (settled by Cheng) and replaced by
``combine_parts``, not moved to ``tools/``. The replacement is a fifth of
the size and has no file-scope state. Section 10.1 records why a
replacement is still needed.

O-6 Generated files are not checked in, except the ``.pyi`` stubs, which
IDEs need without a build.

O-7 ``SNAP_ANOMALY`` stays as an override. It is named in a user-facing
message.

O-8 (withdrawn) The FP32 storage variables no longer get an ``env`` row:
the emulation is removed (settled by Cheng).

O-9 The line estimates come from the skeleton plus the live code they keep.
A +50 % sensitivity is given in section 9.

O-10 The cycle line stays, printed from the budget totals.

4. Flux-register contract
-------------------------

Lane O **posts nothing** to the register or the monitor and **corrects
nothing**. It is a reader only. The names below are lane R's **frozen
header, rev 4.2 (R.rst rev 2.2)**: happysky19/snapy
``4c59162ae37080d0ab307b2823742b44f46da7c3``,
``docs/v3/skeleton/R/flux_register.hpp``, cited as ``R:<line>``. R's
contract text is ``docs/v3/design/R.rst`` at the same sha, cited as
``R.rst:<line>``. Every ``R:``/``R.rst:`` citation in this document is at
that sha.

**What rev 2.2 changed, checked against O** (R.rst:20-50):

* ``weights()`` (R:50) returns the stage's ``harp::IntegratorWeight`` so
  that a **poster** can pass its ``stage_weight``. **O does not need it.**
  O posts nothing, and every value it reads (``step_field``, ``amounts``,
  ``boundary``, ``domain_total``) already carries the weights (R:85;
  R.rst:555-559).
* ``AmountKind::diagnostic`` (R:30) is for amounts that change no row
  (``chem.elements``, ``vic.momentum_t.top``; R.rst:503-504, 790-793).
  **O excludes ``diagnostic`` amounts from the budget residual** and only
  reports them. R's formula adds only ``cell_source``, ``floor`` and
  ``fixer`` (R.rst:550-554).
* ``after_combine`` (R:22) is now entered after **each** stage's combine
  (R.rst:35-36), and its amounts accumulate into the step like the others
  (``end_stage``, R:85). **O reads amounts only once per accepted step**, in
  ``observe`` after the accept and before the next ``begin_step()``
  (R.rst:560). By then every stage's ``after_combine`` amounts are inside
  the step sums, so the per-stage entry changes nothing for O.

**What lane O reads, with R's exact signatures:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - R's declaration
     - lane O's use
   * - ``torch::Tensor FluxRegister::step_field(int dim) const;``
       (R:84, "kRegister, per unit area")
     - ``HydroFluxProfile`` (dim 1).

       * Dim 1 must be requested in ``RegisterRequest::step_field``
         (R:38), which X passes to ``FluxRegister::reset`` (R:43-44).
       * Units are per unit face area (R:4).
       * **The only place O applies an area**: it multiplies by
         ``CoordinateImpl::face_area1()`` (``coordinate.hpp:254``, defined
         at ``coordinate.cpp:614``; ``face_area2/3`` at
         ``coordinate.hpp:259, 264``). The divergence applies the same
         areas (``coordinate.cpp:732-746``).
   * - ``std::vector<std::string> RegisterMonitor::row_names() const;``
       (R:93)
     - The budget rows: the conserved rows in row order, then the derived
       rows ``e+pe`` (lane I) and ``aam`` (lane A) (R:33-34;
       R.rst:528-542). Scalars have their own register instance and
       monitor, one per Scalar module (R:41; R.rst:539-540). ``BudgetConsumer``
       reads every instance.
   * - ``torch::Tensor RegisterMonitor::domain_total() const;`` (R:98,
       "fluid cells, (nrows,) FP64")
     - ``total``. **R owns the domain totals** (R:89-90; R.rst:460-463). O
       sums them over blocks in ``flush``.
   * - ``torch::Tensor RegisterMonitor::boundary(int dim) const;`` (R:97,
       "area-integrated, outward, FP64")
     - ``boundary``, summed over ``dim``. It is **area-integrated**
       (R.rst:555-559), so O applies no area to it. It sums the block's
       physical-boundary faces: wall, outflow and custom (Jiheng, lane S).
       Panel seams are not physical boundaries, and their two entries are
       reconciled by ``replace`` (R:63-64). O's proposal to add the
       immersed-solid faces, all rows, is accepted by Jiheng and **pending
       R** (L-3). Derived rows have no
       ``boundary`` entry; their owners post the transfer as a
       ``derived_amount("<row>.boundary.<dim>", ..., cell_source)``
       (R:72-74; R.rst:542-545), and the formula below needs no special
       case.
   * - ``std::vector<std::string> amount_names() const;`` (R:94),
       ``AmountKind kind(std::string const& name) const;`` (R:95),
       ``torch::Tensor amounts(std::string const& name) const;`` (R:96,
       "(nrows,) FP64")
     - **Single increments**, read one name at a time. ``AmountKind``
       (R:25-31):

       * ``flux_correction`` is already inside ``boundary`` and is reported
         only.
       * ``cell_source``, ``floor`` and ``fixer`` are added.
       * ``diagnostic`` changes no row and is reported only.

       The gravity-work fixer has no poster after Cheng's rulings
       (R.rst:572-573). If it returns, only its correction is a source
       (Xi note 3c). R's gravity-work face posts are inside ``boundary``
       (R:19).
   * - ``counter_names()`` (R:99), ``counters()`` (R:100, int64);
       ``minima()`` (R:101, FP64)
     - The ``counter`` and ``minimum`` variables of ``budget_series.cdl``,
       by name. Counts never enter the budget identity.

**The budget identity** (R.rst:550-554):
``residual = delta domain_total + sum_d boundary(d) - sum amounts(name)``
over ``kind(name)`` in {``cell_source``, ``floor``, ``fixer``}. ``flux_correction``
and ``diagnostic`` are reported, never added.

**When.**

* **Cross-rank amount SUM: lane O's.** Lane X rev 3.1 assigns it to O and
  keeps only the MAX read of the retry flags at step end
  (UCzhangxi/snapy ``bac201d1c06e3eba3f804979789fdf83ff9baa30``,
  ``X.rst:103-105, 165-173``; ``meshblock.cpp:1559-1568`` at the pin).
  In O:

  * ``observe`` accumulates the FP64 amounts, boundary sums and totals on
    the device every accepted step. There is no host read and no
    collective.
  * ``flush`` does the cross-rank SUM of those FP64 sums (``Layout``'s
    process group) and the one host read, once per output interval. The
    output interval is the budget record, and the cycle line every
    ``ncycle_out`` cycles.

  R.rst:460-463 says all monitor values are read in X's step-end read. X
  rev 3.1 narrows that to the retry flags, and O follows X (see 11.7).
* **Positivity census (RULING V3-2).** From R's PR R-1, the positivity
  census lives in R's monitor **per accepted step**. ``EfStatus``'s slots
  move into the monitor in R-1 (R.rst:587-588). The census is:

  * the ``flux_correction`` amounts ``positivity.carry`` and
    ``positivity.withhold`` (R.rst:620-621);
  * the int64 counters (``counters()``, R:100);
  * the FP64 minimum ``pos_min`` (``minima()``, R:101; R.rst:549).

  ``begin_step()`` zeroes it at each step start (R:48), and ``rollback()``
  drops a rejected step's census on every block (R:86; R.rst:464-467). So
  the monitor never holds more than one accepted step.

  **Lane O keeps the lifetime sums.** ``observe`` runs after the accept and
  before the next ``begin_step()`` (R.rst:560). It adds that step's census
  to O's own device sums: counts and amounts are summed, and ``pos_min`` is
  kept as a running minimum. ``flush`` reduces them across ranks (SUM, and
  MIN for the minimum) with the budget sums, and prints them in the cycle
  line's "run-to-date" fields (today ``meshblock.cpp:1329-1335``).

  *Choices and reasons:*

  * A rejected step never reaches O, because ``rollback()`` drops it
    before ``observe``. So no second filter is needed.
  * The lifetime sums are not written to restart. That keeps the stored
    set unchanged (no file-format change) and matches the pin, where the
    run-to-date meters restart at zero (Tianhao's report, chapter 16,
    known limits).
  * The census amounts stay out of the residual, because they are
    ``flux_correction`` and already inside ``boundary``.
* ``begin_step()`` zeroes the step and the monitor (R:48).
* ``rollback()`` zeroes the register and the monitor on every block (R:86;
  R.rst:464-467).
* ``observe`` runs after the accept and before the next ``begin_step()``
  (R.rst:560).
* Lane O calls no ``FluxRegister`` mutator (R:43-86). It uses only
  ``step_field`` and the monitor (``FluxRegister::monitor()``, R:87).

**R-O1, R-O2 and R-O3 are settled** by the frozen header:

* R-O1: ``row_names()`` (R:93), with the derived rows ``e+pe`` and
  ``aam`` (R:34) and scalars per instance. ``boundary(dim)`` is
  area-integrated (R:97).
* R-O2: ``counter_names()`` and int64 ``counters()`` (R:99-100).
* R-O3: ``AmountKind`` and ``kind(name)`` (R:25-31, R:95).

* **Ordering.**

  * ``observe`` runs after the global accept and before the next step's
    register reset.
  * ``flush`` and ``write`` run at their cadence, after ``observe``,
    outside captured segments.
  * On a redo, ``observe`` is not called for the rejected attempt.

* **Writes.** Lane O writes no register state to restart (requirement 10).
  After a restart the budget continues from ``BudgetCarry``, and the
  register starts empty.

Check against the ten requirements:

.. list-table::
   :header-rows: 1
   :widths: 6 40 54

   * - req
     - requirement (as quoted)
     - lane O
   * - 1
     - one owner of every conservative face flux; one frame and area
       weighting
     - Lane O owns no flux. R's posts are per unit face area, in coord's
       ``flux2global<d>_`` frame (R:4-5). O applies
       ``CoordinateImpl::face_area1()`` to ``step_field(1)`` only.
       ``boundary(dim)`` is already area-integrated (R:97).
   * - 2
     - named increments in a declared order; total and parts reportable
     - The budget reports them as ``total``, ``boundary``, ``fixer[f]`` and
       ``residual``, in R's declared order.
   * - 3
     - stage fluxes accumulated with the stage weights into the step
       transfer
     - **Adopted.** It replaces assumption A1 of the first draft. The
       consumers read only the step transfer, never stage fluxes.
   * - 4
     - flux divergence and flux-derived terms read the register, never
       rebuild
     - ``HydroFluxProfile`` reads the register. The legacy recomputation
       from cell centres (``hydro_flux.cpp:56-63``) goes. User callbacks
       (``load_user_output_data.cpp:19``) get a read-only register view, so
       ``*_flux`` outputs need not rebuild fluxes. L-2 answered yes
       (section 11.1).
   * - 5
     - shared face: two entries, one reconciliation rule; rotation owned by
       coord
     - The domain-boundary transfer has one entry per face. Lane O applies
       no rotation and reads scalar sums only. L-3: panel seams are not in
       ``boundary(dim)``; immersed-solid faces pending R (section 11.1).
   * - 6
     - one device-side monitor (mass, species, E+PE, AAM, scalars); no host
       reads in a segment
     - R owns the monitor and the domain totals (R:89-103). The rows are
       ``row_names()``: conserved, then ``e+pe`` and ``aam``, with scalars
       per instance (R:93, R:34, R:41). X reads only the retry flags at step
       end (``X.rst:165-173`` at ``bac201d1``). O's ``observe`` accumulates
       on the device with no host read per step. O's ``flush`` makes the
       cross-rank SUM and the host read once per output interval (section 4,
       "When").
   * - 7
     - every fixer and floor registers what it added
     - **Adopted.** It replaces A2. Each source is
       ``RegisterMonitor::amounts(name)`` (R:96) with ``kind(name)`` in
       {``cell_source``, ``floor``, ``fixer``} (R:25-31, R:95).
       ``aam_closure`` is A's ``AxialClosureFixer`` (R:104), which posts
       ``AmountKind::fixer`` (R:111-112).
   * - 8
     - stated dtype under FP32; fixed storage a graph can capture
     - Step sums and amounts are FP64, and FP32 on MPS, which has no FP64
       (R.rst:453, 644). Every monitor value O reads is FP64 there (R:96-98,
       R:101); counters are int64 (R:100). O casts nothing on CUDA or CPU.
       On MPS, O widens to double on the host after the read, and the
       residual tolerance is FP32-level (section 6). L-4 is closed.
   * - 9
     - no added full-field copy per stage; memory and cost stated
     - Lane O adds no per-stage work and no persistent or graph-visible
       allocation. Per accepted step it adds a few reductions of the
       (nrows,) sums and one (nvar, nc1+1) profile sum. *Estimate*; not
       measured.
   * - 10
     - the register is rebuilt, not written to restart
     - **Adopted.** It replaces the first draft's "derived" assumption.
       ``restart.hpp`` writes no register or monitor state. R: "Nothing is
       written to restart" (R.rst:467).

Assumption A3 of the first draft (output outside captured segments, ``dt``
as a device value) is not a register requirement. It moves to section 7 and
question X-1/X-2. A4 (the budget rows) is now requirement 6. A5 (lane
letters) is in 11.3 and 11.4.

5. Invariants and gates
-----------------------

The identity: for every budget row,
``residual = delta domain_total + sum_d boundary(d) - sum amounts`` over
the kinds ``cell_source``, ``floor`` and ``fixer`` (R.rst:550-554;
section 3.2). With
FP64 accumulation, ``|residual|`` should be at round-off relative to
``total``. The tolerance (proposed ``1e-12 * |total|``, *estimate*) is set
by G0 baselines, not here.

**Every gate below is a specification.** None is written and none has been
run in this round. Each states its identity, the cases it would run on, and
whether it is bitwise or within a tolerance. The PRs that would carry them
are planned in section 10 and not started.

.. list-table::
   :header-rows: 1
   :widths: 8 46 30 16

   * - gate
     - test
     - for which change
     - kind
   * - G-1
     - All 25 ``examples/*.yaml``, 3 steps, every written file compared,
       against the base (the #356 procedure)
     - O1-O3, O5, O7 (no numerical change)
     - bitwise
   * - G-2
     - **Combiner parity.** The same part files are merged by the pin's
       ``mppnccombine`` and by ``combine_parts``, and the merged files are
       compared. Cases listed below.
     - O6a, before O6b deletes ``mppnccombine.cpp``
     - bitwise data, exact metadata
   * - G-3
     - The generated option structs have the members and defaults of the
       pin; each module's deck parse equals the old parse on the in-tree
       decks
     - O4
     - exact
   * - G-4
     - Restart round trip (FP32/FP64, anomaly, 4-block bundle, #367
       mismatches still refused); a truncated bundle now throws. Budget
       cases: a deck with a budget output resumes its interval from
       ``BudgetCarry``; a file without a carry (no budget output, or a
       pin-era file) starts a new interval at the first ``observe``
     - O7
     - bitwise, plus a RED test for the throw
   * - G-5
     - Budget residual on a closed box (no boundary flux) and on a run with
       every fixer active, eager FP64 and FP32
     - O8
     - tolerance (G0)
   * - G-6
     - Ported ``test_rectify`` on ``bc/flip_zero_impl.h<int>``
     - O2
     - exact

**Gate G-2 in detail (specification).** The planned gate PR (O6a) would add ``combine_parts`` next to
``mppnccombine`` and a test. For each case the test would run with
``combine: false``, so the part files stay. It would then merge the same
parts twice: once with ``mppnccombine(-r ...)`` exactly as
``combine_netcdf.cpp:64-73`` calls it, and once with ``combine_parts``.

Cases, all at the pin:

* **One rank, several blocks, Cartesian.**

  * ``examples/straka_mesh2.yaml``: ``nb2: 2``, ``blocks_per_process: 2``
    (``:8-10``); netcdf at ``:52``.
  * ``examples/shallow_xy_mesh4.yaml``: 4 blocks on one rank (``:8-10``);
    netcdf at ``:40``. Already run by ``test_shallow_xy_decomp``
    (``tests/run_shallow_xy_decomp.py:99``).

* **One rank, cubed sphere.** ``examples/shallow_splash_mesh6.yaml``: 6
  panels, ``blocks_per_process: 6`` (``:20-22``); netcdf at ``:54``
  (``run_shallow_splash_decomp.py:93``).
* **Several ranks.**

  * ``examples/shallow_xy_proc2_mesh2.yaml``: 2 ranks × 2 blocks
    (``:8-10``); netcdf at ``:40``
    (``run_shallow_xy_decomp.py:100``).
  * ``examples/shallow_splash_proc2_mesh3.yaml``: 2 ranks × 3 panels
    (``:20-22``); netcdf at ``:54`` (``run_shallow_splash_decomp.py:96-97``).
  * ``examples/straka_proc2.yaml``: 2 ranks × 1 block (``:8-10``); netcdf
    at ``:52``.
  * ``test_output_barrier``, 2 ranks (``tests/CMakeLists.txt:215``).

* **Output options.**

  * ``straka_mesh2.yaml`` again with ``include_ghost_zones: true``,
    ``double_precision: true`` and an ``x2_slice``.
  * The slice cases of ``tests/test_user_output.cpp``, which call
    ``combine_blocks`` at ``:427``.

  These cover the #350 matrix (ghost zones, double precision, slices,
  6-panel combine).

**What G-2 compares, and how strictly:**

* **Data, bitwise.** Every variable's values, dtype and shape:
  ``nccmp -d -f`` with no tolerance, or a byte comparison of each
  variable's array.
* **Metadata, exact.** Dimension names and lengths, including the
  unlimited ``time``; the variable set and order; every variable and
  global attribute (name, type, value, including ``NumFilesInSet`` and
  ``domain_decomposition``); the file format (NETCDF4, as ``nccreate`` at
  ``mppnccombine.cpp:195`` makes it). ``ncdump -h`` text must be
  identical.
* **File bytes: recorded, not gating.** Both writers produce NETCDF4/HDF5
  containers, and HDF5's internal layout (object addresses, B-tree and
  heap placement) can differ for identical content. So a ``cmp`` of the
  whole file is reported but does not decide the gate. **Hypothesis**: with
  the same definition order and fill mode (``NC_NOFILL``,
  ``mppnccombine.cpp:200``) the files may also be byte-identical; not
  tested.

The gate passes only if every case passes. ``mppnccombine.cpp`` is deleted
only after that (O6b).

**Hypotheses the gates test:**

* ``combine_parts`` is complete for snapy's files (G-2).
* The two ``flip_zero`` implementations agree (G-6).

**Hypotheses not covered by a gate:**

* The ``strtok`` race: two blocks of one process writing at once, which
  ``combine_on_root``'s mutex (``writer_core.cpp:81-93``) anticipates. The
  core removes ``strtok`` either way.
* Reading restart tensors from a stream.
* That sedimentation's settling flux is posted as a register increment
  (note 16).

6. FP32 contract
----------------

* **Gridded output.** It reads totals: ``total_u`` adds the FP64 reference
  (``load_hydro_output_data.cpp:38``; the cycle line at
  ``meshblock.cpp:1257``). Files stay ``NC_FLOAT`` by default and
  ``NC_DOUBLE`` with ``double_precision`` (``output_type.hpp:79-83``).
  pnetcdf stays float only (``pnetcdf.cpp:58-61``).
* **Budget.** Accumulation, totals and the file are all FP64. R holds every
  value O reads in FP64 (R:96-98, R:101), so ``observe`` casts nothing (Xi
  note 3a). The exception is MPS, which has no FP64: R's sums are FP32
  there (R.rst:453). O widens them to double after the read, and G-5's
  residual tolerance on MPS is FP32-level.
* **Restart.** #367's rules are unchanged: FP64 totals, anomaly bits in the
  run dtype, FP64 background and grid, and rejection of changed grids.
* **FP32 storage emulation.** Removed (settled by Cheng). Runs that never
  set ``SNAP_FP32_STORAGE`` are unchanged: with the variable unset, the
  emulation is off (``fp32_storage.cpp:92``, default ``"off"``) and
  ``round_storage`` returns at ``meshblock.cpp:623``. Real FP32 arithmetic
  (#361) is not affected.

7. CUDA-graph contract
----------------------

**Rule.** Every host read in lane O is in ``flush``, ``write`` or the
restart code. All of these run after the global accept and outside captured
segments (``V3_ROADMAP.md:101``: "Global CFL/retry decisions and output stay
outside graphs").

``dt`` stays a host double in the first release (Xi's final answer; the
device-``dt`` request at ``meshblock.cpp:1210`` is withdrawn). This breaks
no lane O requirement:

* requirement 6: there is no host read inside a segment, because every O
  call runs after the accept;
* requirement 8: O makes no persistent or graph-visible allocation (Xi
  note 5);
* requirement 10: nothing is restarted.

The only per-step work is ``Consumer::observe`` and
``GriddedWriter::accumulate_stats``. Both run on device, after the global
accept and outside captured segments. They take **the host ``dt`` of the
accepted step** (``double``; Xi note 4), as ``X.rst`` (``77c65d5a``, answer
X2) keeps it: a host double chosen at step start (``meshblock.cpp:773``).
On the first call (the initial output, before any step) ``dt <= 0`` and
``accumulate_stats`` accumulates nothing, as ``output_type.cpp:302-306``
does today. Nothing runs inside a graph,
so no value is baked in, and requirement 6's no-host-read rule holds. A
device ``dt`` comes only if X parameterizes graphs later.

X's step-end host read is only the MAX of the retry flags
(``meshblock.cpp:1566``; ``X.rst:165-173`` at ``bac201d1``). O's monitor
read is in ``flush``: the cross-rank SUM and one host read per output
interval, outside every captured segment. O adds none per step.

Mapping to the roadmap milestones:

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - G
     - lane O contribution
   * - G0
     - The host-read inventory of the output path:
       ``load_diag_output_data.cpp`` 14, ``netcdf.cpp`` 6, ``pnetcdf.cpp``
       6, ``restart.cpp`` 7, ``output_type.cpp`` 3,
       ``load_hydro_output_data.cpp`` 2 (``rg -n
       '\.item<|\.cpu\(\)|to\(torch::kCPU\)|CopyFromTensor|InitFromTensor'``).
       The budget series is the file format for the "budgets" baselines
       (``V3_ROADMAP.md:146``).
   * - G1
     - Consumers are the "persistent device diagnostics" whose aggregation
       is "scheduled outside captured segments" (``:169-171``). The restart
       reader triggers "checkpoint/restart invalidation" (``:172``); the
       caller is question X-3.
   * - G2
     - The ``species/<n>`` and ``energy`` rows are the "water/energy
       budgets" (``:208``).
   * - G3
     - "Keep fallback counters, clamp meters ... gravity work correct on
       every replay" (``:224``): R's amounts and counters (for example
       ``vic.mass_clamped``) are read only in ``flush``.
   * - G4
     - ``combine_on_root`` barriers and the budget collective are outside
       segments. The budget series is the evidence for
       "decomposition-invariance budgets" (``:254``).
   * - G5
     - "Establish FP32 budgets against eager FP64 ... angular momentum"
       (``:273-275``): ``aam`` row, FP64 accumulation.
   * - G6
     - "Preserve global angular-momentum/gravity-work correction ordering"
       (``:287``): the ``aam_closure`` channel is read after the closure.
       Cubed-sphere combine uses the 3 x 2 panel layout
       (``writer_core.cpp:68-76``).
   * - G7
     - The generated options docs carry the execution-selection keys.
       Restart behaviour is documented with ``restart.hpp`` (``:305``).
       Output cadence is held fixed in the benchmarks (``:307``).

8. Cross-repository interface changes and merge order
-----------------------------------------------------

* **kintera**: no change required. Optional: a stream overload of
  ``load_tensors``, so the bundle reader does not extract each block to a
  temporary file (``read_restart_file.cpp:177-216``). **Hypothesis**: kintera
  does not have one today; I did not check.
* **pyharp, pydisort, commux**: none.
* **snapy Python API**:

  * ``OutputOptions.super_resolution`` is removed.
  * The option bindings and stubs become generated.
  * ``load_restart`` keeps its name and binds ``read_restart``.

8.1 Files and blocks moved to other owners (acceptance requested)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Line counts are ``wc -l`` at the pin. Moved code keeps its content; only its
directory or file changes. How the moved lines are counted between lanes X
and O is settled in the lane X thread and is not decided here.

Rows touching the mesh owner (lane X, Xi). **Every row below is agreed with
Xi** (lane X thread, rows 1-5) and is booked once. The last column says
where each row is booked.

* X deletes ``meshblock.cpp:203-230`` (item sv), ``meshblock_options.cpp:65-86``
  (item j) and the FP32 sites.
* The ``signal_handler`` move counts in neither lane.
* The restart-check and printer rows go **to O**, and X counts none of
  their lines.

O's replacement code is inside O's estimates.

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - file or block
     - lines
     - change and reason
   * - ``src/utils/signal_handler.cpp``
     - 146
     - Move to ``src/mesh``. Its only includers are ``mesh.cpp:28`` and
       ``meshblock.cpp:24``, and it takes ``MeshBlockImpl``
       (``signal_handler.hpp:45``). **Counts in neither lane**: a move, not
       a cut.
   * - ``src/utils/signal_handler.hpp``
     - 67
     - Same as above.
   * - ``meshblock.cpp:203-230`` (``validate_slice``)
     - 28
     - **X deletes it (item sv).** It repeats the other two checks. The
       generated output option checks (section 3.3) carry **both** checks,
       once each:

       * the sum check: slice against sum (``output_type.cpp:90-98``; also
         ``meshblock.cpp:205-209``);
       * the bounds and order checks: slice inside the mesh, strictly
         increasing (``meshblock_options.cpp:65-86``, X item j; also
         ``meshblock.cpp:210-220``).

       They run on the host when the options are read, before any step,
       so they do not touch ``dt``.
   * - ``meshblock.cpp:1623-1653`` (#367 restart metadata checks)
     - 31
     - **To O; X counts none of these lines.** They move into
       ``check_restart_metadata`` (``restart.hpp``), next to the file
       contract, and ``_init_from_restart`` calls it. The behaviour and
       messages are unchanged.
   * - ``meshblock.cpp:1226-1339`` (``print_cycle_diagnostics``; the line
       is produced at ``:1232-1338``) and ``mesh/cycle_diagnostics.hpp``
       (16)
     - 112 + 16
     - **To O, except ``:1336-1337``; X counts none of these lines.** They
       become the printer over ``BudgetConsumer``'s totals (section 3.2),
       which merges with O8, after lane R's monitor. The printed content is
       unchanged.

       * ``:1336-1337`` (``fixgrav``) is X's, inside the gravity-work row
         (83), so this row is 114 − 2 = 112 lines plus the header's 16.
       * The ``print_cycle_info`` wrapper (``:1341-1346``) stays in
         ``src/mesh``.
   * - ``meshblock_options.cpp:65-86`` (``check_slice``)
     - 22
     - **X deletes it (item j).** The generated checks carry its bounds and
       order checks (row above).
   * - ``meshblock.cpp:1210`` (``AccumulateStats`` call)
     - 0
     - **No change requested. The earlier request for a device ``dt`` is
       withdrawn** (Xi refused it for the first release). ``dt`` stays a
       host double. ``AccumulateStats`` keeps the host difference
       ``current_time - stat_last_time_`` (``output_type.cpp:308``) and
       runs in ``make_outputs`` after the accept (``meshblock.cpp:1207-1224``),
       outside captured segments.
   * - FP32 emulation sites (joint row 1, agreed): ``meshblock.cpp:22``
       (include), ``:618`` (call), ``:620-635`` (blank + ``round_storage``
       621-634 + blank), ``meshblock.hpp:207-213`` (comment + declaration),
       ``mesh.cpp:403-405`` (call)
     - 28
     - **Delete; booked in X** (1 + 1 + 16 + 7 + 3, Xi's count). Lines X
       already counts in its own items are not counted again:
       ``meshblock.cpp:636-642`` (``round_storage_after_stage``),
       ``meshblock.hpp:214-215`` and ``mesh.cpp:356-358``. O books only
       ``src/utils/fp32_storage.*`` (175).

       Outside both lanes: ``python/csrc/pymesh.cpp:238-247, 331-343`` and
       ``python/snapy/mesh.pyi:175, 220-222`` (the ``round_storage``
       bindings), ``tests/test_fp32_storage.cpp`` (79),
       ``tests/test_anomaly_storage.py:396`` (the ``fp32`` arm) and
       ``tests/CMakeLists.txt:8, 359, 816``.
   * - Gravity-work fixer (joint row 2) and axial closure (joint row 3)
     - 83 / 33 (X books 29 new)
     - X's and A's rows, listed here only because the cycle range overlaps.
       ``meshblock.cpp:1336-1337`` (``fixgrav``) is inside X's 83 and is
       not counted again in the cycle range. The closure is
       ``meshblock.cpp:1009-1035`` (27) and ``meshblock.hpp:223-228`` (6):
       33 lines by ``wc -l``. X books 29 new, because 4 are already in
       X's items (Xi note 6).

To the implicit owner (lane I, Xinyue):

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - file or block
     - lines
     - change and reason
   * - ``src/math/lubksb.h``
     - 71
     - **Accepted by Xinyue (lane I)**, at its current path ``src/math/``.
       Lane I's 2,821 includes it (``I.rst`` rev 2, ``8c2ba226``).
       It counts in lane I and in neither lane's cut. Its only includer is
       ``implicit/forward_sweep_impl.h:14``.
       ``tests/test_lu_failure.cpp:2`` reaches it through
       ``forward_sweep_impl.h``, so the test needs no edit.
   * - ``src/math/ludcmp.h``
     - 95
     - Same; included at ``forward_sweep_impl.h:15``.
   * - ``implicit/implicit_dispatch.cu:15``
     - 1
     - Delete the unused ``#include <snap/utils/cuda_utils.h>``.
       ``checkCudaError`` is never called (section 2.1).

Moved to owners outside lanes X and I, listed for completeness:

* ``src/utils/fp64_store.hpp`` (85) to ``src/hydro``; its only includer is
  ``hydro.hpp:19``.
* ``src/input/command_line.{cpp,hpp}`` (130 + 56) to ``examples/``; its only
  callers are ``run_hydro.cpp`` and ``jupiter_evap_precip_1d.cpp``.
* ``python/csrc/snapy.cpp:10``: delete the unused ``parameter_input.hpp``
  include (section 9.1).

8.2 Merge order
~~~~~~~~~~~~~~~

1. O1-O7 (section 10) depend on no other lane.
2. O8 (consumers and budget) merges after lane R's register and monitor
   (Linfeng).
3. O8's ``observe`` call site merges with lane X's post-accept host phase
   (Xi).
4. Other lanes switch their option headers to the generator after O4, in
   any order.

9. Line budget
--------------

The baseline per file is ``git show 9e5de2d4:<path> | wc -l``.

* "proposed" is what stays in place, or the new file.
* "moved" is not cut.
* cut = pin − proposed − moved.

Proposed sizes are *estimates*. Revision 2 adds a reachability class to
every row (section 9.1) and applies Cheng's decisions of 2026-11-18. These
rows changed from revision 1:

* ``src/utils/fp32_storage.{cpp,hpp}``: 95 + 55 kept → 0 + 0. The
  emulation is removed.
* ``src/output/writer_core.cpp``: 420 → 220. The combiner moves to its own
  **addition row**, ``src/output/combine_parts.cpp`` (+200). The sum is
  unchanged.
* ``tools/snap_combine.cpp`` (+30, outside ``src``): removed. The wrapper is
  dropped (section 10.1).

**Joint rows with lane X: agreed with Xi and booked once** (lane X thread,
Xi's final answer, rows 1-5):

Each file is booked once (note 3 and note 4):

* Booked in **O**:

  * ``src/utils/fp32_storage.*`` (119 + 56 = 175), deleted;
  * ``src/utils/fp64_store.hpp`` (85), moved to ``src/hydro``, not cut;
  * ``src/utils/tensor_cast_cache.hpp`` (53), kept.

* Booked in **X**:

  * the FP32 emulation sites in ``src/mesh`` (28: ``meshblock.cpp:22, 618,
    620-635``; ``meshblock.hpp:207-213``; ``mesh.cpp:403-405``);
  * the gravity-work fixer (83, including ``meshblock.cpp:1336-1337``) and
    the axial closure (33 lines by ``wc -l``, 29 of them new to X,
    ``meshblock.cpp:1009-1035``,
    ``meshblock.hpp:223-228``);
  * ``meshblock.cpp:203-230`` (item sv) and ``meshblock_options.cpp:65-86``
    (item j);
  * all 1,008 lines of ``src/interface``: ``athena_arrays.hpp`` 669,
    ``init_from_tensor.hpp`` 199, ``stride_iterator.hpp`` 140.

* **To O, X counts none of these lines:**

  * ``meshblock.cpp:1623-1653`` (31) go to the restart module;
  * ``meshblock.cpp:1226-1339`` minus ``1336-1337`` (112) and
    ``mesh/cycle_diagnostics.hpp`` (16) go to the printer.

  These lines are outside O's 9,996 baseline (``src/mesh``). O absorbs them
  into its estimated ``restart.cpp`` (260) and ``budget.cpp`` (200), so they
  are a move into O, not an O cut, and O's numbers do not change.
  ``1336-1337`` stays in X's gravity row.
* **In neither lane:** the ``signal_handler.*`` move into ``src/mesh``
  (146 + 67 = 213). It is in O's baseline as ``src/utils``, in the "moved"
  column, so it is not cut.
* **``src/utils`` counts in O**, all 1,717 lines of its baseline.

X's 1,008 ``src/interface`` lines depend on O5: the writer without
``AthenaArray`` and the port of ``tests/test_eos_io_core.cpp``
(sections 3.1 and 10). With O5, X deletes 1,723 of 4,274 (40.3 %);
without it, 827 (19.3 %) (Xi's figures).

None of these rows changes an O number, so O's cut is unchanged by the
agreement.

**Baseline recheck (revision 2).** The per-directory sums at the pin,
counting ``.cpp``/``.hpp``/``.h``/``.cu``/``.cuh`` and leaving out
``CMakeLists.txt`` and ``*.cpp_``:

* ``src/output``: 4,487;
* ``src/input``: 2,041;
* ``src/utils``: 1,717;
* ``src/diagnostics``: 1,585;
* ``src/math``: 166.

The total is 9,996, the brief's number. As a cross-check, plain ``wc -l`` over
all files gives 8,411 for output, input, utils and math (4,487 + 2,041 +
1,717 + 166). It gives 1,785 for ``src/diagnostics``, because that count
includes its ``CMakeLists.txt`` and the two ``.cpp_`` files.

9.1 Reachability evidence for the three largest deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All commands were run on a worktree whose files outside ``docs/v3`` equal
the pin: ``git diff --stat 9e5de2d4 HEAD -- . ':!docs/v3'`` is empty.

**What the CMake lists compile at the pin** (read, not built).

* The top-level ``CMakeLists.txt:151`` adds only ``src``. The other two
  ``add_subdirectory`` calls, at ``:156`` and ``:161``, add ``tests`` and
  ``examples``.
* ``src/CMakeLists.txt:35-53`` globs ``bc coord eos eos/aneos hydro riemann
  recon mesh output scalar utils forcing implicit input sedimentation
  layout``, one directory per line, 37-52. ``output/*.cpp`` is at ``:45``
  and ``input/*.cpp`` at ``:50``.
* The CUDA glob is ``src/CMakeLists.txt:122-136``.
* Tests and examples compile only their own ``<name>.cpp`` against the
  library (``cmake/macros/macro_setup_test.cmake:9``,
  ``macro_setup_problem.cmake:9``).
* The Python extension compiles only ``python/csrc/*.cpp``
  (``setup.py:109``).

**(a) ``src/diagnostics`` (1,585 lines in the baseline; 1,785 with
``CMakeLists.txt`` and the ``.cpp_`` files): never compiled.**

* No glob names it (``src/CMakeLists.txt:35-53``, ``:122-136``).
* Its own ``src/diagnostics/CMakeLists.txt:4`` would glob ``*.cpp`` into an
  ``OBJECT`` library, but no ``add_subdirectory`` reaches it. The only three
  are at ``CMakeLists.txt:151,156,161``.
* ``setup.py:109`` takes only ``python/csrc``.

.. code-block:: text

   $ git grep -n -E 'DiagnosticsFactory|DiagnosticsContainer|DiagnosticsPtr|HydroFlux|HydroMean|
       V1Moments|EddyFlux|EddyKineticEnergy|SphericalAngularMomentum|ConvectiveHeatFlux|
       RadiativeFlux|Buoyancy|Tendency|LatentHeating|diagnostics\.hpp|diagnostics/' 9e5de2d4 \
       -- . ':!src/diagnostics' ':!external' ':!docs' | grep -v cycle_diagnostics
   (no output)
   $ git grep -n -w -E 'Diagnostics|Divergence|Curl|Anomaly' 9e5de2d4 -- . ':!src/diagnostics' ':!external' ':!docs'
   python/csrc/pyeos.cpp:21          // comment ("Anomaly storage ...")
   src/eos/equation_of_state.cpp:221 // comment
   src/hydro/hydro_forward.cpp:1302  log text "Divergence time (s)"
   tests/test_anomaly_storage.py:2   docstring
   tests/test_fp32_science.py:59,111 a local Python class named Diagnostics
   $ git grep -n -E 'stride_iterator|virtual_groups' 9e5de2d4 -- . ':!src/diagnostics' ':!external' ':!docs'
   (no output)
   $ git grep -n 'diagnostics/' 9e5de2d4 -- docs
   docs/doxygen/SUMMARY.md:114   (a doc listing)

Every hit is a comment, a log string, a docstring or an unrelated Python
class. None reaches ``src/diagnostics``. (``cycle_diagnostics`` is
``src/mesh/cycle_diagnostics.hpp``, a different, live file.) **Verdict: not
built, not included, not bound, not used by tests or decks. Dead.**

**(b) Legacy parser ``parameter_input.*`` (993) and ``io_wrapper.*`` (364),
plus the duplicate ``input/signal.*`` (187): compiled by the ``src`` glob, but no symbol is
referenced.**

These files **are compiled** into ``libsnap`` by the ``input/*.cpp`` glob
(``src/CMakeLists.txt:50``). The parser header **is included** in two
compiled files:

* ``python/csrc/snapy.cpp:10``, so it is compiled into the Python
  extension;
* ``src/output/hdf5.cpp:6``, outside that file's ``#ifdef HDF5OUTPUT``.

The search for every public name declared in ``parameter_input.hpp:22-115``
and ``io_wrapper.hpp:16-24``:

.. code-block:: text

   $ rg -n -w -e 'ParameterInput|ParameterInputImpl|InputBlock|InputLine|IOWrapper|IOWrapperFile|
       IOWrapperSizeT|GetOrAddReal|GetOrAddInteger|GetOrAddBoolean|GetOrAddString|LoadFromFile|
       ModifyFromCmdline|ParameterDump|RollbackNextTime|ForwardNextTime' \
       src python tests examples cmake CMakeLists.txt setup.py pyproject.toml \
       -g '!src/input/parameter_input.*' -g '!src/input/io_wrapper.*'
   src/turbulence/turbulence_model.{hpp,cpp}, k_epsilon_turbulence.cpp   (13 lines)
   src/diagnostics/diagnostics.hpp:16,61, diagnostics_factory.cpp:11,15
   src/output/output_formats.hpp:15,23,30,74
   src/input/command_line.cpp:122   // comment
   $ rg -n 'parameter_input\.hpp|io_wrapper\.hpp' (same scope, parser excluded)
   python/csrc/snapy.cpp:10   src/output/hdf5.cpp:6
   src/diagnostics/diagnostics_factory.cpp:5   (<athena/parameter_input.hpp>, another library)
   src/input/io_wrapper.cpp:16   (io_wrapper's own source)

Each hit explained:

* ``src/turbulence`` and ``src/diagnostics`` are not compiled. No glob names
  them, and they use Athena++'s ``ParameterInput``/``MeshBlock``, not
  snapy's ``ParameterInputImpl``.
* ``output_formats.hpp:15,23,30`` sit inside the ``/* ... */`` block at
  lines 11-31, and ``:74`` inside the one at lines 71-75.
* ``command_line.cpp:122`` is a comment.
* The two ``#include`` lines use no symbol. ``snapy.cpp`` binds
  ``load_restart`` (``:47``) and the modules, nothing from the parser.
  ``hdf5.cpp`` has an empty body.

For ``input/signal.*``:

.. code-block:: text

   $ rg -n -w 'Signal' src python tests examples -g '!src/input/signal.*'
   (comments only: signal_handler.cpp:112, signal_handler.hpp:8,35,44,49,50,64)
   $ rg -n 'Signal::|signal\.hpp|mysig_' src python tests examples cmake -g '!src/input/signal.*'
   (no output)

None of the three files does anything when the library loads.
``signal.cpp`` has only zero-initialised statics and a mutex
(``signal.cpp:16, 121-122``) and installs handlers only in a constructor
that nobody calls. ``parameter_input.cpp`` and ``io_wrapper.cpp`` define no
global objects (only ``using Real``, ``parameter_input.cpp:67``). **Verdict:
reachable only at build time, through two unused includes. No call, no
Python binding, no test, no deck. Deleting them requires deleting the two
include lines (``snapy.cpp:10`` and ``hdf5.cpp``, which is deleted anyway).**

If reviewers count "built and included" as reachable even when no symbol is
used, these 1,544 lines come out of the cut (see the tiers below).

**(c) GFDL combiner ``output/mppnccombine.cpp`` (1,014): LIVE.** It is not
dead and is not counted as dead.

* It is compiled by ``src/CMakeLists.txt:45`` (``output/*.cpp``).
* Its one caller, ``combine_netcdf.cpp:73``, runs for every netcdf output
  of a run with more than one part per frame (``combine_netcdf.cpp:61-76``),
  with ``combine`` defaulting to true (``output_type.hpp:85``).

.. code-block:: text

   $ rg -n 'mppnccombine|combine_blocks|CombineBlocks' --hidden -g '!.git' -g '!external/**' -g '!docs/v3/**' .
   src/output/combine_netcdf.cpp:17,73     declaration and the call
   src/output/combine_netcdf.cpp:21,47,67,79, netcdf.cpp:65,577   NetcdfOutput::combine_blocks
   src/output/output_type.hpp:183, output_formats.hpp:39,86,95, restart.cpp:126,
     combine_restart.cpp:60,84,108          (restart and base-class combine; not the GFDL code)
   tests/test_user_output.cpp:427           calls NetcdfOutput::combine_blocks directly
   $ rg -n -w 'process_file|process_vars|flush_decomp|print_debug|nc_type_to_str' src python tests \
       -g '!src/output/mppnccombine.cpp'
   (no output)

What reaches it:

* Tests: ``tests/test_user_output.cpp:427`` (registered at
  ``tests/CMakeLists.txt:158``) and ``test_output_barrier`` (registered at
  ``tests/CMakeLists.txt:215``, two ranks, "a block is missing from the
  combined file", ``test_output_barrier.cpp:81``).
* The decomposition tests ``run_shallow_splash_decomp.py`` and
  ``run_shallow_xy_decomp.py`` (``tests/CMakeLists.txt:719-754``).
* Multi-block decks, for example ``examples/straka_mesh2.yaml``
  (``nb2: 2``, netcdf at ``:52``), ``shallow_xy_mesh4.yaml`` (netcdf at
  ``:40``) and ``jupiter_gcm.yaml`` (netcdf at ``:119,122``).
* The Python ``NetcdfOutput.write_output_file`` binding
  (``pyoutput.cpp:70-80``).

No script calls an external ``mppnccombine``. ``pd-combine``
(``python/api/pd_combine.py``) uses ``ncrcat``, ``ncks`` and ``fitsmerge``,
not this code. **Verdict: live.** Cheng approved removing it
(2026-11-18). It is replaced by ``combine_parts`` (section 10.1), and the
deletion PR merges once gate G-2 (section 5) passes.

Side finding: ``pd_combine.py:72-74`` merges ``*.out?.?????.fits`` files
with ``bin/fitsmerge``, while the library cannot write FITS
(``FITSOutput`` has no definition). That Python helper is outside the
library and keeps working on any FITS files a user already has. The FITS
stub is removed (settled by Cheng).

9.2 Cut, by what supports it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Baseline 9,996 throughout. Xi's final joint-row agreement changes no O number,
so every figure below is unchanged by it.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - cut
     - rev 2 part 1 (``a5325be7``)
     - Cheng's removals (``22cd9512``)
     - combiner plan (this head)
     - depends on
   * - **settled** (unchanged by the note fixes)
     - 4,413 (44.1 %)
     - 5,347 (53.5 %)
     - **5,377 (53.8 %)**
     - 9.1, Cheng's decisions; G-2 before O6b
   * - design target (all rows)
     - 5,627 (56.3 %)
     - 5,777 (57.8 %)
     - **5,807 (58.1 %)**
     - plus G-6 (``flip_zero``, 350) and Cheng on super-resolution
       (``refine``, 80)
   * - strict reading (group (b), compiled by the ``src`` glob but
       unreferenced, 1,544, kept)
     - 2,869 (28.7 %)
     - 3,803 (38.0 %)
     - **3,833 (38.3 %)**
     - above the 30 % target
   * - ``src`` only (no lines outside ``src``)
     - 6,257 (62.6 %)
     - 6,407 (64.1 %)
     - **6,107 (61.1 %)**
     - note 6: ``src/options/*.yaml`` (300) is now counted in ``src``

What the settled cut contains:

* never-compiled code (a);
* code compiled by the ``src`` glob but unreferenced (b), and the other
  unreferenced utilities;
* the rewrites and merges;
* Cheng's approved removals: ``mppnccombine.cpp`` (−1,014, with the
  ``combine_parts.cpp`` addition of +200), the hdf5/fits stubs and
  ``fp32_storage.*`` (−175).

**Pure deletions** (files removed with no replacement code): **3,813**
lines.

* ``src/diagnostics``: 1,585;
* ``parameter_input.*``: 993;
* ``io_wrapper.*``: 364;
* ``input/signal.*``: 187;
* ``pull_neighbors.*``: 249;
* ``utils_dispatch.*``: 131;
* ``cuda_utils.*``: 22;
* ``hdf5.cpp``: 27;
* ``refine.*``: 80;
* ``fp32_storage.*``: 175.

Rev 1's 4,163 also counted 525 replaced lines (note 2).

Commands:

* ``python3 line_table.py`` (worker workspace), which reads
  ``git show 9e5de2d4:<path> | wc -l`` for every row;
* the per-directory baseline check in section 9.

Settled = target − ``flip_zero`` (350) − ``refine`` (80). Strict = settled
− 1,544.

The +30 from revision 2 part 2 comes from dropping the ``snap-combine``
wrapper. If every rewritten or new file is 50 % over its estimate, the
design target is 4,242 lines (42.4 %). Each row of the table below
carries its class, and ``O_lines.csv`` has the same column.

.. list-table::
   :header-rows: 1
   :widths: 34 7 7 7 7 9 29

   * - file
     - pin
     - proposed
     - moved
     - cut
     - action; class
     - note / deletion named
   * - ``src/diagnostics/anomaly.cpp``
     - 97
     - 0
     - 
     - 97
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/buoyancy.cpp``
     - 61
     - 0
     - 
     - 61
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/curl.cpp``
     - 118
     - 0
     - 
     - 118
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics.cpp``
     - 52
     - 0
     - 
     - 52
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics.hpp``
     - 246
     - 0
     - 
     - 246
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics_factory.cpp``
     - 52
     - 0
     - 
     - 52
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/divergence.cpp``
     - 84
     - 0
     - 
     - 84
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/hydro_flux.cpp``
     - 120
     - 0
     - 
     - 120
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/hydro_mean.cpp``
     - 100
     - 0
     - 
     - 100
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/convective_heat_flx.cpp``
     - 130
     - 0
     - 
     - 130
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/eddy_flux.cpp``
     - 125
     - 0
     - 
     - 125
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/eddy_kinetic_energy.cpp``
     - 89
     - 0
     - 
     - 89
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/latent_heating.cpp``
     - 112
     - 0
     - 
     - 112
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/spherical_angular_momentum.cpp``
     - 86
     - 0
     - 
     - 86
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/tendency.cpp``
     - 113
     - 0
     - 
     - 113
     - delete; never-compiled
     - Athena++/canoe; never compiled
   * - ``src/input/check_keys.cpp``
     - 33
     - 33
     - 
     - 0
     - keep; live, kept
     - called by generated parsers
   * - ``src/input/check_keys.hpp``
     - 21
     - 21
     - 
     - 0
     - keep; live, kept
     - 
   * - ``src/input/command_line.cpp``
     - 130
     - 0
     - 130
     - 0
     - move; live, moved
     - to examples/ (only callers: run_hydro.cpp, jupiter_evap_precip_1d.cpp)
   * - ``src/input/command_line.hpp``
     - 56
     - 0
     - 56
     - 0
     - move; live, moved
     - to examples/
   * - ``src/input/io_wrapper.cpp``
     - 305
     - 0
     - 
     - 305
     - delete; compiled by src glob, unreferenced
     - only reached from parameter_input
   * - ``src/input/io_wrapper.hpp``
     - 59
     - 0
     - 
     - 59
     - delete; compiled by src glob, unreferenced
     - 
   * - ``src/input/parameter_input.cpp``
     - 873
     - 0
     - 
     - 873
     - delete; compiled by src glob, unreferenced
     - legacy Athena++ parser; no caller
   * - ``src/input/parameter_input.hpp``
     - 120
     - 0
     - 
     - 120
     - delete; compiled by src glob, unreferenced
     - only an unused include in python/csrc/snapy.cpp:10 and hdf5.cpp:6
   * - ``src/input/read_restart_file.cpp``
     - 236
     - 0
     - 
     - 236
     - merge; live, merged
     - into output/restart.cpp
   * - ``src/input/read_restart_file.hpp``
     - 21
     - 0
     - 
     - 21
     - merge; live, merged
     - into restart.hpp
   * - ``src/input/signal.cpp``
     - 124
     - 0
     - 
     - 124
     - delete; compiled by src glob, unreferenced
     - older copy of utils/signal_handler; no includer
   * - ``src/input/signal.hpp``
     - 63
     - 0
     - 
     - 63
     - delete; compiled by src glob, unreferenced
     - 
   * - ``src/math/lubksb.h``
     - 71
     - 0
     - 71
     - 0
     - move; live, moved
     - to lane I (Xinyue accepted); in neither lane's cut; out of O's roll-up baseline
   * - ``src/math/ludcmp.h``
     - 95
     - 0
     - 95
     - 0
     - move; live, moved
     - to lane I (Xinyue accepted); in neither lane's cut; out of O's roll-up baseline
   * - ``src/output/combine_netcdf.cpp``
     - 87
     - 0
     - 
     - 87
     - merge; live, merged
     - into writer_core.cpp (combine step, no argv building)
   * - ``src/output/combine_restart.cpp``
     - 123
     - 0
     - 
     - 123
     - merge; live, merged
     - into restart.cpp (bundle writer)
   * - ``src/output/hdf5.cpp``
     - 27
     - 0
     - 
     - 27
     - delete; compiled by src glob, unreferenced; removal approved by Cheng
     - stub; body only under never-defined HDF5OUTPUT, signature does not match the header
   * - ``src/output/load_diag_output_data.cpp``
     - 338
     - 260
     - 
     - 78
     - rewrite; live, rewritten
     - field-diagnostic registry replaces ContainVariable chains
   * - ``src/output/load_hydro_output_data.cpp``
     - 186
     - 150
     - 
     - 36
     - rewrite; live, rewritten
     - appends Fields; names from the field registry
   * - ``src/output/load_scalar_output_data.cpp``
     - 64
     - 50
     - 
     - 14
     - rewrite; live, rewritten
     - appends Fields
   * - ``src/output/load_user_output_data.cpp``
     - 27
     - 20
     - 
     - 7
     - rewrite; live, rewritten
     - appends Fields
   * - ``src/output/mppnccombine.cpp``
     - 1014
     - 0
     - 
     - 1014
     - delete; LIVE, removal approved by Cheng; replaced by combine_parts.cpp (gate G-2 first)
     - GFDL tool; replaced by combine_parts in writer_core.cpp
   * - ``src/output/netcdf.cpp``
     - 583
     - 140
     - 
     - 443
     - rewrite; live, rewritten
     - netcdf Sink only
   * - ``src/output/netcdf_utils.hpp``
     - 29
     - 0
     - 
     - 29
     - merge; live, merged
     - sanitize_netcdf_name into writer_core.cpp
   * - ``src/output/output_formats.hpp``
     - 105
     - 0
     - 
     - 105
     - delete; live, folded
     - HDF5Output, FITSOutput, commented classes; Netcdf/PNetcdf classes become Sinks
   * - ``src/output/output_type.cpp``
     - 559
     - 230
     - 
     - 329
     - rewrite; live, rewritten
     - options parser generated; linked list and slice/sum loops replaced by torch ops
   * - ``src/output/output_type.hpp``
     - 219
     - 90
     - 
     - 129
     - rewrite; live, rewritten
     - options struct generated; GriddedWriter only
   * - ``src/output/output_utils.cpp``
     - 298
     - 120
     - 
     - 178
     - rewrite; live, rewritten
     - metadata of live fields only; pin lines here exclude :53-55 (row below)
   * - ``src/output/output_utils.cpp:53-55``
     - 3
     - 0
     - 
     - 3
     - delete; dead names (src/turbulence not compiled); deletion with src/turbulence (F6)
     - eps, tke, mut output names; deleted with src/turbulence (EF/F6, Cheng); O counts them
   * - ``src/output/output_utils.hpp``
     - 65
     - 30
     - 
     - 35
     - rewrite; live, rewritten
     - get_num_variables replaced by Field shape
   * - ``src/output/pnetcdf.cpp``
     - 431
     - 130
     - 
     - 301
     - rewrite; live, rewritten
     - pnetcdf Sink only
   * - ``src/output/restart.cpp``
     - 130
     - 260
     - 
     - -130
     - rewrite; live, rewritten
     - writer + bundle + reader in one file
   * - ``src/output/restart_metadata.hpp``
     - 13
     - 13
     - 
     - 0
     - keep; live, kept
     - #367 contract
   * - ``src/output/writer_core.cpp``
     - 138
     - 220
     - 
     - -82
     - rewrite; live, rewritten
     - Frame transform and packing; the combiner is its own addition row
   * - ``src/output/writer_core.hpp``
     - 48
     - 110
     - 
     - -62
     - rewrite; live, rewritten
     - Frame, Field, Sink, GriddedWriter
   * - ``src/utils/cuda_utils.cu``
     - 19
     - 0
     - 
     - 19
     - delete; compiled by src glob, unreferenced
     - checkCudaError has no caller
   * - ``src/utils/cuda_utils.h``
     - 3
     - 0
     - 
     - 3
     - delete; compiled by src glob, unreferenced
     - one unused include in implicit_dispatch.cu:15
   * - ``src/utils/flip_zero.cpp``
     - 270
     - 0
     - 
     - 270
     - delete; test-only, replaced (gate G-6)
     - int-only copy of bc/flip_zero_impl.h; test ported
   * - ``src/utils/flip_zero.h``
     - 80
     - 0
     - 
     - 80
     - delete; test-only, replaced (gate G-6)
     - 
   * - ``src/utils/fp32_storage.cpp``
     - 119
     - 0
     - 
     - 119
     - delete; LIVE feature, removal approved by Cheng; joint row agreed with Xi, booked once in O
     - FP32 storage emulation; removal approved by Cheng
   * - ``src/utils/fp32_storage.hpp``
     - 56
     - 0
     - 
     - 56
     - delete; LIVE feature, removal approved by Cheng; joint row agreed with Xi, booked once in O
     - same; the 28 emulation lines in src/mesh are X's (joint row 1, agreed with Xi)
   * - ``src/utils/fp64_store.hpp``
     - 85
     - 0
     - 85
     - 0
     - move; live, moved; joint row agreed with Xi, booked once in O
     - to src/hydro (one includer: hydro.hpp)
   * - ``src/utils/log.hpp``
     - 67
     - 67
     - 
     - 0
     - keep; live, kept
     - 11 includers
   * - ``src/utils/loops.cuh``
     - 150
     - 150
     - 
     - 0
     - keep; live, kept
     - 8 CUDA dispatch files
   * - ``src/utils/on_host.hpp``
     - 72
     - 72
     - 
     - 0
     - keep; live, kept
     - 3 dispatch files (MPS)
   * - ``src/utils/pull_neighbors.cpp``
     - 225
     - 0
     - 
     - 225
     - delete; compiled by src glob, unreferenced
     - only commented-out calls (equation_of_state.cpp:254-257)
   * - ``src/utils/pull_neighbors.hpp``
     - 24
     - 0
     - 
     - 24
     - delete; compiled by src glob, unreferenced
     - 
   * - ``src/utils/refine.cpp``
     - 69
     - 0
     - 
     - 69
     - delete; LIVE feature removed (FOR CHENG)
     - only caller: output super-resolution (deleted, FOR CHENG)
   * - ``src/utils/refine.hpp``
     - 11
     - 0
     - 
     - 11
     - delete; LIVE feature removed (FOR CHENG)
     - 
   * - ``src/utils/signal_handler.cpp``
     - 146
     - 0
     - 146
     - 0
     - move; live, moved; joint row agreed with Xi: moved to src/mesh, counts in neither lane
     - to src/mesh (callers mesh.cpp, meshblock.cpp)
   * - ``src/utils/signal_handler.hpp``
     - 67
     - 0
     - 67
     - 0
     - move; live, moved; joint row agreed with Xi: moved to src/mesh, counts in neither lane
     - to src/mesh
   * - ``src/utils/tensor_cast_cache.hpp``
     - 53
     - 53
     - 
     - 0
     - keep; live, kept; joint row agreed with Xi, booked once in O
     - 3 includers
   * - ``src/utils/utils_dispatch.cpp``
     - 26
     - 0
     - 
     - 26
     - delete; compiled by src glob, unreferenced
     - bdot_out has no caller
   * - ``src/utils/utils_dispatch.cu``
     - 92
     - 0
     - 
     - 92
     - delete; compiled by src glob, unreferenced
     - 
   * - ``src/utils/utils_dispatch.hpp``
     - 13
     - 0
     - 
     - 13
     - delete; compiled by src glob, unreferenced
     - 
   * - ``src/utils/vectorize.cpp``
     - 24
     - 0
     - 
     - 24
     - delete; live, folded
     - strtok split; replaced by a split in writer_core
   * - ``src/utils/vectorize.hpp``
     - 46
     - 0
     - 
     - 46
     - delete; live, folded
     - 
   * - ``src/output/combine_parts.cpp``
     - 
     - 200
     - 
     - -200
     - new
     - ADDITION: snapy-specific combiner replacing mppnccombine.cpp (-r form only; gate G-2)
   * - ``src/output/budget_series.cpp``
     - 
     - 120
     - 
     - -120
     - new
     - BudgetSeries netcdf writer
   * - ``src/output/budget_series.hpp``
     - 
     - 30
     - 
     - -30
     - new
     - 
   * - ``src/diagnostics/consumer.hpp``
     - 
     - 90
     - 
     - -90
     - new
     - Consumer, register/monitor views
   * - ``src/diagnostics/budget.cpp``
     - 
     - 200
     - 
     - -200
     - new
     - BudgetConsumer
   * - ``src/diagnostics/hydro_flux_profile.cpp``
     - 
     - 80
     - 
     - -80
     - new
     - HydroFluxProfile from the register (new file; the old hydro_flux.cpp row above is its deletion)
   * - ``tools/gen_options.py``
     - 
     - 300
     - 
     - -300
     - new, outside src
     - options generator
   * - ``src/options/*.yaml (lane S parsing: _pi aliases, cross-option checks)``
     - 
     - 10
     - 
     - -10
     - new
     - ADDITION for S's options parsing (accepted); S counts the deletion
   * - ``tools/gen_options.py (scale/alias column for S's _pi keys)``
     - 
     - 10
     - 
     - -10
     - new, outside src
     - ADDITION for S's options parsing (accepted)
   * - ``src/options/*.yaml``
     - 
     - 300
     - 
     - -300
     - new
     - options tables (data, in src), ~1.4 lines per option
   * - **baseline (71 files, 72 rows)**
     - **9996**
     - **3249**
     - **650**
     - **6097**
     - 
     - src only: 61.0% cut
   * - **including new files outside src**
     - 
     - **310**
     - 
     - **5787**
     - 
     - 57.9% cut; target 30% = 2,999

**Totals (rev 4).**

* Brief's baseline: 9,996 lines in 71 files, rechecked at the pin.
* Kept in place: 2,219. Moved out unchanged: 650.
* Added in ``src``: 1,030, all *estimates*. That is ``combine_parts.cpp``
  200, the option tables 300 + 10, and the rest.
* Added outside ``src``: 310 (``tools/gen_options.py`` 300 + 10).
* **Settled cut: 5,357 lines** (was 5,377). It moved by −20, the new lines
  for S's options rows (section 9.3).
* Design target: 5,787 (was 5,807). Strict: 3,813 (was 3,833). ``src``
  only: 6,097 (was 6,107).
* Not counted: 785 lines deleted outside the baseline. In lane X: all of
  ``src/interface`` (1,008, including ``stride_iterator.hpp`` 140) and the
  FP32 storage application in ``src/mesh`` (section 8.1).

**``output_utils.cpp:53-55`` with ``src/turbulence`` (EF/F6, rev 5).**
Cheng decided F6: ``src/turbulence`` is deleted (EF rev 3.4, ``b7bece9c``).
At the pin, ``git show 9e5de2d4:src/output/output_utils.cpp | sed -n 53,55p``
gives:

.. code-block:: text

   {"eps", "turbulent dissipation", "W/kg", "CCC"},
   {"tke", "turbulent kinetic energy", "J/kg", "CCC"},
   {"mut", "dynamic turbulent viscosity", "kg/(m.s)", "CCC"},

No live code writes these names: ``src/turbulence`` is outside every glob
(section 9.1). Lane O owns the file, so it counts the 3 lines. The line
table now has a separate row, ``output_utils.cpp:53-55``: 3 → 0, cut 3. The
rest of the file, 298 → 120, has cut 178.

**The cut does not move** (choice and reason). The rewrite's 120-line
estimate already kept only live field names (section 3.1, "metadata of live
fields only"), so these 3 lines were already inside the file's 181-line
reduction. Adding them on top would count them twice. Arithmetic:

* before: 301 → 120, cut 181;
* now: (298 → 120, cut 178) + (3 → 0, cut 3) = cut 181.

Settled 5,357, target 5,787, strict 3,813 and ``src`` only 6,097 are
unchanged. The roll-up stays 9,830 → 4,473 (9,830 − 5,357 = 4,473; 54.5 %).

**``src/math`` (Xinyue's lane I answer).** Lane I takes ``lubksb.h`` and
``ludcmp.h`` at their path, ``src/math/``. Lane I's budget is 2,821 to
2,033, and **lane I's 2,821 includes these 166 lines** (Xinyue, lane I
thread; ``I.rst`` rev 2 at XinyueWang05/snapy ``8c2ba226``). So the roll-up
below, 9,830 to 4,473, counts them once, in lane I. **Closed.**

* At the pin, ``git show 9e5de2d4:src/math/lubksb.h | wc -l`` gives 71 and
  ``.../ludcmp.h`` gives 95: 166 lines.
* ``src/math`` is **in neither lane's cut**. O never cut it (its rows are in
  the "moved" column), and lane I keeps the files.
* O's brief baseline (9,996) and O's after figure (4,619 in rev 3) **did
  include** the 166 lines. O's cut (5,377 in rev 3) did not.
* So that the 166 lines are counted once, in lane I, **O's roll-up drops
  them from both baseline and after**:

  * baseline 9,996 − 166 = **9,830**;
  * settled after 9,830 − 5,357 = **4,473**;
  * settled cut 5,357 = **54.5 %**.

* Against the brief's 9,996 the same cut is 53.6 %. The after figure would
  then be 4,639, but that double counts 166 lines with lane I.

.. list-table:: Rev 4 cut. Commands: ``python3 line_table.py``; the
   ``src/math`` ``wc -l`` above.
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - cut
     - lines
     - after (9,830 base)
     - % of 9,830
     - % of 9,996
   * - settled
     - 5,357
     - 4,473
     - 54.5
     - 53.6
   * - design target
     - 5,787
     - 4,043
     - 58.9
     - 57.9
   * - strict
     - 3,813
     - 6,017
     - 38.8
     - 38.1
   * - ``src`` only
     - 6,097
     - 3,733
     - 62.0
     - 61.0

9.3 Lane S's options YAML parsing (accepted into O's schema)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision: accepted** (Jiachen, rev 4). Lane O owns the generated options
schema. Lane X declined move A for that reason (``S.rst`` section 9 at
XinyueWang05/snapy ``d03e48880dc28b33eca6c62ff6ad8ad162bdcdb2``, move A,
``S.rst:1323-1330``). Options rows belong in the one generated table, not
in another hand-written parser. **S counts the deletion; O counts only its
new lines.**

S's move A names 226 lines. Verified at the pin with
``git show 9e5de2d4:<file> | sed -n <a>,<b>p | wc -l``:

* ``coord/coordinate.cpp:62-240``: 179 lines;
* ``layout/layout.cpp:249-272``: 24 lines;
* ``bc/internal_boundary.cpp:14-36``: 23 lines.

Read line by line, 127 of them are options parsing, which O's schema
takes. 99 are grid or layout logic, which stays with S.

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - range at the pin
     - lines
     - content and owner
   * - ``coordinate.cpp:62-114``
     - 53
     - **O schema.** Keys ``geometry/type`` and ``geometry/bounds/*``,
       including the ``x2min_pi``-style keys scaled by pi, read into the
       global bounds.
   * - ``coordinate.cpp:125-149``
     - 25
     - **O schema.** Keys ``geometry/cells/nx1..3``, ``x1rat`` and ``x1f``,
       and the cross-option checks (x1rat xor x1f, x1f size ``nx1+1``,
       x1f end points against the bounds) as ``checks`` rows.
   * - ``coordinate.cpp:213-214``
     - 2
     - **O schema.** ``cells/nghost``, ``cells/interp_order``.
   * - ``layout.cpp:249-272``
     - 24
     - **O schema.** ``distribute/*`` keys, plus the ``BACKEND`` and
       ``DEVICE`` env rows (``:265-266``).
   * - ``internal_boundary.cpp:14-36``
     - 23
     - **O schema.** ``boundary-condition/internal/*`` keys.
   * - ``coordinate.cpp:115-124``
     - 10
     - **Stays with S.** A grid rule: ``bounds`` without ``cells`` sets one
       cell per axis and the block bounds.
   * - ``coordinate.cpp:150-178``
     - 29
     - **Stays with S.** Stretched-x1 face construction from ``x1rat``, the
       monotonicity and width-ratio checks, and ``x1f_interior``.
   * - ``coordinate.cpp:179-212``
     - 34
     - **Stays with S.** Builds a ``Layout`` and checks block divisibility.
   * - ``coordinate.cpp:215-240``
     - 26
     - **Stays with S.** ``repartition`` and the per-block ghost checks
       (#313).
   * - **total**
     - **226**
     - 127 to O's schema, 99 stay with S

**O's new lines: +20** (*estimate*). The options themselves are already
among the 213 ``ADD_ARG`` entries that O's 300-line table estimate covers
(``coordinate.hpp`` 22, ``layout.hpp`` 27, ``internal_boundary.hpp`` 3). O
adds only:

* about 10 lines in ``src/options/*.yaml``: the pi-scaled alias rows and
  the cross-option checks;
* about 10 lines in ``tools/gen_options.py``: an ``alias``/scale column.

These are the two addition rows in the table above. Whether S deletes the
127 lines or all 226 is S's count; O books none of the deletion.
**Final.** Lane S's final split (Jiheng, lane S thread) is the table above:
127 lines to O and 99 stay in S, each range re-verified with ``wc -l`` at
the pin. O maps to schema rows exactly the 127 lines S deletes. Under
RULING V3-1, S counts the deletion and O counts only its +20 new lines.

10. Migration
-------------

**Lane X's precondition for deleting ``src/interface``** (Xi note 8) is one
grep:

``rg -n 'athena_arrays|init_from_tensor' src tests python``

After O1, O5 and X's own deletion, its only hit is
``src/turbulence/turbulence_model.hpp:9``, which is outside the build. At
the pin it has 8 hits: ``tests/test_eos_io_core.cpp:7``,
``src/diagnostics/diagnostics.hpp:8``,
``src/diagnostics/todo/convective_heat_flx.cpp:8``,
``src/output/output_type.hpp:22``, ``src/output/output_utils.hpp:10``,
``src/interface/athena_arrays.hpp:3, 669`` and that one.

The PR sequence below is a **plan**. No PR is started in this round, and
nothing in it has been written, built or run. The combiner gate PR (O6a)
comes before the deletion PR (O6b). "Bitwise" means every state and every written
file is byte-identical to the base (gate G-1). No lane O change touches
arithmetic on the state, so every PR is bitwise except O8, which adds a new
file.

.. list-table::
   :header-rows: 1
   :widths: 6 50 18 26

   * - PR
     - change
     - result
     - why
   * - O1
     - Delete never-compiled code: ``src/diagnostics``, the ``.cpp_``
       files, ``z.junk``, ``stride_iterator.hpp``. The last one's only
       includers are in ``src/diagnostics``; deleting it here closes lane
       X's row for it (X deletes only the other two ``src/interface``
       files, after O5).
     - bitwise
     - no compiled file changes
   * - O2
     - Delete compiled but uncalled code: ``parameter_input``,
       ``io_wrapper``, ``input/signal``, ``pull_neighbors``,
       ``utils_dispatch``, ``cuda_utils``, the hdf5/fits stubs; port
       ``test_rectify``
     - bitwise
     - no caller (section 2.1)
   * - O2b
     - Remove the FP32 storage emulation: ``src/utils/fp32_storage.*``
       (O, 175) and the agreed X sites (X, 28; section 8.1), the
       ``round_storage`` bindings, ``test_fp32_storage`` and the
       ``fp32`` arm of ``test_anomaly_storage.py``
     - bitwise for every run that does not set ``SNAP_FP32_STORAGE``
     - with the variable unset the emulation is off
       (``fp32_storage.cpp:92``) and ``round_storage`` returns at
       ``meshblock.cpp:623``; runs that set it lose the emulation (approved
       by Cheng)
   * - O3
     - Moves: ``command_line`` to ``examples/``, ``fp64_store``,
       ``signal_handler``, ``math``
     - bitwise
     - include paths only
   * - O4
     - Options table and generator; the output and hydro modules first;
       the env rows; the covariance warnings deleted. Then lane S's options
       parsing (section 9.3): the ``geometry``, ``distribute`` and
       ``boundary-condition/internal`` rows and the checks, with the
       ``alias``/scale column for the pi-scaled keys. S deletes its parsing
       lines in its own PR after this one.
     - bitwise
     - same members and defaults (G-3)
   * - O5
     - Writer core without ``AthenaArray`` (section 3.1), netcdf and
       pnetcdf sinks; ``vectorize`` and super-resolution removed. Port
       ``tests/test_eos_io_core.cpp``: its ``AthenaArray`` case
       (``:33-53``, the float32 → float64 copy of #350) becomes the same
       check on ``Frame::to_host()``; the two ``MoistEOS`` cases
       (``:13-31``) are kept as they are. Port ``TestOutputType``
       (``tests/test_user_output.cpp:39-100``, which reads ``pdata->data``
       dims and elements) to ``Frame``/``Field`` accessors. Replace the sink
       element reads (``netcdf.cpp:496-558``, ``pnetcdf.cpp:362-413``, 8
       sites each) and the ``get_num_variables(…, pdata->data)`` calls
       (section 3.1).
     - bitwise files
     - same values cast once, same layout (G-1). **Dependency:** lane X's
       deletion of all 1,008 ``src/interface`` lines needs O5 first. That
       means no ``AthenaArray`` left in O (``output_type.hpp:22, 99``,
       ``output_utils.hpp:10, 47``, ``output_type.cpp:388, 399, 467,
       521, 528, 535``, ``load_diag_output_data.cpp:120-332``) and
       ``tests/test_eos_io_core.cpp:33-53`` ported (X: 1,723 of 4,274
       with O5, 827 without).
   * - O6a
     - **Gate PR.** Add ``src/output/combine_parts.cpp`` and the G-2
       parity test. ``combine_blocks`` still calls ``mppnccombine``.
     - no output change
     - the new code is not called by runs yet
   * - O6b
     - **Deletion PR**, after O6a's G-2 passes. ``combine_netcdf.cpp:73``
       calls ``combine_parts``; ``mppnccombine.cpp`` (1,014) and its
       declaration (``combine_netcdf.cpp:17``) are deleted
     - bitwise data, exact metadata
     - G-2
   * - O7
     - Restart merge; the reader throws
     - bitwise
     - same files (G-4)
   * - O8
     - Consumers, ``BudgetSeries``, the cycle line from the budget totals.
       ``BudgetCarry`` is written only when a budget output is requested; a
       file without a carry starts a new interval at the first ``observe``
       (G-4)
     - new file; cycle line bitwise if the summation order is kept
     - if the reduction order changes, only the last digits of the
       *printed* totals change, never the state

10.1 Is a replacement combiner still needed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision (approved by Cheng; settled): yes.** ``combine_parts``
(``src/output/combine_parts.cpp``, about 200 lines, *estimate*) replaces
``mppnccombine.cpp``. In this round it exists only as an interface
(declaration in ``writer_core.hpp``); it is not written.

* **Takes:** the output path and the part paths of the ``-r <outfile>
  <infiles...>`` form that ``combine_netcdf.cpp:64-73`` builds.
* **Writes:** one merged NETCDF4 file at the output path, then removes the
  parts.
* **Returns:** 0, or a netcdf error code. ``combine_netcdf.cpp:77-83``
  turns a nonzero code into the existing exception.

In-run combining stays, and ``combine`` keeps its
default of true (``src/output/output_type.hpp:85``:
``ADD_ARG(bool, combine) = true;``). Users see no change: one merged netcdf
file per output frame, ``<basename>.out<n>.<kkkkk>.nc``, as today.

**Scope: only what snapy calls.** At the pin the only call is
``mppnccombine(argc, argv)`` at ``src/output/combine_netcdf.cpp:73``
(``rg -n 'mppnccombine\(' src python tests examples cmake`` gives that line,
the declaration at ``:17`` and the definition at ``mppnccombine.cpp:94``).
That call always uses the form ``-r <outfile> <infiles...>``
(``combine_netcdf.cpp:64-71``). ``-r`` tells the tool to remove the
decomposed part files after a successful merge (``mppnccombine.cpp``
``usage``). ``combine_parts`` covers that form only: merge the parts into
the output, then delete the parts on success.

What it does not touch:

* the single-part rename path (``combine_netcdf.cpp:61-62``,
  ``std::rename``);
* the PnetCDF writer, whose ``combine_blocks`` is empty
  (``output_formats.hpp:95``) because it writes one collective file;
* ``combine_restart.cpp``, which bundles restart parts with its own format
  (``combine_restart.cpp:19-58``).

The other ``mppnccombine`` options are not reproduced, because snapy never
passes them: ``-v``, ``-a``, ``-n``, ``-e``, ``-h``, ``-m`` and ``-64``
(``usage``, ``mppnccombine.cpp:336-399``).

**Why a replacement is needed.** Cheng approved removing the GFDL code,
not the behaviour. Without a combiner, every multi-block netcdf output
would become one file per block. That breaks:

* the default above;
* ``test_output_barrier`` (``test_output_barrier.cpp:81``) and
  ``test_user_output.cpp:427``;
* the decomposition tests (``tests/CMakeLists.txt:717-754``);
* the multi-block decks of section 5.

PnetCDF cannot take over: it needs one block per process
(``pnetcdf.cpp:79-83``), writes float only (``:58-61``), takes no slices
(``:51-53``) and exists only in ``PNETCDF=ON`` builds
(``meshblock.cpp:236-241``).

**The ``snap-combine`` wrapper is dropped.** Revision 2 earlier proposed a
30-line command-line wrapper for ``combine: false`` parts. The rule was to
keep it only if it replaced a tool that ``combine: false`` users have at the
pin. They have none:

* the in-tree ``mppnccombine`` is a library function with no executable
  target. ``cmake/macros`` and ``tests``/``examples`` ``CMakeLists`` build
  only ``<name>.cpp`` targets, and ``setup.py:109`` builds only
  ``python/csrc``.
* It has no Python binding (``rg -n mppnccombine python`` is empty).
* ``pd-combine`` does not join blocks; it uses ``ncrcat``/``ncks``
  (``python/api/pd_combine.py``).

Tianhao's report says the same: joining ``combine: false`` parts "needs an
external ``mppnccombine``" (``14-parallelism-io/_netcdf.qmd:106`` at
``919936da``). So the wrapper and its per-part instructions are removed
from the design, and its 30 lines leave the budget. ``combine: false``
users keep exactly what they have today: their own external tool, for
example GFDL FRE-NCtools' ``mppnccombine -r <out.nc> <parts...>``. Nothing
in snapy changes for them.

**Options and paths removed:**

* output type ``hdf5`` (never registered);
* the output key ``super-resolution`` and its Python binding;
* the warnings for ``SNAP_FLUX_COVARIANCE`` and ``SNAP_X1_MASS_COVARIANCE``;
* the ``SNAP_FP32_STORAGE`` and ``SNAP_FP32_STORAGE_ROWS`` variables and
  the Python methods ``MeshBlock.round_storage`` / ``Mesh.round_storage``
  (``python/csrc/pymesh.cpp:238-247, 331-343``; no caller in ``examples``,
  ``tests`` or ``python`` scripts: ``rg -n round_storage examples tests
  python/*.py python/api`` is empty), settled by Cheng;
* the Athena++ input format;
* the built-in ``mppnccombine`` (settled by Cheng; replaced by
  ``combine_parts``, section 10.1). No command-line wrapper is added;
* the duplicate slice validation at ``meshblock.cpp:203-230``.

11. Open questions to other lanes
---------------------------------

**11.1 Lane R** (frozen header rev 4.2, R.rst rev 2.2, happysky19/snapy
``4c59162a``). Requirements 3, 7 and 10 answered the earlier questions on
step integration, per-fixer amounts (the axial closure is a fixer) and
restart. The frozen header closes the rest. L-2 is answered (Tianhao), and
L-3 is pending R:

L-1. **Closed**: rows by ``row_names()`` (R:93; R.rst:528-542); per unit
area in coord's frame (R:4-5).

L-2. **Answered: yes** (Tianhao, lane R thread). The user output callback
becomes
``std::function<Variables(Variables const&, FluxRegister const&)>
user_output_callback``. Through the const accessors it reads the recorded
fluxes as **step transfer per unit area, not a rate**:

* ``total(dim)`` (R:82);
* ``kept(dim, part)`` (R:83);
* ``step_field(dim)`` (R:84).

The lines are the same at ``4c59162a`` and ``2b4c5775``; this document
cites ``4c59162a``. R:82-83 call ``total`` and ``kept`` "kFlux" views, and
only ``step_field`` says "per unit area" in the header. The step-transfer
reading of all three is Tianhao's answer.

At the pin:

* the member is declared at ``src/mesh/meshblock.hpp:125`` as
  ``std::function<Variables(Variables const&)>``. That is lane X's file;
  Xi is asked to carry the two-argument declaration in X rev 3.13.
* O's caller checks it at ``src/output/load_user_output_data.cpp:14`` and
  calls it at ``:19``. In V3 it passes the block's ``FluxRegister``.

*Choice:* the O skeleton does not declare the callback, which is X's
member, or the loader. So it does not differ from the signature and is
unchanged.

L-3. **Pending R.**

* A cubed-sphere panel seam is not a physical boundary (Jiheng, lane S
  thread). Its two entries are reconciled by ``replace`` (R:63-64) and are
  not in ``boundary(dim)``.
* ``boundary(dim)``, area-integrated (R:97; R.rst:555-559), sums the wall,
  outflow and custom faces.
* O proposes adding the **immersed-solid faces, all rows**. Jiheng accepts
  this as a proposal, and it is with Tianhao for R's agreement.

Section 4 and section 3.2 say the same.

L-4. **Closed**: FP64, and FP32 on MPS (R.rst:453, 644).

L-5. **Closed**: ``begin_step()`` (R:48) and ``rollback()`` (R:86;
R.rst:464-467); ``observe`` runs before the next ``begin_step()``
(R.rst:560).

L-6. **Closed**: R's ``RegisterMonitor`` owns the domain totals (R:98;
R:89-90). Lane O prints the cycle line from them.

R-O1, R-O2 and R-O3: **settled** in the frozen header (section 4).

Every ``R:`` and ``R.rst:`` citation in this document, including the
response tables 11.5 and 11.6, is at ``4c59162a`` (rev 4 re-cite).

**11.2 Lane X, Xi.**

X-1. **Closed** (Xi, ``X.rst`` at ``77c65d5a``, answer X1): output,
flushes and restart writes run in the host phase after the global accept,
outside captured segments.

X-2. Closed: O adopts X's host ``dt`` after the accept (``X.rst`` at
``77c65d5a``, answer X2).

X-3. **Closed** (Xi, answer X3): after ``read_restart`` the Mesh calls
``StageExecutor::invalidate()``. Restart never serializes graphs
(``V3_ROADMAP.md:80``).

X-4. **Closed** (Xi's final joint-row answer): the joint rows 1-5 of
sections 8.1 and 9 are accepted. A device ``dt`` at ``meshblock.cpp:1210``
is refused for the first release, and O withdraws that request.

**11.3 Lane C, Tianhao.**

* Lane C's draft names the layout lane "O" and diagnostics "EF"
  (happysky19/snapy ``b7d1300f``, ``c.rst`` section 12). Which lane owns the
  face-floor event counters that feed the monitor?

**11.4 Mesh owner (lane X, Xi), implicit owner (lane I, Xinyue), kickoff thread.**

* Reconcile the lane letters (above).
* Xi: the mesh items of section 8.1 are **accepted** (rows 1-5).
* Xinyue: accept the implicit items of section 8.1.

  **Closed** (Xinyue, lane I thread): lane I accepts ``lubksb.h`` (71) and
  ``ludcmp.h`` (95) at ``src/math/``. Lane I's 2,821 (to 2,033) includes
  them (``I.rst`` rev 2, ``8c2ba226``). They are in neither lane's cut, and
  O's roll-up is 9,830 to 4,473 (section 9).
* Owners of option headers: when to switch to the generator.


**11.5 Responses to Tianhao's round-2 notes 1-17 (on rev 1 ``a39e0e8d``).**
Each citation was checked at the pin before it was accepted. The note texts
reached this revision as one-line summaries.

.. list-table::
   :header-rows: 1
   :widths: 5 13 82

   * - note
     - reply
     - change made, with ``file:line`` at the pin
   * - 1
     - agree
     - Closed as ``agree``. The section titles follow the kickoff's twelve in
       order (1-12), and section 4 checks requirements 1-10 one by one. No
       change was needed.
   * - 2
     - accept
     - Rev 1's "pure deletions 4,163" included 525 lines that are replaced,
       not purely deleted: ``output_formats.hpp`` 105 (folded into the core),
       ``vectorize.*`` 70 (``Field.names``) and ``utils/flip_zero.*`` 350
       (the ``bc`` copy). Section 9 now states pure deletions as **3,813**:
       ``src/diagnostics`` 1,585, ``parameter_input.*`` 993,
       ``io_wrapper.*`` 364, ``input/signal.*`` 187, ``pull_neighbors.*``
       249, ``utils_dispatch.*`` 131, ``cuda_utils.*`` 22, ``hdf5.cpp`` 27,
       ``refine.*`` 80, ``fp32_storage.*`` 175.
   * - 3
     - accept
     - Each file is booked once (section 9, joint rows), agreeing with the
       X/O table Xi confirmed:

       * ``fp32_storage.*`` 175 in O, deleted;
       * ``fp64_store.hpp`` 85 in O, moved to ``src/hydro``, not cut;
       * ``tensor_cast_cache.hpp`` 53 in O, kept.

       X books none of the three.
   * - 4
     - accept
     - Joint accounting (sections 8.1 and 9):

       * X books all of ``src/interface``: ``athena_arrays.hpp`` 669,
         ``init_from_tensor.hpp`` 199, ``stride_iterator.hpp`` 140. This is
         conditional on O5.
       * X deletes ``meshblock.cpp:203-230`` (item sv) and
         ``meshblock_options.cpp:65-86`` (item j).
       * ``:1623-1653`` (to the restart module), ``:1226-1339`` minus
         ``1336-1337`` and ``cycle_diagnostics.hpp`` (16, to the printer)
         go to O. X counts none of these lines.
       * ``meshblock.cpp:1336-1337`` (``fixgrav``) sits in X's gravity row.
       * The rest of ``1603-1833`` stays in ``src/mesh``.
   * - 5
     - accept
     - A third slice check exists: ``meshblock_options.cpp:65-86`` (bounds
       and order, 22 lines). Sections 2.2 and 8.1 now list three sites. The
       generated checks carry both the sum check (``output_type.cpp:90-98``)
       and the bounds/order checks. X deletes ``meshblock_options.cpp:65-86``
       (item j) and ``meshblock.cpp:203-230`` (item sv).
   * - 6
     - accept
     - ``src/options/*.yaml`` (300, *estimate*) is now counted as an
       addition in ``src``.

       * ``src``-only cut: 6,407 → **6,107 (61.1 %)**.
       * Settled cut: 5,377 (53.8 %), unchanged.
       * Design target: 5,807 (58.1 %), unchanged.
       * Strict cut: 3,833 (38.3 %), unchanged.

       The last three are unchanged because they already subtracted every
       addition.
   * - 7
     - accept
     - Section 3.2 now states a double-counting rule.

       * Fixer *amounts* are conserved-quantity changes (requirement 7) and
         enter the identity.
       * Event *meters* (``lim_cut``, ``positivity_severe``, floor and p*
         counts) are counts and never enter it.
       * A fixer that changes face fluxes before they are posted is
         already inside ``boundary``. It is reported as a part
         (requirement 2) and not added again.
       * Only source-type fixers (gravity work, AAM closure, VIC clamp)
         enter ``sum fixer``.
   * - 8
     - accept
     - Lane C posts flux density per unit face area, in coord's
       ``flux2global<d>_`` frame, into ``slot(d, "riemann")`` per direction
       and stage (``riemann.hpp:17-21`` and ``C.rst:1046`` at C rev 6.2,
       ``bcac27c97a9efc1de2cb2041e6b709f07eedcefe``).

       * Lane O now uses R's names (frozen header ``4c59162a``).
         ``FluxRegister::step_field(dim)`` is the FP64 step transfer per
         unit area (R:84), and O multiplies it by
         ``CoordinateImpl::face_area1()`` (``coordinate.hpp:254``; the
         divergence applies the same areas at ``coordinate.cpp:732-746``).
       * The placeholder ``FluxRegisterView`` and ``MonitorView`` types are
         removed (section 4).
   * - 9
     - accept
     - ``budget_series.cdl`` writes C's counters by name through
       ``counter_names()``/``counters()`` (int64). These are C's five names,
       adopted by R (``C.rst:1244-1251`` at ``bcac27c9``):

       * ``face_floor.rho``, ``face_floor.p``, ``face_floor.species``
         (``FaceFloor``);
       * ``riemann.pstar_clamp`` (``FaceFlux::events`` bit 0,
         ``riemann.hpp:30``);
       * ``riemann.llf``.

       Rev 5 final replaced the older ``face_floor_count`` and
       ``pstar_clamp_count`` (C18).

       * Counts are ``int64``.
       * Fractions and minima are ``double``.
   * - 10
     - accept (adopt X)
     - O adopts X's host ``dt`` after the accept (``X.rst`` X2 answer at
       ``77c65d5a``: "dt and the stage weights stay host doubles";
       ``meshblock.cpp:773``).

       * ``observe`` and ``accumulate_stats`` take ``double dt``.
       * This breaks no requirement: both run after the accept, outside
         captured segments, so requirement 6's rule (no host read inside a
         segment) holds and nothing is baked into a graph.
       * X-2 is closed. Skeleton ``diagnostics.hpp`` and
         ``writer_core.hpp`` are changed.
   * - 11
     - accept
     - The monitor is read on the host every step already: X stacks it
       into the step-end retry read (``meshblock.cpp:1566``, "the one
       device-to-host read"; ``X.rst`` :122).

       * O adds no host read per step.
       * ``observe`` accumulates the monitor on device every accepted step,
         before R resets it.
       * ``flush`` reads O's own sums only at its cadence.
   * - 12
     - accept
     - ``BudgetCarry`` (``restart.hpp``) now stores ``total_prev``
       (nrows), ``pending`` (nrows) and ``fixer_pending`` (nfixers,
       nrows), all float64 on CPU, with row and fixer names. Without
       ``total_prev``, the first interval after a restart could not close.
   * - 13
     - accept
     - The line is produced at ``meshblock.cpp:1232-1338``, the function
       body after the signature at ``1226-1231``. Sections 2.3 and 8.1 cite
       it. ``1226-1339`` minus ``1336-1337`` goes to O's printer, as agreed
       with Xi. The ``print_cycle_info`` wrapper (``:1341-1346``) stays in
       ``src/mesh``.
   * - 14
     - accept, in part
     - The read site is found: ``x1_centroid.cpp:68-79``, through
       ``get_env`` (``layout.hpp:39-42``). That closes rev 1's hypothesis.
       ``options_table.yaml`` gets a proposed ``geometry/x1-centroid-exact``
       row with ``env: SNAP_X1_CENTROID_EXACT``; the owning lane decides.

       Counts:

       * ``rg -o 'SNAP_X1_CENTROID_EXACT' --hidden -g '!.git' -g
         '!docs/v3/**' .``: **53** occurrences in **21** files;
       * over ``src python tests examples``: 49 in 18.

       The "14" was not reproduced.
   * - 15
     - accept
     - The #350 merge is cited by its full sha,
       ``ffaffdd25f3b7d3d63204fde84cdcbb476f08517``.
   * - 16
     - accept, static
     - Two static checks:

       * **Cubed-sphere exchange.** It is at ``hydro_forward.cpp:1033-``
         (``if (playout->options->type() == "cubed-sphere")``). O reads
         only R's reconciled transfer (requirement 5). On the sphere the
         domain boundary is x1 only, so the budget's ``boundary`` uses
         only x1 faces there.
       * **``d_sed``.** No such symbol exists at the pin (``rg -n -w d_sed
         src python tests examples`` is empty). The settling flux is part
         of ``flux1``'s species rows (``flux_positivity.hpp:120``), so it
         reaches the budget through ``boundary``. **Hypothesis**:
         sedimentation is posted as a register increment, not a source.
         This is a question for R and A.
   * - 17
     - accept
     - The new file is renamed ``src/diagnostics/hydro_flux_profile.cpp``,
       so it no longer shares a row name with the deleted
       ``hydro_flux.cpp``. ``total(t0)`` is renamed ``total_prev``
       everywhere (``diagnostics.hpp``, ``budget_series.cdl``, sections 3.2
       and 5).

**11.6 Responses to Xi's round-2 notes 1-12 (on ``b66b6745`` and
``d69b5442``).** Citations are at pin ``9e5de2d4`` unless another sha is
named.

.. list-table::
   :header-rows: 1
   :widths: 5 11 84

   * - note
     - reply
     - change made and reason
   * - 1
     - accept
     - ``hydro_w`` stays stored. ``restart.cpp:31-38`` copies every defined
       entry of ``vars``, and keeping it avoids a new EOS call on load.

       * Section 3.4 now lists ``hydro_w`` as stored.
       * ``restart.hpp`` gains ``kRestartStoredKeys`` (12 keys).
       * ``meshblock.cpp:1827-1829`` is now cited for what it does: rebuild
         ``scalar_r``.
       * Registers and ``StateStore`` tensors stay out of the
         ``Variables`` that ``write_restart`` copies (requirement 10).
   * - 2
     - accept
     - ``options_table.yaml`` gets three sample rows:

       * ``dynamics.precision``, which exists at the pin
         (``hydro_options.cpp:55, 67``; ``hydro.hpp:76``, default
         ``float64``);
       * ``integration.execution`` (``eager|graph``);
       * ``integration.execution-strict``.

       The two ``integration`` names are **provisional**. Choosing them is
       Cheng's call (``V3_ROADMAP.md:300-301``, "choose final YAML/Python
       names at G0/G1").
   * - 3
     - accept
     - Three changes:

       * (a) Section 6 no longer casts in ``observe``, because R holds
         FP64, and FP32 on MPS (R.rst:453).
       * (b) Section 4: O reads ``boundary(dim)`` as area-integrated, so
         the area is applied once, only to ``step_field``. R now states it
         (R:97; R.rst:555-559).
       * (c) R-O3: ``gravity_work`` as a source fixer is the fixer's
         correction only (``hydro.hpp:301-311``), not R's face work (R:19),
         which is inside ``boundary``. Since R rev 2 the fixer has no poster
         (R.rst:572-573).
   * - 4
     - accept
     - One choice: the **host dt of the accepted step**.

       * Section 7 and ``writer_core.hpp`` now agree:
         ``accumulate_stats(vars, double dt)``.
       * On the first call (the initial output, before any step) dt ≤ 0
         and nothing is accumulated. That matches
         ``output_type.cpp:302-306`` today.
   * - 5
     - accept
     - Reworded to "no persistent or graph-visible allocation", in the
       requirement 9 row and in section 7.
   * - 6
     - accept
     - The closure is 33 lines by ``wc -l``: ``meshblock.cpp:1009-1035`` (27)
       plus ``meshblock.hpp:223-228`` (6). Section 8.1 and section 9 now
       say so.

       X's booked figure stays at 29 new lines. Four of the 33 are already
       inside X's own items (``1009-1012`` counts 1 new, and
       ``meshblock.hpp:226`` is already counted), per the joint table. O's
       numbers are unaffected.
   * - 7
     - agree
     - Recorded. No change.
   * - 8
     - accept (blocking)
     - O5 now lists all of the following:

       * the port of ``TestOutputType`` (``tests/test_user_output.cpp:39-100``,
         which reads ``pdata->data`` dims and elements) to ``Frame``/``Field``
         accessors;
       * the sink element reads (``netcdf.cpp:496-558``, 8 sites;
         ``pnetcdf.cpp:362-413``, 8 sites) and the
         ``get_num_variables(…, pdata->data)`` calls. Section 3.1 lists the
         same.

       The ``CopyFromTensor``/``InitFromTensor`` count in
       ``load_diag_output_data.cpp`` is 14, not 16 (``rg -c``).

       Section 10 states lane X's precondition as one grep:
       ``rg -n 'athena_arrays|init_from_tensor' src tests python``. The only
       allowed hit is ``src/turbulence/turbulence_model.hpp:9``, which is
       outside the build. At the pin the grep has 8 hits:
       ``test_eos_io_core.cpp:7``, ``diagnostics.hpp:8``,
       ``convective_heat_flx.cpp:8``, ``output_type.hpp:22``,
       ``output_utils.hpp:10``, ``athena_arrays.hpp:3, 669`` and that one.
       O1, O5 and X's deletion remove all the others.
   * - 9
     - accept
     - Section 3.4: on load, each stored tensor is ``copy_``\ ed into its
       ``StateStore`` buffer, cast to that buffer's ``PrecisionPolicy`` row
       dtype. A restart is written only after an accepted step (stage 0,
       register empty after ``begin_step()``, R:48; R.rst:464-467), so the
       file needs no new stage or dtype
       field.
   * - 10
     - accept
     - ``BudgetCarry`` is written only when a budget output is requested. A
       file without a carry, including pin-era files, starts a new budget
       interval at the first ``observe``.

       * Both cases are added to G-4 and to the O8 row.
       * The stored set of a deck without a budget output does not change,
         so there is no FOR CHENG item.
   * - 11
     - accept
     - The requirement 6 row and ``diagnostics.hpp`` now use section 7's
       wording: the monitor's host read is X's step-end retry read
       (``meshblock.cpp:1566``), and O adds none per step. ``flush`` adds
       only the cross-rank FP64 reduction at the budget cadence.
   * - 12
     - accept
     - O1 keeps the ``stride_iterator.hpp`` deletion, whose only includers
       are in ``src/diagnostics``. The O1 row now says it closes lane X's
       row. The line count is unaffected: neither lane cuts it twice.

**11.7 Answer to lane R** (``R.rst:1603-1604`` at ``4c59162a``: "O: confirm
``AmountKind`` and the budget formula of 4.2 against ``c12f78bb``").

**Agree.** I checked R's ``AmountKind`` (R:25-31) and the budget formula of
R.rst 4.2 (R.rst:550-554) against O's identity (section 4, "The budget
identity"; section 5; ``diagnostics.hpp``). They are the same:

``residual = delta domain_total + sum_d boundary(d) - sum amounts`` of
kinds ``cell_source``, ``floor`` and ``fixer``. ``flux_correction`` and
``diagnostic`` are reported only.

Area-integrated ``boundary(dim)`` (R:97; R.rst:555-559) and the
derived-row convention, ``derived_amount("<row>.boundary.<dim>", ...,
cell_source)`` (R.rst:542-545), need no special case in O.

Three alignments on O's side, made in this revision, none of them a
conflict:

1. O's fixer list followed R.rst:572-575. ``gravity_work`` is dropped (no
   poster after Cheng's rulings). ``positivity`` and ``vic.mass_clamped``
   are ``flux_correction``.
2. The reset point is ``begin_step()`` (R:48), not ``begin_stage(0)``.
3. Read timing follows lane X rev 3.1 (``X.rst:165-173`` at ``bac201d1``).
   X reads only the retry flags at step end, and O's ``flush`` makes the
   amount SUM and the host read. R.rst:460-463 still says all monitor
   values are read in X's step-end read. **Request to R:** narrow that
   sentence to the retry flags so R, X and O read the same.

12. Skeleton index
------------------

``docs/v3/skeleton/O/`` holds declarations only: no function bodies, 299
lines in total (``wc -l``). Nothing in it is compiled or run.

.. list-table::
   :header-rows: 1
   :widths: 26 8 66

   * - file
     - lines
     - contents
   * - ``writer_core.hpp``
     - 98
     - ``GridKind``, ``Field``, ``Frame``, ``Sink``, ``GriddedWriter``,
       ``combine_parts``; #350 helpers kept (section 3.1)
   * - ``diagnostics.hpp``
     - 61
     - ``Consumer``, ``BudgetConsumer``, ``HydroFluxProfile`` and
       ``FieldDiagnostic``, reading lane R's ``FluxRegister::step_field`` and
       ``RegisterMonitor`` (frozen R header ``4c59162a``; sections 3.2, 4)
   * - ``restart.hpp``
     - 57
     - The #367 contract, bundle magic, ``OutputSchedule``,
       ``BudgetCarry``, read/write/check; no register state (requirement
       10) (section 3.4)
   * - ``options_table.yaml``
     - 45
     - Table format; output, hydro and geometry sample rows, including
       ``SNAP_ANOMALY`` and the proposed ``SNAP_X1_CENTROID_EXACT`` row (the
       FP32 storage switches are removed, not tabled; section 3.3)
   * - ``gen_options.pyi``
     - 15
     - Generator interface (section 3.3)
   * - ``budget_series.cdl``
     - 23
     - Budget time-series format (section 3.2)
