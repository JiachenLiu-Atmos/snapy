:orphan:

V3 lane O: output, input, utilities, diagnostics, options schema
=================================================================

Revision 1 (restructured), 2026-11-18. Owner: Jiachen (alternate: Isaac).
Reviewers: Tianhao (C) and Isaac (K). Design only: this branch proposes no
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

Deleted but not counted toward the target are 925 lines outside that
baseline: those three files, ``z.junk/load_athena_output_data.cpp_`` (585),
and ``src/interface/stride_iterator.hpp`` (140). The latter's only includers
are in ``src/diagnostics`` (``rg -l stride_iterator src``).

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
  ``pnetcdf``. The slice checks run twice: ``output_type.cpp:90-98`` and
  ``meshblock.cpp:203-230``.
* **Write.** ``make_outputs`` (``meshblock.cpp:1207-1224``) calls
  ``AccumulateStats`` every step (``:1210``) and writes when
  ``current_time >= next_time``.
* **#350's writer core** (186 lines; merge ``ffaffdd5``) shares
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

``print_cycle_diagnostics`` (``meshblock.cpp:1226-1339``) does the following
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
* ``SNAP_X1_CENTROID_EXACT`` appears 49 times, but no ``getenv`` in
  ``src/`` names it. **Hypothesis**: it is read through the
  ``layout.hpp:40`` helper, which I did not trace.

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

**Combining.** ``combine_parts`` reads only the attributes snapy writes,
about 200 lines (*estimate*). It replaces ``mppnccombine.cpp`` once gate G-2
passes (section 5).

**Ownership.** The writer borrows the block's variables for one write and
owns only host copies, its schedule and its statistics tensors.

**Moves out of src.**

* ``tools/snap_combine.cpp`` (about 30 lines), a command-line wrapper around
  ``combine_parts`` for ``combine: false`` runs.
* ``input/command_line.*`` to ``examples/``.

3.2 Diagnostics as register consumers (Q3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Consumer`` has two methods:

* ``observe``: runs on the device every accepted step and reads the
  register's step transfer, the monitor and a device ``dt``;
* ``flush``: makes the collective and the host read, at its own cadence
  only.

Two consumers replace ``src/diagnostics``:

* ``BudgetConsumer``. For each row it accumulates in FP64:

  * ``boundary``, the outward register transfer through the physical domain
    boundary;
  * ``fixer[f]``, the monitor's amount for each fixer.

  At each record it writes ``residual = (total - total_prev) + boundary -
  sum fixer``.

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
* Fixers: ``limiter``, ``positivity``, ``vic_clamp``, ``gravity_work`` and
  ``aam_closure``.
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
* ``SNAP_ANOMALY``: the row ``dynamics/anomaly`` with ``env: SNAP_ANOMALY``,
  the only permanent override. It is named in the user-facing message at
  ``meshblock.cpp:1782`` and used on 5 test lines. **FOR CHENG.**
* ``SNAP_FP32_STORAGE`` and ``SNAP_FP32_STORAGE_ROWS``: **removed, not
  tabled (settled: Cheng approved removing the FP32 storage emulation).**
  The switches get no row in the options table. ``src/utils/fp32_storage.*``
  (175 lines, counted in O) is deleted together with its application in
  ``meshblock.cpp:621-642`` (22 lines, counted in lane X; section 8.1).

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
* ``scalar_s`` (``:30-34``);
* ``fill_solid_hydro_u`` and the boundary references, when present;
* ``last_time`` and ``last_cycle``;
* the output schedule (``:73-99``).

New: ``BudgetCarry``, the pending FP64 sums since the last budget record.
This is output bookkeeping, not register state.

**Derived** (never written):

* ``hydro_w`` and ``scalar_r`` (``meshblock.cpp:1827-1829``);
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
* ``math/*.h`` to ``src/implicit``.

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
nothing**. It is a reader only. What it reads, and the ordering it relies
on:

* **Reads.**

  * The step transfer per axis, per conserved row (requirement 3).
  * The per-fixer amounts and the monitor totals (requirements 6, 7).
  * The row names and the dtype (requirement 8).

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
     - Lane O owns no flux. ``HydroFluxProfile`` and ``boundary`` use R's
       documented weighting. Open question L-1.
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
       ``*_flux`` outputs need not rebuild fluxes. Open question L-2.
   * - 5
     - shared face: two entries, one reconciliation rule; rotation owned by
       coord
     - The domain-boundary transfer has one entry per face. Lane O applies
       no rotation and reads scalar sums only. Open question L-3.
   * - 6
     - one device-side monitor (mass, species, E+PE, AAM, scalars); no host
       reads in a segment
     - Its rows are the budget rows. The monitor is read on the host only in
       ``flush``, outside segments (section 7).
   * - 7
     - every fixer and floor registers what it added
     - **Adopted.** It replaces A2. ``fixer[f]`` is R's per-fixer amount,
       and ``aam_closure`` is a fixer channel, not part of the flux.
   * - 8
     - stated dtype under FP32; fixed storage a graph can capture
     - Lane O needs FP64 sums for the residual. It casts at ``observe`` if
       R holds FP32. Open question L-4.
   * - 9
     - no added full-field copy per stage; memory and cost stated
     - Lane O adds no per-stage work. Per accepted step it adds a few
       reductions of the (nrows,) sums and one (nvar, nc1+1) profile sum.
       *Estimate*; not measured.
   * - 10
     - the register is rebuilt, not written to restart
     - **Adopted.** It replaces the first draft's "derived" assumption.
       ``restart.hpp`` writes no register or monitor state.

Assumption A3 of the first draft (output outside captured segments, ``dt``
as a device value) is not a register requirement. It moves to section 7 and
question X-1/X-2. A4 (the budget rows) is now requirement 6. A5 (lane
letters) is in 11.3 and 11.4.

5. Invariants and gates
-----------------------

The identity: for every budget row,
``total(t) - total(t_prev) = -boundary + sum_f fixer(f) + residual``. With
FP64 accumulation, ``|residual|`` should be at round-off relative to
``total``. The tolerance (proposed ``1e-12 * |total|``, *estimate*) is set
by G0 baselines, not here.

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
     - ``combine_parts`` against ``mppnccombine`` on the example decks and
       the #350 decks (6 panels, slices, ghost zones, double precision):
       every variable, dimension and attribute
     - O6
     - bitwise
   * - G-3
     - The generated option structs have the members and defaults of the
       pin; each module's deck parse equals the old parse on the in-tree
       decks
     - O4
     - exact
   * - G-4
     - Restart round trip (FP32/FP64, anomaly, 4-block bundle, #367
       mismatches still refused); a truncated bundle now throws
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

**Hypotheses the gates test:**

* ``combine_parts`` is complete for snapy's files (G-2).
* The two ``flip_zero`` implementations agree (G-6).

**Hypotheses not covered by a gate:**

* The ``strtok`` race: two blocks of one process writing at once, which
  ``combine_on_root``'s mutex (``writer_core.cpp:81-93``) anticipates. The
  core removes ``strtok`` either way.
* Reading restart tensors from a stream.
* The ``SNAP_X1_CENTROID_EXACT`` read site.

6. FP32 contract
----------------

* **Gridded output.** It reads totals: ``total_u`` adds the FP64 reference
  (``load_hydro_output_data.cpp:38``; the cycle line at
  ``meshblock.cpp:1257``). Files stay ``NC_FLOAT`` by default and
  ``NC_DOUBLE`` with ``double_precision`` (``output_type.hpp:79-83``).
  pnetcdf stays float only (``pnetcdf.cpp:58-61``).
* **Budget.** Accumulation, totals and the file are all FP64. If the
  register holds FP32 (requirement 8 leaves the dtype to R), ``observe``
  casts each step's transfer to FP64 before accumulating. That loses
  nothing that R kept.
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

The only per-step work is ``Consumer::observe`` and
``GriddedWriter::accumulate_stats``. Both are device-only and take ``dt`` as
a device scalar. Today ``AccumulateStats`` uses the host difference
``current_time - stat_last_time_`` (``output_type.cpp:308``); a graph would
bake that value in.

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
       every replay" (``:224``): the ``vic_clamp`` and ``gravity_work``
       channels are read only in ``flush``.
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

To the mesh owner (lane X, Xi):

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
       (``signal_handler.hpp:45``).
   * - ``src/utils/signal_handler.hpp``
     - 67
     - Same as above.
   * - ``meshblock.cpp:203-230`` (``validate_slice``)
     - 28
     - Delete. It repeats ``output_type.cpp:90-98``; the generated output
       option checks (section 3.3) do both checks once.
   * - ``meshblock.cpp:1623-1653`` (#367 restart metadata checks)
     - 31
     - Move into ``check_restart_metadata`` (``restart.hpp``), next to the
       file contract; ``_init_from_restart`` calls it. The behaviour and
       messages are unchanged.
   * - ``meshblock.cpp:1226-1339`` (``print_cycle_diagnostics``) and
       ``mesh/cycle_diagnostics.hpp`` (16)
     - 114 + 16
     - Becomes a printer over ``BudgetConsumer``'s totals (section 3.2). It
       merges with O8, after lane R's monitor. The printed content is
       unchanged.
   * - ``meshblock.cpp:1210`` (``AccumulateStats`` call)
     - 1
     - Takes ``dt`` as a device scalar (section 7).
   * - ``meshblock.cpp:621-642`` (``round_storage``,
       ``round_storage_after_stage``)
     - 22
     - **Delete; counted in lane X, not in O.** The FP32 storage emulation
       is removed (settled by Cheng). Verified at the pin: ``:621``
       ``bool MeshBlockImpl::round_storage(...)`` through ``:642``, the
       closing brace of ``round_storage_after_stage``; 22 lines
       (``git show 9e5de2d4:src/mesh/meshblock.cpp | sed -n 621,642p``).
   * - Other emulation sites in X's files: ``meshblock.cpp:22`` (include),
       ``:618`` (call), ``meshblock.hpp:207-215`` (declarations and
       comment), ``mesh.cpp:358, 405`` (calls)
     - 1 + 1 + 9 + 2
     - Delete with the range above; counted in lane X. Outside both lanes:
       ``python/csrc/pymesh.cpp:238-247, 331-343`` and
       ``python/snapy/mesh.pyi:175, 220-222`` (the ``round_storage``
       bindings), ``tests/test_fp32_storage.cpp`` (79),
       ``tests/test_anomaly_storage.py:396`` (the ``fp32`` arm) and
       ``tests/CMakeLists.txt:8, 359, 816``.

To the implicit owner (lane I, Xinyue):

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - file or block
     - lines
     - change and reason
   * - ``src/math/lubksb.h``
     - 71
     - Move to ``src/implicit``. Its only includer is
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
3. The device ``dt`` part of O8 merges after lane X's executor hook (Xi).
4. Other lanes switch their option headers to the generator after O4, in
   any order.

9. Line budget
--------------

The baseline per file is ``git show 9e5de2d4:<path> | wc -l``.

* "proposed" is what stays in place, or the new file.
* "moved" is not cut.
* cut = pin − proposed − moved.

Proposed sizes are *estimates*. Revision 2 adds a reachability class to
every row (section 9.1) and applies Cheng's decisions of 2026-11-18. One
per-file number changed: ``src/utils/fp32_storage.{cpp,hpp}`` goes from
95 + 55 kept to 0 + 0, because the emulation is removed. Every other row is
as in revision 1. ``meshblock.cpp:621-642`` (22 lines), where the emulation
is applied, **counts in lane X**, not here (section 8.1).

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

**How things get built at the pin.**

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
plus the duplicate ``input/signal.*`` (187): built, but no symbol is
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

Before (revision 2 part 1, ``a5325be7``): 4,413 lines (44.1 %) backed by
evidence; 5,197 (52.0 %) after G-2; 5,627 (56.3 %) after G-6 and Cheng.
After Cheng's decisions:

.. list-table::
   :header-rows: 1
   :widths: 54 14 12 20

   * - counted
     - cut
     - %
     - depends on
   * - Settled: never-compiled code (a), built-but-unreferenced code (b),
       the other unreferenced utilities, the rewrites and merges, plus
       Cheng's approved removals (``mppnccombine.cpp`` replaced by
       ``combine_parts``, the hdf5/fits stubs, ``fp32_storage.*``)
     - **5,347**
     - **53.5**
     - section 9.1 and Cheng's decisions
   * - plus ``utils/flip_zero`` (350, gate G-6) and ``refine`` (80,
       super-resolution, still FOR CHENG)
     - 5,777
     - 57.8
     - G-6 and Cheng
   * - strict reading: (b) kept as well (−1,544 from the first row)
     - 3,803
     - 38.0
     - still above the 30 % target

**Settled cut: 5,347 lines (53.5 %).** It is up from 4,413 because the
combiner (+784 net of its replacement) and the FP32 storage emulation
(+150) are now approved removals. The combiner's replacement still has to
pass G-2 before its deletion PR merges. The design target with every row is
5,777 lines (57.8 %). Each row of the table below carries its class.
``O_lines.csv`` has the same column.

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
     - delete; built, unreferenced
     - only reached from parameter_input
   * - ``src/input/io_wrapper.hpp``
     - 59
     - 0
     - 
     - 59
     - delete; built, unreferenced
     - 
   * - ``src/input/parameter_input.cpp``
     - 873
     - 0
     - 
     - 873
     - delete; built, unreferenced
     - legacy Athena++ parser; no caller
   * - ``src/input/parameter_input.hpp``
     - 120
     - 0
     - 
     - 120
     - delete; built, unreferenced
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
     - delete; built, unreferenced
     - older copy of utils/signal_handler; no includer
   * - ``src/input/signal.hpp``
     - 63
     - 0
     - 
     - 63
     - delete; built, unreferenced
     - 
   * - ``src/math/lubksb.h``
     - 71
     - 0
     - 71
     - 0
     - move; live, moved
     - to src/implicit (one includer: forward_sweep_impl.h)
   * - ``src/math/ludcmp.h``
     - 95
     - 0
     - 95
     - 0
     - move; live, moved
     - to src/implicit
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
     - delete; built, unreferenced; removal approved by Cheng
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
     - delete; LIVE, removal approved by Cheng; replaced by combine_parts (gate G-2)
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
     - 301
     - 120
     - 
     - 181
     - rewrite; live, rewritten
     - metadata of live fields only (unused canoe-era names dropped)
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
     - 420
     - 
     - -282
     - rewrite; live, rewritten
     - Frame transform, packing, combine_parts (~200)
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
     - delete; built, unreferenced
     - checkCudaError has no caller
   * - ``src/utils/cuda_utils.h``
     - 3
     - 0
     - 
     - 3
     - delete; built, unreferenced
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
     - delete; LIVE feature, removal approved by Cheng
     - FP32 storage emulation; removal approved by Cheng
   * - ``src/utils/fp32_storage.hpp``
     - 56
     - 0
     - 
     - 56
     - delete; LIVE feature, removal approved by Cheng
     - same; meshblock.cpp:621-642 counts in lane X
   * - ``src/utils/fp64_store.hpp``
     - 85
     - 0
     - 85
     - 0
     - move; live, moved
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
     - delete; built, unreferenced
     - only commented-out calls (equation_of_state.cpp:254-257)
   * - ``src/utils/pull_neighbors.hpp``
     - 24
     - 0
     - 
     - 24
     - delete; built, unreferenced
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
     - move; live, moved
     - to src/mesh (callers mesh.cpp, meshblock.cpp)
   * - ``src/utils/signal_handler.hpp``
     - 67
     - 0
     - 67
     - 0
     - move; live, moved
     - to src/mesh
   * - ``src/utils/tensor_cast_cache.hpp``
     - 53
     - 53
     - 
     - 0
     - keep; live, kept
     - 3 includers
   * - ``src/utils/utils_dispatch.cpp``
     - 26
     - 0
     - 
     - 26
     - delete; built, unreferenced
     - bdot_out has no caller
   * - ``src/utils/utils_dispatch.cu``
     - 92
     - 0
     - 
     - 92
     - delete; built, unreferenced
     - 
   * - ``src/utils/utils_dispatch.hpp``
     - 13
     - 0
     - 
     - 13
     - delete; built, unreferenced
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
   * - ``src/diagnostics/hydro_flux.cpp``
     - 
     - 80
     - 
     - -80
     - new
     - HydroFluxProfile from the register
   * - ``tools/gen_options.py``
     - 
     - 300
     - 
     - -300
     - new, outside src
     - options generator
   * - ``src/options/*.yaml``
     - 
     - 300
     - 
     - -300
     - new, outside src
     - options tables (data), ~1.4 lines per option
   * - ``tools/snap_combine.cpp``
     - 
     - 30
     - 
     - -30
     - new, outside src
     - CLI around combine_parts for combine: false
   * - **baseline (71 files)**
     - **9996**
     - **2939**
     - **650**
     - **6407**
     - 
     - src only: 64.1% cut
   * - **including new files outside src**
     - 
     - **630**
     - 
     - **5777**
     - 
     - 57.8% cut; target 30% = 2,999

**Totals.**

* Baseline: 9,996 lines in 71 files (rechecked at the pin, section 9).
* Kept in place: 2,419. New in ``src``: 520. Moved out unchanged: 650.
* Cut in ``src`` only: 6,407 lines (64.1 %).
* Cut including the 630 new lines outside ``src``: 5,777 lines (57.8 %).
  **Settled: 5,347 lines (53.5 %)** (section 9.2). The 30 % target is 2,999.
* If every rewritten or new file is 50 % over its estimate: 4,197 lines
  (42.0 %) for the design target.
* Not counted: 925 lines deleted outside the baseline, and the 22 + 13
  lines of FP32 storage application in lane X's files (section 8.1).

10. Migration
-------------

The PR sequence is below. "Bitwise" means every state and every written
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
       files, ``z.junk``, ``stride_iterator.hpp``
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
       (O), ``meshblock.cpp:621-642`` and the other X sites (X, section
       8.1), the ``round_storage`` bindings, ``test_fp32_storage`` and the
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
       the env rows; the covariance warnings deleted
     - bitwise
     - same members and defaults (G-3)
   * - O5
     - Writer core, netcdf and pnetcdf sinks; ``vectorize`` and
       super-resolution removed
     - bitwise files
     - same values cast once, same layout (G-1)
   * - O6
     - ``combine_parts`` replaces ``mppnccombine`` (removal approved by
       Cheng; section 10.1)
     - bitwise variables
     - G-2 is run on the PR before ``mppnccombine.cpp`` is deleted
   * - O7
     - Restart merge; the reader throws
     - bitwise
     - same files (G-4)
   * - O8
     - Consumers, ``BudgetSeries``, the cycle line from the budget totals
     - new file; cycle line bitwise if the summation order is kept
     - if the reduction order changes, only the last digits of the
       *printed* totals change, never the state

10.1 Is a replacement combiner still needed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision: yes. ``combine_parts`` (about 200 lines in
``writer_core.cpp``, *estimate*) replaces ``mppnccombine.cpp``.** The
command-line wrapper ``snap-combine`` (``tools/snap_combine.cpp``, about 30
lines) covers parts written with ``combine: false``.

**Why.** Cheng approved removing the 1,014-line GFDL code, not the
behaviour it provides. Without a combiner, every multi-block netcdf output
would change from one file per frame to one file per block, and that
breaks what depends on the single file:

* the default ``combine: true`` (``output_type.hpp:85``);
* ``test_output_barrier`` ("a block is missing from the combined file",
  ``test_output_barrier.cpp:81``) and ``test_user_output.cpp:427``;
* the decomposition tests (``tests/CMakeLists.txt:719-754``), which run
  ``pd-combine`` on combined ``<case>.out<n>.<kkkkk>.nc`` files;
* the multi-block decks in section 9.1 (c).

The alternatives do not cover this:

* PnetCDF (kept) writes one file collectively, but only with
  ``blocks_per_process=1`` (``pnetcdf.cpp:79-83``), only in float
  (``:58-61``), without slices (``:51-53``) and only in ``PNETCDF=ON``
  builds (``meshblock.cpp:236-241``).
* Gathering every block to the root and writing once removes the part
  files, but puts the whole field in the root's memory.

``combine_parts`` reads only the attributes snapy writes
(``domain_decomposition``, ``NumFilesInSet``; ``netcdf.cpp:163-262``) and
has no file-scope state.

**What a user runs in place of the built-in combiner:**

* Default (``combine: true``): nothing. The run writes the combined
  ``<basename>.out<n>.<kkkkk>.nc`` itself, as today, through
  ``combine_parts``.
* Parts kept with ``combine: false``, or parts from an older run:
  ``snap-combine <basename>.out<n>.<kkkkk>.nc <basename>.block*.out<n>.<kkkkk>.nc``.
  The first argument is the output; the rest are the parts, as the library
  passes them today (``combine_netcdf.cpp:49-71``).
* Equivalent external tool, for users who already have it: GFDL
  FRE-NCtools' standalone ``mppnccombine -r <out.nc> <parts...>``. This is
  the same program and the same ``-r`` call the library makes today
  (``combine_netcdf.cpp:68-73``). **Hypothesis**: a current FRE-NCtools
  build reads snapy's NETCDF4 parts like the in-tree V2.1.7 copy; not
  tested.

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
  ``combine_parts`` and ``snap-combine``, section 10.1);
* the duplicate slice validation at ``meshblock.cpp:203-230``.

11. Open questions to other lanes
---------------------------------

**11.1 Lane R, Linfeng.** Requirements 3, 7 and 10 answered the earlier
questions on step integration, per-fixer amounts (the axial closure is a
fixer) and restart. The rest:

L-1. Which conserved rows does the register hold (species, scalars, the AAM
row of #357)? With which area weighting and frame (requirement 1)?

L-2. May the user output callback receive a read-only register view
(requirement 4), so user ``*_flux`` outputs stop rebuilding fluxes?

L-3. Is the outward transfer through the physical domain boundary available
directly, including on the cubed sphere after reconciliation (requirement
5)? Or does the budget sum it from the per-face entries?

L-4. In what dtype is the step transfer held in an FP32 run (requirement 8)?
The budget needs FP64 sums. Does R accumulate in FP64, or does lane O cast
at ``observe``?

L-5. Who resets the register and monitor on a redo? Is a post-accept hook
where ``observe`` runs guaranteed?

L-6. Does the monitor own the domain totals (requirement 6), or does lane
O's ``BudgetConsumer`` compute them from the state? This decides who owns the
cycle line.

**11.2 Lane X, Xi.**

X-1. Confirm that output, ``flush`` and restart run after the global accept
and outside captured segments.

X-2. Can the executor give post-step consumers ``dt`` as a device scalar?

X-3. Who calls graph and cache invalidation after ``read_restart``:
``MeshBlock`` or the executor?

**11.3 Lane C, Tianhao.**

* Lane C's draft names the layout lane "O" and diagnostics "EF"
  (happysky19/snapy ``b7d1300f``, ``c.rst`` section 12). Which lane owns the
  face-floor event counters that feed the monitor?

**11.4 Mesh owner (lane X, Xi), implicit owner (lane I, Xinyue), kickoff thread.**

* Reconcile the lane letters (above).
* Xi: accept the mesh items of section 8.1.
* Xinyue: accept the implicit items of section 8.1.
* Owners of option headers: when to switch to the generator.

12. Skeleton index
------------------

``docs/v3/skeleton/O/`` holds declarations only: no function bodies, 276
lines in total (``wc -l``), not built. The ``.hpp`` files were
syntax-checked with ``clang++ -fsyntax-only`` against a stub torch header.

.. list-table::
   :header-rows: 1
   :widths: 26 8 66

   * - file
     - lines
     - contents
   * - ``writer_core.hpp``
     - 97
     - ``GridKind``, ``Field``, ``Frame``, ``Sink``, ``GriddedWriter``,
       ``combine_parts``; #350 helpers kept (section 3.1)
   * - ``diagnostics.hpp``
     - 68
     - ``FluxRegisterView`` (requirement 3), ``MonitorView``
       (requirements 6, 7), ``Consumer``, ``BudgetConsumer``,
       ``HydroFluxProfile``, ``FieldDiagnostic`` (sections 3.2, 4)
   * - ``restart.hpp``
     - 48
     - The #367 contract, bundle magic, ``OutputSchedule``,
       ``BudgetCarry``, read/write/check; no register state (requirement
       10) (section 3.4)
   * - ``options_table.yaml``
     - 27
     - Table format; output and hydro sample rows, including
       ``SNAP_ANOMALY`` (the FP32 storage switches are removed, not tabled;
       section 3.3)
   * - ``gen_options.pyi``
     - 17
     - Generator interface (section 3.3)
   * - ``budget_series.cdl``
     - 19
     - Budget time-series format (section 3.2)
