:orphan:

V3 lane O: output, input, utilities, diagnostics, options schema
=================================================================

Revision 1, 2026-11-18. Owner: Jiachen (alternate: Isaac). Reviewers:
Tianhao (C) and Isaac (K). Design only: no implementation change is proposed
in this branch. Declarations are in ``docs/v3/skeleton/O/`` (not built). The
file-by-file line table is section 11 and ``docs/v3/design/O_lines.csv``.

Every ``path:line`` below is at chengcli/snapy
``9e5de2d4ce497d0b2d4da3295fc1fe969fe7e410`` (main, "FP32 step5 validation
(#368)") unless another commit is named. Line counts are ``wc -l`` at that
commit. Numbers marked *estimate* are design targets, not measurements.
Claims marked **hypothesis** have not been tested.

.. contents::
   :local:
   :depth: 1

1. Scope, sources and what was not available
--------------------------------------------

Lane O covers ``src/output``, ``src/input``, ``src/utils``,
``src/diagnostics``, ``src/math`` and ``src/z.junk``, and the options schema
(one table that generates option structs, deck keys, Python stubs and
documentation).

Read for this revision: every file in those directories (including
``radiative_flux.cpp_``, ``v1_moments.cpp_`` and ``z.junk``); the callers in
``src/mesh/meshblock.cpp`` (output set-up 201-255, ``make_outputs``
1207-1224, ``print_cycle_diagnostics`` 1226-1339, ``_init_from_restart``
1603-1830) and ``src/mesh/mesh.cpp:441-463``; ``python/csrc/snapy.cpp``,
``pyoutput.cpp``, ``pyoptions.hpp``; ``V3_ROADMAP.md``; the bodies of PRs
#313, #321, #323, #350, #352, #356 and #367 (GitHub REST API, read-only;
#321 and #323 have empty bodies, so only their titles and sizes were used).

**Tianhao's inventory at 919936da.** No document named as an inventory of
this lane was found in chengcli/snapy or happysky19/snapy. The closest match
is Tianhao's technical report on happysky19/snapy branch
``report/g5g6-919936da`` (head ``2634e2351b04eb04b810046eb431a19a98bb4cd5``),
whose code anchors are at ``919936da``: chapter 13 (``13-diagnostics.qmd:19-20``
states that ``src/diagnostics/`` is legacy Athena++/canoe code outside the
source glob), chapter 14 (``14-parallelism-io/_netcdf.qmd``, output option
table O-1 to O-36) and chapter 16 (known limits, e.g. "run-to-date meters are
not carried across a restart"). This revision uses it as the inventory and
rechecks every claim it takes from it at the pin. If a different inventory
was meant, please point to it; revision 2 will reconcile.

**Not available** (the worker that drafted this cannot read Slack): kickoff
briefs 1/3 and 3/3 beyond what the lane post quotes, the flux-register
requirements in kickoff brief 2/3, and the lane R and lane X designs. Section
3 states what this design assumes about them; section 14 lists the questions
for R and X.

2. Baseline recheck at the pin
------------------------------

``wc -l`` at the pin reproduces the lane post's 9,996 exactly when the
baseline is the ``.cpp``/``.hpp``/``.h``/``.cu``/``.cuh`` files:

.. list-table::
   :header-rows: 1

   * - directory
     - files
     - lines
   * - ``src/output``
     - 20
     - 4,487
   * - ``src/input``
     - 12
     - 2,041
   * - ``src/utils``
     - 22
     - 1,717
   * - ``src/diagnostics`` (excluding ``CMakeLists.txt`` 19 and the two
       ``.cpp_`` files, 108 + 73)
     - 15
     - 1,585
   * - ``src/math``
     - 2
     - 166
   * - **total**
     - **71**
     - **9,996**

Outside that baseline, and deleted here without being counted toward the
target: ``src/z.junk/load_athena_output_data.cpp_`` (585),
``radiative_flux.cpp_`` (108), ``v1_moments.cpp_`` (73),
``src/diagnostics/CMakeLists.txt`` (19), and ``src/interface/stride_iterator.hpp``
(140), whose only includers are in ``src/diagnostics``
(``rg -l stride_iterator src`` lists ``anomaly.cpp``, ``hydro_flux.cpp``,
``hydro_mean.cpp``, ``todo/convective_heat_flx.cpp``). Total 925 lines.

**Option counts in the lane post.** "``hydro.hpp`` has 32 option entries and
``forcing.hpp`` 48" matches ``rg -c ADD_ARG`` (32, 48), which also counts the
``#undef ADD_ARG`` line at the end of each header. The option entries are 31
and 47 (``rg -o 'ADD_ARG\(' src/hydro/hydro.hpp | wc -l`` gives 31; the same
for ``forcing.hpp`` gives 47). Over ``src/`` there are 213 entries
(``add_arg.h`` excluded) and 137 ``ADD_OPTION`` bindings in ``python/csrc``
(``pyoptions.hpp`` excluded).

**Environment switches.** The lane post lists three. At the pin
``rg -n 'getenv\(' src -g '!src/layout/**'`` gives four sites:

.. code-block:: text

   src/hydro/hydro.cpp:47:  if (std::getenv("SNAP_FLUX_COVARIANCE"))
   src/hydro/hydro.cpp:51:  if (std::getenv("SNAP_X1_MASS_COVARIANCE"))
   src/hydro/hydro_options.cpp:84:  if (char const* env = std::getenv("SNAP_ANOMALY")) {
   src/utils/fp32_storage.cpp:25:  char const* v = std::getenv(name);

``SNAP_FLUX_COVARIANCE`` and ``SNAP_X1_MASS_COVARIANCE`` are **only warned
about**: since #352 they have no effect (``hydro.cpp:46-54``, comment "the
environment switches these deck keys replaced are not read any more").
``SNAP_ANOMALY`` overrides the deck (``hydro_options.cpp:83-95``). The lane post
does not list ``SNAP_FP32_STORAGE`` and ``SNAP_FP32_STORAGE_ROWS``, read by
``fp32_storage.cpp:25`` for the Step-0 storage emulation
(``fp32_storage.hpp:19-25``; used at ``meshblock.cpp:622-638``).
``SNAP_X1_CENTROID_EXACT`` appears 49 times, but ``getenv`` reads it in no
file under ``src/``. **Hypothesis**: it is read through the ``layout.hpp:40``
helper, which this revision did not trace. ``MASTER_PORT`` and
``GLOO_SOCKET_IFNAME`` (``src/layout``) are launcher settings and are out of
scope.

3. Assumptions about lane R, lane X and kickoff brief 2/3
---------------------------------------------------------

Neither the flux-register requirements nor the R/X interfaces were available.
This design depends on the following. Each one is a question in section 14.

A1 **Flux register (R).** Per block, R keeps the face fluxes of every
   conserved row integrated over the accepted step with the RK weights
   (sum over stages of ``b_s dt F_s``), per axis, on the device. R resets it
   at the start of a step and restores it on a redo. Consumers read it only
   between accepted steps. Lane C's draft already ends its sweep with
   "register" (happysky19/snapy ``b7d1300f``, ``docs/v3/skeleton/c/sweep.hpp``),
   which agrees with this but does not fix the layout.

A1b The register accumulates in FP64 even in an FP32 run. A budget that
   should close to round-off cannot be built from FP32 face sums.
   (Today ``gravity_work_fix`` is a pending/commit/drop accumulator,
   ``hydro.hpp:301-311``. The register is assumed to generalise it.)

A2 **Monitor (R).** R exposes each fixer's change of each conserved row in
   the accepted step, and today's event counters, as device tensors: limiter
   cut/flux, positivity severe/min, VIC clamp residual, gravity-work fix, AAM
   closure sums (``hydro.hpp:235-311``; read on the host today at
   ``meshblock.cpp:1287-1296, 1336-1337``).

A3 **Execution (X).** Output, diagnostics flushes and restart writes run
   after the accept/retry decision and outside every captured segment.
   ``V3_ROADMAP.md`` "Execution design" already says "Global CFL/retry
   decisions and output stay outside graphs". X passes ``dt`` to consumers as
   a device scalar, so a consumer's per-step ``observe`` can run on device
   without a host value baked into a graph.

A4 **Rows of the budget.** The kickoff post names mass, species, E+PE, AAM,
   scalars and each fixer's contribution. This design takes these as the
   budget rows and fixer channels (section 6). E+PE uses the corrected
   potential energy the cycle line already uses when
   ``radial_exact_work()`` holds (``meshblock.cpp:1270-1283``, test at ``:1273``).

A5 **Lane naming.** Lane C's draft calls the layout lane "Lane O (layout)"
   and diagnostics "lane EF" (happysky19/snapy ``b7d1300f``, ``c.rst``
   section 12). This document follows the lane post: lane O is
   output/input/utils/diagnostics/options. Section 14 asks for the letters
   to be reconciled.

4. Question 1: what is legacy and what is still reachable
---------------------------------------------------------

"Compiled" means matched by ``file(GLOB src_files ...)`` in
``src/CMakeLists.txt:35-53`` (``output/*.cpp``, ``utils/*.cpp``,
``input/*.cpp``; no ``diagnostics/``, ``math/`` or ``z.junk/``) or by the CUDA
glob at ``src/CMakeLists.txt:122-136`` (``utils/*.cu``). Callers come from the
``rg`` runs quoted below, over ``src python tests examples``.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - file(s)
     - compiled
     - callers at the pin (verdict)
   * - ``input/parameter_input.{cpp,hpp}`` (993)
     - yes
     - **none.** The Athena++ ``<block> name=value`` parser
       (``parameter_input.cpp:1-40``). Its header is included only at
       ``python/csrc/snapy.cpp:10`` and ``src/output/hdf5.cpp:6``, and neither
       file uses a symbol from it. The other hits are comments in
       ``output_formats.hpp:15-74`` and code in the uncompiled
       ``src/diagnostics`` and ``src/turbulence``. No ``*.inp``/``athinput*``
       deck exists in the tree (``find . -name '*.inp' -o -name 'athinput*'``
       is empty). **Legacy, unreachable.**
   * - ``input/io_wrapper.{cpp,hpp}`` (364)
     - yes
     - **none** outside ``src/input`` (``rg -n 'io_wrapper|IOWrapper'``: no
       match). Only ``parameter_input.hpp:12`` includes it. **Legacy.**
   * - ``input/signal.{cpp,hpp}`` (187)
     - yes
     - **none** (``rg -n 'input/signal\.hpp|"signal\.hpp"'``: no match). An
       older copy of ``utils/signal_handler`` (the same SIGTERM/SIGINT/SIGALRM
       singleton API, ``signal.hpp:29-60`` vs ``signal_handler.hpp:20-64``),
       which is the live one (``mesh.cpp:28``, ``meshblock.cpp:24``).
       **Duplicate.**
   * - ``input/command_line.{cpp,hpp}`` (186)
     - yes
     - ``examples/run_hydro.cpp:23,212`` and
       ``examples/jupiter_evap_precip_1d.cpp:193,278`` only. **Live, examples
       only.**
   * - ``input/read_restart_file.{cpp,hpp}`` (257)
     - yes
     - ``meshblock.cpp:1609``, ``python/csrc/snapy.cpp:47``, 3 Python tests.
       **Live.**
   * - ``input/check_keys.{cpp,hpp}`` (54)
     - yes
     - 56 call sites over ``src`` (``hydro_options.cpp``, ``layout.cpp``,
       ``output_type.cpp:66``, ...). **Live.**
   * - ``output/mppnccombine.cpp`` (1,014)
     - yes
     - one call, ``combine_netcdf.cpp:73``. The GFDL tool (V2.1.7,
       ``mppnccombine.cpp:1-40``) is compiled into the library as a function.
       It keeps file-scope state (``static unsigned char _first_``,
       ``mppnccombine.cpp:93``) and takes ``argc/argv`` built with 2,048-byte
       ``new[]`` buffers (``combine_netcdf.cpp:64-75``). **Live, but a tool
       inside the library.**
   * - ``output/hdf5.cpp`` (27), ``HDF5Output``/``FITSOutput``
       (``output_formats.hpp:42-69, 98-104``)
     - body not compiled
     - ``HDF5OUTPUT`` is never defined (``configure.h.in`` defines only
       ``NETCDFOUTPUT``/``PNETCDFOUTPUT``, lines 12-16). The ``hdf5.cpp``
       definition has the old signature (``hdf5.cpp:23``, no ``Variables``)
       and would not match the header if it were compiled. The factory branch
       is commented out (``meshblock.cpp:242-244``). ``FITSOutput`` has no
       definition. **Stubs.** #356 kept them as "possibly meant to be filled
       in". This design decides to remove them (section 12).
   * - ``src/diagnostics/*`` (1,585 + 200 outside the baseline)
     - **no**
     - not in any glob, and no ``add_subdirectory`` reaches its
       ``CMakeLists.txt`` (``rg -n 'diagnostics/' src python tests examples
       CMakeLists.txt cmake -g '!src/diagnostics/**'``: no match). Includes
       headers that do not exist in snapy (``<athena/...>``,
       ``<exchanger/...>``, ``diagnostics.hpp:8-13``). Uses ``NVAPOR``,
       ``AthenaArray<Real>``, MPI ``PlanarExchanger``. The live per-cell
       diagnostics (thermo, ``div``, ``div_h``, ``curl``, ``ic_*``,
       ``path_*``, ``avg_*``) are in ``output/load_diag_output_data.cpp:54-338``.
       **Legacy, unreachable.**
   * - ``diagnostics/radiative_flux.cpp_``, ``v1_moments.cpp_``
     - no
     - drafts against Athena++/harp headers (``radiative_flux.cpp_:1-12``).
       **Legacy.**
   * - ``src/z.junk/load_athena_output_data.cpp_`` (585)
     - no
     - Athena++ output loader (``SELF_GRAVITY_ENABLED``, ``pgrav->phi``).
       Doxygen excludes it (``docs/doxygen/Doxyfile:133``). **Junk**; #356
       removed the rest of ``z.junk`` and missed this file, or left it.
   * - ``utils/pull_neighbors.*`` (249), ``utils/utils_dispatch.*`` (131),
       ``utils/cuda_utils.*`` (22)
     - yes
     - **no call.** ``rg -n 'pull_neighbors|bdot_out|checkCudaError' src python
       tests -g '!src/utils/**'`` finds only two commented-out lines
       (``equation_of_state.cpp:254,257``). ``cuda_utils.h`` is included at
       ``implicit_dispatch.cu:15``, but ``checkCudaError`` is never called.
   * - ``utils/flip_zero.{cpp,h}`` (350)
     - yes
     - ``tests/test_rectify.cpp:12`` only. The library uses the templated copy
       ``src/bc/flip_zero_impl.h`` (277 lines, included at
       ``bc/bc_dispatch.cpp:15``). Question 6.
   * - ``src/math/{lubksb,ludcmp}.h`` (166)
     - header-only
     - ``implicit/forward_sweep_impl.h:14-15`` and ``tests/test_lu_failure.cpp``.
       **Live, one caller module.**

Quoted ``rg`` output for the rows above (run at the pin):

.. code-block:: text

   $ rg -n "ParameterInput|parameter_input\.hpp" src python tests examples \
        -g "!src/input/**" -g "!src/diagnostics/**" -g "!src/turbulence/**"
   python/csrc/snapy.cpp:10:#include <snap/input/parameter_input.hpp>
   src/output/hdf5.cpp:6:#include <snap/input/parameter_input.hpp>
   src/output/output_formats.hpp:15:  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
   src/output/output_formats.hpp:23:  ... (lines 23, 30, 74: inside /* */ comments)

   $ rg -n "mppnccombine\(" src python tests examples
   src/output/combine_netcdf.cpp:17:int mppnccombine(int argc, char* argv[]);
   src/output/combine_netcdf.cpp:73:      err = mppnccombine(argc, argv);
   src/output/mppnccombine.cpp:94:int mppnccombine(int argc, char *argv[]) {

   $ rg -n "HDF5Output|FITSOutput|HDF5OUTPUT" src python tests
   src/output/hdf5.cpp:15,21,23,27; src/output/output_formats.hpp:43,46,98,100,101
   src/mesh/meshblock.cpp:244:            std::make_shared<HDF5Output>(out_op));*/

   $ rg -n "utils/flip_zero\.h|utils/refine\.hpp|utils/vectorize\.hpp|utils/fp64_store\.hpp|\
   utils/signal_handler\.hpp|utils/cuda_utils\.h" src python tests examples
   src/hydro/hydro.hpp:19     tests/test_rectify.cpp:12   src/output/netcdf.cpp:19
   src/output/pnetcdf.cpp:12  src/output/hdf5.cpp:8       src/output/output_type.cpp:9
   src/mesh/meshblock.cpp:24  src/mesh/mesh.cpp:28        src/implicit/implicit_dispatch.cu:15
   tests/test_refine.cpp:11

5. Question 2: one writer core
------------------------------

**What #350 did** (merged ``ffaffdd5``). It created ``output/writer_core.{hpp,cpp}``
(186 lines at the pin) with ``output_range``, ``make_output_dir``,
``output_block_location``, ``combine_on_root`` (per-process block counter,
barrier, scope-guard mirror barrier) and ``glob_block_files``. Its body:
"output core: −2" net lines, bitwise outputs on five decks. It shared the
*plumbing*. It did not touch the *format* code.

**What is left**, at the pin:

* Two copies of the frame layout. ``netcdf.cpp`` (583) and ``pnetcdf.cpp``
  (431) each define dims and coordinates, expand vector names
  (``netcdf.cpp:287-318``; ``pnetcdf.cpp:209-236``), map the grid code to
  dimension ids (``netcdf.cpp:320-348``), and pack seven grid kinds with
  hand loops (``netcdf.cpp:490-562``; ``pnetcdf.cpp:350-415``). They differ
  in the library calls and in slicing: ``pnetcdf`` refuses slices
  (``pnetcdf.cpp:51-53``), ``double_precision`` (``:58-61``) and more than
  one block per process (``:79-83``).
* A hand-managed doubly linked list of ``OutputData`` holding
  ``AthenaArray<double>`` copies (``output_type.hpp:93-105, 201-207``;
  ``output_type.cpp:201-244``), with slice and sum as four nested loops per
  case (``output_type.cpp:415-557``).
* Name splitting with ``strtok`` into a 1,028-byte stack buffer
  (``utils/vectorize.cpp:12-22``). The define pass splits on ``;``
  (``netcdf.cpp:269, 288``) and the write pass on ``,`` (``netcdf.cpp:486``,
  ``pnetcdf.cpp:352``). **Hypothesis** (not tested): ``strtok`` is not
  re-entrant, so two blocks of one process writing at the same time would
  race. ``combine_on_root`` uses a mutex (``writer_core.cpp:81-93``), which
  means concurrent block writers are expected. Today ``Mesh::make_outputs``
  loops over blocks serially (``mesh.cpp:447-449``); a Python driver that
  calls ``MeshBlock.make_outputs`` from threads is the case at risk.
* ``mppnccombine.cpp``, a 1,014-line general-purpose combiner, joins files
  that snapy wrote itself. Every part carries the ``domain_decomposition``
  attribute per axis (``netcdf.cpp:163-237``) and ``NumFilesInSet``
  (``:262``).
* Per-element host reads of coordinates: ``.item<double>()`` inside the
  coordinate loops (``netcdf.cpp:433-481``, six sites) and in the slice
  search (``output_type.cpp:438``). Question 7.

**Design** (skeleton ``writer_core.hpp``):

* ``Field`` (names, ``GridKind``, units, long name, one tensor) and
  ``Frame`` (fields, coordinates, per-axis placement) replace ``OutputData``
  and the list. Slice and sum become one ``index_select``/``sum`` per field.
  ``Frame::to_host()`` is the one host copy per field.
* ``Sink`` is the format: ``open/define/end_define/put/put_coords/close``,
  about 120-140 lines each for netcdf and pnetcdf (*estimate*). All layout
  arithmetic (placement, face counts, cubed-sphere 3 x 2 offsets) is computed
  once in the core.
* ``GriddedWriter`` replaces ``OutputType`` and the four format classes. It
  keeps the schedule (``file_number``, ``next_time``, both schedule keys of
  #313) and the time statistics (``output_type.cpp:285-381``) unchanged.
* ``combine_parts(outfile, parts)`` replaces the call into ``mppnccombine``.
  It reads the attributes snapy writes and copies each variable's hyperslab
  into the combined file, about 200 lines (*estimate*). **Gate**: on the 25
  example decks and the #350 decks (6-panel cubed sphere, slices, ghost
  zones, double precision), every variable of every combined file equals
  ``mppnccombine``'s output bit for bit, and so do the dimension and
  attribute lists. Until that gate passes, ``mppnccombine.cpp`` stays.
* Moves out of ``src/``: the combiner CLI (``tools/snap_combine.cpp``,
  about 30 lines, for ``combine: false`` runs; today the tech report says
  "Joining them later needs an external ``mppnccombine``",
  ``_netcdf.qmd:106``), and ``input/command_line.*`` to ``examples/`` (its
  only callers). ``mppnccombine.cpp`` itself is deleted, not moved: once
  ``combine_parts`` passes the gate, nothing calls it.
* Output slice validation runs twice today (``output_type.cpp:90-98`` and
  ``meshblock.cpp:203-230``). It moves into the generated option checks
  (section 7), and the ``meshblock.cpp`` copy goes (that copy is lane M's line,
  noted in section 14).

6. Question 3: diagnostics as consumers of the register and monitor
-------------------------------------------------------------------

**Today.** ``src/diagnostics`` is dead (section 4). The live end-of-step
"monitor" is ``print_cycle_diagnostics`` (``meshblock.cpp:1226-1339``). Every
``ncycle_out`` cycles it computes domain totals of every conserved row, KE and
PE, reduces them over the process group, and prints one line. It reads meters
with ``.cpu()`` (``:1288-1295``) and prints with ``.item<double>()``
(``:1319-1337``). It writes nothing to a file. The convection examples write
their own CSV time series (``examples/planar_convection.cpp:3``,
``spherical_convection.cpp:19``, ``cubed_sphere_convection.cpp:3``). The
legacy ``HydroFlux`` diagnostic recomputed ``rho * v1`` from cell centres
(``diagnostics/hydro_flux.cpp:56-63``). That is not the flux the scheme used.

**Design** (skeleton ``diagnostics.hpp``). A ``Consumer`` has ``observe``
(device only, every accepted step, gets the ``FluxRegisterView``,
``MonitorView`` and a device ``dt``) and ``flush`` (collective, then a host
read, only at its output cadence). Two consumers replace the dead module:

* ``BudgetConsumer``. For each row Q it accumulates, in FP64 on device,
  ``boundary`` (the outward register flux through the physical domain
  boundary, summed over the interval) and ``fixer[f]`` (the monitor's
  per-fixer change). At ``flush`` it also takes ``total`` (the domain
  integral) and writes ``residual = (total - total_prev) + boundary -
  sum fixer``. A budget that closes shows ``residual`` at round-off.
* ``HydroFluxProfile``. The horizontal mean of the registered x1 face flux,
  time-integrated, on grid ``--F``. It is the intent of the legacy
  ``hydro_flux.cpp``, computed from the scheme's own fluxes.

The per-cell diagnostics stay field loaders of the writer core. A
``FieldDiagnostic`` registry (key, grid, units, compute) replaces the
``ContainVariable`` chains in ``load_diag_output_data.cpp`` (11 lines
call it; ``rg -c 'ContainVariable\(' src/output/load_diag_output_data.cpp``).

**Proposed budget time-series format** (skeleton ``budget_series.cdl``). One
file per run, ``<output_dir>/<basename>.budget.nc``. NetCDF4 with an
unlimited ``time`` dimension, appended by the root rank at each budget
interval, and **every variable NC_DOUBLE**. Variables: ``time``, ``cycle``,
``dt``, ``interval``; ``row_name``/``row_units`` (rows: ``mass``,
``species/<name>``, ``energy`` (E+PE), ``aam``, ``scalar/<name>``);
``fixer_name`` (``limiter``, ``positivity``, ``vic_clamp``,
``gravity_work``, ``aam_closure``); ``total(time,row)``,
``boundary(time,row)``, ``fixer(time,fixer,row)``,
``residual(time,row)``; the monitor counters as scalars per record. It is
requested like any output: ``- {type: budget, dt: 3600.}``.

Why this format (open choice O-3 in section 13):

* NetCDF is the format every snapy reader already handles (xarray,
  ``pd_inspect.py``). It carries units and names, and ``time`` UNLIMITED
  makes appending a record cheap. CSV loses units, needs a schema next to it,
  and drifted per example (three different column sets in the convection
  drivers).
* Rows and fixers are *dimensions*, not variable-name suffixes, so adding a
  species or a fixer changes no reader.
* Double only, because the point is relative changes far below the 1e-7
  that the default ``NC_FLOAT`` resolves (``output_type.hpp:79-83``).
* The ``BudgetSeries`` writer goes through the same netcdf ``Sink`` as the
  gridded writer (about 120 lines, *estimate*), so it adds no new I/O
  dependency.
* Restart carries the pending interval sums (``BudgetCarry``, section 8), so
  an interval that straddles a restart closes. The tech report lists the loss
  of run-to-date meters across restart as a known limit (chapter 16).

7. Question 4: one options table
--------------------------------

**Today.** One option is written in up to six places: the ``ADD_ARG``
declaration with its default (``add_arg.h``), the ``report()`` line, the key
list for ``check_keys`` and the typed read in ``from_yaml`` (with a second
default, the bug class #352 fixed for three options), the pybind
``ADD_OPTION`` line, the ``.pyi`` stub and the documentation. The counts show
the drift: 213 ``ADD_ARG`` entries against 137 ``ADD_OPTION`` bindings
(section 2). This revision did not match them name by name, so the 76 is a
count difference, not a list of unbound options. ``hydro.hpp`` alone has 31
entries and 20 bindings (``rg -c 'ADD_OPTION\(' python/csrc/pyhydro.cpp``);
``forcing.hpp`` has 47 and 19.

**Design** (skeleton ``options_table.yaml``, ``gen_options.py``). One YAML
table per module, one row per option: ``name, key, type, default, env,
python, doc``, plus cross-option ``checks``. ``gen_options.py`` emits (1)
the C++ struct (``ADD_ARG``, ``report()``, ``clone()``), (2) ``from_yaml``
with the key list passed to ``check_keys`` and one default per option,
(3) the pybind lines, (4) the ``.pyi`` stubs (checked in, compared in CI)
and (5) an ``rst`` page per module. It runs at CMake configure time and in
``setup.py``. A one-time ``--import`` reads the pin's headers to seed the
tables, and a test checks that the generated structs have the same members
and defaults as the pin.

**Environment switches.** Each switch becomes a table row with an ``env``
column:

* ``SNAP_FLUX_COVARIANCE``, ``SNAP_X1_MASS_COVARIANCE``: delete the warning
  stubs (``hydro.cpp:46-54``) one release after #352. They have had no
  effect since #352.
* ``SNAP_ANOMALY``: keep as the only ``env`` override (``anomaly`` row). It
  is used by restart guidance ("Resume with SNAP_ANOMALY=0",
  ``meshblock.cpp:1782``) and appears on 5 lines in ``tests/``
  (``rg -n SNAP_ANOMALY tests | wc -l``). **FOR CHENG**: keep
  the override, or make tests pass ``dynamics/anomaly``?
* ``SNAP_FP32_STORAGE``/``_ROWS``: become ``dynamics/fp32-storage`` and
  ``dynamics/fp32-storage-rows`` rows. The ``getenv`` path goes
  (``fp32_storage.cpp`` 119 -> 95, *estimate*). **FOR CHENG**: the Step-0
  emulation may no longer be wanted now that FP32 arithmetic is merged
  (#361). If so, the 175 lines go entirely.

**Scope.** The generator and tables are new lines outside ``src/`` (about 600,
*estimate*; counted in section 11). The lines they remove are in other
lanes' headers and bindings (``ADD_ARG`` blocks, ``from_yaml`` bodies,
``python/csrc/py*.cpp``, 2,753 lines of ``.pyi``), so they are **not**
claimed by this lane's line table. Lanes that own those files decide when
to switch.

8. Question 5: restart
----------------------

Skeleton ``restart.hpp`` holds the normative stored/derived table.

**Stored** (all kept from the pin): ``hydro_u`` as FP64 totals
(``restart.cpp:44``); ``hydro_u_stored`` (the anomaly bits in the run dtype,
``:45``); ``hydro_ref``, ``hydro_ref_species`` and the reference grid faces
``hydro_ref_x{1,2,3}f`` (#367, ``:46-61``); ``hydro_restart_metadata``
[version, stored-state bits] (``:50-53``); ``scalar_s`` (``scalar_r`` is
dropped when it can be rebuilt, ``:30-34``); ``fill_solid_hydro_u`` and the
boundary references when present; ``last_time``, ``last_cycle``; the output
schedule (``file_number``, advanced ``next_time``, N legacy plus N v2 keys,
``:73-99``, from #313). New: ``BudgetCarry``, the pending boundary and fixer
sums since the last budget record.

**Derived** (never written): ``hydro_w``, ``scalar_r``
(``meshblock.cpp:1827-1829``); the flux register and monitor counters (reset
at load: they hold one step); CUDA graph executables, captured workspaces and
cast caches (``V3_ROADMAP.md``: "Derived caches and graph executables are
never serialized into restart files"); coordinates, metrics and
reconstruction coefficients.

**#367 metadata kept unchanged**: ``kHydroRestartVersion = 1``, the metadata
name and the grid keys (``restart_metadata.hpp:9-12``), and the checks at
``meshblock.cpp:1623-1653`` (version, dtype bits, shape, grid equality,
perturbation-flux backgrounds). They move into ``check_restart_metadata`` next
to the file contract, and ``MeshBlock`` calls it.

**Changes:**

* The bundle writer (``combine_restart.cpp``) and reader
  (``read_restart_file.cpp``) merge into ``restart.cpp`` with one
  ``kRestartBundleMagic``. Today it is defined twice
  (``combine_restart.cpp:16``, ``read_restart_file.cpp:23``).
* The reader **throws** where it now prints to ``std::cerr`` and returns an
  empty map (``read_restart_file.cpp:157-220``). Today the caller then stops
  on "missing required variable: hydro_u" (``meshblock.cpp:1612``), which
  hides the real cause (a truncated bundle or a wrong block).
* **Hypothesis**: the reader extracts each block to a temporary file before
  ``kintera::load_tensors`` (``read_restart_file.cpp:177-216``). Reading
  through a memory stream would avoid that, if ``kintera::load_tensors``
  accepts one. Not checked in kintera; question for whoever owns that API.
* The bundle format (``SNAPY_RESTART_BUNDLE_V1``) and the file names stay, so
  every existing restart file still loads.

9. Question 6: ``src/utils``
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - helper
     - callers (``rg``, section 4)
     - decision
   * - ``fp64_store.hpp`` (85)
     - ``hydro/hydro.hpp:19`` only
     - **move** to ``src/hydro``
   * - ``signal_handler.*`` (213)
     - ``mesh.cpp:28``, ``meshblock.cpp:24``; takes ``MeshBlockImpl``
       (``signal_handler.hpp:45``)
     - **move** to ``src/mesh``; delete its duplicate ``input/signal.*``
   * - ``refine.*`` (80)
     - ``output_type.cpp:9`` (super-resolution), ``tests/test_refine.cpp``;
       ``conservative_coarsen`` is used only by the test
     - **delete** with ``super_resolution``. It is used by no deck and
       ``tests/test_user_output.cpp:1144`` only, and it builds a whole second
       ``MeshBlockImpl`` with raw ``new`` per write
       (``output_type.cpp:143-187``). **FOR CHENG**: Python bindings expose it
       (``pyoutput.cpp:43``); is it used from a notebook?
   * - ``vectorize.*`` (70)
     - ``netcdf.cpp``, ``pnetcdf.cpp``, ``hdf5.cpp``
     - **delete**; ``Field.names`` is already a vector
   * - ``flip_zero.{cpp,h}`` (350)
     - ``tests/test_rectify.cpp`` only
     - **delete**; port the test to ``bc/flip_zero_impl.h``
   * - ``pull_neighbors.*``, ``utils_dispatch.*``, ``cuda_utils.*``
     - none
     - **delete**
   * - ``fp32_storage.*`` (175)
     - ``meshblock.cpp:622-638``, ``tests/test_fp32_storage.cpp``
     - keep; env parsing replaced by option rows (section 7)
   * - ``log.hpp`` (67), ``loops.cuh`` (150), ``on_host.hpp`` (72),
       ``tensor_cast_cache.hpp`` (53)
     - 11, 8, 3 and 3 including files
     - **keep** (shared infrastructure)
   * - ``src/math/{lubksb,ludcmp}.h`` (166)
     - ``implicit/forward_sweep_impl.h`` only
     - **move** to ``src/implicit`` (lane I decides)

*flip_zero.cpp against src/bc/flip_zero_impl.h.* The ``utils``
copy is the int-only original (``flip_zero.h:30-80``:
``initialize_dp(int n, int *dp, ...)``, ``compute_min_flips(const int
*inSeq, ...)``). The ``bc`` copy is the templated one the library uses
(``flip_zero_impl.h:24-202``: ``initialize_dp(int n, T *dp, ...)``,
``compute_min_flips(const T *inSeq, ...)``). The functions are the same
four (``initialize_dp``, ``can_flip``, ``compute_min_flips``,
``reconstruct_solution``), except that ``reconstruct_solution`` writes
in place in the ``bc`` copy (``flip_zero_impl.h:202``, no ``outSeq``).
**Hypothesis**: the DP is otherwise identical. A line-by-line diff was not
done. The ported ``test_rectify`` (``T = int``) is the check: if the
``bc`` copy passes the ``utils`` copy's test cases, the ``utils`` copy goes.

10. Question 7: output and diagnostics under FP32 and CUDA graphs
-----------------------------------------------------------------

Host reads on the output path at the pin (``rg -n '\.item<|\.cpu\(\)|to\(torch::kCPU\)|CopyFromTensor|InitFromTensor'``,
counted per file): ``load_diag_output_data.cpp`` 14, ``netcdf.cpp`` 6,
``pnetcdf.cpp`` 6, ``restart.cpp`` 7, ``output_type.cpp`` 3,
``load_hydro_output_data.cpp`` 2, ``meshblock.cpp`` 34 (cycle line,
restart load, redo and others).

**Output-only, and outside captured segments** (assumption A3):

* every ``GriddedWriter::write``: ``Frame::to_host`` (one copy per field),
  coordinates (one copy per axis, replacing per-cell ``.item``,
  ``netcdf.cpp:433-481``, ``output_type.cpp:438``);
* restart writes (``.to(kCPU)``, ``restart.cpp:36-66``);
* ``Consumer::flush`` (the budget collective and its host read) and the cycle
  line. The cycle line becomes a printer over the ``BudgetConsumer``'s
  device totals, so its ``.cpu()``/``.item`` (``meshblock.cpp:1288-1337``)
  happen only every ``ncycle_out`` steps, as today.

**On device every step, capturable** (inside or right after a segment, X
decides): ``Consumer::observe`` and ``GriddedWriter::accumulate_stats``.
Today ``AccumulateStats`` takes the host difference ``current_time -
stat_last_time_`` (``output_type.cpp:308``) and runs every step
(``meshblock.cpp:1210``). In V3 it takes ``dt`` as a device scalar (A3), so
a capture does not bake the first step's ``dt`` into the graph.

**FP32.** Gridded outputs read totals: ``total_u`` adds the FP64 reference
(``load_hydro_output_data.cpp:38``, ``meshblock.cpp:1257``), unchanged. The
budget accumulates in FP64 (A1b) and writes ``NC_DOUBLE``. The netcdf
default stays ``NC_FLOAT``, as documented at ``output_type.hpp:79-83``.
Restart keeps #367's dtype rules.

11. Question 8: line table
--------------------------

Baseline per file is ``git show 9e5de2d4:<path> | wc -l``. "proposed" is
what stays in the same file (or the new file), "moved" leaves the lane
unchanged (not counted as cut), "cut" = pin − proposed − moved. Proposed
lines of rewritten and new files are *estimates*. The CSV
(``O_lines.csv``) has the same rows plus the 925 lines deleted outside the
baseline.

.. list-table::
   :header-rows: 1
   :widths: 34 7 7 7 7 9 29

   * - file
     - pin
     - proposed
     - moved
     - cut
     - action
     - note / deletion named
   * - ``src/diagnostics/anomaly.cpp``
     - 97
     - 0
     - 
     - 97
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/buoyancy.cpp``
     - 61
     - 0
     - 
     - 61
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/curl.cpp``
     - 118
     - 0
     - 
     - 118
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics.cpp``
     - 52
     - 0
     - 
     - 52
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics.hpp``
     - 246
     - 0
     - 
     - 246
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/diagnostics_factory.cpp``
     - 52
     - 0
     - 
     - 52
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/divergence.cpp``
     - 84
     - 0
     - 
     - 84
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/hydro_flux.cpp``
     - 120
     - 0
     - 
     - 120
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/hydro_mean.cpp``
     - 100
     - 0
     - 
     - 100
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/convective_heat_flx.cpp``
     - 130
     - 0
     - 
     - 130
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/eddy_flux.cpp``
     - 125
     - 0
     - 
     - 125
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/eddy_kinetic_energy.cpp``
     - 89
     - 0
     - 
     - 89
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/latent_heating.cpp``
     - 112
     - 0
     - 
     - 112
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/spherical_angular_momentum.cpp``
     - 86
     - 0
     - 
     - 86
     - delete
     - Athena++/canoe; never compiled
   * - ``src/diagnostics/todo/tendency.cpp``
     - 113
     - 0
     - 
     - 113
     - delete
     - Athena++/canoe; never compiled
   * - ``src/input/check_keys.cpp``
     - 33
     - 33
     - 
     - 0
     - keep
     - called by generated parsers
   * - ``src/input/check_keys.hpp``
     - 21
     - 21
     - 
     - 0
     - keep
     - 
   * - ``src/input/command_line.cpp``
     - 130
     - 0
     - 130
     - 0
     - move
     - to examples/ (only callers: run_hydro.cpp, jupiter_evap_precip_1d.cpp)
   * - ``src/input/command_line.hpp``
     - 56
     - 0
     - 56
     - 0
     - move
     - to examples/
   * - ``src/input/io_wrapper.cpp``
     - 305
     - 0
     - 
     - 305
     - delete
     - only reached from parameter_input
   * - ``src/input/io_wrapper.hpp``
     - 59
     - 0
     - 
     - 59
     - delete
     - 
   * - ``src/input/parameter_input.cpp``
     - 873
     - 0
     - 
     - 873
     - delete
     - legacy Athena++ parser; no caller
   * - ``src/input/parameter_input.hpp``
     - 120
     - 0
     - 
     - 120
     - delete
     - only an unused include in python/csrc/snapy.cpp:10 and hdf5.cpp:6
   * - ``src/input/read_restart_file.cpp``
     - 236
     - 0
     - 
     - 236
     - merge
     - into output/restart.cpp
   * - ``src/input/read_restart_file.hpp``
     - 21
     - 0
     - 
     - 21
     - merge
     - into restart.hpp
   * - ``src/input/signal.cpp``
     - 124
     - 0
     - 
     - 124
     - delete
     - older copy of utils/signal_handler; no includer
   * - ``src/input/signal.hpp``
     - 63
     - 0
     - 
     - 63
     - delete
     - 
   * - ``src/math/lubksb.h``
     - 71
     - 0
     - 71
     - 0
     - move
     - to src/implicit (one includer: forward_sweep_impl.h)
   * - ``src/math/ludcmp.h``
     - 95
     - 0
     - 95
     - 0
     - move
     - to src/implicit
   * - ``src/output/combine_netcdf.cpp``
     - 87
     - 0
     - 
     - 87
     - merge
     - into writer_core.cpp (combine step, no argv building)
   * - ``src/output/combine_restart.cpp``
     - 123
     - 0
     - 
     - 123
     - merge
     - into restart.cpp (bundle writer)
   * - ``src/output/hdf5.cpp``
     - 27
     - 0
     - 
     - 27
     - delete
     - stub; body only under never-defined HDF5OUTPUT, signature does not match the header
   * - ``src/output/load_diag_output_data.cpp``
     - 338
     - 260
     - 
     - 78
     - rewrite
     - field-diagnostic registry replaces ContainVariable chains
   * - ``src/output/load_hydro_output_data.cpp``
     - 186
     - 150
     - 
     - 36
     - rewrite
     - appends Fields; names from the field registry
   * - ``src/output/load_scalar_output_data.cpp``
     - 64
     - 50
     - 
     - 14
     - rewrite
     - appends Fields
   * - ``src/output/load_user_output_data.cpp``
     - 27
     - 20
     - 
     - 7
     - rewrite
     - appends Fields
   * - ``src/output/mppnccombine.cpp``
     - 1014
     - 0
     - 
     - 1014
     - delete
     - GFDL tool; replaced by combine_parts in writer_core.cpp
   * - ``src/output/netcdf.cpp``
     - 583
     - 140
     - 
     - 443
     - rewrite
     - netcdf Sink only
   * - ``src/output/netcdf_utils.hpp``
     - 29
     - 0
     - 
     - 29
     - merge
     - sanitize_netcdf_name into writer_core.cpp
   * - ``src/output/output_formats.hpp``
     - 105
     - 0
     - 
     - 105
     - delete
     - HDF5Output, FITSOutput, commented classes; Netcdf/PNetcdf classes become Sinks
   * - ``src/output/output_type.cpp``
     - 559
     - 230
     - 
     - 329
     - rewrite
     - options parser generated; linked list and slice/sum loops replaced by torch ops
   * - ``src/output/output_type.hpp``
     - 219
     - 90
     - 
     - 129
     - rewrite
     - options struct generated; GriddedWriter only
   * - ``src/output/output_utils.cpp``
     - 301
     - 120
     - 
     - 181
     - rewrite
     - metadata of live fields only (unused canoe-era names dropped)
   * - ``src/output/output_utils.hpp``
     - 65
     - 30
     - 
     - 35
     - rewrite
     - get_num_variables replaced by Field shape
   * - ``src/output/pnetcdf.cpp``
     - 431
     - 130
     - 
     - 301
     - rewrite
     - pnetcdf Sink only
   * - ``src/output/restart.cpp``
     - 130
     - 260
     - 
     - -130
     - rewrite
     - writer + bundle + reader in one file
   * - ``src/output/restart_metadata.hpp``
     - 13
     - 13
     - 
     - 0
     - keep
     - #367 contract
   * - ``src/output/writer_core.cpp``
     - 138
     - 420
     - 
     - -282
     - rewrite
     - Frame transform, packing, combine_parts (~200)
   * - ``src/output/writer_core.hpp``
     - 48
     - 110
     - 
     - -62
     - rewrite
     - Frame, Field, Sink, GriddedWriter
   * - ``src/utils/cuda_utils.cu``
     - 19
     - 0
     - 
     - 19
     - delete
     - checkCudaError has no caller
   * - ``src/utils/cuda_utils.h``
     - 3
     - 0
     - 
     - 3
     - delete
     - one unused include in implicit_dispatch.cu:15
   * - ``src/utils/flip_zero.cpp``
     - 270
     - 0
     - 
     - 270
     - delete
     - int-only copy of bc/flip_zero_impl.h; test ported
   * - ``src/utils/flip_zero.h``
     - 80
     - 0
     - 
     - 80
     - delete
     - 
   * - ``src/utils/fp32_storage.cpp``
     - 119
     - 95
     - 
     - 24
     - rewrite
     - env parsing replaced by options table key
   * - ``src/utils/fp32_storage.hpp``
     - 56
     - 55
     - 
     - 1
     - keep
     - 
   * - ``src/utils/fp64_store.hpp``
     - 85
     - 0
     - 85
     - 0
     - move
     - to src/hydro (one includer: hydro.hpp)
   * - ``src/utils/log.hpp``
     - 67
     - 67
     - 
     - 0
     - keep
     - 11 includers
   * - ``src/utils/loops.cuh``
     - 150
     - 150
     - 
     - 0
     - keep
     - 8 CUDA dispatch files
   * - ``src/utils/on_host.hpp``
     - 72
     - 72
     - 
     - 0
     - keep
     - 3 dispatch files (MPS)
   * - ``src/utils/pull_neighbors.cpp``
     - 225
     - 0
     - 
     - 225
     - delete
     - only commented-out calls (equation_of_state.cpp:254-257)
   * - ``src/utils/pull_neighbors.hpp``
     - 24
     - 0
     - 
     - 24
     - delete
     - 
   * - ``src/utils/refine.cpp``
     - 69
     - 0
     - 
     - 69
     - delete
     - only caller: output super-resolution (deleted, FOR CHENG)
   * - ``src/utils/refine.hpp``
     - 11
     - 0
     - 
     - 11
     - delete
     - 
   * - ``src/utils/signal_handler.cpp``
     - 146
     - 0
     - 146
     - 0
     - move
     - to src/mesh (callers mesh.cpp, meshblock.cpp)
   * - ``src/utils/signal_handler.hpp``
     - 67
     - 0
     - 67
     - 0
     - move
     - to src/mesh
   * - ``src/utils/tensor_cast_cache.hpp``
     - 53
     - 53
     - 
     - 0
     - keep
     - 3 includers
   * - ``src/utils/utils_dispatch.cpp``
     - 26
     - 0
     - 
     - 26
     - delete
     - bdot_out has no caller
   * - ``src/utils/utils_dispatch.cu``
     - 92
     - 0
     - 
     - 92
     - delete
     - 
   * - ``src/utils/utils_dispatch.hpp``
     - 13
     - 0
     - 
     - 13
     - delete
     - 
   * - ``src/utils/vectorize.cpp``
     - 24
     - 0
     - 
     - 24
     - delete
     - strtok split; replaced by a split in writer_core
   * - ``src/utils/vectorize.hpp``
     - 46
     - 0
     - 
     - 46
     - delete
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
     - **3089**
     - **650**
     - **6257**
     - 
     - src only: 62.6% cut
   * - **including new files outside src**
     - 
     - **630**
     - 
     - **5627**
     - 
     - 56.3% cut; target 30% = 2,999

**Totals.**

* Baseline 9,996 (71 files). Kept or rewritten in place: 2,569. New in
  ``src/``: 520. Moved out unchanged: 650 (``command_line`` 186 to
  ``examples/``, ``fp64_store`` 85 to ``hydro``, ``signal_handler`` 213 to
  ``mesh``, ``math`` 166 to ``implicit``).
* Cut, ``src/`` only: **6,257 lines, 62.6 %**.
* Cut including the new files outside ``src/`` (``gen_options.py``,
  option tables, ``snap_combine``: 630): **5,627 lines, 56.3 %** against the
  30 % target (2,999 lines).
* Sensitivity: if every rewritten and new file comes in 50 % above its
  estimate (3,255 estimated lines, +1,627), the cut is 4,000 lines, 40.0 %.
* Named deletions (pure removals, no replacement): ``parameter_input.*``
  993, ``io_wrapper.*`` 364, ``input/signal.*`` 187, ``src/diagnostics``
  1,585, ``hdf5.cpp`` 27, ``output_formats.hpp`` 105 (folded),
  ``pull_neighbors.*`` 249, ``utils_dispatch.*`` 131, ``cuda_utils.*`` 22,
  ``utils/flip_zero.*`` 350, ``refine.*`` 80, ``vectorize.*`` 70. Total
  4,163. Replaced by smaller code: ``mppnccombine.cpp`` 1,014 (by
  ``combine_parts``, about 200), the netcdf/pnetcdf/``output_type``
  duplication, and the restart merge.
* Plus 925 lines outside the baseline (section 2).

12. Decisions
-------------

**D1 Output formats that remain.**

* ``netcdf`` (per-block parts plus root combine): **stays**, the primary
  format. 18 of the 25 ``examples/*.yaml`` and 4 files in ``tests/`` request
  it (``rg -l 'type: *"?netcdf' examples | wc -l``; same for ``tests``).
* ``restart``: **stays**, format unchanged (bundle ``V1``, #367 metadata).
* ``pnetcdf``: **stays, as a Sink** (about 130 lines instead of 431).
  Reasons: two example decks use it (``examples/straka.yaml:57``,
  ``examples/shallow_splash.yaml:58``); it is the only path that avoids a
  root-side combine for large runs; and with the core it costs one sink.
  **FOR CHENG**: no test runs it on several ranks (the test drivers replace
  it by netcdf, ``tests/run_straka.cmake:28``,
  ``tests/run_shallow_splash.cmake:28``). Is a production workflow on it? If
  not, the 130 lines and the PnetCDF build option can go too.
* ``budget`` (new): the time series of section 6.
* ``hdf5``, ``fits``: **removed** (stubs, never built, section 4).
  **FOR CHENG**: #356 kept them "possibly meant to be filled in". Was an HDF5
  or FITS workflow planned?
* ``super-resolution`` output option: **removed** with ``utils/refine``
  (section 9, FOR CHENG there).
* ``mppnccombine``: **removed from the library** once ``combine_parts``
  passes its gate. **FOR CHENG**: does anyone run snapy's built-in
  ``mppnccombine`` on files that snapy did not write? ``combine_parts`` only
  understands snapy's own attributes.

**D2 The legacy input parser goes.** ``parameter_input.*`` and
``io_wrapper.*`` (1,357 lines) have no caller, no deck in that format exists,
and every module reads YAML through ``from_yaml`` plus ``check_keys``
(section 4). Removing them also removes the dead ``#include`` at
``python/csrc/snapy.cpp:10``. No workflow is known to depend on them. If one
does, ``FOR CHENG``: a converter from the Athena++ format to YAML belongs in
``tools/``, not in the library.

**D3 ``src/diagnostics`` goes and is replaced by register consumers**
(section 6). The todo list it carried (eddy flux, EKE, latent heating,
tendency, convective heat flux, spherical AAM) becomes budget rows or
``FieldDiagnostic`` entries when someone needs them, written against the
register, not ported.

13. Open choices made in this revision, with reasons
----------------------------------------------------

O-1 **Baseline definition**: the ``.cpp_`` files and
    ``diagnostics/CMakeLists.txt`` are outside the 9,996. That is the only
    reading that reproduces the lane post's number. They are deleted, but
    not counted.

O-2 **Moves are not cuts.** Moving ``command_line``, ``fp64_store``,
    ``signal_handler`` and ``math`` out of the lane's directories leaves the
    repository's size unchanged, so they are reported separately.

O-3 **Budget format NetCDF, not CSV/JSON** (section 6). It reuses the writer
    core's sink and the readers people already use, and it keeps units with
    the data.

O-4 **pnetcdf stays** (D1). Keeping it costs about 130 lines with the
    core. Removing it would break two example decks without a decision from
    Cheng.

O-5 **mppnccombine replaced, not moved to tools.** A snapy-specific
    combiner is about a fifth of the size and has no file-scope state. A
    general GFDL tool in ``tools/`` would duplicate it. The replacement is
    gated on bitwise-equal variables.

O-6 **Generated files not checked in, except ``.pyi``.** IDEs need stubs
    without a build. Everything else comes from the table at build time, so
    the table and the code cannot disagree.

O-7 ``SNAP_ANOMALY`` **stays as an env override** (FOR CHENG). It is
    named in a user-facing error message and on 5 test lines. The other
    switches become deck keys.

O-8 **Line estimates** are from the skeleton plus the live code they keep.
    The sensitivity case (+50 %) is given so the target does not rest on the
    estimates.

O-9 **The cycle line stays** (as a printer over the budget's totals). It is
    how people watch runs today. Its content does not change.

14. Cross-lane questions (for the kickoff thread)
-------------------------------------------------

**Lane R (register, monitor):**

R1. Does the register hold the step-integrated face flux ``sum b_s dt F_s``
    of every conserved row per axis (A1), or per-stage fluxes? Which rows
    (species, scalars, the AAM row of #357)?
R2. In what dtype does the register accumulate in an FP32 run (A1b)? The
    budget needs FP64 to close to round-off.
R3. Is the boundary flux available per physical boundary face (the budget
    needs only the domain boundary), and on the cubed sphere after the
    panel-edge flux averaging?
R4. Does the monitor expose each fixer's per-row change (A2), with names?
    Is the AAM closure (#365) a fixer channel or part of the flux?
R5. Who resets the register and monitor on a redo, and is a consumer's
    ``observe`` guaranteed to run only after the accept decision?

**Lane X (execution, graphs):**

X1. Confirm A3: all output, flush and restart work runs outside captured
    segments, after the global accept.
X2. Can X provide ``dt`` (and the stage weights, if needed) as a device
    scalar to post-step consumers, so ``observe``/``accumulate_stats`` can
    sit in or after a captured segment?
X3. Restart must invalidate graphs and derived caches (``V3_ROADMAP.md``). Who
    calls the invalidation after ``read_restart``: ``MeshBlock`` or X's
    executor?

**Other lanes:**

* Lane C / kickoff: lane letters (A5). Lane C's draft calls the layout lane
  "O" and diagnostics "EF". Which lane owns the face-floor event counters
  that feed the monitor?
* Lane M (mesh) or whoever owns ``meshblock.cpp``: drop the duplicate slice
  validation (``meshblock.cpp:203-230``) once the generated option checks
  have it. Accept ``signal_handler`` into ``src/mesh``.
* Lane I: accept ``lubksb.h``/``ludcmp.h`` into ``src/implicit``.
* Every lane that owns option headers: when to switch to the generated
  structs (section 7).

15. Risks, hypotheses and what was not checked
----------------------------------------------

* Nothing was built or run for this revision. All counts are ``wc -l`` and
  ``rg`` at the pin. The proposed line counts are estimates.
* ``combine_parts`` equal to ``mppnccombine`` is a gate, not a result.
  mppnccombine also handles cases snapy does not write (several records,
  missing values, 64-bit offsets). **Hypothesis**: snapy never writes them.
* Hypotheses from sections 2, 5, 8 and 9: where ``SNAP_X1_CENTROID_EXACT``
  is read; the ``strtok`` race; ``load_tensors`` from a stream; the
  ``flip_zero`` DP equivalence.
* The 137-vs-213 option binding count is a count difference. Options were
  not matched name by name.
* The ``;`` vs ``,`` split mismatch between the define and write passes
  (section 5) is harmless for every name list read in this revision, because
  ``GetGridType`` checks the ``path_``/``avg_``/``_flux`` affixes on the
  whole string (``output_utils.cpp:212-238``). It was not tested. The core
  removes it.
* Tianhao's inventory: only the tech-report branch was found (section 1).
