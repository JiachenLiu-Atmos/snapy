Execution: eager and CUDA graphs
================================

``snapy`` runs every stage eagerly by default. A single ``MeshBlock`` on a
CUDA device can instead run each stage of ``forward()`` as one captured CUDA
graph. Eager execution is the reference: a graph replay gives the same bits.

Selecting the execution
-----------------------

The deck selects it under ``integration``::

  integration:
    execution: eager          # eager | cuda-graph
    execution-strict: false   # true: refuse a case that cannot be captured

The two key names are provisional until the V3 naming is confirmed.

- ``eager`` (default): every stage runs eagerly. ``execution-strict: true``
  with ``eager`` is refused when the options are read.
- ``cuda-graph``: a case that can be captured runs each stage as a graph; any
  other case runs eagerly. With ``execution-strict: true`` such a case is
  refused before its first launch, and the error lists the reasons.
- A ``Mesh`` runs its blocks eagerly; ``execution-strict: true`` is refused
  there (graphs for several blocks come with a later milestone).

Capability matrix
-----------------

Numerical support is the eager columns; graph support is separate.

.. csv-table::
   :header: "Case", "eager FP64", "eager FP32", "graph FP64", "graph FP32"

   "Cartesian dry, 1 block", "yes", "yes", "yes (below)", "yes (below)"
   "Cartesian, N blocks, UCX ranks", "yes", "yes", "no (G4)", "no (G4)"
   "kintera moist", "yes", "partly", "no (G2)", "no (G2)"
   "vertical implicit correction", "yes", "Cartesian ideal-moist", "no (G3)", "no (G3)"
   "cubed sphere", "yes", "no (G5)", "no (G6)", "no (G6)"
   "user TorchScript forcing", "yes", "yes", "no", "no"
   "radiation (pyharp/pydisort)", "not on the step path", "no", "no", "no"

CPU and MPS run eager only. The graph rows that say yes cover the dry Straka
case: a CUDA device, Cartesian geometry, one species, no reactions, the EOS
limiter off, an ideal-moist EOS, the hllc Riemann solver, no implicit
scheme, rk3, one block, const-gravity as the only forcing and no solid mask.

``MeshBlock.execution_capability(vars)`` returns a block's row for its own
configuration and variables: ``eager_fp64``/``eager_fp32`` (its precision),
``graph_fp64``/``graph_fp32`` (whether it can be captured) and
``eager_reasons`` (why its stages run eagerly).

What a graph depends on
-----------------------

A stage's graph is captured again when any of these changes: the device, the
shapes, strides or dtypes of the state, the state store's generation, the
configuration generation, the stage or ``dt``. ``dt`` is a host number baked
into the graph, so a new ``dt`` recaptures. The store's generation changes
when its buffers move: ``to()``, a restart, or a variable that the caller
replaced between steps (it is copied into the block's fixed storage).
``bump_config_generation()`` recaptures after a change the graph cannot see,
such as ``implicit-fallback-fp64``. A restart never serializes graphs.

Capture warms each stage up eagerly first, with the step's ``dt``, and then
restores the state, the registered flux-register storage and ``hydro_w``.
Retries stay on the host after the replay: ``check_redo`` restores the
step's backup, and the smaller ``dt`` captures again.

``MeshBlock.execution_report()`` counts captures, replays, eager stages and
held graphs.

Benchmark
---------

``tools/bench_execution.py`` (run from ``tests/``) times the four
precision/execution combinations of the dry Straka case on one GPU, with the
run order alternated between repeats, and reports the wall time per step, its
spread, the captures and replays, the peak memory and the speed-up over the
eager run of the same precision.
