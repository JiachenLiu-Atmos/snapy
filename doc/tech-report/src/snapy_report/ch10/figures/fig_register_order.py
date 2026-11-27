"""The flux register through one stage of a step, at the report's code pin.

Cartoon of the calls MeshBlockImpl::stage_advance, HydroImpl::forward and MeshBlockImpl::stage_combine make on
the hydro register (src/mesh/meshblock_advance.cpp, src/hydro/hydro_forward.cpp), and of the budget output's
observe after an accepted step (MeshBlockImpl::apply_redo, src/mesh/meshblock_redo.cpp); no measured data.
"""
from snapy_report.ch01.flow import make_flow


def make_fig():
    return make_flow([
        "stage 0 only: begin_step zeroes step fields and monitor; wall batch cleared (10.6)",
        "every stage: begin_stage picks the amount slot; wall batch cursor restarts (10.3, 10.4)",
        "sweeps: Riemann total, seam replace, face-floor and event masks counted (10.1, 10.5)",
        "prescribed faces after the seams and after the sources' posts (10.2)",
        "after the implicit correction: x1 prescribed again if it ran or is marked (10.2)",
        "combine: wall batch flushed; end_stage folds step fields, reduces counts (10.3-10.5)",
        "after an accepted step: the budget output observes step fields and counts (10.7)",
    ])
