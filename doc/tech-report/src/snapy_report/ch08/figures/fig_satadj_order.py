"""Where the saturation adjustment sits in one RK step, at the report's code pin.

Cartoon of the stage ops StageExecutor::build_schedule installs (advance, forcing, combine, saturation,
chemistry, boundaries; src/mesh/stage_executor.cpp) and of MeshBlockImpl::stage_saturation and
MeshBlockImpl::local_redo_flag_tensor; no measured data.
"""
from snapy_report.ch01.flow import make_flow


def make_fig():
    return make_flow([
        "stage 0: save the state, zero kintera's failure counter and the marks",
        "every stage: advance, forcing, RK combine, whole-column conserved limiter",
        "last stage only: whole-column limiter again, then the UV adjustment of the interior cells",
        "last stage only: the chemistry slot, if one is registered",
        "every stage: boundary fill of the ghosts",
        "after the step: check_redo reads the failure counter's row 0",
    ])
