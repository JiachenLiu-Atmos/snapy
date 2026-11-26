"""The step loop and its acceptance decision: stages, step-end sources, check_redo, then accept, redo at half
the step, or stop. From examples/run_hydro.cpp:161-200 and MeshBlockImpl::check_redo/apply_redo
(src/mesh/meshblock.cpp:1238-1284, 1355-1357) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5. A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs


def _box(ax, x, y, w, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.6, boxstyle="round,pad=0.03", fc=fc, ec=fs.BLACK, lw=0.8))
    ax.text(x + w / 2, y + 0.3, text, ha="center", va="center", fontsize=8)


def _arrow(ax, a, b, color=fs.BLACK, rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=color, lw=0.9,
                                                    connectionstyle="arc3,rad=%g" % rad))


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    _box(ax, 0.1, 1.5, 2.0, "$\\Delta t=2^{-n_{\\mathrm{redo}}}\\Delta t_0$", fs.SKY)
    _box(ax, 2.6, 1.5, 1.9, "stages $s=1..S$", fs.SKY)
    _box(ax, 5.0, 1.5, 1.9, "step-end sources", "white")
    _box(ax, 7.4, 1.5, 1.8, "check_redo", fs.ORANGE)
    _box(ax, 9.9, 2.6, 1.9, "accept: $n_{\\mathrm{redo}}\\leftarrow0$", fs.GREEN)
    _box(ax, 9.9, 1.5, 1.9, "redo: $n_{\\mathrm{redo}}+1$", fs.YELLOW)
    _box(ax, 9.9, 0.4, 1.9, "stop: $n_{\\mathrm{redo}}>5$", fs.VERMILLION)
    _arrow(ax, (2.1, 1.8), (2.6, 1.8))
    _arrow(ax, (4.5, 1.8), (5.0, 1.8))
    _arrow(ax, (6.9, 1.8), (7.4, 1.8))
    _arrow(ax, (9.2, 1.95), (9.9, 2.9))
    _arrow(ax, (9.2, 1.8), (9.9, 1.8))
    _arrow(ax, (9.2, 1.65), (9.9, 0.7))
    _arrow(ax, (10.85, 1.5), (1.1, 1.5), fs.GREY, rad=-0.35)
    ax.text(5.6, 0.25, "restore $\\mathbf{U}^n$, recompute $\\mathbf{W}$", fontsize=8, color=fs.GREY, ha="center")
    return fig
