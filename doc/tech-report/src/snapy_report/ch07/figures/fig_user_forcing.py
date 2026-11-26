"""Interface of a user stage forcing: what it receives and what it may return.

Drawn from src/mesh/meshblock.cpp:38-67 (stage_forcing_variables, stage_forcing_result) and
:689-751 (application and dry carry) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5.
A cartoon; it reads no data.
"""
from matplotlib.patches import FancyBboxPatch

from snapy_report.figstyle import BLACK, DOUBLE, GREEN, ORANGE, SKY, SMALL_PT, VERMILLION, new_figure


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=BLACK, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=SMALL_PT)


def arrow(ax, a, b, color=BLACK):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=color, lw=1.0))


def make_fig():
    fig, ax = new_figure(DOUBLE, 2.4)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    box(ax, 0.1, 1.0, 2.6, 1.4, "Dict[str, Tensor]:\nblock buffers,\nthen stage variables", SKY)
    box(ax, 3.3, 1.0, 2.2, 1.4, "module.forward(\nvariables,\ndt, stage)", "white")
    box(ax, 6.3, 1.85, 3.6, 0.85, "hydro_du: added to\nthe native $\\Delta\\mathbf{U}$", ORANGE)
    box(ax, 6.3, 0.7, 3.6, 0.85, "scalar_ds: added to\nthe scalar tendency", ORANGE)
    arrow(ax, (2.7, 1.7), (3.3, 1.7))
    arrow(ax, (5.5, 1.9), (6.3, 2.25))
    arrow(ax, (5.5, 1.5), (6.3, 1.1))
    ax.text(8.1, 0.3, "any other key, or a wrong shape: error", ha="center", fontsize=SMALL_PT,
            color=VERMILLION)
    ax.text(8.1, 3.0, "dry-density part carries the tracers", ha="center", fontsize=SMALL_PT, color=GREEN)
    arrow(ax, (8.1, 2.9), (8.1, 2.72), GREEN)
    return fig
