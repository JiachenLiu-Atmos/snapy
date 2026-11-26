"""A segment of two blocks with two in-stage exchanges, captured as graphs with host steps between them.

Mechanism of exchange_cut and StageExecutor::capture / replay (src/mesh/stage_executor.cpp) at
snapy@73c45a7c8c991209924e1137992352c335575062. A cartoon; it reads no data.
"""

from matplotlib.patches import Rectangle

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 1.9)
    steps = [("graph", 2.0), ("host", 0.8), ("graph", 1.6), ("host", 0.8), ("graph", 1.4)]
    for b, y in ((0, 1.2), (1, 0.0)):
        ax.text(-0.15, y + 0.3, "block %d" % b, ha="right", va="center", fontsize=fs.SMALL_PT)
        x = 0.0
        for kind, w in steps:
            graph = kind == "graph"
            ax.add_patch(Rectangle((x, y), w, 0.6, facecolor=fs.CELL_FILL if graph else fs.WHITE,
                                   edgecolor=fs.BLUE if graph else fs.GREEN, lw=1.0))
            ax.text(x + w / 2, y + 0.3, "graph" if graph else "host step", ha="center", va="center",
                    fontsize=fs.SMALL_PT, color=fs.BLUE if graph else fs.GREEN)
            x += w
    x = 0.0
    for kind, w in steps:
        if kind == "host":
            ax.plot([x + w / 2, x + w / 2], [0.6, 1.2], color=fs.GREEN, lw=1.0, ls="--")
        x += w
    ax.text(2.4, 2.0, "rendezvous", ha="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.text(4.8, 2.0, "rendezvous", ha="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set(xlim=(-1.0, 6.8), ylim=(-0.2, 2.3))
    ax.axis("off")
    return fig
