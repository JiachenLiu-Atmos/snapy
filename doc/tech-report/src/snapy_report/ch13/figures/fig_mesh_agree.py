"""The Mesh-agree decision: all blocks capture, or one refusal makes the whole Mesh eager.

Rule of StageExecutor::capture_reasons (src/mesh/stage_executor.cpp) at
snapy@73c45a7c8c991209924e1137992352c335575062. A cartoon; it reads no data.
"""

from matplotlib.patches import Rectangle

from snapy_report import figstyle as fs


def make_fig():
    fig, axes = fs.new_figure(fs.DOUBLE, 1.6, ncols=2)
    cases = [(["graph", "graph", "graph"], [None, None, None]),
             (["eager", "eager", "eager"], ["disagree", "disagree", "refuses"])]
    for ax, (modes, notes) in zip(axes, cases):
        for i, (m, n) in enumerate(zip(modes, notes)):
            x = 1.3 * i
            ok = m == "graph"
            ax.add_patch(Rectangle((x, 0), 1.0, 0.8, facecolor=fs.CELL_FILL,
                                   edgecolor=fs.GREEN if ok else fs.VERMILLION, lw=1.0))
            ax.text(x + 0.5, 0.5, "block %d" % i, ha="center", va="center", fontsize=fs.SMALL_PT)
            ax.text(x + 0.5, 0.2, m, ha="center", va="center", fontsize=fs.SMALL_PT,
                    color=fs.GREEN if ok else fs.VERMILLION)
            if n:
                ax.text(x + 0.5, -0.25, n, ha="center", va="center", fontsize=fs.SMALL_PT)
        ax.set(xlim=(-0.2, 3.8), ylim=(-0.5, 1.0))
        ax.axis("off")
    fs.panel_labels(axes)
    return fig
