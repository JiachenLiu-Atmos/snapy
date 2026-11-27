"""A failed block job ends the other blocks' same-process waits; the pool clears and rethrows.

Mechanism of local_wait, wake_local_exchanges and clear_local_exchange (src/layout/layout.cpp) and of the block
worker pool (src/mesh/mesh.cpp) at snapy@73c45a7c8c991209924e1137992352c335575062. A cartoon; it reads no data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 2.3)
    lanes = [("block 0", 3.0), ("block 1", 2.0), ("block 2", 1.0), ("pool", 0.0)]
    for name, y in lanes:
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=fs.SMALL_PT)
    for y in (3.0, 1.0):  # blocks that reach the meeting and wait
        ax.plot([0, 2.0], [y, y], color=fs.SKY, lw=3, solid_capstyle="butt")
        ax.plot([2.0, 4.0], [y, y], color=fs.GREY, lw=1.5, ls=":")
        ax.text(3.0, y + 0.18, "local_wait", ha="center", fontsize=fs.SMALL_PT, color=fs.GREY)
        ax.plot(4.0, y, "x", color=fs.BLACK, ms=6)
        ax.text(4.1, y, "leaves: same-process wait left", va="center", fontsize=fs.SMALL_PT)
    ax.plot([0, 3.0], [2.0, 2.0], color=fs.SKY, lw=3, solid_capstyle="butt")
    ax.plot(3.0, 2.0, "X", color=fs.VERMILLION, ms=8)
    ax.text(3.15, 2.0, "job throws", va="center", fontsize=fs.SMALL_PT, color=fs.VERMILLION)
    ax.plot([3.0, 7.6], [0.0, 0.0], color=fs.BLACK, lw=0.8)
    for x, label in ((3.0, "store first\nerror"), (4.0, "wake"), (6.2, "all jobs\nended: clear"),
                     (7.4, "rethrow")):
        ax.plot(x, 0.0, "o", color=fs.GREEN, ms=5)
        ax.text(x, -0.35, label, ha="center", va="top", fontsize=fs.SMALL_PT)
    ax.annotate("", xy=(3.0, 0.1), xytext=(3.0, 1.9),
                arrowprops=dict(arrowstyle="->", color=fs.VERMILLION, lw=0.8))
    for y in (3.0, 1.0):
        ax.annotate("", xy=(4.0, y - 0.12), xytext=(4.0, 0.1),
                    arrowprops=dict(arrowstyle="->", color=fs.GREEN, lw=0.8))
    ax.set(xlim=(-1.2, 8.2), ylim=(-1.1, 3.5))
    ax.axis("off")
    return fig
