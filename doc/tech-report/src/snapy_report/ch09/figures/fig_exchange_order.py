"""Where the ghost exchange sits in one stage for the two drivers: MeshBlockImpl::forward exchanges before
advance_local (src/mesh/meshblock.cpp:545-552), MeshImpl::forward after it (src/mesh/mesh.cpp:331-365), at
snapy@e894700ff7aee30b52882e5202b16461413780b0. A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs


def _row(ax, y, items):
    x = 0.2
    for text, fc, w in items:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.55, boxstyle="round,pad=0.02", fc=fc, ec=fs.BLACK, lw=0.7))
        ax.text(x + w / 2, y + 0.275, text, ha="center", va="center", fontsize=8)
        x += w + 0.35
        if (text, fc, w) != items[-1]:
            ax.annotate("", xy=(x, y + 0.275), xytext=(x - 0.35, y + 0.275),
                        arrowprops=dict(arrowstyle="->", color=fs.BLACK, lw=0.7))


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 1.9))
    ax.set_axis_off()
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.3)
    host = ("host source\n(driver)", "white", 1.7)
    ex = ("exchange\nghosts", fs.PURPLE, 1.4)
    adv = ("advance_local\n(stage)", fs.SKY, 1.9)
    ax.text(0.2, 2.05, "MeshBlock::forward", fontsize=8, fontweight="bold")
    _row(ax, 1.3, [host, ex, adv, ("next stage", "white", 1.4)])
    ax.text(0.2, 0.85, "Mesh::forward", fontsize=8, fontweight="bold")
    _row(ax, 0.1, [host, adv, ex, ("next stage", "white", 1.4)])
    return fig
