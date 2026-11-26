"""Small drawing helpers for the chapter 15 cartoons; colours and sizes come from snapy_report.figstyle."""
from matplotlib.patches import FancyArrowPatch

from snapy_report import figstyle as fs


def node(ax, x, y, text, edge=fs.GREEN):
    return ax.text(x, y, text, ha="center", va="center", fontsize=fs.SMALL_PT,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=edge))


def arrow(ax, a, b, style="-|>", ls="-", color=fs.BLACK, lw=1.0):
    """an arrow between two text boxes, clipped to their outlines"""
    p = FancyArrowPatch(posA=a.get_position(), posB=b.get_position(), arrowstyle=style, linestyle=ls, color=color,
                        lw=lw, mutation_scale=9, patchA=a.get_bbox_patch(), patchB=b.get_bbox_patch(),
                        shrinkA=2, shrinkB=2, zorder=5)
    ax.add_patch(p)
    return p


def bare(ax):
    ax.set_axis_off()
    return ax
