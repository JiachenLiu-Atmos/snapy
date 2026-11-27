"""The per-column status bits of ImplicitHydroImpl::forward_masked and what a failed column does
(src/implicit/vic_kernels.h:175-208, src/implicit/implicit_hydro.cpp:564-617 at
snapy@73c45a7c8c991209924e1137992352c335575062). A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.0))
    ax.set_axis_off()
    steps = [("assemble", fs.SKY), ("inputs\nfinite? (1)", fs.ORANGE), ("sweep\nrefused? (2)", fs.ORANGE),
             ("solution\nfinite? (4)", fs.ORANGE), ("redistribute", fs.SKY), ("results\nfinite? (4)", fs.ORANGE),
             ("undo failed\ncolumns", fs.GREY)]
    x = 0.05
    for k, (t, c) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.15), 1.25, 0.6, boxstyle="round,pad=0.03", fc=c if c != fs.GREY else "white",
                                    ec=fs.BLACK, lw=0.7))
        ax.text(x + 0.625, 1.45, t, ha="center", va="center", fontsize=7)
        if c == fs.ORANGE:
            ax.annotate("", xy=(x + 0.625, 0.62), xytext=(x + 0.625, 1.12),
                        arrowprops=dict(arrowstyle="->", color=fs.VERMILLION, lw=0.8))
        if k < len(steps) - 1:
            ax.annotate("", xy=(x + 1.42, 1.45), xytext=(x + 1.28, 1.45),
                        arrowprops=dict(arrowstyle="->", color=fs.BLACK, lw=0.7))
        x += 1.42
    ax.add_patch(FancyBboxPatch((1.4, 0.05), 8.4, 0.55, boxstyle="round,pad=0.03", fc="#FBE3D5", ec=fs.VERMILLION,
                                lw=0.8))
    ax.text(5.6, 0.32, "failed column only: its bit set, du and w restored, "
            "zero correction; check_redo cause 32", ha="center", va="center", fontsize=7.5)
    ax.text(0.68, 1.95, "one thread per column", ha="center", fontsize=7, color=fs.GREY)
    ax.text(10.0, 1.95, "other columns: du - du0", ha="right", fontsize=7, color=fs.GREEN)
    ax.set_xlim(0, 10.1)
    ax.set_ylim(0, 2.15)
    return fig
