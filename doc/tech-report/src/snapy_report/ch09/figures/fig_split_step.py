"""One step with its operator-split pieces: the explicit stages (and VIC), then at the last stage the saturation
adjustment and the physical boundaries, then the driver's kinetics, then the acceptance check. From
src/mesh/meshblock.cpp:795-841 and examples/run_hydro.cpp:166-194 at
snapy@e894700ff7aee30b52882e5202b16461413780b0. A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 1.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 1.3)
    items = [("dynamics:\nRK stages", fs.SKY, 2.2), ("saturation\nadjustment", fs.GREEN, 2.0),
             ("physical\nboundaries", "white", 1.8), ("kinetics\n(driver)", fs.GREEN, 1.8),
             ("check_redo", fs.ORANGE, 1.8)]
    x = 0.1
    for k, (text, fc, w) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 0.7, boxstyle="round,pad=0.03", fc=fc, ec=fs.BLACK, lw=0.8))
        ax.text(x + w / 2, 0.7, text, ha="center", va="center", fontsize=8)
        x += w + 0.35
        if k < len(items) - 1:
            ax.annotate("", xy=(x, 0.7), xytext=(x - 0.35, 0.7),
                        arrowprops=dict(arrowstyle="->", color=fs.BLACK, lw=0.8))
    ax.text(0.1, 0.05, "each piece acts on the result of the one before (a sequential, first-order split)",
            fontsize=8, color=fs.GREY)
    return fig
