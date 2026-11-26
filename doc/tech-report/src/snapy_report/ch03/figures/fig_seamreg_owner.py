"""The panel-seam register exchange: the owner's face rows replace the non-owner's.

Mechanism of reconcile_panel_seams in src/layout/seam_register.cpp at the code pin of the chapter's PROVENANCE
blocks. A cartoon; it reads no data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 2.0)
    for x, label in ((0.0, "panel A (owner, lower rank)"), (4.2, "panel B (non-owner)")):
        fs.cell_box(ax, x, 0.4, width=1.6, height=1.0)
        ax.text(x + 0.8, 1.6, label, ha="center", fontsize=fs.SMALL_PT)
    ax.plot([1.6, 1.6], [0.3, 1.5], color=fs.PURPLE, ls="--", lw=1.4)
    ax.plot([4.2, 4.2], [0.3, 1.5], color=fs.PURPLE, ls="--", lw=1.4)
    ax.text(1.65, 0.15, "seam face", fontsize=fs.SMALL_PT, color=fs.PURPLE)
    steps = ["raise; to\nCartesian", "board buffer;\ncut", "flip, sign;\nto B; lower"]
    for k, s in enumerate(steps):
        x0 = 1.8 + 0.8 * k
        ax.annotate("", xy=(x0 + 0.6, 0.9), xytext=(x0, 0.9),
                    arrowprops=dict(arrowstyle="->", color=fs.ORANGE, lw=1.2))
        ax.text(x0 + 0.3, 0.45 if k % 2 else 1.1, s, ha="center", fontsize=fs.SMALL_PT)
    ax.text(5.0, 0.15, "replace (owner_copy)", ha="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set(xlim=(-0.1, 6.0), ylim=(0.0, 1.9))
    ax.axis("off")
    return fig
