"""The seam rotation cache: a side's operands are reused while the stamps of its grid tensors match.

Mechanism of SeamRotationCache in src/layout/seam_register.hpp and its use in reconcile_panel_seams at the code
pin of the chapter's PROVENANCE blocks. A cartoon; it reads no data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 1.8)
    boxes = [(0.0, "grid tensors\n(x2f or x2v, x3v or x3f,\nface cosine)"),
             (2.2, "stamps: storage, offset,\nversion, dtype,\nsizes, strides; extent"),
             (4.4, "operands\n(metric, cosine)")]
    for x, s in boxes:
        fs.cell_box(ax, x, 0.3, width=1.7, height=1.1)
        ax.text(x + 0.85, 0.85, s, ha="center", va="center", fontsize=fs.SMALL_PT)
    for x in (1.75, 3.95):
        ax.annotate("", xy=(x + 0.4, 0.85), xytext=(x, 0.85),
                    arrowprops=dict(arrowstyle="->", color=fs.ORANGE, lw=1.2))
    ax.text(5.25, 1.55, "equal: reuse; different: rebuild, builds += 1", ha="center",
            fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set(xlim=(-0.1, 6.2), ylim=(0.1, 1.8))
    ax.axis("off")
    return fig
