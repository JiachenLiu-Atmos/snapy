"""The 3x3 neighbour stencil of a block with the buffer id of each offset (get_buffer_id,
src/layout/layout.hpp:35-38 at snapy@73c45a7c8c991209924e1137992352c335575062); face slabs exchanged; corners from the
diagonal neighbour on slab and cubed, synthesised where there is none and on the cubed sphere (CornerRule). A cartoon; it reads no data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.SINGLE, 3.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            bid = (dx % 3) + (dy % 3) * 3
            corner = abs(dx) + abs(dy) == 2
            fs.cell_box(ax, dx, dy, 0.92, 0.92, ghost=(dx, dy) != (0, 0))
            col = fs.PURPLE if corner else fs.BLACK
            box = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none")
            ax.text(dx + 0.46, dy + (0.18 if (dx, dy) == (0, -1) else 0.74), "id %d" % bid, ha="center", va="center", fontsize=fs.SMALL_PT, color=col,
                    bbox=box)
            if corner:
                ax.text(dx + 0.46, dy + 0.3, "diagonal or\nsynthesised", ha="center", va="center",
                        fontsize=fs.SMALL_PT, color=fs.PURPLE, linespacing=1.0, bbox=box)
            elif (dx, dy) != (0, 0):
                ax.annotate("", xy=(0.46 + 0.3 * dx, 0.46 + 0.3 * dy), xytext=(dx + 0.46 - 0.2 * dx, dy + 0.46 - 0.2 * dy),
                            arrowprops=dict(arrowstyle="->", color=fs.ORANGE, lw=1.2))
    ax.text(0.46, 0.2, "block", ha="center", fontsize=fs.SMALL_PT)
    ax.set_xlabel(r"offset in $x_2$")
    ax.set_ylabel(r"offset in $x_3$")
    ax.set(xlim=(-1.1, 2.0), ylim=(-1.1, 2.0), aspect="equal", xticks=[-0.54, 0.46, 1.46], yticks=[-0.54, 0.46, 1.46])
    ax.set_xticklabels(["-1", "0", "+1"])
    ax.set_yticklabels(["-1", "0", "+1"])
    return fig
