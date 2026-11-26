"""Energy-row entries the face gravity work adds to the VIC block row of cell i.

Cartoon of src/implicit/vic_assemble_impl.h:61-66 and 150-159 and src/implicit/vic_gravity_work_impl.h:22-35 at
snapy@73c45a7c8c991209924e1137992352c335575062 (claims C1-C2 of vicwork_check.py); no measured data.
"""
from matplotlib.patches import Rectangle

from snapy_report import figstyle as fs

ROWS = [  # (mode, entries of b[i], a[i], c[i] in the energy row)
    ("face", ["-g lo m\n-g lo A-", "-g(lo+hi) m\n-g(hi A+ - lo A-)", "-g hi m\n+g hi A+"]),
]


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 1.5)
    ax.set_axis_off()
    ax.set(xlim=(0, 12.4), ylim=(1.0, 2.9))
    for k, lab in enumerate(("b[i]: cell i-1", "a[i]: cell i", "c[i]: cell i+1")):
        ax.text(3.0 + 3.2 * k, 2.6, lab, ha="center", fontsize=fs.SMALL_PT)
    for r, (mode, cells) in enumerate(ROWS):
        y = 1.3 - 1.4 * r
        ax.text(.0, y + .45, "gravity-work:\n" + mode, fontsize=fs.SMALL_PT, va="center")
        for k, txt in enumerate(cells):
            ax.add_patch(Rectangle((1.5 + 3.2 * k, y - .1), 3.0, 1.1, facecolor="#EAF5FC", edgecolor=fs.BLACK, lw=.6))
            ax.text(3.0 + 3.2 * k, y + .45, txt, ha="center", va="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    return fig
