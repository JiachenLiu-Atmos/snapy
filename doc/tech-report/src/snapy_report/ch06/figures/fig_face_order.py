"""Order of the face-form gravity work, interior and wall cells (the curvature-flux columns of the data are not drawn:
the code at snapy@73c45a7c8c991209924e1137992352c335575062 has no curvature flux).

Data: data/face_order.csv, written by cellface_check.py (claims C3-C4), which ports
src/hydro/hydro_forward.cpp:794-848 at snapy@e894700ff7aee30b52882e5202b16461413780b0 (exact point fluxes,
g1 = -10, x1 in [0, 1]).
"""
import os

import numpy as np

from snapy_report import figstyle as fs

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "face_order.csv")


def make_fig():
    n, fi, fw, ki, kw = np.loadtxt(DATA, delimiter=",", unpack=True)
    fig, ax = fs.new_figure(fs.SINGLE, 3.4)
    ax.loglog(n, fi, "o-", color=fs.VERMILLION, label="face, interior")
    ax.loglog(n, fw, "s--", color=fs.VERMILLION, label="face, wall")
    for order, y0 in ((2, fi[0]),):
        y1 = y0 * 2. * (n[-1] / n[0]) ** (-order)
        ax.loglog([n[0], n[-1]], [2. * y0, y1], ":", color=fs.BLUE, lw=.9)
        ax.text(n[-1] * 1.1, y1, "$h^%d$" % order, fontsize=fs.SMALL_PT, color=fs.BLUE, va="center")
    ax.set_xticks(n, [str(int(k)) for k in n])
    ax.minorticks_off()
    ax.set_xlabel("cells $n_1$ [-]")
    ax.set_ylabel("max error of the work [-]")
    fig.legend(*ax.get_legend_handles_labels(), fontsize=fs.SMALL_PT, frameon=False, ncol=2, loc="outside lower center")
    return fig
