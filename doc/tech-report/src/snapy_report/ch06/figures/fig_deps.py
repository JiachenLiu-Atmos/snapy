"""Dependency map of the chapter 6 schemes at snapy@73c45a7c8c991209924e1137992352c335575062.

Logical relations read from the code: src/forcing/const_gravity.cpp:29-42 (face is the one form; cell and the fixer
are refused), src/hydro/hydro.cpp:322-327 (radial_exact_work), src/implicit/implicit_hydro.cpp:396-454
(the VIC rows); no measured data.
"""

from snapy_report import figstyle as fs
from snapy_report.ch06._draw import arrow, bare, label, node


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 2.9)
    bare(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1.06))
    face = node(ax, .30, .80, "face form\n(gravity-work: face, the only form)")
    dwork = node(ax, .78, .80, "corrected-PE face work\n(automatic: g1 != 0 and\nthe grid has an x1 measure)")
    vic = node(ax, .78, .22, "VIC gravity work\n(face rows, K rows by Woodbury)")
    removed = node(ax, .22, .22, "cell form, gravity-work fixer\n(refused at setup)", edge=fs.VERMILLION)
    inv = label(ax, .52, .50, "E + P")
    fig.canvas.draw()
    arrow(ax, dwork, face)  # requires
    arrow(ax, dwork, inv, ls="--")  # implies
    arrow(ax, vic, face)  # its face rows are the face form
    arrow(ax, vic, dwork, ls="--")  # carries its rows when it is on
    arrow(ax, face, removed, style="-[", color=fs.VERMILLION)  # replaces
    return fig
