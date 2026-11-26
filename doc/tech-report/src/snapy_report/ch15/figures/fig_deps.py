"""Dependency map of the chapter 15 sections.

Logical relations read from the code the section files cite (the deck reader builds the output entries, the block
builds one writer per entry, the restart writer saves and restores the writers' schedule); no measured data.
"""

from snapy_report import figstyle as fs
from snapy_report.ch15._draw import arrow, bare, node


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 1.6)
    bare(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    options = node(ax, .15, .5, "options schema\n(the deck)")
    output = node(ax, .50, .5, "output writers\n(outputs: netcdf, pnetcdf)")
    restart = node(ax, .85, .5, "restart\n(outputs: restart; -r)")
    fig.canvas.draw()
    arrow(ax, output, options)  # the writers are built from the deck's entries
    arrow(ax, restart, output)  # the restart saves the writers' schedule and uses their combine protocol
    return fig
