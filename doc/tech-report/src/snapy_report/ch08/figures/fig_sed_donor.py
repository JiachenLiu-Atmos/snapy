"""Donor cells and sealing of the settling flux in one x1 column, at the report's code pin.

Cartoon of sedimentation_flux_impl and sedimentation_donor_impl (src/sedimentation/sed_hydro_impl.h) and of the
wall bounds SedHydroImpl::forward hands them (src/sedimentation/sed_hydro.cpp); no measured data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.SINGLE, 3.6)
    ax.set(xlim=(0, 3.4), ylim=(-0.2, 6.2))
    ax.axis("off")
    for k in range(6):
        ghost = k == 0 or k == 5
        fs.cell_box(ax, 1, k, 1, 1, ghost=ghost)
    ax.plot([0.8, 2.2], [1, 1], color=fs.BLACK, lw=2.5)
    ax.plot([0.8, 2.2], [5, 5], color=fs.BLACK, lw=2.5)
    for y in (2, 3, 4):
        ax.annotate("", xy=(1.5, y - 0.3), xytext=(1.5, y + 0.3),
                    arrowprops=dict(arrowstyle="->", color=fs.ORANGE, lw=1.4))
        ax.text(2.3, y, "from cell above\n(settling)", fontsize=fs.SMALL_PT, va="center")
    ax.text(2.3, 1, "physical wall:\nsealed", fontsize=fs.SMALL_PT, va="center")
    ax.text(2.3, 5, "physical wall:\nsealed", fontsize=fs.SMALL_PT, va="center")
    ax.text(0.1, 3, "interior\ncells", fontsize=fs.SMALL_PT, va="center")
    return fig
