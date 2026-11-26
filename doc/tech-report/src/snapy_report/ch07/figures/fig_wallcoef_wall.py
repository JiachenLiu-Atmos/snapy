"""Face coefficient at a reflecting x1 wall: two-cell average with the mirror ghost against the
one-sided extrapolation from the two active cells.

Drawn from extrapolate_to_wall and face_coefficient, src/forcing/diffusion.cpp:110-159, at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, for rho = exp(-x1/H) with h = H/2. A cartoon
with an analytic profile; it reads no data.
"""
import numpy as np
from matplotlib.patches import Rectangle

from snapy_report.figstyle import BLACK, BLUE, CELL_FILL, GHOST_FILL, GREEN, PURPLE, SINGLE, SKY, VERMILLION, new_figure


def make_fig():
    fig, ax = new_figure(SINGLE, 2.8)
    h = 0.5
    xs = np.linspace(-0.25, 1.0, 100)
    for k, x0 in enumerate((-h, 0., h)):
        ghost = x0 < 0.
        ax.add_patch(Rectangle((x0, 0.2), h, 1.1, fc=GHOST_FILL if ghost else CELL_FILL,
                               ec=PURPLE if ghost else BLACK, lw=0.6, hatch="//" if ghost else None))
    ax.plot(xs[xs >= 0], np.exp(-xs[xs >= 0]), color=BLUE, lw=1.2)
    ra, rb = np.exp(-0.25), np.exp(-0.75)
    ax.plot([0.25, 0.75], [ra, rb], "o", color=SKY, ms=5, mec=BLACK, mew=0.5)
    ax.plot([-0.25], [ra], "o", color=PURPLE, ms=5, mec=BLACK, mew=0.5)
    ax.plot([0.], [ra], "s", color=VERMILLION, ms=5, mec=BLACK, mew=0.5)
    ax.plot([0., 0.75], [1.5 * ra - 0.5 * rb, rb], color=GREEN, lw=0.9, ls="--")
    ax.plot([0.], [1.5 * ra - 0.5 * rb], "D", color=GREEN, ms=5, mec=BLACK, mew=0.5)
    ax.axvline(0., color=BLACK, lw=2.5)
    ax.set_xlabel("$x_1/H$ [-]")
    ax.set_ylabel(r"$\rho/\rho(0)$ [-]")
    ax.set_xlim(-0.55, 1.05)
    ax.set_ylim(0.2, 1.3)
    return fig
