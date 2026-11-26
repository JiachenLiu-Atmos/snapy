"""The at-face target of relax-bot-temp: linear extrapolation of the two bottom cells to the
lower face, and the gain that makes the face relax at 1/tau.

Drawn from RelaxBotTempImpl::forward, src/forcing/relax_bot_temp.cpp:80-104, at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, on uniform cells (a = 1/2). A cartoon; it
reads no data.
"""
from matplotlib.patches import Rectangle

from snapy_report.figstyle import BLACK, BLUE, CELL_FILL, GREEN, SINGLE, SKY, SMALL_PT, new_figure


def make_fig():
    fig, ax = new_figure(SINGLE, 2.6)
    for x0 in (0., 1.):
        ax.add_patch(Rectangle((x0, 280.), 1., 22., fc=CELL_FILL, ec=BLACK, lw=0.6))
    T0, T1 = 292., 288.
    ax.plot([0.5, 1.5], [T0, T1], "o", color=SKY, ms=5, mec=BLACK, mew=0.5)
    ax.plot([0., 1.5], [1.5 * T0 - 0.5 * T1, T1], color=GREEN, ls="--", lw=0.9)
    ax.plot([0.], [1.5 * T0 - 0.5 * T1], "D", color=GREEN, ms=5, mec=BLACK, mew=0.5)
    ax.axhline(298., color=BLUE, lw=1.0, ls=":")
    ax.text(1.05, 298.5, "btemp", color=BLUE, fontsize=SMALL_PT)
    ax.axvline(0., color=BLACK, lw=2.5)
    ax.text(0.5, 281., "$i_l$", ha="center", fontsize=SMALL_PT)
    ax.text(1.5, 281., "$i_l+1$", ha="center", fontsize=SMALL_PT)
    ax.set_xlabel("$(x_1 - x_{1,\\mathrm{bot}})/\\Delta x_1$ [-]")
    ax.set_ylabel("$T$ [K]")
    ax.set_xlim(-0.2, 2.05)
    ax.set_ylim(280., 302.)
    return fig
