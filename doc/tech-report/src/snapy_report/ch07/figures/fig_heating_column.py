"""Where body heating, top cooling and bottom heating act in an x1 column.

Drawn from src/forcing/body_heat.cpp:43-55, top_cool.cpp:42-56 and bot_heat.cpp:42-56 at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5. A cartoon; it reads no data.
"""
from matplotlib.patches import Rectangle

from snapy_report.figstyle import BLACK, BLUE, CELL_FILL, ORANGE, SINGLE, SMALL_PT, VERMILLION, new_figure


def make_fig():
    fig, ax = new_figure(SINGLE, 3.2)
    ax.set_axis_off()
    n = 10
    for i in range(n):
        fc = CELL_FILL
        if i >= n - 3:
            fc = BLUE
        elif i < 2:
            fc = VERMILLION
        elif 4 <= i <= 6:
            fc = ORANGE
        ax.add_patch(Rectangle((0, i), 1, 1, fc=fc, ec=BLACK, lw=0.6, alpha=0.6 if fc != CELL_FILL else 1))
    ax.plot([0, 1], [0, 0], color=BLACK, lw=2.5)
    ax.plot([0, 1], [n, n], color=BLACK, lw=2.5)
    ax.text(1.15, n - 1.5, "top-cool: $F/(\\Delta x_{1,\\mathrm{top}}\\,n_{\\mathrm{dep}})$\nin the top $n_{\\mathrm{dep}}$ cells",
            fontsize=SMALL_PT, va="center")
    ax.text(1.15, 5.5, "body-heat: $\\rho c_v\\dot T$\nwhere $p_{\\min}\\leq p\\leq p_{\\max}$",
            fontsize=SMALL_PT, va="center")
    ax.text(1.15, 1.0, "bot-heat: $F/(\\Delta x_{1,\\mathrm{bot}}\\,n_{\\mathrm{dep}})$\nin the bottom $n_{\\mathrm{dep}}$ cells",
            fontsize=SMALL_PT, va="center")
    ax.set_xlim(-0.1, 3.4)
    ax.set_ylim(-0.3, n + 0.3)
    return fig
