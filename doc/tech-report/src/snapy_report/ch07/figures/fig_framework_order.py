"""Registration order of the native forcings and where their tendency enters a stage.

Drawn from HydroImpl::_register_forcings_module (src/hydro/register_forcing_modules.cpp) and the block
"(6) Calculate external forcing" of HydroImpl::forward (src/hydro/hydro_forward.cpp)
at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5. A cartoon; it reads no data.
"""
from matplotlib.patches import FancyBboxPatch

from snapy_report.figstyle import BLACK, DOUBLE, GREEN, ORANGE, SKY, SMALL_PT, new_figure

MODULES = ["const-gravity", "coriolis", "diffusion", "body-heat", "top-cool", "bot-heat",
           "relax-bot-comp", "relax-bot-temp", "relax-bot-velo", "top-sponge-lyr",
           "bot-sponge-lyr", "plume-forcing"]


def make_fig():
    fig, ax = new_figure(DOUBLE, 3.4)
    ax.set_axis_off()
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 6.4)
    x0, w, h, dx = 0.25, 3.05, 0.75, 3.35
    for k, name in enumerate(MODULES):
        row, col = divmod(k, 4)
        x, y = x0 + col * dx, 4.6 - row * 1.4
        dead = name == "plume-forcing"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    fc="white" if dead else SKY, ec=BLACK, lw=0.8,
                                    ls="--" if dead else "-"))
        ax.text(x + w / 2, y + h / 2, "%d  %s" % (k + 1, name), ha="center", va="center",
                fontsize=SMALL_PT)
        if col < 3:
            ax.annotate("", xy=(x + dx, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))
    for row in (0, 1):
        y = 4.6 - row * 1.4
        ax.annotate("", xy=(x0 + w / 2, y - 1.4 + h), xytext=(x0 + 3 * dx + w / 2, y),
                    arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8,
                                    connectionstyle="arc3,rad=0.05"))
    ax.text(x0, 5.85, r"start: $\Delta\mathbf{U} = -\Delta t\,\nabla\cdot\mathbf{F}$ (interior cells)",
            fontsize=SMALL_PT, color=ORANGE, va="center")
    ax.text(x0 + 3 * dx + w, 1.25, r"then: VIC, user stage forcings, RK average",
            fontsize=SMALL_PT, color=GREEN, ha="right", va="center")
    ax.text(x0, 0.45, r"each module adds $\Delta t\,S$ to the same $\Delta\mathbf{U}$ from the "
            r"stage-entry $\mathbf{W}$ and $T$; dashed: never installed at the pin",
            fontsize=SMALL_PT, va="center")
    return fig
