"""The four-cell rows of the cubed-sphere diffusion operator along one block direction.

Drawn from `weights` in src/forcing/diffusion_cubed_sphere.cpp at the code pin: the row of face q (index units,
cell centres at 0..n-1) uses the four cells from max(0, min(n-4, floor(q)-1)). A cartoon of that rule for n = 6;
it reads no data.
"""
import math

from matplotlib.patches import Rectangle

from snapy_report.figstyle import BLACK, CELL_FILL, DOUBLE, GREEN, ORANGE, PURPLE, SKY, SMALL_PT, new_figure

N = 6


def stencil_start(q, n=N):
    return max(0, min(n - 4, math.floor(q) - 1))


def make_fig():
    fig, ax = new_figure(DOUBLE, 2.9)
    ax.set_axis_off()
    rows = N + 1
    for f in range(rows):
        y = rows - 1 - f
        start = stencil_start(f - 0.5)
        for c in range(N):
            used = start <= c < start + 4
            ax.add_patch(Rectangle((c, y), 1, 0.8, fc=SKY if used else CELL_FILL, ec=BLACK, lw=0.5))
        ax.plot([f, f], [y - 0.05, y + 0.85], color=ORANGE, lw=2.2)
        ax.plot(f, y + 0.4, "^", color=ORANGE, ms=6, mec=BLACK, mew=0.5)
        shifted = f in (0, 1, N - 1, N)
        ax.text(-0.4, y + 0.4, f"face {f}", ha="right", va="center", fontsize=SMALL_PT,
                color=GREEN if shifted else BLACK)
    for c in range(N):
        ax.text(c + 0.5, -0.35, f"{c}", ha="center", fontsize=SMALL_PT)
    ax.text(N / 2, -0.8, "interior cells of the block along one direction (no ghost is read)",
            ha="center", fontsize=SMALL_PT)
    for x in (-0.12, N + 0.12):
        ax.plot([x, x], [-0.1, rows], color=PURPLE, lw=1.0, ls="--")
    ax.text(N + 0.8, rows - 0.6, "blue: the four cells of the row", fontsize=SMALL_PT)
    ax.text(N + 0.8, rows - 1.4, "orange: the face the row evaluates", fontsize=SMALL_PT)
    ax.text(N + 0.8, rows - 2.2, "green labels: rows shifted inward", fontsize=SMALL_PT, color=GREEN)
    ax.text(N + 0.8, rows - 3.0, "dashed: block edge; the neighbour's", fontsize=SMALL_PT, color=PURPLE)
    ax.text(N + 0.8, rows - 3.6, "edge flux arrives by exchange", fontsize=SMALL_PT, color=PURPLE)
    ax.set_xlim(-2.2, N + 6.2)
    ax.set_ylim(-1.0, rows + 0.1)
    return fig
