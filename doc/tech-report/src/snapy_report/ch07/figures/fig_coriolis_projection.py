"""Projection of the rotation vector onto the local spherical-polar basis of a cell.

Drawn from CoriolisXYZImpl::reset, src/forcing/coriolis.cpp:100-108, at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, for a rotation about z. A cartoon; it reads no
data.
"""
import numpy as np

from snapy_report.figstyle import BLACK, BLUE, GREEN, LABEL_PT, ORANGE, SINGLE, SKY, SMALL_PT, new_figure


def make_fig():
    fig, ax = new_figure(SINGLE, 3.0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    t = np.linspace(0., np.pi, 100)
    ax.plot(np.sin(t), np.cos(t), color=BLACK, lw=0.8)
    ax.plot([0, 0], [-1.25, 1.35], color=BLACK, lw=0.6, ls=":")
    ax.annotate("", xy=(0, 1.35), xytext=(0, 1.0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.text(0.06, 1.3, r"$\boldsymbol{\Omega}$", color=BLUE, fontsize=LABEL_PT)
    th = 0.8
    p = np.array([np.sin(th), np.cos(th)])
    ax.plot([0, p[0]], [0, p[1]], color=BLACK, lw=0.6, ls="--")
    ax.plot(*p, "o", color=SKY, ms=6, mec=BLACK, mew=0.6)
    r_hat = p
    th_hat = np.array([np.cos(th), -np.sin(th)])
    ax.annotate("", xy=p + 0.45 * r_hat, xytext=p, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("", xy=p + 0.45 * th_hat, xytext=p, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.text(*(p + 0.5 * r_hat + [0.02, 0.0]), r"$\Omega\cos\theta$", color=ORANGE, fontsize=SMALL_PT)
    ax.text(*(p + 0.5 * th_hat + [0.03, -0.05]), r"$-\Omega\sin\theta$", color=GREEN, fontsize=SMALL_PT)
    ax.text(0.08, 0.3, r"$\theta$", fontsize=LABEL_PT)
    ax.text(-0.95, -1.2, "cell centre: sky-blue circle", fontsize=SMALL_PT)
    ax.set_xlim(-1.0, 1.9)
    ax.set_ylim(-1.3, 1.5)
    return fig
