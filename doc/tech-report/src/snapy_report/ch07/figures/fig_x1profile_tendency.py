"""Spurious conductive tendency of the product-of-means face coefficient on a column whose
coefficient s q is constant, against the mean of products used by the code.

Data: data/x1profile_tendency.csv, written by x1profile_check.py (claim C2), which ports
face_scaled_coefficient of src/forcing/diffusion.cpp:168-194 at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5; n1 = 32, rho = (1.25 - x1)^1.5, T = 300 + 50 x1.
"""
import os

import numpy as np

from snapy_report.figstyle import BLACK, GREEN, SINGLE, VERMILLION, new_figure

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "x1profile_tendency.csv")


def make_fig():
    x, mp, pm = np.loadtxt(DATA, delimiter=",", unpack=True)
    fig, ax = new_figure(SINGLE, 2.6)
    ax.plot(x, pm, "s-", color=VERMILLION, ms=3, lw=1.0, label="product of means")
    ax.plot(x, mp, "o-", color=GREEN, ms=3, lw=1.0, label="mean of products (code)")
    ax.axhline(0., color=BLACK, lw=0.5)
    ax.set_xlabel("$x_1$ [-]")
    ax.set_ylabel(r"$\dot E_i$ [-]")
    ax.legend(frameon=False)
    return fig
