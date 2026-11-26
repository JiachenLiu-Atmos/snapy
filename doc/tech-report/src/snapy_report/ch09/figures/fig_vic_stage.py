"""One-step amplification |R(z)| of rk3 with the VIC correction on the decay u' = -lam u, z = lam dt, with the
stage-weighted correction dt (src/hydro/hydro_forward.cpp:1065-1080 at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5)
and with the full dt. Computed by vic_stage_check.amplification.
"""
import matplotlib.pyplot as plt
import numpy as np

from snapy_report import figstyle as fs
from snapy_report.ch09.checks.vic_stage_check import amplification


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.3))
    z = np.logspace(-2, 6, 400)
    ax.semilogx(z, np.abs(amplification("rk3", z)), color=fs.GREEN, label=r"$\Delta t_c=w_2^{(s)}\Delta t$")
    ax.semilogx(z, np.abs(amplification("rk3", z, weighted=False)), color=fs.VERMILLION, ls="--",
                label=r"$\Delta t_c=\Delta t$")
    ax.semilogx(z, np.exp(-np.minimum(z, 50.)), color=fs.BLUE, ls=":", label=r"exact $e^{-\mathcal{Z}}$")
    ax.axhline(1. / 12., color=fs.GREY, lw=0.6)
    ax.axhline(1. / 3., color=fs.GREY, lw=0.6)
    ax.text(2e-2, 1. / 12. + 0.02, "1/12", fontsize=7, color=fs.GREY)
    ax.text(2e-2, 1. / 3. + 0.02, "1/3", fontsize=7, color=fs.GREY)
    ax.set_xlabel(r"$\mathcal{Z}=\mathcal{G}\,\Delta t$")
    ax.set_ylabel(r"$|\mathcal{A}(\mathcal{Z})|$")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right")
    return fig
