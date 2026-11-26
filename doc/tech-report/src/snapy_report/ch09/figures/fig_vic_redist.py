"""The constituent redistribution of one column of vic_redist_check: the total-mass increments phi_i, the face
transfers T_(i-1/2) built from them, and the dry and species changes moved donor-upwind
(vic_constituent_column, src/implicit/vic_redistribute_impl.h:77-161 at
snapy@e894700ff7aee30b52882e5202b16461413780b0). Computed with vic_port.constituent_column.
"""
import matplotlib.pyplot as plt
import numpy as np

from snapy_report import figstyle as fs
from snapy_report.ch09 import vic_port as vp
from snapy_report.ch09.checks.vic_redist_check import column


def make_fig():
    fs.apply()
    du, w, d0, vol = column()
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    n = len(vol)
    phi = (d0 - du.sum(1)) * vol
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.5), sharey=True)
    k = np.arange(n)
    a1.barh(k, phi, height=0.6, color=fs.SKY, label=r"$(\delta\rho_i-\Delta^{\mathrm{stg}}\rho_i)V_i$ (cell)")
    a1.plot(np.append(M, top), np.arange(n + 1) - 0.5, color=fs.ORANGE, marker="o", ms=3,
            label=r"$\mathcal{T}_{i-1/2}$ (face)")
    a1.axvline(0, color=fs.BLACK, lw=0.5)
    a1.set_xlabel("mass per step")
    a1.set_ylabel("layer")
    a1.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    a2.barh(k - 0.2, mass[:, 0] * vol, height=0.35, color=fs.GREEN, label="dry gas")
    a2.barh(k + 0.2, mass[:, 1:].sum(1) * vol * 20, height=0.35, color=fs.PURPLE, label=r"species $\times 20$")
    a2.axvline(0, color=fs.BLACK, lw=0.5)
    a2.set_xlabel("change per step")
    a2.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    return fig
