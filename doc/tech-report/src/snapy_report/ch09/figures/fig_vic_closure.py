"""The column closure: the ghost below the first cell (and above the last) is the mirror of its neighbour,
d_ghost = Bnd d, Bnd = diag(1, -1, 1, 1, 1), folded into the diagonal block
(src/implicit/vic_assemble_full_impl.h:44-45, 122-126 at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5).
A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.0))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.3, 0.4), 1.6, 1.0, fc="#F4E1EC", ec=fs.PURPLE, lw=0.8, ls="--"))
    ax.text(1.1, 1.15, "ghost", ha="center", fontsize=8, color=fs.PURPLE)
    ax.text(1.1, 0.75, r"$\mathsf{P}_{\mathrm{w}}\,\boldsymbol{\delta}_{0}$", ha="center", fontsize=9)
    for k in range(2):
        ax.add_patch(plt.Rectangle((1.9 + 1.6 * k, 0.4), 1.6, 1.0, fc="#EAF5FC", ec=fs.BLACK, lw=0.7))
        ax.text(2.7 + 1.6 * k, 0.75, r"$\boldsymbol{\delta}_{%d}$" % k, ha="center", fontsize=9)
    ax.plot([1.9, 1.9], [0.3, 1.5], color=fs.BLACK, lw=2.2)
    ax.text(1.9, 1.6, "wall", ha="center", fontsize=8)
    ax.annotate("", xy=(1.3, 0.55), xytext=(2.5, 0.55),
                arrowprops=dict(arrowstyle="->", color=fs.PURPLE, lw=0.8, connectionstyle="arc3,rad=0.4"))
    ax.text(5.6, 1.15, r"$\mathsf{B}_0\,\boldsymbol{\delta}_{-1}$ with $\boldsymbol{\delta}_{-1}=\mathsf{P}_{\mathrm{w}}\,\boldsymbol{\delta}_0$:", fontsize=9)
    ax.text(5.6, 0.75, r"$\mathsf{A}_0\leftarrow \mathsf{A}_0+\mathsf{B}_0\,\mathsf{P}_{\mathrm{w}}$, $\;\mathsf{P}_{\mathrm{w}}=\mathrm{diag}(1,-1,1,1,1)$", fontsize=9)
    ax.text(5.6, 0.35, "mass, transverse momentum, energy: no wall flux", fontsize=8, color=fs.GREEN)
    ax.set_xlim(0.2, 11.6)
    ax.set_ylim(0.2, 1.8)
    return fig
