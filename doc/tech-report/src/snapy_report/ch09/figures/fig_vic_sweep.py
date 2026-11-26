"""The block-tridiagonal matrix of one column and the two passes that solve it, ForwardSweep
(src/implicit/forward_sweep_impl.h:25-139) and vic_backward_substitute (src/implicit/vic_redistribute_impl.h:27-36)
at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5. A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.5))
    ax.set_axis_off()
    ax.set_aspect("equal")
    n = 6
    for i in range(n):
        for j, (lab, col) in ((i - 1, ("B", fs.SKY)), (i, ("A", fs.GREEN)), (i + 1, ("C", fs.SKY))):
            if 0 <= j < n:
                ax.add_patch(plt.Rectangle((j * 0.4, (n - 1 - i) * 0.4), 0.36, 0.36, fc=col, ec=fs.BLACK, lw=0.5))
                ax.text(j * 0.4 + 0.18, (n - 1 - i) * 0.4 + 0.18, "$\\mathsf{%s}_%d$" % (lab, i), ha="center", va="center",
                        fontsize=6)
    ax.add_patch(plt.Rectangle((0, 0), n * 0.4, n * 0.4, fc="none", ec=fs.BLACK, lw=0.8))
    ax.annotate("", xy=(2.75, 0.05), xytext=(2.75, 2.35), arrowprops=dict(arrowstyle="->", color=fs.ORANGE, lw=1.4))
    ax.text(2.9, 2.2, "forward sweep, first row to last:", fontsize=8, color=fs.ORANGE)
    ax.text(2.9, 1.9, r"$\tilde{\mathsf{C}}_i=(\mathsf{A}_i-\mathsf{B}_i\tilde{\mathsf{C}}_{i-1})^{-1}\mathsf{C}_i$", fontsize=8)
    ax.text(2.9, 1.6, r"$\tilde{\boldsymbol{\delta}}_i=(\mathsf{A}_i-\mathsf{B}_i\tilde{\mathsf{C}}_{i-1})^{-1}(\Delta\mathbf{U}^{(0)}_{\mathrm{sv},i}/\Delta t_c-\mathsf{B}_i\tilde{\boldsymbol{\delta}}_{i-1})$", fontsize=8)
    ax.text(2.9, 1.3, r"each block: ludcmp, refused if a pivot ratio $\leq 8N\epsilon_{\mathrm{m}}$", fontsize=8, color=fs.VERMILLION)
    ax.annotate("", xy=(6.95, 2.35), xytext=(6.95, 0.05), arrowprops=dict(arrowstyle="->", color=fs.BLUE, lw=1.4))
    ax.text(2.9, 0.75, "back substitution, last row to first:", fontsize=8, color=fs.BLUE)
    ax.text(2.9, 0.45, r"$\boldsymbol{\delta}_i=\tilde{\boldsymbol{\delta}}_i-\tilde{\mathsf{C}}_i\,\boldsymbol{\delta}_{i+1}$", fontsize=8)
    ax.text(2.9, 0.1, r"one column per thread", fontsize=8, color=fs.GREY)
    ax.set_xlim(-0.1, 7.2)
    ax.set_ylim(-0.1, 2.5)
    return fig
