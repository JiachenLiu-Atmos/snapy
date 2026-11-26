"""The three stages of rk3 in Shu-Osher form: each stage mixes the step-start state, the current stage
state and one full-step increment with the weights of integrator.cpp:49-61 at
pyharp@4721715855e937c1e8b218e964c0655f46e56e29, applied in MeshBlockImpl::advance_local
(src/mesh/meshblock.cpp:755 at snapy@e894700ff7aee30b52882e5202b16461413780b0). A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    xs = [0.4, 3.0, 5.6, 8.2]
    labels = [r"$\mathbf{U}^n$", r"$\mathbf{U}^{(1)}$", r"$\mathbf{U}^{(2)}$", r"$\mathbf{U}^{n+1}$"]
    for x, lab in zip(xs, labels):
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.4, 0.7, boxstyle="round,pad=0.03", fc=fs.SKY, ec=fs.BLACK, lw=0.8))
        ax.text(x + 0.7, 1.55, lab, ha="center", va="center")
    weights = [r"$w_1=1,\ w_2=1$", r"$w_0=\frac{3}{4},\ w_1=\frac{1}{4},\ w_2=\frac{1}{4}$",
               r"$w_0=\frac{1}{3},\ w_1=\frac{2}{3},\ w_2=\frac{2}{3}$"]
    for k in range(3):
        ax.annotate("", xy=(xs[k + 1], 1.55), xytext=(xs[k] + 1.4, 1.55),
                    arrowprops=dict(arrowstyle="->", color=fs.BLACK, lw=1.0))
        ax.text((xs[k] + 1.4 + xs[k + 1]) / 2, 2.55, weights[k], ha="center", fontsize=8, color=fs.GREEN)
        ax.text((xs[k] + 1.4 + xs[k + 1]) / 2, 2.05, r"$+\,w_2\,\Delta t\,\mathcal{L}$", ha="center", fontsize=8,
                color=fs.ORANGE)
    for k in (1, 2):
        ax.annotate("", xy=(xs[k + 1] + 0.3, 1.2), xytext=(xs[0] + 0.7, 1.2),
                    arrowprops=dict(arrowstyle="->", color=fs.GREY, lw=0.7, ls="--",
                                    connectionstyle="arc3,rad=0.25"))
    ax.text(0.4, 0.15, r"dashed: the step-start state $\mathbf{U}^n$ re-enters with weight $w_0$", fontsize=8,
            color=fs.GREY)
    return fig
