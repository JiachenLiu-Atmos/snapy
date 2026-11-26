"""The bounds that make up the time step on a 2-D column with an implicit x1 direction, from
HydroImpl::max_time_step (src/hydro/hydro.cpp:311-398 at snapy@e894700ff7aee30b52882e5202b16461413780b0).
A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 2.6))
    ax.set_axis_off()
    ax.set_aspect("equal")
    for i in range(3):
        for j in range(2):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fc="#EAF5FC", ec=fs.BLACK, lw=0.6))
            ax.plot(i + 0.5, j + 0.5, "o", color=fs.SKY, ms=5, mec=fs.BLACK, mew=0.5)
    ax.annotate("", xy=(1.95, 0.25), xytext=(1.05, 0.25), arrowprops=dict(arrowstyle="<->", color=fs.GREEN, lw=1.2))
    ax.text(1.5, -0.25, r"$x_1$: $|v_1|$", ha="center", fontsize=8, color=fs.GREEN)
    ax.annotate("", xy=(1.5, 1.95), xytext=(1.5, 1.05), arrowprops=dict(arrowstyle="<->", color=fs.ORANGE, lw=1.2))
    ax.text(-0.1, 1.5, r"$x_2$: $|v_2|+c_s$", fontsize=8, color=fs.ORANGE, va="center", ha="right")
    ax.plot([2, 2], [1, 2], color=fs.VERMILLION, lw=2.2)
    ax.text(2.0, 2.15, r"shear face: $|v_{2,i+1}-v_{2,i}|\geq c_f$", fontsize=8, color=fs.VERMILLION, ha="center")
    ax.text(3.4, 1.6, r"$\Delta t_1=C_{\mathrm{adv}}\,\Delta x_1/|v_1|$ (implicit $x_1$)", fontsize=8)
    ax.text(3.4, 1.15, r"$\Delta t_2=C\,\Delta x_2/(|v_2|+c_s)$ (explicit $x_2$)", fontsize=8)
    ax.text(3.4, 0.7, r"$\Delta t_{\mathrm{sh}}=C_{\mathrm{sh}}\,c_f\,\Delta x_2/(|v_{2,i}||v_{2,i+1}|)$", fontsize=8)
    ax.text(3.4, 0.25, r"$\Delta t=2^{-n_{\mathrm{redo}}}\min(\Delta t_1,\Delta t_2,\Delta t_{\mathrm{sh}},\Delta t_{\nu})$, all ranks",
            fontsize=8)
    ax.set_xlim(-1.6, 9.6)
    ax.set_ylim(-0.45, 2.4)
    return fig
