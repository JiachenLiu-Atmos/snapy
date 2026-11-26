"""The integration/implicit-scheme value as bits, and what each value builds, from ImplicitOptionsImpl
(src/implicit/implicit_hydro.cpp:71-98, implicit_hydro.hpp:33-39) and the time-step bounds
(src/hydro/hydro.cpp:331-351) at snapy@e894700ff7aee30b52882e5202b16461413780b0. A cartoon; it reads no data.
"""
import matplotlib.pyplot as plt

from snapy_report import figstyle as fs


def make_fig():
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.DOUBLE, 1.9))
    ax.set_axis_off()
    bits = [("bit 3", "full (5 x 5)"), ("bit 2", r"$x_3$ bound"), ("bit 1", r"$x_2$ bound"), ("bit 0", r"$x_1$ implicit")]
    for k, (b, meaning) in enumerate(bits):
        ax.text(0.5 + k * 1.15, 2.05, b, ha="center", fontsize=8)
        ax.text(0.5 + k * 1.15, 0.05, meaning, ha="center", fontsize=7, color=fs.GREY)
    rows = [(0, "0000", "explicit: no solver object, any nb1", fs.GREY),
            (1, "0001", "vic-partial: rho, rho u, E", fs.GREEN),
            (9, "1001", "vic-full: rho, rho u, rho v, rho w, E", fs.GREEN)]
    for r, (val, code, text, col) in enumerate(rows):
        y = 1.55 - 0.45 * r
        for k, ch in enumerate(code):
            ax.add_patch(plt.Rectangle((0.1 + k * 1.15, y - 0.17), 0.8, 0.34, fc=col if ch == "1" else "white",
                                       ec=fs.BLACK, lw=0.6))
            ax.text(0.5 + k * 1.15, y, ch, ha="center", va="center", fontsize=8)
        ax.text(4.85, y, "%d" % val, ha="center", va="center", fontsize=9)
        ax.text(5.3, y, text, va="center", fontsize=8)
    ax.text(5.3, 0.25, "any other value: Unsupported implicit scheme", fontsize=8, color=fs.VERMILLION)
    ax.text(4.85, 2.05, "value", ha="center", fontsize=8)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.1, 2.3)
    return fig
