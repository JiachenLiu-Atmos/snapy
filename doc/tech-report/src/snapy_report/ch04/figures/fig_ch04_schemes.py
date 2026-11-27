"""Figures of the chapter 4 scheme sections drawn in the house style (figstyle only).

Each make_* function returns one figure. The reconstruction panels evaluate ports of the formulas of
src/recon/interpolation.hpp:88-107 (donor cell), src/recon/plm.cpp:10-32 (PLM) and src/recon/interp_impl.h:73-122 with
the weno5.cpp buffers (WENO5) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5; the wave fans are cartoons of
src/riemann/lmars_impl.h, hllc_impl.h, roe_impl.h and shallow_roe_impl.h; the source and face-measure figures
are cartoons of src/coord/spherical_polar.cpp:224-309, src/coord/gnomonic_equiangle.cpp:415 and
src/hydro/hydro_forward.cpp:54-206. Nothing is read from disk.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch

from snapy_report import figstyle as fs

W5 = np.array([[-1 / 6, 5 / 6, 1 / 3, 0, 0], [0, 1 / 3, 5 / 6, -1 / 6, 0], [0, 0, 11 / 6, -7 / 6, 1 / 3],
               [1, -2, 1, 0, 0], [1, -4, 3, 0, 0], [0, 1, -2, 1, 0], [0, -1, 0, 1, 0],
               [0, 0, 1, -2, 1], [0, 0, 3, -4, 1]])


def _profile(n=12):
    x = (np.arange(n) + 0.5) / n
    q = np.where(x < 0.55, 0.3 + 0.9 * x, 1.6 - 0.4 * x)
    q = q + 0.25 * np.sin(2 * np.pi * x)
    return x, q, 1. / n


def make_dc():
    fig, ax = fs.new_figure("single", 2.2)
    x, q, h = _profile()
    xx = np.linspace(0, 1, 400)
    exact = np.interp(xx, x, q)
    ax.plot(xx, exact, color=fs.BLUE, lw=0.9, label="cell values, joined")
    for xi, qi in zip(x, q):
        ax.plot([xi - h / 2, xi + h / 2], [qi, qi], color=fs.GREEN, lw=1.4)
    for k in range(1, len(x)):
        ax.plot([x[k] - h / 2] * 2, [q[k - 1], q[k]], color=fs.ORANGE, lw=1.0)
        fs.face_marker(ax, x[k] - h / 2, q[k - 1])
    ax.plot([], [], color=fs.GREEN, lw=1.4, label="donor-cell face states")
    ax.plot([], [], color=fs.ORANGE, lw=1.0, label="jump the solver sees")
    ax.set_xlabel("$x_1$ (domain fraction)")
    ax.set_ylabel("$q$")
    ax.legend(frameon=False, loc="lower right")
    return fig


def plm_faces(q):
    dl, dr = q[1:-1] - q[:-2], q[2:] - q[1:-1]
    d2 = dl * dr
    dm = np.where(d2 > 0, 2 * d2 / np.where(d2 > 0, dl + dr, 1.), 0.)
    return q[1:-1] - 0.5 * dm, q[1:-1] + 0.5 * dm


def make_plm():
    fig, axes = fs.new_figure("double", 2.3, ncols=2)
    x, q, h = _profile()
    lo, hi = plm_faces(q)
    ax = axes[0]
    for k, xi in enumerate(x[1:-1]):
        ax.plot([xi - h / 2, xi + h / 2], [lo[k], hi[k]], color=fs.GREEN, lw=1.4)
    for xi, qi in zip(x, q):
        fs.cell_marker(ax, xi, qi)
    ext = [k + 1 for k in range(len(x) - 2) if (q[k + 1] - q[k]) * (q[k + 2] - q[k + 1]) <= 0]
    for k in ext:
        ax.plot([x[k]], [q[k]], ls="none", marker="s", ms=8, mfc="none", mec=fs.VERMILLION, mew=1.0)
    ax.plot([], [], ls="none", marker="s", mfc="none", mec=fs.VERMILLION, label="extremum: slope 0")
    ax.plot([], [], color=fs.GREEN, lw=1.4, label="PLM line in each cell")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("$x_1$ (domain fraction)")
    ax.set_ylabel("$q$")
    ax = axes[1]
    r = np.linspace(-1.5, 4, 400)
    hm = np.where(r > 0, 2 * r / (1 + r), 0)
    ax.plot(r, hm, color=fs.GREEN, lw=1.4, label=r"harmonic mean $D/D_-$")
    ax.plot(r, np.where(r > 0, 2 * np.minimum(1, r), 0), color=fs.VERMILLION, lw=0.9, ls="--",
            label=r"bound $2\min(1, D_+/D_-)$")
    ax.plot(r, (1 + r) / 2, color=fs.BLUE, lw=0.9, ls=":", label="centred slope")
    ax.set_xlabel(r"$D_+/D_-$")
    ax.set_ylabel(r"slope $/D_-$")
    ax.set_ylim(-0.5, 2.6)
    ax.legend(frameon=False, loc="upper left")
    fs.panel_labels(axes)
    return fig


def weno5_weights(phi, scale, eps=1e-6):
    vs = np.mean(np.abs(phi)) if scale else 1.
    ph = phi / vs
    be = np.array([13 / 12 * (W5[3 + 2 * k] @ ph) ** 2 + 0.25 * (W5[4 + 2 * k] @ ph) ** 2 for k in range(3)])
    al = np.array([0.3, 0.6, 0.1]) / (be + eps) ** 2
    return al / al.sum()


def make_weno():
    fig, axes = fs.new_figure("double", 3.0, ncols=2)
    n = 40
    x = (np.arange(n) + 0.5) / n
    base = np.where(x < 0.5, 1.0, 2.0) + 0.05 * np.sin(2 * np.pi * x)
    for ax, mag in zip(axes, (1.0, 1e-4)):
        q = mag * base
        xs, w_off, w_on = [], [], []
        for i in range(2, n - 2):
            xs.append(x[i] - 0.5 / n)
            w_off.append(weno5_weights(q[i - 2:i + 3], False))
            w_on.append(weno5_weights(q[i - 2:i + 3], True))
        w_off, w_on = np.array(w_off), np.array(w_on)
        for k, col in zip(range(3), (fs.SKY, fs.GREEN, fs.ORANGE)):
            ax.plot(xs, w_on[:, k], color=col, lw=1.3, label=f"$\\omega_{k}$, scale on")
            ax.plot(xs, w_off[:, k], color=col, lw=0.9, ls="--", label=f"$\\omega_{k}$, scale off")
        ax.axvline(0.5, color=fs.GREY, lw=0.6)
        ax.set_xlabel("left face of cell (domain fraction)")
        ax.set_ylabel("nonlinear weight")
        ax.set_title(f"data magnitude {mag:g}", fontsize=fs.SMALL_PT)
        ax.set_ylim(-0.05, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="outside lower center")
    fs.panel_labels(axes)
    return fig


def _fan(ax, waves, labels, title):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.15)
    ax.axvline(0, color=fs.ORANGE, lw=2.0)
    for (s, col, ls), lab in zip(waves, labels):
        ax.plot([0, s], [0, 1], color=col, lw=1.2, ls=ls)
        ax.text(s * 1.04, 1.03, lab, ha="center", va="bottom", fontsize=fs.SMALL_PT, color=col)
    ax.text(-1.1, 0.12, r"$q^{\mathrm{L}}$", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.text(1.0, 0.12, r"$q^{\mathrm{R}}$", fontsize=fs.SMALL_PT, color=fs.BLUE)
    ax.set_xlabel("$x - x_f$")
    ax.set_ylabel("$t$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fs.SMALL_PT)


def make_wavefan(kind):
    fig, ax = fs.new_figure("single", 2.3)
    if kind == "lmars":
        _fan(ax, [(-0.8, fs.BLACK, "-"), (0.8, fs.BLACK, "-")],
             [r"$v-c$", r"$v+c$"], r"acoustic waves only: $p_f$, $v_f$ between them")
        ax.text(0.05, 0.5, r"$p_f,\ v_f$", fontsize=fs.SMALL_PT, color=fs.ORANGE)
    elif kind == "hllc":
        _fan(ax, [(-0.85, fs.BLACK, "-"), (0.25, fs.GREEN, "--"), (0.9, fs.BLACK, "-")],
             [r"$\lambda^{\mathrm{L}}$", r"$\lambda_{\mathrm{c}}$", r"$\lambda^{\mathrm{R}}$"],
             "two outer waves and the contact")
        ax.text(-0.35, 0.55, r"$\mathbf{U}^{\mathrm{L}}_{\mathrm{c}}$", fontsize=fs.SMALL_PT)
        ax.text(0.38, 0.55, r"$\mathbf{U}^{\mathrm{R}}_{\mathrm{c}}$", fontsize=fs.SMALL_PT)
    elif kind == "roe":
        _fan(ax, [(-0.7, fs.BLACK, "-"), (0.15, fs.GREEN, "--"), (0.95, fs.BLACK, "-")],
             [r"$\lambda_-$", r"$\lambda_0$", r"$\lambda_+$"], r"linear waves, strengths $\alpha_\ell$")
    else:
        _fan(ax, [(-0.7, fs.BLACK, "-"), (0.1, fs.GREEN, "--"), (0.9, fs.BLACK, "-")],
             [r"$v-c$", r"$v$ (shear)", r"$v+c$"], r"gravity waves, $c^2 = (d_{\mathrm{sw}}^{\mathrm{L}}+d_{\mathrm{sw}}^{\mathrm{R}})/2$")
    return fig


def _boxes(ax, items, y, w=2.75, h=0.62, dx=3.05, x0=0.1):
    for k, (text, col) in enumerate(items):
        x = x0 + k * dx
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=col, ec=fs.BLACK, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs.SMALL_PT)
        if k:
            ax.annotate("", xy=(x, y + h / 2), xytext=(x - dx + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color=fs.BLACK, lw=0.8))


def make_sphsrc():
    fig, ax = fs.new_figure("double", 2.0)
    ax.set_axis_off()
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 3.2)
    _boxes(ax, [("divergence\n(all three directions)", fs.SKY), ("centrifugal part\n" r"$\langle r^{-1}\rangle\rho(v_2^2+v_3^2)$",
                fs.CELL_FILL), ("radial pressure force\n(three branches)", fs.CELL_FILL),
                ("$m_2$, $m_3$ sources\n(chapter 3)", fs.CELL_FILL)], 2.2)
    for k, (text, col) in enumerate([("SNAP_X1_CENTROID_EXACT: $r^2$ moment of the face quintic", fs.GREEN),
                                     ("face pressures: area difference minus face gradient", fs.ORANGE),
                                     (r"no face pressure: $2\langle r^{-1}\rangle p$", fs.VERMILLION)]):
        ax.text(6.1, 1.55 - 0.5 * k, text, fontsize=fs.SMALL_PT, color=col)
    ax.text(0.1, 0.2, r"covariance on (curved grid): the source reads $p^{\mathrm{sh}} = p - (x_{1,i}-x^{\mathrm{A}}_{1,i})\,\partial_1 p$",
            fontsize=fs.SMALL_PT, color=fs.GREEN)
    return fig


def make_gnomonic():
    fig, ax = fs.new_figure("double", 1.7)
    ax.set_axis_off()
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 2.4)
    _boxes(ax, [("panel components\n$\\mathbf{W}$", fs.SKY), ("prim2local:\nface-orthonormal frame", fs.CELL_FILL),
                ("Riemann solver\n(lmars, hllc, roe,\nshallow-roe)", fs.CELL_FILL), ("flux2global:\nback to the panel", fs.CELL_FILL)],
           1.4)
    ax.text(0.1, 0.5, "RiemannSolverImpl::solve projects for every face-function solver, roe included",
            fontsize=fs.SMALL_PT, color=fs.BLACK)
    return fig


def make_fluxcov():
    fig, ax = fs.new_figure("single", 2.4)
    r0, h = 1.0, 0.8
    r = np.linspace(r0 - h / 2, r0 + h / 2, 200)
    wA, wV = r / np.trapezoid(r, r), r ** 2 / np.trapezoid(r ** 2, r)
    ax.plot(r, wA, color=fs.ORANGE, lw=1.3, label=r"face measure $\propto r$")
    ax.plot(r, wV, color=fs.SKY, lw=1.3, label=r"cell measure $\propto r^2$")
    xA = np.trapezoid(r * r, r) / np.trapezoid(r, r)
    xc = np.trapezoid(r ** 3, r) / np.trapezoid(r ** 2, r)
    ax.axvline(xA, color=fs.ORANGE, lw=0.8, ls="--")
    ax.axvline(xc, color=fs.SKY, lw=0.8, ls="--")
    ax.annotate("", xy=(xA, 0.35), xytext=(xc, 0.35), arrowprops=dict(arrowstyle="<->", color=fs.GREEN, lw=1.0))
    ax.text(0.5 * (xA + xc), 0.12, r"$x_{1,i}-x^{\mathrm{A}}_{1,i}$", ha="center", fontsize=fs.SMALL_PT, color=fs.GREEN)
    ax.set_xlabel(r"$r$ across one cell ($h = 0.8\,r$)")
    ax.set_ylabel("normalised weight")
    ax.set_ylim(0, 2.5)
    ax.legend(frameon=False, loc="upper left")
    return fig
