"""The stage order of MeshBlockImpl::stage_advance, stage_forcing, stage_combine, stage_saturation, stage_chemistry and
stage_boundaries (src/mesh/meshblock_advance.cpp) at the code pin. No measured data."""
from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.SINGLE, 3.1)
    labels = ['face fluxes + sources', 'VIC + user callbacks', 'RK average + limiter', 'phase solve, chemistry, boundaries']
    for n, label in enumerate(labels):
        y = 3-n
        ax.text(.5, y, label, ha="center", va="center", fontsize=fs.SMALL_PT,
                bbox=dict(boxstyle="round,pad=.5", fc=fs.SKY if n in (0,3) else fs.GREEN, ec=fs.BLACK))
        if n < 3:
            ax.annotate("", xy=(.5,y-.72), xytext=(.5,y-.25), arrowprops=dict(arrowstyle="->", color=fs.BLACK))
    ax.set(xlim=(0,1), ylim=(-.4,3.4))
    ax.axis("off")
    return fig
