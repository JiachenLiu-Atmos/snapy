"""Status of the coordinate types at snapy@73c45a7c8c991209924e1137992352c335575062 (src/coord/coordinate.cpp:702-719, CoordinateImpl::create). A cartoon; it reads no data."""

from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 0.9)
    types = ["cartesian", "spherical-polar", "gnomonic-equiangle", "cylindrical"]
    for n, t in enumerate(types):
        ok = t != "cylindrical"
        ax.text(n, 0, t + ("\nbuilds a metric" if ok else "\nno metric"), ha="center", va="center",
                fontsize=fs.SMALL_PT,
                bbox=dict(boxstyle="round,pad=0.4", fc=fs.GREEN if ok else "white",
                          ec=fs.BLACK, alpha=0.5, hatch=None if ok else "xx"))
    ax.set(xlim=(-0.6, 3.6), ylim=(-0.6, 0.6))
    ax.axis("off")
    return fig
