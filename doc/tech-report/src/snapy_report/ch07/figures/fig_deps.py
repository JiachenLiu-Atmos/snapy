"""Dependency map of the chapter 9 schemes; snapy@6fbb39750f62b83df096058a4746e77d0a634bc5.

Logical relations read from the code cited in each scheme file (the forcing loop of HydroImpl::forward, the
registration in HydroImpl::_register_forcings_module, the tracer carry of MeshBlockImpl::advance_local, the wall
test MeshBlockOptionsImpl::is_wall_boundary and RelaxBotCompImpl::reset); no measured data.
"""

from snapy_report import figstyle as fs

NODES = {
    "user": (0.08, 0.80, "user stage\nforcings"),
    "framework": (0.45, 0.80, "forcing loop\n(forcing:)"),
    "tracer": (0.88, 0.80, "dry-air tracers\n(Chapter 10)"),
    "coriolis": (0.07, 0.42, "Coriolis"),
    "viscosity": (0.29, 0.42, "viscosity and\nconduction"),
    "heating": (0.51, 0.42, "heating and\ncooling"),
    "relax": (0.73, 0.42, "bottom\nrelaxation"),
    "sponge": (0.93, 0.42, "sponge\nlayers"),
    "wallcoef": (0.50, 0.06, "wall face\ncoefficient"),
    "x1profile": (0.29, 0.06, "x1 coefficient\nprofiles"),
    "walls": (0.71, 0.06, "named x1 walls\n(Chapter 11)"),
    "kintera": (0.92, 0.06, "kintera thermo\n(Chapter 2)"),
}
EDGES = [  # solid: is part of; dashed: hands its dry-air increment to; dotted: needs
    ("coriolis", "framework", "-", 0.0), ("viscosity", "framework", "-", 0.0), ("heating", "framework", "-", 0.0),
    ("relax", "framework", "-", 0.0), ("sponge", "framework", "-", 0.0),
    ("wallcoef", "viscosity", "-", 0.0), ("x1profile", "viscosity", "-", 0.0),
    ("framework", "tracer", "--", 0.0), ("user", "tracer", "--", -0.25),
    ("wallcoef", "walls", ":", 0.0), ("relax", "kintera", ":", 0.0),
]


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 3.0)
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.24, 1.02))
    ax.axis("off")
    boxes = {}
    for key, (x, y, label) in NODES.items():
        boxes[key] = ax.text(x, y, label, ha="center", va="center", fontsize=fs.SMALL_PT,
                             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=fs.GREEN))
    fig.canvas.draw()  # the box outlines exist only after a draw; arrows are clipped to them
    for a, b, ls, rad in EDGES:
        xa, ya, _ = NODES[a]
        xb, yb, _ = NODES[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color=fs.BLACK, ls=ls, lw=1.0,
                                    connectionstyle="arc3,rad=%g" % rad,
                                    patchA=boxes[a].get_bbox_patch(), patchB=boxes[b].get_bbox_patch()))
    ax.text(0.5, -0.19, "solid: is part of;   dashed: hands its dry-air increment to;   dotted: needs",
            fontsize=fs.SMALL_PT, ha="center", va="center")
    return fig
