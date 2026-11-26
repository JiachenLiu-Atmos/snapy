"""What the face form reads for cell i.

Cartoon of src/hydro/hydro_forward.cpp:1060-1083 and src/hydro/gravity_work.hpp:16-29 at
snapy@73c45a7c8c991209924e1137992352c335575062;
no measured data.
"""

from snapy_report import figstyle as fs


def make_fig():
    fig, a = fs.new_figure(fs.SINGLE, 2.2)
    a.set_axis_off()
    a.set(xlim=(-.3, 4.3), ylim=(-.9, 1.6))
    a.set_title("face form, cell i", fontsize=fs.LABEL_PT, loc="left")
    for k, lab in enumerate(["i-1", "i", "i+1"]):
        fs.cell_box(a, .5 + k, 0., 1., 1.)
        a.text(1. + k, -.3, lab, ha="center", fontsize=fs.SMALL_PT)
    fs.cell_value(a, 2., .5)
    a.text(2., .72, r"$\phi_i$", ha="center", fontsize=fs.LABEL_PT)
    for x, t, ha in ((1.5, r"$\phi_{i-1/2}F_{i-1/2}$", "right"), (2.5, r"$\phi_{i+1/2}F_{i+1/2}$", "left")):
        fs.face_value(a, x, .5)
        a.text(x, 1.2, t, ha=ha, fontsize=fs.SMALL_PT, color=fs.ORANGE)
    a.text(2., -.75, r"weights $x_{1,i}-x_{1,i-1/2}$ and $x_{1,i+1/2}-x_{1,i}$", ha="center", fontsize=fs.SMALL_PT)
    return fig
