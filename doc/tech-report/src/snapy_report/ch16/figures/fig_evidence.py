"""Conceptual verification paths, no measured data; chapter 15 at the code pin."""
from snapy_report import figstyle


def make_fig():
    fig, ax = figstyle.new_figure("double", 2.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for y, labels in [(0.75, ('Derived identity', 'Symbolic check', 'Formula evidence')),
                      (0.25, ('Pinned assertion', 'Build and run', 'Runtime evidence'))]:
        for x, label in zip((0.15, 0.5, 0.85), labels):
            ax.text(x, y, label, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc=figstyle.WHITE, ec=figstyle.GREY))
        for start, end in ((0.29, 0.36), (0.64, 0.71)):
            ax.annotate('', (end, y), (start, y), arrowprops=dict(arrowstyle='->'))
    return fig
