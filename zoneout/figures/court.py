"""Drawing the court wireframe underneath a figure.

The geometry itself lives in `zoneout.court` so that the figures and the 3D
reconstruction can never disagree about where the lines are; this module only
renders it.
"""

from ..court import CROSS_LINES, OUTLINE_X, OUTLINE_Y, OUTLINE_Z


def draw_court(ax, color="black"):
    """Draw the court outline and cross lines onto a 3D axes."""
    ax.plot(OUTLINE_X, OUTLINE_Y, OUTLINE_Z, color=color)

    for (x0, y0, z0), (x1, y1, z1) in CROSS_LINES:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=color)

    return ax
