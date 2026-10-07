"""Drawing the court underneath a figure, for matplotlib and for plotly.

The geometry itself lives in `zoneout.court` so that the figures and the 3D
reconstruction can never disagree about where the lines are; this module only
renders it.
"""

from ..court import (ANTENNA_HEIGHT, COURT_WIDTH, CROSS_LINES, NET_HEIGHT,
                     OUTLINE_X, OUTLINE_Y, OUTLINE_Z)
from .style import COURT_FILL, COURT_FILL_OPACITY, COURT_LINE, NET, NET_OPACITY

# A volleyball net is 1 m deep, hanging from the top tape down.
NET_DEPTH = 1.0

# Corners of the playing area, counter-clockwise in the z = 0 plane.
_CORNERS = [(0, -9, 0), (9, -9, 0), (9, 9, 0), (0, 9, 0)]


def draw_court(ax):
    """Draw the filled orange court and its white lines onto a 3D matplotlib axes.

    Requires the axes to have `computed_zorder = False`; the caller sets that.
    Matplotlib's 3D renderer otherwise depth-sorts whole artists, which puts
    this large floor polygon in front of parts of the trajectory above it.

    The floor is opaque here, unlike in the interactive figure. Matplotlib
    composites a translucent plane over any geometry it decides is behind it,
    which tinted the blue path purple; plotly renders the same scene correctly
    and keeps its transparency.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    floor = Poly3DCollection(
        [_CORNERS],
        facecolor=COURT_FILL,
        edgecolor="none",
        zorder=0,
    )
    ax.add_collection3d(floor)

    ax.plot(OUTLINE_X, OUTLINE_Y, OUTLINE_Z, color=COURT_LINE, linewidth=1.6,
            zorder=1)
    for (x0, y0, z0), (x1, y1, z1) in CROSS_LINES:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=COURT_LINE, linewidth=1.6,
                zorder=1)

    return ax


def court_traces():
    """Return the plotly traces that render the court: filled floor + white lines."""
    import plotly.graph_objects as go

    xs = [c[0] for c in _CORNERS]
    ys = [c[1] for c in _CORNERS]
    zs = [c[2] for c in _CORNERS]

    traces = [
        go.Mesh3d(
            x=xs, y=ys, z=zs,
            # Two triangles covering the rectangle.
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=COURT_FILL,
            opacity=COURT_FILL_OPACITY,
            flatshading=True,
            hoverinfo="skip",
            showlegend=False,
            name="court",
        ),
        go.Scatter3d(
            x=OUTLINE_X, y=OUTLINE_Y, z=OUTLINE_Z,
            mode="lines",
            line=dict(color=COURT_LINE, width=4),
            hoverinfo="skip",
            showlegend=False,
            name="court outline",
        ),
    ]

    for (x0, y0, z0), (x1, y1, z1) in CROSS_LINES:
        traces.append(
            go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line=dict(color=COURT_LINE, width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def net_traces():
    """Return the plotly traces for the net: its mesh, the top tape and the antennas.

    The net spans the court between the sidelines, where the antennas stand, with
    its top at `court.NET_HEIGHT` and hanging `NET_DEPTH` below that. The
    antennas rise to `court.ANTENNA_HEIGHT` - the same two points the
    calibration clicks.
    """
    import plotly.graph_objects as go

    top, bottom = NET_HEIGHT, NET_HEIGHT - NET_DEPTH
    traces = [
        go.Mesh3d(
            x=[0, COURT_WIDTH, COURT_WIDTH, 0],
            y=[0, 0, 0, 0],
            z=[bottom, bottom, top, top],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=NET,
            opacity=NET_OPACITY,
            flatshading=True,
            hoverinfo="skip",
            showlegend=False,
            name="net",
        ),
        go.Scatter3d(
            x=[0, COURT_WIDTH], y=[0, 0], z=[top, top],
            mode="lines",
            line=dict(color=NET, width=5),
            hoverinfo="skip",
            showlegend=False,
            name="net tape",
        ),
    ]
    for x in (0, COURT_WIDTH):
        traces.append(go.Scatter3d(
            x=[x, x], y=[0, 0], z=[bottom, ANTENNA_HEIGHT],
            mode="lines",
            line=dict(color=NET, width=4),
            hoverinfo="skip",
            showlegend=False,
            name="antenna",
        ))
    return traces
