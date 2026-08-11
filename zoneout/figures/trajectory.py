"""3D ball-trajectory figures.

Two outputs per reception:

- `save_trajectory_figure` writes a PNG - small, quick to flick through.
- `save_trajectory_html` writes a self-contained interactive page you can open
  in a browser and rotate, zoom and hover. This is the one to reach for when a
  reception looks wrong and you need to understand why.

The matplotlib figures are built on `matplotlib.figure.Figure` directly rather
than through pyplot. pyplot keeps every figure it creates in a global registry,
which in a batch run over many receptions means figures pile up in memory
unless each one is explicitly closed; going through Figure avoids that, and
also avoids needing a GUI backend at all.
"""

import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)

from .court import court_traces, draw_court
from .style import MARKER_OUTLINE, PATH, RECEPTION, SERVE


def _valid_points(points):
    """Drop missing frames, keeping each surviving point's original frame index."""
    return [(i, p) for i, p in enumerate(points) if p is not None]


def _split_xyz(points):
    """Split a point list into frame indices and x, y, z lists, skipping gaps."""
    valid = _valid_points(points)
    return (
        [i for i, _ in valid],
        [p[0] for _, p in valid],
        [p[1] for _, p in valid],
        [p[2] for _, p in valid],
    )


def _fit_axes_to_data(ax, x, y, z):
    """Scale the axes so the court and the trajectory both fit, undistorted."""
    all_x = np.concatenate([x, [0, 9]])
    all_y = np.concatenate([y, [-9, 9]])
    all_z = np.concatenate([z, [0]])

    # A degenerate span (e.g. a trajectory flat in z) would make matplotlib
    # reject the box aspect, so keep every extent strictly positive.
    ax.set_box_aspect([
        max(np.ptp(all_x), 1e-6),
        max(np.ptp(all_y), 1e-6),
        max(np.ptp(all_z), 1e-6),
    ])

    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # True court proportions squash z, so thin the tick labels out or they
    # collide with each other.
    ax.zaxis.set_major_locator(MaxNLocator(4))


# ---------------------------------------------------------------------------
# Static PNG
# ---------------------------------------------------------------------------

def plot_trajectory(points, serve_point=None, receive_point=None, title=None):
    """Build a 3D figure of a ball trajectory over the court.

    Args:
        points: sequence of (x, y, z) points, may contain None for frames with
            no detection; those are skipped.
        serve_point: optional (x, y, z) highlighted as the serve contact.
        receive_point: optional (x, y, z) highlighted as the reception contact.
        title: optional figure title.

    Returns:
        matplotlib.figure.Figure
    """
    _, x, y, z = _split_xyz(points)

    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")

    # Draw in explicit zorder rather than matplotlib's depth sort, which would
    # otherwise paint the court floor over the trajectory above it.
    ax.computed_zorder = False

    _fit_axes_to_data(ax, x, y, z)
    draw_court(ax)

    # Line joining consecutive detections, with small dots for the samples.
    ax.plot(x, y, z, color=PATH, linewidth=1.2, zorder=5)
    ax.scatter(x, y, z, color=PATH, s=4, depthshade=False, zorder=6)

    for point, color, label in (
        (serve_point, SERVE, "Serve"),
        (receive_point, RECEPTION, "Reception"),
    ):
        if point is None:
            continue
        # Labelled, ringed and diamond-shaped: identity never rests on hue
        # alone, which the palette's contrast warning requires.
        ax.scatter(*point, color=color, s=55, marker="D",
                   edgecolors=MARKER_OUTLINE, linewidths=1.0,
                   depthshade=False, zorder=7)
        ax.text(point[0], point[1], point[2], f"  {label}", fontsize=7, zorder=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if title:
        ax.set_title(title)

    return fig


def save_trajectory_figure(output_path, points, serve_point=None,
                           receive_point=None, title=None, dpi=300):
    """Build a trajectory figure and write it to `output_path` as a PNG.

    Creates the parent directory if needed. Returns the path written.
    """
    fig = plot_trajectory(points, serve_point, receive_point, title=title)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)

    return output_path


# ---------------------------------------------------------------------------
# Interactive HTML
# ---------------------------------------------------------------------------

def plot_trajectory_plotly(points, serve_point=None, receive_point=None,
                           title=None):
    """Build a rotatable plotly figure of a ball trajectory over the court.

    Same arguments as `plot_trajectory`. Returns a plotly Figure.
    """
    import plotly.graph_objects as go

    frames, x, y, z = _split_xyz(points)

    fig = go.Figure()
    for trace in court_traces():
        fig.add_trace(trace)

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines+markers",
        line=dict(color=PATH, width=3),
        marker=dict(size=2.5, color=PATH),
        name="Ball path",
        customdata=frames,
        hovertemplate=("frame %{customdata}<br>"
                       "x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<extra></extra>"),
    ))

    for point, color, label in (
        (serve_point, SERVE, "Serve"),
        (receive_point, RECEPTION, "Reception"),
    ):
        if point is None:
            continue
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode="markers+text",
            marker=dict(size=7, color=color, symbol="diamond",
                        line=dict(color=MARKER_OUTLINE, width=2)),
            text=[label],
            textposition="top center",
            textfont=dict(size=11),
            name=label,
            hovertemplate=(f"<b>{label}</b><br>"
                           "x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        # Real-world proportions: the court is 9 m x 18 m and the ball only
        # rises a few meters, so don't stretch z into something misleading.
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01),
    )

    return fig


def save_trajectory_html(output_path, points, serve_point=None,
                         receive_point=None, title=None):
    """Write a self-contained interactive trajectory page to `output_path`.

    Open it in a browser: drag to rotate, scroll to zoom, hover for the frame
    number and coordinates. The plotly javascript is embedded, so the file
    works offline and can be moved around on its own (at the cost of a few MB
    per file). Creates the parent directory if needed; returns the path.
    """
    fig = plot_trajectory_plotly(points, serve_point, receive_point, title=title)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    return output_path
