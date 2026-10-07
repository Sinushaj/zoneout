"""3D ball-trajectory figures.

Two outputs per reception:

- `save_trajectory_figure` writes a PNG - small, quick to flick through.
- `save_trajectory_html` writes a self-contained interactive page you can open
  in a browser and rotate, zoom and hover. This is the one to reach for when a
  reception looks wrong and you need to understand why.

Only the interactive page takes the optional `raw_points`: the triangulated
measurements as they were before the ballistic fit replaced them, drawn in a
second color so the fitted parabolas can be checked against what they were
fitted to. Legend clicks toggle either series, so the comparison can be turned
off once it has been made. The PNG deliberately stays a single path - it is
there to be flicked through, and two overlaid trajectories in a static 3D
projection are hard to tell apart without being able to rotate them.

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
from .style import APEX, MARKER_OUTLINE, PATH, RAW_PATH, RECEPTION, SERVE


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

def plot_trajectory(points, serve_point=None, receive_point=None, title=None,
                    markers=None):
    """Build a 3D figure of a ball trajectory over the court.

    Args:
        points: sequence of (x, y, z) points, may contain None for frames with
            no detection; those are skipped.
        serve_point: optional (x, y, z) highlighted as the serve contact.
        receive_point: optional (x, y, z) highlighted as the reception contact.
        title: optional figure title.
        markers: optional `[(label, point, color), ...]`, used instead of the
            serve/reception pair when the highlighted points are neither.

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

    if markers is None:
        markers = [("Serve", serve_point, SERVE),
                   ("Reception", receive_point, RECEPTION)]

    for label, point, color in markers:
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
                           receive_point=None, title=None, dpi=300,
                           markers=None):
    """Build a trajectory figure and write it to `output_path` as a PNG.

    Creates the parent directory if needed. Returns the path written.
    """
    fig = plot_trajectory(points, serve_point, receive_point, title=title,
                          markers=markers)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)

    return output_path


# ---------------------------------------------------------------------------
# Interactive HTML
# ---------------------------------------------------------------------------

def plot_trajectory_plotly(points=None, serve_point=None, receive_point=None,
                           title=None, raw_points=None, markers=None,
                           heights=None):
    """Build a rotatable plotly figure of a ball trajectory over the court.

    Args:
        points: as `plot_trajectory`. May be None, in which case no ball-path
            trace is drawn at all — for a stage of a pipeline that has
            triangulated points but has not fitted a trajectory through them
            yet. Pass those as `raw_points`; drawing them as a *path* would
            join across the frames no pairing survived, which is a claim the
            reconstruction has not made.
        serve_point, receive_point, title: as `plot_trajectory`.
        markers: optional `[(label, point, color), ...]` drawn the same way the
            serve and reception are, for figures whose highlighted points are
            neither. Used instead of `serve_point`/`receive_point`, not as well
            as. Every marker keeps its text label and outline ring: two of the
            data colors sit under 3:1 against the court, so identity must never
            rest on hue alone - see the note in `style.py`.
        raw_points: optional second point list, drawn underneath `points` in
            RAW_PATH. Meant for the reconstruction's own triangulated points,
            so the fitted trajectory can be compared against the measurements
            it came from. Click either legend entry to hide that series.
        heights: optional `[(label, point), ...]`, points whose *height* is
            the measurement - the top of a pass. Each is drawn in APEX ink as
            a cross with a dashed line down to the floor, and labelled with
            its height, in addition to `markers`.

    Returns:
        plotly Figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for trace in court_traces():
        fig.add_trace(trace)

    if raw_points is not None:
        raw_frames, raw_x, raw_y, raw_z = _split_xyz(raw_points)
        # Markers only, and no connecting line: the raw points are what
        # survived matching, gaps included, and joining across a gap would
        # draw a straight segment the reconstruction never claimed. It also
        # keeps the two series apart by shape and not by color alone, which
        # this color needs (see the note on RAW_PATH in style.py).
        fig.add_trace(go.Scatter3d(
            x=raw_x, y=raw_y, z=raw_z,
            mode="markers",
            marker=dict(size=3, color=RAW_PATH),
            name="Raw points",
            customdata=raw_frames,
            hovertemplate=("<b>raw</b> frame %{customdata}<br>"
                           "x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<extra></extra>"),
        ))

    if points is not None:
        frames, x, y, z = _split_xyz(points)
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

    if markers is None:
        markers = [("Serve", serve_point, SERVE),
                   ("Reception", receive_point, RECEPTION)]

    for label, point, color in markers:
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

    for label, point in heights or ():
        if point is None:
            continue
        x, y, z = (float(v) for v in point)
        fig.add_trace(go.Scatter3d(
            x=[x, x], y=[y, y], z=[0.0, z],
            mode="lines",
            line=dict(color=APEX, width=3, dash="dash"),
            legendgroup=label, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text",
            marker=dict(size=6, color=APEX, symbol="cross",
                        line=dict(color=MARKER_OUTLINE, width=2)),
            text=[f"{label} {z:.2f} m"],
            textposition="top center",
            textfont=dict(size=11),
            name=label, legendgroup=label,
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


def save_trajectory_html(output_path, points=None, serve_point=None,
                         receive_point=None, title=None, raw_points=None,
                         markers=None, heights=None):
    """Write a self-contained interactive trajectory page to `output_path`.

    Open it in a browser: drag to rotate, scroll to zoom, hover for the frame
    number and coordinates, click a legend entry to show or hide that series.
    Pass `raw_points` to overlay the measurements the trajectory was fitted to,
    or `points=None` with only `raw_points` to draw the measurements alone.
    The plotly javascript is embedded, so the file works offline and can be
    moved around on its own (at the cost of a few MB per file). Creates the
    parent directory if needed; returns the path.
    """
    fig = plot_trajectory_plotly(points, serve_point, receive_point, title=title,
                                 raw_points=raw_points, markers=markers,
                                 heights=heights)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    return output_path
