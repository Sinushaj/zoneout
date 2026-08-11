"""3D ball-trajectory figures.

Figures are built on `matplotlib.figure.Figure` directly rather than through
pyplot. pyplot keeps every figure it creates in a global registry, which in a
batch run over many receptions means figures pile up in memory unless each one
is explicitly closed; going through Figure avoids that, and also avoids needing
a GUI backend at all.
"""

import os

import numpy as np
from matplotlib.figure import Figure
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)

from .court import draw_court


def _split_xyz(points):
    """Drop missing frames and split a point list into x, y and z lists."""
    valid = [p for p in points if p is not None]
    return (
        [p[0] for p in valid],
        [p[1] for p in valid],
        [p[2] for p in valid],
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


def plot_trajectory(points, serve_point=None, receive_point=None, title=None):
    """Build a 3D figure of a ball trajectory over the court.

    Args:
        points: sequence of (x, y, z) points, may contain None for frames with
            no detection; those are skipped.
        serve_point: optional (x, y, z) highlighted in red.
        receive_point: optional (x, y, z) highlighted in red.
        title: optional figure title.

    Returns:
        matplotlib.figure.Figure
    """
    x, y, z = _split_xyz(points)

    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")

    _fit_axes_to_data(ax, x, y, z)

    ax.scatter(x, y, z)
    for highlight in (serve_point, receive_point):
        if highlight is not None:
            ax.scatter(*highlight, color="red", s=70)

    draw_court(ax)

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

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )

    return output_path
