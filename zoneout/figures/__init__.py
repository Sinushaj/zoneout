"""Figure generation for the ball-tracking pipeline.

Keeps plotting code out of the processing loop in `main.py`. Add new figure
types as further modules here and re-export them below.
"""

from .court import draw_court
from .trajectory import plot_trajectory, save_trajectory_figure

__all__ = ["draw_court", "plot_trajectory", "save_trajectory_figure"]
